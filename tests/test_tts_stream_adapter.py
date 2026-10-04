from __future__ import annotations

import pytest

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.aio.channel import ChanEmpty

from .fake_tts import FakeChunkedStream, FakeTTS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("max_retry", [1, 3])
@pytest.mark.parametrize(
    "error",
    [APIConnectionError(), APITimeoutError(), APIStatusError("rate limited", status_code=429)],
)
async def test_stream_adapter_does_not_retry_after_audio(max_retry: int, error: Exception) -> None:
    provider = FakeTTS(fake_audio_duration=1.0, fake_exception=error, fake_exception_count=1)
    adapter = tts.StreamAdapter(tts=provider)
    provider_errors: list[tts.TTSError] = []
    adapter_errors: list[tts.TTSError] = []
    provider.on("error", provider_errors.append)
    adapter.on("error", adapter_errors.append)
    frames: list[tts.SynthesizedAudio] = []
    try:
        async with adapter.stream(
            conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0)
        ) as stream:
            stream.push_text("Hello world.")
            stream.end_input()
            with pytest.raises(type(error)) as raised:
                async for audio in stream:
                    frames.append(audio)

        assert raised.value is error
        assert 0 < sum(audio.frame.duration for audio in frames) <= 1.0
        assert (await provider.synthesize_ch.recv()).attempt == 1
        assert [event.recoverable for event in provider_errors] == [False]
        assert [event.recoverable for event in adapter_errors] == [False]
    finally:
        await adapter.aclose()
        await provider.aclose()


@pytest.mark.parametrize("streaming", [False, True])
async def test_batch_fallback_preserves_its_consumers_retry_mode(streaming: bool) -> None:
    primary = FakeTTS(
        fake_audio_duration=1.0, fake_exception=APIConnectionError(), fake_exception_count=1
    )
    backup = FakeTTS(fake_audio_duration=1.0)
    primary._capabilities.streaming = backup._capabilities.streaming = False
    fallback = tts.FallbackAdapter([primary, backup], max_retry_per_tts=1)
    adapter = tts.StreamAdapter(tts=fallback)
    try:
        if streaming:
            async with adapter.stream() as stream:
                stream.push_text("Hello world.")
                stream.end_input()
                frames = [audio async for audio in stream]
        else:
            async with fallback.synthesize("Hello world.") as stream:
                frames = [audio async for audio in stream]

        # FallbackAdapter ends gracefully after partial audio and does not switch
        # providers. Buffered synthesis may still restart the same provider.
        assert (await primary.synthesize_ch.recv()).attempt == (1 if streaming else 2)
        duration = sum(audio.frame.duration for audio in frames)
        assert 0 < duration <= 1.03 if streaming else duration > 1.9
        with pytest.raises(ChanEmpty):
            backup.synthesize_ch.recv_nowait()
    finally:
        await adapter.aclose()
        await fallback.aclose()
        await primary.aclose()
        await backup.aclose()


@pytest.mark.parametrize(
    "error",
    [APIConnectionError(), APITimeoutError(), APIStatusError("rate limited", status_code=429)],
)
async def test_stream_adapter_retries_before_audio(error: Exception) -> None:
    provider = FakeTTS(fake_audio_duration=0.0, fake_exception=error, fake_exception_count=1)
    adapter = tts.StreamAdapter(tts=provider)
    errors: list[tts.TTSError] = []
    provider.on("error", errors.append)
    provider.on("error", lambda _: provider.update_options(fake_audio_duration=1.0))
    try:
        async with adapter.stream(
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0)
        ) as stream:
            stream.push_text("Hello world.")
            stream.end_input()
            frames = [audio async for audio in stream]

        assert 1.0 <= sum(audio.frame.duration for audio in frames) <= 1.03
        assert (await provider.synthesize_ch.recv()).attempt == 2
        assert [event.recoverable for event in errors] == [True]
    finally:
        await adapter.aclose()
        await provider.aclose()


@pytest.mark.parametrize("max_retry", [0, 1, 3])
async def test_stream_adapter_honors_retry_limits(max_retry: int) -> None:
    provider = FakeTTS(fake_exception=APIConnectionError())
    adapter = tts.StreamAdapter(tts=provider)
    errors: list[tts.TTSError] = []
    provider.on("error", errors.append)
    try:
        async with adapter.stream(
            conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.0)
        ) as stream:
            stream.push_text("Hello world.")
            stream.end_input()
            with pytest.raises(APIConnectionError):
                async for _ in stream:
                    pytest.fail("a failed attempt should not produce audio")

        assert (await provider.synthesize_ch.recv()).attempt == max_retry + 1
        assert [event.recoverable for event in errors] == [True] * max_retry + [False]
    finally:
        await adapter.aclose()
        await provider.aclose()


@pytest.mark.parametrize("use_adapter", [False, True])
async def test_buffered_synthesis_retains_partial_audio_retries(use_adapter: bool) -> None:
    provider = FakeTTS(
        fake_audio_duration=1.0, fake_exception=APIConnectionError(), fake_exception_count=1
    )
    adapter = tts.StreamAdapter(tts=provider)
    try:
        async with (adapter if use_adapter else provider).synthesize(
            "Hello world.", conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0)
        ) as stream:
            frames = [audio async for audio in stream]

        assert (await provider.synthesize_ch.recv()).attempt == 2
        request_ids = list(dict.fromkeys(audio.request_id for audio in frames))
        assert len(request_ids) == 2
        assert (
            1.0
            <= sum(audio.frame.duration for audio in frames if audio.request_id == request_ids[-1])
            <= 1.02
        )
    finally:
        await adapter.aclose()
        await provider.aclose()


class _FailingSecondSentenceTTS(FakeTTS):
    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> FakeChunkedStream:
        if text == "Second sentence.":
            self.update_options(fake_audio_duration=0.0, fake_exception=APIConnectionError())
        return super().synthesize(text, conn_options=conn_options)


async def test_stream_adapter_can_retry_before_audio_of_a_later_sentence() -> None:
    provider = _FailingSecondSentenceTTS(fake_audio_duration=1.0, fake_exception_count=1)
    adapter = tts.StreamAdapter(
        tts=provider, sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(min_sentence_len=1)
    )
    errors: list[tts.TTSError] = []
    provider.on("error", errors.append)
    provider.on("error", lambda _: provider.update_options(fake_audio_duration=1.0))
    try:
        async with adapter.stream(
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0)
        ) as stream:
            stream.push_text("First sentence. Second sentence.")
            stream.end_input()
            frames = [audio async for audio in stream]

        assert 2.0 <= sum(audio.frame.duration for audio in frames) <= 2.04
        assert (await provider.synthesize_ch.recv()).attempt == 1
        assert (await provider.synthesize_ch.recv()).attempt == 2
        assert [event.recoverable for event in errors] == [True]
    finally:
        await adapter.aclose()
        await provider.aclose()
