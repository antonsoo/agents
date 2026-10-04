from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, APIConnectionError, APIConnectOptions
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities
from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer

from .fake_io import FakeAudioOutput, FakeTextOutput
from .fake_tts import FakeTTS, FakeTTSResponse

pytestmark = pytest.mark.unit
TEXT = "One two three four five six seven eight nine ten."


class _RecordingOutput(FakeAudioOutput):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.finished = asyncio.Event()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.started.set()

    def on_playback_finished(self, **kwargs) -> None:
        super().on_playback_finished(**kwargs)
        self.finished.set()


def _session(model: FakeTTS) -> tuple[AgentSession, _RecordingOutput, TranscriptSynchronizer]:
    session = AgentSession(
        tts=model,
        vad=None,
        turn_handling={"turn_detection": None},
        conn_options=SessionConnectOptions(tts_conn_options=APIConnectOptions(max_retry=0)),
    )
    output = _RecordingOutput()
    sync = TranscriptSynchronizer(next_in_chain_audio=output, next_in_chain_text=FakeTextOutput())
    session.output.audio = sync.audio_output
    session.output.transcription = sync.text_output
    return session, output, sync


@pytest.mark.parametrize("audio_duration", [0.0, 0.2])
async def test_say_reports_the_tts_failure(audio_duration: float) -> None:
    error = APIConnectionError("provider disconnected")
    model = FakeTTS(
        fake_exception=error,
        fake_exception_count=1,
        fake_responses=[
            FakeTTSResponse(input=TEXT, audio_duration=audio_duration, ttfb=0, duration=0.01)
        ],
    )
    session, _, sync = _session(model)
    try:
        await session.start(Agent(instructions="test"))
        handle = session.say(TEXT)
        await asyncio.wait_for(handle, 2)
        assert handle.done()
        assert handle.exception() is error
        assert not [item for item in session.history.items if item.type == "message"]
    finally:
        await session.aclose()
        await sync.aclose()


async def test_failed_say_finishes_playout_before_the_next_speech() -> None:
    model = FakeTTS(
        fake_audio_duration=0.1,
        fake_exception=APIConnectionError("provider disconnected"),
        fake_exception_count=1,
        fake_responses=[FakeTTSResponse(input=TEXT, audio_duration=0.2, ttfb=0, duration=0.01)],
    )
    session, output, sync = _session(model)
    try:
        await session.start(Agent(instructions="test"))
        failed = session.say(TEXT)
        await asyncio.wait_for(failed, 2)
        assert output.started.is_set()
        assert output.finished.is_set()
        await asyncio.wait_for(session.say("Recovered."), 2)
        messages = [item.text_content for item in session.history.items if item.type == "message"]
        assert messages == ["Recovered."]
    finally:
        await session.aclose()
        await sync.aclose()


@pytest.mark.parametrize("after_audio", [False, True])
async def test_say_reports_a_supplied_audio_source_failure(after_audio: bool) -> None:
    error = OSError("audio fixture unavailable")

    async def audio() -> AsyncIterator[rtc.AudioFrame]:
        if after_audio:
            yield rtc.AudioFrame(
                data=b"\0\0" * 2400, sample_rate=24000, num_channels=1, samples_per_channel=2400
            )
        raise error

    session, _, sync = _session(FakeTTS())
    try:
        await session.start(Agent(instructions="test"))
        handle = session.say(TEXT, audio=audio())
        await asyncio.wait_for(handle, 2)
        assert handle.exception() is error
    finally:
        await session.aclose()
        await sync.aclose()


async def test_successful_say_has_no_failure() -> None:
    session, _, sync = _session(FakeTTS(fake_audio_duration=0.1))
    try:
        await session.start(Agent(instructions="test"))
        handle = session.say(TEXT)
        await asyncio.wait_for(handle, 2)
        assert handle.exception() is None
        assert handle.chat_items[0].text_content == TEXT
    finally:
        await session.aclose()
        await sync.aclose()


async def test_interrupted_say_has_no_failure() -> None:
    session, output, sync = _session(FakeTTS(fake_audio_duration=1.0))
    try:
        await session.start(Agent(instructions="test"))
        handle = session.say(TEXT)
        await asyncio.wait_for(output.started.wait(), 2)
        handle.interrupt(force=True)
        await asyncio.wait_for(handle, 2)
        assert handle.interrupted
        assert handle.exception() is None
    finally:
        await session.aclose()
        await sync.aclose()


@pytest.mark.parametrize("tts_also_fails", [False, True])
async def test_say_does_not_wait_for_an_output_that_failed(tts_also_fails: bool) -> None:
    output_error = RuntimeError("playback device disconnected")
    tts_error = APIConnectionError("provider disconnected")

    class BrokenOutput(AudioOutput):
        def __init__(self) -> None:
            super().__init__(
                label="broken playback", capabilities=AudioOutputCapabilities(pause=False)
            )

        async def capture_frame(self, frame: rtc.AudioFrame) -> None:
            await super().capture_frame(frame)
            # The sink failed before playback started; it cannot acknowledge this segment.
            raise output_error

        def flush(self) -> None:
            super().flush()

        def clear_buffer(self) -> None:
            pass

    model = FakeTTS(
        fake_audio_duration=0.2,
        fake_exception=tts_error if tts_also_fails else None,
        fake_responses=[FakeTTSResponse(input=TEXT, audio_duration=0.2, ttfb=0, duration=0.01)],
    )
    session, _, sync = _session(model)
    session.output.audio = BrokenOutput()
    session.output.transcription = FakeTextOutput()
    try:
        await session.start(Agent(instructions="test"))
        handle = session.say(TEXT)
        await asyncio.wait_for(handle, 2)
        assert handle.exception() is (tts_error if tts_also_fails else output_error)
    finally:
        await session.aclose()
        await sync.aclose()
