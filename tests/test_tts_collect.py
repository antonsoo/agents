from __future__ import annotations

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, utils
from livekit.agents.tts import AudioEmitter, SynthesizedAudio

from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("request_ids", [("", ""), ("provider-first", "provider-second")])
async def test_attempts_have_distinct_request_ids(request_ids: tuple[str, str]) -> None:
    channel = utils.aio.Chan[SynthesizedAudio]()
    emitted_ids = []
    for request_id in request_ids:
        emitter = AudioEmitter(label="test", dst_ch=channel)
        emitter.initialize(
            request_id=request_id, sample_rate=24000, num_channels=1, mime_type="audio/pcm"
        )
        try:
            emitter.push_frame(rtc.AudioFrame.create(24000, 1, 480))
            emitter.end_input()
            await emitter.join()
        finally:
            await emitter.aclose()
        events = []
        while not channel.empty():
            events.append(channel.recv_nowait())
        assert events
        assert len({event.request_id for event in events}) == 1
        emitted_ids.append(events[0].request_id)
    channel.close()

    assert emitted_ids[0] != emitted_ids[1]
    if all(request_ids):
        assert emitted_ids == list(request_ids)


@pytest.mark.parametrize("failures", [0, 1, 2])
@pytest.mark.parametrize("num_channels", [1, 2])
async def test_collect_keeps_only_the_successful_attempt(failures: int, num_channels: int) -> None:
    control = FakeTTS(sample_rate=24000, num_channels=num_channels, fake_audio_duration=1.0)
    async with control.synthesize("hello") as stream:
        expected = await stream.collect()
    assert expected.duration == pytest.approx(1.0, abs=0.02)

    tts = FakeTTS(
        sample_rate=24000,
        num_channels=num_channels,
        fake_audio_duration=1.0,
        fake_exception=APIConnectionError("connection lost"),
        fake_exception_count=failures,
    )
    options = APIConnectOptions(max_retry=2, retry_interval=0.0)
    async with tts.synthesize("hello", conn_options=options) as stream:
        audio = await stream.collect()
        assert stream.attempt == failures + 1

    assert audio.sample_rate == tts.sample_rate
    assert audio.num_channels == num_channels
    assert audio.samples_per_channel == expected.samples_per_channel
    assert audio.data.tobytes() == expected.data.tobytes()


async def test_collect_propagates_failure_after_partial_audio() -> None:
    tts = FakeTTS(
        fake_audio_duration=1.0,
        fake_exception=APIConnectionError("connection lost"),
    )
    options = APIConnectOptions(max_retry=1, retry_interval=0.0)
    async with tts.synthesize("hello", conn_options=options) as stream:
        with pytest.raises(APIConnectionError, match="connection lost"):
            await stream.collect()
        assert stream.attempt == 2
