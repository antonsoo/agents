from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from livekit import rtc
from livekit.agents.cli import tcp_console
from livekit.agents.cli.tcp_console import AGENT_SAMPLE_RATE, WIRE_SAMPLE_RATE, TcpAudioOutput
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.protocol.agent_pb import agent_session as agent_pb

pytestmark = pytest.mark.unit


class _Client:
    """The console client's end of the transport: records what the agent sends."""

    def __init__(self) -> None:
        self.messages: list[agent_pb.AgentSessionMessage] = []

    def send_message_threadsafe(self, msg: agent_pb.AgentSessionMessage) -> None:
        self.messages.append(msg)

    def kinds(self) -> list[str]:
        return [str(m.WhichOneof("message")) for m in self.messages]

    def audio(self, start: int = 0) -> bytes:
        return b"".join(
            m.audio_output.data for m in self.messages[start:] if m.HasField("audio_output")
        )


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    # only the module's view of the clock: the event loop keeps the real one
    clock = _Clock()
    monkeypatch.setattr(
        tcp_console, "time", SimpleNamespace(monotonic=clock.monotonic, time=time.time)
    )
    return clock


async def _capture(out: TcpAudioOutput, seconds: float) -> None:
    samples = AGENT_SAMPLE_RATE // 100  # 10 ms
    for i in range(round(seconds * 100)):
        # a ramp, so that a resend from the wrong offset is visible
        data = b"".join(((i * samples + n) % 30000).to_bytes(2, "little") for n in range(samples))
        await out.capture_frame(rtc.AudioFrame(data, AGENT_SAMPLE_RATE, 1, samples))


def _played_bytes(seconds: float) -> int:
    return int(seconds * WIRE_SAMPLE_RATE) * 2


async def test_pause_takes_back_unplayed_audio_and_resume_sends_it_again(clock: _Clock) -> None:
    client = _Client()
    out = TcpAudioOutput(client)  # type: ignore[arg-type]
    finished: list[PlaybackFinishedEvent] = []
    out.on("playback_finished", finished.append)

    await _capture(out, 1.0)
    sent = client.audio()
    assert set(client.kinds()) == {"audio_output"}

    # the client has played 0.25 s of what it was sent; the rest sits in its buffer
    clock.now += 0.25
    out.pause()
    assert client.kinds()[-1] == "audio_playback_clear"

    # audio and the flush that arrive during the pause are held back: a client with an
    # empty buffer would acknowledge the flush at once
    held_from = len(client.messages)
    await _capture(out, 0.2)
    out.flush()
    await asyncio.sleep(0)
    assert client.kinds()[held_from:] == []
    assert finished == []

    clock.now += 2.0
    out.resume()
    resent = client.messages[held_from:]
    assert [str(m.WhichOneof("message")) for m in resent][-1] == "audio_playback_flush"
    assert all(m.HasField("audio_output") for m in resent[:-1])
    resent_audio = client.audio(held_from)
    assert resent_audio[: len(sent) - _played_bytes(0.25)] == sent[_played_bytes(0.25) :]
    assert len(resent_audio) > len(sent) - _played_bytes(0.25)  # plus what came during the pause

    # the client plays the rest and reports it
    clock.now += 0.95
    out.notify_playout_finished()
    await asyncio.sleep(0.01)
    assert len(finished) == 1
    assert finished[0].interrupted is False
    assert finished[0].playback_position == pytest.approx(1.2)


async def test_interruption_during_a_pause_reports_the_position_at_the_pause(clock: _Clock) -> None:
    client = _Client()
    out = TcpAudioOutput(client)  # type: ignore[arg-type]
    finished: list[PlaybackFinishedEvent] = []
    out.on("playback_finished", finished.append)

    await _capture(out, 1.0)
    clock.now += 0.5
    out.pause()

    clock.now += 3.0  # the time spent paused is not playback
    out.flush()
    out.clear_buffer()
    await asyncio.sleep(0.01)

    assert "audio_playback_flush" not in client.kinds()
    assert len(finished) == 1
    assert finished[0].interrupted is True
    assert finished[0].playback_position == pytest.approx(0.5)

    # the pause ended with the segment: the next one plays straight away
    out.resume()
    before = len(client.messages)
    await _capture(out, 0.1)
    assert client.kinds()[before:] and set(client.kinds()[before:]) == {"audio_output"}


async def test_a_pause_between_segments_does_not_shift_the_next_position(clock: _Clock) -> None:
    client = _Client()
    out = TcpAudioOutput(client)  # type: ignore[arg-type]
    finished: list[PlaybackFinishedEvent] = []
    out.on("playback_finished", finished.append)

    out.pause()
    clock.now += 5.0
    out.resume()
    assert client.messages == []

    await _capture(out, 1.0)
    clock.now += 0.5
    out.flush()
    out.clear_buffer()
    await asyncio.sleep(0.01)

    assert len(finished) == 1
    assert finished[0].interrupted is True
    assert finished[0].playback_position == pytest.approx(0.5)


async def test_playout_without_a_pause_is_unchanged(clock: _Clock) -> None:
    client = _Client()
    out = TcpAudioOutput(client)  # type: ignore[arg-type]
    finished: list[PlaybackFinishedEvent] = []
    out.on("playback_finished", finished.append)

    await _capture(out, 0.5)
    out.flush()
    assert client.kinds()[-1] == "audio_playback_flush"
    assert set(client.kinds()[:-1]) == {"audio_output"}

    clock.now += 0.5
    out.notify_playout_finished()
    await asyncio.sleep(0.01)
    assert len(finished) == 1
    assert finished[0].interrupted is False
    assert finished[0].playback_position == pytest.approx(0.5)

    # resume without a pause does nothing
    before = len(client.messages)
    out.resume()
    assert len(client.messages) == before
