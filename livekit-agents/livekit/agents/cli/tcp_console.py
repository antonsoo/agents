from __future__ import annotations

import asyncio
import time
from typing import cast

from livekit import rtc
from livekit.protocol.agent_pb import agent_session as agent_pb

from ..log import logger
from ..voice import io
from ..voice.remote_session import TcpSessionTransport

WIRE_SAMPLE_RATE = 48000
AGENT_SAMPLE_RATE = 24000
_BYTES_PER_SAMPLE = 2  # mono, 16-bit
_RESEND_CHUNK_BYTES = WIRE_SAMPLE_RATE // 50 * _BYTES_PER_SAMPLE  # 20 ms

_SENTINEL = object()


class TcpAudioInput(io.AudioInput):
    """Audio input bridging the producer loop to the agent-session loop.

    push_frame runs on the transport host's loop while __anext__ runs on the
    agent-session loop; these differ when JobExecutorType.THREAD is used. Frames
    are marshalled onto the consumer loop via call_soon_threadsafe, so the queue
    is a plain asyncio.Queue owned by that loop (mirrors TcpAudioOutput).
    """

    def __init__(self) -> None:
        super().__init__(label="TCP Console")
        self._queue: asyncio.Queue[rtc.AudioFrame | object] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._resampler = rtc.AudioResampler(
            input_rate=WIRE_SAMPLE_RATE,
            output_rate=AGENT_SAMPLE_RATE,
            num_channels=1,
        )
        self._closed = False

    def push_frame(self, frame: agent_pb.AgentSessionMessage.ConsoleIO.AudioFrame) -> None:
        # the consumer loop is captured on the first __anext__; frames pushed before
        # that startup window are dropped (same assumption as TcpAudioOutput).
        if self._closed or self._loop is None:
            return
        audio_frame = rtc.AudioFrame(
            data=frame.data,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )
        resampled = self._resampler.push(audio_frame)
        for rf in resampled:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, rf)

    def close(self) -> None:
        """Unblock any waiting consumer and mark as closed."""
        self._closed = True
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, _SENTINEL)

    async def __anext__(self) -> rtc.AudioFrame:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        item = await self._queue.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        return cast(rtc.AudioFrame, item)


class TcpAudioOutput(io.AudioOutput):
    def __init__(self, transport: TcpSessionTransport) -> None:
        super().__init__(
            label="TCP Console",
            next_in_chain=None,
            sample_rate=AGENT_SAMPLE_RATE,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self._transport = transport
        self._resampler = rtc.AudioResampler(
            input_rate=AGENT_SAMPLE_RATE,
            output_rate=WIRE_SAMPLE_RATE,
            num_channels=1,
        )

        self._pushed_duration: float = 0.0
        self._capture_start: float = 0.0
        self._flush_task: asyncio.Task[None] | None = None
        self._playout_done = asyncio.Event()
        self._interrupted_ev = asyncio.Event()
        self._agent_loop: asyncio.AbstractEventLoop | None = None

        # The client plays whatever it has been sent and the protocol has no pause, so a
        # pause takes back the audio it has not played yet (a clear) and a resume sends
        # that part again. For that the current segment is kept here as it went out.
        self._segment_pcm = bytearray()
        self._resume_pos = 0  # bytes of _segment_pcm played when the pause began
        self._flush_requested = False
        self._paused_at: float | None = None
        self._paused_duration: float = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

        if self._agent_loop is None:
            self._agent_loop = asyncio.get_running_loop()

        if self._flush_task and not self._flush_task.done():
            logger.error("capture_frame called while previous flush is in progress")
            await self._flush_task

        if not self._pushed_duration:
            self._capture_start = time.monotonic()
            self._paused_duration = 0.0
            self._resume_pos = 0
            self._flush_requested = False
            if self._paused_at is not None:
                # a segment that starts during a pause has played nothing yet
                self._paused_at = self._capture_start
            self.on_playback_started(created_at=time.time())

        self._pushed_duration += frame.duration

        resampled = self._resampler.push(frame)
        for rf in resampled:
            data = bytes(rf.data)
            if self._paused_at is None:
                self._exclude_underrun()
            self._segment_pcm += data
            if self._paused_at is None:
                self._send_audio(data)

    def flush(self) -> None:
        super().flush()
        self._flush_requested = True
        if self._paused_at is None:
            # while paused the client has nothing buffered and would report the playout
            # as finished at once; resume() sends the flush after the remaining audio
            self._send_flush()

        if self._pushed_duration:
            if self._flush_task and not self._flush_task.done():
                logger.error("flush called while previous flush is in progress")
                self._flush_task.cancel()

            self._playout_done.clear()
            self._interrupted_ev.clear()
            self._flush_task = asyncio.create_task(self._wait_for_playout())

    def clear_buffer(self) -> None:
        self._send_clear()

        if self._pushed_duration:
            self._interrupted_ev.set()

    def pause(self) -> None:
        super().pause()

        if self._paused_at is not None:
            return

        self._exclude_underrun()
        self._paused_at = time.monotonic()
        if self._pushed_duration:
            played = int(self._played_duration() * WIRE_SAMPLE_RATE) * _BYTES_PER_SAMPLE
            self._resume_pos = min(played, len(self._segment_pcm))
            self._send_clear()

    def resume(self) -> None:
        super().resume()

        if self._paused_at is None or self._interrupted_ev.is_set():
            return

        paused_at, self._paused_at = self._paused_at, None
        if not self._pushed_duration:
            return

        self._paused_duration += time.monotonic() - paused_at
        remaining = memoryview(self._segment_pcm)[self._resume_pos :]
        for i in range(0, len(remaining), _RESEND_CHUNK_BYTES):
            self._send_audio(bytes(remaining[i : i + _RESEND_CHUNK_BYTES]))
        if self._flush_requested:
            self._send_flush()

    def _played_duration(self) -> float:
        """Seconds of the current segment the client has played, going by the clock."""
        now = time.monotonic()
        paused = self._paused_duration
        if self._paused_at is not None:
            paused += now - self._paused_at
        sent_duration = len(self._segment_pcm) / (WIRE_SAMPLE_RATE * _BYTES_PER_SAMPLE)
        return min(max(0.0, now - self._capture_start - paused), sent_duration)

    def _exclude_underrun(self) -> None:
        """Exclude silence after the client exhausts the audio sent so far."""
        sent_duration = len(self._segment_pcm) / (WIRE_SAMPLE_RATE * _BYTES_PER_SAMPLE)
        self._capture_start = max(
            self._capture_start, time.monotonic() - self._paused_duration - sent_duration
        )

    def _send_audio(self, data: bytes) -> None:
        audio_frame = agent_pb.AgentSessionMessage.ConsoleIO.AudioFrame(
            data=data,
            sample_rate=WIRE_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=len(data) // _BYTES_PER_SAMPLE,
        )
        self._transport.send_message_threadsafe(
            agent_pb.AgentSessionMessage(audio_output=audio_frame)
        )

    def _send_flush(self) -> None:
        msg = agent_pb.AgentSessionMessage(
            audio_playback_flush=agent_pb.AgentSessionMessage.ConsoleIO.AudioPlaybackFlush()
        )
        self._transport.send_message_threadsafe(msg)

    def _send_clear(self) -> None:
        msg = agent_pb.AgentSessionMessage(
            audio_playback_clear=agent_pb.AgentSessionMessage.ConsoleIO.AudioPlaybackClear()
        )
        self._transport.send_message_threadsafe(msg)

    def notify_playout_finished(self) -> None:
        if self._agent_loop is not None:
            self._agent_loop.call_soon_threadsafe(self._playout_done.set)
        else:
            self._playout_done.set()

    async def _wait_for_playout(self) -> None:
        wait_done = asyncio.create_task(self._playout_done.wait())
        wait_interrupt = asyncio.create_task(self._interrupted_ev.wait())
        try:
            await asyncio.wait(
                [wait_done, wait_interrupt],
                return_when=asyncio.FIRST_COMPLETED,
            )
            interrupted = wait_interrupt.done() and not wait_done.done()
        finally:
            wait_done.cancel()
            wait_interrupt.cancel()

        played = self._played_duration() if interrupted else self._pushed_duration

        self.on_playback_finished(playback_position=played, interrupted=interrupted)

        self._pushed_duration = 0.0
        self._interrupted_ev.clear()
        self._segment_pcm.clear()
        self._resume_pos = 0
        self._flush_requested = False
        self._paused_at = None
        self._paused_duration = 0.0
