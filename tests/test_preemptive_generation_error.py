from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, Callable

import pytest

from livekit.agents import Agent, AgentSession, APIConnectionError, ModelSettings, llm
from livekit.agents.voice.audio_recognition import (
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
    _PreemptiveGenerationInfo,
)

from .fake_llm import FakeLLM

pytestmark = pytest.mark.unit


class _FailFirstAgent(Agent):
    def __init__(self, failure: str | None, *, gated: bool = False) -> None:
        super().__init__(instructions="Answer the question.")
        self.failure = failure
        self.calls = 0
        self.generated = asyncio.Event()
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        if not gated:
            self.gate.set()

    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterable[str]:
        self.calls += 1
        self.started.set()
        await self.gate.wait()
        if self.calls == 1 and self.failure is not None:
            if self.failure == "partial":
                yield "unfinished answer"
            raise APIConnectionError("provider disconnected", retryable=False)
        yield "complete answer"
        self.generated.set()


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(2.0):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("failure", ["before_tokens", "partial", None])
async def test_failed_preemptive_generation_is_not_scheduled(failure: str | None) -> None:
    agent = _FailFirstAgent(failure)
    session: AgentSession[None] = AgentSession(llm=FakeLLM())
    await session.start(agent=agent)
    activity = session._activity
    assert activity is not None

    try:
        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="What is the answer?",
                transcript_confidence=1.0,
                started_speaking_at=time.time(),
            )
        )
        preemptive = activity._preemptive_generation
        assert preemptive is not None
        handle = preemptive.speech_handle

        if failure is not None:
            await _wait_until(lambda: handle._error is not None)
        else:
            await asyncio.wait_for(agent.generated.wait(), timeout=2.0)
        assert not handle.scheduled

        activity.on_end_of_turn(
            _EndOfTurnInfo(
                skip_reply=False,
                new_transcript="What is the answer?",
                transcript_confidence=1.0,
                metrics=_EndOfTurnMetrics(
                    started_speaking_at=None,
                    stopped_speaking_at=None,
                    transcription_delay=None,
                    end_of_turn_delay=None,
                ),
            )
        )
        await _wait_until(lambda: not activity._speech_tasks)

        assert agent.calls == (2 if failure else 1)
        assert [
            (item.role, item.text_content)
            for item in agent.chat_ctx.items
            if item.type == "message" and item.role != "system"
        ] == [("user", "What is the answer?"), ("assistant", "complete answer")]
        assert activity._preemptive_generation is None
        assert handle.interrupted == (failure is not None)
    finally:
        await session.aclose()


@pytest.mark.parametrize("failure", [None, "partial"])
async def test_pending_preemptive_generation_is_reused(failure: str | None) -> None:
    agent = _FailFirstAgent(failure, gated=True)
    session: AgentSession[None] = AgentSession(llm=FakeLLM())
    await session.start(agent=agent)
    activity = session._activity
    assert activity is not None

    try:
        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="What is the answer?",
                transcript_confidence=1.0,
                started_speaking_at=time.time(),
            )
        )
        preemptive = activity._preemptive_generation
        assert preemptive is not None
        handle = preemptive.speech_handle
        await asyncio.wait_for(agent.started.wait(), timeout=2.0)

        activity.on_end_of_turn(
            _EndOfTurnInfo(
                skip_reply=False,
                new_transcript="What is the answer?",
                transcript_confidence=1.0,
                metrics=_EndOfTurnMetrics(
                    started_speaking_at=None,
                    stopped_speaking_at=None,
                    transcription_delay=None,
                    end_of_turn_delay=None,
                ),
            )
        )
        await _wait_until(lambda: handle.scheduled)
        agent.gate.set()
        await _wait_until(lambda: not activity._speech_tasks)

        assert agent.calls == 1
        assert not handle.interrupted
        assert (handle.exception() is not None) == (failure is not None)
        assert [
            item.text_content
            for item in agent.chat_ctx.items
            if item.type == "message" and item.role == "assistant"
        ] == ["unfinished answer" if failure else "complete answer"]
    finally:
        agent.gate.set()
        await session.aclose()
