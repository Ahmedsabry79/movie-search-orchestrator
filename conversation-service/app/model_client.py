from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from .config import Settings
from .schemas import AgentPlan, MemorySummary


class ModelContextLengthError(RuntimeError):
    """Raised when the generation server rejects a request for exceeding context."""


class ModelGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.model_base_url,
            api_key=settings.model_api_key or "unused",
            timeout=settings.model_timeout_seconds,
        )

    async def close(self) -> None:
        await self.client.close()

    @staticmethod
    def _translate_bad_request(exc: BadRequestError) -> None:
        text = str(exc).lower()
        context_markers = (
            "maximum context length",
            "context length",
            "input_tokens",
            "input tokens",
            "too many tokens",
            "prompt contains",
        )
        if any(marker in text for marker in context_markers):
            raise ModelContextLengthError(str(exc)) from exc
        raise exc

    async def create_plan(self, *, planner_prompt: str, messages: list[dict[str, Any]]) -> AgentPlan:
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.generator_model,
                messages=[{"role": "system", "content": planner_prompt}, *messages],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "movie_agent_plan",
                        "strict": True,
                        "schema": AgentPlan.model_json_schema(),
                    },
                },
                temperature=0.1,
                max_tokens=self.settings.planner_max_tokens,
                extra_body={"reasoning_effort": self.settings.planning_reasoning_effort},
            )
        except BadRequestError as exc:
            self._translate_bad_request(exc)
            raise

        content = response.choices[0].message.content or ""
        try:
            return AgentPlan.model_validate_json(content)
        except Exception:
            # Some compatible servers wrap JSON in incidental whitespace/code fences.
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return AgentPlan.model_validate(json.loads(content))


    async def compact_memory(
        self,
        *,
        existing_memory: dict | None,
        messages: list[dict[str, Any]],
    ) -> MemorySummary:
        """Summarize expired conversation turns into durable structured memory.

        This is an internal orchestration call: no MCP tools are exposed and the
        result is persisted by movie-db-app. It preserves active constraints,
        resolved entities/IDs, corrections, established tool-grounded facts and
        unresolved references while dropping conversational filler.
        """
        prompt = (
            "You maintain compact long-term memory for a movie-catalog assistant. "
            "Merge the existing memory with the newly expired conversation messages. "
            "Preserve current user constraints, corrections/overrides, resolved movie/person "
            "names and IDs, facts explicitly established in the conversation, important "
            "retrieval outcomes, unresolved questions, and references such as 'that movie'. "
            "Discard greetings, filler, obsolete constraints, repeated wording, verbose tool "
            "payloads, and private reasoning. Never invent facts. If a later message corrects "
            "an earlier constraint, retain only the active corrected constraint."
        )
        payload = {
            "existing_memory": existing_memory or {},
            "new_messages": messages,
        }
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.generator_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "conversation_memory",
                        "strict": True,
                        "schema": MemorySummary.model_json_schema(),
                    },
                },
                temperature=0.0,
                max_tokens=self.settings.memory_compaction_max_tokens,
                extra_body={"reasoning_effort": "low"},
            )
        except BadRequestError as exc:
            self._translate_bad_request(exc)
            raise
        content = (response.choices[0].message.content or "{}").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return MemorySummary.model_validate_json(content)
        except Exception:
            return MemorySummary.model_validate(json.loads(content))

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str = "auto",
        max_tokens: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.settings.generator_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens or self.settings.agent_max_tokens,
            "extra_body": {"reasoning_effort": self.settings.agent_reasoning_effort},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            self._translate_bad_request(exc)
            raise
        return response.choices[0].message

    async def stream_text(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream only user-visible answer text; model reasoning remains internal."""
        kwargs: dict[str, Any] = {
            "model": self.settings.generator_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens or self.settings.agent_max_tokens,
            "stream": True,
            "extra_body": {"reasoning_effort": self.settings.agent_reasoning_effort},
        }
        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            self._translate_bad_request(exc)
            raise

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except BadRequestError as exc:
            self._translate_bad_request(exc)
            raise

    @staticmethod
    def assistant_message_dict(message: Any) -> dict[str, Any]:
        dumped = message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else {}
        result: dict[str, Any] = {"role": "assistant"}
        if dumped.get("content") is not None:
            result["content"] = dumped["content"]
        if dumped.get("tool_calls"):
            result["tool_calls"] = dumped["tool_calls"]
        # vLLM's interleaved-thinking examples require feeding reasoning back on the
        # assistant tool-call message. It remains internal and is never emitted to users.
        reasoning = dumped.get("reasoning")
        if reasoning:
            result["reasoning"] = reasoning
        return result
