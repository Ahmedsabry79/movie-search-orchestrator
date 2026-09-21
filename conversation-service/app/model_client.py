from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from .config import Settings
from .schemas import AgentPlan


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

    async def create_plan(self, *, planner_prompt: str, messages: list[dict[str, Any]]) -> AgentPlan:
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
        content = response.choices[0].message.content or ""
        try:
            return AgentPlan.model_validate_json(content)
        except Exception:
            # Some compatible servers wrap JSON in incidental whitespace/code fences.
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return AgentPlan.model_validate(json.loads(content))

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str = "auto",
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.settings.generator_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self.settings.agent_max_tokens,
            "extra_body": {"reasoning_effort": self.settings.agent_reasoning_effort},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

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
