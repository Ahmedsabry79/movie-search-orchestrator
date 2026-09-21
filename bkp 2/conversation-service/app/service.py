from __future__ import annotations

import logging
from typing import Any

from .agent import MovieAgent
from .config import Settings
from .runtime_client import RuntimeStoreClient
from .schemas import AgentResult

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(self, settings: Settings, runtime: RuntimeStoreClient, agent: MovieAgent):
        self.settings = settings
        self.runtime = runtime
        self.agent = agent

    async def create_conversation(self, *, title: str | None, metadata: dict) -> dict:
        return await self.runtime.create_conversation(title=title, metadata=metadata)

    async def get_conversation(self, conversation_id: str) -> dict:
        return await self.runtime.get_conversation(
            conversation_id,
            message_limit=max(self.settings.conversation_history_limit, 50),
            event_limit=500,
        )

    async def run_message(
        self,
        *,
        conversation_id: str,
        content: str,
        on_event=None,
    ) -> AgentResult:
        user_message = await self.runtime.append_message(
            conversation_id, role="user", content=content, metadata={}
        )
        run = await self.runtime.create_run(
            conversation_id, user_message_id=user_message["id"], plan=None
        )
        run_id = run["id"]

        async def emit(event_type: str, payload: dict[str, Any]) -> None:
            try:
                if event_type == "plan":
                    await self.runtime.update_run(run_id, plan=payload)
                await self.runtime.append_event(run_id, event_type=event_type, payload=payload)
            except Exception:
                logger.exception("Failed to persist agent event %s", event_type)
            if on_event is not None:
                await on_event(event_type, payload)

        try:
            await emit("run_started", {"run_id": run_id, "conversation_id": conversation_id})
            snapshot = await self.runtime.get_conversation(
                conversation_id,
                message_limit=self.settings.conversation_history_limit,
                event_limit=1,
            )
            answer, plan, tool_calls, tool_failures = await self.agent.run(
                snapshot=snapshot,
                user_text=content,
                emit=emit,
            )
            await self.runtime.update_run(run_id, plan=plan.model_dump(mode="json"))
            assistant = await self.runtime.append_message(
                conversation_id,
                role="assistant",
                content=answer,
                metadata={"run_id": run_id},
            )
            await self.runtime.update_run(
                run_id,
                status="completed",
                assistant_message_id=assistant["id"],
            )
            await emit(
                "completed",
                {"tool_calls": tool_calls, "tool_failures": tool_failures},
            )
            return AgentResult(
                conversation_id=conversation_id,
                run_id=run_id,
                answer=answer,
                plan=plan,
                tool_calls=tool_calls,
                tool_failures=tool_failures,
            )
        except Exception as exc:
            logger.exception("Agent run failed")
            try:
                await self.runtime.update_run(run_id, status="failed", error=str(exc)[:20_000])
                await emit("error", {"message": "The agent run failed before a reliable answer was produced."})
            except Exception:
                logger.exception("Failed to persist run failure")
            raise
