from __future__ import annotations

import logging
from typing import Any

from .agent import MovieAgent
from .config import Settings
from .model_client import ModelContextLengthError
from .runtime_client import RuntimeStoreClient
from .schemas import AgentResult

logger = logging.getLogger(__name__)

# Token deltas are high-volume transport events. Persisting each one would create
# hundreds/thousands of event rows for a single answer. The final assistant message
# is still persisted as a normal conversation message.
EPHEMERAL_EVENTS = {"assistant_delta", "response_started"}


class ConversationService:
    def __init__(self, settings: Settings, runtime: RuntimeStoreClient, agent: MovieAgent):
        self.settings = settings
        self.runtime = runtime
        self.agent = agent


    @staticmethod
    def _older_messages_for_compaction(snapshot: dict, *, active_user_turns: int) -> list[dict[str, Any]]:
        messages = [
            message
            for message in snapshot.get("messages", [])
            if message.get("role") in {"user", "assistant"} and message.get("content")
        ]
        if not messages:
            return []

        # Find the first message belonging to the active recent-window. Everything
        # before it is eligible for durable compaction.
        user_turns = 0
        active_start = len(messages)
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                user_turns += 1
                if user_turns >= max(1, active_user_turns):
                    active_start = index
                    break
        older = messages[:active_start]
        if not older:
            return []

        conversation = snapshot.get("conversation") or {}
        marker = conversation.get("memory_compacted_through_message_id")
        if marker:
            marker_indexes = [i for i, item in enumerate(older) if item.get("id") == marker]
            if marker_indexes:
                older = older[marker_indexes[-1] + 1 :]

        return older

    async def _maybe_compact_memory(
        self,
        *,
        conversation_id: str,
        snapshot: dict,
        emit,
    ) -> dict:
        if not self.settings.memory_compaction_enabled:
            return snapshot

        candidates = self._older_messages_for_compaction(
            snapshot, active_user_turns=self.settings.conversation_history_turns
        )
        if len(candidates) < self.settings.memory_compaction_min_messages:
            return snapshot

        # Compact oldest newly-expired messages first, bounded so memory maintenance
        # can never become the cause of another context-window failure.
        selected: list[dict[str, Any]] = []
        char_count = 0
        for item in candidates:
            content = str(item.get("content") or "")
            remaining = self.settings.memory_compaction_max_chars - char_count
            if remaining <= 0:
                break
            if len(content) > remaining:
                content = content[:remaining] + "…"
            selected.append({"role": item.get("role"), "content": content})
            char_count += len(content)
            if char_count >= self.settings.memory_compaction_max_chars:
                break

        if not selected:
            return snapshot

        through_message = candidates[len(selected) - 1]
        await emit(
            "memory_compaction_started",
            {
                "message_count": len(selected),
                "message": "Compressing older conversation context into durable memory…",
            },
        )
        try:
            memory = await self.agent.model.compact_memory(
                existing_memory=(snapshot.get("conversation") or {}).get("memory_summary"),
                messages=selected,
            )
            record = await self.runtime.update_memory(
                conversation_id,
                summary=memory.model_dump(mode="json"),
                compacted_through_message_id=str(through_message["id"]),
            )
            snapshot.setdefault("conversation", {}).update(record)
            await emit(
                "memory_compaction_completed",
                {
                    "message_count": len(selected),
                    "through_message_id": through_message["id"],
                    "facts": len(memory.established_facts),
                    "constraints": len(memory.active_constraints),
                    "entities": len(memory.entities),
                },
            )
        except Exception as exc:
            # Memory is an optimization/continuity layer; it must never block the
            # user's current request. The full transcript remains safely persisted.
            logger.warning("Conversation memory compaction skipped after failure: %s", exc)
            await emit(
                "memory_compaction_skipped",
                {"message": "Long-term memory refresh was skipped; continuing with recent context."},
            )
        return snapshot

    async def create_conversation(self, *, title: str | None, metadata: dict) -> dict:
        return await self.runtime.create_conversation(title=title, metadata=metadata)

    async def get_conversation(self, conversation_id: str) -> dict:
        return await self.runtime.get_conversation(
            conversation_id,
            message_limit=50,
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
            if event_type not in EPHEMERAL_EVENTS:
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
            # Fetch a bounded transcript window large enough to incrementally compact
            # expired turns. MovieAgent still sends only compacted memory + the recent
            # active turn window to the model. Full persistence remains in DB App.
            snapshot = await self.runtime.get_conversation(
                conversation_id,
                message_limit=max(
                    8,
                    self.settings.conversation_history_turns * 2 + 2,
                    self.settings.memory_compaction_message_limit,
                ),
                event_limit=1,
            )
            snapshot = await self._maybe_compact_memory(
                conversation_id=conversation_id, snapshot=snapshot, emit=emit
            )

            try:
                answer, plan, tool_calls, tool_failures = await self.agent.run(
                    snapshot=snapshot,
                    user_text=content,
                    emit=emit,
                    history_user_turns=self.settings.conversation_history_turns,
                    recovery_mode=False,
                )
            except ModelContextLengthError as exc:
                # Restart the COMPLETE cycle (planning -> tools -> answer) once with
                # only the immediately previous completed turn plus the current user
                # request. This is intentionally not a continuation of the failed cycle.
                await emit(
                    "context_recovery",
                    {
                        "reason": "model_context_window",
                        "message": (
                            "The model context became too large. Restarting this request with only "
                            "the immediately previous turn for continuity."
                        ),
                    },
                )
                logger.warning("Restarting agent cycle after context-window rejection: %s", exc)
                recovery_snapshot = await self.runtime.get_conversation(
                    conversation_id,
                    message_limit=6,
                    event_limit=1,
                )
                answer, plan, tool_calls, tool_failures = await self.agent.run(
                    snapshot=recovery_snapshot,
                    user_text=content,
                    emit=emit,
                    # current user + N previous completed turns
                    history_user_turns=max(1, self.settings.recovery_previous_turns + 1),
                    recovery_mode=True,
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
