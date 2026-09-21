from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from .config import Settings
from .mcp_client import MCPGateway, MCPToolResult
from .model_client import ModelGateway
from .prompt import build_planner_prompt, build_system_prompt
from .schemas import AgentPlan, PlanStep

EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]


class MovieAgent:
    def __init__(self, settings: Settings, mcp: MCPGateway, model: ModelGateway):
        self.settings = settings
        self.mcp = mcp
        self.model = model

    @staticmethod
    def _history_messages(snapshot: dict) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for message in snapshot.get("messages", []):
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                result.append({"role": role, "content": content})
        return result

    @staticmethod
    def _fallback_plan(user_text: str) -> AgentPlan:
        semantic_cues = re.search(
            r"\b(about|plot|story|theme|where|who tries|trying to|meant something|similar to|feels like)\b",
            user_text,
            flags=re.IGNORECASE,
        )
        structured_cues = re.search(
            r"\b(title|tagline|genre|keyword|cast|actor|actress|starring|crew|director|writer|company|studio|country|language|budget|revenue|runtime|minutes|status|rating|rated|vote|popularity|released?|year)\b",
            user_text,
            flags=re.IGNORECASE,
        )
        strategy = "semantic_first" if semantic_cues and not structured_cues else "structured_first"
        action = "semantic_search" if strategy == "semantic_first" else "structured_search"
        return AgentPlan(
            strategy=strategy,
            summary="Retrieve movie candidates using the most appropriate catalog search path, then verify before answering.",
            extracted_criteria=[user_text[:500]],
            has_semantic_description=bool(semantic_cues),
            steps=[
                PlanStep(action=action, purpose="Find candidates matching the user's request."),
                PlanStep(action="fetch_details", purpose="Verify candidate metadata if needed."),
                PlanStep(action="answer", purpose="Answer from retrieved evidence."),
            ],
        )

    @staticmethod
    def _tool_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {tool["function"]["name"]: tool for tool in tools}

    @staticmethod
    def _trace_result(name: str, result: MCPToolResult) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": name,
            "is_error": result.is_error,
            "transport_attempts": result.transport_attempts,
        }
        structured = result.structured
        if isinstance(structured, dict):
            # Keep traces useful but bounded; the model receives the full result separately.
            for key in ("status", "count", "returned_count", "semantic_fallback_recommended", "query", "mode"):
                if key in structured:
                    payload[key] = structured[key]
            if isinstance(structured.get("results"), list):
                payload["result_preview"] = structured["results"][:5]
            elif "movie" in structured:
                payload["result_preview"] = {"movie": structured.get("movie")}
        if result.is_error:
            payload["error"] = result.text[:2000]
        return payload

    @staticmethod
    def _tool_message_content(result: MCPToolResult) -> str:
        if result.structured is not None:
            return json.dumps(result.structured, ensure_ascii=False, default=str)
        if result.text:
            return result.text
        return json.dumps({"error": result.is_error, "message": "Tool returned no content"})

    async def _plan(
        self,
        *,
        schema_guide: str,
        history: list[dict[str, Any]],
        user_text: str,
    ) -> AgentPlan:
        try:
            return await self.model.create_plan(
                planner_prompt=build_planner_prompt(schema_guide=schema_guide),
                messages=history,
            )
        except Exception:
            return self._fallback_plan(user_text)

    async def _final_without_tools(
        self,
        messages: list[dict[str, Any]],
        reason: str,
    ) -> str:
        final_messages = [
            *messages,
            {
                "role": "system",
                "content": (
                    "Tool execution is now disabled for this run. "
                    + reason
                    + " Give a concise, graceful answer using only facts already established by successful tool results "
                    "or the user's own messages. Do not invent missing movie facts."
                ),
            },
        ]
        message = await self.model.complete(messages=final_messages, tools=None, tool_choice="none")
        return (message.content or "I couldn't complete the movie search reliably with the available tools.").strip()

    async def run(
        self,
        *,
        snapshot: dict,
        user_text: str,
        emit: EmitFn,
    ) -> tuple[str, AgentPlan, int, int]:
        context = await self.mcp.get_context()
        history = self._history_messages(snapshot)
        plan = await self._plan(schema_guide=context.schema_guide, history=history, user_text=user_text)
        await emit("plan", plan.model_dump(mode="json"))

        system_prompt = build_system_prompt(
            schema_guide=context.schema_guide,
            tools=context.tools,
            mcp_instructions=context.instructions,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *history,
            {
                "role": "system",
                "content": "Current safe operational plan (you may adapt after tool observations): "
                + json.dumps(plan.model_dump(mode="json"), ensure_ascii=False),
            },
        ]

        if plan.strategy == "direct_answer":
            final = await self.model.complete(messages=messages, tools=None, tool_choice="none")
            answer = (final.content or "How can I help with the movie catalog?").strip()
            await emit("assistant_message", {"content": answer})
            return answer, plan, 0, 0

        tool_map = self._tool_map(context.tools)
        all_names = set(tool_map)
        semantic_enabled = plan.strategy == "semantic_first"
        db_search_attempted = False
        tool_calls_count = 0
        tool_failures = 0
        consecutive_failures = 0

        for _iteration in range(self.settings.max_agent_iterations):
            if plan.strategy == "structured_first" and not semantic_enabled:
                allowed_names = all_names - {"search_movies_semantic"}
            elif plan.strategy == "semantic_first" and not db_search_attempted:
                # Meaning-based requests should not accidentally lead with a SQL-style search.
                allowed_names = all_names - {"search_movies_db"}
            else:
                allowed_names = all_names

            tools = [tool_map[name] for name in tool_map if name in allowed_names]
            model_message = await self.model.complete(messages=messages, tools=tools, tool_choice="auto")
            calls = list(model_message.tool_calls or [])

            if not calls:
                answer = (model_message.content or "").strip()
                if not answer:
                    answer = await self._final_without_tools(
                        messages,
                        "The model ended the retrieval loop without a final textual answer.",
                    )
                await emit("assistant_message", {"content": answer})
                return answer, plan, tool_calls_count, tool_failures

            messages.append(self.model.assistant_message_dict(model_message))

            for call in calls:
                if tool_calls_count >= self.settings.max_tool_calls:
                    answer = await self._final_without_tools(
                        messages,
                        f"The run reached the maximum of {self.settings.max_tool_calls} tool calls.",
                    )
                    await emit("assistant_message", {"content": answer})
                    return answer, plan, tool_calls_count, tool_failures

                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                except Exception as exc:
                    tool_calls_count += 1
                    tool_failures += 1
                    consecutive_failures += 1
                    error_payload = {"error": True, "message": f"Invalid tool arguments: {exc}"}
                    await emit("tool_error", {"tool": name, **error_payload})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": name,
                            "content": json.dumps(error_payload),
                        }
                    )
                    if consecutive_failures >= self.settings.max_tool_failures:
                        answer = await self._final_without_tools(
                            messages,
                            f"Three consecutive tool failures occurred; the last failure was in {name}.",
                        )
                        await emit("assistant_message", {"content": answer})
                        return answer, plan, tool_calls_count, tool_failures
                    continue

                tool_calls_count += 1
                await emit("tool_started", {"tool": name, "arguments": arguments})
                try:
                    result = await self.mcp.call_tool(name, arguments)
                except Exception as exc:
                    result = MCPToolResult(
                        is_error=True,
                        structured=None,
                        text=str(exc),
                        transport_attempts=self.settings.mcp_transport_retries,
                    )

                await emit("tool_result", self._trace_result(name, result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": self._tool_message_content(result),
                    }
                )

                if result.is_error:
                    tool_failures += 1
                    consecutive_failures += 1
                    if consecutive_failures >= self.settings.max_tool_failures:
                        answer = await self._final_without_tools(
                            messages,
                            f"Three consecutive tool failures occurred; the last failing tool was {name}.",
                        )
                        await emit("assistant_message", {"content": answer})
                        return answer, plan, tool_calls_count, tool_failures
                    continue

                consecutive_failures = 0

                if name == "search_movies_db":
                    db_search_attempted = True
                    # Once structured retrieval has been observed, semantic search becomes
                    # available as a fallback if Gemma judges the match insufficient.
                    semantic_enabled = True
                    structured = result.structured if isinstance(result.structured, dict) else {}
                    if structured.get("status") == "no_matches" or structured.get("semantic_fallback_recommended") is True:
                        await emit(
                            "fallback_available",
                            {
                                "from": "search_movies_db",
                                "to": "search_movies_semantic",
                                "reason": "Structured/fuzzy retrieval returned no matches.",
                            },
                        )
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "Structured/fuzzy database retrieval returned no matches. If the user's request "
                                    "contains semantic plot/theme/tagline-meaning clues, use search_movies_semantic now. "
                                    "Do not repeat the identical DB search and do not claim strict filters are satisfied "
                                    "unless the fallback result can actually support them."
                                ),
                            }
                        )

                if name == "search_movies_semantic":
                    # After semantic retrieval, allow DB search only for verification/refinement
                    # if the model discovers an explicit canonical criterion from the candidates.
                    db_search_attempted = True

        answer = await self._final_without_tools(
            messages,
            f"The run reached the maximum of {self.settings.max_agent_iterations} agent iterations.",
        )
        await emit("assistant_message", {"content": answer})
        return answer, plan, tool_calls_count, tool_failures
