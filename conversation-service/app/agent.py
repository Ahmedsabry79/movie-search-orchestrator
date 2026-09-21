from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from .config import Settings
from .mcp_client import MCPGateway, MCPToolResult
from .model_client import ModelContextLengthError, ModelGateway
from .prompt import build_planner_prompt, build_system_prompt
from .schemas import AgentPlan, PlanStep

EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]


class MovieAgent:
    # Tool observations are fed back to the model, but we never allow one unusually
    # large response to consume the entire model context window.
    MAX_TOOL_MESSAGE_CHARS = 24_000

    def __init__(self, settings: Settings, mcp: MCPGateway, model: ModelGateway):
        self.settings = settings
        self.mcp = mcp
        self.model = model

    @staticmethod
    def _history_messages(snapshot: dict, *, max_user_turns: int) -> list[dict[str, str]]:
        """Return only the most recent N user turns and their assistant replies.

        The current user message has already been persisted when the snapshot is read.
        Therefore max_user_turns=3 normally yields:
        previous user/assistant + previous user/assistant + current user.
        """
        filtered: list[dict[str, str]] = []
        for message in snapshot.get("messages", []):
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                filtered.append({"role": role, "content": content})

        if max_user_turns <= 0:
            # Keep the current user request if available, but no older history.
            for message in reversed(filtered):
                if message["role"] == "user":
                    return [message]
            return []

        selected_reversed: list[dict[str, str]] = []
        user_turns = 0
        for message in reversed(filtered):
            selected_reversed.append(message)
            if message["role"] == "user":
                user_turns += 1
                if user_turns >= max_user_turns:
                    break
        return list(reversed(selected_reversed))

    @staticmethod
    def _memory_message(snapshot: dict) -> dict[str, str] | None:
        conversation = snapshot.get("conversation") or {}
        memory = conversation.get("memory_summary")
        if not isinstance(memory, dict) or not memory:
            return None
        return {
            "role": "system",
            "content": (
                "Long-term compacted conversation memory. Treat this as continuity context only; "
                "the current/recent user messages override it when they conflict. Never treat it as "
                "new database evidence:\n"
                + json.dumps(memory, ensure_ascii=False, default=str)
            ),
        }

    def _structured_quality(self, structured: dict[str, Any]) -> tuple[str, float | None]:
        if structured.get("status") == "no_matches" or not structured.get("results"):
            return "no_match", None
        top = structured["results"][0] if isinstance(structured.get("results"), list) else {}
        evidence = top.get("fuzzy_matches") if isinstance(top, dict) else None
        if not evidence:
            # A successful search with only exact/range filters has no fuzzy uncertainty.
            return "strong", 1.0
        similarities = [
            float(item.get("similarity"))
            for item in evidence
            if isinstance(item, dict) and item.get("similarity") is not None
        ]
        if not similarities:
            return "weak", None
        minimum = min(similarities)
        if minimum >= self.settings.fuzzy_strong_similarity:
            return "strong", minimum
        if minimum >= self.settings.fuzzy_accept_similarity:
            return "acceptable", minimum
        return "weak", minimum

    @staticmethod
    def _semantic_quality(structured: dict[str, Any]) -> tuple[str, float | None]:
        quality = str(structured.get("top_quality") or "no_match")
        results = structured.get("results") or []
        score = None
        if results and isinstance(results[0], dict):
            value = results[0].get("normalized_score")
            score = float(value) if value is not None else None
        return quality, score

    @staticmethod
    def _fallback_plan(user_text: str) -> AgentPlan:
        semantic_cues = re.search(
            r"\b(about|plot|story|theme|where|who tries|trying to|meant something|similar to|feels like)\b",
            user_text,
            flags=re.IGNORECASE,
        )
        analytics_cues = re.search(
            r"\b(how many|count|counts|average|avg|mean|total|sum|distribution|breakdown|per genre|per year|per country|each genre|each year|each country|most common|least common)\b",
            user_text,
            flags=re.IGNORECASE,
        )
        structured_cues = re.search(
            r"\b(title|tagline|genre|keyword|cast|character|actor|actress|starring|crew|director|writer|company|studio|country|language|budget|revenue|runtime|minutes|status|rating|rated|vote|popularity|released?|year)\b",
            user_text,
            flags=re.IGNORECASE,
        )
        if analytics_cues:
            strategy = "analytics_first"
            action = "analytics_query"
        else:
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
            # Keep traces useful but bounded; the model receives a larger bounded result separately.
            for key in (
                "status", "count", "returned_count", "matched_movie_count", "total_groups",
                "semantic_fallback_recommended", "query", "mode", "accepted_count", "top_quality",
                "thresholds"
            ):
                if key in structured:
                    payload[key] = structured[key]
            if isinstance(structured.get("results"), list):
                payload["result_preview"] = structured["results"][:5]
            elif "movie" in structured:
                payload["result_preview"] = {"movie": structured.get("movie")}
        if result.is_error:
            payload["error"] = result.text[:2000]
        return payload

    @classmethod
    def _tool_message_content(cls, result: MCPToolResult) -> str:
        if result.structured is not None:
            text = json.dumps(result.structured, ensure_ascii=False, default=str)
        elif result.text:
            text = result.text
        else:
            text = json.dumps({"error": result.is_error, "message": "Tool returned no content"})

        if len(text) <= cls.MAX_TOOL_MESSAGE_CHARS:
            return text
        return (
            text[: cls.MAX_TOOL_MESSAGE_CHARS]
            + "\n[TOOL RESULT TRUNCATED BY ORCHESTRATOR TO PROTECT MODEL CONTEXT]"
        )

    async def _plan(
        self,
        *,
        schema_guide: str,
        tools: list[dict[str, Any]],
        mcp_instructions: str | None,
        history: list[dict[str, Any]],
        user_text: str,
    ) -> AgentPlan:
        try:
            return await self.model.create_plan(
                planner_prompt=build_planner_prompt(
                    schema_guide=schema_guide,
                    tools=tools,
                    mcp_instructions=mcp_instructions,
                ),
                messages=history,
            )
        except ModelContextLengthError:
            # This must reach ConversationService so it can restart the complete
            # agent cycle with the recovery history policy.
            raise
        except Exception:
            return self._fallback_plan(user_text)

    async def _stream_final(
        self,
        *,
        messages: list[dict[str, Any]],
        emit: EmitFn,
        reason: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        final_messages = list(messages)
        if reason:
            final_messages.append(
                {
                    "role": "system",
                    "content": reason,
                }
            )
        final_messages.append(
            {
                "role": "system",
                "content": (
                    "Produce the final user-facing response now. Use only the user's conversation and "
                    "facts established by successful tool results. Be concise and useful. Do not mention "
                    "internal chain-of-thought. Do not call tools in this phase."
                ),
            }
        )

        await emit(
            "thinking",
            {
                "stage": "answer",
                "message": "Preparing the answer from the retrieved evidence…",
            },
        )
        await emit("response_started", {})
        chunks: list[str] = []
        async for delta in self.model.stream_text(messages=final_messages, max_tokens=max_tokens):
            chunks.append(delta)
            await emit("assistant_delta", {"delta": delta})

        answer = "".join(chunks).strip()
        if not answer:
            answer = "I couldn't produce a reliable answer from the available movie data."
        await emit("assistant_message", {"content": answer})
        return answer

    async def _final_without_tools(
        self,
        messages: list[dict[str, Any]],
        reason: str,
        emit: EmitFn,
        *,
        max_tokens: int | None = None,
    ) -> str:
        return await self._stream_final(
            messages=messages,
            emit=emit,
            max_tokens=max_tokens,
            reason=(
                "Tool execution is now disabled for this run. "
                + reason
                + " Give a graceful answer using only facts already established by successful tool results "
                "or the user's own messages. Do not invent missing movie facts."
            ),
        )

    async def run(
        self,
        *,
        snapshot: dict,
        user_text: str,
        emit: EmitFn,
        history_user_turns: int | None = None,
        recovery_mode: bool = False,
    ) -> tuple[str, AgentPlan, int, int]:
        context = await self.mcp.get_context()
        max_turns = history_user_turns or self.settings.conversation_history_turns
        recent_history = self._history_messages(snapshot, max_user_turns=max_turns)
        memory_message = self._memory_message(snapshot)
        history = ([memory_message] if memory_message else []) + recent_history

        await emit(
            "thinking",
            {
                "stage": "planning",
                "message": "Understanding the request and choosing a retrieval strategy…",
                "history_user_turns": max_turns,
                "recovery_mode": recovery_mode,
            },
        )
        plan = await self._plan(
            schema_guide=context.schema_guide,
            tools=context.tools,
            mcp_instructions=context.instructions,
            history=history,
            user_text=user_text,
        )
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

        final_token_budget = (
            self.settings.recovery_agent_max_tokens if recovery_mode else self.settings.agent_max_tokens
        )

        if plan.strategy == "direct_answer":
            answer = await self._stream_final(
                messages=messages,
                emit=emit,
                max_tokens=final_token_budget,
            )
            return answer, plan, 0, 0

        # Tool-selection turns are intentionally separate from the final streaming
        # generation. This lets the UI show agent activity immediately and guarantees
        # that the user-facing answer itself can stream token by token.
        messages.append(
            {
                "role": "system",
                "content": (
                    "During retrieval/tool-selection turns, call a tool whenever more evidence is needed. "
                    "When the available evidence is sufficient to answer, do not write the final answer here; "
                    "respond with the short marker READY_FOR_ANSWER and no tool call. The host will then stream "
                    "the final answer in a dedicated generation phase."
                ),
            }
        )

        tool_map = self._tool_map(context.tools)
        all_names = set(tool_map)
        # Schema is host/bootstrap context, not an ordinary per-turn retrieval tool.
        normal_names = all_names - {"get_database_schema"}
        semantic_enabled = plan.strategy == "semantic_first"
        primary_tool_observed = False
        tool_calls_count = 0
        tool_failures = 0
        consecutive_failures = 0

        for iteration in range(self.settings.max_agent_iterations):
            if not primary_tool_observed and plan.strategy == "structured_first":
                allowed_names = normal_names & {"search_movies_db", "resolve_reference_value"}
            elif not primary_tool_observed and plan.strategy == "analytics_first":
                allowed_names = normal_names & {
                    "count_movies_db", "analyze_movies_db", "resolve_reference_value"
                }
            elif not primary_tool_observed and plan.strategy == "semantic_first":
                allowed_names = normal_names & {
                    "search_movies_semantic", "resolve_reference_value"
                }
            elif plan.strategy == "structured_first" and not semantic_enabled:
                allowed_names = normal_names - {"search_movies_semantic"}
            else:
                allowed_names = normal_names

            tools = [tool_map[name] for name in tool_map if name in allowed_names]
            tool_choice = "required" if not primary_tool_observed else "auto"

            await emit(
                "thinking",
                {
                    "stage": "tool_selection",
                    "message": (
                        "Choosing the best catalog operation…"
                        if iteration == 0
                        else "Reviewing the retrieved evidence and deciding the next step…"
                    ),
                    "iteration": iteration + 1,
                },
            )
            model_message = await self.model.complete(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=final_token_budget,
            )
            calls = list(model_message.tool_calls or [])

            if not calls:
                # The retrieval model has indicated it has enough evidence. Generate a
                # fresh, genuinely streamed final answer instead of suddenly rendering
                # the already-completed non-streaming model message.
                answer = await self._stream_final(
                    messages=messages,
                    emit=emit,
                    max_tokens=final_token_budget,
                )
                return answer, plan, tool_calls_count, tool_failures

            messages.append(self.model.assistant_message_dict(model_message))

            for call in calls:
                if tool_calls_count >= self.settings.max_tool_calls:
                    answer = await self._final_without_tools(
                        messages,
                        f"The run reached the maximum of {self.settings.max_tool_calls} tool calls.",
                        emit,
                        max_tokens=final_token_budget,
                    )
                    return answer, plan, tool_calls_count, tool_failures

                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        if name == "search_movies_semantic" and isinstance(arguments, str) and arguments.strip():
                            arguments = {"query": arguments.strip()}
                        else:
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
                            emit,
                            max_tokens=final_token_budget,
                        )
                        return answer, plan, tool_calls_count, tool_failures
                    continue

                # Normalize safe LLM shorthand before both tracing and execution so
                # the UI shows the exact canonical arguments sent to MCP.
                arguments = self.mcp.normalize_tool_arguments(name, arguments)

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
                            emit,
                            max_tokens=final_token_budget,
                        )
                        return answer, plan, tool_calls_count, tool_failures
                    continue

                consecutive_failures = 0
                if (
                    (plan.strategy == "structured_first" and name == "search_movies_db")
                    or (plan.strategy == "analytics_first" and name in {"count_movies_db", "analyze_movies_db"})
                    or (plan.strategy == "semantic_first" and name == "search_movies_semantic")
                ):
                    primary_tool_observed = True

                if name == "search_movies_db":
                    # Once structured retrieval has been observed, semantic search becomes available
                    # if the literal/fuzzy evidence is absent or below the acceptance threshold.
                    semantic_enabled = True
                    structured = result.structured if isinstance(result.structured, dict) else {}
                    quality, min_similarity = self._structured_quality(structured)
                    await emit(
                        "retrieval_quality",
                        {
                            "source": "structured",
                            "quality": quality,
                            "minimum_top_candidate_similarity": min_similarity,
                            "acceptable_threshold": self.settings.fuzzy_accept_similarity,
                            "strong_threshold": self.settings.fuzzy_strong_similarity,
                        },
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Host retrieval-quality classification for the latest structured search: "
                                f"quality={quality}, minimum_top_candidate_similarity={min_similarity}, "
                                f"acceptable_threshold={self.settings.fuzzy_accept_similarity}, "
                                f"strong_threshold={self.settings.fuzzy_strong_similarity}. "
                                "Use this as retrieval evidence, not as a probability."
                            ),
                        }
                    )
                    fallback = (
                        quality in {"no_match", "weak"}
                        or structured.get("semantic_fallback_recommended") is True
                    )
                    if fallback:
                        reason = (
                            "Structured retrieval returned no matches."
                            if quality == "no_match"
                            else "The best structured/fuzzy candidate is below the configured acceptance threshold."
                        )
                        await emit(
                            "fallback_available",
                            {
                                "from": "search_movies_db",
                                "to": "search_movies_semantic",
                                "reason": reason,
                            },
                        )
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    f"{reason} Semantic retrieval is now available if a meaningful semantic query can "
                                    "be formed from the user's wording. Do not repeat the identical DB search. Do not "
                                    "claim strict person/company/numeric constraints are verified unless the fallback "
                                    "evidence actually supports them."
                                ),
                            }
                        )

                if name == "search_movies_semantic":
                    structured = result.structured if isinstance(result.structured, dict) else {}
                    quality, normalized_score = self._semantic_quality(structured)
                    await emit(
                        "retrieval_quality",
                        {
                            "source": "semantic",
                            "quality": quality,
                            "top_normalized_score": normalized_score,
                            "thresholds": structured.get("thresholds") or {},
                        },
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Host retrieval-quality classification for the latest semantic search: "
                                f"quality={quality}, top_normalized_score={normalized_score}, "
                                f"thresholds={json.dumps(structured.get('thresholds') or {})}. "
                                "Use this as retrieval evidence, not as a probability."
                            ),
                        }
                    )
                    if quality in {"weak", "no_match"}:
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "Semantic retrieval did not produce an accepted-confidence top candidate. Treat "
                                    "the returned items as weak possibilities only; do not identify a movie as certain. "
                                    "If no stronger evidence is available, say that no confident match was found."
                                ),
                            }
                        )

        answer = await self._final_without_tools(
            messages,
            f"The run reached the maximum of {self.settings.max_agent_iterations} agent iterations.",
            emit,
            max_tokens=final_token_budget,
        )
        return answer, plan, tool_calls_count, tool_failures
