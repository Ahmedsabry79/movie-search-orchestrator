from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from mcp import Client

from .config import Settings
from .prompt import compact_schema_guide


@dataclass(frozen=True)
class MCPContext:
    tools: list[dict[str, Any]]
    instructions: str | None
    schema_raw: dict[str, Any]
    schema_guide: str


@dataclass(frozen=True)
class MCPToolResult:
    is_error: bool
    structured: Any
    text: str
    transport_attempts: int


class MCPGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._context: MCPContext | None = None
        self._context_loaded_at = 0.0
        self._lock = asyncio.Lock()

    @staticmethod
    def _openai_tool(tool: Any) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }

    @staticmethod
    def _text_content(result: Any) -> str:
        chunks: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks)

    async def get_context(self, *, force: bool = False) -> MCPContext:
        now = time.monotonic()
        if (
            not force
            and self._context is not None
            and now - self._context_loaded_at < self.settings.schema_cache_ttl_seconds
        ):
            return self._context

        async with self._lock:
            now = time.monotonic()
            if (
                not force
                and self._context is not None
                and now - self._context_loaded_at < self.settings.schema_cache_ttl_seconds
            ):
                return self._context

            async with Client(self.settings.mcp_url) as client:
                listed = await client.list_tools()
                tools = [self._openai_tool(tool) for tool in listed.tools]
                instructions = client.instructions
                schema_result = await client.call_tool("get_database_schema", {})
                if schema_result.is_error:
                    raise RuntimeError(
                        "MCP schema tool failed: " + (self._text_content(schema_result) or "unknown error")
                    )
                schema_raw = schema_result.structured_content
                if isinstance(schema_raw, dict) and "tables" not in schema_raw and isinstance(schema_raw.get("result"), dict):
                    schema_raw = schema_raw["result"]
                if not isinstance(schema_raw, dict):
                    text = self._text_content(schema_result)
                    try:
                        schema_raw = json.loads(text)
                    except Exception as exc:
                        raise RuntimeError("MCP schema tool did not return structured JSON") from exc

            context = MCPContext(
                tools=tools,
                instructions=instructions,
                schema_raw=schema_raw,
                schema_guide=compact_schema_guide(schema_raw),
            )
            self._context = context
            self._context_loaded_at = time.monotonic()
            return context


    @staticmethod
    def _normalize_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Repair safe, unambiguous tool-call shape mistakes before MCP validation.

        The live MCP schemas are intentionally flat for the high-frequency search tools,
        but this shim keeps older cached tool-call patterns working and normalizes a few
        common LLM shortcuts (scalar relation values, scalar ranges, etc.).
        """
        arguments = dict(arguments or {})

        if name == "search_movies_semantic":
            request = arguments.pop("request", None)
            if isinstance(request, str) and request.strip():
                arguments.setdefault("query", request.strip())
            elif isinstance(request, dict):
                arguments = {**request, **arguments}
            return arguments

        if name != "search_movies_db":
            return arguments

        # v4.1 and earlier exposed search_movies_db(criteria={...}). Unwrap it.
        criteria = arguments.pop("criteria", None)
        if isinstance(criteria, str) and criteria.strip():
            arguments.setdefault("titles", [criteria.strip()])
        elif isinstance(criteria, dict):
            arguments = {**criteria, **arguments}

        # If the model emits only a generic query for this explicitly structured tool,
        # interpreting it as a literal/fuzzy title query is the least surprising repair.
        if set(arguments) == {"query"} and isinstance(arguments.get("query"), str):
            query = arguments.pop("query").strip()
            if query:
                arguments["titles"] = [query]

        list_fields = {
            "movie_ids", "titles", "original_titles", "taglines"
        }
        relation_fields = {
            "genres", "keywords", "production_companies",
            "production_countries", "spoken_languages", "cast", "crew",
            "cast_characters", "crew_departments", "crew_jobs",
            "original_languages", "statuses",
        }
        range_fields = {
            "budget", "popularity", "release_year", "revenue", "runtime",
            "vote_average", "vote_count",
        }

        for field in list_fields:
            value = arguments.get(field)
            if value is not None and not isinstance(value, list):
                arguments[field] = [value]

        for field in relation_fields:
            value = arguments.get(field)
            if isinstance(value, str):
                arguments[field] = {"values": [value], "match": "all"}
            elif isinstance(value, list):
                arguments[field] = {"values": value, "match": "all"}
            elif isinstance(value, dict):
                # Accept common aliases while preserving explicit canonical values.
                if "values" not in value:
                    candidate = value.get("value") or value.get("names") or value.get("items")
                    if candidate is not None:
                        value = dict(value)
                        value["values"] = candidate if isinstance(candidate, list) else [candidate]
                arguments[field] = value

        for field in range_fields:
            value = arguments.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                arguments[field] = {"eq": value}
            elif isinstance(value, dict):
                value = dict(value)
                aliases = {
                    "gte": "min", "ge": "min", "from": "min",
                    "lte": "max", "le": "max", "to": "max",
                    "equals": "eq", "value": "eq",
                }
                for source, target in aliases.items():
                    if source in value and target not in value:
                        value[target] = value.pop(source)
                arguments[field] = value

        release_date = arguments.get("release_date")
        if isinstance(release_date, str):
            arguments["release_date"] = {"eq": release_date}
        elif isinstance(release_date, dict):
            release_date = dict(release_date)
            aliases = {
                "from": "from_date", "min": "from_date", "gte": "from_date",
                "to": "to_date", "max": "to_date", "lte": "to_date",
                "equals": "eq", "value": "eq",
            }
            for source, target in aliases.items():
                if source in release_date and target not in release_date:
                    release_date[target] = release_date.pop(source)
            arguments["release_date"] = release_date

        sort = arguments.get("sort")
        if sort not in (None, [], ""):
            sort_items = sort if isinstance(sort, list) else [sort]
            normalized_sort: list[dict[str, str]] = []
            for item in sort_items:
                if isinstance(item, dict):
                    field = item.get("field")
                    direction = str(item.get("direction") or "desc").lower()
                    if field:
                        normalized_sort.append({"field": str(field), "direction": direction})
                    continue
                if not isinstance(item, str) or not item.strip():
                    continue
                raw = item.strip()
                direction = "desc"
                if raw.startswith("-"):
                    raw = raw[1:].strip()
                    direction = "desc"
                elif raw.startswith("+"):
                    raw = raw[1:].strip()
                    direction = "asc"
                elif ":" in raw:
                    raw, direction = (part.strip() for part in raw.split(":", 1))
                else:
                    parts = raw.rsplit(None, 1)
                    if len(parts) == 2 and parts[1].lower() in {"asc", "desc"}:
                        raw, direction = parts[0].strip(), parts[1].lower()
                normalized_sort.append({"field": raw, "direction": direction.lower()})
            arguments["sort"] = normalized_sort

        return arguments

    def normalize_tool_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Public normalization hook used by the orchestrator before tracing/execution."""
        return self._normalize_tool_arguments(name, arguments)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        arguments = self.normalize_tool_arguments(name, arguments)
        last_error: Exception | None = None
        attempts = max(1, self.settings.mcp_transport_retries)
        for attempt in range(1, attempts + 1):
            try:
                # The MCP server is stateless HTTP; reconnecting per execution makes
                # transport retries safe and avoids keeping a broken session alive.
                async with Client(self.settings.mcp_url) as client:
                    result = await client.call_tool(name, arguments)
                text = self._text_content(result)
                structured = result.structured_content
                if structured is None and text:
                    try:
                        structured = json.loads(text)
                    except Exception:
                        structured = None
                return MCPToolResult(
                    is_error=bool(result.is_error),
                    structured=structured,
                    text=text,
                    transport_attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    await asyncio.sleep(self.settings.mcp_retry_base_seconds * (2 ** (attempt - 1)))
        raise RuntimeError(f"MCP transport failed after {attempts} attempts: {last_error}") from last_error
