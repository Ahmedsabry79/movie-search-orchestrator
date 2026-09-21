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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
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
