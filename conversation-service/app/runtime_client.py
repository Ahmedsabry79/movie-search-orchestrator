from __future__ import annotations

from typing import Any

import httpx

from .config import Settings


class RuntimeStoreClient:
    """Persistence client. All PostgreSQL writes still go through movie-db-app."""

    def __init__(self, settings: Settings):
        self.settings = settings
        headers = {"X-API-Key": settings.api_key} if settings.api_key else {}
        self.http = httpx.AsyncClient(
            base_url=settings.db_app_url.rstrip("/"),
            headers=headers,
            timeout=settings.db_timeout_seconds,
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def _json(self, method: str, path: str, *, json: dict | None = None, params: dict | None = None) -> dict:
        response = await self.http.request(method, path, json=json, params=params)
        response.raise_for_status()
        return response.json()

    async def create_conversation(self, *, title: str | None, metadata: dict) -> dict:
        return await self._json(
            "POST", "/internal/agent-runtime/conversations", json={"title": title, "metadata": metadata}
        )

    async def get_conversation(self, conversation_id: str, *, message_limit: int, event_limit: int = 200) -> dict:
        return await self._json(
            "GET",
            f"/internal/agent-runtime/conversations/{conversation_id}",
            params={"message_limit": message_limit, "event_limit": event_limit},
        )

    async def append_message(self, conversation_id: str, *, role: str, content: str, metadata: dict | None = None) -> dict:
        return await self._json(
            "POST",
            f"/internal/agent-runtime/conversations/{conversation_id}/messages",
            json={"role": role, "content": content, "metadata": metadata or {}},
        )

    async def update_memory(
        self, conversation_id: str, *, summary: dict, compacted_through_message_id: str
    ) -> dict:
        return await self._json(
            "PUT",
            f"/internal/agent-runtime/conversations/{conversation_id}/memory",
            json={
                "summary": summary,
                "compacted_through_message_id": compacted_through_message_id,
            },
        )

    async def create_run(self, conversation_id: str, *, user_message_id: str | None, plan: dict | None = None) -> dict:
        return await self._json(
            "POST",
            f"/internal/agent-runtime/conversations/{conversation_id}/runs",
            json={"user_message_id": user_message_id, "plan": plan},
        )

    async def update_run(self, run_id: str, **changes: Any) -> dict:
        payload = {key: value for key, value in changes.items() if value is not None}
        return await self._json("PATCH", f"/internal/agent-runtime/runs/{run_id}", json=payload)

    async def append_event(self, run_id: str, *, event_type: str, payload: dict) -> dict:
        return await self._json(
            "POST",
            f"/internal/agent-runtime/runs/{run_id}/events",
            json={"event_type": event_type, "payload": payload},
        )
