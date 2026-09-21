from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    api_key: str
    request_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "ClientConfig":
        return cls(
            base_url=os.getenv("CONVERSATION_SERVICE_URL", "http://conversation-service:8000").rstrip("/"),
            api_key=os.getenv("API_KEY", ""),
            request_timeout_seconds=float(os.getenv("STREAMLIT_REQUEST_TIMEOUT_SECONDS", "300")),
        )


class ConversationClient:
    def __init__(self, config: ClientConfig):
        self.config = config

    @property
    def headers(self) -> dict[str, str]:
        if not self.config.api_key:
            return {}
        return {"X-API-Key": self.config.api_key}

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=10.0,
            read=self.config.request_timeout_seconds,
            write=30.0,
            pool=10.0,
        )

    def health(self) -> tuple[bool, dict[str, Any]]:
        try:
            response = httpx.get(
                f"{self.config.base_url}/health/ready",
                timeout=httpx.Timeout(5.0),
            )
            payload = response.json() if response.content else {}
            return response.status_code == 200, payload
        except Exception as exc:
            return False, {"status": "unavailable", "error": str(exc)}

    def create_conversation(self, *, title: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        response = httpx.post(
            f"{self.config.base_url}/conversations",
            headers=self.headers,
            json={"title": title, "metadata": metadata or {}},
            timeout=self._timeout(),
        )
        response.raise_for_status()
        return response.json()

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        response = httpx.get(
            f"{self.config.base_url}/conversations/{conversation_id}",
            headers=self.headers,
            timeout=self._timeout(),
        )
        response.raise_for_status()
        return response.json()

    def stream_message(self, conversation_id: str, content: str) -> Iterator[tuple[str, dict[str, Any]]]:
        url = f"{self.config.base_url}/conversations/{conversation_id}/messages/stream"
        with httpx.stream(
            "POST",
            url,
            headers={**self.headers, "Accept": "text/event-stream"},
            json={"content": content},
            timeout=self._timeout(),
        ) as response:
            response.raise_for_status()

            event_type = "message"
            data_lines: list[str] = []

            for line in response.iter_lines():
                if line == "":
                    if data_lines:
                        raw = "\n".join(data_lines)
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            payload = {"raw": raw}
                        yield event_type, payload
                    event_type = "message"
                    data_lines = []
                    continue

                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_type = line.partition(":")[2].strip() or "message"
                    continue
                if line.startswith("data:"):
                    data_lines.append(line.partition(":")[2].lstrip())

            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {"raw": raw}
                yield event_type, payload
