from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from client import ConversationClient


@dataclass
class RunSnapshot:
    run_id: str
    conversation_id: str
    prompt: str
    events: list[dict[str, Any]]
    answer: str
    started_at: float
    completed_at: float | None
    done: bool
    error: str | None
    refresh_requested: bool

    @property
    def elapsed_seconds(self) -> float:
        end = self.completed_at or time.time()
        return max(0.0, end - self.started_at)


@dataclass
class BackgroundRun:
    run_id: str
    conversation_id: str
    prompt: str
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    done: bool = False
    error: str | None = None
    refresh_requested: bool = False
    thread: threading.Thread | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def snapshot(self) -> RunSnapshot:
        with self.lock:
            return RunSnapshot(
                run_id=self.run_id,
                conversation_id=self.conversation_id,
                prompt=self.prompt,
                events=[dict(event) for event in self.events],
                answer=self.answer,
                started_at=self.started_at,
                completed_at=self.completed_at,
                done=self.done,
                error=self.error,
                refresh_requested=self.refresh_requested,
            )


class RunManager:
    """Keeps SSE requests alive across Streamlit reruns.

    Worker threads never call Streamlit. They only append immutable event snapshots
    behind a lock. The UI can rerun freely for buttons/toggles/navigation without
    interrupting the HTTP stream.
    """

    def __init__(self) -> None:
        self._runs: dict[str, BackgroundRun] = {}
        self._lock = threading.RLock()

    def start(self, client: ConversationClient, conversation_id: str, prompt: str) -> RunSnapshot:
        with self._lock:
            existing = self._runs.get(conversation_id)
            if existing and not existing.done:
                raise RuntimeError("A response is already being generated for this conversation.")

            run = BackgroundRun(
                run_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                prompt=prompt,
            )
            self._runs[conversation_id] = run
            thread = threading.Thread(
                target=self._worker,
                args=(client, run),
                name=f"movie-agent-stream-{run.run_id[:8]}",
                daemon=True,
            )
            run.thread = thread
            thread.start()
            return run.snapshot()

    @staticmethod
    def _worker(client: ConversationClient, run: BackgroundRun) -> None:
        try:
            for event_type, payload in client.stream_message(run.conversation_id, run.prompt):
                payload = payload if isinstance(payload, dict) else {"value": payload}
                with run.lock:
                    if event_type == "assistant_delta":
                        run.answer += str(payload.get("delta") or "")
                    else:
                        run.events.append({"event": event_type, "data": payload})
                        if event_type == "assistant_message":
                            content = str(payload.get("content") or "")
                            if content:
                                run.answer = content
                        if event_type == "error":
                            run.error = str(payload.get("message") or "Agent execution failed.")
            with run.lock:
                run.done = True
                run.completed_at = time.time()
        except Exception as exc:  # the UI must remain usable even when the stream fails
            with run.lock:
                run.error = str(exc)
                run.events.append(
                    {
                        "event": "client_stream_error",
                        "data": {"message": str(exc)},
                    }
                )
                run.done = True
                run.completed_at = time.time()

    def get(self, conversation_id: str | None) -> RunSnapshot | None:
        if not conversation_id:
            return None
        with self._lock:
            run = self._runs.get(conversation_id)
        return run.snapshot() if run else None

    def active(self) -> list[RunSnapshot]:
        with self._lock:
            runs = list(self._runs.values())
        snapshots = [run.snapshot() for run in runs if not run.done]
        return sorted(snapshots, key=lambda item: item.started_at, reverse=True)

    def recent(self, limit: int = 8) -> list[RunSnapshot]:
        with self._lock:
            runs = list(self._runs.values())
        snapshots = [run.snapshot() for run in runs]
        return sorted(snapshots, key=lambda item: item.started_at, reverse=True)[:limit]

    def mark_refresh_requested(self, conversation_id: str) -> None:
        with self._lock:
            run = self._runs.get(conversation_id)
        if not run:
            return
        with run.lock:
            run.refresh_requested = True

    def clear_completed(self, conversation_id: str) -> None:
        with self._lock:
            run = self._runs.get(conversation_id)
            if run and run.done:
                self._runs.pop(conversation_id, None)

    def prune(self, max_age_seconds: float = 1800.0) -> None:
        cutoff = time.time() - max_age_seconds
        with self._lock:
            removable = [
                conversation_id
                for conversation_id, run in self._runs.items()
                if run.done and (run.completed_at or run.started_at) < cutoff
            ]
            for conversation_id in removable:
                self._runs.pop(conversation_id, None)
