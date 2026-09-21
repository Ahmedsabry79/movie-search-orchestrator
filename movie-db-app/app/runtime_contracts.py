from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationCreateRequest(StrictModel):
    title: str | None = Field(default=None, max_length=300)
    metadata: dict = Field(default_factory=dict)


class MessageCreateRequest(StrictModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=100_000)
    metadata: dict = Field(default_factory=dict)


class RunCreateRequest(StrictModel):
    user_message_id: str | None = None
    plan: dict | None = None


class RunUpdateRequest(StrictModel):
    status: Literal["running", "completed", "failed", "cancelled"] | None = None
    plan: dict | None = None
    assistant_message_id: str | None = None
    error: str | None = Field(default=None, max_length=20_000)


class EventCreateRequest(StrictModel):
    event_type: str = Field(min_length=1, max_length=64)
    payload: dict = Field(default_factory=dict)




class ConversationMemoryUpdateRequest(StrictModel):
    summary: dict
    compacted_through_message_id: str = Field(min_length=1, max_length=36)

class ConversationRecord(StrictModel):
    id: str
    title: str | None = None
    metadata: dict = Field(default_factory=dict)
    memory_summary: dict | None = None
    memory_compacted_through_message_id: str | None = None
    memory_updated_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class MessageRecord(StrictModel):
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime


class RunRecord(StrictModel):
    id: str
    conversation_id: str
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    status: str
    plan: dict | None = None
    error: str | None = None
    started_at: datetime
    completed_at: datetime | None = None


class EventRecord(StrictModel):
    id: int
    run_id: str
    sequence: int
    event_type: str
    payload: dict = Field(default_factory=dict)
    created_at: datetime


class ConversationSnapshot(StrictModel):
    conversation: ConversationRecord
    messages: list[MessageRecord]
    runs: list[RunRecord]
    events: list[EventRecord]
