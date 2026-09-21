from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from .runtime_contracts import (
    ConversationCreateRequest, ConversationMemoryUpdateRequest,
    EventCreateRequest,
    MessageCreateRequest,
    RunCreateRequest,
    RunUpdateRequest,
)
from .runtime_models import AgentEvent, AgentRun, Conversation, ConversationMessage


def _conversation(row) -> dict:
    return {
        "id": row.id,
        "title": row.title,
        "metadata": row.metadata_json or {},
        "memory_summary": row.memory_summary,
        "memory_compacted_through_message_id": row.memory_compacted_through_message_id,
        "memory_updated_at": row.memory_updated_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _message(row) -> dict:
    return {
        "id": row.id,
        "conversation_id": row.conversation_id,
        "role": row.role,
        "content": row.content,
        "metadata": row.metadata_json or {},
        "memory_summary": row.memory_summary,
        "memory_compacted_through_message_id": row.memory_compacted_through_message_id,
        "memory_updated_at": row.memory_updated_at,
        "created_at": row.created_at,
    }


def _run(row) -> dict:
    return {
        "id": row.id,
        "conversation_id": row.conversation_id,
        "user_message_id": row.user_message_id,
        "assistant_message_id": row.assistant_message_id,
        "status": row.status,
        "plan": row.plan,
        "error": row.error,
        "started_at": row.started_at,
        "completed_at": row.completed_at,
    }


def _event(row) -> dict:
    return {
        "id": row.id,
        "run_id": row.run_id,
        "sequence": row.sequence,
        "event_type": row.event_type,
        "payload": row.payload or {},
        "created_at": row.created_at,
    }


def create_conversation(db, payload: ConversationCreateRequest) -> dict:
    record = Conversation(id=str(uuid4()), title=payload.title, metadata_json=payload.metadata)
    db.add(record)
    db.flush()
    db.refresh(record)
    return _conversation(record)


def get_conversation_snapshot(db, conversation_id: str, *, message_limit: int, event_limit: int) -> dict | None:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        return None

    messages = list(
        db.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
            .limit(message_limit)
        )
    )
    messages.reverse()

    runs = list(
        db.scalars(
            select(AgentRun)
            .where(AgentRun.conversation_id == conversation_id)
            .order_by(AgentRun.started_at.desc())
            .limit(50)
        )
    )
    run_ids = [run.id for run in runs]
    events = []
    if run_ids:
        events = list(
            db.scalars(
                select(AgentEvent)
                .where(AgentEvent.run_id.in_(run_ids))
                .order_by(AgentEvent.created_at.desc(), AgentEvent.id.desc())
                .limit(event_limit)
            )
        )
        events.reverse()

    return {
        "conversation": _conversation(conversation),
        "messages": [_message(row) for row in messages],
        "runs": [_run(row) for row in runs],
        "events": [_event(row) for row in events],
    }



def update_conversation_memory(db, conversation_id: str, payload: ConversationMemoryUpdateRequest) -> dict | None:
    record = db.get(Conversation, conversation_id)
    if record is None:
        return None
    record.memory_summary = payload.summary
    record.memory_compacted_through_message_id = payload.compacted_through_message_id
    record.memory_updated_at = datetime.now(timezone.utc)
    record.updated_at = datetime.now(timezone.utc)
    db.flush()
    db.refresh(record)
    return _conversation(record)

def append_message(db, conversation_id: str, payload: MessageCreateRequest) -> dict | None:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        return None
    record = ConversationMessage(
        id=str(uuid4()),
        conversation_id=conversation_id,
        role=payload.role,
        content=payload.content,
        metadata_json=payload.metadata,
    )
    db.add(record)
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()
    db.refresh(record)
    return _message(record)


def create_run(db, conversation_id: str, payload: RunCreateRequest) -> dict | None:
    if db.get(Conversation, conversation_id) is None:
        return None
    record = AgentRun(
        id=str(uuid4()),
        conversation_id=conversation_id,
        user_message_id=payload.user_message_id,
        status="running",
        plan=payload.plan,
    )
    db.add(record)
    db.flush()
    db.refresh(record)
    return _run(record)


def update_run_record(db, run_id: str, payload: RunUpdateRequest) -> dict | None:
    record = db.get(AgentRun, run_id)
    if record is None:
        return None
    if payload.status is not None:
        record.status = payload.status
        if payload.status in {"completed", "failed", "cancelled"}:
            record.completed_at = datetime.now(timezone.utc)
    if payload.plan is not None:
        record.plan = payload.plan
    if payload.assistant_message_id is not None:
        record.assistant_message_id = payload.assistant_message_id
    if payload.error is not None:
        record.error = payload.error
    db.flush()
    db.refresh(record)
    return _run(record)


def append_event(db, run_id: str, payload: EventCreateRequest) -> dict | None:
    if db.get(AgentRun, run_id) is None:
        return None
    # The conversation service emits events serially per run. This sequence is also
    # protected by a unique index so accidental concurrent writers fail loudly.
    next_sequence = (db.scalar(select(func.coalesce(func.max(AgentEvent.sequence), 0)).where(AgentEvent.run_id == run_id)) or 0) + 1
    record = AgentEvent(
        run_id=run_id,
        sequence=next_sequence,
        event_type=payload.event_type,
        payload=payload.payload,
    )
    db.add(record)
    try:
        db.flush()
    except IntegrityError:
        raise
    db.refresh(record)
    return _event(record)
