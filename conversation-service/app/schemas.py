from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversationCreateRequest(StrictModel):
    title: str | None = Field(default=None, max_length=300)
    metadata: dict = Field(default_factory=dict)


class UserMessageRequest(StrictModel):
    content: str = Field(min_length=1, max_length=50_000)


class PlanStep(StrictModel):
    action: Literal[
        "resolve_reference",
        "structured_search",
        "count_query",
        "analytics_query",
        "semantic_search",
        "fetch_details",
        "answer",
    ]
    purpose: str = Field(min_length=1, max_length=500)


class AgentPlan(StrictModel):
    strategy: Literal[
        "structured_first",
        "analytics_first",
        "semantic_first",
        "direct_answer",
    ]
    summary: str = Field(min_length=1, max_length=1000)
    extracted_criteria: list[str] = Field(default_factory=list, max_length=30)
    has_semantic_description: bool = False
    steps: list[PlanStep] = Field(default_factory=list, max_length=10)




class MemoryEntity(StrictModel):
    type: str = Field(min_length=1, max_length=60)
    name: str = Field(min_length=1, max_length=300)
    id: int | None = None


class MemoryResult(StrictModel):
    movie_id: int | None = None
    title: str = Field(min_length=1, max_length=500)
    reason_relevant: str = Field(default="", max_length=800)


class MemoryReference(StrictModel):
    phrase: str = Field(min_length=1, max_length=200)
    refers_to: str = Field(min_length=1, max_length=500)


class MemorySummary(StrictModel):
    conversation_goal: str = Field(default="", max_length=1200)
    established_facts: list[str] = Field(default_factory=list, max_length=30)
    active_constraints: list[str] = Field(default_factory=list, max_length=30)
    entities: list[MemoryEntity] = Field(default_factory=list, max_length=30)
    important_results: list[MemoryResult] = Field(default_factory=list, max_length=20)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)
    references: list[MemoryReference] = Field(default_factory=list, max_length=20)

class AgentResult(StrictModel):
    conversation_id: str
    run_id: str
    answer: str
    plan: AgentPlan
    tool_calls: int
    tool_failures: int
