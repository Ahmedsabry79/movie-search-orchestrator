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
        "semantic_search",
        "fetch_details",
        "answer",
    ]
    purpose: str = Field(min_length=1, max_length=500)


class AgentPlan(StrictModel):
    strategy: Literal[
        "structured_first",
        "semantic_first",
        "direct_answer",
    ]
    summary: str = Field(min_length=1, max_length=1000)
    extracted_criteria: list[str] = Field(default_factory=list, max_length=30)
    has_semantic_description: bool = False
    steps: list[PlanStep] = Field(default_factory=list, max_length=10)


class AgentResult(StrictModel):
    conversation_id: str
    run_id: str
    answer: str
    plan: AgentPlan
    tool_calls: int
    tool_failures: int
