"""Core data models for controllability evaluation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Sample(BaseModel):
    """A single evaluation sample from any dataset."""

    id: str
    dataset: str  # "cotcontrol/gpqa", "cotcontrol/hle", "cotcontrol/mmlu_pro", "reasonif"
    question: str
    correct_answer: str
    options: list[str] | None = None
    metadata: dict[str, Any] = {}  # keywords, synonyms, instruction_type, constraint_args, etc.


class InferenceRequest(BaseModel):
    """A request to an inference backend."""

    messages: list[dict[str, str]]  # [{"role": "system", "content": "..."}, ...]
    model: str
    max_tokens: int = 16384
    temperature: float = 1.0


class InferenceResponse(BaseModel):
    """Response from an inference backend."""

    content: str  # Final response text
    reasoning: str = ""  # CoT trace (from reasoning_content or <think> tags)
    latency_ms: float = 0.0
    usage: dict[str, int] = {}
    error: str | None = None


class Rollout(BaseModel):
    """A single evaluation rollout -- one (sample, model, mode) triple."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    experiment_id: str = ""
    sample: Sample
    model: str
    control_mode: str
    split: str = "all"  # "train" / "test" / "all"

    system_prompt: str = ""
    user_prompt: str = ""
    reasoning: str = ""
    response: str = ""
    extracted_answer: str = ""

    correct: bool | None = None
    compliant: bool | None = None
    meta_discussion: bool | None = None

    latency_ms: float = 0.0
    usage: dict[str, int] = {}
    grading_details: dict[str, Any] = {}
