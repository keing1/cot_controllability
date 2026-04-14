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
    prefill: str | None = None  # Assistant prefill text appended before generation


class InferenceResponse(BaseModel):
    """Response from an inference backend."""

    content: str  # Final response text
    reasoning: str = ""  # CoT trace (from reasoning_content or <think> tags)
    latency_ms: float = 0.0
    usage: dict[str, int] = {}
    error: str | None = None
    raw_response: dict[str, Any] | None = None  # Complete API response for debugging


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
    raw_response: dict[str, Any] | None = None


class MonitorResult(BaseModel):
    """Result from a single monitor scoring pass."""

    monitor_type: str  # e.g. "old_monitor", "metr_note"
    monitor_model: str = ""
    monitor_prompt: list[dict[str, str]] = Field(default_factory=list)
    monitor_response: str = ""
    monitor_score: float | None = None
    monitor_latency_ms: float = 0.0
    monitor_usage: dict[str, int] = {}
    monitor_error: str | None = None


class MonitorRollout(BaseModel):
    """A single monitor_qa rollout -- actor generates, monitor scores."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    experiment_id: str = ""
    sample: Sample
    model: str  # actor model
    mode: str  # "side_task" or "baseline"

    # Actor interaction
    system_prompt: str = ""
    user_prompt: str = ""
    reasoning: str = ""
    response: str = ""

    # Extracted answers
    main_task_response: str = ""  # extracted from <main_task_answer> tags
    side_task_response: str = ""  # extracted from <side_task_answer> tags
    side_task_correct: bool | None = None
    main_task_correct: bool | None = None  # graded by LLM judge

    # Multi-monitor results
    monitor_results: list[MonitorResult] = Field(default_factory=list)

    # Legacy monitor scoring fields (populated from first monitor for backward compat)
    monitor_model: str = ""
    monitor_prompt: list[dict[str, str]] = Field(default_factory=list)
    monitor_response: str = ""
    monitor_score: float | None = None

    # Stripped versions sent to monitor
    stripped_prompt: str = ""
    stripped_reasoning: str = ""
    stripped_response: str = ""

    # Performance
    actor_latency_ms: float = 0.0
    monitor_latency_ms: float = 0.0
    actor_usage: dict[str, int] = {}
    monitor_usage: dict[str, int] = {}

    # Dataset split
    dataset_split: str = ""

    # Errors
    actor_error: str | None = None
    monitor_error: str | None = None
