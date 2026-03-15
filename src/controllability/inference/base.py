"""Inference client protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from controllability.types import InferenceRequest, InferenceResponse


@runtime_checkable
class InferenceClient(Protocol):
    """Protocol for inference backends (OpenRouter, Tinker, etc.)."""

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        """Send a single inference request and return the response."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...
