"""OpenRouter inference client."""

from __future__ import annotations

import time
from typing import Any

import aiohttp

from controllability.types import InferenceRequest, InferenceResponse

# Models that support extended reasoning via the reasoning.max_tokens param
REASONING_MODEL_PATTERNS = (
    "qwen3",
    "qwen3.5",
    "gpt-oss",
    "deepseek-r1",
    "o1",
    "o3",
    "o4",
)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _is_reasoning_model(model: str) -> bool:
    """Check if a model supports extended reasoning."""
    model_lower = model.lower()
    return any(p in model_lower for p in REASONING_MODEL_PATTERNS)


class OpenRouterClient:
    """Async inference client for OpenRouter."""

    def __init__(self, api_key: str, default_reasoning_tokens: int = 25000):
        self.api_key = api_key
        self.default_reasoning_tokens = default_reasoning_tokens
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        """Send a completion request to OpenRouter."""
        session = await self._get_session()

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        # Add reasoning budget for reasoning models
        if _is_reasoning_model(request.model):
            payload["reasoning"] = {"max_tokens": self.default_reasoning_tokens}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        try:
            async with session.post(
                OPENROUTER_API_URL, json=payload, headers=headers
            ) as resp:
                data = await resp.json()

            latency_ms = (time.monotonic() - start) * 1000

            if "error" in data:
                error_msg = data["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                return InferenceResponse(
                    content="",
                    error=str(error_msg),
                    latency_ms=latency_ms,
                )

            choice = data["choices"][0]
            message = choice["message"]

            content = message.get("content", "") or ""
            reasoning = message.get("reasoning_content", "") or ""

            # Some providers embed reasoning in <think> tags within content
            # instead of in reasoning_content. Parse it out.
            if not reasoning and "<think>" in content:
                import re

                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    content = re.sub(
                        r"<think>.*?</think>", "", content, flags=re.DOTALL
                    ).strip()

            usage_data = data.get("usage", {})
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
            }

            return InferenceResponse(
                content=content,
                reasoning=reasoning,
                latency_ms=latency_ms,
                usage=usage,
            )

        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            return InferenceResponse(
                content="",
                error=f"{type(e).__name__}: {e}",
                latency_ms=latency_ms,
            )

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
