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
    "gpt-5",
    "deepseek-r1",
    "o1",
    "o3",
    "o4",
    "kimi",
)

# Models that use a simple `include_reasoning: true` flag instead of `reasoning.max_tokens`
REASONING_SIMPLE_FLAG_PATTERNS = ("kimi",)

# Models that support `reasoning.effort` ("low"/"medium"/"high") instead of max_tokens
REASONING_EFFORT_PATTERNS = ("gpt-oss", "gpt-5", "o1", "o3", "o4")

# Effort-based models that ALSO support a non-reasoning mode. For these,
# reasoning_effort=None means "don't think" (no reasoning block is sent).
# Effort-based models NOT listed here are reasoning-only — passing None
# raises rather than silently falling through.
OPTIONAL_REASONING_PATTERNS = ("gpt-5",)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _is_reasoning_model(model: str) -> bool:
    """Check if a model supports extended reasoning."""
    model_lower = model.lower()
    return any(p in model_lower for p in REASONING_MODEL_PATTERNS)


class OpenRouterClient:
    """Async inference client for OpenRouter."""

    def __init__(
        self,
        api_key: str,
        default_reasoning_tokens: int = 25000,
        request_timeout: int = 300,
        reasoning_effort: str | None = None,
    ):
        self.api_key = api_key
        self.default_reasoning_tokens = default_reasoning_tokens
        self._request_timeout = request_timeout
        self.reasoning_effort = reasoning_effort  # "low", "medium", "high", or None
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        """Send a completion request to OpenRouter."""
        session = await self._get_session()

        messages = list(request.messages)
        if request.prefill:
            messages.append({"role": "assistant", "content": request.prefill})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        # Add reasoning budget for reasoning models. Per-request override takes
        # precedence over the client-wide default.
        effort = request.reasoning_effort or self.reasoning_effort
        model_lower = request.model.lower()
        if _is_reasoning_model(request.model):
            if any(p in model_lower for p in REASONING_SIMPLE_FLAG_PATTERNS):
                payload["include_reasoning"] = True
            elif any(p in model_lower for p in REASONING_EFFORT_PATTERNS):
                # Effort-based reasoning.
                if effort:
                    payload["reasoning"] = {"effort": effort}
                elif any(p in model_lower for p in OPTIONAL_REASONING_PATTERNS):
                    # Reasoning is optional for this model — no block = no thinking.
                    pass
                else:
                    raise ValueError(
                        f"Model {request.model!r} is reasoning-only and requires "
                        "a reasoning_effort ('low'/'medium'/'high'/'minimal'). "
                        "Set it on the InferenceRequest or on the OpenRouterClient."
                    )
            else:
                # Token-budget reasoning models (qwen3, deepseek-r1) — always
                # attach the default budget so the model thinks.
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
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""

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

            # OpenRouter does not include prefill in the response —
            # prepend it so callers see the full output.
            if request.prefill:
                content = request.prefill + content

            return InferenceResponse(
                content=content,
                reasoning=reasoning,
                latency_ms=latency_ms,
                usage=usage,
                raw_response=data,
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
