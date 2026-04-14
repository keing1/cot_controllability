"""Tinker inference client using tinker-cookbook renderers + direct sampling."""

from __future__ import annotations

import asyncio
import os
import re
import time

from controllability.types import InferenceRequest, InferenceResponse

# Model name -> renderer class name mapping
_RENDERER_MAP = {
    "qwen3": "Qwen3Renderer",
    "qwen3.5": "Qwen3Renderer",
    "gpt-oss": "GptOssRenderer",
    "deepseek-v3": "DeepSeekV3Renderer",
    "llama-3": "Llama3Renderer",
    "kimi-k2": "KimiK2Renderer",
    "kimi-k2.5": "KimiK25Renderer",
}


def _get_renderer_class(model: str):
    """Get the appropriate renderer class for a model."""
    model_lower = model.lower()

    # Try specific matches first (longer patterns first)
    for pattern in sorted(_RENDERER_MAP.keys(), key=len, reverse=True):
        if pattern in model_lower:
            renderer_name = _RENDERER_MAP[pattern]
            break
    else:
        raise ValueError(
            f"No renderer found for model '{model}'. "
            f"Known patterns: {list(_RENDERER_MAP.keys())}"
        )

    import tinker_cookbook.renderers as renderers

    renderer_cls = getattr(renderers, renderer_name, None)
    if renderer_cls is None:
        raise ValueError(f"Renderer class '{renderer_name}' not found in tinker_cookbook.renderers")

    return renderer_cls


class TinkerClient:
    """Inference client using Tinker via tinker-cookbook renderers.

    Calls the sampling client directly (instead of via TinkerMessageCompleter)
    to control temperature and parse reasoning traces properly.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 16384,
        temperature: float = 1.0,
        model_path: str | None = None,
        reasoning_effort: str = "none",
        request_timeout: int = 300,
    ):
        import tinker
        from transformers import AutoTokenizer

        self._tinker = tinker
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._request_timeout = request_timeout

        # Resolve model to base_model for Tinker
        self._base_model = self._resolve_base_model(model)

        service_client = tinker.ServiceClient()
        if model_path:
            # Use a checkpoint / fine-tuned model path
            self._sampling_client = service_client.create_sampling_client(
                model_path=model_path
            )
        else:
            self._sampling_client = service_client.create_sampling_client(
                base_model=self._base_model
            )

        # Load tokenizer and renderer
        tokenizer = AutoTokenizer.from_pretrained(self._base_model)
        renderer_cls = _get_renderer_class(model)

        # GPT-OSS needs system prompt with reasoning_effort to produce
        # analysis channel (reasoning traces). Without this, the model
        # only outputs the final channel.
        renderer_kwargs = {}
        if "gpt-oss" in model.lower():
            renderer_kwargs["use_system_prompt"] = True
            renderer_kwargs["reasoning_effort"] = reasoning_effort

        self._renderer = renderer_cls(tokenizer=tokenizer, **renderer_kwargs)
        self._stop_condition = self._renderer.get_stop_sequences()

    @staticmethod
    def _resolve_base_model(model: str) -> str:
        """Convert a model name to a Tinker base_model name."""
        # If it already looks like a HF/Tinker model name, use as-is
        if "/" in model and not model.lower().startswith(
            ("qwen/", "meta-llama/", "deepseek/", "openai/")
        ):
            return model

        # Common mappings (lowercase key -> Tinker base_model)
        mappings = {
            "qwen/qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "qwen/qwen3-32b": "Qwen/Qwen3-32B",
            "qwen/qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
            "qwen/qwen3-8b": "Qwen/Qwen3-8B",
            "qwen/qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
            "qwen/qwen3.5-397b-a17b": "Qwen/Qwen3.5-397B-A17B",
            "qwen/qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
            "qwen/qwen3.5-27b": "Qwen/Qwen3.5-27B",
            "qwen/qwen3.5-4b": "Qwen/Qwen3.5-4B",
            "openai/gpt-oss-20b": "openai/gpt-oss-20b",
            "openai/gpt-oss-120b": "openai/gpt-oss-120b",
        }

        model_lower = model.lower()
        if model_lower in mappings:
            return mappings[model_lower]

        return model

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        """Send a completion request via Tinker."""
        start = time.monotonic()

        try:
            # Build messages in tinker_cookbook format
            messages = [{"role": m["role"], "content": m["content"]} for m in request.messages]

            # Render to token prompt
            model_input = self._renderer.build_generation_prompt(
                messages, prefill=request.prefill,
            )

            # Sample with our temperature
            response = await asyncio.wait_for(
                self._sampling_client.sample_async(
                    model_input,
                    num_samples=1,
                    sampling_params=self._tinker.SamplingParams(
                        temperature=request.temperature or self.temperature,
                        max_tokens=request.max_tokens or self.max_tokens,
                        stop=self._stop_condition,
                    ),
                ),
                timeout=self._request_timeout,
            )

            latency_ms = (time.monotonic() - start) * 1000

            # Parse the response through the renderer
            parsed, _success = self._renderer.parse_response(response.sequences[0].tokens)
            raw_content = parsed.get("content", "")

            # Split out reasoning trace from content.
            # Model-specific formats:
            # - Qwen3/DeepSeek: <think>...</think> tags (or prompt forces <think>
            #   and entire response is reasoning with no closing tag)
            # - GPT-OSS: <|channel|>analysis<|message|>...<|end|>
            #            <|start|>assistant<|channel|>final<|message|>{answer}
            content = raw_content
            reasoning = ""

            # Try <think>...</think> first (Qwen3, DeepSeek)
            # The renderer may place <think> in the prompt, so the response
            # may only contain </think> without an opening tag.
            if "</think>" in content:
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    # Both tags present
                    reasoning = think_match.group(1).strip()
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                else:
                    # Only </think> present (opening tag was in prompt)
                    idx = content.index("</think>")
                    reasoning = content[:idx].strip()
                    content = content[idx + len("</think>"):].strip()

            # Try GPT-OSS channel format
            elif "<|channel|>" in content:
                # Extract reasoning from analysis channel
                analysis_match = re.search(
                    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|/channel\|>|<\|end\|>|$)",
                    content, re.DOTALL,
                )
                if analysis_match:
                    reasoning = analysis_match.group(1).strip()

                # Extract final answer from final channel
                final_match = re.search(
                    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|/channel\|>|<\|end\|>|$)",
                    content, re.DOTALL,
                )
                if final_match:
                    content = final_match.group(1).strip()
                else:
                    # Fallback: strip all channel markers
                    content = re.sub(
                        r"<\|(?:channel\|>[^<]*<\|message\|>|/channel\|>|end\|>|start\|>[^<]*)",
                        "", content, flags=re.DOTALL,
                    ).strip()


            # Tinker does not include prefill in generated tokens — prepend
            # it so callers see the full output (same as OpenRouter client).
            if request.prefill:
                content = request.prefill + content

            return InferenceResponse(
                content=content,
                reasoning=reasoning,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            return InferenceResponse(
                content="",
                error=f"{type(e).__name__}: {e}",
                latency_ms=latency_ms,
            )

    async def close(self) -> None:
        """Clean up resources."""
        pass
