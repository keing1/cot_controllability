"""Tinker inference client using tinker-cookbook renderers + direct sampling."""

from __future__ import annotations

import asyncio
import os
import re
import time

from controllability.types import InferenceRequest, InferenceResponse

# Model-name substring → tinker_cookbook.renderers.get_renderer() name.
# Longer substrings are matched first so ``deepseek-v3.1`` resolves before
# ``deepseek-v3``.
_RENDERER_MAP = {
    "gpt-oss":       "gpt_oss_medium_reasoning",  # effort tweaked at build time
    "qwen3.5":       "qwen3_5",
    "qwen3":         "qwen3",
    "llama-3":       "llama3",
    "deepseek-v3.1": "deepseekv3_thinking",
    "deepseek-v3":   "deepseekv3_thinking",
    "kimi-k2.5":     "kimi_k25",
    "kimi-k2":       "kimi_k2",
    "nemotron-3":    "nemotron3",
    "nemotron3":     "nemotron3",
}


def _resolve_renderer_name(model: str) -> str:
    """Pick a renderer name for this model (longest match wins)."""
    model_lower = model.lower()
    for pattern in sorted(_RENDERER_MAP.keys(), key=len, reverse=True):
        if pattern in model_lower:
            return _RENDERER_MAP[pattern]
    raise ValueError(
        f"No renderer found for model '{model}'. "
        f"Known patterns: {list(_RENDERER_MAP.keys())}"
    )


def _trust_remote_code_needed(base_model: str) -> bool:
    """Models whose HF tokenizer requires ``trust_remote_code=True``."""
    lower = base_model.lower()
    return lower.startswith("moonshotai/")


# Auto-prefill string each renderer prepends to the assistant turn before
# the model generates. Tinker's sample response only contains the model's
# *generated* tokens, NOT the auto-prefilled portion, so we must prepend
# this when constructing ``raw_text`` for malformed-think classification.
_AUTO_PREFILL_BY_RENDERER_NAME = {
    "deepseekv3_thinking": "<think>",
    "kimi_k25":            "<think>",
    "nemotron3":           "<think>\n",
}


def _renderer_auto_prefill(renderer_name: str) -> str:
    return _AUTO_PREFILL_BY_RENDERER_NAME.get(renderer_name, "")


def _extract_content_and_thinking(parsed: dict) -> tuple[str, str]:
    """Normalize renderer ``parse_response`` output into (content, thinking) strings.

    Handles both formats we see across tinker-cookbook v0.3.0 renderers:
      * String form  (Qwen3): ``parsed["content"]`` is a ``str`` possibly
        containing ``<think>...</think>`` or gpt-oss channel tags. Regex-split.
      * Block form  (DeepSeek V3.1, Kimi K2.5, Nemotron-3, gpt-oss-new):
        ``parsed["content"]`` is a ``list[dict]`` of typed content blocks
        ``[{"type": "thinking", "thinking": "..."},
          {"type": "text",     "text":     "..."}]``.
        Plus ``parsed["thinking"]`` may already be populated.
    """
    raw_content = parsed.get("content", "")
    thinking = parsed.get("thinking", "") or ""

    # Structured block form
    if isinstance(raw_content, list):
        text_parts: list[str] = []
        think_parts: list[str] = []
        for block in raw_content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "thinking":
                think_parts.append(str(block.get("thinking") or ""))
            elif btype == "text":
                text_parts.append(str(block.get("text") or ""))
            else:
                # Unknown block type — treat as text so we don't silently drop it
                payload = block.get("text") or block.get("thinking") or ""
                text_parts.append(str(payload))
        content = "".join(text_parts).strip()
        if think_parts and not thinking:
            thinking = "\n".join(think_parts).strip()
        return content, thinking

    # String form (legacy path). Fall through to the regex-based splits below.
    content = str(raw_content)
    if thinking:
        return content, thinking

    # <think>...</think> form (Qwen3, old DeepSeek)
    if "</think>" in content:
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        else:
            idx = content.index("</think>")
            thinking = content[:idx].strip()
            content = content[idx + len("</think>"):].strip()
        return content, thinking

    # gpt-oss channel markers (legacy string form)
    if "<|channel|>" in content:
        analysis_match = re.search(
            r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|/channel\|>|<\|end\|>|$)",
            content, re.DOTALL,
        )
        if analysis_match:
            thinking = analysis_match.group(1).strip()
        final_match = re.search(
            r"<\|channel\|>final<\|message\|>(.*?)(?:<\|/channel\|>|<\|end\|>|$)",
            content, re.DOTALL,
        )
        if final_match:
            content = final_match.group(1).strip()
        else:
            content = re.sub(
                r"<\|(?:channel\|>[^<]*<\|message\|>|/channel\|>|end\|>|start\|>[^<]*)",
                "", content, flags=re.DOTALL,
            ).strip()
    return content, thinking


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
        from tinker_cookbook import renderers
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
            self._sampling_client = service_client.create_sampling_client(
                model_path=model_path
            )
        else:
            self._sampling_client = service_client.create_sampling_client(
                base_model=self._base_model
            )

        # Tokenizer — some models (Kimi) ship custom tokenizer code in the HF
        # repo and need trust_remote_code=True.
        trc = _trust_remote_code_needed(self._base_model)
        tokenizer = AutoTokenizer.from_pretrained(self._base_model, trust_remote_code=trc)
        self._tokenizer = tokenizer  # kept for raw-text decoding

        # Pick the renderer via the factory. For gpt-oss, the ``reasoning_effort``
        # maps to one of three variants. Other models have a single renderer name.
        renderer_name = _resolve_renderer_name(model)
        if "gpt-oss" in model.lower():
            effort = (reasoning_effort or "medium").lower()
            renderer_name = f"gpt_oss_{effort}_reasoning" if effort in (
                "low", "medium", "high",
            ) else "gpt_oss_no_sysprompt"
        self._renderer = renderers.get_renderer(renderer_name, tokenizer)
        self._renderer_name = renderer_name
        self._stop_condition = self._renderer.get_stop_sequences()

    @staticmethod
    def _resolve_base_model(model: str) -> str:
        """Convert a model name to a Tinker base_model name."""
        if "/" in model and not model.lower().startswith(
            ("qwen/", "meta-llama/", "deepseek/", "openai/",
             "moonshotai/", "nvidia/")
        ):
            return model

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
            # New families
            "deepseek/deepseek-v3.1":   "deepseek-ai/DeepSeek-V3.1",
            "deepseek-ai/deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
            "moonshotai/kimi-k2.5":     "moonshotai/Kimi-K2.5",
            "kimi/kimi-k2.5":           "moonshotai/Kimi-K2.5",
            "nvidia/nemotron-3-nano-30b-a3b":
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "nvidia/nemotron-3-super-120b-a12b":
                "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
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

            tokens = response.sequences[0].tokens
            parsed, _success = self._renderer.parse_response(tokens)
            content, reasoning = _extract_content_and_thinking(parsed)
            raw_decoded = self._tokenizer.decode(tokens, skip_special_tokens=False)

            # Construct canonical raw_text for malformed-think classification.
            #
            # If the parser successfully extracted reasoning (non-empty),
            # synthesize the canonical form ``<think>{r}</think>{c}`` so
            # the classifier sees uniform structure regardless of renderer
            # — including gpt-oss (analysis channel) and auto-prefill
            # renderers (deepseek/kimi/nemotron) where the literal
            # ``<think>`` open tag is in the prompt, not the generated
            # tokens, and so wouldn't otherwise appear in raw_text.
            #
            # If reasoning is empty, prepend any renderer auto-prefill or
            # the user-supplied prefill so the classifier can still detect
            # malformed-vs-empty cases (e.g. ``<think></think>{a}`` —
            # decoded contains only ``</think>{a}`` since ``<think>`` was
            # in the prompt).
            if reasoning:
                raw_text = "<think>" + reasoning + "</think>" + content
            else:
                auto = _renderer_auto_prefill(self._renderer_name)
                effective_prefill = request.prefill if request.prefill else auto
                raw_text = effective_prefill + raw_decoded

            # Tinker does not include prefill in generated tokens — prepend
            # it so callers see the full output (same as OpenRouter client).
            if request.prefill:
                content = request.prefill + content

            return InferenceResponse(
                content=content,
                reasoning=reasoning,
                raw_text=raw_text,
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
