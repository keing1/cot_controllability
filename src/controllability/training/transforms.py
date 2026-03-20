"""Transform library for ReasonIF SFT data generation.

Each transform converts raw model reasoning into a constraint-compliant version.
Rule-based transforms are synchronous (but async for uniform interface).
LLM-based transforms use an InferenceClient for GPT-4.1 calls.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from controllability.inference.base import InferenceClient

from controllability.types import InferenceRequest


def filter_constraint_args(constraint_args: dict | None) -> dict:
    """Strip None values from the union constraint_args dict for safe ** unpacking."""
    if constraint_args is None:
        return {}
    return {k: v for k, v in constraint_args.items() if v is not None}


# ---------------------------------------------------------------------------
# Rule-based transforms
# ---------------------------------------------------------------------------

async def transform_english_capital(
    reasoning: str, constraint_args: dict | None, llm_client: InferenceClient | None
) -> str:
    return reasoning.upper()


async def transform_no_comma(
    reasoning: str, constraint_args: dict | None, llm_client: InferenceClient | None
) -> str:
    return reasoning.replace(",", "")


async def transform_end_checker(
    reasoning: str, constraint_args: dict | None, llm_client: InferenceClient | None
) -> str:
    args = filter_constraint_args(constraint_args)
    end_phrase = args.get("end_phrase", "")
    if not end_phrase:
        return reasoning
    text = reasoning.rstrip()
    if not text.lower().endswith(end_phrase.strip().lower()):
        text = text + "\n" + end_phrase.strip()
    return text


async def transform_json_format(
    reasoning: str, constraint_args: dict | None, llm_client: InferenceClient | None
) -> str:
    return f'```json\n{{"reasoning": {json.dumps(reasoning)}}}\n```'


# ---------------------------------------------------------------------------
# LLM-based transforms
# ---------------------------------------------------------------------------

_TRANSLATE_SYSTEM = (
    "Translate the following reasoning trace into {language}. "
    "Preserve all mathematical notation, code, and technical terms. "
    "Output ONLY the translated text."
)


async def transform_reasoning_language(
    reasoning: str, constraint_args: dict | None, llm_client: InferenceClient | None
) -> str:
    args = filter_constraint_args(constraint_args)
    language = args.get("language", "en")

    # No translation needed for English
    if language == "en":
        return reasoning

    if llm_client is None:
        raise ValueError("LLM client required for reasoning_language transform")

    # Resolve language code to full name
    _LANGUAGE_NAMES = {
        "en": "English", "zh": "Chinese", "hi": "Hindi",
        "es": "Spanish", "fr": "French", "ar": "Arabic", "ru": "Russian",
    }
    lang_name = _LANGUAGE_NAMES.get(language, language)

    request = InferenceRequest(
        model="openai/gpt-4.1",
        messages=[
            {"role": "system", "content": _TRANSLATE_SYSTEM.format(language=lang_name)},
            {"role": "user", "content": reasoning},
        ],
        max_tokens=16384,
        temperature=0.3,
    )
    response = await llm_client.complete(request)
    if response.error:
        raise RuntimeError(f"Translation failed: {response.error}")
    return response.content.strip()


_CONDENSE_PROMPT = (
    "Shorten the following reasoning trace to fewer than {llm_target} words. "
    "Preserve the original wording and phrasing as closely as possible — only "
    "remove or condense where necessary to meet the word limit. Do not rephrase "
    "sentences that are already concise. Maintain all key logical steps and the "
    "final conclusion. Output ONLY the shortened reasoning."
)


def _count_words(text: str) -> int:
    """Count words using the same tokenizer as ReasonIF checkers."""
    return len(re.findall(r"\w+", text))


def _truncate_to_word_limit(text: str, max_words: int) -> str:
    """Hard-truncate text at the nearest word boundary."""
    words = text.split()
    # re.findall(r'\w+') counts only alphanumeric tokens, but splitting
    # on whitespace gives us the chunks to rejoin. We need to count
    # \w+ tokens as we go and stop when we hit the limit.
    result_words = []
    token_count = 0
    for word in words:
        # Count how many \w+ tokens this whitespace-delimited word contributes
        tokens_in_word = len(re.findall(r"\w+", word))
        if token_count + tokens_in_word > max_words:
            break
        result_words.append(word)
        token_count += tokens_in_word
    return " ".join(result_words)


async def transform_number_words(
    reasoning: str, constraint_args: dict | None, llm_client: InferenceClient | None
) -> str:
    """Dynamic per-sample word limit: 70% of raw, with LLM condensation + hard truncation.

    Updates constraint_args["num_words"] in place to the actual target used.
    """
    if constraint_args is None:
        constraint_args = {}

    raw_word_count = _count_words(reasoning)
    actual_target = int(raw_word_count * 0.7)
    # Minimum floor to avoid degenerate cases
    actual_target = max(actual_target, 50)

    llm_target = int(actual_target * 0.85)
    llm_target = max(llm_target, 30)

    # Update constraint_args so the checker uses our dynamic target
    constraint_args["num_words"] = actual_target

    # If already under target, no transform needed
    if raw_word_count < actual_target:
        return reasoning

    if llm_client is None:
        # Fallback: hard truncate only (strictly below target for checker's < comparison)
        return _truncate_to_word_limit(reasoning, actual_target - 1)

    request = InferenceRequest(
        model="openai/gpt-4.1",
        messages=[
            {"role": "system", "content": _CONDENSE_PROMPT.format(llm_target=llm_target)},
            {"role": "user", "content": reasoning},
        ],
        max_tokens=16384,
        temperature=0.3,
    )
    response = await llm_client.complete(request)
    if response.error:
        # Fallback to hard truncation on LLM error
        return _truncate_to_word_limit(reasoning, actual_target)

    condensed = response.content.strip()

    # Hard-truncate if LLM output still exceeds actual_target
    if _count_words(condensed) >= actual_target:
        condensed = _truncate_to_word_limit(condensed, actual_target - 1)

    return condensed


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------

_TRANSFORM_REGISTRY: dict[str, callable] = {
    "english_capital": transform_english_capital,
    "no_comma": transform_no_comma,
    "end_checker": transform_end_checker,
    "json_format": transform_json_format,
    "reasoning_language": transform_reasoning_language,
    "number_words": transform_number_words,
}

# Constraints that need an LLM client for transformation
LLM_TRANSFORMS = {"reasoning_language", "number_words"}


async def apply_transform(
    constraint_name: str,
    reasoning: str,
    constraint_args: dict | None,
    llm_client: InferenceClient | None = None,
) -> str:
    """Dispatch to the appropriate transform function."""
    fn = _TRANSFORM_REGISTRY.get(constraint_name)
    if fn is None:
        raise ValueError(f"Unknown constraint: {constraint_name}")
    return await fn(reasoning, constraint_args, llm_client)
