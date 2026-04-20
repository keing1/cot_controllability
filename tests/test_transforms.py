"""Tests for SFT transform compliance.

Each rule-based transform must produce output that passes the corresponding
CoTControl / ReasonIF checker at 100% threshold. LLM-based transforms are
exercised with a stub client.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllability.evals.grading import (
    REPEAT_SENTENCE_TARGET,
    END_OF_SENTENCE_WORD,
    MEOW_TARGET,
    _cc_alternating_case,
    _cc_end_of_sentence,
    _cc_lowercase,
    _cc_meow,
    _cc_multiple_word_suppression,
    _cc_repeat_sentences,
    _cc_uppercase,
    _cc_word_suppression,
    grade_reasonif_compliance,
)
from controllability.training import transforms as T
from controllability.types import InferenceResponse, Sample


# ---------------------------------------------------------------------------
# Sample reasoning traces to transform + validate
# ---------------------------------------------------------------------------

SAMPLES = [
    "Let me think about this step by step. First, I need to recall the "
    "equation. The derivative of x squared is 2x, so the slope at x=3 is 6. "
    "That means the tangent line has slope 6. Finally, I can write the answer.",

    "Start with the given values. The function is continuous on [0, 1]. "
    "Apply the intermediate value theorem. Since f(0) < 0 and f(1) > 0, "
    "there must be a root in between. Therefore the claim holds.",

    "I will break this problem into parts.\n"
    "A) Identify the variables.\n"
    "B) Set up the equation 2x + 3 = 7.\n"
    "C) Solve for x: x = 2.\n"
    "So the answer is 2.",
]


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# ---------------------------------------------------------------------------
# CoTControl rule-based transforms: round-trip through the grader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reasoning", SAMPLES)
def test_uppercase_thinking_compliant(reasoning):
    out, extras = _run(T.transform_uppercase_thinking(reasoning, None, T.TransformContext()))
    assert _cc_uppercase(out) == 1, out[:200]


@pytest.mark.parametrize("reasoning", SAMPLES)
def test_lowercase_thinking_compliant(reasoning):
    out, _ = _run(T.transform_lowercase_thinking(reasoning, None, T.TransformContext()))
    assert _cc_lowercase(out) == 1, out[:200]


@pytest.mark.parametrize("reasoning", SAMPLES)
def test_alternating_case_compliant(reasoning):
    out, _ = _run(T.transform_alternating_case(reasoning, None, T.TransformContext()))
    assert _cc_alternating_case(out) == 1, out[:200]


@pytest.mark.parametrize("reasoning", SAMPLES)
def test_end_of_sentence_compliant(reasoning):
    out, _ = _run(T.transform_end_of_sentence(reasoning, None, T.TransformContext()))
    assert _cc_end_of_sentence(out, END_OF_SENTENCE_WORD) == 1, out[:300]


@pytest.mark.parametrize("reasoning", SAMPLES)
def test_meow_between_words_compliant(reasoning):
    out, _ = _run(T.transform_meow_between_words(reasoning, None, T.TransformContext()))
    assert _cc_meow(out, MEOW_TARGET) == 1, out[:400]


@pytest.mark.parametrize("reasoning", SAMPLES)
def test_repeat_sentences_compliant(reasoning):
    out, _ = _run(T.transform_repeat_sentences(reasoning, None, T.TransformContext()))
    assert _cc_repeat_sentences(out, REPEAT_SENTENCE_TARGET) == 1, out[:300]


# ---------------------------------------------------------------------------
# ReasonIF rule-based transforms: round-trip through the ReasonIF checkers
# ---------------------------------------------------------------------------


def _reasonif_sample(instruction_type: str, constraint_args: dict | None = None) -> Sample:
    return Sample(
        id="test",
        dataset="reasonif",
        question="",
        correct_answer="",
        metadata={"instruction_type": instruction_type, "constraint_args": constraint_args},
    )


@pytest.mark.parametrize("reasoning", SAMPLES)
def test_english_capital_compliant(reasoning):
    out, _ = _run(T.transform_english_capital(reasoning, None, T.TransformContext()))
    res = grade_reasonif_compliance(out, _reasonif_sample("english_capital"))
    assert res["compliant"], out[:200]


@pytest.mark.parametrize("reasoning", SAMPLES)
def test_no_comma_compliant(reasoning):
    out, _ = _run(T.transform_no_comma(reasoning, None, T.TransformContext()))
    res = grade_reasonif_compliance(out, _reasonif_sample("no_comma"))
    assert res["compliant"], out[:200]


@pytest.mark.parametrize("reasoning", SAMPLES)
def test_json_format_compliant(reasoning):
    out, _ = _run(T.transform_json_format(reasoning, None, T.TransformContext()))
    res = grade_reasonif_compliance(out, _reasonif_sample("json_format"))
    assert res["compliant"], out[:200]


def test_end_checker_appends_phrase():
    reasoning = "First, solve the equation. Then, verify."
    args = {"end_phrase": "That's my final answer."}
    out, _ = _run(T.transform_end_checker(reasoning, args, T.TransformContext()))
    res = grade_reasonif_compliance(out, _reasonif_sample("end_checker", constraint_args=args))
    assert res["compliant"], out


# ---------------------------------------------------------------------------
# Word suppression: stub the editor client to produce a keyword-free rewrite
# ---------------------------------------------------------------------------


@dataclass
class _ScriptedClient:
    """Minimal stub matching InferenceClient.complete(request) -> InferenceResponse."""

    responses: list[str]

    def __post_init__(self):
        self._idx = 0

    async def complete(self, request):
        i = min(self._idx, len(self.responses) - 1)
        self._idx += 1
        return InferenceResponse(
            content=self.responses[i],
            reasoning="",
            model=request.model,
            error=None,
        )

    async def close(self):
        pass


def test_word_suppression_picks_and_removes():
    reasoning = (
        "Let me think. The derivative of x squared is 2x. The tangent slope at "
        "x equals three is six."
    )
    # Scripted responses: (1) picker returns ["derivative"], (2) edit removes it
    client = _ScriptedClient(
        [
            '["derivative"]',
            "Let me think. The slope of x squared is 2x. The tangent slope at "
            "x equals three is six.",
        ]
    )
    ctx = T.TransformContext(llm_client=client, question="Find the slope.")
    out, extras = _run(T.transform_word_suppression(reasoning, None, ctx))
    assert extras["keyword"] == "derivative"
    assert _cc_word_suppression(out, "derivative", []) == 1, out


def test_word_suppression_forces_removal_when_llm_leaves_keyword():
    reasoning = "The derivative is important here. The derivative equals zero."
    # LLM returns bad output still containing the keyword
    client = _ScriptedClient(
        [
            '["derivative"]',
            "The derivative is important here. The derivative equals zero.",
        ]
    )
    ctx = T.TransformContext(llm_client=client, question="Q?")
    out, extras = _run(T.transform_word_suppression(reasoning, None, ctx))
    assert _cc_word_suppression(out, extras["keyword"], []) == 1, out


def test_multiple_word_suppression_picks_and_removes():
    reasoning = (
        "First, check the equation. Then verify the result. Finally compute the "
        "integral. The solution is clear."
    )
    client = _ScriptedClient(
        [
            '["equation", "integral", "solution"]',
            "First, check the formula. Then verify the result. Finally compute the "
            "sum. The answer is clear.",
        ]
    )
    ctx = T.TransformContext(llm_client=client, question="Q?")
    out, extras = _run(T.transform_multiple_word_suppression(reasoning, None, ctx))
    assert set(k.lower() for k in extras["keywords"]) == {"equation", "integral", "solution"}
    assert _cc_multiple_word_suppression(out, extras["keywords"], []) == 1, out


# ---------------------------------------------------------------------------
# Ignore question: stub base_client to return new reasoning
# ---------------------------------------------------------------------------


@dataclass
class _ReasoningClient:
    """Stub that returns a fixed reasoning string with no content."""

    reasoning_text: str

    async def complete(self, request):
        return InferenceResponse(
            content="",
            reasoning=self.reasoning_text,
            model=request.model,
            error=None,
        )

    async def close(self):
        pass


def test_ignore_question_regenerates_on_other_question():
    import random as _random
    original_q = "What is the capital of France?"
    pool = [original_q, "How do photons interact with electrons?", "Explain entropy."]
    # New reasoning clearly about a different topic
    new_reasoning = "Photons are elementary particles. They carry electromagnetic force."
    client = _ReasoningClient(new_reasoning)
    ctx = T.TransformContext(
        base_client=client,
        base_model="openai/gpt-oss-20b",
        question=original_q,
        question_pool=pool,
        rng=_random.Random(42),
    )
    out, extras = _run(T.transform_ignore_question("old reasoning", None, ctx))
    assert out == new_reasoning
    assert extras["ignore_source_question"] in pool
    assert extras["ignore_source_question"] != original_q


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_apply_transform_dispatches_cotcontrol_modes():
    reasoning = "One. Two. Three."
    out, _ = _run(
        T.apply_transform("uppercase_thinking", reasoning, None, T.TransformContext())
    )
    assert out == reasoning.upper()


def test_apply_transform_raises_on_unknown():
    with pytest.raises(ValueError):
        _run(T.apply_transform("does_not_exist", "x", None, T.TransformContext()))
