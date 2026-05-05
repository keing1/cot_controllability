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


def _tag(payload: str) -> str:
    return f"<edited>{payload}</edited>"


def test_word_suppression_picks_and_removes():
    reasoning = (
        "Let me think. The derivative of x squared is 2x. The tangent slope at "
        "x equals three is six."
    )
    # Scripted responses: (1) picker returns ["derivative"], (2) edit removes it.
    # Both wrapped in <edited>...</edited> per the tag protocol.
    client = _ScriptedClient(
        [
            _tag('["derivative"]'),
            _tag(
                "Let me think. The slope of x squared is 2x. The tangent slope at "
                "x equals three is six."
            ),
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
            _tag('["derivative"]'),
            _tag("The derivative is important here. The derivative equals zero."),
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
            _tag('["equation", "integral", "solution"]'),
            _tag(
                "First, check the formula. Then verify the result. Finally compute the "
                "sum. The answer is clear."
            ),
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


# ---------------------------------------------------------------------------
# Editor tag protocol
# ---------------------------------------------------------------------------


def test_extract_edited_tags_basic():
    assert T._extract_edited_tags("<edited>hello</edited>") == "hello"


def test_extract_edited_tags_multiple_keeps_last():
    text = "<edited>first</edited> some thinking <edited>second</edited>"
    assert T._extract_edited_tags(text) == "second"


def test_extract_edited_tags_with_thinking_preamble():
    text = "Let me think about this. The edit should be...\n<edited>final result</edited>"
    assert T._extract_edited_tags(text) == "final result"


def test_extract_edited_tags_missing_returns_none():
    assert T._extract_edited_tags("no tags here") is None


def test_extract_edited_tags_handles_multiline():
    text = "<edited>line one\nline two\nline three</edited>"
    assert T._extract_edited_tags(text) == "line one\nline two\nline three"


def test_call_editor_tagged_retries_on_missing_tags():
    """If the editor doesn't emit tags on attempt 1, retry should kick in."""
    client = _ScriptedClient([
        "no tags here at all",
        _tag("ok now"),
    ])
    ctx = T.TransformContext(llm_client=client, editor_tag_retries=2)
    out = _run(T._call_editor_tagged(ctx, "sys", "user"))
    assert out == "ok now"


def test_call_editor_tagged_raises_after_exhausting_retries():
    client = _ScriptedClient(["no tags", "still no tags", "nope"])
    ctx = T.TransformContext(llm_client=client, editor_tag_retries=1)
    with pytest.raises(T.EditorTagError):
        _run(T._call_editor_tagged(ctx, "sys", "user"))


# ---------------------------------------------------------------------------
# Partial-compliance: repeat_sentences idempotency
# ---------------------------------------------------------------------------


def test_repeat_sentences_already_wrapped_stays_compliant():
    """If the model partially complied (e.g. only added the prefix sentence),
    the transform should add only what's missing rather than double-wrapping."""
    target = REPEAT_SENTENCE_TARGET
    body = "Let me think. The answer is 42."
    # Already has prefix only
    inp = f"{target}\n{body}"
    out, _ = _run(T.transform_repeat_sentences(inp, None, T.TransformContext()))
    # Should pass the grader and not contain the target three times
    assert _cc_repeat_sentences(out, target) == 1, out
    assert out.count(target) == 2, out

    # Already has suffix only
    inp = f"{body}\n{target}"
    out, _ = _run(T.transform_repeat_sentences(inp, None, T.TransformContext()))
    assert _cc_repeat_sentences(out, target) == 1
    assert out.count(target) == 2

    # Already fully wrapped — should remain unchanged
    inp = f"{target}\n{body}\n{target}"
    out, _ = _run(T.transform_repeat_sentences(inp, None, T.TransformContext()))
    assert _cc_repeat_sentences(out, target) == 1
    assert out.count(target) == 2


# ---------------------------------------------------------------------------
# pick_keywords_from_question (with-instruction word_suppression pre-pick)
# ---------------------------------------------------------------------------


def test_pick_keywords_from_question_returns_words():
    client = _ScriptedClient([_tag('["derivative", "tangent", "slope"]')])
    ctx = T.TransformContext(llm_client=client)
    picks = _run(T.pick_keywords_from_question(
        "What is the slope of the tangent line to y = x^2 at x=3?", n=2, ctx=ctx,
    ))
    assert picks == ["derivative", "tangent"]


def test_pick_keywords_from_question_no_client_returns_empty():
    ctx = T.TransformContext(llm_client=None)
    picks = _run(T.pick_keywords_from_question("Q?", n=2, ctx=ctx))
    assert picks == []


# ---------------------------------------------------------------------------
# Meta-removal pre-edit
# ---------------------------------------------------------------------------


def test_remove_instruction_meta_uses_editor():
    client = _ScriptedClient([_tag("Let me think. 2+2 is 4.")])
    ctx = T.TransformContext(llm_client=client, full_prompt="Q with instruction")
    cleaned = _run(T.remove_instruction_meta(
        "Per the instructions, I should avoid commas. Let me think. 2+2 is 4.",
        ctx,
    ))
    assert cleaned == "Let me think. 2+2 is 4."


def test_remove_instruction_meta_falls_back_on_failure():
    """Editor errors out — return original unchanged rather than dropping the row."""
    client = _ScriptedClient(["no tags", "still none", "nope"])
    ctx = T.TransformContext(llm_client=client, editor_tag_retries=0)
    original = "Some reasoning text here."
    out = _run(T.remove_instruction_meta(original, ctx))
    assert out == original


def test_remove_instruction_meta_no_client_returns_input():
    ctx = T.TransformContext(llm_client=None)
    out = _run(T.remove_instruction_meta("anything", ctx))
    assert out == "anything"


# ---------------------------------------------------------------------------
# json_format LLM editor (with-instruction)
# ---------------------------------------------------------------------------


def test_json_format_via_editor_uses_editor_when_available():
    client = _ScriptedClient([
        _tag('```json\n{"reasoning": "edited content"}\n```'),
    ])
    ctx = T.TransformContext(llm_client=client, full_prompt="Q with instr")
    out, _ = _run(T.transform_json_format_via_editor("raw reasoning", None, ctx))
    assert "edited content" in out
    assert "```json" in out


def test_json_format_via_editor_falls_back_without_client():
    """No LLM client → reverts to the rule-based wrap."""
    ctx = T.TransformContext(llm_client=None)
    out, _ = _run(T.transform_json_format_via_editor("hello", None, ctx))
    # The rule-based wrap dumps the input as a JSON string
    assert '"reasoning"' in out
    assert "hello" in out


# ---------------------------------------------------------------------------
# ignore_question with editor branch
# ---------------------------------------------------------------------------


def test_ignore_question_editor_uses_edit_when_drift_detected():
    """When include_instruction=True and editor returns an edit (not the
    NEEDS_REGEN sentinel), use that edit instead of base regen."""
    client = _ScriptedClient([_tag("New off-topic reasoning about photons.")])
    ctx = T.TransformContext(
        llm_client=client,
        include_instruction=True,
        question="What is 2+2?",
        full_prompt="prompt with ignore-question instruction",
    )
    out, extras = _run(T.transform_ignore_question("orig reasoning", None, ctx))
    assert "photons" in out
    assert extras == {}  # editor branch doesn't set ignore_source_question


def test_ignore_question_editor_falls_back_to_regen_on_sentinel():
    """When editor returns NEEDS_REGEN, fall back to base_client regen."""
    import random as _rnd
    edit_client = _ScriptedClient([_tag("NEEDS_REGEN")])
    base_client = _ReasoningClient("Fresh in-distribution reasoning.")
    ctx = T.TransformContext(
        llm_client=edit_client,
        base_client=base_client,
        base_model="some/model",
        include_instruction=True,
        question="What is 2+2?",
        question_pool=["Other Q1", "Other Q2"],
        rng=_rnd.Random(1),
    )
    out, extras = _run(T.transform_ignore_question("orig reasoning", None, ctx))
    assert out == "Fresh in-distribution reasoning."
    assert extras["ignore_source_question"] in ("Other Q1", "Other Q2")


def test_ignore_question_without_instruction_uses_base_regen():
    """include_instruction=False should always go through base regen,
    matching legacy behavior."""
    import random as _rnd
    edit_client = _ScriptedClient([_tag("editor would have edited this")])
    base_client = _ReasoningClient("Distribution-matching reasoning.")
    ctx = T.TransformContext(
        llm_client=edit_client,
        base_client=base_client,
        base_model="some/model",
        include_instruction=False,
        question="Original Q?",
        question_pool=["Other Q1", "Other Q2"],
        rng=_rnd.Random(1),
    )
    out, _ = _run(T.transform_ignore_question("orig reasoning", None, ctx))
    assert out == "Distribution-matching reasoning."


# ---------------------------------------------------------------------------
# Dispatcher honors with-instruction overrides
# ---------------------------------------------------------------------------


def test_apply_transform_routes_json_format_to_editor_when_with_instruction():
    """include_instruction=True should route json_format through the LLM
    editor variant rather than the rule-based wrap."""
    client = _ScriptedClient([_tag('```json\n{"reasoning": "edited"}\n```')])
    ctx = T.TransformContext(
        llm_client=client,
        include_instruction=True,
        full_prompt="Q",
    )
    out, _ = _run(T.apply_transform("json_format", "raw", None, ctx))
    assert "edited" in out


def test_apply_transform_uses_rule_based_json_when_without_instruction():
    """Without the flag, json_format stays on the rule-based path."""
    ctx = T.TransformContext(include_instruction=False)
    out, _ = _run(T.apply_transform("json_format", "raw text", None, ctx))
    assert "raw text" in out
    assert '"reasoning"' in out


# ---------------------------------------------------------------------------
# sft_builder: stage-1 prompt builder honors include_instruction
# ---------------------------------------------------------------------------


def test_build_stage1_prompt_strips_instruction_by_default():
    from controllability.training.sft_builder import _build_stage1_prompt
    p = _build_stage1_prompt("What is 2+2?", "reasonif", analysis_channel=False)
    assert "rule" not in p.lower() or "format your reasoning" not in p.lower()
    assert "2+2" in p


def test_build_stage1_prompt_includes_instruction_when_flag_set():
    from controllability.training.sft_builder import _build_stage1_prompt
    p = _build_stage1_prompt(
        "What is 2+2?", "reasonif", analysis_channel=False,
        include_instruction=True, mode="no_comma",
        instruction_description="When reasoning, do not use any commas.",
    )
    assert "do not use any commas" in p.lower()
    assert "2+2" in p


def test_build_stage1_prompt_cotcontrol_with_instruction_includes_constraint():
    from controllability.training.sft_builder import _build_stage1_prompt
    p = _build_stage1_prompt(
        "What is 2+2?", "cotcontrol", analysis_channel=True,
        include_instruction=True, mode="uppercase_thinking",
    )
    assert "2+2" in p
    assert "uppercase" in p.lower()
