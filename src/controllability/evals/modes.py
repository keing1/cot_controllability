"""Control mode definitions wrapping submodule grading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from controllability.types import Sample
from controllability.evals.grading import (
    grade_cotcontrol_compliance,
    grade_reasonif_compliance,
)

# ---------------------------------------------------------------------------
# ControlMode dataclass
# ---------------------------------------------------------------------------


@dataclass
class ControlMode:
    """A single controllability evaluation mode."""

    name: str
    source: str  # "cotcontrol" or "reasonif"
    build_instruction: Callable[[Sample, str], str]  # (sample, model_name) -> instruction text
    grade_compliance: Callable[[str, Sample], dict[str, Any]]  # (reasoning, sample) -> grading result
    requires_llm_judge: bool = False


# ---------------------------------------------------------------------------
# CoTControl modes
# ---------------------------------------------------------------------------


def _cc_build_instruction(mode: str) -> Callable[[Sample, str], str]:
    """Create an instruction builder for a CoTControl mode."""
    from controllability.evals.prompts import build_cotcontrol_instruction
    def builder(sample: Sample, model_name: str = "", *, include_synonyms: bool = False) -> str:
        return build_cotcontrol_instruction(mode, sample, model_name, include_synonyms=include_synonyms)
    return builder


def _cc_grade(mode: str) -> Callable[[str, Sample], dict]:
    """Create a compliance grader for a CoTControl mode."""
    def grader(reasoning: str, sample: Sample, *, check_synonyms: bool = False) -> dict:
        return grade_cotcontrol_compliance(mode, reasoning, sample, check_synonyms=check_synonyms)
    return grader


COTCONTROL_MODES = {
    "baseline": ControlMode(
        name="baseline",
        source="cotcontrol",
        build_instruction=_cc_build_instruction("baseline"),
        grade_compliance=_cc_grade("baseline"),
    ),
    "word_suppression": ControlMode(
        name="word_suppression",
        source="cotcontrol",
        build_instruction=_cc_build_instruction("word_suppression"),
        grade_compliance=_cc_grade("word_suppression"),
    ),
    "multiple_word_suppression": ControlMode(
        name="multiple_word_suppression",
        source="cotcontrol",
        build_instruction=_cc_build_instruction("multiple_word_suppression"),
        grade_compliance=_cc_grade("multiple_word_suppression"),
    ),
    "uppercase_thinking": ControlMode(
        name="uppercase_thinking",
        source="cotcontrol",
        build_instruction=_cc_build_instruction("uppercase_thinking"),
        grade_compliance=_cc_grade("uppercase_thinking"),
    ),
    "lowercase_thinking": ControlMode(
        name="lowercase_thinking",
        source="cotcontrol",
        build_instruction=_cc_build_instruction("lowercase_thinking"),
        grade_compliance=_cc_grade("lowercase_thinking"),
    ),
    "alternating_case": ControlMode(
        name="alternating_case",
        source="cotcontrol",
        build_instruction=_cc_build_instruction("alternating_case"),
        grade_compliance=_cc_grade("alternating_case"),
    ),
    "repeat_sentences": ControlMode(
        name="repeat_sentences",
        source="cotcontrol",
        build_instruction=_cc_build_instruction("repeat_sentences"),
        grade_compliance=_cc_grade("repeat_sentences"),
    ),
    "end_of_sentence": ControlMode(
        name="end_of_sentence",
        source="cotcontrol",
        build_instruction=_cc_build_instruction("end_of_sentence"),
        grade_compliance=_cc_grade("end_of_sentence"),
    ),
    "meow_between_words": ControlMode(
        name="meow_between_words",
        source="cotcontrol",
        build_instruction=_cc_build_instruction("meow_between_words"),
        grade_compliance=_cc_grade("meow_between_words"),
    ),
    "ignore_question": ControlMode(
        name="ignore_question",
        source="cotcontrol",
        build_instruction=_cc_build_instruction("ignore_question"),
        grade_compliance=_cc_grade("ignore_question"),
        requires_llm_judge=True,
    ),
}


# ---------------------------------------------------------------------------
# ReasonIF modes
# ---------------------------------------------------------------------------

def _rif_build_instruction(instruction_type: str) -> Callable[[Sample, str], str]:
    """Create an instruction builder for a ReasonIF mode."""
    from controllability.evals.prompts import build_reasonif_instruction
    def builder(sample: Sample, model_name: str = "") -> str:
        return build_reasonif_instruction(sample)
    return builder


def _rif_grade(instruction_type: str) -> Callable[[str, Sample], dict]:
    """Create a compliance grader for a ReasonIF instruction type."""
    def grader(reasoning: str, sample: Sample) -> dict:
        return grade_reasonif_compliance(reasoning, sample)
    return grader


REASONIF_MODES = {
    "reasoning_language": ControlMode(
        name="reasoning_language",
        source="reasonif",
        build_instruction=_rif_build_instruction("reasoning_language"),
        grade_compliance=_rif_grade("reasoning_language"),
    ),
    "number_words": ControlMode(
        name="number_words",
        source="reasonif",
        build_instruction=_rif_build_instruction("number_words"),
        grade_compliance=_rif_grade("number_words"),
    ),
    "english_capital": ControlMode(
        name="english_capital",
        source="reasonif",
        build_instruction=_rif_build_instruction("english_capital"),
        grade_compliance=_rif_grade("english_capital"),
    ),
    "end_checker": ControlMode(
        name="end_checker",
        source="reasonif",
        build_instruction=_rif_build_instruction("end_checker"),
        grade_compliance=_rif_grade("end_checker"),
    ),
    "json_format": ControlMode(
        name="json_format",
        source="reasonif",
        build_instruction=_rif_build_instruction("json_format"),
        grade_compliance=_rif_grade("json_format"),
    ),
    "no_comma": ControlMode(
        name="no_comma",
        source="reasonif",
        build_instruction=_rif_build_instruction("no_comma"),
        grade_compliance=_rif_grade("no_comma"),
    ),
}


# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------

ALL_MODES: dict[str, ControlMode] = {**COTCONTROL_MODES, **REASONIF_MODES}


def get_mode(name: str) -> ControlMode:
    """Get a control mode by name."""
    if name not in ALL_MODES:
        available = ", ".join(sorted(ALL_MODES.keys()))
        raise ValueError(f"Unknown mode '{name}'. Available: {available}")
    return ALL_MODES[name]


def list_modes() -> list[str]:
    """Return all available mode names."""
    return sorted(ALL_MODES.keys())
