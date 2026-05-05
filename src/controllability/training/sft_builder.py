"""Generalized SFT dataset builder.

Three-stage pipeline, source-agnostic:

  1. Load questions from a shared pool and plan assignments — one row per
     SFT example, across one or more data sources (reasonif / cotcontrol)
     with even mode distribution.
  2. Stage-1 rollouts: for each *unique* question, generate raw reasoning
     from the base model with the instruction stripped. JSONL-checkpointed
     so re-running is cheap.
  3. Stage-2 transform + verify: apply the source/mode-specific edit to
     the rollout's reasoning, verify it passes the compliance checker,
     retry on failure. Build the final (instruction-wrapped) user prompt.
  4. Stage-3 output: write one parquet matching the existing training schema,
     plus a ``.meta.json`` sidecar with all config options. The same config
     is embedded in parquet Arrow metadata so it travels with the file.

Strategies (currently only ``generate_then_edit`` is implemented) are a
separate axis from sources — the strategy decides *how* Stage-1/Stage-2
produce a compliant (reasoning, content) pair. Future strategies can slot
in without touching source code.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from controllability.config import Settings
from controllability.evals.grading import (
    END_OF_SENTENCE_WORD,
    MEOW_TARGET,
    REPEAT_SENTENCE_TARGET,
    _cc_alternating_case,
    _cc_end_of_sentence,
    _cc_lowercase,
    _cc_meow,
    _cc_multiple_word_suppression,
    _cc_repeat_sentences,
    _cc_uppercase,
    _cc_word_suppression,
    grade_reasonif_compliance,
    judge_ignore_question_async,
)
from controllability.evals.prompts import (
    build_cotcontrol_instruction,
    build_cotcontrol_prompts,
    build_reasonif_baseline_prompt,
    build_reasonif_prompt_from_parts,
)
from controllability.inference.base import InferenceClient
from controllability.inference.batch import run_batch
from controllability.inference.openrouter import OpenRouterClient
from controllability.training.transforms import (
    BASE_MODEL_TRANSFORMS,
    COTCONTROL_MODES,
    LLM_TRANSFORMS,
    LLM_TRANSFORMS_WITH_INSTRUCTION,
    REASONIF_MODES,
    TransformContext,
    apply_transform,
    filter_constraint_args,
    pick_keywords_from_question,
    remove_instruction_meta,
)
from controllability.types import InferenceRequest, Sample


# ---------------------------------------------------------------------------
# Question pool filtering
# ---------------------------------------------------------------------------
#
# A handful of questions in the default ReasonIF-FT pool naturally produce
# non-Latin script in the answer (Japanese hiragana, etc.). These trip the
# strict CoTControl ``isalpha()`` / ``islower()`` graders for
# lowercase_thinking and uppercase_thinking — Japanese ``あ`` is alphabetic
# but has no case, so the row counts as non-compliant even when every Latin
# letter is correctly cased. We filter these out at the pool stage so
# they're never assigned to any mode.
#
# Filter targets only:
#   * Translate-pattern questions whose target language uses a non-Latin
#     script (Japanese, Chinese, Korean, Hindi, etc.).
#   * Questions whose own text contains non-ASCII characters (e.g. German
#     umlauts in pronunciation prompts).
# All other multilingual / language-mentioning questions stay — they
# typically don't require non-Latin output.

_NON_LATIN_LANGUAGES = (
    "japanese", "chinese", "mandarin", "cantonese", "korean", "hindi",
    "arabic", "russian", "thai", "greek", "hebrew", "persian", "farsi",
    "urdu", "bengali", "tamil", "tibetan", "ukrainian", "bulgarian",
    "serbian", "armenian", "georgian", "amharic", "sinhala", "khmer",
    "lao", "burmese", "mongolian",
)

_TRANSLATE_PATTERN = re.compile(
    r"\b(translate|translation|how (do|to) (you )?(say|write|spell|pronounce))\b",
    re.IGNORECASE,
)
_NON_LATIN_LANG_PATTERN = re.compile(
    r"\b(" + "|".join(_NON_LATIN_LANGUAGES) + r")\b",
    re.IGNORECASE,
)
def _is_non_latin_question(q: str) -> bool:
    """True if ``q`` would naturally produce non-Latin script in the answer.

    We ONLY flag translate-style questions whose target is a non-Latin
    script (Japanese, Chinese, Hindi, etc.). A blanket "any non-ASCII char"
    rule was tried earlier but proved too aggressive — it caught hundreds
    of legitimate programming / pronunciation questions whose answer is
    perfectly Latin-scripted (e.g. German `Eichhörnchen` pronunciation,
    code with Unicode quotes).
    """
    return bool(
        _TRANSLATE_PATTERN.search(q) and _NON_LATIN_LANG_PATTERN.search(q)
    )


def filter_non_latin_questions(questions: list[str]) -> tuple[list[str], list[str]]:
    """Return (kept, dropped) — drops questions that would force non-Latin output."""
    kept: list[str] = []
    dropped: list[str] = []
    for q in questions:
        if _is_non_latin_question(q):
            dropped.append(q)
        else:
            kept.append(q)
    return kept, dropped


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default mode sets per source. "baseline" is excluded — nothing to learn.
DEFAULT_REASONIF_MODES = [
    "reasoning_language", "number_words", "english_capital",
    "end_checker", "json_format", "no_comma",
]
DEFAULT_COTCONTROL_MODES = [
    "word_suppression", "multiple_word_suppression",
    "uppercase_thinking", "lowercase_thinking", "alternating_case",
    "repeat_sentences", "end_of_sentence", "meow_between_words",
    "ignore_question",
]


@dataclass
class SourceRequest:
    """One data source's share of the final dataset."""

    name: str                         # "reasonif" | "cotcontrol"
    count: int                        # number of SFT rows from this source
    modes: list[str] | None = None    # None → default mode list for the source

    def resolve_modes(self) -> list[str]:
        if self.modes is not None:
            return list(self.modes)
        if self.name == "reasonif":
            return list(DEFAULT_REASONIF_MODES)
        if self.name == "cotcontrol":
            return list(DEFAULT_COTCONTROL_MODES)
        raise ValueError(f"Unknown source: {self.name}")


@dataclass
class BuildConfig:
    """All knobs for a single ``build_sft_dataset`` invocation.

    Stored verbatim in the output sidecar + parquet metadata for provenance.
    """

    # Output
    output_path: Path
    checkpoint_path: Path

    # Question pool (parquet with an ``original_prompt`` column)
    question_source_parquet: Path

    # Base model (Stage 1 rollouts + ignore_question regeneration)
    base_model: str
    backend: str                                 # "openrouter" | "tinker"

    # Source for canonical reasonif constraint templates (constraint_name,
    # constraint_args, instruction_description columns — like the
    # reasonIF_ft/train-*.parquet file). Defaults to the question source if
    # not set; override when ``question_source_parquet`` is a pool that
    # doesn't carry those template columns.
    reasonif_template_source: Path | None = None

    reasoning_effort: str | None = None
    max_tokens: int = 26000
    temperature: float = 1.0
    request_timeout: int = 1800
    max_concurrency: int = 500
    max_retries: int = 3
    system_prompt: str = ""

    # Post-Stage-3 validation. ``train_max_length`` + ``renderer_type`` enable
    # the length filter (re-renders each row, drops those exceeding the cap).
    # The content validator (catches empty / leaked-token / space-stripped
    # rows) always runs and is free.
    train_max_length: int | None = None
    renderer_type: str | None = None

    # Strategy
    strategy: str = "generate_then_edit"

    # Sources (composition)
    sources: list[SourceRequest] = field(default_factory=list)

    # Editor model for reasonif LLM transforms + cotcontrol word-suppression
    editor_model: str = "openai/gpt-5.4"

    # Stage-1 prompt variant.
    # * False (default, current behavior): the reasoning constraint is stripped
    #   from the Stage-1 prompt, so the model produces unconstrained reasoning
    #   that Stage-2 edits to compliance.
    # * True: the Stage-1 prompt includes the constraint instruction (matches
    #   what the trained model will see at deployment). Stage-2 then runs a
    #   meta-removal pre-edit, a pre-compliance check, and only edits rows
    #   that aren't already compliant. Word-suppression keywords are pre-picked
    #   from the question alone so the constraint can be baked into the prompt.
    include_instruction_at_generation: bool = False

    # Prompt rewriting
    analysis_channel: bool | None = None          # None → auto (True for gpt-oss)

    # Random seed (controls question sampling + ignore_question target pick)
    seed: int = 42

    # Transform retries on non-compliance (within a single Stage-2 attempt
    # for one row).
    max_transform_retries: int = 3

    # After Stage 2 finishes, optionally run ONE additional pass on rows that
    # are still non-compliant. The retry re-runs the full per-row transform
    # pipeline (meta-removal pre-edit → pre-compliance check → main edit) so
    # it picks up editor nondeterminism. Cheap because it touches only the
    # already-failing minority. Set to False to disable.
    final_retry_non_compliant: bool = True

    # Whether to drop non-compliant rows from the output parquet entirely.
    # When True (default), only rows that passed the canonical grader make
    # it into the training data; non-compliant rows are dropped at write
    # time. When False, non-compliant rows are kept and the ``compliant``
    # column records True/False (legacy behavior).
    exclude_non_compliant: bool = True

    # Whether to drop questions whose natural answer requires non-Latin
    # script (e.g. "translate 'thank you' into Japanese"). These trip the
    # strict ``isalpha`` / ``islower`` lowercase_thinking grader because
    # hiragana / Chinese / Hindi / Arabic chars are alphabetic but have
    # no case. Affects ~5/953 questions in the default ReasonIF-FT pool.
    filter_non_latin_questions: bool = True

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot of the config (Paths become strings)."""
        d = asdict(self)
        for k in ("output_path", "checkpoint_path", "question_source_parquet",
                 "reasonif_template_source"):
            v = d.get(k)
            d[k] = str(v) if v is not None else None
        d["sources"] = [asdict(s) for s in self.sources]
        return d


# ---------------------------------------------------------------------------
# Prompt construction — delegates to evals/prompts.py for single-source-of-truth.
#
# Three prompt forms per row:
#   - Stage-1 generation prompt: instruction-stripped, used to elicit a raw CoT
#     from the base model. Equal to the eval prompt minus the reasoning
#     constraint.
#   - SFT training user prompt: the full eval prompt as the trained model will
#     see it at deployment time, *minus* the MCQ format directive (per build
#     spec — keeps SFT prompts a strict subset of eval prompts).
#   - Eval prompt: built by evals/prompts.py during evaluation. Unchanged here.
# ---------------------------------------------------------------------------


def _model_name_for_ac(analysis_channel: bool) -> str:
    """Synthetic model_name for eval helpers.

    Eval helpers detect "analysis channel" vs "reasoning stage" via substring
    match on ``"gpt-oss"`` in model_name. When SFT's analysis_channel is
    explicitly set (overriding the auto-detect from base_model), we synthesize
    a name that yields the desired branch.
    """
    return "openai/gpt-oss" if analysis_channel else "qwen/qwen3"


def _make_cc_sample(
    question: str, mode: str, work_args: dict | None, extras: dict | None,
) -> Sample:
    """Build a Sample for use with CC eval prompt helpers.

    Populates the metadata fields the eval helpers read (currently just
    ``keywords`` for word_suppression / multiple_word_suppression). Always
    leaves ``options=None``: SFT's question pool doesn't carry structured
    MCQ choices, and even if it did, callers pass ``mcq_format=False``.
    """
    extras = extras or {}
    metadata: dict = {}
    if mode == "word_suppression":
        kw = extras.get("keyword") or (work_args or {}).get("keyword") or ""
        metadata["keywords"] = [kw] if kw else []
    elif mode == "multiple_word_suppression":
        kws = extras.get("keywords") or (work_args or {}).get("keywords") or []
        metadata["keywords"] = list(kws)
    return Sample(
        id="_", dataset="_", question=question, correct_answer="",
        options=None, metadata=metadata,
    )


def _build_stage1_prompt(
    question: str,
    source: str,
    analysis_channel: bool,
    *,
    include_instruction: bool = False,
    mode: str | None = None,
    constraint_args: dict | None = None,
    instruction_description: str = "",
) -> str:
    """Source-aware Stage-1 generation prompt.

    With ``include_instruction=False`` (default, legacy behavior): returns the
    eval prompt with the reasoning constraint stripped — i.e. the question
    presented in the same shape the trained model will see at eval time,
    but without the constraint. For CC also without the MCQ format line.

    With ``include_instruction=True``: returns the FULL eval-equivalent
    prompt including the constraint instruction. Matches what the trained
    model is expected to handle at deployment. Mode/constraint_args/
    instruction_description are required so the constraint text can be
    rendered (and for word_suppression the keyword(s) must already be
    populated in constraint_args).
    """
    if not include_instruction:
        if source == "reasonif":
            return build_reasonif_baseline_prompt(question, analysis_channel)
        if source == "cotcontrol":
            sample = Sample(
                id="_", dataset="_", question=question, correct_answer="",
                options=None, metadata={},
            )
            _, user = build_cotcontrol_prompts(
                "baseline", sample,
                model_name=_model_name_for_ac(analysis_channel),
                mcq_format=False,
            )
            return user
        raise ValueError(f"Unknown source: {source}")

    # With-instruction path: build the full eval-equivalent prompt.
    if mode is None:
        raise ValueError("mode is required when include_instruction=True")

    if source == "reasonif":
        desc = instruction_description or ""
        # number_words descriptions reference the word count in
        # constraint_args; if num_words is set, ensure the description
        # reflects it (Stage-2 may later rewrite this dynamically per row).
        if mode == "number_words" and constraint_args and constraint_args.get("num_words"):
            desc = f"When reasoning, respond with less than {int(constraint_args['num_words'])} words."
        if analysis_channel:
            desc = _rewrite_rif_desc_for_ac(desc)
        return build_reasonif_prompt_from_parts(
            question, desc, analysis_channel=analysis_channel,
        )
    if source == "cotcontrol":
        sample = _make_cc_sample(question, mode, constraint_args, {})
        model_name = _model_name_for_ac(analysis_channel)
        _, user = build_cotcontrol_prompts(
            mode, sample, model_name=model_name, mcq_format=False,
        )
        return user
    raise ValueError(f"Unknown source: {source}")


# ---------------------------------------------------------------------------
# Canonical reasonif templates (constraint_args + instruction_description)
# ---------------------------------------------------------------------------

_RIF_CATEGORY_TO_SHORT = {
    "punctuation:no_comma": "no_comma",
    "startend:end_checker": "end_checker",
    "detectable_format:json_format": "json_format",
    "language:reasoning_language": "reasoning_language",
    "change_case:english_capital": "english_capital",
    "length_constraint_checkers:number_words": "number_words",
}


def _unwrap(value):
    if isinstance(value, np.ndarray):
        return value[0] if len(value) else None
    return value


def _load_reasonif_templates(source_parquet: Path) -> dict[str, list[dict]]:
    """Group source parquet rows by reasonif mode, returning constraint_args
    and instruction_description templates per mode."""
    df = pd.read_parquet(source_parquet)
    by_mode: dict[str, list[dict]] = {m: [] for m in DEFAULT_REASONIF_MODES}
    for _, row in df.iterrows():
        raw_name = str(_unwrap(row["constraint_name"]) or "")
        short = _RIF_CATEGORY_TO_SHORT.get(raw_name, raw_name)
        if short not in by_mode:
            continue
        raw_args = _unwrap(row["constraint_args"])
        args = dict(raw_args) if isinstance(raw_args, dict) else None
        desc = str(_unwrap(row.get("instruction_description", "")) or "")
        by_mode[short].append({
            "constraint_args": args,
            "constraint_name_raw": np.array([raw_name]) if raw_name else row["constraint_name"],
            "instruction_description": desc,
        })
    return by_mode


# ---------------------------------------------------------------------------
# Assignment planning
# ---------------------------------------------------------------------------


@dataclass
class Assignment:
    """One planned SFT row before generation runs."""

    row_idx: int                 # output row index
    question_idx: int            # index into the question pool
    question: str
    source: str                  # "reasonif" | "cotcontrol"
    mode: str
    constraint_args: dict | None
    instruction_description: str
    canonical_constraint_name_raw: Any | None = None  # only for reasonif, preserves ndarray shape


def _even_mode_counts(total: int, modes: list[str]) -> dict[str, int]:
    """Split ``total`` across ``modes`` as evenly as possible.

    Remainder is distributed to the first modes in ``modes`` so the result
    is deterministic given the input order.
    """
    n = len(modes)
    if n == 0 or total <= 0:
        return {}
    base = total // n
    rem = total % n
    out = {}
    for i, m in enumerate(modes):
        out[m] = base + (1 if i < rem else 0)
    return out


def plan_assignments(
    config: BuildConfig,
    questions: list[str],
    reasonif_templates: dict[str, list[dict]],
) -> list[Assignment]:
    """Build the list of SFT assignments before any rollouts happen.

    Each question in the pool is used at most once, so there's a 1:1
    mapping between questions and SFT rows. Per-source / per-mode counts
    follow even distribution; mode labels are shuffled globally before
    pairing so the resulting data set isn't ordered by mode.
    """
    rng = random.Random(config.seed)

    # Stable question indexing: assign an index per *unique* question.
    seen: dict[str, int] = {}
    unique_questions: list[str] = []
    for q in questions:
        if q and q not in seen:
            seen[q] = len(unique_questions)
            unique_questions.append(q)

    # Build the full list of (source, mode) labels based on the per-source
    # counts and the per-mode even split within each source.
    labels: list[tuple[str, str]] = []
    for src in config.sources:
        per_mode = _even_mode_counts(src.count, src.resolve_modes())
        for mode, k in per_mode.items():
            labels.extend([(src.name, mode)] * k)

    if len(labels) > len(unique_questions):
        raise ValueError(
            f"Pool has {len(unique_questions)} unique questions but "
            f"{len(labels)} assignments were requested. Either enlarge the "
            "question pool or reduce the per-source counts."
        )

    # Shuffle both — one question ↔ one label pairing, no duplicates on either
    # side.
    rng.shuffle(labels)
    q_order = list(range(len(unique_questions)))
    rng.shuffle(q_order)

    assignments: list[Assignment] = []
    for row_idx, ((source, mode), q_idx) in enumerate(zip(labels, q_order)):
        question = unique_questions[q_idx]
        args, desc, raw_cname = _canonical_args_for(
            source, mode, reasonif_templates, rng,
        )
        assignments.append(Assignment(
            row_idx=row_idx,
            question_idx=q_idx,
            question=question,
            source=source,
            mode=mode,
            constraint_args=args,
            instruction_description=desc,
            canonical_constraint_name_raw=raw_cname,
        ))
    return assignments


def _sample_with_optional_replacement(
    pool: list[int], k: int, rng: random.Random,
) -> list[int]:
    """Sample ``k`` elements; use sampling-without-replacement when feasible."""
    if k <= 0:
        return []
    if k <= len(pool):
        return rng.sample(pool, k)
    # Need more than the pool has — sample the pool exhaustively, then fill
    # the remainder with random picks.
    result = list(pool)
    rng.shuffle(result)
    while len(result) < k:
        result.append(rng.choice(pool))
    return result[:k]


def _canonical_args_for(
    source: str, mode: str,
    reasonif_templates: dict[str, list[dict]],
    rng: random.Random,
) -> tuple[dict | None, str, Any | None]:
    """Return (constraint_args, instruction_description, raw_constraint_name)
    for a newly-planned row."""
    if source == "reasonif":
        templates = reasonif_templates.get(mode, [])
        if not templates:
            raise ValueError(f"No reasonif template rows for mode: {mode}")
        t = rng.choice(templates)
        args = dict(t["constraint_args"]) if t["constraint_args"] else None
        return args, t["instruction_description"], t["constraint_name_raw"]

    if source == "cotcontrol":
        # For rule-based modes we pre-set the control value; for LLM-picked
        # modes (word_suppression) we leave keyword unset (filled in Stage 2).
        args: dict | None = None
        desc = ""
        if mode == "word_suppression":
            args = {"keyword": None}
        elif mode == "multiple_word_suppression":
            args = {"keywords": None}
        elif mode == "end_of_sentence":
            args = {"target_word": END_OF_SENTENCE_WORD}
        elif mode == "meow_between_words":
            args = {"target_word": MEOW_TARGET}
        elif mode == "repeat_sentences":
            args = {"target_sentence": REPEAT_SENTENCE_TARGET}
        # instruction_description is assembled in Stage 2 once we know the
        # control value + analysis_channel setting.
        return args, desc, None

    raise ValueError(f"Unknown source: {source}")


# ---------------------------------------------------------------------------
# Stage-1 checkpointing
# ---------------------------------------------------------------------------


# (question_id, source, mode_or_blank, include_instruction).
# ``question_id`` is a stable hash of the question text — robust to changes
# in pool ordering, additions, or filtering (which would shift integer
# question_idx values and silently mismatch cached rollouts to the wrong
# question).
# When include_instruction=False, mode_or_blank is "" — the Stage-1 prompt is
# mode-independent so a single rollout is shared across modes for a given
# (question, source). When include_instruction=True, the Stage-1 prompt is
# mode-specific so each (question, source, mode) gets its own rollout.
_CheckpointKey = tuple[str, str, str, bool]


def _question_id(question: str) -> str:
    """Stable identifier for a question — hash of the trimmed text.

    Decoupled from question_idx so that changes to the question pool
    (filtering, reordering, additions) don't silently mismatch cached
    Stage-1 rollouts to the wrong assignments.
    """
    import hashlib
    return hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:16]


def _load_checkpoint(path: Path) -> dict[_CheckpointKey, dict]:
    """Load Stage-1 checkpoint records keyed by (q_id, source, mode, include).

    Records produced by older builds (without a `source` field) are rejected
    with a clear error. Records that have an integer ``question_idx`` but no
    ``question_id`` field are migrated on-the-fly: we recover the question
    text from the ``generation_prompt`` and hash it.
    """
    completed: dict[_CheckpointKey, dict] = {}
    if not path.exists():
        return completed
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "source" not in rec:
                raise ValueError(
                    f"Checkpoint at {path} predates source-aware Stage-1 prompts "
                    "(missing 'source' field). Delete the checkpoint and re-run "
                    "to regenerate rollouts under the new prompt format."
                )
            include = bool(rec.get("include_instruction", False))
            mode = rec.get("mode", "") if include else ""
            qid = rec.get("question_id")
            if not qid:
                # Migrate older records: extract question text from the
                # generation_prompt and hash it.
                qtext = _extract_question_from_prompt(
                    rec.get("generation_prompt", ""), rec.get("source", ""),
                )
                if not qtext:
                    raise ValueError(
                        f"Checkpoint record in {path} has neither "
                        "'question_id' nor a recognizable question in "
                        "'generation_prompt' — delete and re-run."
                    )
                qid = _question_id(qtext)
                rec["question_id"] = qid
            completed[(qid, rec["source"], mode, include)] = rec
    return completed


def _extract_question_from_prompt(prompt: str, source: str) -> str:
    """Best-effort recovery of the original question from a saved Stage-1
    generation prompt. Used to migrate pre-question_id checkpoints."""
    if not prompt:
        return ""
    # CC baseline / with-instruction prompt format: starts with "Question: ..."
    m = re.search(r"^Question:\s*(.+?)(?:\n\n|$)", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    # ReasonIF baseline format: "...Here is the question:\n\n<question>"
    m = re.search(r"Here is the question:\s*(.+?)$", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    return prompt.strip()


def _append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Stage 1: generation
# ---------------------------------------------------------------------------


def _think_prefill_for_base_model(model: str) -> str | None:
    """Stage-1 base-model prefill.

    Returns None for every family — empirically, no manual prefill is
    needed:
      - gpt-oss: uses analysis channel, no `<think>`.
      - qwen3: model self-emits `<think>\\n` after the assistant header
        (its training format). Manual prefill caused double `<think>`.
      - deepseek/kimi/nemotron: their renderers auto-prefill `<think>`
        (or `<think>\\n` for nemotron) when building the request,
        so a manual prefill here would either replace or double up.
    """
    return None


def _classify_rollout(
    reasoning: str | None,
    content: str | None,
    raw_text: str | None = None,
) -> str:
    """Classify a stage-1 rollout's structure.

    With ``raw_text`` (assistant turn including any prefill), substring-checks
    `<think>` / `</think>` for the principled distinction. Returns one of:

      "ok"                  — `<think>{r}</think>{a}` in correct order, with
                              non-empty `a`. Also accepts `<think></think>{a}`
                              (clean empty think + answer).
      "no_open"             — has `</think>` but no `<think>`.
      "no_close"            — has `<think>` but no `</think>`.
      "no_markers"          — has neither `<think>` nor `</think>`.
      "wrong_order"         — `</think>` appears before `<think>`.
      "no_answer"           — well-formed `<think>...</think>` but nothing after.

    Falls back to a length heuristic when ``raw_text`` is missing.
    """
    if raw_text is None or not raw_text:
        # Legacy fallback. Existing callers without raw_text get the prior
        # heuristic; "ok" is the only label preserved across both forms.
        r = (reasoning or "").strip()
        c = (content or "").strip()
        if not r and not c:
            return "no_answer"
        if not r and len(c) > 200:
            return "no_markers"
        if r and not c:
            return "no_answer"
        return "ok"

    has_open = "<think>" in raw_text
    has_close = "</think>" in raw_text

    if not has_open and not has_close:
        return "no_markers"
    if has_open and not has_close:
        return "no_close"
    if has_close and not has_open:
        return "no_open"

    open_idx = raw_text.index("<think>")
    close_idx = raw_text.index("</think>")
    if close_idx < open_idx:
        return "wrong_order"

    after_close = raw_text[close_idx + len("</think>"):].strip()
    if not after_close:
        return "no_answer"
    return "ok"


async def _build_stage1_request(
    config: BuildConfig, generation_prompt: str,
) -> InferenceRequest:
    messages: list[dict] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": generation_prompt})
    return InferenceRequest(
        messages=messages,
        model=config.base_model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        prefill=_think_prefill_for_base_model(config.base_model),
    )


async def stage1_generate(
    config: BuildConfig,
    assignments: list[Assignment],
    base_client: InferenceClient,
    analysis_channel: bool,
) -> dict[_CheckpointKey, dict]:
    """Run Stage-1 rollouts.

    Dedup behavior depends on ``config.include_instruction_at_generation``:
      * False → one rollout per (question, source) (mode-independent prompt).
      * True  → one rollout per (question, source, mode) (mode-specific prompt
        because the constraint instruction is embedded).

    Validates each rollout against the malformed criteria. On detection,
    retries once. Records that remain malformed after retry are stored with
    ``error="malformed_think:<kind>"`` so stage 2/3 skip them.

    Returns a mapping ``key -> {reasoning, content, error, generation_prompt,
    source, mode, include_instruction}``.
    """
    include_instruction = config.include_instruction_at_generation

    # Dedupe by the appropriate key shape — keyed by question text hash
    # (NOT integer question_idx) so cached rollouts survive pool reorderings.
    unique: dict[_CheckpointKey, Assignment] = {}
    for a in assignments:
        mode_key = a.mode if include_instruction else ""
        key = (_question_id(a.question), a.source, mode_key, include_instruction)
        unique.setdefault(key, a)

    completed = _load_checkpoint(config.checkpoint_path)
    pending = [(key, a) for key, a in unique.items() if key not in completed]

    if not pending:
        print(f"  All {len(unique)} rollouts already in checkpoint")
        return completed

    label = "(question, source, mode)" if include_instruction else "(question, source)"
    print(f"  {len(pending)} pending / {len(unique)} unique {label} "
          f"keys (resumed {len(completed)})")

    requests: list[InferenceRequest] = []
    ordered_keys: list[_CheckpointKey] = []
    ordered_prompts: list[str] = []
    for key, a in pending:
        generation_prompt = _build_stage1_prompt(
            a.question, a.source, analysis_channel,
            include_instruction=include_instruction,
            mode=a.mode,
            constraint_args=a.constraint_args,
            instruction_description=a.instruction_description,
        )
        requests.append(await _build_stage1_request(config, generation_prompt))
        ordered_keys.append(key)
        ordered_prompts.append(generation_prompt)

    responses = await run_batch(
        base_client, requests,
        max_concurrency=config.max_concurrency,
        max_retries=config.max_retries,
        desc="Stage 1: Generate",
    )

    # Map qid → assignment so we can store the question text alongside
    # each new record (helps debugging + future migrations).
    qid_to_assignment: dict[str, Assignment] = {
        _question_id(a.question): a for a in assignments
    }

    def _record(key: _CheckpointKey, resp, gen_prompt: str, error: str | None) -> dict:
        qid, source, mode_key, include = key
        a = qid_to_assignment.get(qid)
        return {
            "question_id": qid,
            "question_idx": a.question_idx if a else None,  # legacy / debug
            "question": a.question if a else "",
            "source": source,
            "mode": mode_key,
            "include_instruction": include,
            "reasoning": resp.reasoning,
            "content": resp.content,
            "error": error,
            "generation_prompt": gen_prompt,
        }

    # First pass: write good rollouts; collect malformed for one retry.
    retry_queue: list[tuple[_CheckpointKey, str, str]] = []  # (key, gen_prompt, kind)
    n_first_pass_ok = 0
    n_first_pass_inference_err = 0
    for key, resp, gen_prompt in zip(ordered_keys, responses, ordered_prompts):
        if resp.error:
            # Inference-level error — don't validate, don't retry here
            # (run_batch already retried on the wire).
            record = _record(key, resp, gen_prompt, resp.error)
            completed[key] = record
            _append_checkpoint(config.checkpoint_path, record)
            n_first_pass_inference_err += 1
            continue
        kind = _classify_rollout(
            resp.reasoning, resp.content, getattr(resp, "raw_text", "") or "",
        )
        if kind == "ok":
            record = _record(key, resp, gen_prompt, None)
            completed[key] = record
            _append_checkpoint(config.checkpoint_path, record)
            n_first_pass_ok += 1
        else:
            retry_queue.append((key, gen_prompt, kind))

    print(f"  First pass: {n_first_pass_ok} ok, "
          f"{n_first_pass_inference_err} inference errors, "
          f"{len(retry_queue)} malformed queued for retry")

    # Retry pass — single retry for malformed rollouts.
    if retry_queue:
        retry_requests = [await _build_stage1_request(config, gp) for _, gp, _ in retry_queue]
        retry_responses = await run_batch(
            base_client, retry_requests,
            max_concurrency=config.max_concurrency,
            max_retries=config.max_retries,
            desc="Stage 1: Retry malformed",
        )
        n_retry_ok = 0
        n_retry_still_bad = 0
        kind_counts: dict[str, int] = {}
        for (key, gen_prompt, first_kind), resp in zip(retry_queue, retry_responses):
            kind_counts[first_kind] = kind_counts.get(first_kind, 0) + 1
            if resp.error:
                record = _record(key, resp, gen_prompt, resp.error)
            else:
                kind = _classify_rollout(
                    resp.reasoning, resp.content, getattr(resp, "raw_text", "") or "",
                )
                if kind == "ok":
                    record = _record(key, resp, gen_prompt, None)
                    n_retry_ok += 1
                else:
                    record = _record(key, resp, gen_prompt, f"malformed_think:{kind}")
                    n_retry_still_bad += 1
            completed[key] = record
            _append_checkpoint(config.checkpoint_path, record)
        print(f"  Retry pass: {n_retry_ok} recovered, {n_retry_still_bad} still malformed")
        if kind_counts:
            print(f"    Malformed kinds in first pass: {kind_counts}")

    n_err = sum(1 for r in completed.values() if r.get("error"))
    n_malformed_final = sum(
        1 for r in completed.values()
        if isinstance(r.get("error"), str) and r["error"].startswith("malformed_think:")
    )
    print(f"  Completions: {len(completed)} total, {n_err} errors "
          f"({n_malformed_final} malformed-think remaining)")
    return completed


# ---------------------------------------------------------------------------
# Stage 2: transform + verify
# ---------------------------------------------------------------------------


def _verify_compliance(
    source: str, mode: str, reasoning: str,
    constraint_args: dict | None, extras: dict,
) -> bool:
    """Invoke the canonical grader for (source, mode)."""
    if source == "reasonif":
        # Strip None values — the reasonif checker does **unpack and will
        # choke on keys from sibling modes that happen to be stored in the
        # same union dict.
        clean_args = filter_constraint_args(constraint_args)
        sample = Sample(
            id="_",
            dataset="reasonif",
            question="",
            correct_answer="",
            metadata={"instruction_type": mode, "constraint_args": clean_args or None},
        )
        return bool(grade_reasonif_compliance(reasoning, sample).get("compliant"))

    if source == "cotcontrol":
        if mode == "uppercase_thinking":
            return bool(_cc_uppercase(reasoning))
        if mode == "lowercase_thinking":
            return bool(_cc_lowercase(reasoning))
        if mode == "alternating_case":
            return bool(_cc_alternating_case(reasoning))
        if mode == "end_of_sentence":
            return bool(_cc_end_of_sentence(reasoning, END_OF_SENTENCE_WORD))
        if mode == "meow_between_words":
            return bool(_cc_meow(reasoning, MEOW_TARGET))
        if mode == "repeat_sentences":
            return bool(_cc_repeat_sentences(reasoning, REPEAT_SENTENCE_TARGET))
        if mode == "word_suppression":
            kw = extras.get("keyword", "") or (constraint_args or {}).get("keyword", "")
            return bool(kw) and bool(_cc_word_suppression(reasoning, kw, []))
        if mode == "multiple_word_suppression":
            kws = extras.get("keywords", []) or (constraint_args or {}).get("keywords", []) or []
            return bool(kws) and bool(_cc_multiple_word_suppression(reasoning, kws, []))
        if mode == "ignore_question":
            # ignore_question uses an LLM judge — verified separately (async).
            return True
        raise ValueError(f"Unknown cotcontrol mode: {mode}")

    raise ValueError(f"Unknown source: {source}")


async def _verify_ignore_question(
    reasoning: str, original_question: str, judge_model: str = "gpt-5-mini",
) -> bool:
    """LLM-judge verification for cotcontrol ignore_question SFT rows.

    Returns False on any error to avoid accepting non-compliant rows.
    """
    try:
        result = await judge_ignore_question_async(
            reasoning=reasoning,
            user_prompt=original_question,
            model=judge_model,
            triple_check=False,  # single pass is sufficient for build-time filtering
        )
    except Exception:  # noqa: BLE001
        return False
    return bool(result.get("compliant"))


async def stage2_transform(
    config: BuildConfig,
    assignments: list[Assignment],
    rollouts: dict[_CheckpointKey, dict],
    editor_client: InferenceClient | None,
    base_client: InferenceClient,
    question_pool: list[str],
    analysis_channel: bool,
    judge_ignore: bool = False,
) -> list[dict]:
    """Edit each rollout to be compliant, retrying up to
    ``config.max_transform_retries`` times on failure. Returns output rows.

    Transforms run concurrently — up to ``config.max_concurrency`` rows in
    flight. Per-mode compliance counters are updated atomically on the main
    task after each row settles (asyncio is single-threaded so no lock needed).
    """
    compliance = {src.name: {m: {"total": 0, "ok": 0, "err": 0, "retries": 0}
                             for m in src.resolve_modes()}
                  for src in config.sources}

    from tqdm.asyncio import tqdm_asyncio

    sem = asyncio.Semaphore(max(1, config.max_concurrency))

    async def _process(a: Assignment) -> dict:
        async with sem:
            return await _process_one_assignment(
                a, rollouts, editor_client, base_client, question_pool,
                config, analysis_channel, judge_ignore,
            )

    tasks = [_process(a) for a in assignments]
    results = await tqdm_asyncio.gather(*tasks, desc="Stage 2: Transform")

    # Optional final-retry pass: re-run any rows still non-compliant (and
    # not errored). Cheap because it touches only the failing minority and
    # picks up editor nondeterminism. Replaces the original result whenever
    # the retry succeeds; keeps the original otherwise so we don't regress.
    if config.final_retry_non_compliant:
        retry_indices = [
            i for i, r in enumerate(results)
            if r.get("error") is None and not r.get("compliant")
        ]
        if retry_indices:
            print(f"\n  Final retry: {len(retry_indices)} non-compliant rows")
            retry_tasks = [_process(results[i]["assignment"]) for i in retry_indices]
            retry_results = await tqdm_asyncio.gather(
                *retry_tasks, desc="Stage 2: Final retry",
            )
            n_recovered = 0
            for i, retry_r in zip(retry_indices, retry_results):
                if retry_r.get("error") is None and retry_r.get("compliant"):
                    results[i] = retry_r
                    n_recovered += 1
            print(f"  Final retry: recovered {n_recovered}/{len(retry_indices)}")

    # Aggregate compliance stats from per-row results
    for r in results:
        a: Assignment = r["assignment"]
        per = compliance[a.source][a.mode]
        per["total"] += 1
        if r.get("error") is not None:
            per["err"] += 1
        elif r.get("compliant"):
            per["ok"] += 1
        per["retries"] += r.get("retries", 0)

    # Compliance summary
    print("\n  Compliance summary:")
    for src_name, per_mode in compliance.items():
        for mode, stats in per_mode.items():
            tot = stats["total"]
            if tot == 0:
                continue
            ok = stats["ok"]
            err = stats["err"]
            denom = tot - err
            rate = f"{(ok / denom):.1%}" if denom > 0 else "N/A"
            retries = stats["retries"]
            extra = f", {retries} retries" if retries else ""
            print(f"    {src_name:>10s}/{mode:<28s}  {ok}/{denom} ({rate}), {err} errors{extra}")

    return results


def _can_pre_check_compliance(
    a: Assignment, work_args: dict | None, include_instruction: bool,
) -> bool:
    """Whether the pre-compliance check can be run before any transform.

    Skipped when the grader needs information that hasn't been picked yet:
      * cotcontrol/word_suppression without a keyword (without-instruction
        case picks it after seeing the rollout).
      * cotcontrol/multiple_word_suppression without keywords.
      * cotcontrol/ignore_question — uses an LLM judge; pre-checking would
        burn a judge call on every row regardless of whether it matters.
    """
    if a.source == "cotcontrol":
        if a.mode == "ignore_question":
            return False
        if a.mode == "word_suppression":
            kw = (work_args or {}).get("keyword")
            return bool(kw)
        if a.mode == "multiple_word_suppression":
            kws = (work_args or {}).get("keywords") or []
            return len(kws) > 0
    return True


def _materialize_extras_for_compliant(
    a: Assignment, work_args: dict | None,
) -> dict:
    """Build the ``extras`` dict that downstream record-building expects when
    we skip the transform because the rollout was already compliant."""
    extras: dict = {}
    if a.source == "cotcontrol":
        if a.mode == "word_suppression":
            kw = (work_args or {}).get("keyword") or ""
            extras["keyword"] = kw
        elif a.mode == "multiple_word_suppression":
            kws = (work_args or {}).get("keywords") or []
            extras["keywords"] = list(kws)
    return extras


async def _prepick_suppression_keywords(
    config: BuildConfig,
    assignments: list[Assignment],
    editor_client: InferenceClient | None,
) -> None:
    """For the with-instruction case, pre-pick suppression keywords from the
    QUESTION (not the rollout) so the constraint can be baked into the
    Stage-1 prompt. Mutates ``a.constraint_args`` in place.

    Without-instruction rows are left alone — keywords are picked from the
    rollout in Stage-2 as before.
    """
    if not config.include_instruction_at_generation:
        return
    if editor_client is None:
        return

    targets: list[tuple[Assignment, int]] = []
    for a in assignments:
        if a.source != "cotcontrol":
            continue
        if a.mode == "word_suppression":
            n = 1
        elif a.mode == "multiple_word_suppression":
            n = 3
        else:
            continue
        existing = a.constraint_args or {}
        if a.mode == "word_suppression" and existing.get("keyword"):
            continue
        if a.mode == "multiple_word_suppression" and existing.get("keywords"):
            continue
        targets.append((a, n))

    if not targets:
        return

    print(f"  Pre-picking suppression keywords for {len(targets)} rows "
          f"(with-instruction generation)")

    sem = asyncio.Semaphore(max(1, config.max_concurrency))

    async def _one(a: Assignment, n: int) -> tuple[Assignment, int, list[str]]:
        # Per-assignment deterministic RNG matches _process_one_assignment's
        # seed scheme so a given row always gets the same coin flip.
        local_rng = random.Random(config.seed * 1_000_003 + a.row_idx)
        ctx = TransformContext(
            llm_client=editor_client,
            editor_model=config.editor_model,
            question=a.question,
            full_prompt=a.question,
            include_instruction=True,
            rng=local_rng,
            max_tokens=400,
        )
        # 50/50 inclusion only meaningful for n=1; n>1 keeps default
        # ("at least one not in question").
        prompt_inclusion: str | None = None
        if n == 1:
            prompt_inclusion = "in_prompt" if local_rng.random() < 0.5 else "not_in_prompt"
        async with sem:
            try:
                picks = await pick_keywords_from_question(
                    a.question, n, ctx, prompt_inclusion=prompt_inclusion,
                )
            except Exception:  # noqa: BLE001
                picks = []
        return a, n, picks

    from tqdm.asyncio import tqdm_asyncio
    results = await tqdm_asyncio.gather(
        *[_one(a, n) for (a, n) in targets],
        desc="Stage 0.5: Pre-pick suppression keywords",
    )

    n_filled = 0
    for a, n, picks in results:
        if not picks:
            continue
        a.constraint_args = dict(a.constraint_args) if a.constraint_args else {}
        if a.mode == "word_suppression":
            a.constraint_args["keyword"] = picks[0]
        elif a.mode == "multiple_word_suppression":
            a.constraint_args["keywords"] = list(picks[:n])
        n_filled += 1
    print(f"  Pre-picked keywords for {n_filled}/{len(targets)} rows")


async def _process_one_assignment(
    a: Assignment,
    rollouts: dict[_CheckpointKey, dict],
    editor_client: InferenceClient | None,
    base_client: InferenceClient,
    question_pool: list[str],
    config: BuildConfig,
    analysis_channel: bool,
    judge_ignore: bool,
) -> dict:
    """Transform + verify compliance for a single assignment.

    Flow:
      1. Look up the matching Stage-1 rollout. Skip if missing/errored.
      2. (with-instruction only) Run the meta-removal pre-edit so the
         reasoning doesn't narrate the constraint.
      3. Pre-check compliance against the canonical grader. If already
         compliant, skip the mode-specific edit entirely. (Skipped for
         modes whose target/keyword is unknown until Stage-2 picks it.)
      4. Otherwise run the mode-specific transform with retries.

    Returns a dict with ``assignment``, ``error``, ``compliant``, ``retries``,
    and (on success) a ``record`` ready for stage3 output.
    """
    include_instruction = config.include_instruction_at_generation
    mode_key = a.mode if include_instruction else ""
    rollout_key = (_question_id(a.question), a.source, mode_key, include_instruction)
    rollout = rollouts.get(rollout_key)
    if rollout is None or rollout.get("error"):
        err = rollout.get("error") if rollout else "missing rollout"
        return {"assignment": a, "error": err, "compliant": None, "retries": 0}

    raw_reasoning = rollout["reasoning"] or ""
    content = rollout["content"] or ""
    generation_prompt = rollout.get("generation_prompt", "")

    # Per-assignment deterministic RNG so re-runs pick the same
    # ignore_question target and keyword fallbacks.
    local_rng = random.Random(config.seed * 1_000_003 + a.row_idx)

    ctx = TransformContext(
        llm_client=editor_client,
        editor_model=config.editor_model,
        base_client=base_client,
        base_model=config.base_model,
        question=a.question,
        full_prompt=generation_prompt,
        include_instruction=include_instruction,
        question_pool=question_pool,
        rng=local_rng,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
    )

    # Mutable args copy so transforms can write back (e.g. number_words).
    work_args = dict(a.constraint_args) if a.constraint_args else None

    # 2. Meta-removal pre-edit (with-instruction only). Falls back to the
    #    original reasoning if the editor call fails — better to keep a
    #    partially-meta'd row than to drop it entirely.
    if include_instruction and editor_client is not None and raw_reasoning.strip():
        try:
            raw_reasoning = await remove_instruction_meta(raw_reasoning, ctx)
        except Exception:  # noqa: BLE001 — never fail the row on pre-edit issues
            pass

    # 3. Pre-compliance check. Skip for modes where the keyword/target isn't
    #    known yet (without-instruction word_suppression / multi) — there's
    #    nothing meaningful to check before the transform picks them.
    can_pre_check = _can_pre_check_compliance(a, work_args, include_instruction)
    if can_pre_check:
        try:
            already_compliant = _verify_compliance(
                a.source, a.mode, raw_reasoning, work_args, {},
            )
        except Exception:  # noqa: BLE001
            already_compliant = False
        if already_compliant:
            extras = _materialize_extras_for_compliant(a, work_args)
            record = _build_output_record(
                a, raw_reasoning, content, generation_prompt, work_args, extras,
                analysis_channel,
            )
            record["compliant"] = True
            record["error"] = None
            return {
                "assignment": a, "error": None, "compliant": True,
                "retries": 0, "record": record,
            }

    transformed: str | None = None
    extras: dict = {}
    last_error: str | None = None
    compliant = False
    retries_used = 0

    for attempt in range(1, config.max_transform_retries + 1):
        try:
            transformed, extras = await apply_transform(
                a.mode, raw_reasoning, work_args, ctx,
            )
        except Exception as e:  # noqa: BLE001
            last_error = f"transform error: {e}"
            continue

        compliant = _verify_compliance(
            a.source, a.mode, transformed, work_args, extras,
        )
        if compliant:
            break
        retries_used += 1

    # number_words: hard-truncate if the LLM still exceeded the dynamic limit.
    if (not compliant and a.source == "reasonif" and a.mode == "number_words"
            and work_args and "num_words" in work_args):
        from controllability.training.transforms import _truncate_to_word_limit
        transformed = _truncate_to_word_limit(
            transformed or raw_reasoning, int(work_args["num_words"]) - 1,
        )
        compliant = _verify_compliance(a.source, a.mode, transformed, work_args, extras)

    # ignore_question: optional async LLM-judge verification.
    if compliant and judge_ignore and a.source == "cotcontrol" and a.mode == "ignore_question":
        compliant = await _verify_ignore_question(transformed, a.question)

    if transformed is None:
        return {
            "assignment": a, "error": last_error or "transform failed",
            "compliant": None, "retries": retries_used,
        }

    record = _build_output_record(
        a, transformed, content, generation_prompt, work_args, extras,
        analysis_channel,
    )
    record["compliant"] = compliant
    record["error"] = None
    return {
        "assignment": a, "error": None, "compliant": compliant,
        "retries": retries_used, "record": record,
    }


def _build_output_record(
    a: Assignment,
    transformed: str,
    content: str,
    generation_prompt: str,
    work_args: dict | None,
    extras: dict,
    analysis_channel: bool,
) -> dict:
    """Shape a result row matching the existing SFT parquet schema."""
    source = a.source
    mode = a.mode

    if source == "reasonif":
        desc = a.instruction_description
        # number_words: constraint_args["num_words"] was rewritten in place
        if mode == "number_words" and work_args and "num_words" in work_args:
            desc = f"When reasoning, respond with less than {int(work_args['num_words'])} words."
        if analysis_channel:
            desc = _rewrite_rif_desc_for_ac(desc)
        user_prompt = build_reasonif_prompt_from_parts(
            a.question, desc, analysis_channel=analysis_channel,
        )
        constraint_name_raw = a.canonical_constraint_name_raw
        constraint_args_out = (
            np.array([work_args]) if work_args else np.array([{}], dtype=object)
        )
        instruction_desc_out = np.array([desc])
    elif source == "cotcontrol":
        sample = _make_cc_sample(a.question, mode, work_args, extras)
        model_name = _model_name_for_ac(analysis_channel)
        _, user_prompt = build_cotcontrol_prompts(
            mode, sample, model_name=model_name, mcq_format=False,
        )
        desc = build_cotcontrol_instruction(
            mode, sample, model_name=model_name, include_no_restate=True,
        )
        # Normalize cotcontrol constraint_args for storage
        ca = dict(work_args) if work_args else {}
        if mode == "word_suppression":
            ca["keyword"] = extras.get("keyword", ca.get("keyword", ""))
        elif mode == "multiple_word_suppression":
            ca["keywords"] = list(extras.get("keywords", ca.get("keywords", [])) or [])
        elif mode == "ignore_question":
            if extras.get("ignore_source_question"):
                ca["ignore_source_question"] = extras["ignore_source_question"]
        constraint_name_raw = np.array([f"cotcontrol:{mode}"])
        constraint_args_out = np.array([ca], dtype=object)
        instruction_desc_out = np.array([desc])
    else:
        raise ValueError(f"Unknown source: {source}")

    # messages_original is filled in stage3 with the pre-edit rollout reasoning.
    return {
        "row_idx": a.row_idx,
        "question_idx": a.question_idx,
        "original_prompt": a.question,
        "generation_prompt": generation_prompt,
        "source": source,
        "constraint_name": constraint_name_raw,
        "constraint_args": constraint_args_out,
        "instruction_description": instruction_desc_out,
        "user_prompt": user_prompt,
        "messages": _messages_row(user_prompt, transformed, content),
        "content": content,
    }


def _rewrite_rif_desc_for_ac(desc: str) -> str:
    desc = desc.replace("When reasoning,", "When using your analysis channel,")
    desc = desc.replace(
        "No other reasoning words should follow",
        "No other words should follow",
    )
    return desc


def _messages_row(user_prompt: str, reasoning: str, content: str) -> list[dict]:
    msgs = [{"role": "user", "content": user_prompt}]
    # Always include `thinking` (empty string when there's no reasoning) so
    # the parquet schema is unambiguous and downstream code can audit empty
    # thinking explicitly rather than seeing pyarrow-filled None.
    asst = {
        "role": "assistant",
        "content": content,
        "thinking": reasoning or "",
    }
    msgs.append(asst)
    return msgs


# ---------------------------------------------------------------------------
# Stage 3: write parquet + metadata sidecar
# ---------------------------------------------------------------------------


def stage3_write(
    results: list[dict],
    rollouts: dict[_CheckpointKey, dict],
    config: BuildConfig,
    output_path: Path,
) -> dict:
    """Write the parquet (with raw reasoning in ``messages_original``) and
    the metadata sidecar. Returns a summary dict."""
    include_instruction = config.include_instruction_at_generation
    good_rows: list[dict] = []
    n_excluded_non_compliant = 0
    for r in results:
        if r.get("error") is not None:
            continue
        rec = r["record"]
        if config.exclude_non_compliant and not rec.get("compliant"):
            n_excluded_non_compliant += 1
            continue
        a: Assignment = r["assignment"]
        mode_key = a.mode if include_instruction else ""
        rollout = rollouts.get(
            (_question_id(a.question), a.source, mode_key, include_instruction), {},
        )
        raw_reasoning = rollout.get("reasoning", "") or ""
        rec["messages_original"] = _messages_row(
            rec["user_prompt"], raw_reasoning, rec["content"],
        )
        good_rows.append(rec)
    if n_excluded_non_compliant:
        print(f"  Excluded {n_excluded_non_compliant} non-compliant rows "
              f"(exclude_non_compliant=True)")

    print(f"\n  Writing {len(good_rows)} rows to parquet "
          f"({len(results) - len(good_rows)} errored, dropped)")

    if not good_rows:
        print("  WARNING: no successful rows")
        return {"n_rows": 0, "n_compliant": 0}

    # Build DataFrame preserving schema expected by the trainer / viewer
    records = []
    for rec in good_rows:
        records.append({
            "original_prompt": rec["original_prompt"],
            "generation_prompt": rec["generation_prompt"],
            "source": rec["source"],
            "constraint_name": rec["constraint_name"],
            "constraint_args": rec["constraint_args"],
            "instruction_description": rec["instruction_description"],
            "messages": rec["messages"],
            "messages_original": rec["messages_original"],
            "compliant": rec.get("compliant", False),
        })

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Embed build config in parquet metadata
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=False)
    cfg_bytes = json.dumps(config.to_dict()).encode("utf-8")
    existing = table.schema.metadata or {}
    merged = {
        **existing,
        b"controllability_build_config": cfg_bytes,
        b"controllability_build_version": b"1",
    }
    table = table.replace_schema_metadata(merged)
    pq.write_table(table, output_path)

    # Sidecar JSON for easy extraction / human inspection
    sidecar_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    summary = {
        "n_rows": len(good_rows),
        "n_compliant": int(sum(1 for r in records if r["compliant"])),
        "sources": [
            {
                "name": s.name,
                "count": s.count,
                "modes": s.resolve_modes(),
            }
            for s in config.sources
        ],
    }
    sidecar = {
        "config": config.to_dict(),
        "summary": summary,
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  Parquet: {output_path}")
    print(f"  Sidecar: {sidecar_path}")

    return summary


# ---------------------------------------------------------------------------
# Post-Stage-3 validators
#
# These run after the parquet is written. They guarantee the trainer never
# sees a row that:
#   - exceeds ``max_length`` (would be silently right-truncated, chopping
#     `</think>` and the answer), or
#   - has empty content / empty thinking, leaked special tokens, or text
#     produced by a decoder bug (e.g. space-stripped).
# ---------------------------------------------------------------------------


# Modes that legitimately suppress normal spacing — exempt from space-ratio
# sanity check.
_LOW_SPACE_OK_MODES = {
    "uppercase_thinking", "lowercase_thinking", "alternating_case",
    "english_capital", "no_comma", "json_format", "ignore_question",
    "meow_between_words", "end_of_sentence", "repeat_sentences",
    "word_suppression", "multiple_word_suppression",
}

# Special-token strings that should never appear inside stored content/thinking
# — they indicate decoder leakage from the renderer/tokenizer.
_LEAK_PATTERNS = [
    "<|im_start|>", "<|im_end|>",
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
    "<|start|>", "<|message|>", "<|channel|>",
    "<|endoftext|>",
]


def validate_parquet_content(parquet_path: Path) -> dict:
    """Drop rows from an SFT parquet that are obviously malformed.

    Drops rows where:
      - assistant content (final answer) is empty,
      - assistant thinking is empty (Stage 1 should never store empty
        thinking on success — if it's empty, the row is unusable),
      - content/thinking contains a leaked special token,
      - thinking has <1% spaces in a mode that should preserve spaces
        (catches re-occurrence of decode bugs like the deepseek
        ``transformers==5.3.0`` Hub-tokenizer issue).

    Always preserves the prior parquet as ``.prevalidate.bak`` if any rows
    are dropped. Returns ``{"kept", "dropped", "drop_reasons": {...}}``.
    """
    import shutil
    if not parquet_path.exists():
        return {"kept": 0, "dropped": 0, "drop_reasons": {}}
    df = pd.read_parquet(parquet_path)
    keep: list[bool] = []
    reasons: dict[str, int] = {}

    def _bump(key: str) -> None:
        reasons[key] = reasons.get(key, 0) + 1

    for i in range(len(df)):
        row = df.iloc[i]
        msgs = list(row["messages"])
        asst = next((m for m in msgs if m.get("role") == "assistant"), None)
        if asst is None:
            keep.append(False); _bump("no_assistant"); continue
        content = (asst.get("content") or "").strip()
        thinking = (asst.get("thinking") or "")
        cn = row.get("constraint_name") or []
        cn_str = str(cn).lower()
        is_low_space_ok = any(m in cn_str for m in _LOW_SPACE_OK_MODES)

        if not content:
            keep.append(False); _bump("empty_content"); continue
        if not thinking.strip():
            keep.append(False); _bump("empty_thinking"); continue

        leaked = False
        for pat in _LEAK_PATTERNS:
            if pat in content or pat in thinking:
                _bump(f"leaked:{pat}")
                leaked = True
                break
        if leaked:
            keep.append(False); continue

        if not is_low_space_ok:
            n = len(thinking)
            if n >= 200 and (thinking.count(" ") / n) < 0.01:
                keep.append(False); _bump("low_space_ratio"); continue

        keep.append(True)

    n_kept = sum(keep)
    n_dropped = len(keep) - n_kept
    if n_dropped > 0:
        bak = parquet_path.with_suffix(parquet_path.suffix + ".prevalidate.bak")
        if not bak.exists():
            shutil.copy2(parquet_path, bak)
        df_clean = df[keep].reset_index(drop=True)
        df_clean.to_parquet(parquet_path, index=False)
    return {"kept": n_kept, "dropped": n_dropped, "drop_reasons": reasons}


def filter_parquet_by_max_length(
    parquet_path: Path,
    base_model: str,
    renderer_type: str,
    max_length: int,
    reasoning_effort: str = "none",
) -> dict:
    """Drop rows whose rendered token-length exceeds ``max_length``.

    Re-renders each row through the same renderer the trainer will use, and
    drops anything over the cap. Without this, ``tinker_cookbook`` silently
    right-truncates over-long rows at training time, chopping `</think>`
    and the answer — recreating the malformed pattern we filter at Stage 1.

    Imports ``ft_runner`` lazily to avoid import cycles. Preserves the
    pre-filter parquet as ``.prefilter.bak``.
    """
    import shutil
    from transformers import AutoTokenizer
    from controllability.training.ft_runner import (
        FTModelSpec, _to_structured_parts, make_renderer,
    )

    if not parquet_path.exists() or max_length is None or max_length <= 0:
        return {"kept": 0, "dropped": 0, "max_seen": 0}

    spec = FTModelSpec(
        model=base_model, base_model=base_model,
        parquet=str(parquet_path.name),
        log_path=parquet_path.stem,
        renderer_type=renderer_type,
        reasoning_effort=reasoning_effort,
    )
    trc = "moonshotai" in base_model.lower()
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trc)
    renderer = make_renderer(spec, tok)

    df = pd.read_parquet(parquet_path)
    keep_mask: list[bool] = []
    max_seen = 0
    for i in range(len(df)):
        msgs = list(df.iloc[i]["messages"])
        conv = [{
            "role": m.get("role"),
            "content": _to_structured_parts(
                m.get("role"),
                m.get("content") or "",
                m.get("thinking") or "",
            ),
        } for m in msgs]
        try:
            mi, _w = renderer.build_supervised_example(conv)
            n = sum(c.length for c in mi.chunks)
        except Exception:
            n = max_length + 1  # treat as too long → drop
        max_seen = max(max_seen, n)
        keep_mask.append(n <= max_length)

    n_kept = sum(keep_mask)
    n_dropped = len(keep_mask) - n_kept
    if n_dropped > 0:
        bak = parquet_path.with_suffix(parquet_path.suffix + ".prefilter.bak")
        if not bak.exists():
            shutil.copy2(parquet_path, bak)
        df_filtered = df[keep_mask].reset_index(drop=True)
        df_filtered.to_parquet(parquet_path, index=False)
    return {"kept": n_kept, "dropped": n_dropped, "max_seen": max_seen}


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


async def build_sft_dataset(
    config: BuildConfig, judge_ignore_question: bool = False,
) -> dict:
    """Run the full builder end-to-end. Returns a summary dict."""
    start = time.monotonic()
    print(f"\n{'=' * 60}\nSFT BUILD\n{'=' * 60}")
    print(f"Base model:      {config.base_model} ({config.backend})")
    print(f"Strategy:        {config.strategy}")
    print(f"Question source: {config.question_source_parquet}")
    print("Sources:")
    for s in config.sources:
        print(f"  - {s.name}: {s.count}  modes={s.resolve_modes()}")

    if config.strategy != "generate_then_edit":
        raise NotImplementedError(f"Strategy not yet implemented: {config.strategy}")

    # Load question pool
    df_src = pd.read_parquet(config.question_source_parquet)
    if "original_prompt" not in df_src.columns:
        raise ValueError(
            f"question_source parquet is missing 'original_prompt' column: {config.question_source_parquet}"
        )
    questions = [str(_unwrap(x) or "").strip() for x in df_src["original_prompt"].tolist()]
    questions = [q for q in questions if q]

    # Optional pool-level filter — drop questions whose natural answer
    # forces non-Latin script (those break the strict lowercase / uppercase
    # graders even when the model behaves correctly).
    if config.filter_non_latin_questions:
        questions, dropped_nl = filter_non_latin_questions(questions)
        if dropped_nl:
            print(f"  Filtered {len(dropped_nl)} non-Latin-output questions "
                  f"from pool (keep {len(questions)})")
            for q in dropped_nl[:5]:
                print(f"    - {q[:120]}")
            if len(dropped_nl) > 5:
                print(f"    ... and {len(dropped_nl) - 5} more")

    # ReasonIF template cache (only needed if any source is reasonif).
    # Defaults to the question source if no dedicated template parquet was
    # provided — backwards compatible with runs where both live in one file.
    reasonif_templates: dict[str, list[dict]] = {}
    if any(s.name == "reasonif" for s in config.sources):
        template_path = config.reasonif_template_source or config.question_source_parquet
        reasonif_templates = _load_reasonif_templates(template_path)

    # Plan assignments
    assignments = plan_assignments(config, questions, reasonif_templates)
    if not assignments:
        raise ValueError("No assignments produced — check source counts.")
    print(f"\nPlanned {len(assignments)} SFT rows across "
          f"{len({a.question_idx for a in assignments})} unique questions")

    # Resolve analysis_channel default
    ac = config.analysis_channel
    if ac is None:
        ac = "gpt-oss" in config.base_model.lower()

    # Clients
    settings = Settings()
    base_client, editor_client = _make_clients(config, settings)
    question_pool = [questions[i] for i in sorted({a.question_idx for a in assignments})]

    try:
        # Stage 0.5 — pre-pick suppression keywords for with-instruction so
        # the keyword can be embedded in the Stage-1 prompt.
        if config.include_instruction_at_generation:
            print("\n" + "-" * 60)
            print("STAGE 0.5: Pre-pick suppression keywords (with-instruction)")
            print("-" * 60)
            await _prepick_suppression_keywords(
                config, assignments, editor_client,
            )

        # Stage 1
        print("\n" + "-" * 60)
        if config.include_instruction_at_generation:
            print("STAGE 1: Generate raw rollouts (instruction included)")
        else:
            print("STAGE 1: Generate raw rollouts (instruction stripped)")
        print("-" * 60)
        rollouts = await stage1_generate(
            config, assignments, base_client, analysis_channel=ac,
        )

        # Stage 2
        print("\n" + "-" * 60)
        print("STAGE 2: Transform + verify compliance")
        print(f"  analysis_channel = {ac}")
        print("-" * 60)
        results = await stage2_transform(
            config, assignments, rollouts, editor_client, base_client,
            question_pool, analysis_channel=ac,
            judge_ignore=judge_ignore_question,
        )

        # Stage 3
        print("\n" + "-" * 60)
        print("STAGE 3: Write output parquet + metadata sidecar")
        print("-" * 60)
        summary = stage3_write(results, rollouts, config, config.output_path)

        # Stage 4: post-write validation. Length filter is opt-in (needs
        # renderer config); content validator always runs.
        print("\n" + "-" * 60)
        print("STAGE 4: Post-write validation")
        print("-" * 60)
        if config.train_max_length and config.renderer_type:
            lstats = filter_parquet_by_max_length(
                config.output_path,
                base_model=config.base_model,
                renderer_type=config.renderer_type,
                max_length=config.train_max_length,
                reasoning_effort=config.reasoning_effort or "none",
            )
            print(f"  Length filter (max_length={config.train_max_length}): "
                  f"kept={lstats['kept']} dropped={lstats['dropped']} "
                  f"max_seen={lstats['max_seen']}")
        else:
            print("  Length filter: skipped (set train_max_length + renderer_type to enable)")
        vstats = validate_parquet_content(config.output_path)
        print(f"  Content validator: kept={vstats['kept']} "
              f"dropped={vstats['dropped']} reasons={vstats['drop_reasons']}")
        summary["validation"] = {
            "length_filter": locals().get("lstats"),
            "content_validator": vstats,
        }

    finally:
        await base_client.close()
        if editor_client is not None and editor_client is not base_client:
            await editor_client.close()

    elapsed = time.monotonic() - start
    print(f"\nDone in {elapsed:.0f}s")
    return summary


def _make_clients(
    config: BuildConfig, settings: Settings,
) -> tuple[InferenceClient, InferenceClient | None]:
    """Build the base (generation + ignore_question regen) + editor clients."""
    if config.backend == "openrouter":
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        base = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            request_timeout=config.request_timeout,
            reasoning_effort=config.reasoning_effort,
        )
    elif config.backend == "tinker":
        from controllability.inference.tinker_client import TinkerClient
        base = TinkerClient(
            model=config.base_model,
            reasoning_effort=config.reasoning_effort or "none",
            request_timeout=config.request_timeout,
        )
    else:
        raise ValueError(f"Unknown backend: {config.backend}")

    # Editor client (always OpenRouter) — needed for reasonif LLM transforms
    # and cotcontrol word-suppression keyword picking/minimal-edit.
    editor: InferenceClient | None = None
    if _needs_editor(config):
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY required for editor model (LLM-based transforms)"
            )
        editor = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            request_timeout=config.request_timeout,
        )
    return base, editor


def _needs_editor(config: BuildConfig) -> bool:
    """Whether an editor client is needed.

    With-instruction generation always benefits from an editor (meta-removal
    pre-edit + pre-pick suppression keywords + with-instruction overrides
    like json_format). Without-instruction only needs the editor for the
    legacy LLM_TRANSFORMS that aren't base-model regen.
    """
    if config.include_instruction_at_generation:
        return True
    for s in config.sources:
        for m in s.resolve_modes():
            if m in LLM_TRANSFORMS and m not in BASE_MODEL_TRANSFORMS:
                return True
    return False
