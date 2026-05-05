"""Helpers for the v4 pipeline:

1. ``reset_malformed_in_checkpoint``: rewrites a stage-1 checkpoint.jsonl
   without records that have ``error="malformed_think:..."`` so the pipeline
   re-runs them on resume (now with bumped max_tokens).

2. ``filter_parquet_by_max_length``: re-renders each parquet row with the
   model's renderer and drops rows whose tokenized length exceeds
   ``max_length``. Writes the filtered parquet back in-place (with .bak).

3. ``validate_parquet_content``: drops rows that are obviously malformed
   (empty content/thinking, leaked special tokens, space-stripped text).

Self-contained so the running pipeline can lazy-import this module without
depending on a particular cached version of sft_builder. The canonical
versions of (2) and (3) also live in sft_builder.py for future runs.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from controllability.training.ft_runner import (
    FTModelSpec, _to_structured_parts, make_renderer,
)


def reset_malformed_in_checkpoint(checkpoint_path: Path) -> dict:
    """Strip records with `error="malformed_think:..."` from a stage-1 checkpoint.

    Returns a stats dict: {"kept": int, "dropped": int, "dropped_kinds": dict}.
    """
    if not checkpoint_path.exists():
        return {"kept": 0, "dropped": 0, "dropped_kinds": {}}
    kept_lines: list[str] = []
    dropped = 0
    kinds: dict[str, int] = {}
    with open(checkpoint_path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue
            err = rec.get("error") or ""
            if isinstance(err, str) and err.startswith("malformed_think:"):
                dropped += 1
                kind = err.split(":", 1)[1] if ":" in err else "unknown"
                kinds[kind] = kinds.get(kind, 0) + 1
            else:
                kept_lines.append(line)
    bak = checkpoint_path.with_suffix(checkpoint_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(checkpoint_path, bak)
    with open(checkpoint_path, "w") as f:
        for line in kept_lines:
            f.write(line if line.endswith("\n") else line + "\n")
    return {"kept": len(kept_lines), "dropped": dropped, "dropped_kinds": kinds}


def filter_parquet_by_max_length(
    parquet_path: Path,
    spec: FTModelSpec,
    tokenizer,
    max_length: int,
) -> dict:
    """Drop parquet rows whose rendered token-length exceeds ``max_length``.

    Renders each row through the model's renderer (same path as
    ``ft_runner.SFTDataset``) and counts tokens. Rows over the cap are
    dropped. The original parquet is preserved as ``.bak`` and the
    filtered version replaces the original.

    Returns: {"kept": int, "dropped": int, "max_seen": int}.
    """
    renderer = make_renderer(spec, tokenizer)
    df = pd.read_parquet(parquet_path)
    keep_mask = []
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
            n = max_length + 1
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


# Modes that legitimately produce text without normal spaces. Rows under these
# constraints are exempt from the space-ratio sanity check.
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
      - assistant content is empty,
      - assistant thinking is empty,
      - content/thinking contains a leaked special token,
      - thinking has <1% spaces in a mode that should preserve spaces.

    Always preserves the prior parquet as ``.prevalidate.bak`` if any rows
    are dropped. Returns ``{"kept", "dropped", "drop_reasons": {...}}``.
    """
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
