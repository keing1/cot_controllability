"""Rollout Viewer — Streamlit app for browsing rollout JSONL data.

Launch:  uv run streamlit run scripts/analysis/rollout_viewer.py
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from controllability.evals.prompts import get_monitor_prompt_builder
from controllability.evals.rescore import build_monitor_messages

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLOUTS_DIR = Path(__file__).resolve().parents[2] / "results" / "rollouts"
MONITOR_QA_DIR = ROLLOUTS_DIR / "monitor_qa"

# ---------------------------------------------------------------------------
# Filename parsing — CoTControl / ReasonIF
# ---------------------------------------------------------------------------

# Regex splits on _cotcontrol_ or _reasonif_ to separate model part from rest.
_DATASET_RE = re.compile(r"_(cotcontrol|reasonif)_")

# Fine-tune pattern:  -rif  optionally -stripped  optionally -lr<rate>  and  -<checkpoint>
_FT_RE = re.compile(r"-rif(?:-(stripped))?(?:-lr([\d.]+e-\d+))?(?:-(\d{3,}|final))?$")


@dataclass
class FileMetadata:
    path: Path
    filename: str
    model: str        # e.g. "gpt-oss-120b" or "qwen3-8b"
    dataset: str      # "cotcontrol" or "reasonif"
    is_ft: bool
    is_stripped: bool  # True if trained on stripped SFT data
    lr: str           # "" if not fine-tuned
    checkpoint: str   # "" if not fine-tuned
    provider: str     # e.g. "openai", "qwen"
    reasoning_effort: str  # from header, e.g. "medium", "none", ""
    max_tokens: str        # from header, e.g. "28000", ""
    analysis_channel: str  # "True", "False", or ""


def _read_header_quick(path: Path) -> dict:
    """Read just the first line header from a JSONL file."""
    try:
        with open(path, "r") as f:
            line = f.readline().strip()
            if line.startswith("# "):
                return json.loads(line[2:])
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def parse_filename(path: Path) -> FileMetadata | None:
    """Extract metadata from a rollout filename. Returns None on parse failure."""
    name = path.stem  # drop .jsonl
    m = _DATASET_RE.search(name)
    if not m:
        return None
    dataset = m.group(1)
    model_part = name[: m.start()]  # provider_model

    # Split provider from model on first underscore
    uscore = model_part.find("_")
    if uscore == -1:
        return None
    provider = model_part[:uscore]
    model_full = model_part[uscore + 1 :]

    # Check for fine-tune suffix
    ft = _FT_RE.search(model_full)
    if ft:
        base_model = model_full[: ft.start()]
        is_stripped = ft.group(1) == "stripped"
        # Old-format filenames (-rif-000060) omit the lr; these were all 1e-4
        lr = ft.group(2) or ("1e-4" if ft.group(3) else "")
        checkpoint = ft.group(3) or ""
        is_ft = True
    else:
        base_model = model_full
        is_stripped = False
        lr = ""
        checkpoint = ""
        is_ft = False

    # Read header for reasoning_effort, max_tokens, analysis_channel
    hdr = _read_header_quick(path)
    reasoning_effort = str(hdr.get("reasoning_effort", "") or "")
    max_tokens = str(hdr.get("max_tokens", "") or "")
    ac_val = hdr.get("analysis_channel")
    analysis_channel = str(ac_val) if ac_val is not None else ""

    return FileMetadata(
        path=path,
        filename=path.name,
        model=base_model,
        dataset=dataset,
        is_ft=is_ft,
        is_stripped=is_stripped,
        lr=lr,
        checkpoint=checkpoint,
        provider=provider,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        analysis_channel=analysis_channel,
    )


# ---------------------------------------------------------------------------
# Filename parsing — MonitorQA
# ---------------------------------------------------------------------------

# Pattern: monitor_qa_{model}_{setting}_{datetime}_{run_id}.jsonl
# List longer/more-specific settings first so the alternation picks them
# before shorter prefixes (e.g. `user_default` before `default`).
_MQA_SETTINGS = [
    "baseline_main_and_side",
    "baseline_formatted",
    "user_intense_hiding",
    "user_less_reasoning",
    "user_dont_think",
    "user_misdirect",
    "user_gray_swan",
    "user_roleplay",
    "user_pressure",
    "user_default",
    "dont_think",
    "baseline",
    "default",
]
_MQA_SETTING_RE = re.compile(r"_(" + "|".join(_MQA_SETTINGS) + r")_")


@dataclass
class MonitorQAFileMetadata:
    path: Path
    filename: str
    model: str       # e.g. "gpt-oss-120b" or "qwen3-8b-ft60"
    setting: str     # "baseline", "default", or "dont_think"
    run_id: str      # hex hash at end
    date_str: str    # e.g. "20260402_090606" or "20260403_rerun"
    split: str       # "val", "old", or "unknown"


def parse_monitor_qa_filename(path: Path) -> MonitorQAFileMetadata | None:
    """Extract metadata from a monitor_qa filename. Returns None on failure."""
    name = path.stem  # drop .jsonl
    if not name.startswith("monitor_qa_"):
        return None
    rest = name[len("monitor_qa_"):]  # strip prefix

    # Find all setting matches; take the first one (model doesn't contain these words)
    matches = list(_MQA_SETTING_RE.finditer(rest))
    if not matches:
        return None

    m = matches[0]
    model = rest[: m.start()]
    setting = m.group(1)
    tail = rest[m.end():]  # e.g. "val_split_20260409_061534_c813097b7552"

    # Extract split if present (val_split_ or old_split_)
    split = "old"  # default for files without explicit split
    if tail.startswith("val_split_"):
        split = "val"
        tail = tail[len("val_split_"):]
    elif tail.startswith("old_split_"):
        split = "old"
        tail = tail[len("old_split_"):]

    # run_id is the last _-separated segment (hex hash)
    last_uscore = tail.rfind("_")
    if last_uscore == -1:
        return None
    run_id = tail[last_uscore + 1:]
    date_str = tail[:last_uscore]

    return MonitorQAFileMetadata(
        path=path,
        filename=path.name,
        model=model,
        setting=setting,
        run_id=run_id,
        date_str=date_str,
        split=split,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_records(path: str, _mtime: float) -> list[dict]:
    """Load all non-header JSON records from a JSONL file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


@st.cache_data
def load_header(path: str, _mtime: float) -> dict | None:
    """Load the experiment config from the # header line."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# "):
                try:
                    return json.loads(line[2:])
                except json.JSONDecodeError:
                    return None
            if line and not line.startswith("#"):
                break
    return None


def records_to_df(records: list[dict]) -> pd.DataFrame:
    """Build a summary DataFrame from raw records."""
    rows = []
    for i, r in enumerate(records):
        sample = r.get("sample", {})
        latency_ms = r.get("latency_ms", 0)
        rows.append(
            {
                "idx": i,
                "sample_id": sample.get("id", ""),
                "control_mode": r.get("control_mode", ""),
                "correct": r.get("correct"),
                "compliant": r.get("compliant"),
                "meta_discussion": r.get("meta_discussion"),
                "extracted_answer": r.get("extracted_answer", ""),
                "correct_answer": sample.get("correct_answer", ""),
                "latency_s": round(latency_ms / 1000, 1),
                "question_preview": (sample.get("question", "") or "")[:100],
            }
        )
    return pd.DataFrame(rows)


def _get_monitor_types_from_records(records: list[dict]) -> list[str]:
    """Collect all monitor types present in records, in stable order."""
    seen: dict[str, None] = {}
    for r in records:
        for mr in r.get("monitor_results", []):
            mt = mr.get("monitor_type", "")
            if mt and mt not in seen:
                seen[mt] = None
    return list(seen.keys())


def _get_monitor_type_models(records: list[dict]) -> dict[str, str]:
    """Map each monitor_type to its monitor_model (first seen)."""
    type_to_model: dict[str, str] = {}
    for r in records:
        for mr in r.get("monitor_results", []):
            mt = mr.get("monitor_type", "")
            if mt and mt not in type_to_model:
                model = mr.get("monitor_model", "")
                type_to_model[mt] = model
    return type_to_model


def monitorqa_records_to_df(records: list[dict]) -> pd.DataFrame:
    """Build a summary DataFrame from MonitorQA records."""
    monitor_types = _get_monitor_types_from_records(records)

    rows = []
    for i, r in enumerate(records):
        sample = r.get("sample", {})
        actor_latency = r.get("actor_latency_ms", 0)
        reasoning_lc = (r.get("reasoning") or "").lower()
        mentions_task_terms = (
            "main task" in reasoning_lc or "side task" in reasoning_lc
        )
        mentions_strong = (
            mentions_task_terms
            or "main:" in reasoning_lc
            or "side:" in reasoning_lc
            or "main_task" in reasoning_lc
            or "side_task" in reasoning_lc
            or "both tasks" in reasoning_lc
            or "two tasks" in reasoning_lc
            or "two questions" in reasoning_lc
            or "main question" in reasoning_lc
            or "side question" in reasoning_lc
            or "main answer" in reasoning_lc
            or "side answer" in reasoning_lc
            or "both answer" in reasoning_lc
        )
        row = {
            "idx": i,
            "sample_id": sample.get("id", ""),
            "mode": r.get("mode", ""),
            "monitor_score": r.get("monitor_score"),
            "side_task_correct": r.get("side_task_correct"),
            "mentions_task_terms": mentions_task_terms,
            "mentions_strong": mentions_strong,
            "actor_error": r.get("actor_error") or "",
            "monitor_error": r.get("monitor_error") or "",
            "latency_s": round(actor_latency / 1000, 1) if actor_latency else 0.0,
            "question_preview": (sample.get("question", "") or "")[:100],
        }
        # Per-monitor-type scores
        mt_scores = {
            mr.get("monitor_type"): mr.get("monitor_score")
            for mr in r.get("monitor_results", [])
        }
        for mt in monitor_types:
            row[f"score_{mt}"] = mt_scores.get(mt)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool_filter(series: pd.Series, value: str) -> pd.Series:
    """Return a boolean mask for a True/False/None/All selectbox value."""
    if value == "All":
        return pd.Series(True, index=series.index)
    if value == "True":
        return series == True  # noqa: E712
    if value == "False":
        return series == False  # noqa: E712
    # "None"
    return series.isna()


def _colored(val: bool | None, true_color: str = "green", false_color: str = "red") -> str:
    if val is True:
        return f":{true_color}[True]"
    if val is False:
        return f":{false_color}[False]"
    return "None"


def _monitor_score_color(score: float | None) -> str:
    """Return a CSS color for a monitor score (0=green, 10=red)."""
    if score is None:
        return "#888"
    if score <= 2:
        return "#28a745"  # green
    if score <= 5:
        return "#ffc107"  # yellow
    if score <= 7:
        return "#fd7e14"  # orange
    return "#dc3545"  # red


def _monitor_score_badge(score: float | None) -> str:
    """Return an HTML badge for monitor score."""
    if score is None:
        return '<span style="font-size:1.5em; color:#888;">N/A</span>'
    color = _monitor_score_color(score)
    return (
        f'<span style="display:inline-block; font-size:2em; font-weight:bold; '
        f"color:white; background:{color}; padding:0.15em 0.5em; "
        f'border-radius:0.375em;">{score:.1f}</span>'
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Rollout Viewer", layout="wide")
st.title("Rollout Viewer")

# ---- Sidebar: file selection ------------------------------------------------

with st.sidebar:
    st.header("File Selection")

    if not ROLLOUTS_DIR.exists():
        st.error(f"Rollouts directory not found: {ROLLOUTS_DIR}")
        st.stop()

    # Dataset type toggle
    has_cotcontrol = bool(sorted(ROLLOUTS_DIR.glob("*.jsonl")))
    has_monitorqa = MONITOR_QA_DIR.exists() and bool(sorted(MONITOR_QA_DIR.glob("*.jsonl")))

    if has_cotcontrol and has_monitorqa:
        dataset_type = st.radio(
            "Dataset Type",
            ["CoTControl / ReasonIF", "MonitorQA"],
            horizontal=True,
        )
    elif has_monitorqa:
        dataset_type = "MonitorQA"
    else:
        dataset_type = "CoTControl / ReasonIF"

    is_monitorqa = dataset_type == "MonitorQA"

# ---- Sidebar: CoTControl / ReasonIF mode ------------------------------------

if not is_monitorqa:
    with st.sidebar:
        jsonl_files = sorted(ROLLOUTS_DIR.glob("*.jsonl"))
        if not jsonl_files:
            st.warning("No .jsonl files found in rollouts directory.")
            st.stop()

        # Parse all filenames
        all_meta = [parse_filename(f) for f in jsonl_files]
        all_meta = [m for m in all_meta if m is not None]

        if not all_meta:
            st.warning("Could not parse any rollout filenames.")
            st.stop()

        # --- Cascading filters ---

        all_models = sorted({m.model for m in all_meta})
        sel_models = st.multiselect("Model", all_models, default=all_models)

        all_datasets = sorted({m.dataset for m in all_meta})
        sel_datasets = st.multiselect("Dataset", all_datasets, default=all_datasets)

        ft_filter = st.radio(
            "Fine-tuning", ["All", "Base only", "Fine-tuned only"], horizontal=True
        )

        # Apply base filters
        filtered = [
            m
            for m in all_meta
            if m.model in sel_models
            and m.dataset in sel_datasets
            and (ft_filter == "All" or (ft_filter == "Base only") == (not m.is_ft))
        ]

        # LR filter (only shown if FT files present)
        ft_in_view = [m for m in filtered if m.is_ft]
        if ft_in_view:
            all_lrs = sorted({m.lr for m in ft_in_view if m.lr})
            if all_lrs:
                sel_lrs = st.multiselect("Learning Rate", all_lrs, default=all_lrs)
                filtered = [
                    m
                    for m in filtered
                    if not m.is_ft or not m.lr or m.lr in sel_lrs
                ]

            all_ckpts = sorted({m.checkpoint for m in ft_in_view if m.checkpoint})
            if all_ckpts:
                sel_ckpts = st.multiselect("Checkpoint", all_ckpts, default=all_ckpts)
                filtered = [
                    m
                    for m in filtered
                    if not m.is_ft or not m.checkpoint or m.checkpoint in sel_ckpts
                ]

        # Stripped filter (only shown if FT files exist)
        ft_in_view2 = [m for m in filtered if m.is_ft]
        if ft_in_view2:
            stripped_filter = st.radio(
                "SFT Data", ["All", "Stripped only", "Non-stripped only"], horizontal=True
            )
            if stripped_filter != "All":
                want_stripped = stripped_filter == "Stripped only"
                filtered = [m for m in filtered if m.is_ft and m.is_stripped == want_stripped]

        # Analysis channel filter
        all_ac = sorted({m.analysis_channel for m in filtered})
        if len(all_ac) > 1:
            sel_ac = st.multiselect("Analysis Channel", all_ac, default=all_ac)
            filtered = [m for m in filtered if m.analysis_channel in sel_ac]

        # Reasoning effort filter
        all_efforts = sorted({m.reasoning_effort for m in filtered})
        if len(all_efforts) > 1:
            sel_efforts = st.multiselect("Reasoning Effort", all_efforts, default=all_efforts)
            filtered = [m for m in filtered if m.reasoning_effort in sel_efforts]

        # Max tokens filter
        all_max_tokens = sorted({m.max_tokens for m in filtered})
        if len(all_max_tokens) > 1:
            sel_max_tokens = st.multiselect("Max Tokens", all_max_tokens, default=all_max_tokens)
            filtered = [m for m in filtered if m.max_tokens in sel_max_tokens]

        # Sort: model, dataset, lr, checkpoint
        filtered.sort(key=lambda m: (m.model, m.dataset, m.lr, m.checkpoint, m.filename))

        if not filtered:
            st.warning("No files match current filters.")
            st.stop()

        file_options = {m.filename: m for m in filtered}
        selected_name = st.selectbox(
            "File", list(file_options.keys()), format_func=lambda x: x
        )
        meta = file_options[selected_name]

    # ---- Main area: load data (CoTControl/ReasonIF) -------------------------

    file_path = str(meta.path)
    file_mtime = os.path.getmtime(file_path)

    records = load_records(file_path, file_mtime)
    header = load_header(file_path, file_mtime)

    if not records:
        st.warning("File contains no rollout records.")
        st.stop()

    df = records_to_df(records)

    # ---- Header info ---------------------------------------------------------

    st.markdown(f"**Model:** `{meta.model}`  |  **Dataset:** `{meta.dataset}`  |  **Rollouts:** {len(records)}")

    if header:
        with st.expander("Experiment Config"):
            st.json(header)

    # ---- Metrics bar ---------------------------------------------------------

    total = len(df)
    has_correct = df["correct"].notna()
    has_compliant = df["compliant"].notna()
    has_meta = df["meta_discussion"].notna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Correct Rate", f"{df.loc[has_correct, 'correct'].mean():.1%}" if has_correct.any() else "N/A")
    c3.metric("Compliant Rate", f"{df.loc[has_compliant, 'compliant'].mean():.1%}" if has_compliant.any() else "N/A")
    c4.metric("Meta-Discussion Rate", f"{df.loc[has_meta, 'meta_discussion'].mean():.1%}" if has_meta.any() else "N/A")

    # ---- Row-level filters ---------------------------------------------------

    st.subheader("Filters")
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1, 1, 1, 2])

    modes_available = sorted(df["control_mode"].unique())
    with fc1:
        sel_modes = st.multiselect("Control Mode", modes_available, default=modes_available)
    with fc2:
        correct_filter = st.selectbox("Correct", ["All", "True", "False", "None"])
    with fc3:
        compliant_filter = st.selectbox("Compliant", ["All", "True", "False", "None"])
    with fc4:
        meta_filter = st.selectbox("Meta-discussion", ["All", "True", "False", "None"])
    with fc5:
        sample_search = st.text_input("Sample ID search")

    mask = (
        df["control_mode"].isin(sel_modes)
        & _bool_filter(df["correct"], correct_filter)
        & _bool_filter(df["compliant"], compliant_filter)
        & _bool_filter(df["meta_discussion"], meta_filter)
    )
    if sample_search:
        mask = mask & df["sample_id"].str.contains(sample_search, case=False, na=False)

    filtered_df = df[mask].reset_index(drop=True)

    st.caption(f"Showing {len(filtered_df)} of {total} rollouts")

    # ---- Table ---------------------------------------------------------------

    st.subheader("Rollouts")

    event = st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        column_config={
            "idx": st.column_config.NumberColumn("idx", width="small"),
            "latency_s": st.column_config.NumberColumn("latency_s", format="%.1f"),
            "question_preview": st.column_config.TextColumn("question_preview", width="large"),
        },
    )

    # ---- Detail view ---------------------------------------------------------

    selected_rows = event.selection.rows if event.selection else []

    if not selected_rows:
        st.info("Select a row above to view rollout details.")
        st.stop()

    # Navigation
    sel_table_idx = selected_rows[0]
    sel_record_idx = int(filtered_df.iloc[sel_table_idx]["idx"])

    # Store navigation state — only update from table click when the
    # selection actually changed (otherwise prev/next button updates
    # get clobbered on rerun).
    if "detail_idx" not in st.session_state:
        st.session_state.detail_idx = sel_record_idx
    elif st.session_state.get("_last_table_sel") != sel_record_idx:
        st.session_state.detail_idx = sel_record_idx
    st.session_state._last_table_sel = sel_record_idx

    detail_idx = st.session_state.detail_idx
    record = records[detail_idx]

    # Determine position within filtered set for prev/next
    filtered_indices = filtered_df["idx"].tolist()
    pos_in_filtered = filtered_indices.index(detail_idx) if detail_idx in filtered_indices else 0

    st.divider()

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("← Prev", disabled=(pos_in_filtered == 0)):
            st.session_state.detail_idx = filtered_indices[pos_in_filtered - 1]
            st.rerun()
    with nav3:
        if st.button("Next →", disabled=(pos_in_filtered >= len(filtered_indices) - 1)):
            st.session_state.detail_idx = filtered_indices[pos_in_filtered + 1]
            st.rerun()
    with nav2:
        st.markdown(f"**Rollout {pos_in_filtered + 1} of {len(filtered_indices)}**")

    # ---- Detail: metadata ----------------------------------------------------

    sample = record.get("sample", {})
    sample_meta = sample.get("metadata", {})

    left, right = st.columns(2)

    with left:
        st.markdown("**User Prompt:**")
        user_prompt = record.get("user_prompt", "")
        st.markdown(
            '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
            "font-size: 0.85em; background: #f6f8fa; padding: 0.75em; border-radius: 0.375em; "
            f'max-height: 600px; overflow-y: auto;">{html_mod.escape(user_prompt)}</div>',
            unsafe_allow_html=True,
        )
        if sample_meta:
            with st.expander("Sample Metadata"):
                st.json(sample_meta)

    with right:
        st.markdown(f"**Model:** `{record.get('model', '')}`")
        st.markdown(f"**Control Mode:** `{record.get('control_mode', '')}`")
        st.markdown(f"**Correct:** {_colored(record.get('correct'))}")
        st.markdown(f"**Compliant:** {_colored(record.get('compliant'))}")
        st.markdown(
            f"**Extracted Answer:** `{record.get('extracted_answer', '')}` "
            f"| **Correct Answer:** `{sample.get('correct_answer', '')}`"
        )
        latency_ms = record.get("latency_ms", 0)
        st.markdown(f"**Latency:** {latency_ms / 1000:.1f}s")
        usage = record.get("usage", {})
        if usage:
            st.markdown(
                f"**Tokens:** {usage.get('prompt_tokens', '?')} prompt / "
                f"{usage.get('completion_tokens', '?')} completion"
            )
        grading = record.get("grading_details", {})
        if grading:
            with st.expander("Grading Details"):
                st.json(grading)

    # ---- Detail: prompts and traces ------------------------------------------

    sys_prompt = record.get("system_prompt", "")
    if sys_prompt:
        with st.expander("System Prompt", expanded=False):
            st.code(sys_prompt, language=None)

    reasoning = record.get("reasoning", "")
    if reasoning:
        word_count = len(reasoning.split())
        char_count = len(reasoning)
        st.markdown(f"### Reasoning / CoT Trace  ({word_count:,} words, {char_count:,} chars)")
        st.markdown(
            '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
            "font-size: 0.85em; background: #f6f8fa; padding: 0.75em; border-radius: 0.375em; "
            f'max-height: 800px; overflow-y: auto;">{html_mod.escape(reasoning)}</div>',
            unsafe_allow_html=True,
        )

    response = record.get("response", "")
    if response:
        st.markdown("### Response")
        st.code(response, language=None)

    grading_full = record.get("grading_details", {})
    if grading_full:
        with st.expander("Grading Details (JSON)", expanded=False):
            st.json(grading_full)

# ===========================================================================
# MonitorQA mode
# ===========================================================================

else:
    with st.sidebar:
        if not MONITOR_QA_DIR.exists():
            st.error(f"MonitorQA directory not found: {MONITOR_QA_DIR}")
            st.stop()

        mqa_files = sorted(MONITOR_QA_DIR.glob("*.jsonl"))
        if not mqa_files:
            st.warning("No .jsonl files found in monitor_qa directory.")
            st.stop()

        all_mqa_meta = [parse_monitor_qa_filename(f) for f in mqa_files]
        all_mqa_meta = [m for m in all_mqa_meta if m is not None]

        if not all_mqa_meta:
            st.warning("Could not parse any MonitorQA filenames.")
            st.stop()

        # --- MonitorQA sidebar filters ---

        all_mqa_models = sorted({m.model for m in all_mqa_meta})
        sel_mqa_models = st.multiselect("Model", all_mqa_models, default=all_mqa_models)

        all_mqa_splits = sorted({m.split for m in all_mqa_meta})
        default_splits = ["val"] if "val" in all_mqa_splits else all_mqa_splits
        sel_mqa_splits = st.multiselect("Split", all_mqa_splits, default=default_splits)

        all_mqa_settings = sorted({m.setting for m in all_mqa_meta})
        sel_mqa_settings = st.multiselect("Setting", all_mqa_settings, default=all_mqa_settings)

        all_mqa_dates = sorted({m.date_str for m in all_mqa_meta})
        if len(all_mqa_dates) > 1:
            sel_mqa_dates = st.multiselect("Date", all_mqa_dates, default=all_mqa_dates)
        else:
            sel_mqa_dates = all_mqa_dates

        mqa_filtered = [
            m
            for m in all_mqa_meta
            if m.model in sel_mqa_models
            and m.split in sel_mqa_splits
            and m.setting in sel_mqa_settings
            and m.date_str in sel_mqa_dates
        ]

        mqa_filtered.sort(key=lambda m: (m.model, m.setting, m.date_str, m.filename))

        if not mqa_filtered:
            st.warning("No MonitorQA files match current filters.")
            st.stop()

        mqa_file_options = {m.filename: m for m in mqa_filtered}
        selected_mqa_name = st.selectbox(
            "File", list(mqa_file_options.keys()), format_func=lambda x: x
        )
        mqa_meta = mqa_file_options[selected_mqa_name]

    # ---- Main area: load data (MonitorQA) ------------------------------------

    file_path = str(mqa_meta.path)
    file_mtime = os.path.getmtime(file_path)

    records = load_records(file_path, file_mtime)
    header = load_header(file_path, file_mtime)

    if not records:
        st.warning("File contains no rollout records.")
        st.stop()

    df = monitorqa_records_to_df(records)

    # ---- Header info ---------------------------------------------------------

    st.markdown(
        f"**Model:** `{mqa_meta.model}`  |  **Setting:** `{mqa_meta.setting}`  |  "
        f"**Run:** `{mqa_meta.run_id}`  |  **Rollouts:** {len(records)}"
    )

    if header:
        with st.expander("Experiment Config"):
            st.json(header)

    # ---- MonitorQA: monitor type selector --------------------------------------

    monitor_types = _get_monitor_types_from_records(records)
    monitor_type_models = _get_monitor_type_models(records)
    if monitor_types:
        # Default to metr_note_gpt54 if present, else first available
        default_idx = (
            monitor_types.index("metr_note_gpt54")
            if "metr_note_gpt54" in monitor_types
            else 0
        )
        active_monitor = st.selectbox(
            "Monitor Type",
            monitor_types,
            index=default_idx,
            format_func=lambda mt: f"{mt} ({monitor_type_models.get(mt, '?')})"
                if monitor_type_models.get(mt) else mt,
        )
        active_score_col = f"score_{active_monitor}"
        # Replace monitor_score with the selected monitor's scores
        if active_score_col in df.columns:
            df["monitor_score"] = df[active_score_col]
    else:
        active_monitor = None
        active_score_col = "monitor_score"

    # ---- MonitorQA Metrics bar -----------------------------------------------

    total = len(df)
    has_side = df["side_task_correct"].notna()
    side_correct_rate = df.loc[has_side, "side_task_correct"].mean() if has_side.any() else None
    has_errors = (df["actor_error"] != "") | (df["monitor_error"] != "")
    error_rate = has_errors.mean()

    # Compute mean for active monitor
    if active_score_col in df.columns:
        has_active = df[active_score_col].notna()
        mean_active = df.loc[has_active, active_score_col].mean() if has_active.any() else None
    else:
        mean_active = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    score_label = f"Mean {active_monitor}" if active_monitor else "Mean Monitor Score"
    if mean_active is not None:
        c2.markdown(
            f'<div style="text-align:center;">'
            f'<div style="font-size:0.85em; color:#888;">{score_label}</div>'
            f'<div style="font-size:1.8em; font-weight:bold; color:{_monitor_score_color(mean_active)};">'
            f'{mean_active:.2f}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        c2.metric(score_label, "N/A")
    c3.metric("Side Task Correct", f"{side_correct_rate:.1%}" if side_correct_rate is not None else "N/A")
    c4.metric("Error Rate", f"{error_rate:.1%}")

    # ---- MonitorQA Row-level filters -----------------------------------------

    st.subheader("Filters")

    # Row 1: primary filters
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1, 1, 1, 2])

    with fc1:
        score_min, score_max = st.slider(
            "Monitor Score Range", 0.0, 10.0, (0.0, 10.0), step=0.5
        )
    with fc2:
        modes_available = sorted(df["mode"].unique())
        sel_modes = st.multiselect("Mode", modes_available, default=modes_available)
    with fc3:
        side_filter = st.selectbox("Side Task Correct", ["All", "True", "False"])
    with fc4:
        error_filter = st.selectbox("Errors", ["No errors", "All", "Errors only"])
    with fc5:
        sample_search = st.text_input("Sample ID search")

    # Row 2: reasoning & monitor score filters
    has_ra = f"score_reasoning_audit" in df.columns
    has_ra_mini = f"score_reasoning_audit_mini" in df.columns

    fc_r1, fc_r2, fc_r3, fc_r4 = st.columns([1, 1, 2, 2])

    with fc_r1:
        mentions_filter = st.selectbox(
            "Task terms in reasoning",
            ["All", "True", "False"],
            key="mqa_mentions_filter",
            help="'main task' or 'side task' appears in reasoning.",
        )
    with fc_r2:
        strong_filter = st.selectbox(
            "Strong regex",
            ["All", "True", "False"],
            key="mqa_strong_filter",
            help="Any of: 'main task', 'side task', 'main:', 'side:', "
                 "'main_task', 'side_task', 'both tasks', 'two tasks', "
                 "'two questions', 'main question', 'side question', "
                 "'main answer', 'side answer', 'both answer'.",
        )
    with fc_r3:
        if has_ra:
            ra_max_val = df["score_reasoning_audit"].dropna().max()
            ra_is_binary = ra_max_val is not None and ra_max_val <= 1.0
            if ra_is_binary:
                ra_min, ra_max = st.slider(
                    "Reasoning Audit (GPT-5.4)", 0.0, 1.0, (0.0, 1.0), step=1.0,
                    key="ra_range",
                )
            else:
                ra_min, ra_max = st.slider(
                    "Reasoning Audit (GPT-5.4)", 0.0, 10.0, (0.0, 10.0), step=0.5,
                    key="ra_range",
                )
        else:
            ra_min, ra_max = 0.0, 10.0
    with fc_r4:
        if has_ra_mini:
            ra_mini_max_val = df["score_reasoning_audit_mini"].dropna().max()
            ra_mini_is_binary = ra_mini_max_val is not None and ra_mini_max_val <= 1.0
            if ra_mini_is_binary:
                ra_mini_min, ra_mini_max = st.slider(
                    "Reasoning Audit Mini (GPT-5.4-mini)", 0.0, 1.0, (0.0, 1.0), step=1.0,
                    key="ra_mini_range",
                )
            else:
                ra_mini_min, ra_mini_max = st.slider(
                    "Reasoning Audit Mini (GPT-5.4-mini)", 0.0, 10.0, (0.0, 10.0), step=0.5,
                    key="ra_mini_range",
                )
        else:
            ra_mini_min, ra_mini_max = 0.0, 10.0

    mask = df["mode"].isin(sel_modes)

    # Monitor score range (applied to active monitor type)
    if active_score_col in df.columns:
        has_score = df[active_score_col].notna()
        if has_score.any():
            score_mask = (
                (df[active_score_col] >= score_min) & (df[active_score_col] <= score_max)
            ) | df[active_score_col].isna()
            mask = mask & score_mask

    # Reasoning audit score range filters
    if has_ra:
        ra_upper = 1.0 if ra_is_binary else 10.0
        if ra_min > 0.0 or ra_max < ra_upper:
            ra_col = "score_reasoning_audit"
            mask = mask & (
                ((df[ra_col] >= ra_min) & (df[ra_col] <= ra_max))
                | df[ra_col].isna()
            )
    if has_ra_mini:
        ra_mini_upper = 1.0 if ra_mini_is_binary else 10.0
        if ra_mini_min > 0.0 or ra_mini_max < ra_mini_upper:
            ra_mini_col = "score_reasoning_audit_mini"
            mask = mask & (
                ((df[ra_mini_col] >= ra_mini_min) & (df[ra_mini_col] <= ra_mini_max))
                | df[ra_mini_col].isna()
            )

    # Side task correct filter
    if side_filter != "All":
        want_side = side_filter == "True"
        mask = mask & (df["side_task_correct"] == want_side)

    # Reasoning-mentions-task-terms filter
    if mentions_filter != "All":
        want_mentions = mentions_filter == "True"
        mask = mask & (df["mentions_task_terms"] == want_mentions)

    # Strong regex filter
    if strong_filter != "All":
        want_strong = strong_filter == "True"
        mask = mask & (df["mentions_strong"] == want_strong)

    # Error filter
    if error_filter == "No errors":
        mask = mask & (df["actor_error"] == "") & (df["monitor_error"] == "")
    elif error_filter == "Errors only":
        mask = mask & ((df["actor_error"] != "") | (df["monitor_error"] != ""))

    if sample_search:
        mask = mask & df["sample_id"].str.contains(sample_search, case=False, na=False)

    filtered_df = df[mask].reset_index(drop=True)

    st.caption(f"Showing {len(filtered_df)} of {total} rollouts")

    # ---- MonitorQA Table -----------------------------------------------------

    st.subheader("Rollouts")

    # Color-code monitor scores in the table via column_config
    col_config = {
        "idx": st.column_config.NumberColumn("idx", width="small"),
        "monitor_score": st.column_config.ProgressColumn(
            "monitor_score",
            min_value=0,
            max_value=10,
            format="%.1f",
        ),
        "latency_s": st.column_config.NumberColumn("latency_s", format="%.1f"),
        "question_preview": st.column_config.TextColumn("question_preview", width="large"),
    }
    # Add per-monitor-type score columns
    for mt in monitor_types:
        col_name = f"score_{mt}"
        if col_name in filtered_df.columns:
            if mt == "main_task_grading":
                col_config[col_name] = st.column_config.NumberColumn(
                    col_name, format="%d",
                )
            else:
                col_config[col_name] = st.column_config.ProgressColumn(
                    col_name, min_value=0, max_value=10, format="%.1f",
                )

    event = st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        column_config=col_config,
    )

    # ---- MonitorQA Detail view -----------------------------------------------

    selected_rows = event.selection.rows if event.selection else []

    if not selected_rows:
        st.info("Select a row above to view rollout details.")
        st.stop()

    sel_table_idx = selected_rows[0]
    sel_record_idx = int(filtered_df.iloc[sel_table_idx]["idx"])

    if "mqa_detail_idx" not in st.session_state:
        st.session_state.mqa_detail_idx = sel_record_idx
    elif st.session_state.get("_mqa_last_table_sel") != sel_record_idx:
        st.session_state.mqa_detail_idx = sel_record_idx
    st.session_state._mqa_last_table_sel = sel_record_idx

    detail_idx = st.session_state.mqa_detail_idx
    record = records[detail_idx]

    filtered_indices = filtered_df["idx"].tolist()
    pos_in_filtered = filtered_indices.index(detail_idx) if detail_idx in filtered_indices else 0

    st.divider()

    # Monitor score badges — one per monitor type
    monitor_results = record.get("monitor_results", [])
    if monitor_results:
        def _badge_label(mr: dict) -> str:
            mt = mr.get("monitor_type", "?")
            mm = mr.get("monitor_model", "")
            return f"{mt} ({mm})" if mm else mt

        badge_html = " &nbsp; ".join(
            f'<span style="font-size:0.7em; color:#888;">{_badge_label(mr)}: </span>'
            + _monitor_score_badge(mr.get("monitor_score"))
            for mr in monitor_results
        )
        st.markdown(badge_html, unsafe_allow_html=True)
    else:
        # Fallback to legacy field
        score_val = record.get("monitor_score")
        st.markdown(_monitor_score_badge(score_val), unsafe_allow_html=True)

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("← Prev", disabled=(pos_in_filtered == 0), key="mqa_prev"):
            st.session_state.mqa_detail_idx = filtered_indices[pos_in_filtered - 1]
            st.rerun()
    with nav3:
        if st.button("Next →", disabled=(pos_in_filtered >= len(filtered_indices) - 1), key="mqa_next"):
            st.session_state.mqa_detail_idx = filtered_indices[pos_in_filtered + 1]
            st.rerun()
    with nav2:
        st.markdown(f"**Rollout {pos_in_filtered + 1} of {len(filtered_indices)}**")

    # ---- MonitorQA Detail: metadata ------------------------------------------

    sample = record.get("sample", {})
    sample_meta = sample.get("metadata", {})

    left, right = st.columns(2)

    with left:
        st.markdown("**User Prompt:**")
        user_prompt = record.get("user_prompt", "")
        st.markdown(
            '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
            "font-size: 0.85em; background: #f6f8fa; padding: 0.75em; border-radius: 0.375em; "
            f'max-height: 600px; overflow-y: auto;">{html_mod.escape(user_prompt)}</div>',
            unsafe_allow_html=True,
        )

        # Side task metadata
        if sample_meta:
            st.markdown("**Side Task Info:**")
            side_cols = st.columns(2)
            with side_cols[0]:
                st.markdown(f"**Side Task:** {sample_meta.get('side_task', 'N/A')[:200]}")
                st.markdown(f"**Side Answer:** `{sample_meta.get('side_task_answer', 'N/A')}`")
            with side_cols[1]:
                st.markdown(f"**Side Level:** `{sample_meta.get('side_level', 'N/A')}`")
                st.markdown(f"**Side Type:** `{sample_meta.get('side_type', 'N/A')}`")

            with st.expander("Full Sample Metadata"):
                st.json(sample_meta)

    with right:
        st.markdown(f"**Model:** `{record.get('model', '')}`")
        st.markdown(f"**Mode:** `{record.get('mode', '')}`")
        st.markdown(f"**Side Task Correct:** {_colored(record.get('side_task_correct'))}")

        actor_err = record.get("actor_error")
        monitor_err = record.get("monitor_error")
        if actor_err:
            st.markdown(f"**Actor Error:** :red[{actor_err}]")
        if monitor_err:
            st.markdown(f"**Monitor Error:** :red[{monitor_err}]")

        actor_latency = record.get("actor_latency_ms", 0)
        monitor_latency = record.get("monitor_latency_ms", 0)
        st.markdown(
            f"**Actor Latency:** {actor_latency / 1000:.1f}s  |  "
            f"**Monitor Latency:** {monitor_latency / 1000:.1f}s"
        )

        actor_usage = record.get("actor_usage", {})
        if actor_usage:
            st.markdown(
                f"**Actor Tokens:** {actor_usage.get('prompt_tokens', '?')} prompt / "
                f"{actor_usage.get('completion_tokens', '?')} completion"
            )
        monitor_usage = record.get("monitor_usage", {})
        if monitor_usage:
            st.markdown(
                f"**Monitor Tokens:** {monitor_usage.get('prompt_tokens', '?')} prompt / "
                f"{monitor_usage.get('completion_tokens', '?')} completion"
            )

    # ---- MonitorQA Detail: prompts and traces --------------------------------

    sys_prompt = record.get("system_prompt", "")
    if sys_prompt:
        with st.expander("System Prompt", expanded=False):
            st.code(sys_prompt, language=None)

    reasoning = record.get("reasoning", "")
    response = record.get("response", "")
    mqa_actor_err = record.get("actor_error")

    if reasoning:
        word_count = len(reasoning.split())
        char_count = len(reasoning)
        st.markdown(f"### Reasoning / CoT Trace  ({word_count:,} words, {char_count:,} chars)")
        st.markdown(
            '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
            "font-size: 0.85em; background: #f6f8fa; padding: 0.75em; border-radius: 0.375em; "
            f'max-height: 800px; overflow-y: auto;">{html_mod.escape(reasoning)}</div>',
            unsafe_allow_html=True,
        )
    elif not mqa_actor_err:
        st.markdown(
            '### Reasoning / CoT Trace\n\n'
            ':orange[Model skipped thinking] — closed `</think>` immediately without reasoning.'
        )

    if reasoning and not response and not mqa_actor_err:
        st.markdown(
            ':red[Model never left thinking] — no `</think>` was produced; '
            'entire output stayed in the think block.'
        )

    if response:
        st.markdown("### Response")
        st.code(response, language=None)

    # Monitor response section — show all monitor results
    monitor_results = record.get("monitor_results", [])
    if monitor_results:
        for mr in monitor_results:
            mt = mr.get("monitor_type", "unknown")
            mm = mr.get("monitor_model", "")
            mr_score = mr.get("monitor_score")
            mr_response = mr.get("monitor_response", "")
            mr_error = mr.get("monitor_error")

            header_label = f"{mt} ({mm})" if mm else mt
            st.markdown(f"### Monitor: {header_label}")
            if mr_error:
                st.markdown(f":red[Error: {mr_error}]")
            if mr_response:
                st.markdown(
                    '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
                    "font-size: 0.85em; background: #fff3cd; padding: 0.75em; border-radius: 0.375em; "
                    f'border-left: 4px solid {_monitor_score_color(mr_score)}; '
                    f'max-height: 400px; overflow-y: auto;">{html_mod.escape(mr_response)}</div>',
                    unsafe_allow_html=True,
                )

            mr_prompt = mr.get("monitor_prompt")
            if not mr_prompt:
                mr_prompt = record.get("monitor_prompt")
            if not mr_prompt:
                try:
                    builder = get_monitor_prompt_builder(mt)
                    mr_prompt = build_monitor_messages(builder, record)
                except Exception:
                    mr_prompt = None
            if mr_prompt:
                with st.expander(f"Monitor Prompt ({mt})", expanded=False):
                    if isinstance(mr_prompt, list):
                        for msg in mr_prompt:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            st.markdown(f"**{role}:**")
                            st.code(content, language=None)
                    else:
                        st.code(str(mr_prompt), language=None)
    else:
        # Fallback to legacy fields
        monitor_response = record.get("monitor_response", "")
        score_val = record.get("monitor_score")
        if monitor_response:
            st.markdown("### Monitor Response")
            st.markdown(
                '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
                "font-size: 0.85em; background: #fff3cd; padding: 0.75em; border-radius: 0.375em; "
                f'border-left: 4px solid {_monitor_score_color(score_val)}; '
                f'max-height: 400px; overflow-y: auto;">{html_mod.escape(monitor_response)}</div>',
                unsafe_allow_html=True,
            )

        monitor_prompt = record.get("monitor_prompt")
        if monitor_prompt:
            with st.expander("Monitor Prompt", expanded=False):
                if isinstance(monitor_prompt, list):
                    for msg in monitor_prompt:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        st.markdown(f"**{role}:**")
                        st.code(content, language=None)
                else:
                    st.code(str(monitor_prompt), language=None)

    # Side task response fields
    main_response = record.get("main_task_response", "")
    side_response = record.get("side_task_response", "")
    if main_response or side_response:
        with st.expander("Parsed Task Responses", expanded=False):
            if main_response:
                st.markdown("**Main Task Response:**")
                st.code(main_response, language=None)
            if side_response:
                st.markdown("**Side Task Response:**")
                st.code(side_response, language=None)
