"""Rollout Viewer — Streamlit app for browsing rollout JSONL data.

Launch:  uv run streamlit run scripts/analysis/rollout_viewer.py
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLOUTS_DIR = Path(__file__).resolve().parents[2] / "results" / "rollouts"

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Regex splits on _cotcontrol_ or _reasonif_ to separate model part from rest.
_DATASET_RE = re.compile(r"_(cotcontrol|reasonif)_")

# Fine-tune pattern:  -rif  optionally followed by  -lr<rate>  and  -<checkpoint>
_FT_RE = re.compile(r"-rif(?:-lr([\d.]+e-\d+))?(?:-(\d{3,}|final))?$")


@dataclass
class FileMetadata:
    path: Path
    filename: str
    model: str        # e.g. "gpt-oss-120b" or "qwen3-8b"
    dataset: str      # "cotcontrol" or "reasonif"
    is_ft: bool
    lr: str           # "" if not fine-tuned
    checkpoint: str   # "" if not fine-tuned
    provider: str     # e.g. "openai", "qwen"


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
        # Old-format filenames (-rif-000060) omit the lr; these were all 1e-4
        lr = ft.group(1) or ("1e-4" if ft.group(2) else "")
        checkpoint = ft.group(2) or ""
        is_ft = True
    else:
        base_model = model_full
        lr = ""
        checkpoint = ""
        is_ft = False

    return FileMetadata(
        path=path,
        filename=path.name,
        model=base_model,
        dataset=dataset,
        is_ft=is_ft,
        lr=lr,
        checkpoint=checkpoint,
        provider=provider,
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

# ---- Main area: load data ---------------------------------------------------

file_path = str(meta.path)
file_mtime = os.path.getmtime(file_path)

records = load_records(file_path, file_mtime)
header = load_header(file_path, file_mtime)

if not records:
    st.warning("File contains no rollout records.")
    st.stop()

df = records_to_df(records)

# ---- Header info -------------------------------------------------------------

st.markdown(f"**Model:** `{meta.model}`  |  **Dataset:** `{meta.dataset}`  |  **Rollouts:** {len(records)}")

if header:
    with st.expander("Experiment Config"):
        st.json(header)

# ---- Metrics bar -------------------------------------------------------------

total = len(df)
has_correct = df["correct"].notna()
has_compliant = df["compliant"].notna()
has_meta = df["meta_discussion"].notna()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", total)
c2.metric("Correct Rate", f"{df.loc[has_correct, 'correct'].mean():.1%}" if has_correct.any() else "N/A")
c3.metric("Compliant Rate", f"{df.loc[has_compliant, 'compliant'].mean():.1%}" if has_compliant.any() else "N/A")
c4.metric("Meta-Discussion Rate", f"{df.loc[has_meta, 'meta_discussion'].mean():.1%}" if has_meta.any() else "N/A")

# ---- Row-level filters -------------------------------------------------------

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

# ---- Table -------------------------------------------------------------------

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

# ---- Detail view -------------------------------------------------------------

selected_rows = event.selection.rows if event.selection else []

if not selected_rows:
    st.info("Select a row above to view rollout details.")
    st.stop()

# Navigation
sel_table_idx = selected_rows[0]
sel_record_idx = int(filtered_df.iloc[sel_table_idx]["idx"])

# Store navigation state
if "detail_idx" not in st.session_state:
    st.session_state.detail_idx = sel_record_idx
else:
    # If user clicked a new row, update
    st.session_state.detail_idx = sel_record_idx

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

# ---- Detail: metadata --------------------------------------------------------

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

# ---- Detail: prompts and traces ----------------------------------------------

sys_prompt = record.get("system_prompt", "")
if sys_prompt:
    with st.expander("System Prompt", expanded=False):
        st.code(sys_prompt, language=None)

reasoning = record.get("reasoning", "")
if reasoning:
    word_count = len(reasoning.split())
    char_count = len(reasoning)
    st.markdown(f"### Reasoning / CoT Trace  ({word_count:,} words, {char_count:,} chars)")
    st.code(reasoning, language=None)

response = record.get("response", "")
if response:
    st.markdown("### Response")
    st.code(response, language=None)

grading_full = record.get("grading_details", {})
if grading_full:
    with st.expander("Grading Details (JSON)", expanded=False):
        st.json(grading_full)
