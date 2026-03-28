"""SFT Dataset Viewer — Streamlit app for browsing SFT parquet data.

Launch:  uv run streamlit run scripts/analysis/sft_viewer.py
"""

from __future__ import annotations

import html as html_mod
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SFT_DIR = Path(__file__).resolve().parents[2] / "results" / "sft"

CNAME_TO_SHORT = {
    "punctuation:no_comma": "no_comma",
    "length_constraint_checkers:number_words": "number_words",
    "detectable_format:json_format": "json_format",
    "change_case:english_capital": "english_capital",
    "change_style:english_capital": "english_capital",
    "startend:end_checker": "end_checker",
    "language:reasoning_language": "reasoning_language",
}

# Model name from filename: e.g. "gpt-oss-20b-reasonif-sft.parquet" -> "GPT-OSS-20B"
# Also handles suffixes like "-stripped", "-stripped-ac"
_MODEL_RE = re.compile(r"^(.+?)-reasonif-sft(?:-[a-z]+)*\.parquet$")

# Variant parsing: extract (stripped, ac) from filename suffix
_VARIANT_RE = re.compile(r"-reasonif-sft((?:-[a-z]+)*)\.parquet$")


def _parse_variant(filename: str) -> tuple[bool, bool]:
    """Parse suffix to determine (is_stripped, is_ac)."""
    m = _VARIANT_RE.search(filename)
    if not m:
        return False, False
    suffix = m.group(1)  # e.g. "", "-stripped", "-stripped-ac"
    return "-stripped" in suffix, "-ac" in suffix


def _variant_label(stripped: bool, ac: bool) -> str:
    if stripped and ac:
        return "stripped + ac"
    elif stripped:
        return "stripped"
    elif ac:
        return "ac"
    return "original"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_val(x):
    """Unwrap numpy arrays (parquet stores some fields as 1-element arrays)."""
    if isinstance(x, np.ndarray):
        return x[0] if len(x) > 0 else None
    return x


def _model_from_filename(name: str) -> str | None:
    m = _MODEL_RE.match(name)
    if m:
        return m.group(1).upper()
    return None


def _extract_parts(messages):
    """Extract prompt, reasoning, and response from messages array."""
    prompt = ""
    reasoning = ""
    response = ""
    if isinstance(messages, np.ndarray):
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            if role == "user":
                prompt = m.get("content", "")
            elif role == "assistant":
                reasoning = m.get("thinking", "") or ""
                response = m.get("content", "")
    return prompt, reasoning, response


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def discover_files() -> list[tuple[str, str, Path, bool, bool]]:
    """Return list of (model_label, filename, path, is_stripped, is_ac) for all SFT parquets."""
    if not SFT_DIR.exists():
        return []
    results = []
    for p in sorted(SFT_DIR.glob("*.parquet")):
        model = _model_from_filename(p.name)
        if model:
            stripped, ac = _parse_variant(p.name)
            results.append((model, p.name, p, stripped, ac))
    return results


@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SFT Dataset Viewer", layout="wide")
st.title("SFT Dataset Viewer")

# ---- Sidebar: file & filter selection ------------------------------------

with st.sidebar:
    st.header("Dataset Selection")

    all_files = discover_files()
    if not all_files:
        st.error(f"No SFT parquet files found in {SFT_DIR}")
        st.stop()

    # Model selector
    models = sorted({f[0] for f in all_files})
    sel_model = st.selectbox("Model", models)

    # Files matching model
    matching = [f for f in all_files if f[0] == sel_model]

    # Variant filters (only show options that exist for this model)
    available_stripped = sorted({f[3] for f in matching})
    available_ac = sorted({f[4] for f in matching})

    # Stripped filter
    if len(available_stripped) > 1:
        stripped_options = {True: "Stripped", False: "Original"}
        sel_stripped = st.radio(
            "Prompt variant",
            available_stripped,
            format_func=lambda x: stripped_options[x],
            horizontal=True,
        )
        matching = [f for f in matching if f[3] == sel_stripped]
    elif len(available_stripped) == 1:
        lbl = "Stripped" if available_stripped[0] else "Original"
        st.caption(f"Prompt variant: **{lbl}** (only option)")

    # AC filter
    if len(available_ac) > 1:
        ac_options = {True: "Analysis channel", False: "Reasoning"}
        sel_ac = st.radio(
            "Terminology",
            available_ac,
            format_func=lambda x: ac_options[x],
            horizontal=True,
        )
        matching = [f for f in matching if f[4] == sel_ac]
    elif len(available_ac) == 1:
        lbl = "Analysis channel" if available_ac[0] else "Reasoning"
        st.caption(f"Terminology: **{lbl}** (only option)")

    # Final file selection (should be 1, but fallback to selectbox)
    if len(matching) == 1:
        sel_file = matching[0]
    else:
        file_names = [name for _, name, _, _, _ in matching]
        sel_name = st.selectbox("File", file_names)
        sel_file = next(f for f in matching if f[1] == sel_name)

    st.caption(f"`{sel_file[1]}`")

# ---- Load data -----------------------------------------------------------

df = load_parquet(str(sel_file[2]))

# Parse instruction types
df["instruction_type"] = df["constraint_name"].apply(
    lambda x: CNAME_TO_SHORT.get(_get_val(x), _get_val(x))
)

# ---- Sidebar: instruction filter -----------------------------------------

with st.sidebar:
    st.header("Filters")
    all_types = sorted(df["instruction_type"].unique())
    sel_types = st.multiselect("Instruction Type", all_types, default=all_types)

mask = df["instruction_type"].isin(sel_types)
filtered = df[mask].reset_index(drop=True)

# ---- Metrics -------------------------------------------------------------

st.markdown(f"**Model:** `{sel_model}`  |  **Total rows:** {len(df)}  |  **Showing:** {len(filtered)}")

type_counts = filtered["instruction_type"].value_counts().sort_index()
cols = st.columns(min(len(type_counts), 6))
for i, (itype, count) in enumerate(type_counts.items()):
    cols[i % len(cols)].metric(itype, count)

# ---- Navigation ----------------------------------------------------------

if len(filtered) == 0:
    st.warning("No rows match current filters.")
    st.stop()

# Session state for current index
if "sft_idx" not in st.session_state:
    st.session_state.sft_idx = 0

# Clamp index
if st.session_state.sft_idx >= len(filtered):
    st.session_state.sft_idx = 0

st.divider()

nav1, nav2, nav3 = st.columns([1, 3, 1])
idx = st.session_state.sft_idx

with nav1:
    if st.button("← Prev (Left)", disabled=(idx == 0)):
        st.session_state.sft_idx = idx - 1
        st.rerun()
with nav3:
    if st.button("Next (Right) →", disabled=(idx >= len(filtered) - 1)):
        st.session_state.sft_idx = idx + 1
        st.rerun()
with nav2:
    st.markdown(f"**Example {idx + 1} of {len(filtered)}**")

# Jump to specific index
jump = st.number_input(
    "Jump to example #", min_value=1, max_value=len(filtered), value=idx + 1, step=1
)
if jump - 1 != idx:
    st.session_state.sft_idx = jump - 1
    st.rerun()

# ---- Display current example ---------------------------------------------

row = filtered.iloc[idx]
prompt, reasoning, response = _extract_parts(row["messages"])
itype = row["instruction_type"]
cargs = _get_val(row.get("constraint_args"))
if isinstance(cargs, dict):
    cargs = {k: v for k, v in cargs.items() if v is not None}
instruction_desc = _get_val(row.get("instruction_description", ""))

# Metadata bar
meta_cols = st.columns(3)
meta_cols[0].markdown(f"**Instruction Type:** `{itype}`")
if cargs:
    meta_cols[1].markdown(f"**Constraint Args:** `{cargs}`")
if instruction_desc:
    meta_cols[2].markdown(f"**Description:** {instruction_desc[:100]}")

# Prompt
st.markdown("### Prompt")
st.markdown(
    '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
    "font-size: 0.85em; background: #f6f8fa; padding: 0.75em; border-radius: 0.375em; "
    f'max-height: 400px; overflow-y: auto;">{html_mod.escape(prompt)}</div>',
    unsafe_allow_html=True,
)

# Generation prompt (the actual prompt sent to the model, may differ from final prompt)
if "generation_prompt" in row.index:
    gen_prompt = _get_val(row.get("generation_prompt", ""))
    if gen_prompt and gen_prompt != prompt:
        with st.expander("Generation Prompt (sent to model)"):
            st.markdown(
                '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
                "font-size: 0.85em; background: #e3f2fd; padding: 0.75em; border-radius: 0.375em; "
                f'max-height: 400px; overflow-y: auto;">{html_mod.escape(gen_prompt)}</div>',
                unsafe_allow_html=True,
            )

# Reasoning
if reasoning:
    word_count = len(re.findall(r"\w+", reasoning))
    char_count = len(reasoning)
    st.markdown(f"### Reasoning ({word_count:,} words, {char_count:,} chars)")
    st.markdown(
        '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
        "font-size: 0.85em; background: #e8f5e9; padding: 0.75em; border-radius: 0.375em; "
        f'max-height: 600px; overflow-y: auto;">{html_mod.escape(reasoning)}</div>',
        unsafe_allow_html=True,
    )

# Response
if response:
    st.markdown("### Response")
    st.markdown(
        '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
        "font-size: 0.85em; background: #fff3e0; padding: 0.75em; border-radius: 0.375em; "
        f'max-height: 400px; overflow-y: auto;">{html_mod.escape(response)}</div>',
        unsafe_allow_html=True,
    )

# Original (pre-transform) comparison
if "messages_original" in row.index:
    orig = row["messages_original"]
    orig_prompt, orig_reasoning, orig_response = _extract_parts(orig)
    if orig_reasoning and orig_reasoning != reasoning:
        with st.expander("Original Reasoning (pre-transform)"):
            orig_wc = len(orig_reasoning.split())
            st.caption(f"{orig_wc:,} words, {len(orig_reasoning):,} chars")
            st.code(orig_reasoning, language=None)

# Full metadata
with st.expander("Full Row Data"):
    display = {}
    for col in row.index:
        val = row[col]
        if isinstance(val, np.ndarray):
            display[col] = [_get_val(v) if isinstance(v, np.ndarray) else v for v in val]
        else:
            display[col] = _get_val(val)
    # Remove large fields for readability
    for key in ["messages", "messages_original"]:
        if key in display:
            display[key] = f"<{key}: {len(row[key])} items>"
    st.json(display, expanded=False)

# ---- Keyboard navigation via query params --------------------------------
# Streamlit doesn't natively support keyboard events, but we can use
# a small JS snippet for left/right arrow keys.

st.markdown(
    """
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === 'ArrowLeft') {
            const btn = document.querySelectorAll('button');
            for (const b of btn) {
                if (b.textContent.includes('Prev')) { b.click(); break; }
            }
        } else if (e.key === 'ArrowRight') {
            const btn = document.querySelectorAll('button');
            for (const b of btn) {
                if (b.textContent.includes('Next')) { b.click(); break; }
            }
        }
    });
    </script>
    """,
    unsafe_allow_html=True,
)
