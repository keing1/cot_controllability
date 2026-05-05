"""SFT Dataset Viewer — Streamlit app for browsing SFT parquet data.

Launch:  uv run streamlit run scripts/analysis/sft_viewer.py
"""

from __future__ import annotations

import difflib
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
    # ReasonIF canonical names (legacy "category:short")
    "punctuation:no_comma": "no_comma",
    "length_constraint_checkers:number_words": "number_words",
    "detectable_format:json_format": "json_format",
    "change_case:english_capital": "english_capital",
    "change_style:english_capital": "english_capital",
    "startend:end_checker": "end_checker",
    "language:reasoning_language": "reasoning_language",
    # CoTControl names emitted by the mixed-source builder
    "cotcontrol:word_suppression": "word_suppression",
    "cotcontrol:multiple_word_suppression": "multiple_word_suppression",
    "cotcontrol:uppercase_thinking": "uppercase_thinking",
    "cotcontrol:lowercase_thinking": "lowercase_thinking",
    "cotcontrol:alternating_case": "alternating_case",
    "cotcontrol:end_of_sentence": "end_of_sentence",
    "cotcontrol:meow_between_words": "meow_between_words",
    "cotcontrol:repeat_sentences": "repeat_sentences",
    "cotcontrol:ignore_question": "ignore_question",
}

# Model name from filename:
#   v1-v3 (with -sft):   "gpt-oss-20b-reasonif-sft.parquet"
#                        "gpt-oss-20b-mixed5k-sft-stripped-ac.parquet"
#   v4+   (no -sft):     "gpt-oss-20b-mixed5k-v4-stripped-ac.parquet"
#   smoke runs:          "gpt-oss-20b-smoke-with-instr.parquet"
#                        "gpt-oss-20b-smoke-without-instr.parquet"
# Composition can carry a -v<N> suffix (e.g. "mixed5k-v4"). The "-sft"
# infix is optional. Trailing variant tags ("-stripped", "-ac",
# "-with-instr", "-without-instr") are matched by the variant suffix.
_COMPOSITION_RE = r"(?:reasonif|cotcontrol|mixed[0-9a-z]*(?:-v\d+)?|smoke)"
# Suffix tokens are alpha-only or alpha-with-internal-hyphen (e.g.
# "with-instr", "without-instr"). Match one-or-more dash-separated tokens.
_SUFFIX_RE = r"(?:-[a-z]+(?:-[a-z]+)*)*"
_MODEL_RE = re.compile(rf"^(.+?)-{_COMPOSITION_RE}(?:-sft)?{_SUFFIX_RE}\.parquet$")
_VARIANT_RE = re.compile(rf"-{_COMPOSITION_RE}(?:-sft)?({_SUFFIX_RE})\.parquet$")
_COMPOSITION_EXTRACT_RE = re.compile(rf"-({_COMPOSITION_RE})(?:-sft)?{_SUFFIX_RE}\.parquet$")


def _parse_composition(filename: str) -> str:
    m = _COMPOSITION_EXTRACT_RE.search(filename)
    return m.group(1) if m else "unknown"


def _parse_variant(filename: str) -> tuple[bool, bool, str]:
    """Parse suffix to determine (is_stripped, is_ac, generation_mode).

    ``generation_mode`` is one of:
      "with-instr"     — Stage-1 prompt included the constraint instruction
      "without-instr"  — Stage-1 prompt stripped of the constraint
      "stripped"       — legacy synonym for without-instr (older runs)
      ""               — unspecified
    """
    m = _VARIANT_RE.search(filename)
    if not m:
        return False, False, ""
    suffix = m.group(1)
    is_with = "-with-instr" in suffix
    is_without = "-without-instr" in suffix
    is_stripped = "-stripped" in suffix
    is_ac = "-ac" in suffix
    if is_with:
        gen = "with-instr"
    elif is_without:
        gen = "without-instr"
    elif is_stripped:
        gen = "stripped"
    else:
        gen = ""
    return is_stripped, is_ac, gen


def _variant_label(stripped: bool, ac: bool, gen: str = "") -> str:
    pieces: list[str] = []
    if gen == "with-instr":
        pieces.append("with-instr")
    elif gen == "without-instr":
        pieces.append("without-instr")
    elif stripped:
        pieces.append("stripped")
    if ac:
        pieces.append("ac")
    if not pieces:
        return "original"
    return " + ".join(pieces)


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
def discover_files() -> list[tuple[str, str, Path, bool, bool, str, str]]:
    """Return (model_label, display_name, path, is_stripped, is_ac,
    composition, generation_mode) for every SFT parquet under ``results/sft``.

    Searches recursively so files in subdirectories (e.g.
    ``results/sft/smoke_with_without/``) are picked up too. ``display_name``
    includes the relative subdir for disambiguation.
    """
    if not SFT_DIR.exists():
        return []
    results = []
    for p in sorted(SFT_DIR.rglob("*.parquet")):
        model = _model_from_filename(p.name)
        if not model:
            continue
        stripped, ac, gen = _parse_variant(p.name)
        comp = _parse_composition(p.name)
        rel = p.relative_to(SFT_DIR)
        # Use the relative path for files in subdirectories so the user can
        # tell smoke runs apart from production runs in the picker.
        display = str(rel) if rel.parent != Path(".") else p.name
        results.append((model, display, p, stripped, ac, comp, gen))
    return results


@st.cache_data
def load_build_metadata(path: str) -> dict | None:
    """Load the .meta.json sidecar next to a parquet, if present."""
    sidecar = Path(path + ".meta.json")
    if not sidecar.exists():
        return None
    try:
        with open(sidecar) as f:
            return json.load(f)
    except Exception:
        return None


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

    # Composition filter (reasonif / cotcontrol / mixed)
    available_comps = sorted({f[5] for f in matching})
    if len(available_comps) > 1:
        sel_comp = st.radio(
            "Composition", available_comps, horizontal=True,
        )
        matching = [f for f in matching if f[5] == sel_comp]
    elif len(available_comps) == 1:
        st.caption(f"Composition: **{available_comps[0]}**")

    # Variant filters (only show options that exist for this model)
    available_stripped = sorted({f[3] for f in matching})
    available_ac = sorted({f[4] for f in matching})
    available_gen = sorted({f[6] for f in matching})

    # Generation-mode filter (with-instr vs without-instr) — surfaces the new
    # smoke-run variant axis. Falls back gracefully when not applicable.
    meaningful_gen = [g for g in available_gen if g]
    if len(meaningful_gen) > 1:
        gen_options = {
            "with-instr": "With instruction",
            "without-instr": "Without instruction",
            "stripped": "Stripped (legacy)",
            "": "(unspecified)",
        }
        sel_gen = st.radio(
            "Generation mode",
            available_gen,
            format_func=lambda x: gen_options.get(x, x or "(unspecified)"),
            horizontal=True,
        )
        matching = [f for f in matching if f[6] == sel_gen]
    elif len(meaningful_gen) == 1:
        lbl = {
            "with-instr": "With instruction",
            "without-instr": "Without instruction",
            "stripped": "Stripped (legacy)",
        }.get(meaningful_gen[0], meaningful_gen[0])
        st.caption(f"Generation mode: **{lbl}** (only option)")

    # Stripped filter (only meaningful when generation mode isn't already set)
    if not meaningful_gen and len(available_stripped) > 1:
        stripped_options = {True: "Stripped", False: "Original"}
        sel_stripped = st.radio(
            "Prompt variant",
            available_stripped,
            format_func=lambda x: stripped_options[x],
            horizontal=True,
        )
        matching = [f for f in matching if f[3] == sel_stripped]
    elif not meaningful_gen and len(available_stripped) == 1:
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
        file_names = [f[1] for f in matching]
        sel_name = st.selectbox("File", file_names)
        sel_file = next(f for f in matching if f[1] == sel_name)

    st.caption(f"`{sel_file[1]}`")

    # Build metadata sidecar (if present)
    build_meta = load_build_metadata(str(sel_file[2]))
    if build_meta is not None:
        with st.expander("Build config (from sidecar)"):
            st.json(build_meta, expanded=False)

# ---- Load data -----------------------------------------------------------

df = load_parquet(str(sel_file[2]))

# Parse instruction types
df["instruction_type"] = df["constraint_name"].apply(
    lambda x: CNAME_TO_SHORT.get(_get_val(x), _get_val(x))
)


def _short_source(constraint: str) -> str:
    """Map a constraint short-name to its source family (cotcontrol/reasonif)."""
    if constraint in {
        "word_suppression", "multiple_word_suppression", "uppercase_thinking",
        "lowercase_thinking", "alternating_case", "end_of_sentence",
        "meow_between_words", "repeat_sentences", "ignore_question",
    }:
        return "cotcontrol"
    return "reasonif"


df["source_short"] = df["instruction_type"].apply(_short_source)

# ---- Sidebar: instruction filter -----------------------------------------

with st.sidebar:
    st.header("Filters")

    # Source family filter (cotcontrol vs reasonif vs all)
    source_choice = st.radio(
        "Source", ["All", "CoTControl", "ReasonIF"], horizontal=True,
    )

    # Compliance filter
    compliant_choice = st.radio(
        "Compliance", ["All", "Compliant only", "Non-compliant only"],
        horizontal=True,
    )

    # Instruction-type multiselect — narrowed by source choice
    pool = df
    if source_choice == "CoTControl":
        pool = pool[pool["source_short"] == "cotcontrol"]
    elif source_choice == "ReasonIF":
        pool = pool[pool["source_short"] == "reasonif"]
    all_types = sorted(pool["instruction_type"].unique())
    sel_types = st.multiselect(
        "Instruction Type / Mode", all_types, default=all_types,
        help="Pick one or more modes (uppercase_thinking, repeat_sentences, "
             "json_format, etc.). Narrowed automatically by Source above.",
    )

mask = df["instruction_type"].isin(sel_types)
if source_choice == "CoTControl":
    mask = mask & (df["source_short"] == "cotcontrol")
elif source_choice == "ReasonIF":
    mask = mask & (df["source_short"] == "reasonif")
if compliant_choice == "Compliant only":
    mask = mask & (df["compliant"] == True)  # noqa: E712
elif compliant_choice == "Non-compliant only":
    mask = mask & (df["compliant"] == False)  # noqa: E712

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
meta_cols = st.columns(4)
row_source = _get_val(row.get("source", "")) if "source" in row.index else ""
meta_cols[0].markdown(f"**Instruction Type:** `{itype}`")
meta_cols[1].markdown(f"**Source:** `{row_source or 'n/a'}`")
if cargs:
    meta_cols[2].markdown(f"**Constraint Args:** `{cargs}`")
if instruction_desc:
    meta_cols[3].markdown(f"**Description:** {instruction_desc[:100]}")

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

# Reasoning — side-by-side diff when an original (pre-edit) reasoning exists.
# Word-level diff highlighting: removed text shown in red, added text in green.

_DEL_SPAN = '<span style="background:#ffd0d0;color:#900;">{}</span>'
_INS_SPAN = '<span style="background:#d0f4d0;color:#0a0;">{}</span>'


def _word_tokens(text: str) -> list[str]:
    """Token list that preserves whitespace as separate tokens — keeps the
    diff aligned to natural word boundaries while still rendering the
    original whitespace verbatim."""
    return re.findall(r"\S+|\s+", text or "")


def _structural_chunks(text: str) -> list[str]:
    """Split text into structural chunks for the OUTER diff pass.

    Strategy:
      1. If the text has multiple lines, split on newlines (keeping the
         newline character with each chunk).
      2. Otherwise (single long line, common in model rollouts), split on
         sentence boundaries (`.!?` followed by whitespace) while keeping
         the punctuation.

    The OUTER diff matches whole chunks against whole chunks. This stops
    `SequenceMatcher` from producing spurious matches on isolated stop
    words across long deleted/inserted regions, which was the failure
    mode the user reported.
    """
    if not text:
        return [""]
    if "\n" in text:
        # Split keeping newlines attached so we can re-render whitespace
        # exactly. ``splitlines(keepends=True)`` does this.
        return text.splitlines(keepends=True)
    # Single-line text — fall back to sentence-level chunks. Regex captures
    # everything up to a terminator + the terminator + any trailing whitespace.
    chunks = re.findall(r"[^.!?]*[.!?]+\s*|[^.!?]+$", text)
    return chunks if chunks else [text]


def _word_level_diff_html(a_text: str, b_text: str) -> tuple[str, str]:
    """Inner pass — word-level diff within a pair of replaced chunks."""
    a = _word_tokens(a_text)
    b = _word_tokens(b_text)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    left_parts: list[str] = []
    right_parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_chunk = "".join(a[i1:i2])
        b_chunk = "".join(b[j1:j2])
        if tag == "equal":
            esc = html_mod.escape(a_chunk)
            left_parts.append(esc)
            right_parts.append(esc)
        elif tag == "delete":
            left_parts.append(_DEL_SPAN.format(html_mod.escape(a_chunk)))
        elif tag == "insert":
            right_parts.append(_INS_SPAN.format(html_mod.escape(b_chunk)))
        elif tag == "replace":
            left_parts.append(_DEL_SPAN.format(html_mod.escape(a_chunk)))
            right_parts.append(_INS_SPAN.format(html_mod.escape(b_chunk)))
    return "".join(left_parts), "".join(right_parts)


def _render_diff_html(orig: str, edited: str) -> tuple[str, str]:
    """Two-level diff: structural chunks first, words within replaced
    chunks. Returns (orig_html, edited_html).

    Removed text gets a red background in the original column; added
    text gets a green background in the edited column.
    """
    a_chunks = _structural_chunks(orig)
    b_chunks = _structural_chunks(edited)
    sm = difflib.SequenceMatcher(a=a_chunks, b=b_chunks, autojunk=False)
    left_parts: list[str] = []
    right_parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_text = "".join(a_chunks[i1:i2])
        b_text = "".join(b_chunks[j1:j2])
        if tag == "equal":
            esc = html_mod.escape(a_text)
            left_parts.append(esc)
            right_parts.append(esc)
        elif tag == "delete":
            left_parts.append(_DEL_SPAN.format(html_mod.escape(a_text)))
        elif tag == "insert":
            right_parts.append(_INS_SPAN.format(html_mod.escape(b_text)))
        elif tag == "replace":
            # Recurse into a word-level diff for the replaced region.
            sub_left, sub_right = _word_level_diff_html(a_text, b_text)
            left_parts.append(sub_left)
            right_parts.append(sub_right)
    return "".join(left_parts), "".join(right_parts)


def _reasoning_box_html(content_html: str, bg: str = "#e8f5e9") -> str:
    """Wrap a (possibly highlighted) chunk of reasoning HTML in the same
    monospace box style we use elsewhere."""
    return (
        f'<div style="white-space: pre-wrap; word-wrap: break-word; '
        f'font-family: monospace; font-size: 0.85em; background: {bg}; '
        f'padding: 0.75em; border-radius: 0.375em; max-height: 600px; '
        f'overflow-y: auto;">{content_html}</div>'
    )


# Recover the original (pre-edit) reasoning if present.
orig_reasoning = ""
if "messages_original" in row.index:
    _, orig_reasoning, _ = _extract_parts(row["messages_original"])

word_count = len(re.findall(r"\w+", reasoning)) if reasoning else 0
char_count = len(reasoning) if reasoning else 0

if reasoning and orig_reasoning and orig_reasoning != reasoning:
    # Side-by-side diff view
    orig_wc = len(re.findall(r"\w+", orig_reasoning))
    st.markdown(
        f"### Reasoning — original vs edited "
        f"(orig: {orig_wc:,} words / {len(orig_reasoning):,} chars; "
        f"edited: {word_count:,} words / {char_count:,} chars)"
    )
    st.caption(
        '<span style="background:#ffd0d0;color:#900;padding:0 4px;'
        'border-radius:3px;">red = removed</span> &nbsp; '
        '<span style="background:#d0f4d0;color:#0a0;padding:0 4px;'
        'border-radius:3px;">green = added</span>',
        unsafe_allow_html=True,
    )
    left_html, right_html = _render_diff_html(orig_reasoning, reasoning)
    col_orig, col_edit = st.columns(2)
    with col_orig:
        st.markdown("**Original (pre-edit)**")
        st.markdown(_reasoning_box_html(left_html, bg="#fff5f5"),
                    unsafe_allow_html=True)
    with col_edit:
        st.markdown("**Edited (final)**")
        st.markdown(_reasoning_box_html(right_html, bg="#f0fff4"),
                    unsafe_allow_html=True)
elif reasoning:
    # Single-column view (no diff to show)
    st.markdown(f"### Reasoning ({word_count:,} words, {char_count:,} chars)")
    if orig_reasoning and orig_reasoning == reasoning:
        st.caption("(no edits — original and final reasoning are identical)")
    st.markdown(_reasoning_box_html(html_mod.escape(reasoning)),
                unsafe_allow_html=True)

# Response
if response:
    st.markdown("### Response")
    st.markdown(
        '<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; '
        "font-size: 0.85em; background: #fff3e0; padding: 0.75em; border-radius: 0.375em; "
        f'max-height: 400px; overflow-y: auto;">{html_mod.escape(response)}</div>',
        unsafe_allow_html=True,
    )

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
