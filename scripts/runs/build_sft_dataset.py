#!/usr/bin/env python3
"""Build a mixed-source SFT dataset for fine-tuning.

Replaces ``scripts/runs/generate_sft_data.py`` with a source/strategy-aware
pipeline. The builder lives in ``controllability.training.sft_builder``;
this script is a thin CLI.

Key defaults:
  * Stage-1 prompts have the reasoning instruction stripped, so the model
    produces unconstrained reasoning that Stage-2 edits into compliant form.
  * analysis_channel defaults to True for gpt-oss base models, False otherwise.
  * Editor model defaults to openai/gpt-4.1 (used for reasonif LLM transforms
    and cotcontrol word-suppression keyword picking + minimal rewriting).
  * Question pool defaults to external/reasonIF_ft/train-00000-of-00001.parquet
    (HuggingFaceH4/Multilingual-Thinking) — same pool used by the existing
    reasonif SFT pipeline.

Examples:

  # 500 reasonif + 500 cotcontrol for gpt-oss-20b (mixed dataset, 1000 rows):
  python scripts/runs/build_sft_dataset.py \\
      --base-model openai/gpt-oss-20b --backend tinker \\
      --sources reasonif:500,cotcontrol:500 \\
      --reasoning-effort medium

  # Pure cotcontrol, qwen3-8b, smaller:
  python scripts/runs/build_sft_dataset.py \\
      --base-model qwen/qwen3-8b --backend openrouter \\
      --sources cotcontrol:100

  # Dry-run to inspect the assignment plan:
  python scripts/runs/build_sft_dataset.py \\
      --base-model openai/gpt-oss-20b --backend tinker \\
      --sources reasonif:500,cotcontrol:500 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from controllability.training.sft_builder import (
    BuildConfig,
    SourceRequest,
    build_sft_dataset,
    plan_assignments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTION_POOL = PROJECT_ROOT / "external" / "reasonIF_ft" / "train-00000-of-00001.parquet"


def _parse_sources(raw: str) -> list[SourceRequest]:
    """'reasonif:500,cotcontrol:500' -> [SourceRequest(...), ...]"""
    out: list[SourceRequest] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Malformed --sources entry '{chunk}' (expected name:N)")
        name, n = chunk.split(":", 1)
        name = name.strip()
        try:
            count = int(n.strip())
        except ValueError:
            raise ValueError(f"Non-integer count in --sources: {chunk}")
        if name not in ("reasonif", "cotcontrol"):
            raise ValueError(f"Unknown source name: {name}")
        out.append(SourceRequest(name=name, count=count))
    return out


def _default_output_stem(
    base_model: str, sources: list[SourceRequest], analysis_channel: bool,
) -> str:
    model_short = base_model.split("/")[-1].lower()
    names = [s.name for s in sources]
    if len(names) == 1:
        composition = names[0]
    else:
        composition = "mixed"
    # keep "-stripped" suffix for consistency with legacy naming; ac optional
    suffix = "-stripped"
    if analysis_channel:
        suffix += "-ac"
    return f"{model_short}-{composition}-sft{suffix}"


def _resolve_analysis_channel(flag: bool | None, base_model: str) -> bool:
    if flag is not None:
        return flag
    return "gpt-oss" in base_model.lower()


def _build_config_from_args(args: argparse.Namespace) -> BuildConfig:
    sources = _parse_sources(args.sources)
    if not sources:
        raise ValueError("--sources produced an empty list")

    ac = _resolve_analysis_channel(args.analysis_channel, args.base_model)
    stem = args.output_stem or _default_output_stem(args.base_model, sources, ac)
    output_dir = Path(args.output_dir)
    output_path = args.output or output_dir / f"{stem}.parquet"
    checkpoint_path = args.checkpoint or output_dir / f"{stem}.checkpoint.jsonl"

    return BuildConfig(
        output_path=Path(output_path),
        checkpoint_path=Path(checkpoint_path),
        question_source_parquet=Path(args.question_source),
        reasonif_template_source=(
            Path(args.reasonif_template_source) if args.reasonif_template_source else None
        ),
        base_model=args.base_model,
        backend=args.backend,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
        max_concurrency=args.max_concurrency,
        max_retries=args.max_retries,
        system_prompt=args.system_prompt or "",
        strategy=args.strategy,
        sources=sources,
        editor_model=args.editor_model,
        analysis_channel=ac,
        seed=args.seed,
        max_transform_retries=args.max_transform_retries,
        run_id=args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )


def _dry_run(config: BuildConfig) -> None:
    import pandas as pd
    from controllability.training.sft_builder import (
        _load_reasonif_templates, _unwrap,
    )

    print("\n[DRY RUN] Configuration:")
    print(f"  base_model:        {config.base_model} ({config.backend})")
    print(f"  reasoning_effort:  {config.reasoning_effort}")
    print(f"  strategy:          {config.strategy}")
    print(f"  editor_model:      {config.editor_model}")
    print(f"  analysis_channel:  {config.analysis_channel}")
    print(f"  output:            {config.output_path}")
    print(f"  checkpoint:        {config.checkpoint_path}")
    print(f"  question_source:   {config.question_source_parquet}")
    print(f"  seed:              {config.seed}")
    print("  sources:")
    for s in config.sources:
        print(f"    - {s.name}: {s.count}  modes={s.resolve_modes()}")

    df = pd.read_parquet(config.question_source_parquet)
    questions = [str(_unwrap(x) or "").strip() for x in df["original_prompt"].tolist()]
    questions = [q for q in questions if q]
    print(f"\n  Question pool: {len(questions)} entries")

    reasonif_templates = {}
    if any(s.name == "reasonif" for s in config.sources):
        reasonif_templates = _load_reasonif_templates(config.question_source_parquet)
        print("  ReasonIF template counts:")
        for mode, rows in reasonif_templates.items():
            print(f"    {mode:24s}  {len(rows)} source rows")

    assignments = plan_assignments(config, questions, reasonif_templates)
    print(f"\n  Planned {len(assignments)} SFT rows "
          f"({len({a.question_idx for a in assignments})} unique questions)")

    # Per-source mode breakdown
    breakdown: dict[tuple[str, str], int] = {}
    for a in assignments:
        breakdown[(a.source, a.mode)] = breakdown.get((a.source, a.mode), 0) + 1
    print("\n  Mode breakdown:")
    for (src, mode), n in sorted(breakdown.items()):
        print(f"    {src:>10s}/{mode:<28s}  {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mixed-source SFT dataset")
    parser.add_argument(
        "--base-model", required=True,
        help="Model used for Stage-1 rollouts (and ignore_question regen)",
    )
    parser.add_argument(
        "--backend", choices=["openrouter", "tinker"], default="openrouter",
        help="Inference backend for the base model",
    )
    parser.add_argument(
        "--sources", required=True,
        help="Comma-separated source:count (e.g. 'reasonif:500,cotcontrol:500')",
    )
    parser.add_argument(
        "--strategy", default="generate_then_edit",
        choices=["generate_then_edit"],
        help="SFT generation strategy (currently only one implemented)",
    )
    parser.add_argument(
        "--question-source", default=str(DEFAULT_QUESTION_POOL),
        help="Parquet with 'original_prompt' column (question pool)",
    )
    parser.add_argument(
        "--reasonif-template-source", default=None,
        help=(
            "Parquet holding reasonif canonical constraint templates "
            "(constraint_name, constraint_args, instruction_description). "
            "Defaults to --question-source if not set."
        ),
    )
    parser.add_argument("--output-dir", default="results/sft")
    parser.add_argument("--output", default=None, help="Override parquet output path")
    parser.add_argument("--output-stem", default=None, help="Override auto-computed filename stem")
    parser.add_argument("--checkpoint", default=None, help="Override Stage-1 checkpoint path")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--max-concurrency", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--editor-model", default="openai/gpt-4.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-transform-retries", type=int, default=3)
    parser.add_argument("--run-id", default=None, help="Optional run identifier")

    ac_group = parser.add_mutually_exclusive_group()
    ac_group.add_argument(
        "--analysis-channel", action="store_true", default=None, dest="analysis_channel",
        help="Use 'analysis channel' terminology (auto-enabled for gpt-oss)",
    )
    ac_group.add_argument(
        "--no-analysis-channel", action="store_false", dest="analysis_channel",
        help="Force 'reasoning' terminology regardless of model",
    )

    parser.add_argument(
        "--judge-ignore-question", action="store_true",
        help="Additionally run an LLM judge over cotcontrol/ignore_question rows",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config + assignment plan without running",
    )

    args = parser.parse_args()
    config = _build_config_from_args(args)

    if args.dry_run:
        _dry_run(config)
        return

    asyncio.run(build_sft_dataset(config, judge_ignore_question=args.judge_ignore_question))


if __name__ == "__main__":
    main()
