#!/usr/bin/env python3
"""Rescore all monitor_qa rollouts from the v5 14-FT run with the
``metr_note_v3`` monitor (graded by gpt-5.4 instant).

Pointed at ``v5_14m_20260505_051845`` rollouts. After the v5 run finishes
(and optionally the continuation run too), call this to add metr_note_v3
scores in-place on each monitor_qa jsonl.

Coverage when fully complete:
    14 models × {000050, 000100, final} × {baseline, less_reasoning_v2,
    vocab_control, dont_think_v2}
    = 14 × 3 × 4 = 168 files

Files that don't yet exist (because evals haven't landed) are skipped
silently — re-run later when more arrive.

Usage:
    uv run python scripts/runs/rescore_metr_note_v3_v5.py --dry-run
    uv run python scripts/runs/rescore_metr_note_v3_v5.py
    uv run python scripts/runs/rescore_metr_note_v3_v5.py --include-continuation
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from controllability.config import Settings  # noqa: E402
from controllability.evals.prompts import get_monitor_prompt_builder  # noqa: E402
from controllability.evals.rescore import rescore_file  # noqa: E402
from controllability.inference.openrouter import OpenRouterClient  # noqa: E402

ROLLOUTS_DIR = REPO_ROOT / "results" / "rollouts"
MQA_DIR = ROLLOUTS_DIR / "monitor_qa"

MONITOR_TYPE = "metr_note_v3"
MONITOR_MODEL = "openai/gpt-5.4"   # reasoning_effort=None → "instant" mode

# In-flight budget = PARALLEL_FILES × PER_FILE_CONCURRENCY ≈ 5000
PARALLEL_FILES = 10
PER_FILE_CONCURRENCY = 500

V5_RUN_ID = "v5_14m_20260505_051845"
V5_CKPTS = ["000050", "000100", "final"]

# Continuation run produces only a "final" ckpt with run-id v5cont_<ts>.
# Discovered at runtime by globbing if --include-continuation is set.

VARIANTS = ["baseline", "less_reasoning_v2", "vocab_control", "dont_think_v2"]


def _collect_v5_files() -> list[Path]:
    """All monitor_qa rollouts produced by the v5 run."""
    out: list[Path] = []
    for variant in VARIANTS:
        for ckpt in V5_CKPTS:
            # File pattern: monitor_qa_<spec.model>-ft-<run>-<ckpt>_<variant>_*.jsonl
            # spec.model is the same across -with/-without variants, so multiple
            # files may match per (model, ckpt, variant) — we pick all of them.
            pattern = f"monitor_qa_*-ft-{V5_RUN_ID}-{ckpt}_{variant}_*.jsonl"
            out.extend(sorted(MQA_DIR.glob(pattern)))
    # Dedupe by path
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _collect_continuation_files() -> list[Path]:
    """All monitor_qa rollouts produced by any v5 continuation run."""
    out: list[Path] = []
    for variant in VARIANTS:
        # Continuation only saves the "final" ckpt.
        pattern = f"monitor_qa_*-ft-v5cont_*-final_{variant}_*.jsonl"
        out.extend(sorted(MQA_DIR.glob(pattern)))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


async def main(dry_run: bool, include_continuation: bool,
               parallel_files: int, per_file_concurrency: int) -> None:
    settings = Settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    files = _collect_v5_files()
    if include_continuation:
        files += _collect_continuation_files()

    logging.info(
        "Collected %d files. Rescore: %s (%s, instant). "
        "Per-file concurrency=%d, parallel_files=%d → in-flight ≈ %d",
        len(files), MONITOR_TYPE, MONITOR_MODEL,
        per_file_concurrency, parallel_files,
        per_file_concurrency * parallel_files,
    )

    if not files:
        logging.warning("No matching files yet — re-run after evals land.")
        return

    if dry_run:
        from controllability.evals.rescore import dry_run as dr
        total = 0
        for path in files:
            total += dr(path.parent, path.name, MONITOR_TYPE)
        logging.info("Dry-run: %d rollouts would be scored across %d files",
                     total, len(files))
        return

    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=120,
    )
    builder = get_monitor_prompt_builder(MONITOR_TYPE)

    file_sem = asyncio.Semaphore(parallel_files)
    total_scored = 0
    total_errors = 0
    finished = 0
    t0 = time.time()

    async def process_one(idx: int, path: Path) -> dict:
        nonlocal total_scored, total_errors, finished
        async with file_sem:
            ts = time.time()
            summary = await rescore_file(
                path,
                client,
                monitor_type=MONITOR_TYPE,
                monitor_model=MONITOR_MODEL,
                builder=builder,
                monitor_max_tokens=1024,
                monitor_temperature=0.0,
                concurrency=per_file_concurrency,
                max_retries=3,
            )
            elapsed = time.time() - ts
            total_scored += summary["scored"]
            total_errors += summary["errors"]
            finished += 1
            skipped = max(0, summary.get("total", 0) - summary.get("scored", 0)
                            - summary.get("errors", 0))
            logging.info(
                "[%3d/%d] %s — total=%d scored=%d errors=%d skipped=%d (%.1fs)",
                finished, len(files), path.name,
                summary.get("total", 0), summary["scored"],
                summary["errors"], skipped, elapsed,
            )
            return summary

    tasks = [asyncio.create_task(process_one(i, p))
             for i, p in enumerate(files)]
    await asyncio.gather(*tasks)
    await client.close()

    print()
    print("=" * 70)
    print(f"DONE — {finished} files, {total_scored} rollouts scored, "
          f"{total_errors} errors")
    print(f"Total wall time: {time.time() - t0:.0f}s")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Count rollouts that would be scored; make no API calls")
    p.add_argument("--include-continuation", action="store_true",
                   help="Also rescore monitor_qa files from the v5 "
                        "continuation run(s) (run-id prefix 'v5cont_')")
    p.add_argument("--parallel-files", type=int, default=PARALLEL_FILES)
    p.add_argument("--per-file-concurrency", type=int, default=PER_FILE_CONCURRENCY)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    asyncio.run(main(args.dry_run, args.include_continuation,
                     args.parallel_files, args.per_file_concurrency))
