"""Main evaluation orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path

from controllability.config import ExperimentConfig, Settings
from controllability.datasets.loader import load_dataset
from controllability.datasets.splits import proportional_sample, split_dataset
from controllability.evals.grading import (
    detect_meta_discussion,
    extract_answer,
    grade_correctness,
    judge_ignore_question_async,
)
from controllability.evals.metrics import compute_metrics, format_metrics
from controllability.evals.modes import get_mode
from controllability.evals.prompts import (
    apply_number_words_calibration,
    build_cotcontrol_prompts,
    build_reasonif_prompts,
)
from controllability.inference.base import InferenceClient
from controllability.inference.batch import run_batch
from controllability.inference.openrouter import OpenRouterClient
from controllability.rollouts.store import (
    append_rollouts,
    load_completed_keys,
    write_experiment_header,
)
from controllability.types import InferenceRequest, Rollout, Sample


def _create_client(config: ExperimentConfig, settings: Settings) -> InferenceClient:
    """Create the appropriate inference client based on config."""
    if config.backend == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Check your .env file.")
        return OpenRouterClient(
            api_key=settings.openrouter_api_key,
            request_timeout=config.request_timeout,
        )
    elif config.backend == "tinker":
        from controllability.inference.tinker_client import TinkerClient

        if not settings.tinker_api_key:
            raise ValueError("TINKER_API_KEY not set. Check your .env file.")
        return TinkerClient(
            model=config.model,
            model_path=config.model_path,
            reasoning_effort=config.reasoning_effort,
            request_timeout=config.request_timeout,
        )
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


def _build_prompts(
    mode_name: str,
    sample: Sample,
    model_name: str,
    adversarial_pressure: str,
    analysis_channel: bool = False,
    include_synonyms: bool = False,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for a given mode and sample."""
    mode = get_mode(mode_name)

    if mode.source == "reasonif":
        return build_reasonif_prompts(sample, analysis_channel=analysis_channel)
    else:
        return build_cotcontrol_prompts(
            mode=mode_name,
            sample=sample,
            model_name=model_name,
            adversarial_pressure=adversarial_pressure,
            include_synonyms=include_synonyms,
        )


async def run_experiment(config: ExperimentConfig) -> list[Rollout]:
    """Run a full evaluation experiment.

    Steps:
    1. Load dataset
    2. Apply split and fraction
    3. Create inference client
    4. For each mode: build prompts, run inference, grade, save
    5. Print summary
    """
    settings = Settings()
    output_path = Path(config.output_dir) / config.output_filename

    # Write experiment header
    write_experiment_header(output_path, config.model_dump())

    # Load and split dataset
    print(f"Loading dataset: {config.dataset}")
    samples = load_dataset(config.dataset)
    samples = split_dataset(
        samples, split=config.split, seed=config.seed, fraction=config.fraction
    )

    # Proportional stratified sampling when n_samples is set
    if config.n_samples is not None:
        require_kw = any(m in config.modes for m in ("word_suppression", "multiple_word_suppression"))
        samples = proportional_sample(
            samples, n=config.n_samples, seed=config.seed,
            require_valid_keywords=require_kw,
        )

    print(f"  {len(samples)} samples after split={config.split}, fraction={config.fraction}"
          + (f", n_samples={config.n_samples}" if config.n_samples else ""))

    # Apply model-specific number_words calibration (if available)
    if "number_words" in config.modes or config.dataset == "reasonif":
        n_patched = apply_number_words_calibration(
            samples, model=config.model, reasoning_effort=config.reasoning_effort,
        )
        if n_patched:
            print(f"  Applied model-specific number_words thresholds to {n_patched} samples")

    # Check for completed work (resume support)
    completed_keys = load_completed_keys(output_path)
    if completed_keys:
        print(f"  Found {len(completed_keys)} completed rollouts, will skip them")

    # Create inference client
    client = _create_client(config, settings)
    all_rollouts: list[Rollout] = []

    try:
        # Collect all pending work across modes, then run in one big batch
        work_items: list[tuple[str, Sample, str, str, InferenceRequest]] = []
        # (mode_name, sample, sys_prompt, usr_prompt, request)

        for mode_name in config.modes:
            mode = get_mode(mode_name)

            if mode.source == "reasonif":
                mode_samples = [
                    s for s in samples
                    if s.metadata.get("instruction_type") == mode_name
                ]
            else:
                mode_samples = samples

            pending_samples = [
                s for s in mode_samples if (s.id, mode_name) not in completed_keys
            ]

            if not pending_samples:
                print(f"\nMode: {mode_name}")
                print(f"  All {len(mode_samples)} samples already completed, skipping")
                continue

            print(f"\nMode: {mode_name}")
            print(f"  {len(pending_samples)} pending samples (of {len(mode_samples)} total)")

            for sample in pending_samples:
                sys_prompt, usr_prompt = _build_prompts(
                    mode_name, sample, config.model, config.adversarial_pressure,
                    analysis_channel=config.analysis_channel,
                    include_synonyms=config.word_suppression_synonyms,
                )

                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": usr_prompt})

                request = InferenceRequest(
                    messages=messages,
                    model=config.model,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
                work_items.append((mode_name, sample, sys_prompt, usr_prompt, request))

        if not work_items:
            print("\nAll work already completed.")
        else:
            # Run ALL requests in one batch (single shared semaphore)
            all_requests = [item[4] for item in work_items]
            print(f"\nRunning {len(all_requests)} requests across {len(config.modes)} modes...")
            responses = await run_batch(
                client,
                all_requests,
                max_concurrency=config.max_concurrency,
                max_retries=config.max_retries,
                desc="  all_modes",
            )

            # Grade and create rollouts, grouped by mode for saving.
            # ignore_question grading uses async LLM calls — collect and batch.
            mode_rollouts_map: dict[str, list[Rollout]] = {}

            # First pass: grade everything except ignore_question, collect
            # pending judge items with their pre-computed rollout fields.
            graded: list[tuple[dict, dict]] = []  # (rollout_kwargs, grading_placeholder)
            pending_judge: list[tuple[int, str, str]] = []  # (index, reasoning, user_prompt)

            for idx, ((mode_name, sample, sys_prompt, usr_prompt, _request), resp) in enumerate(
                zip(work_items, responses)
            ):
                mode = get_mode(mode_name)

                extracted = extract_answer(resp.content, sample.options)
                correct = grade_correctness(extracted, sample.correct_answer, sample.options)

                if resp.error:
                    grading_result = {"error": resp.error}
                    compliant = None
                elif not resp.reasoning and mode_name == "ignore_question":
                    grading_result = {"compliant": False, "mode": "ignore_question", "note": "no_reasoning"}
                    compliant = False
                elif not resp.reasoning:
                    grading_result = {"error": "no_reasoning_trace"}
                    compliant = None
                elif mode_name == "ignore_question":
                    # Defer to async batch below
                    grading_result = None
                    compliant = None
                    pending_judge.append((idx, resp.reasoning, usr_prompt))
                else:
                    grading_result = mode.grade_compliance(
                        resp.reasoning, sample,
                        check_synonyms=config.word_suppression_synonyms,
                    )
                    compliant = grading_result.get("compliant")

                meta = detect_meta_discussion(resp.reasoning) if resp.reasoning else None

                rollout_kwargs = dict(
                    experiment_id=config.experiment_id,
                    sample=sample,
                    model=config.model,
                    control_mode=mode_name,
                    split=config.split,
                    system_prompt=sys_prompt,
                    user_prompt=usr_prompt,
                    reasoning=resp.reasoning,
                    response=resp.content,
                    extracted_answer=extracted,
                    correct=correct,
                    compliant=compliant,
                    meta_discussion=meta,
                    latency_ms=resp.latency_ms,
                    usage=resp.usage,
                    grading_details=grading_result,
                    raw_response=resp.raw_response,
                )
                graded.append((rollout_kwargs, {}))

            # Batch async LLM judge for ignore_question
            if pending_judge:
                print(f"  Judging {len(pending_judge)} ignore_question samples (async)...")
                judge_coros = [
                    judge_ignore_question_async(reasoning=reasoning, user_prompt=usr_prompt)
                    for _, reasoning, usr_prompt in pending_judge
                ]
                judge_results = await asyncio.gather(*judge_coros)
                for (rollout_idx, _, _), result in zip(pending_judge, judge_results):
                    graded[rollout_idx][0]["grading_details"] = result
                    graded[rollout_idx][0]["compliant"] = result.get("compliant")

            # Create rollouts and group by mode
            for rollout_kwargs, _ in graded:
                rollout = Rollout(**rollout_kwargs)
                mode_rollouts_map.setdefault(rollout.control_mode, []).append(rollout)

            # Save per-mode and print summaries
            for mode_name, mode_rollouts in mode_rollouts_map.items():
                append_rollouts(output_path, mode_rollouts)
                all_rollouts.extend(mode_rollouts)

                mode_metrics = compute_metrics(mode_rollouts)
                for _, m in mode_metrics.get("by_mode", {}).items():
                    acc = f"{m['accuracy']:.1%}"
                    comply = f"{m['compliance_rate']:.1%}" if m['compliance_rate'] is not None else "N/A"
                    print(f"  {mode_name}: accuracy={acc}  compliance={comply}  n={m['n']}")

    finally:
        await client.close()

    # Print overall summary
    if all_rollouts:
        print("\n" + "=" * 60)
        print(format_metrics(compute_metrics(all_rollouts)))

    print(f"\nResults saved to: {output_path}")
    return all_rollouts
