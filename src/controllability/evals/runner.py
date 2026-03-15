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
    judge_ignore_question,
)
from controllability.evals.metrics import compute_metrics, format_metrics
from controllability.evals.modes import get_mode
from controllability.evals.prompts import build_cotcontrol_prompts, build_reasonif_prompts
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
        return OpenRouterClient(api_key=settings.openrouter_api_key)
    elif config.backend == "tinker":
        from controllability.inference.tinker_client import TinkerClient

        if not settings.tinker_api_key:
            raise ValueError("TINKER_API_KEY not set. Check your .env file.")
        return TinkerClient(model=config.model)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


def _build_prompts(
    mode_name: str, sample: Sample, model_name: str, adversarial_pressure: str
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for a given mode and sample."""
    mode = get_mode(mode_name)

    if mode.source == "reasonif":
        return build_reasonif_prompts(sample)
    else:
        return build_cotcontrol_prompts(
            mode=mode_name,
            sample=sample,
            model_name=model_name,
            adversarial_pressure=adversarial_pressure,
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

    # Check for completed work (resume support)
    completed_keys = load_completed_keys(output_path)
    if completed_keys:
        print(f"  Found {len(completed_keys)} completed rollouts, will skip them")

    # Create inference client
    client = _create_client(config, settings)
    all_rollouts: list[Rollout] = []

    try:
        for mode_name in config.modes:
            mode = get_mode(mode_name)
            print(f"\nMode: {mode_name}")

            # For ReasonIF modes, filter to samples matching the instruction type
            if mode.source == "reasonif":
                mode_samples = [
                    s for s in samples
                    if s.metadata.get("instruction_type") == mode_name
                ]
            else:
                mode_samples = samples

            # Filter out already-completed samples
            pending_samples = [
                s for s in mode_samples if (s.id, mode_name) not in completed_keys
            ]

            if not pending_samples:
                print(f"  All {len(mode_samples)} samples already completed, skipping")
                continue

            print(f"  {len(pending_samples)} pending samples (of {len(mode_samples)} total)")

            # Build prompts and inference requests
            prompt_pairs: list[tuple[str, str]] = []
            requests: list[InferenceRequest] = []

            for sample in pending_samples:
                sys_prompt, usr_prompt = _build_prompts(
                    mode_name, sample, config.model, config.adversarial_pressure
                )
                prompt_pairs.append((sys_prompt, usr_prompt))

                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": usr_prompt})

                requests.append(
                    InferenceRequest(
                        messages=messages,
                        model=config.model,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                    )
                )

            # Run batch inference
            responses = await run_batch(
                client,
                requests,
                max_concurrency=config.max_concurrency,
                max_retries=config.max_retries,
                desc=f"  {mode_name}",
            )

            # Grade and create rollouts
            mode_rollouts: list[Rollout] = []
            for sample, (sys_prompt, usr_prompt), resp in zip(
                pending_samples, prompt_pairs, responses
            ):
                # Extract answer and grade correctness
                extracted = extract_answer(resp.content, sample.options)
                correct = grade_correctness(extracted, sample.correct_answer, sample.options)

                # Grade compliance -- only on reasoning trace, not final response
                if resp.error:
                    grading_result = {"error": resp.error}
                    compliant = None
                elif not resp.reasoning and mode_name == "ignore_question":
                    # No reasoning = non-compliant (constraint says "think about anything else")
                    grading_result = {"compliant": False, "mode": "ignore_question", "note": "no_reasoning"}
                    compliant = False
                elif not resp.reasoning:
                    grading_result = {"error": "no_reasoning_trace"}
                    compliant = None
                elif mode_name == "ignore_question":
                    grading_result = judge_ignore_question(
                        reasoning=resp.reasoning,
                        user_prompt=usr_prompt,
                    )
                    compliant = grading_result.get("compliant")
                else:
                    grading_result = mode.grade_compliance(resp.reasoning, sample)
                    compliant = grading_result.get("compliant")

                # Detect meta-discussion in reasoning trace
                meta = detect_meta_discussion(resp.reasoning) if resp.reasoning else None

                rollout = Rollout(
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
                )
                mode_rollouts.append(rollout)

            # Save incrementally
            append_rollouts(output_path, mode_rollouts)
            all_rollouts.extend(mode_rollouts)

            # Print mode summary
            mode_metrics = compute_metrics(mode_rollouts)
            for _, m in mode_metrics.get("by_mode", {}).items():
                acc = f"{m['accuracy']:.1%}"
                comply = f"{m['compliance_rate']:.1%}" if m['compliance_rate'] is not None else "N/A"
                print(f"  accuracy={acc}  compliance={comply}  n={m['n']}")

    finally:
        await client.close()

    # Print overall summary
    if all_rollouts:
        print("\n" + "=" * 60)
        print(format_metrics(compute_metrics(all_rollouts)))

    print(f"\nResults saved to: {output_path}")
    return all_rollouts
