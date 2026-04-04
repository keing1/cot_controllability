"""Main evaluation orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path

from controllability.config import ExperimentConfig, MonitorQAConfig, Settings
from controllability.datasets.loader import load_dataset, load_monitor_dataset, subsample_dataset
from controllability.datasets.splits import proportional_sample, split_dataset
from controllability.evals.grading import (
    detect_meta_discussion,
    extract_answer,
    extract_main_task_response,
    extract_side_task_response,
    grade_correctness,
    grade_side_task,
    has_extraneous_text,
    judge_ignore_question_async,
)
from controllability.evals.metrics import (
    compute_metrics,
    compute_monitor_metrics,
    format_metrics,
    format_monitor_metrics,
    write_monitor_metrics_csv,
)
from controllability.evals.modes import get_mode
from controllability.evals.prompts import (
    apply_number_words_calibration,
    build_cotcontrol_prompts,
    build_reasonif_prompts,
    get_monitor_prompt_builder,
    get_monitor_qa_prompt_builder,
)
from controllability.inference.base import InferenceClient
from controllability.inference.batch import _single_with_retry, run_batch
from controllability.inference.openrouter import OpenRouterClient
from controllability.rollouts.store import (
    append_monitor_rollouts,
    append_rollouts,
    dedup_monitor_rollout_file,
    load_completed_keys,
    load_monitor_completed_keys,
    write_experiment_header,
)
from controllability.types import InferenceRequest, MonitorResult, MonitorRollout, Rollout, Sample


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


# ---------------------------------------------------------------------------
# Monitor QA experiment runner
# ---------------------------------------------------------------------------


def _create_monitor_actor_client(
    config: MonitorQAConfig, settings: Settings,
) -> InferenceClient:
    """Create the actor inference client for monitor_qa."""
    if config.actor_backend == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Check your .env file.")
        return OpenRouterClient(
            api_key=settings.openrouter_api_key,
            request_timeout=config.request_timeout,
            reasoning_effort=config.reasoning_effort,
        )
    elif config.actor_backend == "tinker":
        from controllability.inference.tinker_client import TinkerClient

        if not settings.tinker_api_key:
            raise ValueError("TINKER_API_KEY not set. Check your .env file.")
        return TinkerClient(
            model=config.actor_model,
            model_path=config.model_path,
            reasoning_effort=config.reasoning_effort,
            request_timeout=config.request_timeout,
        )
    else:
        raise ValueError(f"Unknown actor backend: {config.actor_backend}")


def _create_monitor_client(settings: Settings, config: MonitorQAConfig) -> OpenRouterClient:
    """Create the monitor inference client (always OpenRouter, non-reasoning)."""
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Check your .env file.")
    return OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=config.request_timeout,
    )


async def _run_monitor_streaming(
    pending: list[Sample],
    mode: str,
    config: MonitorQAConfig,
    actor_client: InferenceClient,
    monitor_client: OpenRouterClient,
    actor_prompt_builder,
    monitor_prompt_builders: dict[str, object],
    output_path: Path,
) -> list[MonitorRollout]:
    """Streaming actor->monitor pipeline.

    All samples are launched concurrently (bounded by semaphores). Each sample
    independently flows through actor inference -> monitor inference -> buffer.
    Every ``config.save_every`` completions the buffer is flushed to disk, so
    there is no idle time waiting for stragglers at chunk boundaries.

    Each sample is scored by all monitors in ``monitor_prompt_builders``.
    """
    save_every = config.save_every
    actor_sem = asyncio.Semaphore(config.actor_concurrency)
    monitor_sem = asyncio.Semaphore(config.monitor_concurrency)

    all_rollouts: list[MonitorRollout] = []
    buffer: list[MonitorRollout] = []
    buffer_lock = asyncio.Lock()
    saved_count = 0
    # Track saved keys to prevent duplicates — both from prior runs and
    # within the current run (guards against overlapping invocations).
    saved_keys: set[tuple[str, str]] = set(
        load_monitor_completed_keys(output_path)
    )

    # Ordered list of monitor types for deterministic first-monitor selection
    monitor_type_order = list(monitor_prompt_builders.keys())

    def _flush_sync(force: bool = False) -> None:
        """Flush buffer to disk if it has >= save_every items (or force).

        Must be called while holding buffer_lock. Fully synchronous (no await
        points) so the locked section is atomic w.r.t. the event loop.
        """
        nonlocal saved_count
        if not buffer:
            return
        if not force and len(buffer) < save_every:
            return
        # Deduplicate against saved_keys (and within the batch).
        seen: set[tuple[str, str]] = set()
        new_rollouts: list[MonitorRollout] = []
        for r in buffer:
            key = (r.sample.id, r.mode)
            if key not in saved_keys and key not in seen:
                seen.add(key)
                new_rollouts.append(r)
        buffer.clear()
        if new_rollouts:
            append_monitor_rollouts(output_path, new_rollouts)
            for r in new_rollouts:
                saved_keys.add((r.sample.id, r.mode))
            all_rollouts.extend(new_rollouts)
            saved_count += len(new_rollouts)
        n_errors = sum(1 for r in new_rollouts if r.actor_error)
        print(f"    Saved {saved_count}/{len(pending)} rollouts ({n_errors} errors in batch)")

    async def process_one(sample: Sample) -> MonitorRollout | None:
        # Skip if another coroutine (or a prior run) already saved this key.
        # This is a fast pre-check *before* doing expensive inference.
        if (sample.id, mode) in saved_keys:
            return None

        # --- Actor ---
        sys_prompt, usr_prompt = actor_prompt_builder.build_actor_prompt(
            sample, mode, model=config.actor_model,
        )
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": usr_prompt})

        actor_request = InferenceRequest(
            messages=messages,
            model=config.actor_model,
            max_tokens=config.actor_max_tokens,
            temperature=config.actor_temperature,
        )
        actor_resp = await _single_with_retry(
            actor_client, actor_request, actor_sem, config.actor_max_retries,
        )
        if actor_resp.error:
            print(f"    [actor error] sample={sample.id}: {actor_resp.error}")

        # --- Build monitor input ---
        main_task_resp = extract_main_task_response(actor_resp.content)
        side_task_resp = extract_side_task_response(actor_resp.content)
        expected_answer = sample.metadata.get("side_task_answer", "")
        side_task_correct = grade_side_task(side_task_resp, expected_answer)

        # Fail if there is text outside the expected tags
        if side_task_correct and has_extraneous_text(actor_resp.content or ""):
            side_task_correct = False

        stripped_prompt, stripped_reasoning, stripped_response = (
            actor_prompt_builder.build_stripped_versions(
                sample, usr_prompt, actor_resp.reasoning, actor_resp.content,
            )
        )

        # --- Run all monitors ---
        monitor_results: list[MonitorResult] = []
        total_monitor_latency = 0.0
        total_monitor_usage: dict[str, int] = {}

        for monitor_type in monitor_type_order:
            builder = monitor_prompt_builders[monitor_type]
            monitor_messages = builder.build_monitor_prompt(
                stripped_prompt, stripped_reasoning, stripped_response,
            )

            monitor_request = InferenceRequest(
                messages=monitor_messages,
                model=config.monitor_model,
                max_tokens=config.monitor_max_tokens,
                temperature=config.monitor_temperature,
            )
            monitor_resp = await _single_with_retry(
                monitor_client, monitor_request, monitor_sem, config.monitor_max_retries,
            )
            if monitor_resp.error:
                print(f"    [{monitor_type} error] sample={sample.id}: {monitor_resp.error}")

            score = builder.parse_score(monitor_resp.content)
            monitor_results.append(MonitorResult(
                monitor_type=monitor_type,
                monitor_model=config.monitor_model,
                monitor_prompt=monitor_messages,
                monitor_response=monitor_resp.content,
                monitor_score=score,
                monitor_latency_ms=monitor_resp.latency_ms,
                monitor_usage=monitor_resp.usage,
                monitor_error=monitor_resp.error,
            ))
            total_monitor_latency += monitor_resp.latency_ms
            for k, v in monitor_resp.usage.items():
                total_monitor_usage[k] = total_monitor_usage.get(k, 0) + v

        # --- Assemble rollout ---
        # Legacy flat fields from the first monitor
        first = monitor_results[0] if monitor_results else None
        rollout = MonitorRollout(
            experiment_id=config.experiment_id,
            sample=sample,
            model=config.actor_model,
            mode=mode,
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            reasoning=actor_resp.reasoning,
            response=actor_resp.content,
            main_task_response=main_task_resp,
            side_task_response=side_task_resp,
            side_task_correct=side_task_correct,
            monitor_results=monitor_results,
            # Legacy fields from first monitor
            monitor_model=first.monitor_model if first else "",
            monitor_prompt=first.monitor_prompt if first else [],
            monitor_response=first.monitor_response if first else "",
            monitor_score=first.monitor_score if first else None,
            stripped_prompt=stripped_prompt,
            stripped_reasoning=stripped_reasoning,
            stripped_response=stripped_response,
            actor_latency_ms=actor_resp.latency_ms,
            monitor_latency_ms=total_monitor_latency,
            actor_usage=actor_resp.usage,
            monitor_usage=total_monitor_usage,
            actor_error=actor_resp.error,
            monitor_error=first.monitor_error if first else None,
        )

        # --- Buffer & flush ---
        async with buffer_lock:
            buffer.append(rollout)
            completed_count[0] += 1
            c = completed_count[0]
            # Log progress every 25 completions
            if c % 25 == 0 or c == total:
                print(f"    Completed {c}/{total}")
            _flush_sync()

        return rollout

    total = len(pending)
    completed_count = [0]  # mutable counter for nested scope

    # Launch all samples concurrently; semaphores control actual parallelism.
    tasks = [process_one(s) for s in pending]
    print(f"  Streaming {total} samples with {len(monitor_type_order)} monitor(s): {monitor_type_order} (save every {save_every})...")
    await asyncio.gather(*tasks)

    # Flush any remaining rollouts
    async with buffer_lock:
        _flush_sync(force=True)

    return all_rollouts


async def run_monitor_experiment(config: MonitorQAConfig) -> list[MonitorRollout]:
    """Run a monitor_qa experiment.

    Streaming pipeline: each sample independently flows through
    actor → monitor → buffer → disk (flushed every save_every completions).
    No idle time waiting for stragglers at batch boundaries.

    Supports resume via (sample_id, mode) keys.
    """
    from pathlib import Path

    settings = Settings()
    output_path = Path(config.output_dir) / config.output_filename

    # Write experiment header
    write_experiment_header(output_path, {
        "type": "monitor_qa",
        "actor_model": config.actor_model,
        "monitor_model": config.monitor_model,
        "dataset_path": config.dataset_path,
        "modes": config.modes,
        "experiment_id": config.experiment_id,
    })

    # Load dataset
    print(f"Loading dataset: {config.dataset_path}")
    samples = load_monitor_dataset(config.dataset_path)
    samples = subsample_dataset(samples, config.n_samples, config.seed)
    print(f"  {len(samples)} samples" + (
        f" (subsampled to {config.n_samples})" if config.n_samples else ""
    ))

    # Clean up any duplicates left by orphaned workers from prior killed runs.
    n_deduped = dedup_monitor_rollout_file(output_path)
    if n_deduped:
        print(f"  Cleaned {n_deduped} duplicate rows from previous run")

    # Check for completed work
    completed_keys = load_monitor_completed_keys(output_path)
    if completed_keys:
        print(f"  Found {len(completed_keys)} completed rollouts, will skip them")

    # Get prompt builders
    actor_prompt_builder = get_monitor_qa_prompt_builder(config.prompt_type)
    monitor_prompt_builders = {
        mt: get_monitor_prompt_builder(mt) for mt in config.monitor_types
    }

    # Create clients
    actor_client = _create_monitor_actor_client(config, settings)
    monitor_client = _create_monitor_client(settings, config)

    all_rollouts: list[MonitorRollout] = []

    try:
        for mode in config.modes:
            print(f"\nMode: {mode}")

            # Re-read completed keys before each mode to pick up rollouts
            # saved by a prior mode (or a concurrent run) since we started.
            completed_keys = load_monitor_completed_keys(output_path)

            # Deduplicate pending by sample id (keep first occurrence)
            seen_ids: set[str] = set()
            pending: list[Sample] = []
            for s in samples:
                if (s.id, mode) not in completed_keys and s.id not in seen_ids:
                    seen_ids.add(s.id)
                    pending.append(s)
            if not pending:
                print(f"  All {len(samples)} samples already completed, skipping")
                continue
            print(f"  {len(pending)} pending samples (of {len(samples)} total)")

            mode_rollouts = await _run_monitor_streaming(
                pending=pending,
                mode=mode,
                config=config,
                actor_client=actor_client,
                monitor_client=monitor_client,
                actor_prompt_builder=actor_prompt_builder,
                monitor_prompt_builders=monitor_prompt_builders,
                output_path=output_path,
            )
            all_rollouts.extend(mode_rollouts)

            mode_metrics = compute_monitor_metrics(mode_rollouts)
            for m_name, m in mode_metrics.get("by_mode", {}).items():
                st_acc = f"{m['side_task_accuracy']:.1%}" if m["side_task_accuracy"] is not None else "N/A"
                avg_sc = f"{m['avg_monitor_score']:.2f}" if m["avg_monitor_score"] is not None else "N/A"
                print(f"  {m_name}: side_task_acc={st_acc}  avg_monitor_score={avg_sc}  n={m['n']}")

    finally:
        await actor_client.close()
        await monitor_client.close()

    # Print overall summary
    if all_rollouts:
        print("\n" + "=" * 60)
        print(format_monitor_metrics(compute_monitor_metrics(all_rollouts)))

        # Write CSV metrics
        csv_path = output_path.with_suffix(".csv")
        write_monitor_metrics_csv(all_rollouts, csv_path)
        print(f"\nCSV metrics saved to: {csv_path}")

    print(f"Results saved to: {output_path}")
    return all_rollouts
