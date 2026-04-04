"""Batch inference with concurrency control and retry."""

from __future__ import annotations

import asyncio
import random

from tqdm.asyncio import tqdm_asyncio

from controllability.inference.base import InferenceClient
from controllability.types import InferenceRequest, InferenceResponse


async def _single_with_retry(
    client: InferenceClient,
    request: InferenceRequest,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    sample_timeout: float | None = 960,
) -> InferenceResponse:
    """Execute a single request with semaphore and exponential backoff retry.

    Args:
        sample_timeout: Total wall-clock timeout in seconds across all attempts
            for a single sample. None disables. Default 720s (12 min).
    """
    last_response: InferenceResponse | None = None
    deadline = asyncio.get_event_loop().time() + sample_timeout if sample_timeout else None

    for attempt in range(max_retries + 1):
        if deadline and asyncio.get_event_loop().time() >= deadline:
            return InferenceResponse(
                content="",
                error=f"sample_timeout: {sample_timeout}s exceeded after {attempt} attempts",
            )

        async with semaphore:
            response = await client.complete(request)

        if response.error is None:
            return response

        last_response = response

        if attempt < max_retries:
            delay = (2**attempt) + random.uniform(0, 1)
            if deadline and asyncio.get_event_loop().time() + delay >= deadline:
                break
            await asyncio.sleep(delay)

    # All retries exhausted -- return last error response
    assert last_response is not None
    return last_response


async def run_batch(
    client: InferenceClient,
    requests: list[InferenceRequest],
    max_concurrency: int = 30,
    max_retries: int = 3,
    desc: str = "Inference",
) -> list[InferenceResponse]:
    """Run a batch of inference requests with concurrency control.

    Returns responses in the same order as requests.
    Failed requests have the `error` field set.
    """
    if not requests:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        _single_with_retry(client, req, semaphore, max_retries) for req in requests
    ]

    responses: list[InferenceResponse] = await tqdm_asyncio.gather(
        *tasks, desc=desc, total=len(tasks)
    )

    return responses
