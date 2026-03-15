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
) -> InferenceResponse:
    """Execute a single request with semaphore and exponential backoff retry."""
    last_response: InferenceResponse | None = None

    for attempt in range(max_retries + 1):
        async with semaphore:
            response = await client.complete(request)

        if response.error is None:
            return response

        last_response = response

        if attempt < max_retries:
            delay = (2**attempt) + random.uniform(0, 1)
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
