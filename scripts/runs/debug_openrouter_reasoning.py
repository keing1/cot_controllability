"""Debug script: call Qwen3.5-27b on OpenRouter for no_comma ReasonIF samples.

Saves raw API responses to diagnose missing reasoning traces.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

# Add project root so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllability.datasets.loader import load_dataset
from controllability.evals.prompts import build_reasonif_prompts

load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.5-27b"
OUTPUT_PATH = Path("results/debug_openrouter_qwen3.5-27b_no_comma.jsonl")


async def call_openrouter(session: aiohttp.ClientSession, api_key: str, messages: list[dict], sample_id: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0.6,
        "reasoning": {"max_tokens": 25000},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=600)
    async with session.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=timeout) as resp:
        raw = await resp.json()
    latency_ms = (time.monotonic() - start) * 1000

    # Extract what our pipeline would extract
    reasoning = ""
    content = ""
    if "choices" in raw:
        msg = raw["choices"][0]["message"]
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not reasoning and "<think>" in content:
            import re
            m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if m:
                reasoning = m.group(1).strip()

    return {
        "sample_id": sample_id,
        "model": MODEL,
        "backend": "openrouter",
        "latency_ms": round(latency_ms, 1),
        "has_reasoning": bool(reasoning),
        "reasoning_len": len(reasoning),
        "content_len": len(content),
        "reasoning_preview": reasoning[:200] if reasoning else None,
        "content_preview": content[:200] if content else None,
        "raw_response": raw,
    }


async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    # Load no_comma samples from reasonif
    samples = load_dataset("reasonif")
    no_comma = [s for s in samples if s.metadata.get("instruction_type") == "no_comma"]
    print(f"Found {len(no_comma)} no_comma samples")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed sample IDs for resume support
    done_ids = set()
    if OUTPUT_PATH.exists():
        for line in OUTPUT_PATH.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["sample_id"])
    remaining = [s for s in no_comma if s.id not in done_ids]
    print(f"Already done: {len(done_ids)}, remaining: {len(remaining)}")

    sem = asyncio.Semaphore(15)
    lock = asyncio.Lock()
    null_count = 0
    done_count = 0

    async def process(session, sample, idx):
        nonlocal null_count, done_count
        _, user_prompt = build_reasonif_prompts(sample)
        messages = [{"role": "user", "content": user_prompt}]

        async with sem:
            result = await call_openrouter(session, api_key, messages, sample.id)

        async with lock:
            done_count += 1
            if not result["has_reasoning"]:
                null_count += 1
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(result) + "\n")
            status = "OK" if result["has_reasoning"] else "NO REASONING"
            print(f"[{done_count}/{len(remaining)}] {sample.id}: {status} (reasoning={result['reasoning_len']}, content={result['content_len']}, {result['latency_ms']:.0f}ms)")

    async with aiohttp.ClientSession() as session:
        tasks = [process(session, s, i) for i, s in enumerate(remaining)]
        await asyncio.gather(*tasks)

        total = len(no_comma)
        # Recount from file for accuracy
        all_results = [json.loads(l) for l in OUTPUT_PATH.read_text().splitlines() if l.strip()]
        null_total = sum(1 for r in all_results if not r["has_reasoning"])
        print(f"\nDone. {null_total}/{len(all_results)} missing reasoning ({null_total/len(all_results):.0%})")
        print(f"Raw responses saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
