"""Verify the save_at monkey-patch in scripts/runs/launch_ft.py drops
periodic saves at non-allowed steps and lets every other save through.

Mirrors the two real call patterns Tinker emits at training time:
  - periodic save:  name=f"{step:06d}"            (pure digits, e.g. "000075")
  - rolling save:   name=f"rolling_checkpoint_{step:06d}"
  - final save:     name="final"
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "runs"))

ALLOWED = {25, 50, 100, 200, 300, 400, 500}


@pytest.fixture
def patched_module():
    """Install the filter onto a stub `checkpoint_utils.save_checkpoint_async`
    and yield the stubbed module so we can inspect call counts.
    """
    import tinker_cookbook.checkpoint_utils as cu

    calls: list[dict] = []

    async def fake_original(*args, **kwargs):
        calls.append(kwargs)
        return {"saved": True, "name": kwargs.get("name")}

    # Save and replace the real one
    real = cu.save_checkpoint_async
    cu.save_checkpoint_async = fake_original

    # Now invoke the patcher (it wraps whatever's currently in cu)
    launch_ft = importlib.import_module("launch_ft")
    launch_ft._install_save_at_filter(ALLOWED)

    try:
        yield cu, calls
    finally:
        cu.save_checkpoint_async = real


def _call(cu, **kwargs):
    return asyncio.run(cu.save_checkpoint_async(**kwargs))


@pytest.mark.parametrize("step", sorted(ALLOWED))
def test_allowed_periodic_save_passes_through(patched_module, step):
    cu, calls = patched_module
    name = f"{step:06d}"
    result = _call(cu, name=name, kind="both")
    assert result == {"saved": True, "name": name}
    assert calls and calls[-1]["name"] == name


@pytest.mark.parametrize("step", [75, 125, 150, 175, 225, 250, 275, 325, 350, 375, 425, 450, 475])
def test_disallowed_periodic_save_is_filtered(patched_module, step):
    cu, calls = patched_module
    name = f"{step:06d}"
    result = _call(cu, name=name, kind="both")
    assert result is None
    assert not calls or calls[-1]["name"] != name


def test_final_save_always_passes_through(patched_module):
    cu, calls = patched_module
    result = _call(cu, name="final", kind="both")
    assert result == {"saved": True, "name": "final"}


@pytest.mark.parametrize("step", [25, 75, 100, 475])
def test_rolling_save_always_passes_through(patched_module, step):
    """Rolling saver names look like `rolling_checkpoint_000075` — they should
    NOT be filtered by step number, even when the embedded step would be
    rejected as a periodic save.
    """
    cu, calls = patched_module
    name = f"rolling_checkpoint_{step:06d}"
    result = _call(cu, name=name, kind="both")
    assert result == {"saved": True, "name": name}


def test_no_kwargs_passes_through(patched_module):
    """A call with no `name` kwarg defaults to "" (not a digit) — pass through."""
    cu, calls = patched_module
    result = _call(cu, kind="both")
    assert result == {"saved": True, "name": None}


def test_filter_uses_int_parse_correctly(patched_module):
    """A pathological all-digit name still parses as int. e.g. "0050" → 50.
    But Tinker's periodic naming is always 6-digit zero-padded — verify
    both 6-digit and short forms work."""
    cu, calls = patched_module
    # 6-digit zero-padded (Tinker's actual format)
    assert asyncio.run(cu.save_checkpoint_async(name="000050"))["saved"] is True
    assert asyncio.run(cu.save_checkpoint_async(name="000075")) is None
    # Short form (defense in depth — should still work)
    assert asyncio.run(cu.save_checkpoint_async(name="50"))["saved"] is True
    assert asyncio.run(cu.save_checkpoint_async(name="75")) is None


def test_gcd_of_save_at_is_correct():
    """The launcher derives save_every = GCD(save_at). Verify the GCD covers
    every required step."""
    from functools import reduce
    import math

    save_at = [25, 50, 100, 200, 300, 400, 500]
    save_every = reduce(math.gcd, save_at)
    assert save_every == 25
    # Tinker emits a save attempt at every step where step % save_every == 0
    # and step > 0. Confirm every required step is one of those attempts.
    for step in save_at:
        assert step % save_every == 0, f"step {step} not reachable with save_every={save_every}"

    # And also confirm we predicted the unwanted attempts correctly
    expected_offers = list(range(save_every, 501, save_every))  # 25,50,75,...,500
    unwanted = [s for s in expected_offers if s not in save_at]
    assert unwanted == [75, 125, 150, 175, 225, 250, 275, 325, 350, 375, 425, 450, 475]
