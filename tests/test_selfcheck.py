"""pytest wrappers around ``core.selfcheck``.

One test per check group, run in quick mode. The authoritative full run is
``python -m core.selfcheck``. The ``performance`` group measures wall-clock
speed and is skipped here unless ``RUN_PERF=1`` is set, because it fails
spuriously when the CPU is busy (for example while training).
"""
import os

import pytest

from core import selfcheck

GROUPS = list(selfcheck.GROUPS)


@pytest.mark.parametrize("group", GROUPS)
def test_selfcheck_group(group):
    if group == "performance" and os.environ.get("RUN_PERF") != "1":
        pytest.skip("timing check; set RUN_PERF=1 on an idle machine")
    results = selfcheck.run([group], quick=True)
    assert results, f"group {group!r} produced no checks"
    failed = [f"{r.name} ({r.detail})" for r in results if not r.passed]
    assert not failed, "failed checks:\n  " + "\n  ".join(failed)


def test_every_invariant_group_is_registered():
    expected = {"network", "physics", "signals", "determinism", "observation", "reward", "routing",
                "perturbations", "scenarios", "performance", "baselines", "agents", "adversary"}
    assert set(GROUPS) == expected


def test_unknown_group_is_rejected():
    with pytest.raises(KeyError):
        selfcheck.run(["no_such_group"])
