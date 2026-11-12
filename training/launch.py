"""Start several training runs side by side, each in its own process.

    python -m training.launch --runs cooperative=configs/default.yaml selfish=configs/selfish.yaml \
                              --seeds 0 1 2 --threads 2 --wait

Each run is named ``<label>_s<seed>`` and logs to ``runs/<name>/train.log``.
The children are started detached, so they keep running if this launcher is
closed; ``--wait-only`` re-attaches and waits for the ``DONE`` / ``FAILED``
markers. Runs that are already DONE are skipped; interrupted ones resume.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def run_names(runs, seeds) -> list:
    return [(f"{spec.split('=', 1)[0]}_s{seed}", spec.split("=", 1)[1], seed) for spec in runs for seed in seeds]


def _spawn(cmd, log_path: Path) -> subprocess.Popen:
    log = open(log_path, "ab")
    kw = dict(stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if sys.platform == "win32":
        new_group, no_window, breakaway = 0x00000200, 0x08000000, 0x01000000
        try:
            return subprocess.Popen(cmd, creationflags=new_group | no_window | breakaway, **kw)
        except OSError:
            return subprocess.Popen(cmd, creationflags=new_group | no_window, **kw)
    return subprocess.Popen(cmd, start_new_session=True, **kw)


def progress(root: Path, name: str) -> str:
    run = root / "runs" / name
    if (run / "DONE").exists():
        return "done"
    if (run / "FAILED").exists():
        return "FAILED"
    try:
        with open(run / "episodes.csv", "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return f"ep {rows[-1]['episode']}" if rows else "starting"
    except OSError:
        return "starting"


def wait(root: Path, names, poll: float = 15.0, report: float = 120.0) -> int:
    t0 = last = time.time()
    while True:
        state = {n: progress(root, n) for n in names}
        if all(s in ("done", "FAILED") for s in state.values()):
            print(f"[launch] finished after {(time.time() - t0) / 60:.1f} min: {state}", flush=True)
            return 1 if "FAILED" in state.values() else 0
        if time.time() - last >= report:
            last = time.time()
            print(f"[launch] {(time.time() - t0) / 60:5.1f} min  " + "  ".join(f"{n}: {s}" for n, s in state.items()),
                  flush=True)
        time.sleep(poll)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Launch several training runs in parallel.")
    p.add_argument("--runs", nargs="+", default=["cooperative=configs/default.yaml", "selfish=configs/selfish.yaml"],
                   help="label=config.yaml pairs")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=2, help="PyTorch threads per run")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--root", default=".")
    p.add_argument("--wait", action="store_true", help="block until every run has finished")
    p.add_argument("--wait-only", action="store_true", help="do not start anything, just wait")
    p.add_argument("--force", action="store_true", help="restart runs that are already DONE")
    a = p.parse_args(argv)
    root = Path(a.root)
    plan = run_names(a.runs, a.seeds)
    names = [n for n, _, _ in plan]
    if not a.wait_only:
        for name, config, seed in plan:
            run = root / "runs" / name
            if (run / "DONE").exists() and not a.force:
                print(f"[launch] {name}: already done, skipping", flush=True)
                continue
            run.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "training.train", "--config", config, "--name", name,
                   "--seed", str(seed), "--threads", str(a.threads), "--root", str(root)]
            if a.episodes is not None:
                cmd += ["--episodes", str(a.episodes)]
            if (root / "checkpoints" / name / "resume.pt").exists() and not a.force:
                cmd.append("--resume")
            proc = _spawn(cmd, run / "train.log")
            print(f"[launch] {name}: started (pid {proc.pid})", flush=True)
    if a.wait or a.wait_only:
        return wait(root, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
