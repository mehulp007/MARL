"""Benchmark every controller on the scenario library.

    python -m training.evaluate --groups cooperative selfish --promote

Controllers: fixed timer, max-pressure, random, and every checkpoint
``checkpoints/<group>_s<seed>.pt``. Every controller sees exactly the same
vehicles on each (scenario, seed) pair (common random numbers).

Writes into ``--out`` (default ``runs/eval``, plus a committed copy of the small
summary files in ``results/``):
    results.csv    one row per controller run x scenario x evaluation seed
    summary.csv    mean of every KPI per controller group x scenario
    summary.md     readable table of time in system
    gates.json     the acceptance gates G1-G5 with the measured numbers

``--promote`` copies the median training seed of each group (by time in system
at medium demand) to ``checkpoints/<group>.pt`` and its snapshots to
``checkpoints/<group>/``, which is what the server and the UI load.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

EVAL_SEEDS = (2001, 2002, 2003, 2004, 2005)          # never used for tuning or for training evaluation
BASELINES = ("fixed", "maxpressure", "random")
KPIS = ("time_in_system", "att", "delay", "mean_queue", "throughput", "stops_per_veh", "wait_per_veh",
        "wait_p95", "longest_wait_mean", "longest_wait_peak", "gws", "gws_arterial", "progression_len",
        "ev_count", "ev_delay", "ev_stops", "completion_ratio", "backlog", "mean_speed", "return",
        "return_local", "recovery_time", "excess_veh_hours", "peak_excess")
COLUMNS = ("controller", "run", "scenario", "seed", "gridlock") + KPIS
MEDIUM = "steady_medium"
RECOVERY_SCENARIOS = ("arterial_incident", "center_closure", "stadium_surge", "compound_chaos")
GATE_RECOVERY = ("arterial_incident", "stadium_surge")


def _evaluate_task(task: dict) -> list:
    """Worker: one controller on one scenario for all seeds. Runs in its own process."""
    import torch
    torch.set_num_threads(1)
    from core import Config, Network, TrafficEnv
    from core.baselines import FixedTimeController, MaxPressureController, RandomController, run_episode
    from core.scenarios import get_scenario
    from training.train import load_controller

    cfg = Config()
    net = Network(cfg)
    env = TrafficEnv(cfg, net=net)
    kind = task["kind"]
    if kind == "fixed":
        ctrl = FixedTimeController()
    elif kind == "maxpressure":
        ctrl = MaxPressureController()
    elif kind == "random":
        ctrl = RandomController()
    else:
        ctrl = load_controller(task["checkpoint"])
    from core.metrics import excess_recovery
    from core.scenarios import Scenario

    scenario = get_scenario(task["scenario"], cfg, net)
    paired = task["scenario"] in RECOVERY_SCENARIOS
    no_event = Scenario(scenario.name + "_no_event", (), dict(scenario.demand))
    nan = float("nan")
    rows = []
    for seed in task["seeds"]:
        s = run_episode(env, ctrl, seed=int(seed), scenario=scenario)
        effect = {"recovery_time": nan, "excess_veh_hours": nan, "peak_excess": nan}
        if paired:
            # the same seed without the events: the difference is what the perturbation caused
            series = list(env.metrics.series)
            run_episode(env, ctrl, seed=int(seed), scenario=no_event)
            effect = excess_recovery(series, env.metrics.series, scenario.first_event_time(),
                                     cfg.signal.decision_dt)
        row = {"controller": task["group"], "run": task["run"], "scenario": task["scenario"],
               "seed": int(seed), "gridlock": int(s["gridlock"])}
        row.update({k: float(s[k]) for k in KPIS if k in s})
        row.update(effect)
        rows.append(row)
    return rows


def discover(root: Path, groups) -> list:
    """(group, run name, checkpoint path) for every per-seed checkpoint of the given groups."""
    found = []
    for g in groups:
        for p in sorted((root / "checkpoints").glob(f"{g}_s*.pt")):
            found.append((g, p.stem, str(p)))
    return found


def aggregate(rows: list) -> dict:
    """{(controller, scenario): {kpi: mean, "n": rows, "runs": {run: mean time in system}}}"""
    out: dict = {}
    for r in rows:
        d = out.setdefault((r["controller"], r["scenario"]), {"_rows": []})
        d["_rows"].append(r)
    for key, d in out.items():
        rs = d.pop("_rows")
        d["n"] = len(rs)
        for k in KPIS:
            vals = np.array([r[k] for r in rs], dtype=float)
            d[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        d["gridlock"] = float(np.mean([r["gridlock"] for r in rs]))
        runs = sorted({r["run"] for r in rs})
        d["runs"] = {run: {k: float(np.mean([r[k] for r in rs if r["run"] == run]))
                           for k in ("time_in_system", "gws_arterial", "ev_stops", "longest_wait_mean")}
                     for run in runs}
    return out


def compute_gates(agg: dict, coop: str = "cooperative", selfish: str = "selfish") -> dict:
    """Acceptance gates with the numbers behind each verdict."""
    def get(c, s, k):
        return agg.get((c, s), {}).get(k, float("nan"))
    tis = lambda c: get(c, MEDIUM, "time_in_system")
    gates = {}
    c, f, m = tis(coop), tis("fixed"), tis("maxpressure")
    gates["G1"] = {"rule": "cooperative time in system at least 15% lower than the fixed timer (medium demand)",
                   "cooperative": c, "fixed": f, "change_pct": 100 * (c / f - 1), "passed": bool(c <= 0.85 * f)}
    gates["G2"] = {"rule": "cooperative time in system within 5% of max-pressure, or better (medium demand)",
                   "cooperative": c, "maxpressure": m, "change_pct": 100 * (c / m - 1), "passed": bool(c <= 1.05 * m)}
    g = {k: get(k, MEDIUM, "gws_arterial") for k in (coop, selfish, "random", "fixed", "maxpressure")}
    gates["G3"] = {"rule": "gws_arterial: cooperative > selfish > random (medium demand)", **g,
                   "passed": bool(g[coop] > g[selfish] > g["random"])}
    who = {"cooperative": coop, "selfish": selfish, "fixed": "fixed", "maxpressure": "maxpressure"}
    eff = {s: {k: {name: get(c, s, k) for name, c in who.items()}
               for k in ("excess_veh_hours", "recovery_time", "peak_excess")} for s in RECOVERY_SCENARIOS}
    gates["G4"] = {"rule": "the incident and the surge cause less extra queueing under cooperative than under "
                           "the fixed timer (same seed with and without the event)",
                   "effect": eff,
                   "passed": bool(all(eff[s]["excess_veh_hours"]["cooperative"] < eff[s]["excess_veh_hours"]["fixed"]
                                      for s in GATE_RECOVERY))}
    ev = {s: {"cooperative": get(coop, s, "ev_stops"), "fixed": get("fixed", s, "ev_stops"),
              "maxpressure": get("maxpressure", s, "ev_stops")} for s in (MEDIUM, "emergency_corridor")}
    gates["G5"] = {"rule": "emergency vehicles stop less often under cooperative than under the fixed timer",
                   "ev_stops": ev, "passed": bool(all(v["cooperative"] < v["fixed"] for v in ev.values()))}
    cs = agg.get((coop, MEDIUM), {}).get("runs", {})
    ss = agg.get((selfish, MEDIUM), {}).get("runs", {})
    gates["cooperative_vs_selfish"] = {
        "note": "time in system at medium demand per training seed; lower is better",
        "cooperative_runs": {k: v["time_in_system"] for k, v in cs.items()},
        "selfish_runs": {k: v["time_in_system"] for k, v in ss.items()},
        "cooperative_mean": c, "selfish_mean": tis(selfish),
        "cooperative_better": bool(c < tis(selfish)),
        "ranges_overlap": bool(cs and ss and min(v["time_in_system"] for v in cs.values())
                               <= max(v["time_in_system"] for v in ss.values())
                               and min(v["time_in_system"] for v in ss.values())
                               <= max(v["time_in_system"] for v in cs.values())),
    }
    return gates


def write_outputs(rows: list, out: Path, order: list) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    agg = aggregate(rows)
    scenarios = sorted({r["scenario"] for r in rows}, key=lambda s: [r["scenario"] for r in rows].index(s))
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["controller", "scenario", "n", "gridlock"] + list(KPIS))
        for c in order:
            for s in scenarios:
                if (c, s) in agg:
                    d = agg[(c, s)]
                    w.writerow([c, s, d["n"], f"{d['gridlock']:.2f}"] + [f"{d[k]:.3f}" for k in KPIS])
    lines = ["# Evaluation summary", "",
             "Mean time in system (seconds; lower is better). Every controller sees the same vehicles.", "",
             "| Scenario | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for s in scenarios:
        lines.append(f"| {s} | " + " | ".join(
            f"{agg[(c, s)]['time_in_system']:.1f}" if (c, s) in agg else "-" for c in order) + " |")
    for title, key, fmt in (("Arterial Green Wave Score (higher is better)", "gws_arterial", "{:.2f}"),
                            ("Emergency-vehicle stops per trip (lower is better)", "ev_stops", "{:.2f}"),
                            ("Typical longest wait at a junction, seconds (lower is better)", "longest_wait_mean", "{:.0f}")):
        lines += ["", f"## {title}", "", "| Scenario | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
        for s in scenarios:
            lines.append(f"| {s} | " + " | ".join(
                fmt.format(agg[(c, s)][key]) if (c, s) in agg else "-" for c in order) + " |")
    lines += ["", "## What each perturbation cost", "",
              "Extra queueing caused by the events, in vehicle-hours: each episode is compared with the same "
              "seed run without the events. In brackets: seconds until the extra queue is gone.", "",
              "| Scenario | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for s in scenarios:
        if s in RECOVERY_SCENARIOS:
            lines.append(f"| {s} | " + " | ".join(
                f"{agg[(c, s)]['excess_veh_hours']:.1f} ({agg[(c, s)]['recovery_time']:.0f} s)"
                if (c, s) in agg else "-" for c in order) + " |")
    gates = compute_gates(agg)
    lines += ["", "## Acceptance gates", ""]
    for name in ("G1", "G2", "G3", "G4", "G5"):
        lines.append(f"- **{name} {'PASSED' if gates[name]['passed'] else 'NOT MET'}** - {gates[name]['rule']}")
    cvs = gates["cooperative_vs_selfish"]
    lines += ["", "## Cooperative versus selfish (medium demand, per training seed)", "",
              "- cooperative: " + ", ".join(f"{k} {v:.1f}" for k, v in cvs["cooperative_runs"].items())
              + f" (mean {cvs['cooperative_mean']:.1f})",
              "- selfish: " + ", ".join(f"{k} {v:.1f}" for k, v in cvs["selfish_runs"].items())
              + f" (mean {cvs['selfish_mean']:.1f})",
              f"- per-seed ranges overlap: {cvs['ranges_overlap']}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    return {"agg": agg, "gates": gates}


def promote(root: Path, agg: dict, groups) -> dict:
    """Copy the median training seed of each group to ``checkpoints/<group>.pt`` (+ snapshots)."""
    chosen = {}
    for g in groups:
        runs = agg.get((g, MEDIUM), {}).get("runs", {})
        if not runs:
            continue
        ranked = sorted(runs, key=lambda r: runs[r]["time_in_system"])
        pick = ranked[len(ranked) // 2]
        ck = root / "checkpoints"
        shutil.copyfile(ck / f"{pick}.pt", ck / f"{g}.pt")
        dst = ck / g
        dst.mkdir(exist_ok=True)
        for p in (ck / pick).glob("ep_*.pt"):
            shutil.copyfile(p, dst / p.name)
        chosen[g] = {"run": pick, "time_in_system": runs[pick]["time_in_system"],
                     "all_runs": {r: runs[r]["time_in_system"] for r in ranked}}
    (root / "checkpoints" / "SELECTION.json").write_text(json.dumps(
        {"rule": "median training seed by time in system at medium demand (not the best one)",
         "selected": chosen}, indent=2), encoding="utf-8")
    return chosen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate all controllers on the scenario library.")
    p.add_argument("--groups", nargs="*", default=["cooperative", "selfish"])
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=list(EVAL_SEEDS))
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="runs/eval")
    p.add_argument("--publish", default="results", help="committed copy of the small summary files ('' to skip)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--promote", action="store_true")
    a = p.parse_args(argv)
    from core.scenarios import LIBRARY
    root = Path(a.root)
    scenarios = a.scenarios or list(LIBRARY)
    ckpts = discover(root, a.groups)
    tasks = [{"kind": b, "group": b, "run": b, "checkpoint": None, "scenario": s, "seeds": a.seeds}
             for b in BASELINES for s in scenarios]
    tasks += [{"kind": "dqn", "group": g, "run": run, "checkpoint": path, "scenario": s, "seeds": a.seeds}
              for g, run, path in ckpts for s in scenarios]
    print(f"[evaluate] {len(tasks)} tasks x {len(a.seeds)} seeds, {len(ckpts)} checkpoints, "
          f"{a.workers} workers", flush=True)
    t0 = time.perf_counter()
    rows = []
    if a.workers <= 1:
        for t in tasks:
            rows += _evaluate_task(t)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for i, part in enumerate(pool.map(_evaluate_task, tasks), 1):
                rows += part
                if i % 10 == 0 or i == len(tasks):
                    print(f"[evaluate] {i}/{len(tasks)} tasks ({time.perf_counter() - t0:.0f} s)", flush=True)
    order = [g for g in a.groups if any(r["controller"] == g for r in rows)] + ["maxpressure", "fixed", "random"]
    res = write_outputs(rows, root / a.out, order)
    if a.promote:
        print("[evaluate] promoted:", json.dumps(promote(root, res["agg"], a.groups)), flush=True)
    if a.publish:
        pub = root / a.publish
        pub.mkdir(parents=True, exist_ok=True)
        for name in ("summary.csv", "summary.md", "gates.json", "results.csv"):
            shutil.copyfile(root / a.out / name, pub / name)
    print((root / a.out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
