"""Static charts for the README and the results write-up.

    python -m training.plots --out docs/figures

Reads ``runs/<group>_s*/eval.csv`` (learning curves) and ``results/results.csv``
(written by ``training.evaluate``). Every chart is rendered twice, for light and
dark backgrounds. The numbers behind every chart are in ``results/summary.md``.

Colour follows the controller, never its rank: cooperative = blue, selfish =
orange, max-pressure = aqua, fixed timer = yellow, in that fixed order (checked
for colour-vision-deficiency separation in both themes). Text never takes a
series colour.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from training.evaluate import KPIS, MEDIUM, aggregate  # noqa: E402

ORDER = ("cooperative", "selfish", "maxpressure", "fixed")
LABELS = {"cooperative": "Cooperative agents", "selfish": "Selfish agents",
          "maxpressure": "Max-pressure", "fixed": "Fixed timer"}
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": {"cooperative": "#2a78d6", "selfish": "#eb6834", "maxpressure": "#1baf7a", "fixed": "#eda100"}},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "series": {"cooperative": "#3987e5", "selfish": "#d95926", "maxpressure": "#199e70", "fixed": "#c98500"}},
}
SCENARIO_LABELS = {"calm_low": "Calm (low demand)", "steady_medium": "Steady (medium demand)",
                   "heavy_high": "Heavy (high demand)", "rush_hour_wave": "Rush-hour wave",
                   "arterial_incident": "Arterial accident", "center_closure": "Central road closure",
                   "stadium_surge": "Stadium surge", "emergency_corridor": "Emergency corridor",
                   "compound_chaos": "Compound chaos"}


def _style(th: dict) -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "figure.facecolor": th["surface"], "axes.facecolor": th["surface"], "savefig.facecolor": th["surface"],
        "text.color": th["ink"], "axes.labelcolor": th["ink2"], "xtick.color": th["muted"],
        "ytick.color": th["muted"], "axes.edgecolor": th["axis"], "axes.linewidth": 0.8,
        "grid.color": th["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
    })


def _titles(fig, th, title: str, subtitle: str, top: float = 0.965) -> None:
    fig.text(0.06, top, title, fontsize=14, fontweight="bold", color=th["ink"], va="top")
    fig.text(0.06, top - 0.055, subtitle, fontsize=10, color=th["ink2"], va="top")


def read_results(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in KPIS + ("gridlock",):
                r[k] = float(r[k]) if r.get(k) not in (None, "") else float("nan")
            rows.append(r)
    return rows


def read_eval_curves(root: Path, group: str) -> dict:
    """{run name: (episodes, time in system)} from runs/<group>_s*/eval.csv"""
    out = {}
    for p in sorted((root / "runs").glob(f"{group}_s*/eval.csv")):
        with open(p, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            out[p.parent.name] = (np.array([float(r["episode"]) for r in rows]),
                                  np.array([float(r["time_in_system"]) for r in rows]))
    return out


def baseline_reference() -> dict:
    """Baselines on the seeds and demand the training evaluation uses, so the reference lines are exact."""
    from core import Config, Network, TrafficEnv
    from core.baselines import FixedTimeController, MaxPressureController, run_episode
    cfg = Config()
    env = TrafficEnv(cfg, net=Network(cfg))
    out = {}
    for name, ctrl in (("maxpressure", MaxPressureController()), ("fixed", FixedTimeController())):
        out[name] = float(np.mean([run_episode(env, ctrl, seed=s)["time_in_system"] for s in cfg.train.eval_seeds]))
    return out


# --------------------------------------------------------------------------
def learning_curves(root: Path, out: Path, theme: str, ref: dict) -> Path | None:
    th = THEMES[theme]
    _style(th)
    curves = {g: read_eval_curves(root, g) for g in ("cooperative", "selfish")}
    if not any(curves.values()):
        return None
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.07, right=0.83, top=0.80, bottom=0.11)
    x_max, ends = 0.0, []
    for g in ("cooperative", "selfish"):
        col = th["series"][g]
        runs = curves[g]
        if not runs:
            continue
        for ep, y in runs.values():
            ax.plot(ep, y, color=col, lw=1.0, alpha=0.38, solid_capstyle="round")
        n = min(len(ep) for ep, _ in runs.values())
        ep = next(iter(runs.values()))[0][:n]
        mean = np.mean([y[:n] for _, y in runs.values()], axis=0)
        ax.plot(ep, mean, color=col, lw=2.0, label=f"{LABELS[g]} (mean of {len(runs)} seeds)", solid_capstyle="round")
        ax.plot(ep[-1], mean[-1], "o", ms=8, color=col, mec=th["surface"], mew=2, zorder=5)
        ends.append((mean[-1], f"{mean[-1]:.0f} s"))
        x_max = max(x_max, ep[-1])
    for name in ("fixed", "maxpressure"):
        ax.axhline(ref[name], color=th["muted"], lw=1.0, zorder=1)
        ends.append((ref[name], f"{LABELS[name]} {ref[name]:.0f} s"))
    # direct labels at the right edge, nudged apart so they never collide
    ends.sort()
    lo, hi = ax.get_ylim()
    min_gap = 0.045 * (min(hi, 520) - lo)
    placed = []
    for y, text in ends:
        yy = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(yy)
        ax.annotate(text, xy=(x_max, y), xytext=(x_max * 1.025, yy), fontsize=9.5, color=th["ink2"], va="center",
                    annotation_clip=False)
    top = max(max(y.max() for _, y in runs.values()) for runs in curves.values() if runs)
    ax.set_ylim(min(lo, ref["maxpressure"] - 15), min(top * 1.04, 520))
    ax.set_xlim(0, x_max * 1.01)
    ax.grid(axis="y")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Time in system (seconds)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), labelcolor=th["ink2"])
    clipped = " Early values above 520 s are cut off." if top > 520 else ""
    _titles(fig, th, "Learning curves: time in system at medium demand",
            "Greedy policy on fixed evaluation seeds. Thin lines are individual training seeds. Lower is better." + clipped)
    path = out / f"learning_curves_{theme}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def scenario_comparison(rows: list, out: Path, theme: str) -> Path:
    th = THEMES[theme]
    _style(th)
    agg = aggregate(rows)
    scenarios = [s for s in SCENARIO_LABELS if any((c, s) in agg for c in ORDER)]
    present = [c for c in ORDER if any((c, s) in agg for s in scenarios)]
    fig, ax = plt.subplots(figsize=(10, 7.4))
    fig.subplots_adjust(left=0.215, right=0.95, top=0.82, bottom=0.08)
    h = 0.19
    for j, c in enumerate(present):
        vals = [agg.get((c, s), {}).get("time_in_system", np.nan) for s in scenarios]
        y = np.arange(len(scenarios)) + (j - (len(present) - 1) / 2) * h
        ax.barh(y, vals, height=h * 0.86, color=th["series"][c], label=LABELS[c], edgecolor=th["surface"], linewidth=1.2)
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in scenarios], color=th["ink2"])
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Time in system (seconds)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.005), ncol=len(present), labelcolor=th["ink2"],
              handlelength=1.1, columnspacing=1.6)
    _titles(fig, th, "Time in system by scenario",
            "Mean over 5 evaluation seeds (and 3 training seeds for the agents). Lower is better. "
            "Exact values: results/summary.md", top=0.975)
    path = out / f"scenario_comparison_{theme}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def key_metrics(rows: list, out: Path, theme: str) -> Path:
    th = THEMES[theme]
    _style(th)
    agg = aggregate(rows)
    panels = (("time_in_system", "Time in system (s)", "lower is better", "{:.0f}"),
              ("ev_stops", "Emergency-vehicle stops per trip", "lower is better", "{:.2f}"),
              ("gws_arterial", "Arterial Green Wave Score", "higher is better", "{:.2f}"),
              ("longest_wait_mean", "Typical longest wait (s)", "lower is better", "{:.0f}"))
    present = [c for c in ORDER if (c, MEDIUM) in agg]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 4.9))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.66, bottom=0.05, wspace=0.28)
    for ax, (key, title, sense, fmt) in zip(axes, panels):
        vals = [agg[(c, MEDIUM)][key] for c in present]
        x = np.arange(len(present))
        ax.bar(x, vals, width=0.62, color=[th["series"][c] for c in present], edgecolor=th["surface"], linewidth=1.2)
        top = max(vals)
        for i, c in enumerate(present):
            per_run = [r[key] for r in agg[(c, MEDIUM)]["runs"].values()]
            hi = vals[i]
            if len(per_run) > 1:                           # range across training seeds
                ax.plot([i, i], [min(per_run), max(per_run)], color=th["ink"], lw=1.2, solid_capstyle="butt")
                hi = max(hi, max(per_run))
            ax.text(i, hi + 0.025 * top, fmt.format(vals[i]), ha="center", va="bottom", fontsize=9.5, color=th["ink"])
        ax.set_ylim(0, top * 1.16)
        ax.set_xticks([])
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10.5, color=th["ink"], loc="left", pad=22)
        ax.text(0.0, 1.02, sense, transform=ax.transAxes, fontsize=9, color=th["muted"], va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, color=th["series"][c]) for c in present]
    fig.legend(handles, [LABELS[c] for c in present], loc="upper left", bbox_to_anchor=(0.04, 0.845),
               ncol=len(present), labelcolor=th["ink2"], handlelength=1.1, columnspacing=1.8)
    _titles(fig, th, "Four measures at medium demand",
            "Mean over 5 evaluation seeds. The thin vertical line on the agents' bars is the range across 3 training seeds.",
            top=0.97)
    path = out / f"key_metrics_{theme}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render the result charts.")
    p.add_argument("--root", default=".")
    p.add_argument("--results", default="results/results.csv")
    p.add_argument("--out", default="docs/figures")
    a = p.parse_args(argv)
    root, out = Path(a.root), Path(a.root) / a.out
    out.mkdir(parents=True, exist_ok=True)
    made = []
    ref = baseline_reference()
    results = root / a.results
    rows = read_results(results) if results.exists() else []
    for theme in THEMES:
        made.append(learning_curves(root, out, theme, ref))
        if rows:
            made.append(scenario_comparison(rows, out, theme))
            made.append(key_metrics(rows, out, theme))
    for m in made:
        if m is not None:
            print(f"[plots] wrote {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
