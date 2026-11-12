"""One training run.

    python -m training.train --config configs/default.yaml --name cooperative_s0 --seed 0

Writes
    runs/<name>/config.yaml       the fully resolved configuration
    runs/<name>/episodes.csv      one row per training episode
    runs/<name>/eval.csv          greedy evaluation on fixed seeds every train.eval_every episodes
    runs/<name>/tb/               TensorBoard scalars (unless --no-tensorboard)
    runs/<name>/DONE | FAILED     completion marker (JSON summary or traceback)
    checkpoints/<name>/ep_XXXX.pt snapshots at train.snapshot_episodes (ep_0000 = untrained)
    checkpoints/<name>/resume.pt  everything needed to continue (not committed to git)
    checkpoints/<name>.pt         final policy

Checkpoints are the dict from ``MADQN.state_dict()`` without the target network,
plus a ``meta`` entry. Load them with ``load_checkpoint`` below, or
``DQNController.from_state_dict(torch.load(path, weights_only=False))``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

from core import Config, Network, TrafficEnv
from core.agents import DQNController, Trainer

EPISODE_KPIS = ("time_in_system", "att", "mean_queue", "gws_arterial", "stops_per_veh",
                "longest_wait_mean", "ev_stops", "completion_ratio", "throughput")
EPISODE_COLUMNS = ("episode", "total_steps", "epsilon", "ret", "ret_local", "loss", "q_mean", "td_abs",
                   "buffer", "demand_rate", "profile", "n_events", "wall_time") + EPISODE_KPIS
RESUME_EVERY = 10


def load_config(path: str | Path | None, seed: int | None = None, episodes: int | None = None,
                threads: int | None = None) -> Config:
    """YAML overrides on top of the defaults, then command-line overrides on top of that."""
    data = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    if seed is not None:
        data.setdefault("train", {})["seed"] = int(seed)
    if episodes is not None:
        data.setdefault("train", {})["episodes"] = int(episodes)
    if threads is not None:
        data.setdefault("dqn", {})["torch_threads"] = int(threads)
    return Config.from_dict(data)


def slim_state(state: dict, meta: dict) -> dict:
    """Policy-only checkpoint: drop the target network and optimiser, attach metadata."""
    out = {k: v for k, v in state.items() if k not in ("q_target", "opt")}
    out["meta"] = meta
    return out


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint written by this project. Only use on files you produced yourself."""
    return torch.load(str(path), map_location="cpu", weights_only=False)


def load_controller(path: str | Path) -> DQNController:
    return DQNController.from_state_dict(load_checkpoint(path))


class _Csv:
    def __init__(self, path: Path, columns, append: bool):
        new = not (append and path.exists())
        self.fh = open(path, "w" if new else "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.fh, fieldnames=list(columns), extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.fh.flush()

    def row(self, d: dict) -> None:
        self.w.writerow(d)
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def train(cfg: Config, name: str, root: str | Path = ".", resume: bool = False,
          tensorboard: bool = True, quiet: bool = False) -> dict:
    """Run (or continue) one training run. Returns the summary also written to ``DONE``."""
    root = Path(root)
    run_dir, ckpt_dir = root / "runs" / name, root / "checkpoints" / name
    final_path, resume_path = root / "checkpoints" / f"{name}.pt", ckpt_dir / "resume.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for marker in ("DONE", "FAILED"):
        (run_dir / marker).unlink(missing_ok=True)
    say = (lambda *_: None) if quiet else (lambda s: print(s, flush=True))
    t_start = time.perf_counter()

    env = TrafficEnv(cfg, seed=cfg.train.seed, net=Network(cfg))
    trainer = Trainer(cfg, env=env)
    resumed_from = 0
    if resume and resume_path.exists():
        state = load_checkpoint(resume_path)
        trainer.learner.load_state_dict(state)
        trainer.episode = int(state["trainer"]["episode"])
        trainer.total_steps = int(state["trainer"]["total_steps"])
        trainer.eval_history = list(state["trainer"]["eval_history"])
        trainer.rng.bit_generator.state = state["trainer"]["rng_state"]
        resumed_from = trainer.episode
        say(f"[{name}] resuming at episode {resumed_from}")
    else:
        resume = False
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)

    ep_csv = _Csv(run_dir / "episodes.csv", EPISODE_COLUMNS, append=resume)
    eval_csv = {"w": None}
    tb = None
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(run_dir / "tb"))
        except Exception as exc:                          # noqa: BLE001 - logging is optional
            say(f"[{name}] TensorBoard disabled: {exc}")
    meta_base = {"name": name, "seed": cfg.train.seed, "coop_beta": cfg.reward.coop_beta,
                 "planned_episodes": cfg.train.episodes, "shared_parameters": cfg.dqn.shared_parameters}

    def save_resume() -> None:
        state = trainer.learner.state_dict(with_optimizer=True)
        state["trainer"] = {"episode": trainer.episode, "total_steps": trainer.total_steps,
                            "eval_history": trainer.eval_history,
                            "rng_state": trainer.rng.bit_generator.state}
        tmp = resume_path.with_suffix(".tmp")
        torch.save(state, str(tmp))
        tmp.replace(resume_path)

    def on_episode_end(s) -> None:
        row = s.to_dict()
        row.update({k: s.summary.get(k) for k in EPISODE_KPIS})
        ep_csv.row(row)
        if tb is not None:
            for k in ("epsilon", "ret", "ret_local", "loss", "q_mean", "td_abs", "demand_rate"):
                tb.add_scalar(f"train/{k}", row[k], s.episode)
            for k in ("time_in_system", "mean_queue", "gws_arterial", "ev_stops"):
                tb.add_scalar(f"train_kpi/{k}", row[k], s.episode)
        say(f"[{name}] ep {s.episode:3d}/{trainer.planned_episodes} eps {s.epsilon:.2f} "
            f"demand {s.demand_rate:4.0f} {s.profile:5s} return {s.ret:8.1f} loss {s.loss:7.4f} "
            f"TIS {s.summary['time_in_system']:6.1f} ({s.wall_time:.1f}s)")
        if s.episode % RESUME_EVERY == 0:
            save_resume()

    def on_eval(r: dict) -> None:
        if eval_csv["w"] is None:
            cols = ["episode"] + sorted(k for k in r if k != "episode")
            eval_csv["w"] = _Csv(run_dir / "eval.csv", cols, append=resume)
        eval_csv["w"].row(r)
        if tb is not None:
            for k in ("time_in_system", "att", "mean_queue", "gws_arterial", "ev_stops",
                      "longest_wait_mean", "return"):
                tb.add_scalar(f"eval/{k}", r[k], r["episode"])
        say(f"[{name}]   eval @ {r['episode']:3d}: TIS {r['time_in_system']:6.1f}  ATT {r['att']:6.1f}  "
            f"queue {r['mean_queue']:6.1f}  gws_art {r['gws_arterial']:.2f}  ev_stops {r['ev_stops']:.2f}")

    def on_snapshot(episode: int, state: dict) -> None:
        torch.save(slim_state(state, {**meta_base, "episode": episode}), str(ckpt_dir / f"ep_{episode:04d}.pt"))

    trainer.on_episode_end, trainer.on_eval, trainer.on_snapshot = on_episode_end, on_eval, on_snapshot
    try:
        trainer.run()
        last_eval = trainer.eval_history[-1] if trainer.eval_history else {}
        torch.save(slim_state(trainer.learner.state_dict(),
                              {**meta_base, "episode": trainer.episode, "final_eval": last_eval}), str(final_path))
        save_resume()
        summary = {"name": name, "episodes": trainer.episode, "resumed_from": resumed_from,
                   "seed": cfg.train.seed, "coop_beta": cfg.reward.coop_beta,
                   "updates": trainer.learner.updates, "final_eval": last_eval,
                   "wall_seconds": round(time.perf_counter() - t_start, 1), "checkpoint": str(final_path)}
        (run_dir / "DONE").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        say(f"[{name}] done: {trainer.episode} episodes in {summary['wall_seconds']:.0f} s -> {final_path}")
        return summary
    except BaseException:
        (run_dir / "FAILED").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        ep_csv.close()
        if eval_csv["w"] is not None:
            eval_csv["w"].close()
        if tb is not None:
            tb.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the traffic-signal agents.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--name", default=None, help="run name (default: the config file's stem)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads for this run")
    p.add_argument("--root", default=".", help="directory that holds runs/ and checkpoints/")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<name>/resume.pt if present")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.seed, a.episodes, a.threads)
    train(cfg, a.name or Path(a.config).stem, a.root, a.resume, not a.no_tensorboard, a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
