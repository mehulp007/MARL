"""Plumbing tests for ``training/`` (tiny runs in a temporary directory)."""
import csv
import json

import numpy as np
import pytest

from core import Config
from core.agents import DQNController
from training import evaluate, launch, train as train_mod

TINY = {"env": {"episode_len": 100.0},
        "dqn": {"learn_starts": 64, "batch_size": 32, "buffer_size": 5000},
        "train": {"episodes": 2, "eval_every": 1, "eval_seeds": [1], "snapshot_episodes": [0, 2]}}


def _rows(path):
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_train_writes_logs_checkpoints_and_resumes(tmp_path):
    cfg = Config.from_dict(TINY)
    summary = train_mod.train(cfg, "tiny", root=tmp_path, tensorboard=False, quiet=True)
    run, ck = tmp_path / "runs" / "tiny", tmp_path / "checkpoints"
    assert summary["episodes"] == 2 and (run / "DONE").exists() and not (run / "FAILED").exists()
    assert len(_rows(run / "episodes.csv")) == 2 and len(_rows(run / "eval.csv")) == 2
    assert Config.from_dict(__import__("yaml").safe_load((run / "config.yaml").read_text())) == cfg
    assert (ck / "tiny" / "ep_0000.pt").exists() and (ck / "tiny" / "ep_0002.pt").exists()
    state = train_mod.load_checkpoint(ck / "tiny.pt")
    assert state["format"] == "marl-traffic-madqn-v1" and "q_target" not in state and "opt" not in state
    assert state["meta"]["episode"] == 2 and state["meta"]["coop_beta"] == cfg.reward.coop_beta
    assert isinstance(train_mod.load_controller(ck / "tiny.pt"), DQNController)
    # continue the same run to 4 episodes
    longer = cfg.override({"train": {"episodes": 4, "snapshot_episodes": [0, 2, 4]}})
    summary = train_mod.train(longer, "tiny", root=tmp_path, resume=True, tensorboard=False, quiet=True)
    assert summary["episodes"] == 4 and summary["resumed_from"] == 2
    eps = [int(r["episode"]) for r in _rows(run / "episodes.csv")]
    assert eps == [1, 2, 3, 4]
    assert (ck / "tiny" / "ep_0004.pt").exists()
    assert json.loads((run / "DONE").read_text())["episodes"] == 4


def test_load_config_applies_yaml_then_cli(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("reward:\n  coop_beta: 0.0\ntrain:\n  episodes: 7\n", encoding="utf-8")
    cfg = train_mod.load_config(p, seed=5, episodes=3, threads=1)
    assert cfg.reward.coop_beta == 0.0 and cfg.train.episodes == 3 and cfg.train.seed == 5
    assert cfg.dqn.torch_threads == 1
    assert train_mod.load_config(None) == Config()
    with pytest.raises(KeyError):
        (tmp_path / "bad.yaml").write_text("nope: 1\n", encoding="utf-8")
        train_mod.load_config(tmp_path / "bad.yaml")


def test_shipped_configs_are_valid():
    coop, selfish, smoke = (train_mod.load_config(f"configs/{n}.yaml") for n in ("default", "selfish", "smoke"))
    assert coop.reward.coop_beta == 0.4 and selfish.reward.coop_beta == 0.0
    assert coop.train.episodes == selfish.train.episodes == 300 and smoke.train.episodes == 10


def test_launch_names_runs_by_label_and_seed():
    plan = launch.run_names(["cooperative=configs/default.yaml", "selfish=configs/selfish.yaml"], [0, 1])
    assert [n for n, _, _ in plan] == ["cooperative_s0", "cooperative_s1", "selfish_s0", "selfish_s1"]
    assert plan[2][1] == "configs/selfish.yaml" and plan[3][2] == 1


def _fake_rows():
    rng = np.random.default_rng(0)
    base = {"fixed": 200.0, "maxpressure": 165.0, "random": 620.0, "cooperative": 160.0, "selfish": 168.0}
    rows = []
    for ctrl, tis in base.items():
        runs = [f"{ctrl}_s{k}" for k in range(3)] if ctrl in ("cooperative", "selfish") else [ctrl]
        for run in runs:
            for scenario in ("steady_medium", "arterial_incident", "stadium_surge", "emergency_corridor"):
                for seed in (1, 2):
                    r = {k: 1.0 for k in evaluate.KPIS}
                    r.update(controller=ctrl, run=run, scenario=scenario, seed=seed, gridlock=0,
                             time_in_system=tis + rng.normal(0, 1),
                             gws_arterial={"cooperative": 0.6, "selfish": 0.5, "random": 0.1}.get(ctrl, 0.5),
                             ev_stops=0.8 if ctrl in ("cooperative", "selfish") else 2.5,
                             recovery_time=300.0 if ctrl == "cooperative" else 500.0,
                             excess_veh_hours=1.0 if ctrl == "cooperative" else 3.0)
                    rows.append(r)
    return rows


def test_gates_and_summary_files(tmp_path):
    rows = _fake_rows()
    res = evaluate.write_outputs(rows, tmp_path / "eval", ["cooperative", "selfish", "maxpressure", "fixed", "random"])
    g = res["gates"]
    assert all(g[k]["passed"] for k in ("G1", "G2", "G3", "G4", "G5"))
    assert g["cooperative_vs_selfish"]["cooperative_better"] and not g["cooperative_vs_selfish"]["ranges_overlap"]
    assert len(_rows(tmp_path / "eval" / "results.csv")) == len(rows)
    assert "PASSED" in (tmp_path / "eval" / "summary.md").read_text(encoding="utf-8")
    worse = [dict(r, time_in_system=r["time_in_system"] + 60) if r["controller"] == "cooperative" else r for r in rows]
    g2 = evaluate.compute_gates(evaluate.aggregate(worse))
    assert not g2["G1"]["passed"] and not g2["G2"]["passed"] and not g2["cooperative_vs_selfish"]["cooperative_better"]


def test_evaluate_task_runs_a_baseline_and_a_checkpoint(tmp_path):
    cfg = Config.from_dict(TINY)
    train_mod.train(cfg, "agent_s0", root=tmp_path, tensorboard=False, quiet=True)
    found = evaluate.discover(tmp_path, ["agent"])
    assert [(g, r) for g, r, _ in found] == [("agent", "agent_s0")]
    for kind, ckpt in (("maxpressure", None), ("dqn", found[0][2])):
        rows = evaluate._evaluate_task({"kind": kind, "group": kind, "run": kind, "checkpoint": ckpt,
                                        "scenario": "arterial_incident", "seeds": [3]})
        assert len(rows) == 1 and np.isfinite(rows[0]["time_in_system"]) and rows[0]["recovery_time"] >= 0


def test_excess_recovery_measures_the_event_against_the_same_seed():
    from core.metrics import excess_recovery
    t = np.arange(5.0, 605.0, 5.0)
    base = [(x, 50.0, 10.0, 0, 0) for x in t]
    bump = np.where((t > 200) & (t <= 320), 30.0, 0.0)            # +30 vehicles for 120 s
    event = [(x, 50.0 + b, 10.0, 0, 0) for x, b in zip(t, bump)]
    r = excess_recovery(event, base, t_event=200.0, decision_dt=5.0)
    assert r["peak_excess"] == 30.0 and abs(r["excess_veh_hours"] - 30.0 * 120.0 / 3600.0) < 1e-9
    assert r["recovery_time"] == 125.0                             # first calm sample is at t = 325
    same = excess_recovery(base, base, t_event=200.0, decision_dt=5.0)
    assert same == {"excess_veh_hours": 0.0, "peak_excess": 0.0, "recovery_time": 0.0}
    never = excess_recovery([(x, 90.0, 10.0, 0, 0) if x > 200 else (x, 50.0, 10.0, 0, 0) for x in t], base, 200.0, 5.0)
    assert never["recovery_time"] == 400.0                         # capped at the time left in the episode