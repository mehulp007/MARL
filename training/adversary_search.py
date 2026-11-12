"""Search for the scenarios on which a trained policy does worst relative to a reference.

    python -m training.adversary_search --checkpoint checkpoints/cooperative.pt --out scenarios/adversarial

Writes ``adv_01.json`` ... (worst first; each is a ``Scenario`` plus the measured
regret) and ``index.json`` with the unperturbed regret and the search history.
The "unleash adversary" button of the UI loads the scenario files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from core import Config, Network
from core.adversary import Adversary
from core.baselines import FixedTimeController, MaxPressureController
from training.train import load_controller


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cross-entropy search for adversarial scenarios.")
    p.add_argument("--checkpoint", default="checkpoints/cooperative.pt")
    p.add_argument("--reference", choices=("maxpressure", "fixed"), default="maxpressure")
    p.add_argument("--out", default="scenarios/adversarial")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    over = {k: v for k, v in (("population", a.population), ("iterations", a.iterations)) if v is not None}
    cfg = Config.from_dict({"adversary": over}) if over else Config()
    net = Network(cfg)
    policy = load_controller(a.checkpoint)
    reference = MaxPressureController() if a.reference == "maxpressure" else FixedTimeController()
    adv = Adversary(policy, reference, cfg, np.random.default_rng(a.seed), net=net,
                    base_demand={"base_rate": cfg.demand.medium})
    t0 = time.perf_counter()
    base_regret, base_policy, base_ref = adv.baseline()
    print(f"[adversary] unperturbed: policy {base_policy:.1f}  {a.reference} {base_ref:.1f}  "
          f"regret {base_regret:+.1f} s", flush=True)

    def log(rec):
        print(f"[adversary] iteration {rec['iteration']}: best {rec['best']:+.1f}  mean {rec['mean']:+.1f}  "
              f"best so far {rec['best_so_far']:+.1f}  ({rec['evaluations']} evaluations, "
              f"{time.perf_counter() - t0:.0f} s)", flush=True)
    results = adv.search(top_k=a.top, on_iteration=log)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("adv_*.json"):
        old.unlink()
    index = {"policy": a.checkpoint, "reference": a.reference, "metric": adv.metric, "seeds": list(adv.seeds),
             "episode_len": adv.episode_len, "unperturbed": {"regret": base_regret, "policy": base_policy,
                                                             "reference": base_ref},
             "history": adv.history, "scenarios": []}
    for i, r in enumerate(results, 1):
        d = r.to_dict()
        d["scenario"]["name"] = f"adversarial_{i:02d}"
        d["found_regret_over_unperturbed"] = r.regret - base_regret
        (out / f"adv_{i:02d}.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
        index["scenarios"].append({"file": f"adv_{i:02d}.json", "regret": r.regret,
                                   "added_by_adversary": r.regret - base_regret,
                                   "events": [f"{e.t:.0f}s {e.kind}" for e in r.scenario.events]})
        print(f"[adversary] #{i}: regret {r.regret:+.1f} s ({r.regret - base_regret:+.1f} over unperturbed)  "
              + ", ".join(f"{e.t:.0f}s {e.kind}" for e in r.scenario.events), flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[adversary] done in {time.perf_counter() - t0:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
