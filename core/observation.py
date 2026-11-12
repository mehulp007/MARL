"""Per-agent observation and the phase features used by baselines.

Every agent sees only its own intersection plus a small, fixed set of features
from its (up to) four grid neighbours. Layout, ``OBS_DIM = 81``, all in [0, 1]:

    [0:8]    queue length            halted / CAP_LANE              per own incoming lane
    [8:16]   cumulative wait         tanh(sum wait_link / W_REF)
    [16:24]  approaching count       moving / CAP_LANE
    [24:32]  approaching speed       mean speed of moving / V0
    [32:40]  emergency proximity     0, or floor + (1-floor) * pos/len
    [40:44]  downstream occupancy    vehicles on out_link[i, h] / capacity
    [44:48]  own phase               one-hot (target phase)
    [48]     own green elapsed       / G_MAX
    [49:81]  4 neighbour blocks (N, E, S, W) of 8:
             exists, phase one-hot (4), green elapsed, upstream feed, congestion

Own lanes are ordered ``j = 2*h + k`` (heading N, E, S, W; outer lane then inner).
"""
from __future__ import annotations

import numpy as np

from .config import LANES, N_PHASES, OBS_DIM
from .simulator import LaneStats, TrafficSim

OBS_SLICES = {
    "queue": slice(0, 8),
    "wait": slice(8, 16),
    "approach_count": slice(16, 24),
    "approach_speed": slice(24, 32),
    "emergency": slice(32, 40),
    "downstream": slice(40, 44),
    "phase": slice(44, 48),
    "green_elapsed": slice(48, 49),
    "neighbours": slice(49, 81),
}
NEIGHBOUR_FIELDS = ("exists", "phase0", "phase1", "phase2", "phase3", "green_elapsed", "feed", "congestion")


def phase_demand(sim: TrafficSim, ls: LaneStats) -> np.ndarray:
    """(N, 4) vehicles on the lanes of each phase."""
    cnt = ls.count[: 8 * sim.n].reshape(sim.n, 8)
    return cnt[:, :N_PHASES] + cnt[:, N_PHASES:]


def phase_wait(sim: TrafficSim, ls: LaneStats) -> np.ndarray:
    """(N, 4) longest wait (seconds) of any vehicle on the lanes of each phase."""
    w = ls.wait_max[: 8 * sim.n].reshape(sim.n, 8)
    return np.maximum(w[:, :N_PHASES], w[:, N_PHASES:])


def phase_pressure(sim: TrafficSim, ls: LaneStats) -> np.ndarray:
    """(N, 4) max-pressure weight of each phase.

    For every movement (lane l, move m) served by the phase:
    ``max(0, x(l, m) - x(target link) / LANES)`` with x = vehicles; summed per phase.
    """
    net, n = sim.net, sim.n
    mc = ls.move_count[: 8 * n].astype(np.float64)                 # (8N, 3)
    link = net.lane_link[: 8 * n]
    down = ls.link_count[net.move_target[link]] / LANES             # (8N, 3)
    w = np.maximum(0.0, mc - down).sum(axis=1).reshape(n, 8)
    return w[:, :N_PHASES] + w[:, N_PHASES:]


def build_observation(sim: TrafficSim, ls: LaneStats | None = None) -> np.ndarray:
    """(N, OBS_DIM) float32 observation matrix; row i is all agent i may use."""
    cfg, net, sg = sim.cfg, sim.net, sim.signals
    if ls is None:
        ls = sim.lane_stats()
    n, cap = sim.n, float(cfg.cap_lane)
    a = 8 * n
    ar = np.arange(n)
    halted = ls.halted[:a].reshape(n, 8)
    moving = ls.moving[:a].reshape(n, 8)
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    obs[:, 0:8] = np.minimum(halted / cap, 1.0)
    obs[:, 8:16] = np.tanh(ls.wait_sum[:a].reshape(n, 8) / cfg.obs.w_ref)
    obs[:, 16:24] = np.minimum(moving / cap, 1.0)
    obs[:, 24:32] = np.minimum(ls.mean_speed[:a].reshape(n, 8) / cfg.vehicle.v0, 1.0)
    obs[:, 32:40] = ls.ev_prox[:a].reshape(n, 8)
    out = net.out_link                                               # (N, 4)
    out_cap = LANES * np.where(net.is_exit[out], cfg.cap_exit_lane, cfg.cap_lane)
    obs[:, 40:44] = np.minimum(ls.link_count[out] / out_cap, 1.0)
    phase = sg.target_phase.astype(np.intp)
    elapsed = np.minimum(sg.green_elapsed / cfg.signal.g_max, 1.0)
    obs[ar, 44 + phase] = 1.0
    obs[:, 48] = elapsed
    congestion = np.minimum(halted.sum(axis=1) / (2.0 * cap), 1.0)
    for d in range(4):
        base = 49 + 8 * d
        j = net.nbr[:, d]
        m = j >= 0
        js = j[m]
        im = ar[m]
        obs[im, base] = 1.0
        obs[im, base + 1 + phase[js]] = 1.0
        obs[im, base + 5] = elapsed[js]
        # the neighbour in direction d feeds the link that arrives at i with heading opp(d)
        feed_link = net.in_link[im, (d + 2) % 4]
        obs[im, base + 6] = np.minimum(ls.inbound[feed_link] / cap, 1.0)
        obs[im, base + 7] = congestion[js]
    return obs
