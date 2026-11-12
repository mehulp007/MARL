"""Reward function.

    r_local_i = - w_h*h_i + w_x*x_i - w_e*e_i - w_m*m_i - w_b*b_i - w_s*s_i
    r_i       = (1 - beta) * r_local_i + beta * mean_{j in N(i)} r_local_j

``beta = reward.coop_beta`` is the cooperation mandate: that share of every
agent's reward is the local outcome of its grid neighbours. ``beta = 0`` gives
purely selfish agents.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .network import Network
from .simulator import IntervalStats

TERM_NAMES = ("h", "x", "e", "m", "b", "s")


def reward_terms(stats: IntervalStats, cfg: Config) -> dict:
    """Normalised, unweighted terms, each an (N,) float64 array.

    h  halting       mean halted vehicles on incoming lanes / C_H
    x  throughput    stop-line crossings per decision interval / X_REF
    e  emergency     mean number of halted emergency vehicles
    m  starvation    clip((longest wait - W_TOL) / W_TOL, 0, M_CLIP)
    b  wasted green  mean green lanes held by don't-block-the-box / 2
    s  switching     1 if a phase change was initiated
    """
    rc = cfg.reward
    k = max(1, stats.steps)
    intervals = k / cfg.substeps                      # 1 for a normal decision interval
    return {
        "h": stats.halt_sum / (k * rc.c_h),
        "x": stats.crossings / (rc.x_ref * intervals),
        "e": stats.ev_halt_sum / k,
        "m": np.clip((stats.wait_max - rc.w_tol) / rc.w_tol, 0.0, rc.m_clip),
        "b": stats.blocked_sum / (k * 2.0),
        "s": stats.switched.astype(np.float64),
    }


def local_reward(terms: dict, cfg: Config) -> np.ndarray:
    rc = cfg.reward
    return (-rc.w_h * terms["h"] + rc.w_x * terms["x"] - rc.w_e * terms["e"]
            - rc.w_m * terms["m"] - rc.w_b * terms["b"] - rc.w_s * terms["s"])


def mix_rewards(r_local: np.ndarray, net: Network, beta: float) -> np.ndarray:
    """Blend each agent's local reward with the mean of its existing neighbours'."""
    if beta == 0.0:
        return r_local.copy()
    nb = np.where(net.nbr_mask, r_local[np.maximum(net.nbr, 0)], 0.0).sum(axis=1)
    has = net.n_nbr > 0
    nb_mean = np.where(has, nb / np.maximum(net.n_nbr, 1), r_local)
    return (1.0 - beta) * r_local + beta * nb_mean


def compute_reward(stats: IntervalStats, net: Network, cfg: Config):
    """Returns ``(r, r_local, terms)``; ``r`` and ``r_local`` are (N,) float32."""
    terms = reward_terms(stats, cfg)
    r_local = local_reward(terms, cfg)
    r = mix_rewards(r_local, net, cfg.reward.coop_beta)
    return r.astype(np.float32), r_local.astype(np.float32), terms
