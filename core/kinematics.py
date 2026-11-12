"""Vectorised vehicle kinematics. Pure functions on NumPy arrays.

Conventions: ``pos`` is the FRONT bumper along the lane; a leader's rear is
``pos - length``; ``gap = leader_rear - pos``; ``dv = v - v_leader``.
"""
from __future__ import annotations

import numpy as np


def idm_accel(v, v0, gap, dv, a_max, headway, decel, s0, decel_max):
    """Intelligent Driver Model acceleration with smooth over-speed braking.

        s*     = s0 + max(0, v*T + v*dv / (2*sqrt(A*B)))
        a_free = A * (1 - (v/v0)^4)                    if v <= v0
               = -B * (1 - (v0/v)^(4A/B))              if v >  v0
        a      = clip(a_free - A * (s*/max(gap, 0.1))^2, -B_MAX, A)

    ``v, v0, gap, dv, a_max`` are arrays of equal length; the rest are scalars.
    """
    s_star = s0 + np.maximum(0.0, v * headway + v * dv / (2.0 * np.sqrt(a_max * decel)))
    r = v / v0
    r2 = r * r
    free = a_max * (1.0 - r2 * r2)
    over = r > 1.0
    if over.any():
        free[over] = -decel * (1.0 - np.power(1.0 / r[over], 4.0 * a_max[over] / decel))
    q = s_star / np.maximum(gap, 0.1)
    acc = free - a_max * q * q
    return np.minimum(np.maximum(acc, -decel_max), a_max)


def integrate(pos, v, acc, dt):
    """Ballistic update. Returns ``(pos_new, v_new)``; speed never goes negative.

    A vehicle that would reach v < 0 within the step stops after ``v^2 / (2|a|)``.
    """
    v_new = v + acc * dt
    dx = (v + 0.5 * acc * dt) * dt
    neg = v_new < 0.0
    if neg.any():
        dx[neg] = v[neg] * v[neg] / (-2.0 * acc[neg])
        v_new[neg] = 0.0
    return pos + dx, v_new


def project(pos, pos_new, v_new, x_limit, v_limit):
    """Hard projection: never pass ``x_limit`` and never move backwards (in place)."""
    over = pos_new > x_limit
    if over.any():
        pos_new[over] = np.maximum(pos[over], x_limit[over])
        v_new[over] = np.minimum(v_new[over], v_limit[over])
    return pos_new, v_new


def can_stop(dist, v, decel_yellow):
    """Yellow rule: True where the stop line ``dist`` ahead can still be respected."""
    return dist >= v * v / (2.0 * decel_yellow)


def lane_change_ok(gap_front, v_self, v_lead, gap_back, v_follower, vc) -> bool:
    """Gap acceptance for an instantaneous lane change (scalars).

    Ahead:  gap >= s0 + margin + braking distance for the speed surplus.
    Behind: gap >= s0 + headway * v_follower + braking distance for its surplus.
    ``vc`` is the VehicleConfig.
    """
    need_front = vc.s0 + vc.lc_front_margin
    if v_self > v_lead:
        need_front += (v_self * v_self - v_lead * v_lead) / (2.0 * vc.decel_max)
    need_back = vc.s0 + vc.lc_rear_headway * v_follower
    if v_follower > v_self:
        need_back += (v_follower * v_follower - v_self * v_self) / (2.0 * vc.decel_max)
    return gap_front >= need_front and gap_back >= need_back
