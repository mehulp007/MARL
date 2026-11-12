"""Single source of every constant in the project.

Nothing outside this module may hard-code a number that appears here.
All config objects are frozen dataclasses; build variants with
``Config.from_dict({...})`` or ``cfg.override({...})`` (nested dicts, e.g. the
result of ``yaml.safe_load``). Units are SI: metres, seconds, m/s, radians.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any, Mapping

# --------------------------------------------------------------------------
# Enumerations (plain ints so they can live inside NumPy arrays)
# --------------------------------------------------------------------------

# Heading = direction of travel. World coordinates: x east, y SOUTH.
HEAD_N, HEAD_E, HEAD_S, HEAD_W = 0, 1, 2, 3
HEADING_NAMES = ("N", "E", "S", "W")
HEADING_DX = (0.0, 1.0, 0.0, -1.0)
HEADING_DY = (-1.0, 0.0, 1.0, 0.0)

# Movement at an intersection. New heading = (h + MOVE_DH[m]) % 4.
LEFT, THROUGH, RIGHT = 0, 1, 2
MOVE_NAMES = ("LEFT", "THROUGH", "RIGHT")
MOVE_DH = (-1, 0, 1)

# Lanes per link. Fixed by design: lane 0 (outer) = THROUGH + RIGHT,
# lane 1 (inner) = LEFT only. lane_id = LANES * link + k.
LANES = 2
LANE_OF_MOVE = (1, 0, 0)  # indexed by movement

# Signal phases (conflict-free by construction).
N_PHASES = 4
PHASE_NS_THROUGH, PHASE_NS_LEFT, PHASE_EW_THROUGH, PHASE_EW_LEFT = 0, 1, 2, 3
PHASE_NAMES = ("NS_THROUGH", "NS_LEFT", "EW_THROUGH", "EW_LEFT")

# Stage of an intersection's signal state machine.
STAGE_GREEN, STAGE_YELLOW, STAGE_ALL_RED = 0, 1, 2

# Per-lane signal colour.
SIG_RED, SIG_YELLOW, SIG_GREEN = 0, 1, 2

# Entity type stored in the vehicle arrays.
ETYPE_CAR, ETYPE_EMERGENCY, ETYPE_OBSTACLE = 0, 1, 2

# Link kind.
LINK_INTERNAL, LINK_ENTRY, LINK_EXIT = 0, 1, 2

# Observation layout. Fixed by the lane/phase design above.
OBS_OWN_LANES = 4 * LANES                      # 8 incoming lanes per agent
OBS_NEIGHBOUR_BLOCK = 8                        # features per neighbour
OBS_DIM = 5 * OBS_OWN_LANES + 4 + N_PHASES + 1 + 4 * OBS_NEIGHBOUR_BLOCK  # = 81
N_ACTIONS = N_PHASES


# --------------------------------------------------------------------------
# Config sections
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GridConfig:
    rows: int = 4
    cols: int = 4
    link_len: float = 200.0          # incoming links: start -> stop line
    exit_len: float = 60.0           # exit links
    lane_w: float = 3.2
    box: float = 8.0                 # junction box half-size
    # Junction path length per movement (LEFT, THROUGH, RIGHT).
    conn_len: tuple = (17.0, 16.0, 6.0)
    v_turn_left: float = 8.0         # speed cap while turning left
    v_turn_right: float = 7.0        # speed cap while turning right
    turn_slow_dist: float = 25.0     # cap also applies this close to the stop line
    # Routing penalty per movement in seconds (LEFT, THROUGH, RIGHT).
    turn_penalty: tuple = (8.0, 0.0, 3.0)
    pick_radius: float = 6.0         # max distance for Network.pick()


@dataclass(frozen=True)
class VehicleConfig:
    v0: float = 13.89                # desired speed (50 km/h)
    v0_sigma: float = 0.08           # per-driver factor ~ N(1, sigma)
    v0_clip: tuple = (0.8, 1.2)
    headway: float = 1.1             # IDM T  (calibrated: ~1,780 veh/h/lane discharge)
    accel: float = 2.0               # IDM A
    decel: float = 2.5               # IDM B (comfortable)
    decel_max: float = 6.0           # hard limit B_MAX
    decel_yellow: float = 3.0        # B_YELLOW in the can-stop rule
    s0: float = 2.0                  # standstill gap
    length: float = 4.5
    v_halt: float = 0.1              # below this a vehicle counts as halted
    v0_ev: float = 18.0              # emergency vehicle desired speed
    accel_ev: float = 2.5
    stop_margin: float = 1.0         # front bumper stops this far before the line
    wall_clamp: float = 0.5          # hard projection distance before a wall
    overlap_margin: float = 0.1      # hard no-overlap projection margin
    incident_len: float = 6.0        # length of an incident obstacle
    lc_lookahead: float = 60.0
    lc_cooldown: float = 3.0
    lc_front_margin: float = 1.0     # gap ahead >= s0 + this
    lc_rear_headway: float = 0.8     # gap behind >= s0 + this * v_follower
    lat_time: float = 1.0            # seconds a lane change takes visually
    box_block_speed: float = 1.0     # target-lane tail slower than this counts as "not clearing"
    box_block_dist: float = 15.0     # a held front this close to the line counts as wasted green
    max_veh: int = 2048              # struct-of-arrays capacity


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.5


@dataclass(frozen=True)
class SignalConfig:
    decision_dt: float = 5.0
    yellow: float = 3.0
    all_red: float = 1.0
    g_min: float = 6.0
    g_max: float = 60.0


@dataclass(frozen=True)
class DemandConfig:
    low: float = 300.0               # veh/h per entry link
    medium: float = 500.0
    high: float = 700.0
    base_rate: float = 500.0         # rate used unless overridden per episode
    side_bias: tuple = (1.0, 1.0, 1.0, 1.0)   # multiplier per city edge N, E, S, W
    profile: str = "flat"            # "flat" | "rush" | "waves"
    rush_peak: float = 1.6           # raised-cosine peak multiplier
    rush_center_frac: float = 0.5    # peak position as a fraction of the episode
    rush_width_frac: float = 0.6     # full width of the peak as a fraction
    wave_period: float = 600.0       # seconds per NS-heavy/EW-heavy cycle
    wave_amp: float = 0.5            # +/- modulation depth
    p_straight: float = 0.6          # P(destination is straight across)
    ev_rate: float = 12.0            # emergency vehicles per hour, network-wide
    ev_p_straight: float = 0.7


@dataclass(frozen=True)
class ObsConfig:
    w_ref: float = 300.0             # veh*s scale inside tanh for cumulative wait
    ev_floor: float = 0.1            # proximity value of an EV at pos = 0


@dataclass(frozen=True)
class RewardConfig:
    w_h: float = 1.0                 # halting penalty
    w_x: float = 0.5                 # throughput reward
    w_e: float = 10.0                # halted emergency vehicle penalty
    w_m: float = 0.25                # starvation penalty (a guard, not the main signal)
    w_b: float = 0.3                 # wasted green penalty
    w_s: float = 0.1                 # switching penalty
    c_h: float = 20.0                # halting normaliser (vehicles)
    x_ref: float = 5.0               # throughput normaliser (crossings per decision)
    w_tol: float = 120.0             # tolerated longest wait (s)
    m_clip: float = 2.0              # upper clip of the starvation term
    coop_beta: float = 0.4           # share of reward tied to neighbours


@dataclass(frozen=True)
class EnvConfig:
    episode_len: float = 1800.0


@dataclass(frozen=True)
class BaselineConfig:
    # Both baselines were tuned over low/medium/high demand so neither is a strawman.
    fixed_greens: tuple = (36.0, 6.0, 36.0, 6.0)   # green seconds per phase (fixed timer, 100 s cycle)
    mp_min_green: float = 20.0       # max-pressure: hold a phase with pressure this long
    mp_margin: float = 0.0           # max-pressure: advantage needed to switch
    mp_max_wait: float = 120.0       # max-pressure: starvation guard (s); inf = textbook max-pressure


@dataclass(frozen=True)
class MetricsConfig:
    rolling_window: float = 300.0
    gridlock_speed: float = 0.5
    gridlock_time: float = 120.0


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.95
    n_step: int = 3
    buffer_size: int = 500_000
    batch_size: int = 512
    lr: float = 3e-4
    grad_clip: float = 10.0
    tau: float = 0.005
    learn_starts: int = 5_000
    updates_per_step: int = 1
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.6            # fraction of training over which eps decays
    hidden: int = 256
    head_hidden: int = 128
    shared_parameters: bool = True
    reward_scale: float = 0.2        # learner sees reward * scale (keeps Q-values O(1..10))
    reward_clip: float = 20.0        # raw rewards are clipped to [-clip, +clip] before scaling
    torch_threads: int = 0           # 0 = leave PyTorch's default


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 300
    seed: int = 0
    demand_range: tuple = (250.0, 750.0)
    side_bias_range: tuple = (0.6, 1.4)
    profiles: tuple = ("flat", "rush", "waves")
    p_scenario: float = 0.3
    snapshot_episodes: tuple = (0, 5, 10, 20, 40, 80, 120, 160, 200, 250, 300)
    eval_every: int = 10
    eval_seeds: tuple = (1001, 1002, 1003)


@dataclass(frozen=True)
class AdversaryConfig:
    population: int = 24
    elites: int = 6
    iterations: int = 8
    episode_len: float = 900.0
    n_seeds: int = 2
    max_events: int = 3
    max_closure_time: float = 900.0
    max_surge: float = 3.0
    max_incidents: int = 2


@dataclass(frozen=True)
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    adversary: AdversaryConfig = field(default_factory=AdversaryConfig)

    def __post_init__(self) -> None:
        self.validate()

    # ---- derived quantities -------------------------------------------
    @property
    def n_agents(self) -> int:
        return self.grid.rows * self.grid.cols

    @property
    def n_exits(self) -> int:
        return 2 * (self.grid.rows + self.grid.cols)

    @property
    def n_entries(self) -> int:
        return self.n_exits

    @property
    def n_links(self) -> int:
        return 4 * self.n_agents + self.n_exits

    @property
    def n_lanes(self) -> int:
        return LANES * self.n_links

    @property
    def block(self) -> float:
        """Centre-to-centre spacing of intersections."""
        return self.grid.link_len + 2.0 * self.grid.box

    @property
    def cap_lane(self) -> int:
        """Jam capacity of one incoming lane (vehicles)."""
        return int(math.floor(self.grid.link_len / (self.vehicle.length + self.vehicle.s0)))

    @property
    def cap_exit_lane(self) -> int:
        return max(1, int(math.floor(self.grid.exit_len / (self.vehicle.length + self.vehicle.s0))))

    @property
    def substeps(self) -> int:
        """Simulation steps per agent decision (K in the reward)."""
        return int(round(self.signal.decision_dt / self.sim.dt))

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    # ---- construction / serialisation ---------------------------------
    @classmethod
    def from_dict(cls, overrides: Mapping[str, Any] | None = None) -> "Config":
        """Defaults with a nested dict of overrides applied on top."""
        return cls().override(overrides or {})

    def override(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new Config; unknown keys and wrong types raise."""
        return _override(self, overrides, path="")

    def to_dict(self) -> dict:
        return _plain(asdict(self))

    # ---- invariants ----------------------------------------------------
    def validate(self) -> None:
        g, v, s, sg, d, r, q = (self.grid, self.vehicle, self.sim, self.signal,
                                self.demand, self.reward, self.dqn)
        _req(g.rows >= 1 and g.cols >= 1, "grid must be at least 1x1")
        _req(len(g.conn_len) == 3 and len(g.turn_penalty) == 3,
             "grid.conn_len and grid.turn_penalty need 3 entries (LEFT, THROUGH, RIGHT)")
        _req(min(g.conn_len) > 0, "grid.conn_len must be positive")
        _req(LANES * g.lane_w <= g.box, "junction box too small for the lanes")
        _req(g.link_len > 4 * (v.length + v.s0) and g.exit_len > v.length + v.s0,
             "links too short")
        _req(s.dt > 0, "sim.dt must be positive")
        k = sg.decision_dt / s.dt
        _req(abs(k - round(k)) < 1e-9 and round(k) >= 1,
             "signal.decision_dt must be a whole multiple of sim.dt")
        for name in ("yellow", "all_red", "g_min"):
            x = getattr(sg, name) / s.dt
            _req(abs(x - round(x)) < 1e-9, f"signal.{name} must be a whole multiple of sim.dt")
        # Guarantees every intersection is GREEN at every decision instant.
        _req(sg.yellow + sg.all_red < sg.decision_dt,
             "signal.yellow + signal.all_red must be < signal.decision_dt")
        _req(0 < sg.g_min <= sg.g_max, "need 0 < g_min <= g_max")
        # A vehicle that proceeds on yellow must clear the line before red.
        _req(v.v0_ev / (2.0 * v.decel_yellow) <= sg.yellow + sg.all_red + 1e-9,
             "yellow + all_red too short for decel_yellow at v0_ev")
        _req(0 < v.accel and 0 < v.decel <= v.decel_max, "bad IDM accel/decel")
        _req(len(v.v0_clip) == 2 and 0 < v.v0_clip[0] <= 1 <= v.v0_clip[1], "bad v0_clip")
        _req(len(d.side_bias) == 4, "demand.side_bias needs 4 entries (N, E, S, W)")
        _req(d.profile in ("flat", "rush", "waves"), "unknown demand.profile")
        _req(0 <= d.p_straight <= 1 and 0 <= d.ev_p_straight <= 1, "probabilities out of range")
        _req(0 <= r.coop_beta <= 1, "reward.coop_beta must be in [0, 1]")
        _req(len(self.baseline.fixed_greens) == N_PHASES, "baseline.fixed_greens needs 4 entries")
        _req(0 < q.gamma < 1 and q.n_step >= 1, "bad dqn.gamma / dqn.n_step")
        _req(0 <= q.eps_end <= q.eps_start <= 1 and 0 < q.eps_frac <= 1, "bad epsilon schedule")
        _req(self.adversary.elites <= self.adversary.population, "adversary.elites > population")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _req(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"invalid config: {msg}")


def _plain(x: Any) -> Any:
    """Tuples -> lists so the result is YAML/JSON friendly."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_plain(v) for v in x]
    return x


def _coerce(path: str, current: Any, value: Any) -> Any:
    if is_dataclass(current):
        if not isinstance(value, Mapping):
            raise TypeError(f"{path}: expected a mapping")
        return _override(current, value, path)
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise TypeError(f"{path}: expected bool, got {value!r}")
        return value
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != int(value):
            raise TypeError(f"{path}: expected int, got {value!r}")
        return int(value)
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{path}: expected number, got {value!r}")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise TypeError(f"{path}: expected str, got {value!r}")
        return value
    if isinstance(current, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path}: expected a list, got {value!r}")
        if current:
            return tuple(_coerce(f"{path}[{i}]", current[0], v) for i, v in enumerate(value))
        return tuple(value)
    return value


def _override(obj: Any, overrides: Mapping[str, Any], path: str) -> Any:
    known = {f.name for f in fields(obj)}
    changes = {}
    for key, value in overrides.items():
        where = f"{path}.{key}" if path else str(key)
        if key not in known:
            raise KeyError(f"unknown config key: {where}")
        changes[key] = _coerce(where, getattr(obj, key), value)
    return replace(obj, **changes) if changes else obj
