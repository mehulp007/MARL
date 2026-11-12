"""MARL Traffic Optimizer — pure math/logic library.

Rules: no I/O, no printing, no global
state, no dependency on the server or UI. Only numpy (and torch inside
``core.agents``).
"""
from .config import Config
from .demand import DemandSpec
from .network import Network, RoutingTables
from .simulator import IntervalStats, LaneStats, TrafficSim, TripLog
from .scenarios import Event, Scenario
from .env import TrafficEnv

__version__ = "0.3.0"
__all__ = ["Config", "DemandSpec", "Network", "RoutingTables", "TrafficSim", "TrafficEnv",
           "LaneStats", "IntervalStats", "TripLog", "Event", "Scenario", "__version__"]
