"""Learning agents (the only part of ``core`` that needs PyTorch)."""
from .dqn import DQNController, MADQN
from .networks import DuelingQNet, QEnsemble
from .replay import NStepAssembler, ReplayBuffer
from .trainer import EpisodeStats, Trainer

__all__ = ["MADQN", "DQNController", "DuelingQNet", "QEnsemble", "NStepAssembler",
           "ReplayBuffer", "Trainer", "EpisodeStats"]
