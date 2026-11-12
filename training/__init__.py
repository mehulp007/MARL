"""Training pipeline: command-line plumbing around ``core.agents.Trainer``.

    python -m training.train     one training run
    python -m training.launch    several runs side by side (seeds x settings)
    python -m training.evaluate  benchmark every controller on the scenario library
    python -m training.plots     learning curves and comparison charts
    python -m training.adversary_search   worst-case scenarios for a trained policy
"""
