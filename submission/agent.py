"""DeadReckon-EgoMap DOOM agent (docs/CONTRACT.md): a thin wrapper around decision.DecisionEngine.
Every action is sampled from the trained network (policy.py)."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from decision import DecisionEngine  # noqa: E402
from model import sample  # noqa: E402  (module-level so evaluation tools can swap in greedy sampling)

# Observation-based backward mask (action_mask.py). Must match how the shipped weights were trained:
# the current weights (it2734) were trained and evaluated with it (training --mask-back 1).
MASK_BACKWARD = True


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)
        dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.engine = DecisionEngine(HERE / "weights" / "model.pt", dev)
        # warm-up (first torch call is slow; __init__ has 60 s, first act() only 10 s)
        self.reset()
        self.act(observation_space.sample())
        self.reset()

    @property
    def net(self):  # kept for tools that load other checkpoints into the agent
        return self.engine.net

    def reset(self) -> None:
        self.engine.reset()

    def act(self, observation: dict) -> np.ndarray:
        return self.engine.act(observation, mask_backward=MASK_BACKWARD, sampler=sample)
