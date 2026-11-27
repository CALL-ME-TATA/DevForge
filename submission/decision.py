"""DecisionEngine: observation -> EgoMap features -> trained network -> action.

Owns the network (policy.py via model.py), the observation encoder (features.Features), the recurrent
state (GRU h, learned map L) and the observation-based action masks. Every action is sampled from the
network's logits; a mask can only RELEASE a button, never press one (docs/RULES.md allows action
masking from observation fields):
  - wmask (in Net.step): no weapon switch while the held weapon still has ammo
  - action_mask.mask_actions: no MOVE_BACKWARD unless an enemy is in view
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from action_mask import enemy_in_view, mask_actions
from features import Features
from model import Net, heads_to_buttons, load_compat, sample

KEYS_LONG = ("etype", "crop_idx", "pos_idx")
# Sampling temperature: logits / TEMP before sampling (1.0 = the trained distribution; < 1 sharper).
# Argmax (TEMP -> 0) was clearly worse (6 vs 13 of 50 maps: stuck against walls).
TEMP = 1.0
# Stuck escape (off unless enabled): when the agent has moved < UNSTUCK_DIST map units over the last
# UNSTUCK_WINDOW decisions and no enemy is in view, sample from logits / UNSTUCK_TEMP (a flatter version of
# the network's own distribution). The action still comes from the trained network.
import os  # noqa: E402
UNSTUCK = os.environ.get("DF_UNSTUCK", "0") == "1"
UNSTUCK_WINDOW, UNSTUCK_DIST, UNSTUCK_TEMP = 24, 48.0, float(os.environ.get("DF_UNSTUCK_TEMP", "1.8"))


class DecisionEngine:
    def __init__(self, weights: Path, device: torch.device):
        self.dev = device
        self.net = Net().to(device).eval()
        # names of tensors that were not loaded as-is (new layers start neutral: zero fuse/vec columns)
        self.reinitialised = None
        if Path(weights).exists():
            self.reinitialised = load_compat(self.net, torch.load(weights, map_location=device)["model"])
        self.feat = Features()
        self.reset()

    def reset(self) -> None:
        self.feat.reset()
        self.state = self.net.init_state(1, self.dev)
        self.prev = None
        self.trail = []

    @torch.no_grad()
    def act(self, observation: dict, mask_backward: bool = True, sampler=sample) -> np.ndarray:
        f, _ = self.feat.step(observation, self.prev)
        o = {k: torch.as_tensor(np.asarray(v)[None]).to(self.dev) for k, v in f.items()}
        o = {k: (v.long() if k in KEYS_LONG else v.float()) for k, v in o.items()}
        logits, _, _, self.state = self.net.step(o, self.state)
        t = TEMP
        if UNSTUCK:
            self.trail = (self.trail + [(self.feat.x, self.feat.y)])[-UNSTUCK_WINDOW:]
            if len(self.trail) == UNSTUCK_WINDOW and not enemy_in_view(observation):
                xs, ys = zip(*self.trail)
                if max(max(xs) - min(xs), max(ys) - min(ys)) < UNSTUCK_DIST:
                    t = UNSTUCK_TEMP
        idx, _, _ = sampler(logits / t if t != 1.0 else logits)
        a = heads_to_buttons(idx)[0].cpu().numpy().astype(np.int64)
        if mask_backward:
            a = mask_actions(a, observation)
        self.prev = a  # the executed action drives dead reckoning
        return a
