"""DeadReckon-EgoMap DOOM agent: every action is sampled from the trained network (see model.py)."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from features import Features  # noqa: E402
from model import Net, heads_to_buttons, sample  # noqa: E402

KEYS_LONG = ("etype", "crop_idx", "pos_idx")


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)
        self.dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.dev).eval()
        w = HERE / "weights" / "model.pt"
        if w.exists():
            self.net.load_state_dict(torch.load(w, map_location=self.dev)["model"])
        self.feat = Features()
        # warm-up (first torch call is slow; __init__ has 60 s, first act() only 10 s)
        self.reset()
        self.act(observation_space.sample())
        self.reset()

    def reset(self) -> None:
        self.feat.reset()
        self.state = self.net.init_state(1, self.dev)
        self.prev = None

    @torch.no_grad()
    def act(self, observation: dict) -> np.ndarray:
        f, _ = self.feat.step(observation, self.prev)
        o = {k: torch.as_tensor(np.asarray(v)[None]).to(self.dev) for k, v in f.items()}
        o = {k: (v.long() if k in KEYS_LONG else v.float()) for k, v in o.items()}
        logits, _, _, self.state = self.net.step(o, self.state)
        idx, _, _ = sample(logits)
        a = heads_to_buttons(idx)[0].cpu().numpy().astype(np.int64)
        self.prev = a
        return a
