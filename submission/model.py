"""DeadReckon-EgoMap agent network.

Encoders: circular ray conv | depth/vitals/prev-action MLP | entity self-attention (+ null token) |
hybrid EgoMap CNN with a K-step shared-weight value-iteration-style conv (learned planner).
Core: GRU. Heads: 6 factored action heads, value, DFP-style future-measurement aux, visited aux.
Learned memory: the core writes an 8-dim gated vector into its own map at the dead-reckoned cell;
those channels are read back with the fixed ones (Neural Map style, gradients flow through writes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import CROP, G, N_FIXED, VEC_DIM

torch.backends.mha.set_fastpath_enabled(False)  # fast path is slower for 33 tokens

N_LEARNED = 8
HEADS = (3, 3, 3, 2, 2, 2)  # move F/B/-, strafe L/R/-, turn L/R/-, attack, use, next weapon
N_AUX = 6 + 1  # d(health, kills, keys) at +8 and +32 steps; visited-before
# Second aux head (training-only targets from privileged state; Arnold's game-feature heads / DFP style):
# corridor distance to the current goal (1), direction of the shortest path relative to heading (8 bins),
# "exit reached within the next 64 steps" (1). Only trains the shared trunk; never used to act.
N_DIR = 8
N_AUX2 = 1 + N_DIR + 1
# Asymmetric critic (Pinto et al. 2018): during TRAINING the value head also reads privileged ground
# truth (distance to goal, goal kind, keys, health, time). The actor never sees it, and at evaluation
# o has no "priv" key, so the value head gets zeros (its output is not used to act).
PRIV_DIM = 8


def heads_to_buttons(idx: torch.Tensor) -> torch.Tensor:
    """(..., 6) head choices -> (..., 9) MultiBinary buttons."""
    b = torch.zeros(*idx.shape[:-1], 9, dtype=torch.int64, device=idx.device)
    for k, (i0, i1) in enumerate(((0, 1), (2, 3), (4, 5))):
        b[..., i0] = (idx[..., k] == 0).long()
        b[..., i1] = (idx[..., k] == 1).long()
    b[..., 6], b[..., 7], b[..., 8] = idx[..., 3], idx[..., 4], idx[..., 5]
    return b


class Net(nn.Module):
    def __init__(self, hid: int = 256, vin_k: int = 4):
        super().__init__()
        self.vin_k = vin_k
        self.rays = nn.Sequential(
            nn.Conv1d(7, 32, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 32, 256), nn.ReLU())
        self.vec = nn.Sequential(nn.Linear(VEC_DIM, 128), nn.ReLU())
        self.etype = nn.Embedding(12, 16)
        self.eproj = nn.Linear(6 + 16, 64)
        self.null = nn.Parameter(torch.zeros(1, 1, 64))
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True), 2, enable_nested_tensor=False)
        c = N_FIXED + N_LEARNED
        self.map_in = nn.Sequential(nn.Conv2d(c, 32, 3, stride=2, padding=1), nn.ReLU())  # 16x16
        self.vin = nn.Conv2d(64, 32, 3, padding=1)  # shared across K iterations
        self.map_out = nn.Sequential(nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # 8x8
                                     nn.Flatten(), nn.Linear(32 * 64, 256), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(256 + 128 + 64 + 256, hid), nn.ReLU())
        self.gru = nn.GRUCell(hid, hid)
        self.pi = nn.Linear(hid, sum(HEADS))
        self.vpriv = nn.Sequential(nn.Linear(PRIV_DIM, 64), nn.ReLU())
        self.v = nn.Linear(hid + 64, 1)
        self.aux = nn.Linear(hid, N_AUX)
        self.aux2 = nn.Linear(hid, N_AUX2)
        self.write = nn.Linear(hid, N_LEARNED + 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def init_state(self, b: int, device):
        return (torch.zeros(b, self.gru.hidden_size, device=device),
                torch.zeros(b, N_LEARNED, G * G + 1, device=device))

    def step(self, o: dict, state):
        """One time step for a batch. o: tensors from features.py (batched). Returns
        (logits, value, aux, new_state)."""
        h, L = state
        B = h.shape[0]
        learned = torch.gather(L, 2, o["crop_idx"].unsqueeze(1).expand(B, N_LEARNED, -1))
        m = torch.cat([o["map"].float(), learned.view(B, N_LEARNED, CROP, CROP)], 1)
        x = self.map_in(m)
        r = x
        for _ in range(self.vin_k):
            r = F.relu(self.vin(torch.cat([x, r], 1)))
        zmap = self.map_out(r)

        e = self.eproj(torch.cat([o["ents"], self.etype(o["etype"])], -1))
        e = torch.cat([self.null.expand(B, 1, 64), e], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=e.device), o["ents"][..., 5] < 0.5], 1)
        e = self.attn(e, src_key_padding_mask=pad)
        valid = (~pad).float().unsqueeze(-1)
        zent = (e * valid).sum(1) / valid.sum(1)

        z = self.fuse(torch.cat([self.rays(o["rays"]), self.vec(o["vec"]), zent, zmap], -1))
        h = self.gru(z, h)
        w = self.write(h)
        g = torch.sigmoid(w[:, :1])  # (B,1)
        pos = o["pos_idx"].view(B, 1, 1).expand(B, N_LEARNED, 1)
        old = torch.gather(L, 2, pos).squeeze(-1)
        new = (1 - g) * old + g * torch.tanh(w[:, 1:])
        L = L.scatter(2, pos, new.unsqueeze(-1))
        pv = self.vpriv(o["priv"]) if "priv" in o else h.new_zeros(B, 64)
        v = self.v(torch.cat([h, pv], -1)).squeeze(-1)
        return self.pi(h), v, torch.cat([self.aux(h), self.aux2(h)], -1), (h, L)


def load_compat(net: nn.Module, sd: dict) -> list[str]:
    """Load weights saved before/after the privileged-critic change. The old value head (1, hid) is
    copied into the first hid columns of the new one; missing new layers keep their init.
    Returns the names of tensors that were not loaded as-is."""
    own = net.state_dict()
    out, changed = {}, []
    for k, t in own.items():
        s = sd.get(k)
        if s is None:
            changed.append(k)
            out[k] = t
        elif s.shape == t.shape:
            out[k] = s
        elif k == "v.weight" and s.shape[0] == t.shape[0] and s.shape[1] < t.shape[1]:
            w = torch.zeros_like(t)
            w[:, :s.shape[1]] = s
            out[k] = w
            changed.append(k)
        else:
            changed.append(k)
            out[k] = t
    net.load_state_dict(out)
    return changed


def split_logits(logits: torch.Tensor):
    return torch.split(logits, HEADS, dim=-1)


def _head_stats(lg, a=None, greedy=False):
    lp = F.log_softmax(lg, -1)
    if a is None:
        if greedy:
            a = lg.argmax(-1)
        else:  # Gumbel-max sampling (no torch.distributions overhead)
            a = (lg - torch.log(-torch.log(torch.rand_like(lg).clamp_(1e-9, 1.0)))).argmax(-1)
    return a, lp.gather(-1, a.unsqueeze(-1)).squeeze(-1), -(lp.exp() * lp).sum(-1)


def sample(logits: torch.Tensor, greedy: bool = False):
    out = [_head_stats(lg, greedy=greedy) for lg in split_logits(logits)]
    return torch.stack([o[0] for o in out], -1), sum(o[1] for o in out), sum(o[2] for o in out)


def evaluate(logits: torch.Tensor, idx: torch.Tensor):
    out = [_head_stats(lg, idx[..., k]) for k, lg in enumerate(split_logits(logits))]
    return sum(o[1] for o in out), sum(o[2] for o in out)
