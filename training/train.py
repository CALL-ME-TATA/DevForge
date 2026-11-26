"""Recurrent PPO with a hybrid learned EgoMap, auxiliary heads, curiosity and a PLR-lite curriculum.

Run (Linux container, see README):  python training/train.py --envs 16 --hours 8
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(ROOT / "env")]

FAMILY_TICKS = {"tiny": 1050, "arena": 2100, "multi": 3150, "key": 4200, "gauntlet": 4200, "freedoom": 4200}
LEVELS = 5
EPISODES_PER_MAP = 8


# ------------------------------------------------------------------ curriculum (per worker)
HARD_FAMILIES = ("multi", "key", "gauntlet")


class Curriculum:
    def __init__(self, rng, hard_floor: float = 0.05, key_gate: float = 0.5):
        self.rng = rng
        self.p = defaultdict(float)  # EMA exit rate per bin
        # PLR-style floor (Jiang et al. 2021): unlocked-but-unsolved navigation bins keep being revisited
        self.hard_floor = hard_floor
        self.key_gate = key_gate  # multi0 exit rate needed before key0 unlocks

    def eligible(self):
        p = self.p
        out = [("tiny", 0)]
        if p[("tiny", 0)] > 0.6:
            out += [("arena", 0), ("multi", 0)]
        for fam in ("arena", "multi", "key", "gauntlet"):
            for li in range(1, LEVELS):
                if p[(fam, li - 1)] > 0.5:
                    out.append((fam, li))
        if p[("multi", 0)] > self.key_gate:
            out.append(("key", 0))
        if p[("key", 0)] > 0.5:
            out.append(("gauntlet", 0))
        if p[("key", 1)] > 0.3:
            out.append(("freedoom", 0))
        return sorted(set(out))

    def sample(self):
        bins = self.eligible()
        w = [self.p[b] * (1 - self.p[b]) + (self.hard_floor if b[0] in HARD_FAMILIES else 0.05) for b in bins]
        return self.rng.choices(bins, w)[0]

    def update(self, b, success):
        self.p[b] = 0.9 * self.p[b] + 0.1 * float(success)


# ------------------------------------------------------------------ environment worker
def cur_to_plain(p):
    """Curriculum EMA dict -> {"fam|level": rate} (plain types, loads with torch weights_only)."""
    return {f"{f}|{li}": float(v) for (f, li), v in p.items()}


# reward / map-pool settings (CLI flags in main(); defaults = original behaviour)
DEFAULT_CFG = dict(maps_per_bin=40, dmg_pen=0.01, death_pen=3.0, cell_rew=0.01,
                   shape_k=0.0, hard_floor=0.05, key_gate=0.5, weapon_pen=0.0, shot_pen=0.0, door_use=0.0)

KEY_IDX = {"B": 0, "Y": 1, "R": 2}  # lock colour -> doom_env KEY_NAMES index (Blue/Yellow/Red card)
KEY_THING_COLOUR = {13: "R", 5: "B", 6: "Y"}  # mapgen KEY_THING inverse


class GoalShaper:
    """TRAINING-ONLY reward scaffolding: potential-based shaping (Ng, Harada & Russell 1999).

    Uses the generator's own layout of the current training map (privileged, training time only;
    the submitted agent never sees or computes any of this). Potential = -k * corridor distance
    (4-connected BFS over the 64-unit grid) to the current goal: the exit if it is reachable with
    the keys already held, otherwise the nearest still-needed reachable key. Locked doors are
    passable only with their key. Shaping reward = phi(s') - phi(s), so walking back and forth nets 0.
    """

    def __init__(self, grid, things):
        self.g = grid
        self.H, self.W = len(grid), max(len(r) for r in grid)
        self.exit = [(y, x) for y, row in enumerate(grid) for x, c in enumerate(row) if c == "E"]
        self.keys = {KEY_THING_COLOUR[t]: (y, x) for (y, x, t, _) in things if t in KEY_THING_COLOUR}
        self.cache = {}

    def _bfs(self, targets, held):
        INF = 1 << 30
        dist = [[INF] * self.W for _ in range(self.H)]
        q = []
        for y, x in targets:
            dist[y][x] = 0
            q.append((y, x))
        i = 0
        while i < len(q):
            y, x = q[i]
            i += 1
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < self.H and 0 <= nx < len(self.g[ny]) and dist[ny][nx] == INF:
                    c = self.g[ny][nx]
                    if c == "#" or (c in KEY_IDX and c not in held):
                        continue
                    dist[ny][nx] = dist[y][x] + 1
                    q.append((ny, nx))
        return dist

    def dist(self, x: float, y: float, held: frozenset):
        """(goal id, corridor distance in cells) or None if unknown/unreachable."""
        if held not in self.cache:
            need = [p for c, p in self.keys.items() if c not in held]
            self.cache[held] = (self._bfs(self.exit, held), self._bfs(need, held) if need else None)
        d_exit, d_key = self.cache[held]
        r, c = math.floor(-y / 64.0), math.floor(x / 64.0)
        if not (0 <= r < self.H and 0 <= c < len(self.g[r])):
            return None
        if d_exit[r][c] < (1 << 30):
            return ("exit", held), d_exit[r][c]
        if d_key is not None and d_key[r][c] < (1 << 30):
            return ("key", held), d_key[r][c]
        return None


def worker(conn, wid: int, map_dir: str, seed: int, cur0=None, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    os.environ["OMP_NUM_THREADS"] = "1"
    from doom_env import make_env
    from doom_env.wrapper import IWAD
    from features import Features

    from mapgen import FAMILIES, generate

    rng = random.Random(seed)
    cur = Curriculum(rng, hard_floor=cfg["hard_floor"], key_gate=cfg["key_gate"])
    if cur0:  # restore unlocked bins from the checkpoint so a resume does not restart at "tiny"
        for k, v in cur0.items():
            f, li = k.split("|")
            cur.p[(f, int(li))] = float(v)
    feat = Features()
    env = None
    S = {}

    eps_left = 0

    def start():
        # creating a DoomGame costs ~0.4 s (vs 0.01 s reset) and stalls the whole batch,
        # so each sampled map is replayed for EPISODES_PER_MAP episodes with new seeds
        nonlocal env, eps_left
        if env is None or eps_left == 0:
            if env is not None:
                env.close()
            b = cur.sample()
            fam, li = b
            S["shaper"] = None
            if fam == "freedoom":
                wad, mp_ = IWAD, f"MAP{rng.randint(1, 24):02d}"
            else:
                mi = rng.randrange(cfg["maps_per_bin"])
                wad, mp_ = os.path.join(map_dir, f"{fam}_{li}_{mi}.wad"), "MAP01"
                if cfg["shape_k"] > 0:  # rebuild this map's layout from its generator seed (same as build_pool)
                    grid, things = generate(fam, li / (LEVELS - 1), FAMILIES.index(fam) * 1_000_000 + li * 10_000 + mi)
                    S["shaper"] = GoalShaper(grid, things)
            env = make_env(wad, mp_, episode_ticks=FAMILY_TICKS[fam])
            S["bin"], eps_left = b, EPISODES_PER_MAP
        eps_left -= 1
        b = S["bin"]
        obs, _ = env.reset(seed=rng.randrange(1 << 30))
        feat.reset()
        f, _ = feat.step(obs, None)
        ps = env.privileged_state()
        S.update(bin=b, ps=ps, visited={(int(ps["x"] // 64), int(ps["y"] // 64))}, opened=set(),
                 last_use=-99, t=0, ret=0.0, use_prev=0, phi=None, goal=None, shaped=0.0, obs=obs)
        return f, ps

    f, ps = start()
    conn.send(f)
    while True:
        cmd, a = conn.recv()
        if cmd == "cur":
            conn.send(cur_to_plain(cur.p))
            continue
        if cmd == "close":
            env.close()
            return
        obs_prev = S["obs"]  # what the agent saw when it chose `a`
        obs, _, term, trunc, info = env.step(a)
        S["obs"] = obs
        p0, p1 = S["ps"], env.privileged_state()
        S["t"] += 1
        r = -0.001
        # --- edge-case fixes (training-time reward only; see README "Edge cases") ---
        vit = obs_prev["vitals"]
        if cfg["weapon_pen"] > 0:
            if a[8] and vit[3] > 0:  # switched away from a weapon that still had ammo
                r -= cfg["weapon_pen"]
            if vit[3] <= 0 and (vit[4] > 0 or vit[5] > 0):  # holding fists/empty gun while bullets/shells exist
                r -= 0.2 * cfg["weapon_pen"]
        if cfg["shot_pen"] > 0 and a[6]:  # firing with no enemy in view wastes ammo
            ve = obs_prev["visible_entities"]
            if not np.isin(ve["entity_type"][ve["mask"].astype(bool)], (1, 2, 3)).any():
                r -= cfg["shot_pen"]
        r += 1.0 * (p1["kills"] - p0["kills"])
        r -= cfg["dmg_pen"] * max(0.0, p1["damage_taken"] - p0["damage_taken"])
        r += 3.0 * float(p1["keys"].sum() - p0["keys"].sum())
        r += 0.2 * max(0, p1["items"] - p0["items"])
        lock, fl, ce = p1["door_lock"], p1["sector_floor"], p1["sector_ceiling"]
        for s in np.nonzero((lock >= 0) & (ce - fl > 40))[0]:
            if s not in S["opened"]:
                S["opened"].add(s)
                r += 1.0 + 2.0 * (lock[s] > 0)
        cell = (int(p1["x"] // 64), int(p1["y"] // 64))
        visited_before = cell in S["visited"]
        if not visited_before:
            S["visited"].add(cell)
            r += cfg["cell_rew"]
        if a[7] and not S["use_prev"]:
            if S["t"] - S["last_use"] < 8:
                r -= 0.02  # re-pressing USE bounces a moving door
            elif cfg["door_use"] > 0 and S.get("shaper") is not None:  # fresh USE right next to a door
                g_ = S["shaper"].g
                rr, cc = math.floor(-p0["y"] / 64.0), math.floor(p0["x"] / 64.0)
                if any(0 <= rr + dy < len(g_) and 0 <= cc + dx < len(g_[rr + dy]) and g_[rr + dy][cc + dx] in "DRBY"
                       for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    r += cfg["door_use"]
            S["last_use"] = S["t"]
        S["use_prev"] = int(a[7])
        done = term or trunc
        if S.get("shaper") is not None and not done:
            held = frozenset(c for c, i in KEY_IDX.items() if p1["keys"][i] > 0)
            gd = S["shaper"].dist(p1["x"], p1["y"], held)
            if gd is not None:
                goal, dcell = gd
                phi = -cfg["shape_k"] * dcell
                if S["phi"] is not None and S["goal"] == goal:  # no shaping on goal switches (key pickup)
                    r += phi - S["phi"]
                    S["shaped"] += phi - S["phi"]
                S["phi"], S["goal"] = phi, goal
        if info["exit_reached"]:
            r += 10.0 + 2.0 * p1["health"] / 100 + 2.0 * (1 - p1["tic"] / p1["episode_ticks"])
        elif term:
            r -= cfg["death_pen"]
        f_next, new_cell = feat.step(obs, a)
        r += 0.02 * new_cell  # agent-internal episodic curiosity (own dead-reckoned map)
        S["ret"] += r
        aux = np.array([p1["health"] / 100, p1["kills"], p1["keys"].sum(), visited_before], np.float32)
        ep = None
        S["ps"] = p1
        if done:
            cur.update(S["bin"], info["exit_reached"])
            ep = dict(bin=S["bin"], exit=info["exit_reached"], ret=S["ret"], kills=p1["kills"],
                      keys=float(p1["keys"].sum()), cells=len(S["visited"]), dead=p1["dead"])
            f_next, _ = start()
        conn.send((f_next, r, done, aux, ep))


class Envs:
    def __init__(self, n, map_dir, seed, cur0=None, cfg=None):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            c0 = cur0[i % len(cur0)] if cur0 else None
            p = ctx.Process(target=worker, args=(b, i, map_dir, seed * 1000 + i, c0, cfg), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.n = n
        self.first = [c.recv() for c in self.conns]

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", a))
        return [c.recv() for c in self.conns]

    def curricula(self):
        """Per-worker curriculum state (called between steps, so workers are idle)."""
        for c in self.conns:
            c.send(("cur", None))
        return [c.recv() for c in self.conns]

    def close(self):
        for c in self.conns:
            c.send(("close", None))


# ------------------------------------------------------------------ helpers
KEYS = ("vec", "rays", "ents", "etype", "map", "crop_idx", "pos_idx")


def stack(fs):
    return {k: np.stack([f[k] for f in fs]) for k in KEYS}


def to_t(d, dev, sl=None):
    import torch
    out = {}
    for k in KEYS:
        v = d[k] if sl is None else d[k][sl]
        t = torch.from_numpy(np.ascontiguousarray(v)).to(dev, non_blocking=True)
        out[k] = t.long() if k in ("etype", "crop_idx", "pos_idx") else t.float()
    return out


def aux_targets(aux, dones, ks=(8, 32)):
    """aux: (T,N,4) [health, kills, keys, visited]; dones: (T,N). Future deltas within the episode."""
    T, N, _ = aux.shape
    end = np.empty((T, N), np.int64)
    nxt = np.full(N, T - 1)
    for t in range(T - 1, -1, -1):
        nxt = np.where(dones[t], t, nxt)
        end[t] = nxt
    tgt = []
    for k in ks:
        idx = np.minimum(np.arange(T)[:, None] + k, end)
        fut = np.take_along_axis(aux[..., :3], idx[..., None], 0)
        tgt.append(fut - aux[..., :3])
    return np.concatenate(tgt + [aux[..., 3:4]], -1).astype(np.float32)  # (T,N,7)


class GraphPolicy:
    """Rollout policy step captured as a CUDA graph (kernel-launch latency 9 ms -> 3 ms).
    Holds the recurrent state (h, learned map L) for all envs; parameters are updated in place."""

    def __init__(self, net, obs, dev):
        import torch
        from model import sample
        self.dev = dev
        self.o = to_t(obs, dev)
        self.m = torch.ones(len(obs["vec"]), device=dev)
        self.h, self.L = net.init_state(len(obs["vec"]), dev)

        def f():
            keep = 1 - self.m
            lg, v, _, (h2, L2) = net.step(self.o, (self.h * keep[:, None], self.L * keep[:, None, None]))
            idx, lp, _ = sample(lg)
            return idx, lp, v, h2, L2

        self.graph = None
        if dev.type == "cuda":
            with torch.no_grad():
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        f()
                torch.cuda.current_stream().wait_stream(s)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.out = f()
        self.f = f

    def __call__(self, obs, starts):
        import torch
        for k, v in to_t(obs, self.dev).items():
            self.o[k].copy_(v)
        self.m.copy_(torch.from_numpy(starts.astype(np.float32)))
        if self.graph is not None:
            self.graph.replay()
            idx, lp, v, h2, L2 = self.out
        else:
            idx, lp, v, h2, L2 = self.f()
        self.h.copy_(h2)
        self.L.copy_(L2)
        return idx.clone(), lp.clone(), v.clone()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--mb", type=int, default=64, help="sequences per minibatch")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--maps", default=str(ROOT / "training" / "maps_gen"))
    ap.add_argument("--out", default=str(ROOT / "training" / "ckpt"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=0, help="stop after N iterations (smoke test)")
    ap.add_argument("--maps-per-bin", type=int, default=DEFAULT_CFG["maps_per_bin"],
                    help="layouts per (family, level) bin to sample from (grow pool: training/extend_maps.py)")
    ap.add_argument("--dmg-pen", type=float, default=DEFAULT_CFG["dmg_pen"], help="penalty per HP of damage")
    ap.add_argument("--death-pen", type=float, default=DEFAULT_CFG["death_pen"], help="penalty for dying")
    ap.add_argument("--cell-rew", type=float, default=DEFAULT_CFG["cell_rew"], help="reward per new 64u cell")
    ap.add_argument("--shape-k", type=float, default=DEFAULT_CFG["shape_k"],
                    help="training-only potential shaping: reward per cell of corridor distance gained toward "
                         "the current goal (key, then exit); 0 = off")
    ap.add_argument("--hard-floor", type=float, default=DEFAULT_CFG["hard_floor"],
                    help="minimum curriculum weight for unlocked multi/key/gauntlet bins")
    ap.add_argument("--key-gate", type=float, default=DEFAULT_CFG["key_gate"],
                    help="multi0 exit rate needed before key0 unlocks")
    ap.add_argument("--weapon-pen", type=float, default=DEFAULT_CFG["weapon_pen"],
                    help="penalty for switching away from a loaded weapon (x0.2 per step holding fists/empty "
                         "while ammo exists)")
    ap.add_argument("--shot-pen", type=float, default=DEFAULT_CFG["shot_pen"],
                    help="penalty per shot fired with no enemy in view")
    ap.add_argument("--door-use", type=float, default=DEFAULT_CFG["door_use"],
                    help="reward for a fresh USE press next to a door (generated maps)")
    args = ap.parse_args()

    import torch
    from model import Net, heads_to_buttons, sample, evaluate
    from mapgen import build_pool

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(maps_per_bin=args.maps_per_bin, dmg_pen=args.dmg_pen, death_pen=args.death_pen,
               cell_rew=args.cell_rew, shape_k=args.shape_k, hard_floor=args.hard_floor, key_gate=args.key_gate,
               weapon_pen=args.weapon_pen, shot_pen=args.shot_pen, door_use=args.door_use)
    print(f"config {cfg}", flush=True)
    if not Path(args.maps, f"gauntlet_4_{args.maps_per_bin - 1}.wad").exists():
        print("generating maps ...", flush=True)
        build_pool(args.maps, per_bin=args.maps_per_bin, levels=tuple(i / (LEVELS - 1) for i in range(LEVELS)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    it0, steps, cur0 = 0, 0, None
    if args.resume and (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location=dev)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        for g_ in opt.param_groups:  # the CLI --lr wins over the lr stored in the checkpoint
            g_["lr"] = args.lr
        it0, steps = ck["iter"], ck["steps"]
        cur0 = ck.get("cur")
        print(f"resumed at iter {it0}, {steps} steps, lr {args.lr}, ent {args.ent}, "
              f"curriculum {'restored' if cur0 else 'fresh'}", flush=True)

    N, T, C = args.envs, args.T, args.chunk
    nch = T // C
    envs = Envs(N, args.maps, args.seed, cur0, cfg)
    obs = stack(envs.first)
    starts = np.ones(N, bool)
    pol = GraphPolicy(net, obs, dev)
    t_end = time.time() + args.hours * 3600
    eps_log = []
    it = it0
    while time.time() < t_end and (args.iters == 0 or it - it0 < args.iters):
        t0 = time.time()
        buf = {k: [] for k in KEYS}
        acts, logps, vals, rews, dones, auxs, strt = [], [], [], [], [], [], []
        h0 = torch.empty(nch, N, net.gru.hidden_size)
        L0 = torch.empty(nch, N, *pol.L.shape[1:], dtype=torch.float16)
        net.eval()
        with torch.no_grad():
            for t in range(T):
                if t % C == 0:
                    m = torch.from_numpy(starts).to(dev).float()
                    h0[t // C] = (pol.h * (1 - m)[:, None]).cpu()
                    L0[t // C] = (pol.L * (1 - m)[:, None, None]).half().cpu()
                idx, lp, v = pol(obs, starts)
                btn = heads_to_buttons(idx).cpu().numpy()
                res = envs.step(list(btn))
                for k in KEYS:
                    buf[k].append(obs[k])
                strt.append(starts.copy())
                acts.append(idx.cpu().numpy())
                logps.append(lp.cpu().numpy())
                vals.append(v.cpu().numpy())
                rews.append([x[1] for x in res])
                d = np.array([x[2] for x in res])
                dones.append(d)
                auxs.append(np.stack([x[3] for x in res]))
                eps_log += [x[4] for x in res if x[4] is not None]
                obs = stack([x[0] for x in res])
                starts = d
            h_keep, L_keep = pol.h.clone(), pol.L.clone()
            _, _, v_last = pol(obs, starts)  # bootstrap value; restore state (action is re-sampled next iter)
            pol.h.copy_(h_keep); pol.L.copy_(L_keep)
        t_roll = time.time() - t0

        rews, dones, vals = np.array(rews, np.float32), np.array(dones), np.array(vals)
        v_last = v_last.cpu().numpy()
        adv = np.zeros_like(rews)
        last = 0
        for t in range(T - 1, -1, -1):
            nv = v_last if t == T - 1 else vals[t + 1]
            nd = 1.0 - dones[t]
            delta = rews[t] + args.gamma * nv * nd - vals[t]
            last = delta + args.gamma * args.lam * nd * last
            adv[t] = last
        ret = adv + vals
        aux_t = aux_targets(np.array(auxs), dones)
        data = {k: np.array(v) for k, v in buf.items()}  # (T,N,...)
        acts, logps, strt = np.array(acts), np.array(logps), np.array(strt)

        # ---- PPO update over (chunk, env) sequences
        net.train()
        seqs = [(c, n) for c in range(nch) for n in range(N)]
        stats = defaultdict(list)
        for _ in range(args.epochs):
            random.shuffle(seqs)
            for i in range(0, len(seqs), args.mb):
                mbs = seqs[i:i + args.mb]
                cs, ns = np.array([s[0] for s in mbs]), np.array([s[1] for s in mbs])
                h = h0[cs, ns].to(dev)
                L = L0[cs, ns].to(dev).float()
                st = (h, L)
                lp_l, ent_l, v_l, aux_l = [], [], [], []
                for k in range(C):
                    ts = cs * C + k
                    sl = (ts, ns)
                    mk = torch.from_numpy(strt[ts, ns]).to(dev).float()
                    st = (st[0] * (1 - mk)[:, None], st[1] * (1 - mk)[:, None, None])
                    logits, v, aux, st = net.step(to_t(data, dev, sl), st)
                    lp, en = evaluate(logits, torch.from_numpy(acts[ts, ns]).to(dev))
                    lp_l.append(lp); ent_l.append(en); v_l.append(v); aux_l.append(aux)
                ts_all = (cs[None, :] * C + np.arange(C)[:, None])  # (C,B)
                ns_all = np.broadcast_to(ns[None, :], ts_all.shape)
                g = lambda x: torch.from_numpy(np.ascontiguousarray(x[ts_all, ns_all])).to(dev)
                lp, en, v, aux = torch.stack(lp_l), torch.stack(ent_l), torch.stack(v_l), torch.stack(aux_l)
                A = g(adv)
                A = (A - A.mean()) / (A.std() + 1e-8)
                ratio = torch.exp(lp - g(logps))
                pg = -torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A).mean()
                vl = 0.5 * ((v - g(ret)) ** 2).mean()
                at = g(aux_t)
                al = ((aux[..., :6] - at[..., :6]) ** 2).mean() + \
                    torch.nn.functional.binary_cross_entropy_with_logits(aux[..., 6], at[..., 6])
                loss = pg + vl - args.ent * en.mean() + 0.1 * al
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats["pg"].append(pg.item()); stats["v"].append(vl.item())
                stats["aux"].append(al.item()); stats["ent"].append(en.mean().item())
                stats["kl"].append((g(logps) - lp).mean().item())
        it += 1
        steps += N * T
        dt = time.time() - t0
        recent = eps_log[-200:]
        fam = defaultdict(list)
        for e in recent:
            fam[f"{e['bin'][0]}{e['bin'][1]}"].append(e["exit"])
        fs = " ".join(f"{k}:{np.mean(v):.2f}({len(v)})" for k, v in sorted(fam.items()))
        print(f"it {it} steps {steps/1e6:.2f}M sps {N*T/dt:.0f} (roll {t_roll:.1f}s, upd {dt-t_roll:.1f}s) "
              f"ret {np.mean([e['ret'] for e in recent]) if recent else 0:.2f} "
              + " ".join(f"{k} {np.mean(v):.3f}" for k, v in stats.items()) + f" | exit {fs}", flush=True)
        eps_log = recent
        if it % 10 == 0 or time.time() >= t_end:
            ck = {"model": net.state_dict(), "opt": opt.state_dict(), "iter": it, "steps": steps,
                  "cur": envs.curricula()}
            torch.save(ck, out / "latest.pt.tmp")
            os.replace(out / "latest.pt.tmp", out / "latest.pt")
            if it % 50 == 0:
                torch.save({"model": net.state_dict()}, out / f"it{it}.pt")
    envs.close()


if __name__ == "__main__":
    main()
