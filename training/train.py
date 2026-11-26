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
import vizdoom as vd

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
        # Level 0 of EVERY family (and Freedoom) is always in the pool: hard gates (key0 behind multi0,
        # gauntlet behind key0) meant the agent never saw a single Tier-4 map in 9M steps.
        p = self.p
        out = [(fam, 0) for fam in ("tiny", "arena", "multi", "key", "gauntlet")] + [("freedoom", 0)]
        for fam in ("arena", "multi", "key", "gauntlet"):
            for li in range(1, LEVELS):
                if p[(fam, li - 1)] > 0.4:
                    out.append((fam, li))
        return sorted(set(out))

    def floor(self, b):
        if b[0] == "gauntlet":
            return 1.5 * self.hard_floor  # Tier 4 carries the most score
        if b[0] in HARD_FAMILIES:
            return self.hard_floor
        return 0.1 if b[0] == "freedoom" else 0.05

    def sample(self):
        bins = self.eligible()
        w = [self.p[b] * (1 - self.p[b]) + self.floor(b) for b in bins]
        return self.rng.choices(bins, w)[0]

    def update(self, b, success):
        self.p[b] = 0.9 * self.p[b] + 0.1 * float(success)


# ------------------------------------------------------------------ environment worker
def cur_to_plain(p):
    """Curriculum EMA dict -> {"fam|level": rate} (plain types, loads with torch weights_only)."""
    return {f"{f}|{li}": float(v) for (f, li), v in p.items()}


# reward / map-pool settings (CLI flags in main(); defaults = original behaviour)
DEFAULT_CFG = dict(maps_per_bin=40, dmg_pen=0.01, death_pen=3.0, cell_rew=0.01,
                   shape_k=0.0, hard_floor=0.05, key_gate=0.5, weapon_pen=0.0, shot_pen=0.0, door_use=0.0,
                   rev_p=0.0, assist=0.0, assist_until=0.6, edge_p=0.0, edge_every=32, edge_dir="",
                   back_pen=0.0, turn_pen=0.0, use_bounce=0.02, hit_rew=0.0, aim_k=0.0, mask_back=0,
                   dagger=0, dagger_beta=0.0)

KEY_IDX = {"B": 0, "Y": 1, "R": 2}  # lock colour -> doom_env KEY_NAMES index (Blue/Yellow/Red card)
KEY_THING_COLOUR = {13: "R", 5: "B", 6: "Y"}  # mapgen KEY_THING inverse
MONSTER_THINGS = (3004, 9, 3001, 3002)  # mapgen MONSTERS

# ------------------------------------------------------------------ reverse curriculum (training only)
# Florensa et al. 2017 / Go-Explore's backward algorithm: on long key -> door -> exit maps, start some
# episodes part-way along the level's solution route and move the start back as the agent succeeds.
# REV_FRACS = fraction of the route still ahead of the agent; the full start (1.0) is the normal map.
REV_FAMILIES = ("multi", "key", "gauntlet")
REV_FRACS = (0.1, 0.25, 0.45, 0.7)


class RevCurriculum:
    def __init__(self, rng):
        self.rng, self.p = rng, defaultdict(float)

    def sample(self, fam):
        unlocked = [0] + [k for k in range(1, len(REV_FRACS)) if self.p[(fam, k - 1)] > 0.5]
        top = max(unlocked)
        return top if self.rng.random() < 0.6 else self.rng.choice(unlocked)

    def update(self, fam, lvl, success):
        self.p[(fam, lvl)] = 0.9 * self.p[(fam, lvl)] + 0.1 * float(success)


def solution_route(grid, things):
    """[(cell, keys held)] from the player start to the exit, following the GoalShaper fields
    (nearest needed key first, then the exit). Training-time map preparation only."""
    sh = GoalShaper(grid, things)
    pos = next(((y, x) for y, x, t, _ in things if t == 1), None)
    if pos is None:
        return None
    held, route, INF = frozenset(), [], 1 << 30
    for _ in range(5000):
        route.append((pos, held))
        if grid[pos[0]][pos[1]] == "E":
            return route
        sh.dist(pos[1] * 64 + 32, -(pos[0] * 64 + 32), held)
        d_exit, d_key = sh.cache[held]
        y, x = pos
        field = d_exit if d_exit[y][x] < INF else d_key
        if field is None or field[y][x] >= INF:
            return None
        nxt = next(((ny, nx) for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1))
                    if 0 <= ny < sh.H and 0 <= nx < len(grid[ny]) and field[ny][nx] == field[y][x] - 1), None)
        if nxt is None:
            return None
        pos = nxt
        held = held | {c for c, kp in sh.keys.items() if kp == pos}
    return None


def map_variant(grid, things, frac, assist, rng):
    """Move the player start to `frac` of the solution route from the end (keys already passed are put on
    the new start cell), and remove a fraction `assist` of monsters (fading combat assist)."""
    things = list(things)
    if frac < 1.0:
        route = solution_route(grid, things)
        if route and len(route) > 3:
            i = min(len(route) - 2, int(len(route) * (1 - frac)))
            (sy, sx), held = route[i]
            new = []
            for y, x, t, a in things:
                if t == 1:
                    new.append((sy, sx, 1, rng.choice(range(0, 360, 45))))
                elif t in KEY_THING_COLOUR and KEY_THING_COLOUR[t] in held:
                    new.append((sy, sx, t, 0))
                else:
                    new.append((y, x, t, a))
            things = new
    sy, sx = next((y, x) for y, x, t, _ in things if t == 1)
    return [th for th in things if not (th[2] in MONSTER_THINGS and
                                        (rng.random() < assist or abs(th[0] - sy) + abs(th[1] - sx) < 3))]


# ------------------------------------------------------------------ edge-case mining (training only)
# The engine itself finds the edge cases: every worker snapshots the full game state (vizdoom save) every
# `edge_every` steps. When an episode fails, the failure is classified (died, stuck, stalled at a door,
# reached the goal cell without finishing, lost after progress) and the snapshot from just BEFORE the
# failure is kept as a case. The learner keeps a bank of cases and sends them back to workers, which start
# a fraction `edge_p` of episodes by loading a case's exact state (monsters, health, ammo, doors, keys,
# clock). Cases the agent now solves are retired; this is prioritized replay of hard *states*, not just
# hard levels (PLR, Jiang et al. 2021; Go-Explore "return then explore", Ecoffet et al. 2021).
STILL_STEPS = 120     # longest stand-still (moved < 1 cell) that counts as "stuck"
PLATEAU_STEPS = 150   # steps after the closest approach to the goal with no further progress
DEATH_MARGIN = 40     # replay a death from a snapshot at least this many steps before it


def still_window(trace):
    """(length, start index) of the longest window in which the agent stayed within one 64u cell."""
    best, best_i, i0 = 0, 0, 0
    for j in range(len(trace)):
        while i0 < j and max(abs(trace[j][0] - trace[i0][0]), abs(trace[j][1] - trace[i0][1])) > 64:
            i0 += 1
        if j - i0 > best:
            best, best_i = j - i0, i0
    return best, best_i


def blocked_by(grid, x, y):
    """'door' / 'locked' if a door cell is next to world position (x, y) on the generator grid."""
    r, c = math.floor(-y / 64.0), math.floor(x / 64.0)
    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]) and grid[nr][nc] in "DRBY":
            return "door" if grid[nr][nc] == "D" else "locked"
    return None


def pick_case(trace, snaps, dead, exit_, grid=None):
    """Classify a failed episode and choose the snapshot to replay. trace: [(x, y, goal dist or None)] per
    step; snaps: [{"t": step, ...}] in order. Returns (kind, snap) or None."""
    if exit_ or not snaps:
        return None
    T = len(trace)

    def before(t):
        c = [s for s in snaps if s["t"] <= t]
        return c[-1] if c else None

    if dead:
        s = before(T - DEATH_MARGIN)
        return ("death", s) if s else None
    n, i0 = still_window(trace)
    if n >= STILL_STEPS:
        s = before(i0 + 8)
        if s:
            return "stuck", s
    ds = [(t[2], j) for j, t in enumerate(trace) if t[2] is not None]
    if ds:
        dmin, j = min(ds)
        if dmin < ds[0][0] and T - j >= PLATEAU_STEPS:  # made progress, then stopped making any
            s = before(j)
            if s:
                kind = "goal_cell" if dmin <= 1 else \
                    (blocked_by(grid, trace[j][0], trace[j][1]) if grid else None) or "plateau"
                return kind, s
    return None


class EdgeBank:
    """Learner-side bank of mined edge cases with PLR-style replay priorities."""

    def __init__(self, store: Path, cap: int = 400, state=None):
        self.store, self.cap = store, cap
        self.cases, self.inflight = {}, set()
        self.stats = defaultdict(int)  # mined:<kind>, mined_fam:<fam>, replays, solved, retired:<why>
        if state:
            self.cases = {c["id"]: c for c in state["cases"] if Path(c["sav"]).exists()}
            self.stats.update(state["stats"])

    @staticmethod
    def score(c):
        h = c["hist"][-4:]
        return 2.0 if not h else 1.05 - sum(h) / len(h)  # unseen first, then the ones still failing

    def add(self, c, it):
        c.update(hist=[], tries=0, born=it)
        self.cases[c["id"]] = c
        self.stats["mined:" + c["kind"]] += 1
        self.stats["mined_fam:" + c["fam"]] += 1
        while len(self.cases) > self.cap:  # evict the oldest of the least useful
            free = [k for k in self.cases if k not in self.inflight] or list(self.cases)
            self._drop(min(free, key=lambda k: (self.score(self.cases[k]), -self.cases[k]["born"])), "evicted")

    def result(self, cid, success):
        self.inflight.discard(cid)
        self.stats["replays"] += 1
        self.stats["solved"] += int(success)
        c = self.cases.get(cid)
        if c is None:
            return
        c["hist"].append(int(success))
        c["tries"] += 1
        if len(c["hist"]) >= 3 and sum(c["hist"][-3:]) == 3:
            self._drop(cid, "learned")
        elif c["tries"] >= 8 and sum(c["hist"]) == 0:
            self._drop(cid, "hopeless")

    def _drop(self, cid, why):
        c = self.cases.pop(cid)
        self.inflight.discard(cid)
        self.stats["retired:" + why] += 1
        for k in ("sav", "own_wad"):
            if c.get(k):
                Path(c[k]).unlink(missing_ok=True)

    def take(self, rng, n):
        pool = [c for k, c in self.cases.items() if k not in self.inflight]
        out = []
        for _ in range(min(n, len(pool))):
            c = rng.choices(pool, [self.score(x) for x in pool])[0]
            pool.remove(c)
            self.inflight.add(c["id"])
            out.append(c)
        return out

    def state(self):
        return {"cases": list(self.cases.values()), "stats": dict(self.stats)}

    def report(self, it):
        open_ = defaultdict(int)
        for c in self.cases.values():
            open_[f"{c['kind']}@{c['fam']}{c['li']}"] += 1
        s = self.stats
        return {"iter": it, "bank": len(self.cases),
                "mined_by_kind": {k[6:]: v for k, v in s.items() if k.startswith("mined:")},
                "mined_by_family": {k[10:]: v for k, v in s.items() if k.startswith("mined_fam:")},
                "replays": s["replays"], "solved": s["solved"],
                "retired": {k[8:]: v for k, v in s.items() if k.startswith("retired:")},
                "open_by_kind_and_level": dict(sorted(open_.items(), key=lambda kv: -kv[1]))}


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

    def path_dir(self, x: float, y: float, angle_deg: float, held: frozenset, k: int = 3, bins: int = 8):
        """Aux-head TARGET only: direction (bin 0 = straight ahead, counter-clockwise) from the agent to the
        cell k steps down the shortest corridor path to the current goal, or None."""
        gd = self.dist(x, y, held)
        if gd is None or gd[1] == 0:
            return None
        d_exit, d_key = self.cache[held]
        r, c = math.floor(-y / 64.0), math.floor(x / 64.0)
        field = d_exit if gd[0][0] == "exit" else d_key
        for _ in range(min(k, gd[1])):
            nxt = next(((nr, nc) for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
                        if 0 <= nr < self.H and 0 <= nc < len(self.g[nr]) and field[nr][nc] == field[r][c] - 1), None)
            if nxt is None:
                break
            r, c = nxt
        rel = math.atan2(-(r + 0.5) * 64 - y, (c + 0.5) * 64 - x) - math.radians(angle_deg)
        return int(round(rel / (2 * math.pi / bins))) % bins


def worker(conn, wid: int, map_dir: str, seed: int, cur0=None, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    os.environ["OMP_NUM_THREADS"] = "1"
    from doom_env import make_env
    from doom_env.wrapper import IWAD
    from features import Features
    from action_mask import mask_actions
    from teacher import Teacher, idx_to_buttons

    import tempfile
    from mapgen import FAMILIES, generate, write_map

    rng = random.Random(seed)
    cur = Curriculum(rng, hard_floor=cfg["hard_floor"], key_gate=cfg["key_gate"])
    rev = RevCurriculum(rng)
    prog = [0.0]  # fraction of the run done (sent by the learner), drives the fading combat assist
    var_wad = os.path.join(tempfile.gettempdir(), f"doomvar_{wid}_{os.getpid()}.wad")

    def priv_of(ps):
        """Privileged critic input (training only): distance/goal from the generator layout + state."""
        gd = None
        if S.get("shaper") is not None:
            held = frozenset(c for c, i in KEY_IDX.items() if ps["keys"][i] > 0)
            gd = S["shaper"].dist(ps["x"], ps["y"], held)
        return np.array([min(gd[1], 100) / 50 if gd else 2.0,
                         float(bool(gd) and gd[0][0] == "exit"), float(bool(gd) and gd[0][0] == "key"),
                         float(S.get("shaper") is not None), float(ps["keys"].sum()) / 3, ps["health"] / 100,
                         ps["tic"] / ps["episode_ticks"],
                         float(S.get("rev") is not None or S.get("replay") is not None)], np.float32)
    if cur0:  # restore unlocked bins from the checkpoint so a resume does not restart at "tiny"
        for k, v in cur0.items():
            f, li = k.split("|")
            cur.p[(f, int(li))] = float(v)
    feat = Features()
    env = None
    S = {}

    eps_left = 0
    # edge-case mining: per-episode snapshot files, cases queued by the learner, a lockstep step counter
    import shutil
    from doom_env.wrapper import GAME_VARIABLES
    mine = cfg["edge_p"] > 0
    ring = os.path.join(tempfile.gettempdir(), f"doomsnap_{wid}_{os.getpid()}")
    queue, gstep, ncase, failed_loads = [], [0], [0], []

    def layout(fam, li, mi):
        return generate(fam, li / (LEVELS - 1), FAMILIES.index(fam) * 1_000_000 + li * 10_000 + mi)

    def load_case(case):
        """Restore a mined engine snapshot (called right after env.reset). Returns obs or None."""
        g = env._game
        try:
            g.load(case["sav"])
        except Exception:
            return None
        if g.is_episode_finished():
            return None
        tic, full = g.get_episode_time(), FAMILY_TICKS[case["fam"]]
        if full - tic < full // 2:  # late failures (door stalls, plateaus) still get half an episode
            env.episode_ticks = tic + full // 2
            g.set_episode_timeout(env.episode_ticks)
        obs = env._observe(g.get_state(), first=True)
        env._keys.held[:] = case["held"]  # key flags are tracked by the wrapper, not stored in the save
        obs["vitals"][8:8 + len(case["held"])] = env._keys.held
        return obs

    def finish_now():
        """The episode ended during the ~2 tics a save advances: report it the way wrapper.step would."""
        g = env._game
        dead, to = g.is_player_dead(), g.is_episode_timeout_reached()
        env._exit_reached = not dead and not to
        env._last_vars = np.array([g.get_game_variable(v) for v in GAME_VARIABLES], np.float64)
        term = dead or env._exit_reached
        return env._last_obs, term, to and not term, env._info()

    def start():
        # creating a DoomGame costs ~0.4 s (vs 0.01 s reset) and stalls the whole batch,
        # so each sampled map is replayed for EPISODES_PER_MAP episodes with new seeds
        nonlocal env, eps_left
        case = queue.pop(0) if queue and rng.random() < cfg["edge_p"] else None
        if case is not None:  # replay a mined edge case on its own map, then go back to the curriculum
            if env is not None:
                env.close()
            env = make_env(case["wad"], case["map"], episode_ticks=FAMILY_TICKS[case["fam"]])
            S["shaper"] = None if case["fam"] == "freedoom" else GoalShaper(*layout(case["fam"], case["li"], case["mi"]))
            S.update(rev=None, bin=(case["fam"], case["li"]), wad=case["wad"], map=case["map"], mi=case["mi"],
                     variant=False)
            eps_left = 1
        elif env is None or eps_left <= 0:
            if env is not None:
                env.close()
            b = cur.sample()
            fam, li = b
            S["shaper"], S["rev"], mi = None, None, -1
            if fam == "freedoom":
                wad, mp_ = IWAD, f"MAP{rng.randint(1, 24):02d}"
            else:
                mi = rng.randrange(cfg["maps_per_bin"])
                wad, mp_ = os.path.join(map_dir, f"{fam}_{li}_{mi}.wad"), "MAP01"
                # rebuild this map's layout from its generator seed (same as build_pool)
                grid, things = layout(fam, li, mi)
                lvl = rev.sample(fam) if fam in REV_FAMILIES and rng.random() < cfg["rev_p"] else None
                assist = cfg["assist"] * max(0.0, 1 - prog[0] / cfg["assist_until"]) if fam == "gauntlet" else 0.0
                if lvl is not None or assist > 0:
                    things = map_variant(grid, things, REV_FRACS[lvl] if lvl is not None else 1.0, assist, rng)
                    wad = write_map(var_wad, grid, things)
                S["shaper"], S["rev"] = GoalShaper(grid, things), lvl
            env = make_env(wad, mp_, episode_ticks=FAMILY_TICKS[fam])
            S.update(bin=b, wad=wad, map=mp_, mi=mi, variant=wad == var_wad)
            eps_left = EPISODES_PER_MAP
        eps_left -= 1
        b = S["bin"]
        obs, _ = env.reset(seed=rng.randrange(1 << 30))
        S["replay"] = None
        if case is not None:
            o2 = load_case(case)
            if o2 is None:
                failed_loads.append(case["id"])  # play the map normally; the learner drops the case
            else:
                obs, S["replay"] = o2, case
        feat.reset()
        f, _ = feat.step(obs, None)
        ps = env.privileged_state()
        # doors already open at spawn (common on Freedoom) must not pay the door-opened bonus
        lock, fl, ce = ps["door_lock"], ps["sector_floor"], ps["sector_ceiling"]
        opened0 = set(np.nonzero((lock >= 0) & (ce - fl > 40))[0].tolist())
        visited = {(int(ps["x"] // 64), int(ps["y"] // 64))}
        used = set()
        if S["replay"] is not None:  # cells / doors already paid for before the snapshot stay paid
            visited |= {tuple(c) for c in case["visited"]}
            used = {tuple(d) for d in case["used"]}
        S.update(bin=b, ps=ps, visited=visited, opened=opened0,
                 used_doors=used, last_use=-99, last_sw=-99, t=0, ret=0.0, use_prev=0, phi=None, goal=None,
                 shaped=0.0, obs=obs, terms=defaultdict(float), trace=[], snaps=[],
                 hits=env._game.get_game_variable(vd.GameVariable.HITCOUNT))
        # DAgger teacher for this episode (generated maps only; Freedoom has no layout -> no labels)
        S["teacher"] = Teacher(S["shaper"], track_executed=True) if cfg["dagger"] and S.get("shaper") is not None else None
        f["priv"] = priv_of(ps)
        return f, ps

    def progress_of(ps):
        """Scalar 'distance left' for the miner: corridor distance + 200 per key still on the map."""
        if S.get("shaper") is None:
            return None
        held = frozenset(c for c, i in KEY_IDX.items() if ps["keys"][i] > 0)
        gd = S["shaper"].dist(ps["x"], ps["y"], held)
        return None if gd is None else gd[1] + 200 * sum(c not in held for c in S["shaper"].keys)

    def mine_case(p1, info):
        pk = pick_case(S["trace"], S["snaps"], p1["dead"], info["exit_reached"],
                       S["shaper"].g if S.get("shaper") is not None else None)
        if pk is None:
            return None
        kind, snap = pk
        ncase[0] += 1
        cid = f"{wid}_{os.getpid()}_{ncase[0]}"
        store = Path(cfg["edge_dir"])
        sav = str(store / f"{cid}.sav")
        shutil.copyfile(snap["path"], sav)
        wad, own = S["wad"], None
        if S["variant"]:  # reverse-curriculum / assist variants are rewritten per map: keep a copy
            own = wad = str(store / f"{cid}.wad")
            shutil.copyfile(var_wad, wad)
        fam, li = S["bin"]
        return dict(id=cid, kind=kind, fam=fam, li=li, mi=S["mi"], wad=wad, map=S["map"], own_wad=own, sav=sav,
                    t=snap["t"], ep_len=len(S["trace"]), tic=snap["tic"], pos=snap["pos"], health=snap["health"],
                    held=snap["held"], visited=snap["visited"],
                    used=snap["used"])

    f, ps = start()
    conn.send(f)
    while True:
        cmd, a = conn.recv()
        if cmd == "cur":
            conn.send(cur_to_plain(cur.p))
            continue
        if cmd == "prog":
            prog[0] = float(a[0])
            queue.extend(a[1])
            conn.send(len(queue))
            continue
        if cmd == "close":
            env.close()
            for p_ in [var_wad] + [s["path"] for s in S.get("snaps", [])]:
                if os.path.exists(p_):
                    os.remove(p_)
            return
        gstep[0] += 1
        obs_prev = S["obs"]  # what the agent saw when it chose `a`
        # ---- DAgger label: what the privileged teacher would do in the STUDENT's current situation
        lab = np.full(7, -1, np.int64)  # 6 head indices + valid flag
        if S.get("teacher") is not None:
            tidx, tvalid = S["teacher"].act(obs_prev, S["ps"])
            if tvalid:
                lab[:6], lab[6] = tidx, 1
                beta = cfg["dagger_beta"] * max(0.0, 1.0 - prog[0] / 0.5)  # mixing fades out by mid-run
                if beta > 0 and rng.random() < beta:
                    a = idx_to_buttons(tidx)  # execute the teacher's action (DAgger beta-mixture)
        if cfg["mask_back"]:  # same observation-based action mask as submission/agent.py
            a = mask_actions(a, obs_prev)
        if S.get("teacher") is not None:
            S["teacher"].note_executed(a)
        if S.pop("pending_end", False):
            obs, term, trunc, info = finish_now()
        else:
            obs, _, term, trunc, info = env.step(a)
        S["obs"] = obs
        p0, p1 = S["ps"], env.privileged_state()
        S["t"] += 1
        R = S["terms"]  # per-term episode sums (logged, so no single term can silently dominate)

        def add(name, x):
            R[name] += x
            return x

        r = add("step", -0.001)
        ve_prev = obs_prev["visible_entities"]
        enemy_seen = np.isin(ve_prev["entity_type"][ve_prev["mask"].astype(bool)], (1, 2, 3)).any()
        # walking backwards blinds the 90-degree entity view and points USE/ATTACK away from the path;
        # backing off while an enemy is in view (kiting) stays free
        if cfg["back_pen"] > 0 and obs["vitals"][14] < -1.0 and not enemy_seen:
            r += add("back", -cfg["back_pen"])
        if cfg["turn_pen"] > 0 and a[4] != a[5]:  # constant turning + moving = circling in place
            r += add("turn", -cfg["turn_pen"])
        # --- edge-case fixes (training-time reward only; see README "Edge cases") ---
        vit, vit1 = obs_prev["vitals"], obs["vitals"]
        if cfg["weapon_pen"] > 0:
            # rapid weapon cycling only: a single switch (e.g. to a freshly picked-up shotgun) is free
            if a[8]:
                if S["t"] - S["last_sw"] < 16:
                    r += add("weapon", -cfg["weapon_pen"])
                S["last_sw"] = S["t"]
            if vit[3] <= 0 and (vit[4] > 0 or vit[5] > 0):  # holding fists/empty gun while bullets/shells exist
                r += add("weapon", -0.2 * cfg["weapon_pen"])
        # --- combat (training-time reward only) ---
        hits = env._game.get_game_variable(vd.GameVariable.HITCOUNT)
        if cfg["hit_rew"] > 0:  # dense aiming signal: engine hit counter (never seen by the agent)
            r += add("hit", cfg["hit_rew"] * max(0.0, hits - S["hits"]))
        S["hits"] = hits
        if cfg["aim_k"] > 0:
            # potential-based aim shaping: phi = -aim_k * (angle to the nearest visible enemy / 45 deg),
            # only while an enemy is visible before and after the step (so it cannot be farmed)
            def aim(o):
                ve = o["visible_entities"]
                m = ve["mask"].astype(bool) & np.isin(ve["entity_type"], (1, 2, 3))
                if not m.any():
                    return None
                ft = ve["features"][m]
                return float(np.abs(np.arctan2(ft[:, 1], ft[:, 2])).min()) / math.radians(45)
            g0, g1 = aim(obs_prev), aim(obs)
            if g0 is not None and g1 is not None:
                r += add("aim", cfg["aim_k"] * (min(g0, 1.0) - min(g1, 1.0)))
        if cfg["shot_pen"] > 0 and a[6]:  # firing with no enemy in view wastes ammo
            ve = obs_prev["visible_entities"]
            if not np.isin(ve["entity_type"][ve["mask"].astype(bool)], (1, 2, 3)).any():
                spent = float(np.sum(vit[4:8]) - np.sum(vit1[4:8]))
                r += add("shot", -cfg["shot_pen"] * max(1.0, spent))  # per round actually fired
        r += add("kill", 1.0 * (p1["kills"] - p0["kills"]))
        r += add("damage", -cfg["dmg_pen"] * max(0.0, p1["damage_taken"] - p0["damage_taken"]))
        dkeys = float(p1["keys"].sum() - p0["keys"].sum())
        if S["rev"] is None or S["t"] > 3:  # reverse-curriculum starts hand over passed keys for free
            r += add("key", 3.0 * dkeys)
        # pickups: ItemCount only counts special items, so health / armor / ammo come from the vitals
        picked = int(vit1[0] > vit[0]) + int(vit1[1] > vit[1]) + int(vit1[4:8].sum() > vit[4:8].sum())
        r += add("item", 0.2 * (picked + max(0, p1["items"] - p0["items"])))
        lock, fl, ce = p1["door_lock"], p1["sector_floor"], p1["sector_ceiling"]
        for s in np.nonzero((lock >= 0) & (ce - fl > 40))[0]:
            if s not in S["opened"]:
                S["opened"].add(s)
                r += add("door", 1.0 + 2.0 * (lock[s] > 0))
        cell = (int(p1["x"] // 64), int(p1["y"] // 64))
        if dkeys > 0:
            # new key phase: re-arm the exploration bonus so backtracking to the locked door pays again
            S["visited"] = set()
        visited_before = cell in S["visited"]
        if not visited_before:
            S["visited"].add(cell)
            r += add("explore", cfg["cell_rew"])
        if a[7] and not S["use_prev"]:
            if S["t"] - S["last_use"] < 8:
                r += add("use", -cfg["use_bounce"])  # re-pressing USE bounces a moving door
            elif cfg["door_use"] > 0 and S.get("shaper") is not None:
                # fresh USE next to a door: paid ONCE per door per episode, never for a locked door without
                # its key (otherwise pressing USE at a door every 8 steps out-earns the exit)
                g_ = S["shaper"].g
                rr, cc = math.floor(-p0["y"] / 64.0), math.floor(p0["x"] / 64.0)
                held_c = {c for c, i in KEY_IDX.items() if p0["keys"][i] > 0}
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    y_, x_ = rr + dy, cc + dx
                    if 0 <= y_ < len(g_) and 0 <= x_ < len(g_[y_]) and g_[y_][x_] in "DRBY" \
                            and (g_[y_][x_] == "D" or g_[y_][x_] in held_c) and (y_, x_) not in S["used_doors"]:
                        S["used_doors"].add((y_, x_))
                        r += add("use", cfg["door_use"])
                        break
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
                    r += add("shape", phi - S["phi"])
                    S["shaped"] += phi - S["phi"]
                S["phi"], S["goal"] = phi, goal
        if info["exit_reached"]:
            r += add("exit", 10.0 + 2.0 * p1["health"] / 100 + 2.0 * (1 - p1["tic"] / p1["episode_ticks"]))
        elif term:
            r += add("death", -cfg["death_pen"])
        f_next, new_cell = feat.step(obs, a)
        # agent-internal curiosity on its own dead-reckoned map: kept small, since the true-cell bonus above
        # already pays for exploration and dead-reckoning drift makes old cells look new
        r += add("explore_dr", 0.005 * new_cell)
        S["ret"] += r
        # aux2 targets describe the state the agent OBSERVED when acting (p0), from the training-only layout
        gd0, dir0 = None, None
        if S.get("shaper") is not None:
            held0 = frozenset(c for c, i in KEY_IDX.items() if p0["keys"][i] > 0)
            gd0 = S["shaper"].dist(p0["x"], p0["y"], held0)
            dir0 = S["shaper"].path_dir(p0["x"], p0["y"], p0["angle"], held0)
        aux = np.array([p1["health"] / 100, p1["kills"], p1["keys"].sum(), visited_before,
                        min(gd0[1], 100) / 50 if gd0 else 0.0, dir0 if dir0 is not None else 0,
                        float(gd0 is not None), float(dir0 is not None), float(info["exit_reached"])], np.float32)
        ep = None
        S["ps"] = p1
        if mine:
            S["trace"].append((p1["x"], p1["y"], progress_of(p1)))
        if done:
            rp = S["replay"]
            if rp is not None:  # edge-case replays move neither curriculum
                pass
            elif S["rev"] is None:  # only full-start episodes move the level curriculum
                cur.update(S["bin"], info["exit_reached"])
            else:
                rev.update(S["bin"][0], S["rev"], info["exit_reached"])
            edge = mine_case(p1, info) if mine and rp is None else None
            # a death case is solved by getting out alive (exit or survive to the clock), the rest by the exit
            solved = info["exit_reached"] or (rp is not None and rp["kind"] == "death" and not p1["dead"])
            ep = dict(bin=S["bin"], exit=info["exit_reached"], ret=S["ret"], kills=p1["kills"],
                      keys=float(p1["keys"].sum()), cells=len(S["visited"]), dead=p1["dead"], rev=S["rev"],
                      terms=dict(S["terms"]), edge=edge,
                      replay=None if rp is None else (rp["id"], rp["kind"], bool(solved)),
                      failed_loads=failed_loads[:])
            failed_loads.clear()
            f_next, _ = start()
        else:
            if mine and S["replay"] is None and gstep[0] % cfg["edge_every"] == 0:
                path = f"{ring}_{len(S['snaps'])}.sav"
                env._game.save(path)  # advances the game ~2 tics; the episode may end inside them
                S["snaps"].append(dict(t=S["t"], path=path, held=env._keys.held.tolist(), tic=p1["tic"],
                                       pos=[round(p1["x"]), round(p1["y"])], health=p1["health"],
                                       visited=[list(c) for c in S["visited"]],
                                       used=[list(d) for d in S["used_doors"]]))
                if env._game.is_episode_finished():
                    S["pending_end"] = True
            f_next["priv"] = priv_of(p1)
        conn.send((f_next, r, done, aux, ep, lab))


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

    def progress(self, frac: float, cases=None):
        """Tell workers how far the run is (drives the fading combat assist) and hand them edge cases to
        replay. Returns each worker's queue length."""
        for i, c in enumerate(self.conns):
            c.send(("prog", (frac, cases[i] if cases else [])))
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
KEYS = ("vec", "rays", "ents", "etype", "map", "crop_idx", "pos_idx", "wmask", "priv")  # priv: critic only


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


EXIT_SOON = 64


def aux_targets(aux, dones, ks=(8, 32)):
    """aux: (T,N,9) [health, kills, keys, visited, goal dist, path dir, dist valid, dir valid, exit now];
    dones: (T,N). Returns (T,N,13): future deltas within the episode (6), visited (1), goal dist (1),
    path dir (1), dist valid (1), dir valid (1), exit within EXIT_SOON steps (1), its valid mask (1)."""
    T, N, _ = aux.shape
    end = np.empty((T, N), np.int64)
    nxt = np.full(N, T - 1)
    ex = np.full(N, 1 << 30)
    nxt_exit = np.empty((T, N), np.int64)
    for t in range(T - 1, -1, -1):
        nxt = np.where(dones[t], t, nxt)
        end[t] = nxt
        ex = np.where(dones[t], np.where(aux[t, :, 8] > 0, t, 1 << 30), ex)
        nxt_exit[t] = ex
    tgt = []
    for k in ks:
        idx = np.minimum(np.arange(T)[:, None] + k, end)
        fut = np.take_along_axis(aux[..., :3], idx[..., None], 0)
        tgt.append(fut - aux[..., :3])
    ar = np.arange(T)[:, None]
    soon = (nxt_exit - ar) <= EXIT_SOON
    # the label is known if the window lies inside the rollout or the episode ended inside it
    known = (ar + EXIT_SOON <= T - 1) | dones[end, np.arange(N)[None, :]]
    extra = np.stack([soon, known], -1).astype(np.float32)
    return np.concatenate(tgt + [aux[..., 3:4], aux[..., 4:8], extra], -1).astype(np.float32)


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
                    help="reward for the first USE press next to each closed door (generated maps)")
    ap.add_argument("--bc-w", type=float, default=0.0,
                    help="DAgger / learning-by-cheating: weight of the cross-entropy loss toward the privileged "
                         "teacher's action (training/teacher.py); 0 = off")
    ap.add_argument("--ppo-w", type=float, default=1.0, help="weight of the PPO policy+value+entropy terms (0 = pure imitation)")
    ap.add_argument("--dagger-beta", type=float, default=0.0,
                    help="probability of executing the teacher's action instead of the student's (fades to 0 by mid-run)")
    ap.add_argument("--mask-back", type=int, default=DEFAULT_CFG["mask_back"],
                    help="1 = apply submission/action_mask.py (MOVE_BACKWARD only with an enemy in view), "
                         "exactly as the submitted agent does")
    ap.add_argument("--back-pen", type=float, default=DEFAULT_CFG["back_pen"],
                    help="penalty per step moving backwards with no enemy in view")
    ap.add_argument("--turn-pen", type=float, default=DEFAULT_CFG["turn_pen"], help="penalty per turning step")
    ap.add_argument("--use-bounce", type=float, default=DEFAULT_CFG["use_bounce"],
                    help="penalty for re-pressing USE within 8 steps (slams the opening door shut)")
    ap.add_argument("--hit-rew", type=float, default=DEFAULT_CFG["hit_rew"],
                    help="reward per enemy hit (engine HITCOUNT, training only)")
    ap.add_argument("--aim-k", type=float, default=DEFAULT_CFG["aim_k"],
                    help="potential-based shaping toward the nearest visible enemy (training only)")
    ap.add_argument("--ent-end", type=float, default=None,
                    help="entropy coefficient at the end of the run (linear anneal from --ent); default = --ent")
    ap.add_argument("--rev-p", type=float, default=DEFAULT_CFG["rev_p"],
                    help="probability that a multi/key/gauntlet episode uses a reverse-curriculum start")
    ap.add_argument("--assist", type=float, default=DEFAULT_CFG["assist"],
                    help="initial fraction of gauntlet monsters removed (fades to 0)")
    ap.add_argument("--edge-p", type=float, default=DEFAULT_CFG["edge_p"],
                    help="edge-case mining: probability that a new episode starts from a mined failure snapshot "
                         "(0 = mining off)")
    ap.add_argument("--edge-every", type=int, default=DEFAULT_CFG["edge_every"],
                    help="steps between engine snapshots (all workers save on the same step)")
    ap.add_argument("--edge-cap", type=int, default=400, help="max edge cases kept in the bank")
    ap.add_argument("--assist-until", type=float, default=DEFAULT_CFG["assist_until"],
                    help="fraction of the run after which the combat assist is 0")
    args = ap.parse_args()
    if args.ent_end is None:
        args.ent_end = args.ent

    import torch
    from model import Net, heads_to_buttons, load_compat, sample, evaluate, split_logits
    from mapgen import build_pool

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(maps_per_bin=args.maps_per_bin, dmg_pen=args.dmg_pen, death_pen=args.death_pen,
               cell_rew=args.cell_rew, shape_k=args.shape_k, hard_floor=args.hard_floor, key_gate=args.key_gate,
               weapon_pen=args.weapon_pen, shot_pen=args.shot_pen, door_use=args.door_use,
               rev_p=args.rev_p, assist=args.assist, assist_until=args.assist_until,
               edge_p=args.edge_p, edge_every=args.edge_every, edge_dir=str(Path(args.out) / "edge_cases"),
               back_pen=args.back_pen, turn_pen=args.turn_pen, use_bounce=args.use_bounce,
               hit_rew=args.hit_rew, aim_k=args.aim_k, mask_back=args.mask_back,
               dagger=int(args.bc_w > 0), dagger_beta=args.dagger_beta)
    print(f"config {cfg}", flush=True)
    if not Path(args.maps, f"gauntlet_4_{args.maps_per_bin - 1}.wad").exists():
        print("generating maps ...", flush=True)
        build_pool(args.maps, per_bin=args.maps_per_bin, levels=tuple(i / (LEVELS - 1) for i in range(LEVELS)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    it0, steps, cur0, edge0 = 0, 0, None, None
    if args.resume and (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location=dev)
        changed = load_compat(net, ck["model"])
        if changed:  # architecture changed (e.g. privileged critic): Adam moments no longer match
            print(f"resume: re-initialised {changed}; fresh optimizer", flush=True)
        else:
            opt.load_state_dict(ck["opt"])
        for g_ in opt.param_groups:  # the CLI --lr wins over the lr stored in the checkpoint
            g_["lr"] = args.lr
        it0, steps = ck["iter"], ck["steps"]
        cur0 = ck.get("cur")
        edge0 = ck.get("edge")
        print(f"resumed at iter {it0}, {steps} steps, lr {args.lr}, ent {args.ent}, "
              f"curriculum {'restored' if cur0 else 'fresh'}", flush=True)

    N, T, C = args.envs, args.T, args.chunk
    nch = T // C
    bank, qlens, erng = None, [0] * N, random.Random(args.seed)
    if args.edge_p > 0:
        Path(cfg["edge_dir"]).mkdir(parents=True, exist_ok=True)
        bank = EdgeBank(Path(cfg["edge_dir"]), args.edge_cap, edge0)
        print(f"edge-case mining on: {len(bank.cases)} cases restored", flush=True)
    envs = Envs(N, args.maps, args.seed, cur0, cfg)
    obs = stack(envs.first)
    starts = np.ones(N, bool)
    pol = GraphPolicy(net, obs, dev)
    t_start = time.time()
    t_end = t_start + args.hours * 3600
    eps_log = []
    it = it0
    while time.time() < t_end and (args.iters == 0 or it - it0 < args.iters):
        t0 = time.time()
        frac = min(1.0, (t0 - t_start) / (args.hours * 3600))
        if args.iters:
            frac = max(frac, (it - it0) / args.iters)
        qlens = envs.progress(frac, [bank.take(erng, max(0, 2 - q)) for q in qlens] if bank else None)
        n_eps0, mined_it, rep_it = len(eps_log), 0, []
        ent_coef = args.ent + (args.ent_end - args.ent) * frac
        buf = {k: [] for k in KEYS}
        acts, logps, vals, rews, dones, auxs, strt, labs = [], [], [], [], [], [], [], []
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
                labs.append(np.stack([x[5] for x in res]))
                eps_log += [x[4] for x in res if x[4] is not None]
                obs = stack([x[0] for x in res])
                starts = d
            h_keep, L_keep = pol.h.clone(), pol.L.clone()
            _, _, v_last = pol(obs, starts)  # bootstrap value; restore state (action is re-sampled next iter)
            pol.h.copy_(h_keep); pol.L.copy_(L_keep)
        t_roll = time.time() - t0
        if bank is not None:  # bank the failures the engine found this iteration, score the replays
            for e in eps_log[n_eps0:]:
                if e["edge"] is not None:
                    bank.add(e["edge"], it)
                    mined_it += 1
                if e["replay"] is not None:
                    bank.result(e["replay"][0], e["replay"][2])
                    rep_it.append(e["replay"])
                for cid in e["failed_loads"]:
                    if cid in bank.cases:
                        bank._drop(cid, "load_failed")

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
        acts, logps, strt, labs = np.array(acts), np.array(logps), np.array(strt), np.array(labs)  # labs (T,N,7)

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
                bc_num = bc_den = agree_num = 0.0
                for k in range(C):
                    ts = cs * C + k
                    sl = (ts, ns)
                    mk = torch.from_numpy(strt[ts, ns]).to(dev).float()
                    st = (st[0] * (1 - mk)[:, None], st[1] * (1 - mk)[:, None, None])
                    logits, v, aux, st = net.step(to_t(data, dev, sl), st)
                    lp, en = evaluate(logits, torch.from_numpy(acts[ts, ns]).to(dev))
                    lp_l.append(lp); ent_l.append(en); v_l.append(v); aux_l.append(aux)
                    if args.bc_w > 0:  # DAgger: cross-entropy of every action head toward the teacher's label
                        lab_k = torch.from_numpy(labs[ts, ns]).to(dev)
                        valid = (lab_k[:, 6] == 1).float()
                        heads = split_logits(logits)
                        ce_k = sum(torch.nn.functional.cross_entropy(lg, lab_k[:, j].clamp(min=0), reduction="none")
                                   for j, lg in enumerate(heads))
                        bc_num = bc_num + (ce_k * valid).sum()
                        bc_den = bc_den + valid.sum()
                        with torch.no_grad():  # move + turn agreement with the teacher
                            agree = ((heads[0].argmax(-1) == lab_k[:, 0]) & (heads[2].argmax(-1) == lab_k[:, 2])).float()
                            agree_num += float((agree * valid).sum())
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
                # aux2 (privileged training-only targets): goal distance, shortest-path direction, exit soon
                F_ = torch.nn.functional
                mdist, mdir, mes = at[..., 9], at[..., 10], at[..., 12]
                l_dist = (mdist * (aux[..., 7] - at[..., 7]) ** 2).sum() / mdist.sum().clamp(min=1)
                ce = F_.cross_entropy(aux[..., 8:16].reshape(-1, 8), at[..., 8].long().reshape(-1), reduction="none")
                l_dir = (mdir.reshape(-1) * ce).sum() / mdir.sum().clamp(min=1)
                l_es = (mes * F_.binary_cross_entropy_with_logits(aux[..., 16], at[..., 11], reduction="none")).sum() \
                    / mes.sum().clamp(min=1)
                al2 = l_dist + l_dir + l_es
                stats["aux_dist"].append(l_dist.item()); stats["aux_dir"].append(l_dir.item())
                stats["aux_exit"].append(l_es.item())
                al = al + al2
                loss = args.ppo_w * (pg + vl - ent_coef * en.mean()) + 0.1 * al
                if args.bc_w > 0:
                    bc = bc_num / max(float(bc_den), 1.0)
                    loss = loss + args.bc_w * bc
                    stats["bc"].append(float(bc))
                    stats["agree"].append(agree_num / max(float(bc_den), 1.0))
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
            if e.get("replay") is None:  # replays start mid-level: keep the per-level exit rates honest
                fam[f"{e['bin'][0]}{e['bin'][1]}"].append(e["exit"])
        fs = " ".join(f"{k}:{np.mean(v):.2f}({len(v)})" for k, v in sorted(fam.items()))
        tsum = defaultdict(float)
        for e in recent:
            for k, v in e.get("terms", {}).items():
                tsum[k] += v / len(recent)
        ts = " ".join(f"{k}:{v:+.2f}" for k, v in sorted(tsum.items()))
        nrev = sum(e.get("rev") is not None for e in recent)
        print(f"it {it} steps {steps/1e6:.2f}M sps {N*T/dt:.0f} (roll {t_roll:.1f}s, upd {dt-t_roll:.1f}s) "
              f"ret {np.mean([e['ret'] for e in recent]) if recent else 0:.2f} entc {ent_coef:.4f} "
              + " ".join(f"{k} {np.mean(v):.3f}" for k, v in stats.items()) + f" | exit {fs}"
              + f" | rev {nrev} | terms {ts}" + (
                  f" | edge bank {len(bank.cases)} +{mined_it} replays {len(rep_it)} "
                  f"solved {sum(r[2] for r in rep_it)} retired {bank.stats['retired:learned']}L/"
                  f"{bank.stats['retired:hopeless']}H" if bank is not None else ""), flush=True)
        eps_log = recent
        if it % 10 == 0 or time.time() >= t_end:
            ck = {"model": net.state_dict(), "opt": opt.state_dict(), "iter": it, "steps": steps,
                  "cur": envs.curricula(), "edge": bank.state() if bank is not None else edge0}
            torch.save(ck, out / "latest.pt.tmp")
            os.replace(out / "latest.pt.tmp", out / "latest.pt")
            if bank is not None:
                import json
                (out / "edge_report.json").write_text(json.dumps(bank.report(it), indent=1))
            if it % 50 == 0:
                torch.save({"model": net.state_dict()}, out / f"it{it}.pt")
    envs.close()


if __name__ == "__main__":
    main()
