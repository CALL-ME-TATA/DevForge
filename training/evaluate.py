"""Held-out evaluation: play checkpoints on maps never used in training and report score categories.

Held-out maps come from the same generator as training but with unseen seeds (training uses
i = 0..39 per bin; eval uses i = 5000..). Optionally adds Freedoom levels. Each (checkpoint, map)
pair is played with the same fixed seed so checkpoints are compared on identical episodes.

  python training/evaluate.py --ckpts training/ckpt/latest.pt submission/weights/model.pt
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(ROOT / "env")]

FAMILY_TICKS = {"tiny": 1050, "arena": 2100, "multi": 3150, "key": 4200, "gauntlet": 4200, "freedoom": 4200}
LEVELS = 5


def build_eval_maps(out_dir: Path, per_bin: int, levels: list[int], start: int = 5000,
                    families=None) -> list[tuple[str, str, str, int]]:
    from mapgen import FAMILIES, generate, write_map
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = []
    for fi, fam in enumerate(FAMILIES):
        if families and fam not in families:
            continue
        for li in levels:
            d = li / (LEVELS - 1)
            for i in range(start, start + per_bin):
                p = out_dir / f"{fam}_{li}_{i}.wad"
                if not p.exists():
                    grid, things = generate(fam, d, fi * 1_000_000 + li * 10_000 + i)
                    write_map(p, grid, things)
                maps.append((f"{fam}{li}", str(p), "MAP01", FAMILY_TICKS[fam]))
    return maps


def run_one(job):
    ckpt, (tag, wad, mapname, ticks), seed, greedy = job
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from doom_env import make_env
    import agent as agent_mod
    from agent import DoomAgent
    if greedy:  # always take each head's most likely action instead of sampling
        import functools
        import model
        agent_mod.sample = functools.partial(model.sample, greedy=True)

    env = make_env(wad, mapname, episode_ticks=ticks)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu")
    from model import load_compat
    load_compat(agent.net, torch.load(ckpt, map_location="cpu")["model"])
    # diagnostics (generated maps only): rebuild the layout to measure distance-to-goal over time
    shaper = None
    if wad.endswith(".wad") and "maps_eval" in wad.replace("\\", "/"):
        from mapgen import FAMILIES, generate
        from train import GoalShaper
        fam_, li_, mi_ = Path(wad).stem.rsplit("_", 2)
        grid, things = generate(fam_, int(li_) / (LEVELS - 1), FAMILIES.index(fam_) * 1_000_000 + int(li_) * 10_000 + int(mi_))
        shaper = GoalShaper(grid, things)
    KEY_IDX = {"B": 0, "Y": 1, "R": 2}

    agent.reset()
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    cells, steps, t0 = set(), 0, time.time()
    trace = []  # (x, y, health, goal-kind, dist)
    while True:
        ps = env.privileged_state()
        cells.add((int(ps["x"] // 64), int(ps["y"] // 64)))
        gd = None
        if shaper is not None:
            held = frozenset(c for c, i in KEY_IDX.items() if ps["keys"][i] > 0)
            gd = shaper.dist(ps["x"], ps["y"], held)
        trace.append((ps["x"], ps["y"], ps["health"], gd[0][0] if gd else None, gd[1] if gd else None))
        obs, _, term, trunc, info = env.step(agent.act(obs))
        steps += 1
        if term or trunc:
            break
    ps = env.privileged_state()
    env.close()

    # longest window where the agent moved less than one cell (stuck on a wall / corner / door)
    longest_still, i0 = 0, 0
    for j in range(len(trace)):
        while i0 < j and max(abs(trace[j][0] - trace[i0][0]), abs(trace[j][1] - trace[i0][1])) > 64:
            i0 += 1
        longest_still = max(longest_still, j - i0)
    ds = [t[4] for t in trace if t[4] is not None]
    diag = dict(start_dist=ds[0] if ds else None, min_dist=min(ds) if ds else None,
                final_dist=ds[-1] if ds else None, still_steps=longest_still,
                revisit_ratio=round(steps / max(1, len(cells)), 1),
                got_key_phase=any(t[3] == "exit" for t in trace) and any(t[3] == "key" for t in trace),
                blocked_by=None)
    if shaper is not None and ds:  # what is in the next cell toward the goal at the closest approach?
        j = min(range(len(trace)), key=lambda k: trace[k][4] if trace[k][4] is not None else 1 << 30)
        x, y = trace[j][0], trace[j][1]
        r, c = int(-y // 64), int(x // 64)
        best = None
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < shaper.H and 0 <= nc < len(shaper.g[nr]):
                ch = shaper.g[nr][nc]
                if ch in "DRBY":
                    best = ch
        diag["blocked_by"] = {"D": "door", "R": "locked", "B": "locked", "Y": "locked"}.get(best)
    return dict(ckpt=ckpt, tag=tag, map=Path(wad).name + ":" + mapname, exit=bool(info["exit_reached"]),
                dead=bool(ps["dead"]), kills=int(ps["kills"]), keys=float(np.sum(ps["keys"])),
                cells=len(cells), health=float(ps["health"]), tic=int(ps["tic"]), steps=steps,
                death_frac=round(steps / max(1, ticks // 4), 2), secs=round(time.time() - t0, 1), **diag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--per-bin", type=int, default=4)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--freedoom", type=int, default=2, help="number of Freedoom levels (MAP01..) to add")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=5000, help="first map index (training uses 0..499)")
    ap.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    ap.add_argument("--maps-dir", default=str(ROOT / "training" / "maps_eval"))
    ap.add_argument("--out", default=str(ROOT / "training" / "eval_results.json"))
    ap.add_argument("--families", nargs="+", default=None,
                    help="only these generator families (e.g. --families gauntlet for a Tier-4-only eval)")
    args = ap.parse_args()

    maps = build_eval_maps(Path(args.maps_dir), args.per_bin, args.levels, args.seed_start, args.families)
    if args.freedoom:
        from doom_env.wrapper import IWAD
        maps += [("freedoom", IWAD, f"MAP{i:02d}", FAMILY_TICKS["freedoom"]) for i in range(1, args.freedoom + 1)]
    jobs = [(c, m, 12345 + j, args.greedy) for c in args.ckpts for j, m in enumerate(maps)]
    print(f"{len(maps)} held-out maps x {len(args.ckpts)} checkpoints = {len(jobs)} episodes, "
          f"{args.workers} workers", flush=True)

    t0, res = time.time(), []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            res.append(r)
            if len(res) % 10 == 0:
                print(f"  {len(res)}/{len(jobs)} episodes ({time.time() - t0:.0f}s)", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))

    for c in args.ckpts:
        rs = [r for r in res if r["ckpt"] == c]
        by = defaultdict(list)
        for r in rs:
            by[r["tag"]].append(r)
        print(f"\n== {c}   ({len(rs)} episodes)")
        print(f"{'map type':<10}{'exit%':>7}{'keys':>7}{'kills':>7}{'cells':>7}{'died%':>7}")
        for tag in sorted(by):
            g = by[tag]
            n = len(g)
            print(f"{tag:<10}{100 * sum(r['exit'] for r in g) / n:>6.0f}%{sum(r['keys'] for r in g) / n:>7.2f}"
                  f"{sum(r['kills'] for r in g) / n:>7.1f}{sum(r['cells'] for r in g) / n:>7.0f}"
                  f"{100 * sum(r['dead'] for r in g) / n:>6.0f}%")
        n = len(rs)
        print(f"{'OVERALL':<10}{100 * sum(r['exit'] for r in rs) / n:>6.0f}%{sum(r['keys'] for r in rs) / n:>7.2f}"
              f"{sum(r['kills'] for r in rs) / n:>7.1f}{sum(r['cells'] for r in rs) / n:>7.0f}"
              f"{100 * sum(r['dead'] for r in rs) / n:>6.0f}%")


if __name__ == "__main__":
    main()
