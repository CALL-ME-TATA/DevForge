"""Go/no-go check: let the privileged Teacher drive alone on held-out generated maps.

  python training/test_teacher.py --per-bin 6 --levels 0 2 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(ROOT / "env")]
TICKS = {"tiny": 1050, "arena": 2100, "multi": 3150, "key": 4200, "gauntlet": 4200}
TIER = {"tiny": "T0", "arena": "T1", "multi": "T2", "key": "T3", "gauntlet": "T4"}


def run(job):
    fam, li, mi = job
    import warnings
    warnings.filterwarnings("ignore")
    from mapgen import FAMILIES, generate
    from train import GoalShaper
    from teacher import Teacher, idx_to_buttons
    from doom_env import make_env
    grid, things = generate(fam, li / 4, FAMILIES.index(fam) * 1_000_000 + li * 10_000 + mi)
    t = Teacher(GoalShaper(grid, things))
    env = make_env(str(ROOT / "training" / "maps_eval" / f"{fam}_{li}_{mi}.wad"), "MAP01", episode_ticks=TICKS[fam])
    obs, _ = env.reset(seed=12345)
    steps, invalid, back_noenemy, turning, still = 0, 0, 0, 0, 0
    while True:
        ps = env.privileged_state()
        idx, valid = t.act(obs, ps)
        invalid += int(not valid)
        a = idx_to_buttons(idx)
        turning += int(a[4] != a[5])
        obs, _, term, trunc, info = env.step(a)
        still += int(abs(obs["vitals"][14]) <= 1 and abs(obs["vitals"][15]) <= 1)
        steps += 1
        if term or trunc:
            break
    ps = env.privileged_state()
    env.close()
    out = "exit" if info["exit_reached"] else ("died" if ps["dead"] else "timeout")
    return dict(fam=fam, li=li, mi=mi, out=out, steps=steps, kills=ps["kills"], keys=float(ps["keys"].sum()),
                invalid=invalid, turning=turning, still=still)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bin", type=int, default=6)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--start", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    jobs = [(f, li, i) for f in TICKS for li in args.levels for i in range(args.start, args.start + args.per_bin)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = list(pool.imap_unordered(run, jobs))
    by = defaultdict(list)
    for r in res:
        by[r["fam"]].append(r)
    print(f"{'tier':<14}{'maps':>5}{'exit':>6}{'died':>6}{'time':>6}{'kills':>7}{'keys':>6}   exit by level")
    for fam in TICKS:
        g = by[fam]
        c = Counter(r["out"] for r in g)
        lv = "/".join(f"{sum(r['out'] == 'exit' for r in g if r['li'] == li)}" for li in args.levels)
        print(f"{TIER[fam] + ' ' + fam:<14}{len(g):>5}{c['exit']:>6}{c['died']:>6}{c['timeout']:>6}"
              f"{sum(r['kills'] for r in g) / len(g):>7.1f}{sum(r['keys'] for r in g) / len(g):>6.2f}   {lv}")
    tot = Counter(r["out"] for r in res)
    print(f"{'ALL':<14}{len(res):>5}{tot['exit']:>6}{tot['died']:>6}{tot['timeout']:>6}")
    n = sum(r["steps"] for r in res)
    print(f"no-label steps {100 * sum(r['invalid'] for r in res) / n:.1f}% | turning on {100 * sum(r['turning'] for r in res) / n:.0f}% of steps"
          f" | standing still {100 * sum(r['still'] for r in res) / n:.0f}% of steps")


if __name__ == "__main__":
    main()
