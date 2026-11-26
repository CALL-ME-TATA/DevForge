"""Env throughput benchmark: steps/s for 1 process and N processes (random actions)."""
import sys, time
from multiprocessing import Pool
import numpy as np
from doom_env import make_env

def run(args):
    wad, n, render = args
    env = make_env(wad, "MAP01", episode_ticks=4200, render_buffers=render)
    rng = np.random.default_rng(0)
    env.reset(seed=0); t = time.perf_counter(); steps = 0
    while steps < n:
        _, _, term, trunc, _ = env.step(rng.integers(0, 2, 9))
        steps += 1
        if term or trunc: env.reset()
    dt = time.perf_counter() - t; env.close()
    return steps / dt

if __name__ == "__main__":
    wad = sys.argv[1] if len(sys.argv) > 1 else "maps/two_rooms.wad"
    for render in (True, False):
        print(f"render={render} 1 proc: {run((wad, 2000, render)):.0f} steps/s")
    for k in (6, 10):
        t = time.perf_counter()
        with Pool(k) as p: r = p.map(run, [(wad, 2000, True)] * k)
        print(f"{k} procs: {sum(r):.0f} steps/s aggregate (per-proc {np.mean(r):.0f})")
