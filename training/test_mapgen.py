"""Engine check for generated maps: door/locked-door ray types, key pickup opens lock, exit ends episode."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from doom_env import make_env  # noqa: E402
from mapgen import FAMILIES, generate, write_map  # noqa: E402

FWD, USE = np.eye(9, dtype=int)[0], np.eye(9, dtype=int)[7]
NOOP = np.zeros(9, int)


def run(env, a, n):
    for _ in range(n):
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            return obs, term, info
    return obs, False, info


def main():
    tmp = Path(tempfile.mkdtemp())
    # 1) plain door straight ahead -> ray 0 is 'door'(2); USE opens it
    w = write_map(tmp / "door.wad", ["#######", "#...D.#", "#######"], [(1, 1, 1, 0)])
    env = make_env(w, episode_ticks=700)
    obs, _ = env.reset(seed=0)
    assert obs["rays"]["hit_type"][0] == 2, obs["rays"]["hit_type"][:4]
    obs, _, _ = run(env, FWD, 15)
    assert obs["rays"]["distance"][0] < 0.05, obs["rays"]["distance"][0]
    run(env, USE, 1)
    obs, _, _ = run(env, NOOP, 30)
    assert obs["rays"]["hit_type"][0] != 2, "door did not open"
    env.close()

    # 2) locked red door: USE fails without key; picking up key then USE opens; exit pad ends episode
    w = write_map(tmp / "lock.wad", ["###########", "#....R..E.#", "###########"], [(1, 1, 1, 0), (1, 3, 13, 0)])
    env = make_env(w, episode_ticks=1400)
    obs, _ = env.reset(seed=0)
    assert obs["rays"]["hit_type"][0] == 5, "key item should block ray 0"
    obs, _, _ = run(env, FWD, 12)
    assert obs["vitals"][10] == 1, f"red key not picked up {obs['vitals'][8:14]}"
    assert obs["rays"]["hit_type"][0] == 3, obs["rays"]["hit_type"][:4]
    run(env, USE, 1)
    obs, _, _ = run(env, NOOP, 30)
    assert obs["rays"]["hit_type"][0] != 3, "locked door did not open with key"
    obs, term, info = run(env, FWD, 60)
    assert term and info["exit_reached"], "exit not reached"
    env.close()

    # 3) every family loads and runs a few random steps
    rng = np.random.default_rng(0)
    for fam in FAMILIES:
        for d in (0.0, 1.0):
            grid, things = generate(fam, d, 7)
            env = make_env(write_map(tmp / f"{fam}{d}.wad", grid, things), episode_ticks=200)
            env.reset(seed=0)
            run(env, rng.integers(0, 2, 9), 20)
            assert len(env.sector_polygons()) > 0
            env.close()
    print("mapgen OK")


if __name__ == "__main__":
    main()
