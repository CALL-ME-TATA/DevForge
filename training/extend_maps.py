"""Grow the training map pool to N layouts per (family, level) bin without touching existing files.

Uses the same seed scheme as mapgen.build_pool, so maps 0..39 are identical to the original pool and
40..N-1 are new layouts. Held-out eval maps use i >= 5000, so they never overlap.

  python training/extend_maps.py --per-bin 200
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "env")]
LEVELS = 5


def make(job):
    from mapgen import generate, write_map
    path, fam, d, seed = job
    grid, things = generate(fam, d, seed)
    write_map(path, grid, things)


def main():
    from mapgen import FAMILIES
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bin", type=int, default=200)
    ap.add_argument("--maps", default=str(ROOT / "training" / "maps_gen"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.maps)
    jobs = []
    for fi, fam in enumerate(FAMILIES):
        for li in range(LEVELS):
            for i in range(args.per_bin):
                p = out / f"{fam}_{li}_{i}.wad"
                if not p.exists():
                    jobs.append((str(p), fam, li / (LEVELS - 1), fi * 1_000_000 + li * 10_000 + i))
    print(f"generating {len(jobs)} new maps", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(make, jobs, chunksize=20), 1):
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
