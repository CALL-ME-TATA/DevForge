"""Classify failed held-out episodes (evaluate.py JSON with diagnostics) into edge cases.

  python training/edge_cases.py training/eval_results.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

TIER = {"tiny": "T0", "arena": "T1", "multi": "T2", "key": "T3", "gauntlet": "T4", "freedoom": "T4"}


def classify(r):
    if r["exit"]:
        return "cleared"
    if r["dead"]:
        return "died early (<40% of time)" if r["death_frac"] < 0.4 else "died late"
    if r.get("min_dist") is None:
        return "timeout (no layout info)"
    if r["min_dist"] <= 1:
        return "reached goal cell, did not finish"
    if r["blocked_by"] == "locked":
        return "stalled at a LOCKED door"
    if r["blocked_by"] == "door":
        return "stalled at a door (never opened / passed)"
    if r["got_key_phase"]:
        return "got key, never reached exit"
    if r["still_steps"] >= 150:
        return "stuck in place >=150 steps"
    if r["start_dist"] and r["min_dist"] > 0.6 * r["start_dist"]:
        return "lost: barely got closer to goal"
    return "partial progress, ran out of time"


res = json.loads(Path(sys.argv[1]).read_text())
for c in dict.fromkeys(r["ckpt"] for r in res):
    rs = [r for r in res if r["ckpt"] == c]
    print(f"\n== {Path(c).name}: {len(rs)} episodes, {sum(r['exit'] for r in rs)} cleared")
    by_tier = defaultdict(Counter)
    for r in rs:
        by_tier[TIER[r["tag"].rstrip("0123456789")]][classify(r)] += 1
    allc = Counter(classify(r) for r in rs)
    tiers = sorted(by_tier)
    print(f"{'edge case':<44}" + "".join(f"{t:>5}" for t in tiers) + f"{'all':>6}")
    for case, n in allc.most_common():
        print(f"{case:<44}" + "".join(f"{by_tier[t][case]:>5}" for t in tiers) + f"{n:>6}")
    fails = [r for r in rs if not r["exit"]]
    if fails:
        print(f"\nfailed episodes: avg longest stand-still {sum(r['still_steps'] for r in fails) / len(fails):.0f} steps, "
              f"avg steps per unique cell {sum(r['revisit_ratio'] for r in fails) / len(fails):.1f}")
        deaths = [r for r in fails if r["dead"]]
        if deaths:
            print(f"deaths: {len(deaths)}, avg at {100 * sum(r['death_frac'] for r in deaths) / len(deaths):.0f}% of time, "
                  f"avg kills before dying {sum(r['kills'] for r in deaths) / len(deaths):.1f}")
        g = [r for r in fails if r.get("start_dist")]
        if g:
            print(f"timeouts/deaths with layout: avg start distance {sum(r['start_dist'] for r in g) / len(g):.0f} cells, "
                  f"closest approach {sum(r['min_dist'] for r in g) / len(g):.0f}, final {sum(r['final_dist'] for r in g) / len(g):.0f}")
