"""Group held-out eval results (evaluate.py JSON) by the competition's 4 tiers.

  python training/tier_report.py training/eval_results.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

TIERS = [  # (name, map-type prefixes) -- mapping of our generator families onto docs/EVALUATION.md tiers
    ("T0 Basics (tiny)", ("tiny",)),
    ("T1 Combat & Survival (arena)", ("arena",)),
    ("T2 Multi-Room Exploration (multi)", ("multi",)),
    ("T3 Long-Horizon Progression (key)", ("key",)),
    ("T4 Full Gauntlet (gauntlet+freedoom)", ("gauntlet", "freedoom")),
]


def tier_of(tag):
    for name, prefixes in TIERS:
        if tag.startswith(prefixes):
            return name
    return "other"


def row(name, g):
    n = len(g)
    return (f"{name:<38}{n:>4}{100 * sum(r['exit'] for r in g) / n:>7.0f}%"
            f"{sum(r['keys'] for r in g) / n:>7.2f}{sum(r['kills'] for r in g) / n:>7.1f}"
            f"{sum(r['cells'] for r in g) / n:>7.0f}{100 * sum(r['dead'] for r in g) / n:>7.0f}%")


res = json.loads(Path(sys.argv[1]).read_text())
for c in dict.fromkeys(r["ckpt"] for r in res):
    rs = [r for r in res if r["ckpt"] == c]
    by = defaultdict(list)
    for r in rs:
        by[tier_of(r["tag"])].append(r)
    print(f"\n== {Path(c).name}")
    print(f"{'tier':<38}{'eps':>4}{'exit':>8}{'keys':>7}{'kills':>7}{'cells':>7}{'died':>8}")
    for name, _ in TIERS:
        if by.get(name):
            print(row(name, by[name]))
    print(row("ALL", rs))
    # difficulty split inside each tier (level 0 / 2 / 4)
    lv = defaultdict(list)
    for r in rs:
        if r["tag"][-1].isdigit():
            lv[(tier_of(r["tag"]), r["tag"][-1])].append(r)
    print("  exit% by level:", "  ".join(
        f"{name.split()[0]}:" + "/".join(f"{100 * sum(x['exit'] for x in lv[(name, l)]) / len(lv[(name, l)]):.0f}"
                                         for l in "024" if lv.get((name, l)))
        for name, _ in TIERS if any(lv.get((name, l)) for l in "024")))
