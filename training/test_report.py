"""Tables for the never-seen test set (evaluate.py JSON): per tier x difficulty, per-tier totals,
hand-made Freedoom levels, and failure breakdown. Several files can be compared side by side.

  python training/test_report.py name1=results1.json name2=results2.json
"""
import json
import sys
from collections import defaultdict

FAMS = [("tiny", "T0 Basics"), ("arena", "T1 Combat"), ("multi", "T2 Multi-room"), ("key", "T3 Keycards"),
        ("gauntlet", "T4 Gauntlet")]


def load(arg):
    name, path = arg.split("=", 1)
    return name, json.load(open(path))


def main():
    runs = [load(a) for a in sys.argv[1:]]
    for name, r in runs:
        gen = [x for x in r if not x["tag"].startswith("freedoom")]
        fd = [x for x in r if x["tag"].startswith("freedoom")]
        print(f"\n### {name}: never-seen generated maps ({len(gen)}) — levels cleared per difficulty")
        print("| Tier | L0 | L1 | L2 | L3 | L4 | Total | Died | Kills/map | Keys/map | Cells/map |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for fam, label in FAMS:
            g = [x for x in gen if x["tag"].rstrip("0123456789") == fam]
            if not g:
                continue
            by = defaultdict(list)
            for x in g:
                by[int(x["tag"][len(fam):])].append(x)
            lv = " | ".join(f"{sum(e['exit'] for e in by[l])}/{len(by[l])}" if by[l] else "–" for l in range(5))
            n = len(g)
            print(f"| {label} | {lv} | **{sum(e['exit'] for e in g)}/{n}** ({100 * sum(e['exit'] for e in g) / n:.0f}%) | "
                  f"{100 * sum(e['dead'] for e in g) / n:.0f}% | {sum(e['kills'] for e in g) / n:.1f} | "
                  f"{sum(e['keys'] for e in g) / n:.2f} | {sum(e['cells'] for e in g) / n:.0f} |")
        n = len(gen)
        print(f"| **All generated** | | | | | | **{sum(e['exit'] for e in gen)}/{n}** ({100 * sum(e['exit'] for e in gen) / n:.0f}%) | "
              f"{100 * sum(e['dead'] for e in gen) / n:.0f}% | {sum(e['kills'] for e in gen) / n:.1f} | "
              f"{sum(e['keys'] for e in gen) / n:.2f} | {sum(e['cells'] for e in gen) / n:.0f} |")
        if fd:
            print(f"\n**{name}: hand-made Freedoom levels never used in training**")
            print("| Level | Result | Steps | Kills | Keys | Cells explored |")
            print("|---|---|---|---|---|---|")
            for x in sorted(fd, key=lambda e: e["map"]):
                out = "exit" if x["exit"] else ("died" if x["dead"] else "time up")
                print(f"| {x['map'].split(':')[1]} | {out} | {x['steps']} | {x['kills']} | {int(x['keys'])} | {x['cells']} |")
            print(f"Freedoom cleared: {sum(e['exit'] for e in fd)}/{len(fd)}")
        fails = [x for x in gen if not x["exit"]]
        if fails:
            died = sum(x["dead"] for x in fails)
            key_none = sum(1 for x in fails if x["tag"].startswith("key") and not x["dead"] and x["keys"] == 0)
            key_after = sum(1 for x in fails if x["tag"].startswith("key") and not x["dead"] and x["keys"] > 0)
            other = len(fails) - died - key_none - key_after
            print(f"\n{name} failures ({len(fails)}): died {died} | keycard: never found key {key_none}, "
                  f"failed after key {key_after} | ran out of time elsewhere {other}")


if __name__ == "__main__":
    main()
