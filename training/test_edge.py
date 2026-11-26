"""Unit tests for the edge-case miner (no engine needed):  python -m pytest training/test_edge.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import EdgeBank, pick_case, still_window  # noqa: E402

SNAPS = [{"t": t} for t in range(0, 400, 32)]


def walk(n, d0=50):
    return [(i * 20.0, 0.0, max(0, d0 - i)) for i in range(n)]


def test_still_window():
    tr = walk(10) + [(200.0, 0.0, 40)] * 130 + walk(5)
    n, i0 = still_window(tr)
    assert n >= 129 and 5 <= i0 <= 10


def test_exit_is_never_a_case():
    assert pick_case(walk(100), SNAPS, dead=False, exit_=True) is None


def test_death_replays_from_before_the_death():
    kind, s = pick_case(walk(300), SNAPS, dead=True, exit_=False)
    assert kind == "death" and s["t"] <= 300 - 40


def test_stuck():
    tr = walk(40) + [(800.0, 0.0, 10)] * 200
    kind, s = pick_case(tr, SNAPS, dead=False, exit_=False)
    assert kind == "stuck" and s["t"] <= 48


def test_plateau_and_door():
    tr = walk(40, d0=50) + [(800.0 + (i % 2) * 100, 0.0, 30) for i in range(200)]
    kind, s = pick_case(tr, SNAPS, dead=False, exit_=False)
    assert kind == "plateau" and s["t"] <= 39
    grid = ["#" * 20 for _ in range(3)]
    grid[1] = "#" * 12 + "D" + "#" * 7  # door right next to the closest-approach cell (x=780 -> col 12)
    kind, _ = pick_case(tr, SNAPS, dead=False, exit_=False, grid=[list(r) for r in grid])
    assert kind == "door"


def test_bank_retires_learned_and_hopeless(tmp_path):
    b = EdgeBank(tmp_path, cap=3)
    for i in range(5):
        (tmp_path / f"{i}.sav").write_bytes(b"x")
        b.add(dict(id=str(i), kind="death", fam="arena", li=0, sav=str(tmp_path / f"{i}.sav")), it=i)
    assert len(b.cases) == 3 and b.stats["retired:evicted"] == 2
    a, c = b.take(random.Random(0), 2)
    for _ in range(3):
        b.result(a["id"], True)
    for _ in range(8):
        b.result(c["id"], False)
    assert b.stats["retired:learned"] == 1 and b.stats["retired:hopeless"] == 1 and len(b.cases) == 1
    assert not Path(a["sav"]).exists()
