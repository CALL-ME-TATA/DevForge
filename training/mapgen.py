"""Procedural training maps: room lattice -> ASCII grid -> UDMF (zdoom namespace) WAD.

Grid legend (64-unit cells): '#' wall, '.' floor, 'D' door, 'R'/'B'/'Y' locked door (red/blue/yellow),
'E' exit pad. Things are placed separately. Rooms sit on a lattice of S x S cell slots; every room
contains its slot's centre row and column, so neighbouring rooms connect with straight 1-wide corridors.

Families mirror the evaluation tiers: arena (combat), multi (rooms + doors), key (key -> locked door
-> exit), gauntlet (all + more monsters, scarce items). `d` in [0, 1] scales size and danger.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))
from doom_env.wad import Lump, write_wad  # noqa: E402

CELL = 64
S = 9  # slot size in cells
LOCKS = {"R": 1, "B": 2, "Y": 3}  # mapinfo.py LOCK_RED/BLUE/YELLOW
KEY_THING = {"R": 13, "B": 5, "Y": 6}  # red/blue/yellow keycard
MONSTERS = [(3004, 1.0), (9, 0.6), (3001, 1.0), (3002, 0.4)]  # zombieman, shotgun guy, imp, demon
ITEMS = [2011, 2012, 2007, 2008, 2018, 2001]  # stimpack, medikit, clip, shells, armor, shotgun
# --- extended content (generate(..., extended=True)); drawn from a SEPARATE random stream so the
# original maps of every seed stay byte-identical (training pool, held-out eval and test sets) ---
# high-DPS / tanky monsters the evaluation maps use: weight per unit of difficulty d (absent at d=0)
TOUGH_MONSTERS = [(65, 1.0), (3005, 0.7), (69, 0.5), (58, 0.5), (3006, 0.5), (3003, 0.15)]
# chaingunner, cacodemon, hell knight, spectre, lost soul, baron of hell
EXTRA_ITEMS = [2002, 82, 2048, 2049, 2012, 2018]  # chaingun, super shotgun, bullet box, shell box, medikit, armor
SWITCH_EXIT_P = 0.35  # share of extended maps whose exit is a wall switch (USE) instead of a floor pad
SWITCH_TEX = "SW1STARG"
# realism (Freedoom diagnosis: stuck on barrels/pillars, damage from liquids, guns running dry)
OBSTACLES = [2035, 48, 2028, 30, 31, 32]  # barrel, tech column, floor lamp, tall/short green & red columns
AMMO_ITEMS = (2007, 2008, 2048, 2049)  # clip, shells, bullet box, shell box
REALISM_P = 0.5  # share of extended maps that get the realism block
SCARCE_P = 0.3  # share of extended maps with almost no ammo pickups
HAZARD_P = 0.4  # share of extended maps with damaging floor patches ('H' cells)
HAZARD_TEX, HAZARD_DMG = "NUKAGE1", 5  # 5 HP every 32 tics while standing in it


def _route_cells(g, start, goals):
    """Cells on shortest paths from start to each goal (4-connected, any non-wall cell passable)."""
    prev, q = {start: None}, [start]
    for cy, cx in q:
        for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
            if 0 <= ny < len(g) and 0 <= nx < len(g[ny]) and g[ny][nx] != "#" and (ny, nx) not in prev:
                prev[(ny, nx)] = (cy, cx)
                q.append((ny, nx))
    cells = set()
    for gl in goals:
        node = gl
        while node is not None and node in prev:
            cells.add(node)
            node = prev[node]
    return cells


def switch_side(grid, y, x):
    """Wall side carrying the exit switch for an 'X' cell: first wall neighbour in the order
    right, left, up, down (to_udmf and the training teacher both use this)."""
    for dy, dx in ((0, 1), (0, -1), (-1, 0), (1, 0)):
        ny, nx = y + dy, x + dx
        if not (0 <= ny < len(grid) and 0 <= nx < len(grid[ny])) or grid[ny][nx] == "#":
            return dy, dx
    return None
FAMILIES = ("tiny", "arena", "multi", "key", "gauntlet")


def _lattice(rng: random.Random, rows: int, cols: int, min_room: int = 3, loop_p: float = 0.15):
    """Rooms on a rows x cols lattice + spanning tree (+ a few loops). Returns grid, rooms, edges."""
    H, W = rows * S, cols * S
    g = [["#"] * W for _ in range(H)]
    rooms = {}
    for r in range(rows):
        for c in range(cols):
            cr, cc = r * S + S // 2, c * S + S // 2
            h, w = rng.randint(min_room, S - 2), rng.randint(min_room, S - 2)
            r0 = rng.randint(max(r * S + 1, cr - h + 1), min(cr, r * S + S - 1 - h))
            c0 = rng.randint(max(c * S + 1, cc - w + 1), min(cc, c * S + S - 1 - w))
            for y in range(r0, r0 + h):
                for x in range(c0, c0 + w):
                    g[y][x] = "."
            rooms[(r, c)] = (r0, c0, h, w)
    # random spanning tree (randomised DFS), then extra loop edges
    nodes = list(rooms)
    stack, edges = [rng.choice(nodes)], []
    seen = {stack[0]}
    while stack:
        cur = stack[-1]
        nb = [(cur[0] + dr, cur[1] + dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))]
        nb = [n for n in nb if n in rooms and n not in seen]
        if not nb:
            stack.pop()
            continue
        n = rng.choice(nb)
        seen.add(n)
        edges.append((cur, n))
        stack.append(n)
    for a in nodes:
        for n in ((a[0], a[1] + 1), (a[0] + 1, a[1])):
            if n in rooms and (a, n) not in edges and (n, a) not in edges and rng.random() < loop_p:
                edges.append((a, n))
    corridors = {}
    for a, b in edges:
        cells = _corridor(g, rooms, a, b)
        corridors[frozenset((a, b))] = cells
    return g, rooms, edges, corridors


def _corridor(g, rooms, a, b):
    """Carve a straight corridor between lattice-neighbour rooms; return its cells (outside rooms)."""
    if a > b:
        a, b = b, a
    (r0a, c0a, ha, wa), (r0b, c0b, hb, wb) = rooms[a], rooms[b]
    cells = []
    if a[0] == b[0]:  # horizontal neighbours share the slot centre row
        y = a[0] * S + S // 2
        cells = [(y, x) for x in range(c0a + wa, c0b)]
    else:
        x = a[1] * S + S // 2
        cells = [(y, x) for y in range(r0a + ha, r0b)]
    for y, x in cells:
        g[y][x] = "."
    return cells


def _bfs_tree(edges, start):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    dist, parent, q = {start: 0}, {start: None}, [start]
    for n in q:
        for m in adj.get(n, []):
            if m not in dist:
                dist[m], parent[m] = dist[n] + 1, n
                q.append(m)
    return dist, parent, adj


SLOT_RANGE = (6, 13)  # extended maps: per-map slot size -> room sizes / corridor lengths vary (structural DR)


def generate(family: str, d: float, seed: int, extended: bool = False):
    if not extended:
        return _generate(family, d, seed)
    # structural domain randomisation (Tobin et al. 2017; PAIRED/ACCEL): a per-map lattice slot size, from
    # its own random stream, so the agent cannot overfit one room/corridor geometry
    global S
    s_default = S
    min_room = 5 if family == "arena" else 3  # as in _generate: rooms need S - 2 >= min_room (+1 margin)
    S = random.Random(seed * 31337 + 7).randint(max(SLOT_RANGE[0], min_room + 3), SLOT_RANGE[1])
    try:
        grid, things = _generate(family, d, seed)
    finally:
        S = s_default
    return _extend(family, d, seed, grid, things)


def _extend(family, d, seed, grid, things):
    """Extended content from its own random stream: upgrade some monsters to high-DPS / tanky types
    (more with difficulty, most in gauntlets), add weapons/ammo/health to make them winnable, and turn
    some exits into wall switches (USE)."""
    rng = random.Random(seed * 7919 + 104729)
    g = [list(r) for r in grid]
    mon_ids = {m for m, _ in MONSTERS}
    p_tough = {"arena": 0.35, "gauntlet": 0.5}.get(family, 0.25) * d
    out = []
    for y, x, t, a in things:
        if t in mon_ids and rng.random() < p_tough:
            t = rng.choices([m for m, _ in TOUGH_MONSTERS], [w for _, w in TOUGH_MONSTERS])[0]
        out.append((y, x, t, a))
    used = {(y, x) for y, x, _, _ in out}
    free = [(y, x) for y in range(len(g)) for x in range(len(g[y])) if g[y][x] == "." and (y, x) not in used]
    n_extra = round((1 + 3 * d) * (1.5 if family in ("arena", "gauntlet") else 0.5))
    for _ in range(n_extra):
        if not free:
            break
        p = free.pop(rng.randrange(len(free)))
        out.append((p[0], p[1], rng.choice(EXTRA_ITEMS), 0))
    if family != "arena" and rng.random() < SWITCH_EXIT_P:
        ex = [(y, x) for y in range(len(g)) for x in range(len(g[y])) if g[y][x] == "E"]
        if ex:
            # the exit pad sits mid-room: move it to the nearest floor cell of the same room that touches a
            # wall (BFS over floor), then mount the switch on that wall
            ey, ex_ = ex[0]
            seen, q, target = {(ey, ex_)}, [(ey, ex_)], None
            occupied = {(y, x) for y, x, _, _ in out}
            for cy, cx in q:
                if (cy, cx) != (ey, ex_) and (cy, cx) not in occupied and switch_side(g, cy, cx) is not None:
                    target = (cy, cx)
                    break
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < len(g) and 0 <= nx < len(g[ny]) and (ny, nx) not in seen and g[ny][nx] == ".":
                        seen.add((ny, nx))
                        q.append((ny, nx))
            if target is not None:
                g[ey][ex_] = "."
                g[target[0]][target[1]] = "X"
    rng2 = random.Random(seed * 104723 + 17)  # own stream: older extended content is unchanged
    # --- realism: ammo scarcity, obstacles, hazard floors (never on the route to keys / exit) ---
    # Only on REALISM_P of maps: the rest rehearse the previous distribution, which prevents forgetting
    # when fine-tuning (Rolnick et al. 2019; Wolczyk et al. 2024).
    if rng2.random() >= REALISM_P:
        return ["".join(r) for r in g], out
    if rng2.random() < SCARCE_P:
        ammo = [th for th in out if th[2] in AMMO_ITEMS]
        keep = set(rng2.sample(range(len(ammo)), min(1, len(ammo))))
        drop = {id(th) for i, th in enumerate(ammo) if i not in keep}
        out = [th for th in out if id(th) not in drop]
    start = next(((y, x) for y, x, t, _ in out if t == 1), None)
    if start is not None:
        goals = [(y, x) for y in range(len(g)) for x in range(len(g[y])) if g[y][x] in "EX"]
        goals += [(y, x) for y, x, t, _ in out if t in (5, 6, 13)]
        route = _route_cells(g, start, goals)
        near_route = {(y + dy, x + dx) for y, x in route for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        occupied = {(y, x) for y, x, _, _ in out}
        spots = [(y, x) for y in range(len(g)) for x in range(len(g[y]))
                 if g[y][x] == "." and (y, x) not in near_route and (y, x) not in occupied
                 and all(g[y + dy][x + dx] not in "DRBY" for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                         if 0 <= y + dy < len(g) and 0 <= x + dx < len(g[y + dy]))]
        rng2.shuffle(spots)
        for _ in range(min(len(spots), round(1 + 4 * d))):  # obstacles off the route
            p = spots.pop()
            out.append((p[0], p[1], rng2.choice(OBSTACLES), 0))
        if spots and rng2.random() < HAZARD_P:  # 1-3 hazard patches of up to 4 cells
            for _ in range(rng2.randint(1, 3)):
                if not spots:
                    break
                y, x = spots.pop()
                for cy, cx in ((y, x), (y + 1, x), (y, x + 1), (y + 1, x + 1)):
                    if (0 <= cy < len(g) and 0 <= cx < len(g[cy]) and g[cy][cx] == "." and (cy, cx) not in near_route
                            and (cy, cx) not in occupied):
                        g[cy][cx] = "H"
    return ["".join(r) for r in g], out


def _generate(family: str, d: float, seed: int):
    """Return (grid rows as strings, things list [(x, y, type, angle)])."""
    rng = random.Random(seed)
    if family == "tiny":
        rows, cols = 1, rng.randint(1, 2)
    elif family == "arena":
        rows, cols = 1, rng.randint(1, 2)
    else:
        n = 2 + round(d * 2)
        rows, cols = rng.randint(max(1, n - 1), n), rng.randint(n, n + 1)
    # no loops in key families: a loop could bypass a locked door
    g, rooms, edges, corridors = _lattice(rng, rows, cols, min_room=5 if family == "arena" else 3,
                                          loop_p=0.0 if family in ("key", "gauntlet") else 0.15)
    nodes = list(rooms)
    start = rng.choice(nodes)
    dist, parent, adj = _bfs_tree(edges, start)
    far = max(nodes, key=lambda n: (dist[n], rng.random()))

    def centre(n):
        r0, c0, h, w = rooms[n]
        return r0 + h // 2, c0 + w // 2

    def rand_cell(n, avoid=()):
        r0, c0, h, w = rooms[n]
        for _ in range(50):
            p = (rng.randrange(r0, r0 + h), rng.randrange(c0, c0 + w))
            if g[p[0]][p[1]] == "." and p not in avoid:
                return p
        return centre(n)

    used = set()
    things = []
    sy, sx = rand_cell(start)
    used.add((sy, sx))
    things.append((sy, sx, 1, rng.choice(range(0, 360, 45))))

    # doors on corridors (plain), locks on the path into the exit room for key families
    door_p = {"tiny": 0.0, "arena": 0.0}.get(family, 0.3 + 0.4 * d)
    for cells in corridors.values():
        if len(cells) >= 1 and rng.random() < door_p:
            y, x = cells[len(cells) // 2]
            g[y][x] = "D"
    keys = []
    if family in ("key", "gauntlet") and far != start:
        n_locks = 1 + (d > 0.6 and rng.random() < 0.5)
        colours = rng.sample(list(LOCKS), n_locks)
        target, blocked = far, set()
        for col in colours:
            # lock the corridor into `target`; key goes in a room reachable without crossing it
            if parent.get(target) is None:
                break
            cells = corridors[frozenset((target, parent[target]))]
            y, x = cells[len(cells) // 2]
            g[y][x] = col
            blocked |= _subtree(target, parent, adj)
            options = [n for n in nodes if n not in blocked and n != start] or [n for n in nodes if n not in blocked]
            # prefer far dead ends for the key
            kroom = max(options, key=lambda n: (len(adj.get(n, [])) == 1, dist[n], rng.random()))
            keys.append((kroom, col))
            target = kroom  # next lock (if any) guards the key room
    ey, ex = centre(far)
    if (ey, ex) in used:
        ey, ex = rand_cell(far, used)
    g[ey][ex] = "E"
    used.add((ey, ex))
    for kroom, col in keys:
        p = rand_cell(kroom, used)
        used.add(p)
        things.append((p[0], p[1], KEY_THING[col], 0))

    n_mon = {"tiny": 0, "arena": 2 + round(6 * d), "multi": round(4 * d), "key": round(4 * d),
             "gauntlet": 4 + round(8 * d)}[family]
    n_item = {"tiny": 0, "arena": 3, "multi": 2, "key": 2, "gauntlet": 2}[family]
    safe = {start} if family != "arena" else set()
    mon_rooms = [n for n in nodes if n not in safe] or nodes
    for _ in range(n_mon):
        t = rng.choices([m for m, _ in MONSTERS], [w * (1 + d if m in (3001, 3002) else 1) for m, w in MONSTERS])[0]
        p = rand_cell(rng.choice(mon_rooms), used)
        if p == (sy, sx) or abs(p[0] - sy) + abs(p[1] - sx) < 3:
            continue
        used.add(p)
        things.append((p[0], p[1], t, rng.choice(range(0, 360, 45))))
    for _ in range(n_item):
        p = rand_cell(rng.choice(nodes), used)
        used.add(p)
        things.append((p[0], p[1], rng.choice(ITEMS), 0))
    return ["".join(row) for row in g], things


def _subtree(root, parent, adj):
    """Rooms reachable from `root` without going through parent[root] (tree edges only)."""
    out, q = {root}, [root]
    for n in q:
        for m in adj.get(n, []):
            if m not in out and m != parent[root] and parent.get(m) == n:
                out.add(m)
                q.append(m)
    return out


# ---------------------------------------------------------------- grid -> UDMF
def to_udmf(grid: list[str], things) -> str:
    H, W = len(grid), len(grid[0])
    # sector id per non-wall cell: floor cells by 4-connected component, each door/exit cell its own
    sid = [[-1] * W for _ in range(H)]
    sectors = []  # (kind, char)
    for y in range(H):
        for x in range(W):
            ch = grid[y][x]
            if ch == "#" or sid[y][x] >= 0:
                continue
            k = len(sectors)
            sectors.append(ch)
            if ch != ".":
                sid[y][x] = k
                continue
            q = [(y, x)]
            sid[y][x] = k
            for cy, cx in q:
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == "." and sid[ny][nx] < 0:
                        sid[ny][nx] = k
                        q.append((ny, nx))

    verts, vidx, sides, lines = [], {}, [], []

    def v(x, y):
        if (x, y) not in vidx:
            vidx[(x, y)] = len(verts)
            verts.append((x, y))
        return vidx[(x, y)]

    def side(sec, mid="-", top="-"):
        sides.append(f'sidedef {{ sector = {sec}; texturetop = "{top}"; texturemiddle = "{mid}"; texturebottom = "-"; }}')
        return len(sides) - 1

    def cell_val(y, x):
        return grid[y][x] if 0 <= y < H and 0 <= x < W else "#"

    for y in range(H):
        for x in range(W):
            a = cell_val(y, x)
            if a == "#":
                continue
            X0, X1, YT = x * CELL, (x + 1) * CELL, -y * CELL
            YB = YT - CELL
            # (neighbour cell, v1, v2) with this cell on the right of v1->v2
            for (ny, nx), p1, p2 in (((y, x + 1), (X1, YT), (X1, YB)), ((y, x - 1), (X0, YB), (X0, YT)),
                                     ((y - 1, x), (X0, YT), (X1, YT)), ((y + 1, x), (X1, YB), (X0, YB))):
                b = cell_val(ny, nx)
                if b == "#":
                    mid = "DOORTRAK" if a in "DRBY" else "STARTAN2"
                    special = ""
                    if a == "X" and switch_side(grid, y, x) == (ny - y, nx - x):  # wall exit switch
                        mid, special = SWITCH_TEX, "special = 243; playeruse = true;"
                    extra = f" {special}" if special else ""
                    lines.append(f"linedef {{ v1 = {v(*p1)}; v2 = {v(*p2)}; sidefront = {side(sid[y][x], mid)}; blocking = true; dontpegbottom = true;{extra} }}")
                    continue
                sa, sb = sid[y][x], sid[ny][nx]
                if sa == sb:
                    continue
                special = ""
                if b in "DRBY" and a not in "DRBY":  # room side faces the door; door sector on the back
                    if b == "D":
                        special = "special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; repeatspecial = true;"
                    else:
                        special = f"special = 13; arg0 = 0; arg1 = 16; arg2 = 150; arg3 = {LOCKS[b]}; playeruse = true; repeatspecial = true;"
                    top = "BIGDOOR2"
                elif b == "E" and a != "E":
                    special, top = "special = 243; playercross = true;", "-"
                elif a in "DRBYEX" or (a == "H" and b != "H") or (b == a and a in ".H" and sa > sb):
                    continue  # emitted from the other side
                else:
                    top = "-"
                f, bk = side(sa, "-", top), side(sb, "-", top)
                lines.append(f"linedef {{ v1 = {v(*p1)}; v2 = {v(*p2)}; sidefront = {f}; sideback = {bk}; twosided = true; {special} }}")

    out = ['namespace = "zdoom";']
    out += [f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}" for x, y in verts]
    for ch in sectors:
        ceil = 0 if ch in "DRBY" else 128
        floor_tex = "GATE3" if ch == "E" else (HAZARD_TEX if ch == "H" else "FLOOR0_1")
        dmg = f" damageamount = {HAZARD_DMG}; damageinterval = 32;" if ch == "H" else ""
        out.append(f'sector {{ texturefloor = "{floor_tex}"; textureceiling = "CEIL1_1"; heightfloor = 0; heightceiling = {ceil}; lightlevel = 192;{dmg} }}')
    out += sides + lines
    for cy, cx, t, ang in things:
        out.append(f"thing {{ x = {cx * CELL + 32:.1f}; y = {-cy * CELL - 32:.1f}; angle = {ang}; type = {t}; "
                   "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }")
    return "\n".join(out) + "\n"


def write_map(path: str | Path, grid, things) -> str:
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", to_udmf(grid, things).encode()), Lump("ENDMAP", b"")])
    return str(path)


def build_pool(out_dir: str | Path, per_bin: int = 40, levels=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Write per_bin maps for every (family, difficulty) bin. Returns {(family, d): [wad paths]}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = {}
    for fi, fam in enumerate(FAMILIES):
        for li, d in enumerate(levels):
            paths = []
            for i in range(per_bin):
                seed = fi * 1_000_000 + li * 10_000 + i
                grid, things = generate(fam, d, seed)
                paths.append(write_map(out / f"{fam}_{li}_{i}.wad", grid, things))
            pool[(fam, d)] = paths
    return pool


if __name__ == "__main__":
    for fam in FAMILIES:
        grid, things = generate(fam, 0.7, 1)
        print(fam, len(things), "things")
        print("\n".join(grid))
