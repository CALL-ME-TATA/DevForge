"""Privileged TEACHER for imitation learning (DAgger / "learning by cheating"). TRAINING ONLY.

Never imported by submission/. It reads the generator's layout of the current training map (through
GoalShaper) and the engine's true pose, and outputs the action the student should have taken, as the
6 head indices the network uses: (move F/B/-, strafe L/R/-, turn L/R/-, attack, use, next weapon).

Scenarios handled (see README "Teacher"):
 navigation  - goal = exit if reachable with the keys held, else the nearest still-needed reachable key
               (chains of several locks fall out of this rule); BFS corridor distance from GoalShaper
             - steer at the farthest cell of the shortest path that is in straight line of sight
               (cuts corners without clipping walls); rotate in place when the error is large
             - plain doors and (with the key) locked doors: press USE on alternate steps while facing it
               up close (USE only fires on button-down; locked doors close again after ~4 s)
             - stuck against a wall / door frame / monster: strafe out, alternating sides
             - position outside the known layout -> no label (the step is excluded from the loss)
 combat      - only uses what the STUDENT can see (visible_entities), so the behaviour is imitable
             - turn to the nearest enemy, fire only when aligned (tolerance shrinks with distance),
               back off from enemies that are very close, never shoot through a closer barrel
             - dodge an approaching projectile by strafing
             - empty weapon while other ammo exists: switch weapon; no ammo at all: just navigate
             - within 3 cells of the exit: ignore enemies and finish the level
 never       - drives backward without an enemy in view, spins in place, or switches a loaded weapon
"""
from __future__ import annotations

import math

import numpy as np

CELL = 64.0
ENEMY_TYPES = (1, 2, 3)  # doom_env.spaces.ENTITY_TYPES: enemy_melee, enemy_hitscan, enemy_projectile
PROJECTILE, BARREL = 4, 10
KEY_IDX = {"B": 0, "Y": 1, "R": 2}  # lock colour -> vitals/keys index (Blue/Yellow/Red card)
F, B, N = 0, 1, 2  # move head
L, R = 0, 1  # strafe / turn heads (2 = none)
INF = 1 << 30
USE_COOLDOWN = 20  # steps (~2.3 s at 8.75 steps/s) between USE presses
# Steering hysteresis: start turning only when clearly off course, stop once nearly aligned. The engine's
# turn speed ramps up while the button is held, so a tight deadband overshoots and zig-zags.
TURN_START, TURN_STOP = 9.0, 0.0  # = plain 9 deg deadband (hysteresis 15/5 cost keycard clears)
# Aiming: one TURN press rotates ~7 deg, so a ~1.5 deg deadband jittered left/right on every step an enemy
# was in view (38% of all steps). Doom's pistol/chaingun spread is ~+-5.6 deg anyway.
AIM_DEADBAND = 6.0
SIGHT_RAYS, SIGHT_STEPS = 32, 32  # frontier teacher: 32 sight lines x 32 steps of 32 units = 1024 u
ROTATE_IN_PLACE = 40.0  # heading error above which the teacher stops to turn (90 tested: 74/90 vs 81/90)


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def idx_to_buttons(idx) -> np.ndarray:
    """numpy twin of model.heads_to_buttons for one action."""
    b = np.zeros(9, np.int64)
    for k, (i0, i1) in enumerate(((0, 1), (2, 3), (4, 5))):
        b[i0], b[i1] = int(idx[k] == 0), int(idx[k] == 1)
    b[6], b[7], b[8] = idx[3], idx[4], idx[5]
    return b


class Teacher:
    def __init__(self, shaper, track_executed: bool = False, explore: bool = False):
        self.sh = shaper  # GoalShaper of this exact map variant (grid, keys, exit)
        # DAgger: the student drives, so the USE cooldown must follow the presses actually executed
        # (note_executed), not the teacher's own labels
        self.track_executed = track_executed
        # Information-matched teacher (closes the "imitation gap", Weihs et al. 2021; Choudhury et al. 2018):
        # until the current goal (needed key / exit) has been SEEN, explore like Yamauchi (1997) -- nearest
        # frontier of the seen area, reached through seen space -- instead of walking straight to a goal the
        # student cannot know about. Once seen, take the shortest route as before.
        self.explore = explore
        self.reset()

    def note_executed(self, buttons):
        if buttons[7] and not self.exec_use_prev:
            self.use_cd = USE_COOLDOWN
        self.exec_use_prev = int(buttons[7])

    def reset(self):
        self.pos_hist = []
        self.fwd_hist = []
        self.unstick, self.unstick_dir = 0, 1
        self.use_prev = self.wpn_prev = 0
        self.use_cd = 0  # USE on a moving door reverses it (CONTRACT.md): press once, then wait ~2 s
        self.exec_use_prev = 0
        self.dodge_dir = 1
        self.turn_dir = None  # steering hysteresis state (L / R / None)
        self.seen = set()  # grid cells the student could have seen this episode

    # ------------------------------------------------------------------ what the student can know
    def _observe(self, x, y):
        """Mark cells visible from (x, y): SIGHT_RAYS sight lines up to 1024 units (the student's ray
        range); walls and doors block the view (the blocking cell itself is seen)."""
        g = self.sh.g
        for k in range(SIGHT_RAYS):
            a = 2.0 * math.pi * k / SIGHT_RAYS
            dx, dy = math.cos(a), math.sin(a)
            for s in range(1, SIGHT_STEPS + 1):
                px, py = x + dx * s * 32.0, y + dy * s * 32.0
                r, c = math.floor(-py / CELL), math.floor(px / CELL)
                if not (0 <= r < len(g) and 0 <= c < len(g[r])):
                    break
                self.seen.add((r, c))
                ch = g[r][c]
                if ch == "#" or ch == "D" or ch in KEY_IDX:
                    break

    def _goal_known(self, kind, held):
        if kind == "exit":
            return any(p in self.seen for p in self.sh.exit)
        return any(p in self.seen for col, p in self.sh.keys.items() if col not in held)

    def _frontier_path(self, r, c, held, n=6):
        """Yamauchi (1997): breadth-first search through SEEN passable cells to the nearest unseen
        passable cell (the frontier). Returns the first n cells of that path, or None."""
        start, prev, q, i = (r, c), {(r, c): None}, [(r, c)], 0
        target = None
        while i < len(q):
            cur = q[i]
            i += 1
            if cur != start and cur not in self.seen:
                target = cur
                break
            y, x = cur
            for nb in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if nb not in prev and self._free(*nb, held):
                    prev[nb] = cur
                    q.append(nb)
        if target is None:
            return None
        path, node = [], target
        while node != start:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path[:n]

    def _press_use(self):
        if self.track_executed:
            return int(self.use_cd == 0 and not self.exec_use_prev)
        if self.use_cd == 0 and not self.use_prev:
            self.use_cd = USE_COOLDOWN
            return 1
        return 0

    # ------------------------------------------------------------------ geometry
    def _free(self, r, c, held, allow_doors=True):
        g = self.sh.g
        if not (0 <= r < len(g) and 0 <= c < len(g[r])):
            return False
        ch = g[r][c]
        if ch == "#":
            return False
        if ch in KEY_IDX:
            return ch in held and allow_doors
        if ch == "D":
            return allow_doors
        return True

    def _los(self, x0, y0, x1, y1, held):
        """Straight segment stays in free, non-door cells (except the end cell)."""
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 12))
        end = (math.floor(-y1 / CELL), math.floor(x1 / CELL))
        for i in range(1, n):
            t = i / n
            # sample the segment and also +-14 units sideways so the player's body (radius 16) fits
            for off in (-14.0, 0.0, 14.0):
                dx, dy = x1 - x0, y1 - y0
                ln = math.hypot(dx, dy) or 1.0
                px, py = x0 + dx * t - dy / ln * off, y0 + dy * t + dx / ln * off
                cell = (math.floor(-py / CELL), math.floor(px / CELL))
                if cell != end and not self._free(*cell, held, allow_doors=False):
                    return False
        return True

    def _path(self, r, c, held, n=6):
        self.sh.dist(c * CELL + 32, -(r * CELL + 32), held)  # fills the cache for `held`
        d_exit, d_key = self.sh.cache[held]
        if 0 <= r < len(d_exit) and 0 <= c < len(d_exit[r]) and d_exit[r][c] < INF:
            field, kind = d_exit, "exit"
        elif d_key is not None and 0 <= r < len(d_key) and 0 <= c < len(d_key[r]) and d_key[r][c] < INF:
            field, kind = d_key, "key"
        else:
            return None, None, None
        path, (y, x) = [], (r, c)
        for _ in range(n):
            if field[y][x] == 0:
                break
            nxt = min(((ny, nx) for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1))
                       if 0 <= ny < len(field) and 0 <= nx < len(field[ny]) and field[ny][nx] < field[y][x]),
                      key=lambda p: field[p[0]][p[1]], default=None)
            if nxt is None:
                break
            path.append(nxt)
            y, x = nxt
        return path, kind, field[r][c]

    # ------------------------------------------------------------------ policy
    def act(self, obs, ps):
        """-> (idx[6] int array, valid). `ps` = env.privileged_state() BEFORE the step."""
        x, y, ang = ps["x"], ps["y"], ps["angle"]
        held = frozenset(c for c, i in KEY_IDX.items() if ps["keys"][i] > 0)
        r, c = math.floor(-y / CELL), math.floor(x / CELL)
        move, strafe, turn, attack, use, wpn = N, 2, 2, 0, 0, 0

        path, kind, dist = self._path(r, c, held)
        valid = path is not None
        if self.explore:
            self._observe(x, y)
            if path is not None and not self._goal_known(kind, held):
                fp = self._frontier_path(r, c, held)
                if fp:  # goal not seen yet: go to the nearest frontier instead
                    path, kind, dist = fp, "explore", None

        # ---- navigation target: farthest path cell in line of sight
        nav_err, target_cell = None, None
        if path:
            tx, ty = path[0][1] * CELL + 32, -(path[0][0] * CELL + 32)
            target_cell = path[0]
            for cell in path[1:]:
                cx, cy = cell[1] * CELL + 32, -(cell[0] * CELL + 32)
                if self._los(x, y, cx, cy, held):
                    tx, ty, target_cell = cx, cy, cell
                else:
                    break
            nav_err = wrap(math.degrees(math.atan2(ty - y, tx - x)) - ang)

        # ---- what the student can see
        ve = obs["visible_entities"]
        m = np.asarray(ve["mask"]).astype(bool)
        types = np.asarray(ve["entity_type"])[m]
        feats = np.asarray(ve["features"])[m]
        ents = [(float(f[0]) * 1024.0, math.degrees(math.atan2(f[1], f[2])), float(f[3]), int(t))
                for f, t in zip(feats, types)]
        enemies = sorted([e for e in ents if e[3] in ENEMY_TYPES])
        vit = obs["vitals"]
        cur_ammo, bullets, shells = vit[3], vit[4], vit[5]
        near_exit = kind == "exit" and dist is not None and dist <= 3

        # ---- weapon: switch only if the current weapon is empty and other ammo exists
        if cur_ammo <= 0 and (bullets > 0 or shells > 0):
            wpn = 0 if self.wpn_prev else 1
        self.wpn_prev = wpn

        engaged = False
        if enemies and not near_exit and (cur_ammo > 0):
            d, rel, _, _ = enemies[0]
            if d < 1100:
                engaged = True
                tol = max(AIM_DEADBAND, math.degrees(math.atan2(18.0, max(d, 1.0))))
                turn = L if rel > AIM_DEADBAND else (R if rel < -AIM_DEADBAND else 2)
                blocked = any(e[3] == BARREL and e[0] < d and e[0] < 220 and abs(e[1] - rel) < 8 for e in ents)
                attack = int(abs(rel) < tol and not blocked)
                move = B if d < 160 else (F if d > 700 and abs(rel) < 20 else N)

        # ---- dodge projectiles coming at us
        for d, rel, vrel, t in ents:
            if t == PROJECTILE and d < 450 and abs(rel) < 35 and vrel < 0:
                strafe = R if rel > 0 else L
                break

        if engaged or path is None or not path:
            self.turn_dir = None
        if not engaged and path is not None:
            if not path:  # standing on the goal cell
                move, turn = F, 2
            else:
                err = nav_err
                if self.turn_dir is not None and (abs(err) < TURN_STOP or (self.turn_dir == L) != (err > 0)):
                    self.turn_dir = None  # aligned (or crossed over): drive straight
                if self.turn_dir is None and abs(err) > TURN_START:
                    self.turn_dir = L if err > 0 else R
                turn = self.turn_dir if self.turn_dir is not None else 2
                # keep moving while turning; rotate in place only for U-turns. A 40 deg threshold made the
                # student (who cannot see the route) copy "stand and rotate" -> spinning in place.
                move = F if abs(err) < ROTATE_IN_PLACE else N
                nxt = path[0]
                ch = self.sh.g[nxt[0]][nxt[1]]
                if ch == "D" or (ch in KEY_IDX and ch in held):
                    dcx, dcy = nxt[1] * CELL + 32, -(nxt[0] * CELL + 32)
                    door_err = wrap(math.degrees(math.atan2(dcy - y, dcx - x)) - ang)
                    if math.hypot(dcx - x, dcy - y) < 110:
                        turn = L if door_err > 9 else (R if door_err < -9 else 2)
                        self.turn_dir = None  # door alignment overrides the cruise steering
                        if abs(door_err) < 30:
                            use = self._press_use()

        # ---- stuck recovery (pressing forward but not moving)
        self.pos_hist = (self.pos_hist + [(x, y)])[-14:]
        self.fwd_hist = (self.fwd_hist + [move == F])[-14:]
        if self.unstick > 0:
            self.unstick -= 1
            strafe = self.unstick_dir
        elif len(self.pos_hist) == 14 and all(self.fwd_hist) and \
                math.hypot(x - self.pos_hist[0][0], y - self.pos_hist[0][1]) < 20:
            self.unstick_dir = L if self.unstick_dir == R else R
            self.unstick = 8
            strafe = self.unstick_dir
            use = self._press_use()  # maybe it is an unopened door

        self.use_cd = max(0, self.use_cd - 1)
        self.use_prev = use
        return np.array([move, strafe, turn, attack, use, wpn], np.int64), valid
