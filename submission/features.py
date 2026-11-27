"""Observation encoding shared by training and the submitted agent (numpy only).

- Dead reckoning: heading from our own turn actions (table refit in-engine, residual 0),
  position from the integrated ego velocities in `vitals`. Frame = spawn pose, no privileged data.
- EgoMap: G x G allocentric grid (64-unit cells) written from rays + visible entities, read as a
  heading-aligned 32 x 32 crop. Writing is deterministic observation encoding; all reading,
  planning and acting is learned (see model.py, which adds learned write channels on top).
"""
from __future__ import annotations

import os

import numpy as np

G, CELL, CROP = 128, 64.0, 32
DOOR_RESET = os.environ.get("DF_DOOR_RESET", "0") == "1"  # experiment switch, off in the submission
N_FIXED = 9  # free, wall, door, locked, key, item, enemy, visits, recent (decaying trail)
RECENT_DECAY = 0.97  # trail half-life ~23 steps
RAY_RANGE, NUM_RAYS = 1024.0, 64
KEY_T, ENEMY_T = 9, (1, 2, 3)
PICKUP_T = (5, 6, 7, 8, 9)  # health, armor, ammo, weapon, key (doom_env.spaces.ENTITY_TYPES)
N_KEYMEM = 2  # [any key held, steps since last key pickup / 200]: "backtrack to the gate" mode
# Landmark compass (goal vector, Wijmans et al. 2020 / MERLIN "right information in the right format"):
# for the nearest REMEMBERED locked door (M[3]) and key (M[4]) on the agent's own map:
# [known, forward component, right component, distance / 2048]. Straight-line bearing, no path search.
LANDMARK_CH = (3, 4)
N_LANDMARK = 4 * len(LANDMARK_CH)
VEC_DIM = 17 + 9 + 40 + N_KEYMEM + N_LANDMARK
# Coarse global map: 16 x 16 heading-aligned cells of 512 units (= 8 x 8 fine cells), max-pooled, centred
# on the agent, so remembered keys / locked doors beyond the 32 x 32 crop (> 1024 units) stay visible.
GMAP, GCELL, GSUB = 16, 512.0, 4
FRONT_RAYS = np.r_[0:8, 56:64]  # +-45 deg cone in front (ray 0 = straight ahead, CCW)
CLEAR_FAR = 384.0  # free space seen this close in the front cone holds no phantom key/item marker
VITALS_SCALE = np.array([100, 100, 7, 50, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 10, 10, 10], np.float32)

# mean heading change (deg) for (turn[t-2], turn[t-1], turn[t]); turn = left - right
# Refit on 19.7k engine steps: with these values the residual is exactly 0 (the engine's turn ramp is
# deterministic given the last three turn inputs). Keys with turn[t] == 0 give 0.
_TURN = {(-1, -1, -1): -14.06, (-1, -1, 1): 14.06, (-1, 0, -1): -7.03, (-1, 0, 1): 7.03,
         (-1, 1, -1): -14.06, (-1, 1, 1): 14.06, (0, -1, -1): -12.3, (0, -1, 1): 12.3,
         (0, 0, -1): -7.03, (0, 0, 1): 7.03, (0, 1, -1): -12.3, (0, 1, 1): 12.3,
         (1, -1, -1): -14.06, (1, -1, 1): 14.06, (1, 0, -1): -7.03, (1, 0, 1): 7.03,
         (1, 1, -1): -14.06, (1, 1, 1): 14.06}

_RAY_ANG = np.arange(NUM_RAYS) * (2 * np.pi / NUM_RAYS)  # CCW from forward
_SAMPLES = np.arange(1, 33) * 32.0  # free-space samples along each ray
_ci, _cj = np.meshgrid(np.arange(CROP), np.arange(CROP), indexing="ij")
_FWD = ((CROP / 2 - 0.5) - _ci).ravel() * CELL  # crop row 0 = far ahead
_RIGHT = (_cj - (CROP / 2 - 0.5)).ravel() * CELL
PAD = G * G  # index of an always-empty cell
# global-map sample points: GSUB x GSUB per coarse cell, same orientation as the crop (row 0 = far ahead)
_gi, _gj = np.meshgrid(np.arange(GMAP), np.arange(GMAP), indexing="ij")
_gsub = (np.arange(GSUB) + 0.5) / GSUB
_GFWD = np.broadcast_to(((GMAP / 2 - 1 - _gi)[..., None, None] + _gsub[None, None, :, None]) * GCELL,
                        (GMAP, GMAP, GSUB, GSUB)).ravel()
_GRIGHT = np.broadcast_to(((_gj - GMAP / 2)[..., None, None] + _gsub[None, None, None, :]) * GCELL,
                          (GMAP, GMAP, GSUB, GSUB)).ravel()


class Features:
    def __init__(self):
        self.reset()

    def reset(self):
        self.map = np.zeros((N_FIXED, G * G + 1), np.float32)
        self.x = self.y = self.h = 0.0
        self.turns = [0, 0]
        self.n = 0  # integrated steps since reset
        self.keys_held = 0.0
        self.since_key = 0  # steps since the last key pickup (counts only once a key is held)
        self.prev_vel = np.zeros(2, np.float32)
        self.prev_action = np.zeros(9, np.float32)
        self.first = True
        self.door_known = False

    def _landmark(self, ch: int, h: float):
        """Bearing to the nearest remembered cell of channel `ch`, in the heading frame."""
        idx = np.flatnonzero(self.map[ch, :PAD] > 0.5)
        if idx.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        dx = ((idx % G) - G // 2 + 0.5) * CELL - self.x
        dy = ((idx // G) - G // 2 + 0.5) * CELL - self.y
        d2 = dx * dx + dy * dy
        k = int(np.argmin(d2))
        dist = max(float(np.sqrt(d2[k])), 1.0)
        fwd = (dx[k] * np.cos(h) + dy[k] * np.sin(h)) / dist
        right = (dx[k] * np.sin(h) - dy[k] * np.cos(h)) / dist
        return [1.0, float(fwd), float(right), min(dist / 2048.0, 2.0)]

    def _cell(self, wx, wy):
        cx = np.floor(np.asarray(wx) / CELL).astype(np.int64) + G // 2
        cy = np.floor(np.asarray(wy) / CELL).astype(np.int64) + G // 2
        ok = (cx >= 0) & (cx < G) & (cy >= 0) & (cy < G)
        return np.where(ok, cy * G + cx, PAD)

    def step(self, obs: dict, action) -> tuple[dict, bool]:
        """Encode `obs`, which resulted from `action` (ignored on the first call after reset).
        Returns (features, entered_new_cell)."""
        vit = np.asarray(obs["vitals"], np.float32)
        vel = vit[14:16]
        if not self.first:
            a = np.asarray(action, np.float32)
            t = int(a[4]) - int(a[5])
            self.n += 1
            if self.n == 1:  # engine ignores the turn input on the first step after reset
                deg = 0.0
            elif self.n == 2:  # then a short startup ramp (measured for all 81 four-step starts)
                deg = t * (3.52 if self.turns[-1] != 0 else 1.76)
            else:
                deg = _TURN.get((self.turns[-2], self.turns[-1], t), 0.0)
            dh = np.radians(deg)
            self.turns.append(t)
            del self.turns[:-2]
            h_mid = self.h + dh / 2
            f, r = (self.prev_vel + vel) * 2.0  # average velocity x 4 tics
            self.x += f * np.cos(h_mid) + r * np.sin(h_mid)
            self.y += f * np.sin(h_mid) - r * np.cos(h_mid)
            self.h += dh
            self.prev_action = a
        self.first = False
        self.prev_vel = vel.copy()
        ch, sh = np.cos(self.h), np.sin(self.h)

        M = self.map
        M[6] *= 0.9  # enemy memory fades
        M[8] *= RECENT_DECAY
        # rays -> free space + typed hits
        dist = np.asarray(obs["rays"]["distance"], np.float32) * RAY_RANGE
        ht = np.asarray(obs["rays"]["hit_type"], np.int64)
        wa = self.h + _RAY_ANG
        dx, dy = np.cos(wa), np.sin(wa)
        free = _SAMPLES[None, :] < (dist[:, None] - 16)
        fx = self.x + dx[:, None] * _SAMPLES[None, :]
        fy = self.y + dy[:, None] * _SAMPLES[None, :]
        fidx = self._cell(fx[free], fy[free])
        M[0, fidx] = 1
        M[1:3, fidx] = 0
        # locked-door marks are long-horizon landmarks: only erase them from close range, so heading
        # drift on far rays cannot wipe a remembered locked door
        near = free & (_SAMPLES[None, :] < 192)
        nidx = self._cell(fx[near], fy[near])
        M[3, nidx] = 0
        # rays stop at pickups, so open space seen up close holds no key/item any more (picked up)
        M[4:6, nidx] = 0
        # phantom key/item markers: a pickup spotted at 200-500 units and never approached head-on used to
        # stay on the map forever. If no pickup is in view, free space in the front cone up to CLEAR_FAR
        # clears them too (locked doors M[3] keep the close-range-only rule above).
        ent = obs["visible_entities"]
        et = np.asarray(ent["entity_type"], np.int64)
        mask = np.asarray(ent["mask"], bool)
        if not np.isin(et[mask], PICKUP_T).any():
            cone = np.zeros_like(free)
            cone[FRONT_RAYS] = True
            far = free & cone & (_SAMPLES[None, :] < CLEAR_FAR)
            M[4:6, self._cell(fx[far], fy[far])] = 0
        hit = ht > 0
        hidx = self._cell(self.x + dx[hit] * (dist[hit] + 8), self.y + dy[hit] * (dist[hit] + 8))
        chan = np.array([0, 1, 2, 3, 6, 5])[ht[hit]]
        M[chan, hidx] = 1
        # visible keys / enemies from entity list (rotation by heading, no angle inversion)
        feat = np.asarray(ent["features"], np.float32)
        d = feat[:, 0] * RAY_RANGE
        ex = self.x + d * (ch * feat[:, 2] - sh * feat[:, 1])
        ey = self.y + d * (sh * feat[:, 2] + ch * feat[:, 1])
        eidx = self._cell(ex, ey)
        M[4, eidx[mask & (et == KEY_T)]] = 1
        M[6, eidx[mask & np.isin(et, ENEMY_T)]] = 1
        # visits (curiosity source)
        pidx = int(self._cell(self.x, self.y))
        kh = float(vit[8:14].sum())
        if kh > self.keys_held:  # just picked a key up: drop the key mark around us
            cx, cy = pidx % G, pidx // G
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    if 0 <= cx + ox < G and 0 <= cy + oy < G:
                        M[4, (cy + oy) * G + cx + ox] = 0
            # backtrack mode: earlier rooms are no longer "done" (visits reset) so the way back to the
            # remembered locked door is open again; the recent trail M[8] stays, so the agent does not
            # ping-pong straight back into the dead end it just left. Only if a locked door IS remembered:
            # otherwise (e.g. a two-lock chain, door not seen yet) the reset made searched rooms look new
            # and the agent looped in them (key_4_9000: 15-30 distinct cells per 150 steps for 700 steps).
            if M[3, :PAD].max() > 0.5:
                M[7] = 0
            self.since_key = 0
        elif kh > 0:
            self.since_key += 1
        # optional (off by default, DF_DOOR_RESET=1): key found BEFORE its locked door -> when the first locked
        # door is mapped while a key is held, reset visits once so the way back is open (as at key pickup)
        door_now = bool(M[3, :PAD].max() > 0.5)
        if DOOR_RESET and door_now and not self.door_known and kh > 0 and not kh > self.keys_held:
            M[7] = 0
        self.door_known = door_now
        self.keys_held = kh
        new_cell = pidx != PAD and M[7, pidx] == 0
        M[7, pidx] += 1
        M[8, pidx] = 1.0
        M[:, PAD] = 0

        # heading-aligned crop indices
        wx = self.x + _FWD * ch + _RIGHT * sh
        wy = self.y + _FWD * sh - _RIGHT * ch
        crop_idx = self._cell(wx, wy)
        crop = M[:, crop_idx].reshape(N_FIXED, CROP, CROP).copy()
        crop[7] = np.log1p(crop[7]) / 3.0

        # coarse global map (max-pooled, heading-aligned, agent-centred)
        gidx = self._cell(self.x + _GFWD * ch + _GRIGHT * sh, self.y + _GFWD * sh - _GRIGHT * ch)
        gmap = M[:, gidx].reshape(N_FIXED, GMAP * GMAP, GSUB * GSUB).max(-1).reshape(N_FIXED, GMAP, GMAP)
        gmap[7] = np.log1p(gmap[7]) / 3.0

        depth = np.asarray(obs["depth"], np.float32)[30:90].min(0).reshape(40, 4).min(1) / 255.0
        keymem = np.array([float(kh > 0), min(self.since_key / 200.0, 5.0)], np.float32)
        landmarks = np.array([v for c in LANDMARK_CH for v in self._landmark(c, self.h)], np.float32)
        vec = np.concatenate([vit / VITALS_SCALE, self.prev_action, depth, keymem, landmarks]).astype(np.float32)
        rays = np.concatenate([(dist / RAY_RANGE)[None], np.eye(6, dtype=np.float32)[ht].T]).astype(np.float32)
        ents = np.concatenate([np.clip(feat, -10, 10) * np.array([1, 1, 1, 0.1, 0.1], np.float32),
                               mask[:, None].astype(np.float32)], 1)
        out = {"vec": vec, "rays": rays, "ents": ents.astype(np.float32), "etype": et * mask,
               "map": crop.astype(np.float16), "gmap": gmap.astype(np.float16),
               "crop_idx": crop_idx, "pos_idx": np.int64(pidx),
               # action mask (allowed by RULES.md): no weapon switch while the held weapon has ammo
               "wmask": np.float32(0.0 if vit[3] > 0 else 1.0)}
        return out, bool(new_cell)
