"""Video that makes key -> locked door backtracking visible: game screen + top-down map of the level.

The top-down map is drawn from the generator's layout FOR VISUALISATION ONLY (the agent never sees it).
Trail colour: blue while searching for the key, orange once a key is held (backtracking to the gate),
green after the locked door is open. Markers: key (yellow), locked door (red), exit (green).

  python training/record_backtrack.py --ckpt training/ckpt/latest.pt --mask 1 --maps key_1_8001 --out bt.mp4
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(ROOT / "env")]

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

H_SCREEN = 360  # game view 160x120 scaled x3
BAR = 60
PHASE = {"search": ((80, 150, 255), "SEARCHING FOR THE KEY"),
         "back": ((255, 150, 40), "KEY HELD -> BACKTRACKING TO THE LOCKED DOOR"),
         "exit": ((60, 220, 90), "DOOR UNLOCKED -> HEADING FOR THE EXIT")}


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", type=int, default=1)
    ap.add_argument("--fps", type=int, default=18)
    args = ap.parse_args()

    from mapgen import FAMILIES, generate
    from doom_env import make_env
    import agent as agent_mod
    from agent import DoomAgent
    from model import load_compat
    agent_mod.MASK_BACKWARD = bool(args.mask)
    sd = torch.load(args.ckpt, map_location="cpu")
    f_big, f_small = font(20), font(16)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)

    for name in args.maps:
        fam, li, mi = name.rsplit("_", 2)
        grid, things = generate(fam, int(li) / 4, FAMILIES.index(fam) * 1_000_000 + int(li) * 10_000 + int(mi))
        rows, cols = len(grid), max(len(r) for r in grid)
        px = max(4, min(12, H_SCREEN // rows, 420 // cols))
        base = Image.new("RGB", (cols * px, rows * px), (25, 25, 30))
        bd = ImageDraw.Draw(base)
        colours = {".": (70, 70, 80), "D": (150, 110, 60), "E": (60, 220, 90),
                   "R": (230, 50, 50), "B": (230, 50, 50), "Y": (230, 50, 50)}
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if ch in colours:
                    bd.rectangle([c * px, r * px, (c + 1) * px - 1, (r + 1) * px - 1], fill=colours[ch])
        keys = [(y, x) for y, x, t, _ in things if t in (5, 6, 13)]
        for y, x in keys:
            bd.ellipse([x * px + 1, y * px + 1, (x + 1) * px - 2, (y + 1) * px - 2], fill=(255, 230, 40))

        env = make_env(str(ROOT / "training" / "maps_eval" / f"{name}.wad"), "MAP01", episode_ticks=4200)
        ag = DoomAgent(env.observation_space, env.action_space, "cpu")
        load_compat(ag.net, sd["model"])
        ag.reset()
        torch.manual_seed(0)
        obs, _ = env.reset(seed=12345)
        trail, phase, step, t_key, t_door = [], "search", 0, None, None
        while True:
            ps = env.privileged_state()
            if ps["keys"].sum() > 0 and phase == "search":
                phase, t_key = "back", step
            lock, fl, ce = ps["door_lock"], ps["sector_floor"], ps["sector_ceiling"]
            if phase == "back" and np.any((lock > 0) & (ce - fl > 40)):
                phase, t_door = "exit", step
            cx, cy = ps["x"] / 64.0, -ps["y"] / 64.0
            trail.append((cx * px, cy * px, PHASE[phase][0]))

            mp = base.copy()
            md = ImageDraw.Draw(mp)
            for i in range(1, len(trail)):
                md.line([trail[i - 1][:2], trail[i][:2]], fill=trail[i][2], width=max(2, px // 3))
            a = math.radians(ps["angle"])
            hx, hy = cx * px + math.cos(a) * px * 1.5, cy * px - math.sin(a) * px * 1.5
            md.line([(cx * px, cy * px), (hx, hy)], fill=(255, 255, 255), width=2)
            md.ellipse([cx * px - px // 2, cy * px - px // 2, cx * px + px // 2, cy * px + px // 2], fill=(255, 255, 255))

            screen = Image.fromarray(np.repeat(np.repeat(obs["rgb"], 3, 0), 3, 1))
            W = screen.width + 12 + max(mp.width, 360)
            canvas = Image.new("RGB", (W + (W % 2), max(H_SCREEN, mp.height) + BAR + (max(H_SCREEN, mp.height) + BAR) % 2),
                               (12, 12, 16))
            canvas.paste(screen, (0, BAR))
            canvas.paste(mp, (screen.width + 12, BAR))
            d = ImageDraw.Draw(canvas)
            col, label = PHASE[phase]
            d.text((10, 6), f"{name}  |  step {step}  |  keys held {int(ps['keys'].sum())}  |  kills {ps['kills']}",
                   fill=(230, 230, 230), font=f_small)
            d.text((10, 30), label, fill=col, font=f_big)
            writer.append_data(np.asarray(canvas))
            obs, _, term, trunc, info = env.step(ag.act(obs))
            step += 1
            if term or trunc:
                ps = env.privileged_state()
                out = "EXIT REACHED" if info["exit_reached"] else ("DIED" if ps["dead"] else "TIME UP")
                end = canvas.copy()
                ImageDraw.Draw(end).text((10, 30), f"RESULT: {out} in {step} steps (key at {t_key}, gate at {t_door})",
                                         fill=(255, 255, 255), font=f_big)
                for _ in range(args.fps * 3):
                    writer.append_data(np.asarray(end))
                print(f"{name}: {out} in {step} steps | key picked up at {t_key} | locked door opened at {t_door}", flush=True)
                break
        env.close()
    writer.close()


if __name__ == "__main__":
    main()
