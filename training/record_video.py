"""Record a checkpoint playing held-out maps (one per tier) to an MP4 with a caption bar.

  python training/record_video.py --ckpt training/ckpt/latest.pt --out agent.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(ROOT / "env")]

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

TICKS = {"tiny": 1050, "arena": 2100, "multi": 3150, "key": 4200, "gauntlet": 4200}
TIER = {"tiny": "T0 Basics", "arena": "T1 Combat", "multi": "T2 Multi-room", "key": "T3 Keycards",
        "gauntlet": "T4 Gauntlet"}
SCALE, BAR = 4, 56


def caption(frame, lines):
    img = Image.fromarray(np.repeat(np.repeat(frame, SCALE, 0), SCALE, 1))
    canvas = Image.new("RGB", (img.width, img.height + BAR), (15, 15, 20))
    canvas.paste(img, (0, BAR))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for i, t in enumerate(lines):
        d.text((10, 6 + 24 * i), t, fill=(235, 235, 235), font=font)
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maps", nargs="+", default=["tiny_4_5000", "arena_2_5000", "multi_0_5000",
                                                    "key_0_5000", "gauntlet_0_5000"])
    ap.add_argument("--fps", type=int, default=18, help="game runs at 8.75 decisions/s; 18 fps = ~2x speed")
    ap.add_argument("--mask", type=int, default=0, help="1 = backward mask on (checkpoints trained with --mask-back 1)")
    args = ap.parse_args()

    from doom_env import make_env
    import agent as agent_mod
    from agent import DoomAgent
    agent_mod.MASK_BACKWARD = bool(args.mask)
    from model import load_compat

    sd = torch.load(args.ckpt, map_location="cpu")
    it = sd.get("iter", "?")
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    for name in args.maps:
        fam = name.split("_")[0]
        env = make_env(str(ROOT / "training" / "maps_eval" / f"{name}.wad"), "MAP01", episode_ticks=TICKS[fam])
        agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        load_compat(agent.net, sd["model"])
        agent.reset()
        torch.manual_seed(0)
        obs, _ = env.reset(seed=12345)
        step, kills = 0, 0
        while True:
            ps = env.privileged_state()
            v = obs["vitals"]
            lines = [f"{TIER[fam]}  |  unseen map {name}  |  iteration {it}",
                     f"step {step}   health {int(v[0])}   ammo {int(v[3])}   kills {ps['kills']}   "
                     f"keys {int(ps['keys'].sum())}"]
            writer.append_data(caption(obs["rgb"], lines))
            obs, _, term, trunc, info = env.step(agent.act(obs))
            step += 1
            if term or trunc:
                ps = env.privileged_state()
                outcome = "EXIT REACHED" if info["exit_reached"] else ("DIED" if ps["dead"] else "TIME UP")
                end = caption(obs["rgb"], [f"{TIER[fam]}  |  {name}", f"RESULT: {outcome} after {step} steps, "
                                                                       f"{ps['kills']} kills"])
                for _ in range(args.fps * 2):  # hold the result card for 2 s
                    writer.append_data(end)
                print(f"{name}: {outcome} in {step} steps, kills {ps['kills']}", flush=True)
                break
        env.close()
    writer.close()


if __name__ == "__main__":
    main()
