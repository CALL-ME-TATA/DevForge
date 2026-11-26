# Artifacts

Everything produced while training and evaluating the agent: gameplay videos, evaluation results,
logs, maps and every checkpoint. Nothing here is needed to run the submission (`submission/`).

Large binaries are stored as plain git files (not Git LFS), so they don't use LFS bandwidth that the
scorer needs for `submission/weights/model.pt`. Total ≈ 800 MB, under the 1 GB repository limit.

## Videos (`videos/`, 14 files)

Gameplay of the agent on **held-out maps it never trained on** (generator seeds ≥ 5000; training uses
0–499). The bar at the top shows the tier, map, iteration, health, ammo, kills and keys; each episode
ends on a result card. `training/record_video.py` makes them.

| File | Checkpoint | Maps |
| --- | --- | --- |
| `doom_agent_latest.mp4` | it1540 (PPO) | one per tier, seeds 5000 |
| `doom_agent_it1660.mp4` | it1660 (PPO) | same 5 maps |
| `doom_agent_it1660_fresh_maps.mp4` | it1660 (PPO) | 5 brand-new maps (seeds 7000) |
| `doom_agent_latest_masked.mp4` | it1720 (PPO + backward mask) | same 5 maps |
| `doom_agent_dagger.mp4` | it1860 (DAgger) | same 5 maps |
| `doom_agent_dagger_latest.mp4` | it2150 (DAgger) | same 5 maps |
| `doom_agent_dagger_fresh_maps.mp4` | it2180 (DAgger, shipped) | the seed-7000 maps |
| `doom_agent_it2180_4_new_maps.mp4` | it2180 (shipped) | 4 harder new maps (seeds 8000) |
| `tier3_key_level0..4.mp4` | it2220 (DAgger) | keycard maps, levels 0–4 (seeds 8001) |
| `agent_changes_explainer.mp4` | — | explainer video |

## Evaluations (`evals/`)

Per-episode JSON from `training/evaluate.py` (exit, death, kills, keys, cells explored, steps, and
failure diagnostics). Summaries: `python training/tier_report.py <file>` and `training/edge_cases.py`.

| File | What |
| --- | --- |
| `eval_results.json` | it150, it450 and the repo's original weights on the 92-map set |
| `eval_it700.json`, `eval_it1500.json`, `eval_it1660.json` | PPO checkpoints, same 92 maps |
| `eval_dagger.json` … `eval_dagger4.json` | DAgger checkpoints it1860, it2110, it2180, it2360 |
| `eval_push.json` | it2250 (57/92; not shipped because it2180 scored 61/92) |
| `eval_all*.json`, `eval_cmp*.json` | merged files used for side-by-side tables |
| `eval50.json`, `eval50_greedy.json` | 50 fresh maps: sampled vs greedy actions (13 vs 6 cleared) |
| `diag.json` | it900 with failure diagnostics (the edge-case analysis) |
| `loss_per_iteration.csv` | policy/value/aux loss, entropy and KL for iterations 1–448 |

## Logs (`logs/`)

`train.log` (every training iteration, all runs), `training_health.log` (15-minute summaries and
alerts), Docker build logs for the training image and the official scorer image, a smoke-test log
and the fine-tune scheduler log.

## Maps (`maps/`)

- `maps_eval/`: the 154 held-out evaluation maps (seeds 5000+, 6000+, 7000, 8000, 8001).
- `maps_gen.tar.gz`: all 12,500 training maps (5 families × 5 levels × 500). They can also be rebuilt
  exactly with `python training/extend_maps.py --per-bin 500` (fixed seeds).

## Checkpoints (`checkpoints/`)

`checkpoints.tar.part00`–`part06` (90 MiB parts) hold every checkpoint: `training_ckpt/` (all files from
`training/ckpt/`, including full resume points with optimizer state) and `extra_snapshots/` (unique
weights saved during evals and videos; the shipped it2180 resume point is kept with its optimizer).
`CONTENTS.txt` lists all 70 files.

```bash
cat checkpoints.tar.part* > checkpoints.tar && sha256sum -c SHA256SUMS && tar -xf checkpoints.tar
```

## Scripts (`scripts/`)

`watch_training.sh` (health watcher) and `extend_training.sh` (guarded scheduler for the fine-tune).
