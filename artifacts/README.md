# Artifacts

Everything produced while training and evaluating the agent: evaluation results, logs, maps and every
checkpoint. Nothing here is needed to run the submission (`submission/`).

Only small files are kept here (evaluation results, logs, maps). Videos and checkpoints stay out of
the repository.

## Videos

Videos are not stored in the repository. Record them locally with `training/record_video.py`
(game view per map) or `training/record_backtrack.py` (game view + top-down path by phase).

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

## Checkpoints

Training checkpoints are not stored in the repository: the official validator flags large
non-LFS files, and putting them in LFS would compete with the scorer's download of
`submission/weights/model.pt`. The shipped weights are `submission/weights/model.pt` (Git LFS).

## Scripts (`scripts/`)

`watch_training.sh` (health watcher) and `extend_training.sh` (guarded scheduler for the fine-tune).
