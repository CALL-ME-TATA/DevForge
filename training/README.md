# Training: DeadReckon-EgoMap agent

A 1.49M-parameter recurrent agent that builds its **own map of the level from its own motion** and
learns to read, plan over and act on that map, trained with a custom recurrent PPO on 12,500
procedurally generated levels.

## The agent (what runs at evaluation)

Every action is sampled from the trained network. The submitted agent uses only the official
`doom_env` observations (`rays`, `vitals`, `visible_entities`; no privileged state, no map files).

```
observations ──► features.py (deterministic encoding, numpy)
                 ├─ dead reckoning: heading from our own turn actions (in-engine turn table),
                 │  position from integrated ego velocities in `vitals`; frame = spawn pose
                 └─ EgoMap: 128×128 grid of 64-unit cells written from rays + visible entities
                    (free / wall / door / locked / key / item / enemy / visits), read as a
                    heading-aligned 32×32 crop
          ──► model.py (all learned)
                 ├─ circular 1-D conv over the 64 rays
                 ├─ MLP over vitals, depth summary, previous action
                 ├─ self-attention over visible entities (+ null token)
                 ├─ map CNN + learned planner: a K=4 step shared-weight convolution in the style of
                 │  Value Iteration Networks. Its weights are trained end to end; nothing is hand-coded.
                 ├─ learned memory: the GRU writes an 8-dim gated vector into its own map channels at
                 │  the dead-reckoned cell (Neural Map style; gradients flow through the writes)
                 ├─ GRU core (256)
                 └─ heads: 6 factored action heads (move, strafe, turn, attack, use, weapon),
                    value, auxiliary predictions
```

Why it is built this way:

- **No privileged position.** Published map-memory agents (EgoMap, Beeching et al. 2019; Neural Map,
  Parisotto & Salakhutdinov 2017) read the true pose from the game. The evaluation gives no pose,
  so the agent dead-reckons its own and the map is written in that estimated frame.
- **Partial observability.** Multi-room levels need memory of where the agent has been, which doors
  exist and where a key was seen. The map plus GRU carry that across the episode.
- **Tight compute budget.** 1 CPU thread, 40 ms per step, 2 GB VRAM: measured p50 10 ms per step and
  ~128 MB GPU memory (`validate.py`).

## Rules compliance

| Rule (docs/RULES.md) | How this repo complies |
| --- | --- |
| Actions from trained neural parameters | `agent.act` = network logits → `sample` → buttons. Nothing else chooses actions. |
| No A*, Dijkstra, BFS over a map | None in `submission/`. The "planner" is a trained convolution (VIN-style), not a search. |
| No aimbots | No angle arithmetic toward entities; aiming is learned. |
| Not off-the-shelf SB3 | PPO, losses, recurrence, curriculum and model are written from scratch in this folder. |
| Training may use anything | Training rewards use the engine's privileged state and the generator's layouts (below). This is training-time scaffolding only; none of it reaches the agent. |

## Training pipeline (`train.py`)

- **Recurrent PPO**: 40 parallel ViZDoom games × 256 steps per iteration, truncated BPTT over 32-step
  chunks carrying the GRU state and learned map, 2 epochs, minibatch 64 sequences, clip 0.2,
  GAE (γ = 0.995, λ = 0.95), Adam lr 3e-4, gradient clip 0.5. The rollout policy step runs as a CUDA graph.
- **Loss** = PPO policy loss + value loss − entropy bonus (0.01) + 0.1 × auxiliary loss.
  Auxiliary targets: change in health, kills and keys 8 and 32 steps ahead (DFP-style,
  Dosovitskiy & Koltun 2017) and "have I visited this cell before" (loop-closure-style, Mirowski et al. 2017).
- **Levels**: `mapgen.py` generates room lattices as UDMF WADs in 5 families mirroring the tiers
  (`tiny`, `arena` combat, `multi` rooms + doors, `key` key → locked door → exit, `gauntlet` all of it)
  × 5 difficulty levels, 500 layouts each (12,500 maps; `extend_maps.py`), plus Freedoom MAP01–24.
- **Curriculum**: per-worker success rate (EMA) per (family, level); levels unlock as earlier ones are
  solved, and sampling favours levels near 50% success (PLR-style, Jiang et al. 2021) with a minimum
  weight for unlocked navigation families so unsolved ones keep being practised. Curriculum state is
  saved in checkpoints so resumes continue where they stopped.

### Reward

| Term | Value | Purpose |
| --- | --- | --- |
| Exit reached | +10, + up to 2 for health left, + up to 2 for time left | main objective |
| Key picked up | +3 | progression |
| Door opened (locked door) | +1 (+2 more) | progression |
| Kill | +1 | combat |
| Item picked up | +0.2 | survival |
| New 64-unit cell visited | +0.02, plus +0.02 on the agent's own dead-reckoned map | exploration |
| Damage taken | −0.03 per HP | survival |
| Death | −5 | survival |
| Every step | −0.001 | urgency |
| Re-pressing USE within 8 steps | −0.02 | stop bouncing doors |
| **Distance shaping** (generated maps only) | +0.1 per cell closer to the current goal, −0.1 per cell farther | long-horizon navigation |
| Switching away from a loaded weapon | −0.05 (−0.01 per step holding fists while ammo exists) | edge case: fist fighting |
| Firing with no enemy in view | −0.005 per shot | edge case: wasted ammo |
| Fresh USE press next to a door (generated maps) | +0.1 | edge case: doors never opened |

**Distance shaping** is potential-based reward shaping (Ng, Harada & Russell 1999), which leaves the
optimal policy unchanged. Each training map's layout is rebuilt from its generator seed. A BFS over that
grid gives the corridor distance to the current goal: the exit if it is reachable with the keys held,
otherwise the nearest key still needed (locked doors are passable only with their key). The reward is
Φ(s′) − Φ(s) with Φ = −0.1 × distance, so pacing back and forth earns nothing. **This BFS exists only in
the training reward;** the agent never sees distances and never runs a search.

### Edge cases found on unseen maps (`evaluate.py` diagnostics, `edge_cases.py`)

At 9.0M steps, failures on 90 unseen maps broke down as: lost / barely closer to the goal 24,
died early 23, stalled at a door 14, reached the goal cell without finishing 5. Probing behaviour:

- **Fist fighting.** The agent pressed SELECT_NEXT_WEAPON on 43% of steps and held fists for 44% of
  every arena episode while it still had bullets (switching within the first 1–5 steps). This is
  the main cause of early combat deaths. Fix: weapon-switch penalty (table above).
- **Aiming.** Attack rate was the same with or without an enemy in view (37% vs 38%), and it turned
  away from the nearest enemy as often as toward it. Fix: small penalty for firing at nothing.
- **Doors.** It spent 0.5% of steps next to a door and pressed USE there 19% of the time, about its
  14% base rate, so the +1 door-opening reward almost never fired. Fix: distance shaping now leads it
  to doors, plus a small reward for pressing USE next to one.
- **Greedy vs sampled actions.** Argmax actions cleared 6/50 maps vs 13/50 sampled (stuck against
  walls), so the agent keeps sampling.

### Automatic edge-case mining during training (`--edge-p`)

The edge cases above were found by hand between runs. With `--edge-p > 0` the engine now finds them
inside every iteration and feeds them straight back into PPO (prioritized replay of hard *states*,
PLR / Go-Explore style):

1. **Snapshot.** Every `--edge-every` steps (default 32) each worker saves the full ViZDoom game state
   (`DoomGame.save`: monsters, health, ammo, doors, clock). All workers save on the same lockstep step, so
   the ~30 ms save stalls the batch once, not 40 times.
2. **Mine.** When an episode fails, `pick_case` classifies it, using the same taxonomy as `edge_cases.py`:
   `death`, `stuck` (≥120 steps inside one cell), `door` / `locked` (closest approach to the goal next to a
   door), `goal_cell` (reached the exit cell, never finished), or `plateau` (got closer, then ≥150 steps
   with no progress). The snapshot from just before the failure is kept: ≥40 steps before a death, at the
   start of a stall, or at the closest approach.
3. **Bank.** The learner's `EdgeBank` (cap `--edge-cap`, saved in checkpoints) samples cases by priority:
   unseen cases first, then the ones that still fail. A case is **retired as learned** after 3 solves in
   a row and **dropped as hopeless** after 8 straight failures.
4. **Replay.** A fraction `--edge-p` of new episodes load a case's exact state and play on from there (late
   failures get at least half an episode of clock). A death case counts as solved if the agent gets out
   alive, every other kind only by reaching the exit. Replays move neither curriculum and are left out of
   the per-level exit rates in the log.

The log line gets `| edge bank N +mined replays R solved S retired L/H`, and
`training/ckpt/edge_report.json` breaks down mined / open cases by kind and level. Engine snapshots and
reward bookkeeping are training-time scaffolding only; the agent observes nothing new.

### Training timeline (one RTX 4050 laptop, Docker/Linux, ~1,000 env steps/s)

| Steps | Change |
| --- | --- |
| 0 – 0.6M | Windows, 16 envs, 40 maps per bin |
| 0.6M – 5.0M | Docker, 40 envs |
| 5.0M – 8.1M | 200 maps per bin; damage −0.01 → −0.03, death −3 → −5, cell +0.01 → +0.02 |
| 8.1M – 9.5M | 500 maps per bin; distance shaping; curriculum floor for multi/key/gauntlet |
| 9.5M – 16.6M | edge-case fixes (weapon/shot/door), reverse curriculum, asymmetric critic, auxiliary goal heads |
| 16.6M – 17.7M | backward mask (`--mask-back 1`, same as the agent) |
| 17.7M – end | **DAgger from a privileged teacher, PPO weight 0** (below) |

### Final stage: imitation from a privileged teacher (DAgger / learning by cheating)

`teacher.py` is a scripted expert that exists **only during training**. It reads the generator's layout
of the current map and the engine's true pose, follows the shortest route (needed key, then locked door,
then exit), presses USE once per door (a second press reverses a moving door), aims and fires at
visible enemies, and never reverses or spins. Alone it clears 81/90 held-out generated maps
(`test_teacher.py`). The student (this network) drives itself; at every step it is trained with
cross-entropy on all 6 action heads toward the teacher's action for the student's own situation
(DAgger, Ross et al. 2011; Learning by Cheating, Chen et al. 2019; kickstarting, Schmitt et al. 2018).
For the first half of the run the teacher's action is executed with a fading probability (≤ 0.3).
Nothing from the teacher reaches `submission/`: the agent sees only the official observations.

## Evaluation on unseen maps (`evaluate.py`, `tier_report.py`)

Held-out levels come from the same generator with seeds never used in training (index ≥ 5000; training
uses 0–499). Every checkpoint plays the same maps with the same seeds. The families map onto the
evaluation tiers: T1 `arena`, T2 `multi`, T3 `key`, T4 `gauntlet` + Freedoom, plus `tiny` as a basic check.

Levels cleared on 92 unseen maps (levels 0/2/4 × 6 maps per family + 2 Freedoom):

| Tier | Repo's original weights | 4.5M (PPO) | 14.4M (PPO) | 16.1M (PPO) | **21.4M (DAgger, shipped)** |
| --- | --- | --- | --- | --- | --- |
| T0 Basics (18) | 6 | 8 | 17 | 17 | 16 |
| T1 Combat (18) | 1 | 3 | 12 | **16** | 13 |
| T2 Multi-room (18) | 1 | 1 | 6 | 6 | **11** |
| T3 Keys (18) | 0 | 0 | 4 | 4 | **13** |
| T4 Gauntlet (20) | 0 | 0 | 1 | 2 | **8** |
| **Total (92)** | **8** | **12** | **40** | **45** | **61 (66%)** |
| Died | 37% | 33% | 45% | 30% | **12%** |

The final checkpoint is chosen by this evaluation, not by recency. Sampled actions beat greedy
(argmax) actions clearly (13 vs 6 of 50 maps; greedy gets stuck against walls), so the agent samples.

## Reproduce

```bash
# Linux container (ViZDoom IPC is ~3.5x faster than on Windows)
docker build -t doom-train -f training/Dockerfile training
python training/extend_maps.py --per-bin 500           # grow the map pool (seconds)
python training/train.py --envs 40 --hours 8 --resume \
  --maps-per-bin 500 --dmg-pen 0.03 --death-pen 5 --cell-rew 0.02 \
  --shape-k 0.1 --hard-floor 0.2 --key-gate 0.3 --weapon-pen 0.05 --shot-pen 0.005 --door-use 0.1 \
  --edge-p 0.2   # mine failure snapshots from the engine and replay them
# Windows host: powershell -File training\start_training.ps1 -Hours 8 -Extra "<flags above>"

python training/evaluate.py --ckpts training/ckpt/latest.pt --per-bin 6 --levels 0 2 4 --freedoom 2
python training/tier_report.py training/eval_results.json
```

## References

- Ng, Harada & Russell (1999). Policy invariance under reward transformations: theory and application to reward shaping. ICML.
- Tamar et al. (2016). Value Iteration Networks. NeurIPS.
- Parisotto & Salakhutdinov (2017). Neural Map: structured memory for deep reinforcement learning.
- Dosovitskiy & Koltun (2017). Learning to act by predicting the future. ICLR.
- Mirowski et al. (2017). Learning to navigate in complex environments. ICLR.
- Lample & Chaplot (2017). Playing FPS games with deep reinforcement learning. AAAI.
- Wu & Tian (2017). Training agent for first-person shooter game with actor-critic curriculum learning. ICLR.
- Beeching et al. (2019). EgoMap: projective mapping and structured egocentric memory for deep RL.
- Cobbe et al. (2020). Leveraging procedural generation to benchmark reinforcement learning. ICML.
- Jiang, Grefenstette & Rocktäschel (2021). Prioritized level replay. ICML.
