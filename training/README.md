# Training: DeadReckon-EgoMap agent

A ~1.5M-parameter recurrent agent that builds its **own map of the level from its own motion** and
learns to read, plan over and act on that map. Trained on 12,500 procedurally generated levels with a
custom recurrent PPO, then with imitation of a training-only privileged teacher (DAgger), then with PPO
and a fading imitation term together (DAPG-style).

## The agent (what runs at evaluation)

Every action is sampled from the trained network. The submitted agent uses only the official
`doom_env` observations (`rays`, `vitals`, `visible_entities`, `depth`; never `rgb`, no privileged
state, no map files). `agent.py` is a thin `DoomAgent` wrapper around `decision.DecisionEngine`
(features, network, recurrent state, action masks); the network lives in `policy.py`
(`model.py` re-exports it).

```
observations ──► features.py (deterministic encoding, numpy)
                 ├─ dead reckoning: heading from our own turn actions (in-engine turn table),
                 │  position from integrated ego velocities in `vitals`; frame = spawn pose
                 ├─ EgoMap: 128×128 grid of 64-unit cells written from rays + visible entities
                 │  (free / wall / door / locked / key / item / enemy / visits / recent trail), read as a
                 │  heading-aligned 32×32 crop
                 ├─ coarse global map: 16×16 heading-aligned max-pool of the whole EgoMap (512-unit cells),
                 │  so keys / locked doors beyond the crop stay visible
                 ├─ key memory: [key held, steps since pickup]; on pickup the visit counts reset
                 │  (backtrack mode) while the recent trail is kept
                 └─ phantom pickups cleared: free space seen < 384 u in the front cone erases stale
                    key/item marks when no pickup is in view
          ──► policy.py (all learned)
                 ├─ circular 1-D conv over the 64 rays
                 ├─ MLP over vitals, depth summary, previous action
                 ├─ self-attention over visible entities (+ null token)
                 ├─ small CNN over the coarse global map (Neural Map-style global read)
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
| Action masking from observation fields (allowed) | Two masks can only *release* a button, never press one: no weapon switch while the held weapon has ammo (`wmask`), and no MOVE_BACKWARD unless an enemy is in view (`action_mask.py`; the same mask is applied during training). |
| Training may use anything | Training rewards use the engine's privileged state and the generator's layouts (below). This is training-time scaffolding only; none of it reaches the agent. |

**About heuristics.** The training-only teacher (`teacher.py`) *is* a hand-written expert: BFS routes
over the generator's layout, rule-based steering, USE timing and aiming. It runs only inside training to
produce imitation targets, is never imported by `submission/`, and the rules allow any training-time
scaffolding. At evaluation the agent is the network alone; `features.py` is observation encoding (dead
reckoning and map writing), not decision-making.

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
| 17.7M – 21.4M | **DAgger from a privileged teacher, PPO weight 0** (below) |
| 21.4M – 23.4M | mentor review: global map, key memory, visits reset, phantom-pickup fix; PPO weight 0.3 + imitation 1.0 → 0.5 (DAPG-style) |
| 23.4M – end | **information-matched frontier teacher** (below) |

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

### Information-matched frontier teacher (`--teacher-explore 1`)

Diagnostics showed most keycard failures were *never finding the key*. The clairvoyant teacher walks
straight to a key the student cannot see, so the student cannot learn that decision (the "imitation gap":
Weihs et al. 2021; Walsman et al. 2023; Choudhury et al. 2018). In explore mode the teacher tracks which
cells the student could have seen (32 sight lines, 1024 u, walls and doors block) and, until the needed
key or the exit has been seen, goes to the **nearest frontier through seen space** (Yamauchi 1997);
once the goal is seen it takes the shortest route. Its decisions now depend only on information the
student also has in its own map. Alone it clears 81/90 held-out maps, the same as the clairvoyant teacher.
`record_backtrack.py` renders the search / key-held / gate-open phases on a top-down view.

## Final submission: it2734 (27.1M steps)

Trained to the end with the information-matched frontier teacher + PPO on fresh, structurally randomised
maps (high-DPS monsters, exit switches), landmark compass, failed-level replay. Same maps and seeds for both:

| Set | it2410 | **it2734 (shipped)** |
| --- | --- | --- |
| Test: 100 never-seen generated maps, all 5 difficulties | 70/100 | **72/100** (T0 19, T1 15, T2 15, T3 12, T4 11 of 20) |
| Test: 60 hard never-seen maps (tough monsters, switches, varied layouts) | 23/60 | **25/60** |
| Test: hand-made Freedoom MAP25–32 (never seen) | 0/8 exits | 0/8 exits |
| Validation: 90 generated maps (levels 0/2/4) | 62/90 | 59/90 |

Differences between the two are within the measured seed noise (±2 maps). Official-image validate: PASSED
(1 CPU, p50 6.7 ms/step, 1.5 GB RAM). Stress tests: 0 crashes / 0 invalid actions on 3,000 fuzzed observations;
largest hand-made maps within budget.

## Known limitations

- **Hand-made levels.** 0/8 exits on Freedoom MAP25–32 (never seen) and 0/6 on MAP01–06, while earning
  partial credit (4–13 kills per level on MAP01–06). These are full campaign levels; the 4,200-tic cap is
  about two minutes, too short to finish most of them. Per-death diagnosis: (1) losing fights against
  groups (up to 6 enemies in view), (2) holding an empty weapon for up to 75% of steps and never switching
  (a habit from generated maps where ammo rarely runs out; now rewarded with `--empty-switch-rew`),
  (3) damage with no enemy in view (hazard floors / off-screen attackers, absent from the generator).
  Transfer to human-designed levels "highly depends on the design of the level generators"
  (Justesen et al. 2018); the generator now adds high-DPS monsters, wall switches and varied structure.
  Next steps: the same privileged teacher on rasterised real maps (learning by cheating on the target
  domain), hazard floors and ammo scarcity in the generator, regret-based level design (PAIRED, ACCEL).
- **Key and lock colours are not observable.** Rays report "locked door" and entities report "key" without
  a colour (only held keys are known from `vitals`), so two-lock levels are solved by trying.
- **Search is the weakest skill.** On failed runs the agent covers only ~25–35% of the map; the frontier
  teacher targets exactly this.
- **Imitation of a heuristic teacher.** Allowed as training scaffolding (see Rules compliance), but the
  final policy is not pure end-to-end RL; the pure-PPO checkpoint (16.1M steps) scored 45/92 vs 61/92.

### Post-submission fixes (from a frame-by-frame review of it2734 videos)

| Failure seen | Cause found | Fix |
|---|---|---|
| T3: loops in already-searched rooms after picking a key | EgoMap visit channel was wiped on every key pickup, so searched rooms looked new | keep visits unless a locked door is already mapped (`features.py`) |
| Freedoom MAP25: 485 shots, ~5 hits, then empty gun and death | no cost for missing; ammo never scarce in the generator | miss / miss-streak penalty, ammo pickup reward scaled by need (`--miss-pen`, `--miss-streak-pen`, `--ammo-need-k`); 30% of extended maps nearly ammo-free |
| Freedoom MAP01: stuck 115 steps, 63/75 damage with no enemy in view | no obstacles or damaging floors in the generator | barrels / columns / lamps and NUKAGE hazard patches (5 HP / 32 tics), placed off the key-exit route on their own RNG stream (older maps unchanged) |

Also: the shaper / teacher grid now marks solid decorations as walls (the teacher was steering into
barrels), and only half of the extended maps get the new realism so the rest rehearse the old distribution
(forgetting during fine-tuning: Rolnick et al. 2019; Wołczyk et al. 2024). `training/wise_ft.py` averages
old and fine-tuned weights (WiSE-FT, Wortsman et al. 2022).

**Result: the fine-tunes did not beat it2734, so it2734 stays the submission.** Exits on the same 54 fixed
never-seen maps (seeds 9000–9001, all tiers and levels, Freedoom MAP25–28):

| Model | Total | T1 | T2 | T3 | T4 | Freedoom |
|---|---|---|---|---|---|---|
| it2734 (two runs: original / with the `features.py` fix) | 37 / 41 | 7 / 9 | 8 / 8 | 6 / 7 | 6 / 7 | 0 / 0 |
| run 1 (all fixes, 146 iters) | 33 | 8 | 5 | 6 | 4 | 0 |
| run 1 WiSE-FT 0.5 with it2734 | 34 | 8 | 6 | 6 | 4 | 0 |
| run 3 snapshot (obstacle-aware teacher, rehearsal, smaller rewards) | 35 | 8 | 6 | 6 | 5 | 0 |
| run 3 final (it ~2810) | 36 | 7 | 8 | 7 | 4 | 0 |
| it2734, third run (other episode seed) | 39 | 9 | 9 | 7 | 5 | 0 |
| it2734 + sampling temperature 0.85 | 34 | 5 | 8 | 6 | 5 | 0 |
| it2734 + visit reset when the locked door is first seen with a key held | 39 | 9 | 8 | 7 | 5 | 0 |

it2734 with the shipped code averages 40/54 over its three current runs. A sharper sampling temperature
hurts combat (5/10, as greedy did earlier), and the door-time visit reset ties the baseline, so neither ships
(both stay as off-by-default switches: `--temp`, `DF_DOOR_RESET=1`).

The same 54 maps vary by about ±3 exits between runs of one model (sampled actions), but T2 dropped in every
fine-tune (5–6 vs 8): obstacles and hazard floors change room navigation, and about 100 iterations is not
enough to adapt. The `features.py` memory fix ships (no regression: 41/54).

**Next steps for differently-built maps** (the real weakness: on another team member's own map set, exits
fell to 60 / 20 / 0 / 0 % for T1–T4):
1. Teacher on rasterised real WAD geometry (Freedoom MAP01–24, community maps), learning by cheating on
   the target domain (Chen et al. 2019); hold out MAP25–32 and unseen community maps for testing.
2. A far more varied generator: non-rectangular rooms, height changes, lifts, remote switches, keys off the
   main path (Tobin 2017; Cobbe 2020; Justesen 2018).
3. Regret-driven map editing toward the agent's failures (PAIRED, Dennis 2020; ACCEL, Parker-Holder 2022).
4. Memory for big maps: the 128 × 128 EgoMap covers ±4,096 units from spawn, so larger maps fall off it and
   the agent re-walks explored areas; plus an episodic novelty reward (Never Give Up, Badia et al. 2020).

## Evaluation on unseen maps (`evaluate.py`, `tier_report.py`)

Held-out levels come from the same generator with seeds never used in training (index ≥ 5000; training
uses 0–499). Every checkpoint plays the same maps with the same seeds. The families map onto the
evaluation tiers: T1 `arena`, T2 `multi`, T3 `key`, T4 `gauntlet` + Freedoom, plus `tiny` as a basic check.

**Validation set (92 maps):** 90 unseen generated maps (levels 0/2/4 × 6 per family; seeds checked:
0 overlap with the 12,500 training seeds, 0 identical WAD files) + Freedoom MAP01–02. *Correction:*
those two Freedoom levels are **not** unseen (training samples MAP01–24). The same set was used all night
to compare checkpoints and to choose the shipped one, so it is a validation set and its score is
optimistic. The unbiased number comes from a separate **test set** used once at the end: 100 never-used
generated maps (seeds 9000–9003, all 5 levels) + Freedoom MAP25–32 (never trained on).

Levels cleared on the validation set:

| Tier | Repo's original weights | 4.5M (PPO) | 14.4M (PPO) | 16.1M (PPO) | **21.4M (DAgger)** |
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
  --shape-k 0.15 --hard-floor 0.2 --key-gate 0.3 --weapon-pen 0.05 --shot-pen 0.02 --hit-rew 0.1 \
  --door-use 0.5 --use-bounce 0.12 --turn-pen 0.002 --mask-back 1 --key-rew 3.5 --stall-pen 0.01 \
  --rev-p 0.3 --chunk 64 --mb 32 --lr 1e-4 \
  --bc-w 1.0 --bc-end 0.5 --ppo-w 0.3 --dagger-beta 0.1 --dagger-fade 0 --teacher-explore 1
python training/test_teacher.py --explore 1   # the teacher alone on held-out maps
# Windows host: powershell -File training\start_training.ps1 -Hours 8 -Extra "<flags above>"

python training/evaluate.py --ckpts training/ckpt/latest.pt --per-bin 6 --levels 0 2 4 --freedoom 2
python training/tier_report.py training/eval_results.json
```

## References

- Yamauchi (1997). A frontier-based approach for autonomous exploration. CIRA.
- Ross, Gordon & Bagnell (2011). A reduction of imitation learning to no-regret online learning (DAgger). AISTATS.
- Rajeswaran et al. (2018). Learning complex dexterous manipulation with deep RL and demonstrations (DAPG). RSS.
- Schmitt et al. (2018). Kickstarting deep reinforcement learning.
- Choudhury et al. (2018). Data-driven planning via imitation learning. IJRR.
- Chen et al. (2019). Learning by cheating. CoRL.
- Chaplot et al. (2020). Learning to explore using Active Neural SLAM. ICLR; Object goal navigation using goal-oriented semantic exploration. NeurIPS.
- Weihs et al. (2021). Bridging the imitation gap by adaptive insubordination. NeurIPS.
- Justesen et al. (2018). Illuminating generalization in deep RL through procedural level generation. NeurIPS DRL workshop.
- Cobbe et al. (2019). Quantifying generalization in reinforcement learning. ICML.
- Kirk et al. (2023). A survey of zero-shot generalisation in deep reinforcement learning. JAIR.
- Dennis et al. (2020). Emergent complexity and zero-shot transfer via unsupervised environment design (PAIRED). NeurIPS.
- Parker-Holder et al. (2022). Evolving curricula with regret-based environment design (ACCEL). ICML.
- Wijmans et al. (2020). DD-PPO: learning near-perfect PointGoal navigators. ICLR.
- Wayne et al. (2018). Unsupervised predictive memory in a goal-directed agent (MERLIN).
- Tobin et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. IROS.
- Walsman et al. (2023). Impossibly good experts and how to follow them. ICLR.

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
