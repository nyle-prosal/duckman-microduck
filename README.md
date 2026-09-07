# Duck-Man — maze tag for Microduck

Entry for the HIM Arena **Microduck · Best Sports Sim** challenge (`microduck-sports-sim-2026`).

Five real Microducks play maze tag in MuJoCo. One Duck-Man collects coins by physically knocking
them over, four ghosts chase it, and a power pellet reverses the chase for twelve seconds. Every duck
walks with Pollen Robotics' published learned gait through the BAM actuator model. Two policies were
trained for this entry: a **stand-up policy** (reinforcement learning, Pollen's training stack, one
GPU) that lets a tagged or fallen duck get back up, and the **Duck-Man strategy network**
(behaviour cloning of our scripted planner, then evolution strategies inside this simulation on a
laptop CPU). Simulation only; no hardware claims.

## What is learned and what is scripted (please read)

| Layer | Type | Who made it | Evidence |
|---|---|---|---|
| **Stand-up policy** (14 joint targets at 50 Hz, recovers from sitting and face-down) | **trained here** — PPO on `Mjlab-StandUp-Flat-MicroDuck` from Pollen's `microduck_rl`, 7,000 iterations × 4,096 envs on one 24 GB GPU (HIM Arena machine) | this entry | `assets/policies/standup.onnx` (exported with Pollen's exporter, normalizer baked in); training logs and checkpoints in the repository's `training/standup/` notes |
| **Duck-Man strategy** (which cell to go to next) | **trained here** — initialised by behaviour cloning of our scripted planner (8,928 decisions), then evolution strategies with full MuJoCo rollouts, elitism on a 6-layout pool | this entry | `checkpoints/strategy_final.npz` (sha256 in `results/learned_seed0.json`), `checkpoints/curve.csv`, learning-curve frame in the video |
| Cell navigation (turn, kick-start, walk to a cell centre) | scripted controller | this entry | `duckman/navigator.py` |
| Ghost behaviour (chase / ambush / mirror / shy, scatter waves, flee, go home) | scripted controller | this entry | `duckman/ghosts.py` |
| Scripted planner baseline (BFS with ghost-danger cost) | scripted controller | this entry | `duckman/planner.py` |
| Walking, standing, sit-and-stand gaits | learned (PPO), **unmodified** | Pollen Robotics, `pollen-robotics/microduck-policies` | `assets/policies/alpha_*.onnx` |
| Actuator physics | BAM XL330 M6 voltage model, **one controller per duck** | Rhoban / Pollen | `duckman/bam_loader.py` |

**Balance assistance: none.** No fixed base, no external stabilisation, no joint animation. All 70
servos of all five ducks are driven only by the ONNX gait networks, every control step, and each
policy object's `act()` is the only writer of joint targets (`duckman/policies.py`).
**Observation:** the strategy layer sees perfect game state (positions, coin states, ghost modes);
every gait network sees proprioception only, exactly as on the robot.
**Render-only cues:** collected tokens become invisible and inert (they still rest on the floor);
frightened ghosts turn blue; the powered Duck-Man flashes. No physics is changed by these.

## Rules

3 lives, 240 s clock. Coin +10 (7 cm token; counts when the Duck-Man has touched it and it is
toppled or knocked ≥ 6 cm from its spawn), power pellet +50 (four, one per corner; 12 s of power),
ghost caught in power mode +200 (it walks home), maze cleared +500. A **tag** is body contact between
a ghost and the unpowered Duck-Man: it loses a life, holds still while the ghosts turn away, sits,
and stands back up (sit-and-stand policy when seated, stand-up policy when it was knocked over)
once every ghost has backed off three cells. Ghosts alternate arcade scatter and chase waves.
Nothing is ever teleported. Ghosts pass through each other and through tokens (collision groups); all
other contact is ordinary physics.

## Results

Held-out seeds 0–19 (training used layouts 1000–1005 only). Same seed = same maze jitter, spawn
noise and ghost RNG for every policy.

<!-- bench:start -->
| Policy | Seeds | Mean score | Std | Mean coins | Ghosts caught | Lives lost | Rounds to the clock | Falls |
|---|---|---|---|---|---|---|---|---|
| learned | 20 | **426** | 114 | 19.1 | 0.25 | 2.75 | 5/20 | 17 |
| planner | 20 | **378** | 115 | 19.0 | 0.10 | 2.10 | 13/20 | 31 |
| neutral | 20 | **0** | 0 | 0.0 | 0.00 | 0.15 | 0/20 | 0 |

Learned beats planner on 10/20 seeds (same seed = same maze layout, spawn jitter and ghost RNG).
<!-- bench:end -->

Seed 0, the causality seed used by `./run.sh`:

<!-- results:start -->
RESULTS_TABLE
<!-- results:end -->

**Causality test** (`tests/test_eval_causality.py`, run by `./run.sh`): the trained strategy must
score ≥ 200 with ≥ 12 coins on seed 0; the neutral Duck-Man (strategy always "stay") collects 0
coins; every policy's call count equals the number of control steps; gait actions are finite and
bounded. The learned Duck-Man's edge over the planner is real but modest (+48 points, 10/20 seeds);
what it learned that the planner never does is to hunt ghosts during power windows.

## Reproduce

```bash
./run.sh                    # venv, pinned deps, tests, evaluations, result.mp4   (CPU only, ~10 min)
./run.sh train --run r --generations 300 --pop 48 --seeds 3 --workers 10 --sigma 0.03 --lr 0.005 \
    --resume checkpoints/strategy_final.npz            # optional: continue evolving the strategy
python -m duckman.imitate --rounds 40                  # optional: rebuild the behaviour-cloning init
python -m duckman.bench --seeds 20                     # optional: the 20-seed table above
python -m duckman.eval --policy learned --seed 7 --checkpoint checkpoints/strategy_final.npz --video x.mp4
```

Python ≥ 3.12. Evaluation is deterministic on CPU for a given seed. Retraining the stand-up policy
needs a CUDA GPU and Pollen's `microduck_rl` at commit 29e887e:
`uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 7000 --agent.logger tensorboard`
then `uv run scripts/export.py Mjlab-StandUp-Flat-MicroDuck --checkpoint-file <model_7000.pt>`.

## Video

`result.mp4` = title card → generation-0 clip (untrained strategy network, 2× speed, labelled) →
final trained round on seed 0 (2× speed, labelled) → learning curve → end card with the numbers
from `results/*.json`. Clips are unedited renders of the evaluation runs; only the playback speed is
changed, and it is burned into the frame.

## Limitations

- Simulation only. Nothing here is evidence of hardware behaviour.
- The strategy layer uses perfect game state; a real robot would need perception.
- Pollen's gait walks at ~0.15 m/s and cannot strafe, so the game is slow and turn-heavy; 2×
  playback is used for watchability.
- **Face-up recovery does not work.** The stand-up policy rises from sitting (5/5) and face-down
  (6/6) but not from its back (0/8 across roll angles) after 7,000 iterations. A Duck-Man knocked
  onto its back stays down and loses its remaining lives to tags — a fair knockout, but a gap.
- The learned strategy is aggressive: it usually spends all three lives by ~160 s hunting ghosts.
  Under the scoring rules that is the higher-scoring choice; it does make rounds shorter.
- Training seeds are six maze layouts; generalisation to held-out seeds is shown above, but the maze
  shape itself is fixed.

## Next ideas

1. **Duck-Man Dodgeball.** Replace power pellets with a ball the Duck-Man kicks down a corridor to
   knock out a ghost. Corridors solve the aiming problem (blind kick spread ±8° → ±6–12 cm at one to
   two cells, inside a 45 cm corridor); the hard part is kick alignment (±1.5 cm) with a navigator
   that arrives within 6 cm.
2. Train the strategy end to end from its own rollouts with more seeds per generation once compute
   allows; the current policy plateaued around fitness 200 on the pool.
3. Face-up recovery: Pollen's roll-noise curriculum did not transfer within our budget; a
   reverse-curriculum from near-on-side starts is the obvious next experiment.
4. Fewer, faster ghosts and a larger maze once a faster gait is available.

## Attribution

See `THIRD_PARTY_NOTICES.md`. Microduck model and policies © Pollen Robotics (Apache-2.0 code and
policies, CC BY-SA-NC 4.0 meshes). BAM by Rhoban. This entry's code is Apache-2.0.
