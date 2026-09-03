# Duck-Man — maze tag for Microduck

Entry for the HIM Arena **Microduck · Best Sports Sim** challenge.

Five real Microducks play maze tag in MuJoCo. One Duck-Man collects coins by physically knocking
them over, four ghosts chase it, and a power pellet reverses the chase for ten seconds. The
Duck-Man's strategy network was **trained by evolution strategies inside this simulation on a
laptop CPU**. Every duck walks with Pollen Robotics' published learned gait through the BAM
actuator model. Simulation only; no hardware claims.

## What is learned and what is scripted (please read)

| Layer | Type | Who made it | Evidence |
|---|---|---|---|
| Duck-Man strategy (which cell to go to next) | **trained** — evolution strategies, full MuJoCo rollouts | this entry | `checkpoints/strategy_final.npz` (sha256 in `results/learned_seed0.json`), `checkpoints/curve.csv`, learning-curve frame in the video |
| Cell navigation (turn, walk to a cell centre) | scripted controller | this entry | `duckman/navigator.py` |
| Ghost behaviour (chase / ambush / mirror / shy, flee, go home) | scripted controller | this entry | `duckman/ghosts.py` |
| Scripted planner baseline | scripted controller | this entry | `duckman/planner.py` |
| Walking, standing, sit-and-stand gaits (14 joint targets at 50 Hz) | learned (PPO), **unmodified** | Pollen Robotics, `pollen-robotics/microduck-policies` | `assets/policies/*.onnx` |
| Actuator physics | BAM XL330 M6 voltage model, one controller per duck | Rhoban / Pollen | `duckman/bam_loader.py` |

**Balance assistance: none.** No fixed base, no external stabilisation, no joint animation. All 70
servos of all five ducks are driven only by the ONNX gait networks, every control step.
**Observation:** the strategy layer sees perfect game state (positions, coin states, ghost modes);
the gait networks see proprioception only, exactly as on the robot.

## Rules

3 lives, 240 s clock. Coin +10 (7 cm token, counts when the Duck-Man has touched it and it is
toppled or knocked ≥ 6 cm from its spawn), power pellet +50 (10 s of power), ghost caught in power
mode +200 (the ghost walks home), maze cleared +500. A tag is body contact between a ghost and the
unpowered Duck-Man: it loses a life, holds still, sits while the ghosts walk back to their house,
then stands up with the sit-and-stand policy and play resumes. Nothing is ever teleported. Ghosts
pass through each other and through tokens (collision groups), as in the arcade; every other
contact is ordinary physics.

## Results (seed 0, same maze, same ghosts)

RESULTS_TABLE

Causality test (`tests/test_eval_causality.py`, run by `./run.sh`): the trained policy must score
≥ 200 with ≥ 12 coins on seed 0; the neutral policy (strategy always "stay") collects 0 coins and
loses a life; policy call counts equal control steps; gait actions are finite and bounded.

## Reproduce

```bash
./run.sh                       # venv, pinned deps, tests, 4 evaluations, result.mp4   (~10 min, CPU only)
./run.sh train --run r2 --generations 300 --pop 48 --seeds 2 --workers 10   # optional: retrain
python -m duckman.eval --policy learned --checkpoint checkpoints/strategy_final.npz --seed 7 --video x.mp4
```

Python ≥ 3.12. Deterministic on CPU for a given seed.

## Video

`result.mp4` = title card → generation-0 clip (untrained network, 2× speed, labelled) → final
policy full round (2× speed, labelled) → learning curve → end card with the numbers from
`results/*.json`. Clips are unedited renders of the evaluation runs; only the playback speed is
changed and it is burned into the frame.

## Limitations

- Simulation only. Nothing here is evidence of hardware behaviour.
- The strategy layer uses perfect state; a real robot would need perception.
- Pollen's gait walks at ~0.15 m/s and cannot strafe, so the game is slow and turn-heavy.
- A tag is a sit-down, not a knockdown: Pollen did not publish a stand-up policy, so recovery
  uses the published sit-and-stand policy. A tagged duck that has actually fallen cannot get up and
  the round ends ("fell_unrecoverable").
- Ghosts and tokens use collision groups so ghosts do not knock coins or each other over.

## Attribution

See `THIRD_PARTY_NOTICES.md`. Microduck model and policies © Pollen Robotics (Apache-2.0 code and
policies, CC BY-SA-NC 4.0 meshes). BAM by Rhoban. This entry's code is Apache-2.0.
