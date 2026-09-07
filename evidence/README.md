# Evidence index — where every number in the README comes from

| Claim in README | Produced by | File |
|---|---|---|
| 20-seed benchmark table (learned / planner / neutral / frozen), paired difference, CI, sign test | `python -m duckman.bench --seeds 20` | `results/bench.json` (60+ rows), `results/bench.md` |
| Seed-2 showcase round: score, coins, ghosts caught, lives, policy call counts, gait action bounds, checkpoint sha256 | `python -m duckman.eval --policy learned --seed 2 --checkpoint checkpoints/strategy_final.npz` | `results/learned_seed2.json` |
| Seed-2 planner comparison | `python -m duckman.eval --policy planner --seed 2` | `results/planner_seed2.json` |
| Seed-0 causality set: learned, planner, neutral (full 240 s), frozen (no network, full 240 s), generation-0 untrained | `./run.sh` | `results/learned_seed0.json`, `results/planner_seed0.json`, `results/neutral_seed0.json`, `results/frozen_seed0.json`, `results/gen0_seed0.json` |
| Causality assertions (learned ≥ 200 & ≥ 12 coins on seed 0; neutral and frozen collect 0; call count == control steps; finite bounded actions) | `pytest tests/test_eval_causality.py` | `tests/test_eval_causality.py` |
| No rollout code writes simulator state (qpos/qvel/ctrl/xfrc/mocap) | `pytest tests/test_no_sim_writes.py` | `tests/test_no_sim_writes.py` |
| No teleport of any duck or token between control steps; deterministic replays | `pytest tests/test_game.py` | `tests/test_game.py` |
| Strategy learning curve (generations 1–26, unperturbed-policy fitness and population score) | `duckman/train_es.py` | `checkpoints/curve.csv` |
| Stand-up policy training: 7,000 PPO iterations, mean reward per iteration, recovery table | Pollen's `uv run train Mjlab-StandUp-Flat-MicroDuck` on a HIM GPU machine; our recovery harness | `training/standup/mean_reward.csv`, `training/standup/mean_reward.png`, `training/standup/README.md` |
| Every shipped asset and checkpoint: sha256, size, upstream URL, revision, licence | `python -m duckman.provenance` | `assets/PROVENANCE.json` |
| Video provenance | `duckman/eval.py --video` (unedited episode renders) + `duckman/make_video.py` (cards and concatenation only) | `result.mp4` |

Rejected or superseded experiments are described in the README's Limitations and Next-ideas sections
(face-up recovery 0/8 at iterations 4499 and 7000; evolution without elitism regressing within 8
generations; sigma 0.1 destroying the cloned policy). Their raw logs live in the working repository under
`runs/` and are excluded from the ZIP by the challenge's packaging rules.
