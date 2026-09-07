# Stand-up policy training record

Trained for this entry with Pollen Robotics' own stack (`pollen-robotics/microduck_rl` @ 29e887e,
mjlab 1.3.0 / MuJoCo Warp / rsl_rl PPO), unmodified task `Mjlab-StandUp-Flat-MicroDuck`.

| Item | Value |
|---|---|
| Hardware | 1× NVIDIA RTX 4090 (24 GB), HIM Arena SSH machine `gpu-l4-workspace`, 2026-09-04 |
| Environments | 4,096 parallel, 50 Hz control, BAM actuator model, Pollen's domain randomisation |
| Iterations | 0 → 4,499 (2.0 h, 1.55 s/iteration), resumed 4,499 → 7,000 (1.1 h) |
| Logger | tensorboard (`mean_reward.csv` here is the exported `Train/mean_reward` curve; `mean_reward.png` plots it) |
| Export | `uv run scripts/export.py Mjlab-StandUp-Flat-MicroDuck --checkpoint-file model_4499.pt` → `assets/policies/standup.onnx` (observation normaliser baked in, `[1,61] -> [1,14]`) |
| Cost | ≈ 395 HIM credits (4 h 15 min machine time) |

## Measured recovery (our CPU harness, 8 s window, success = upright > 0.75 and trunk height > 0.10 m)

| Start pose | iteration 4499 | iteration 7000 |
|---|---|---|
| Sitting (from Pollen's sit-and-stand policy) | 4/4 | 4/4 |
| Face-down, random yaw, z 5–9 cm | 6/6 | 6/6 |
| Face-up, spine roll 0°/±45°/90° | 0/8 | 0/8 |

The shipped file is the 4,499 checkpoint (identical measured behaviour, validated in-game over more
rounds). Face-up recovery did not emerge in either run; see the README limitations. The mean-reward
curve steps down at iterations 3,000 and 4,000 because Pollen's curriculum lowers the standing
reward weights there, not because the policy degrades.

Full checkpoints (`model_*.pt`, ~5 MB each) and raw tensorboard event files are kept out of the
submission ZIP; they are available on request and were produced exactly by the command in the top-level
README.
