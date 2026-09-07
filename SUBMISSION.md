# Submission sheet — Duck-Man

| HIM Arena field | Value |
|---|---|
| Challenge | `microduck-sports-sim-2026` (Microduck · Best Sports Sim) |
| Name | `Duck-Man — maze tag for Microduck` |
| Kind | `both` (policy + simulator) |
| Robot / simulator | `microduck` / `mujoco` |
| Run command | `./run.sh` (full, ~20 min CPU) — `./run.sh quick` for a 4-minute check |
| Caption | see below |
| Status | ready (not work-in-progress) |

**Caption**

> Maze tag for 5 real Microducks in MuJoCo with Pollen's BAM actuator model. Trained here: a stand-up
> policy (PPO on Pollen's Microduck task, one GPU) and the Duck-Man's strategy network (imitation of our
> scripted planner, then evolution strategies in-sim on a laptop). Over 60 held-out seeds: learned 406,
> scripted planner 360, disabled baselines 0. Ghosts and cell navigation are scripted; every gait is
> Pollen's learned policy; coins count only when physically knocked over. Balance assistance: none.
> Simulation only.

**Publish command**

```
him upload ./submission.zip --video ./result.mp4 --kind both --robot microduck --simulator mujoco \
  --challenge microduck-sports-sim-2026 --name "Duck-Man — maze tag for Microduck" \
  --caption "<caption above>" --run-command "./run.sh" --ready --yes --json
```

## What to watch for in `result.mp4` (3 min 40 s)

| Time | Segment | Look for |
|---|---|---|
| 0:00 | Cold open: seed 2, first power window (2×) | pellet → all four ghosts turn blue → the Duck-Man catches Pinky (+200) |
| 0:10 | Title card | what is learned, what is scripted, simulation only |
| 0:13 | Real time, unedited, seed 2 | five ducks under learned gaits, physics at 1× |
| 0:33 | Generation 0 (untrained network), 2× | it collects a few coins and camps — the baseline for the learning curve |
| 0:53 | Neutral baseline, real time | strategy never moves; the ghosts converge and tag it at 61 s — the causality control on camera |
| 1:13 | Final policy, full seed-2 round, 2× | side panel: score, lives, power bar, ticker, legend; 780 points, two ghosts caught, ends on the clock |
| 3:13 | Stand-up policy demo | spawned face-down, the policy trained on the GPU stands it up; labelled as a demo, not a scored round |
| 3:24 | Strategy learning curves, then the stand-up training curve | all four evolution runs, including the two that failed |
| 3:36 | End card | the numbers from `results/*.json` |

Everything in the arena view is the evaluation itself; the panel, ticker and legend are drawn from game
state. Speed-ups are labelled on screen. No frame shows an outcome the evaluation did not produce.

## Where the evidence is

`README.md` (results, causality, limitations) · `evidence/README.md` (every number → its file) ·
`assets/PROVENANCE.json` (every shipped file hashed with its upstream) · `tests/` (26 tests incl. the
no-simulator-write AST check) · `training/` (both trained policies' records).
