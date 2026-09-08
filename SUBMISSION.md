# Submission sheet: Duck-Man

| HIM Arena field | Value |
|---|---|
| Challenge | `microduck-sports-sim-2026` (Microduck · Best Sports Sim) |
| Name | `Duck-Man: maze tag for Microduck` |
| Kind | `both` (policy + simulator) |
| Robot / simulator | `microduck` / `mujoco` |
| Run command | `./run.sh` (full, 20-30 min CPU); `./run.sh quick` runs a 4-minute check |
| Caption | see below |
| Status | ready (not work-in-progress) |
| Public repository | https://github.com/nyle-prosal/duckman-microduck |

**Caption**

> Maze tag for 5 real Microducks in MuJoCo (Pollen's BAM actuators). Trained here: a stand-up policy (PPO) and the Duck-Man strategy net (imitation, then evolution in-sim); ghosts scripted, gaits Pollen's. 60 held-out seeds: learned 406, planner 360, disabled 0. Sim only.

**Published**

Uploaded through the HIM Arena web form on 2026-09-08 (the CLI rejected the challenge id that day):
https://arena.himrobotics.com/uploads/source_b69ff406a5b1b17ca51c (upload id `source_b69ff406a5b1b17ca51c`,
kind policy + simulator, status ready). ZIP SHA-256 on the page: `855d6b24ffc1227fb2d5f093d12e3c3fd6a6c465b3360fa47adb612337940113`.

The equivalent CLI command:

```
him upload ./submission.zip --video ./result.mp4 --kind policy --robot microduck --simulator mujoco \
  --challenge microduck-sports-sim-2026 --name "Duck-Man: maze tag for Microduck" \
  --caption "<caption above>" --run-command "./run.sh" --ready --yes --json
```

## What to watch for in `result.mp4` (3 min 41 s)

| Time | Segment | Look for |
|---|---|---|
| 0:00 | Cold open: seed 2, first power window (2×) | pellet → all four ghosts turn blue → the Duck-Man catches Pinky (+200) |
| 0:10 | Title card | what is learned, what is scripted, simulation only |
| 0:13 | Real time, unedited, seed 2 | five ducks under learned gaits, physics at 1× |
| 0:33 | Generation 0 (untrained network), 2× | it collects a few coins and camps; the baseline for the learning curve |
| 0:53 | Neutral baseline, real time | strategy never moves; the ghosts converge and tag it at 61 s; the causality control on camera |
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
