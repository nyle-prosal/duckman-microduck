#!/usr/bin/env bash
# Reproduces the Duck-Man entry on CPU: tests, evaluations (learned / planner / neutral), result.mp4.
#   ./run.sh            full reproduction (~10 min on a laptop)
#   ./run.sh train ...  optional: launch evolution-strategies training (e.g. --run r2 --generations 300)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then "$PY" -m venv .venv; fi
.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements.txt
export PYTHONPATH="$PWD"
if [ "${1:-}" = "train" ]; then shift; exec .venv/bin/python -m duckman.train_es "$@"; fi
if [ "${1:-}" = "quick" ]; then
  # ~4 min: tests + the causality pair on seed 0, no video
  .venv/bin/python -m pytest -q tests
  mkdir -p results
  .venv/bin/python -m duckman.eval --policy learned --seed 0 --checkpoint checkpoints/strategy_final.npz --out results/learned_seed0.json
  .venv/bin/python -m duckman.eval --policy neutral --seed 0 --out results/neutral_seed0.json
  echo "quick check done: results/learned_seed0.json results/neutral_seed0.json"; exit 0
fi
START=$(date +%s)
.venv/bin/python -m pytest -q tests
mkdir -p results
CK=checkpoints/strategy_final.npz
.venv/bin/python -m duckman.eval --policy learned --seed 0 --checkpoint "$CK" --out results/learned_seed0.json
.venv/bin/python -m duckman.eval --policy planner --seed 0 --out results/planner_seed0.json
.venv/bin/python -m duckman.eval --policy neutral --seed 0 --out results/neutral_seed0.json
.venv/bin/python -m duckman.eval --policy frozen --seed 0 --out results/frozen_seed0.json
# clips for the video: 30 s of seed 2 in real time, the neutral baseline in real time, the stand-up policy demo
.venv/bin/python -m duckman.eval --policy learned --seed 2 --checkpoint "$CK" --video results/realtime_seed2.mp4 --speed 1 --max-t 20 \
    --label "Real time, unedited, seed 2 (first 20 s)"
.venv/bin/python -m duckman.eval --policy neutral --seed 0 --video results/neutral_full.mp4 --speed 1 --max-t 70 \
    --label "Neutral baseline: gait runs, strategy never moves - 0 coins"
.venv/bin/python -m duckman.cut results/neutral_full.mp4 results/neutral_clip.mp4 --start 50 --end 70   # window around its first tag (61 s)
.venv/bin/python -m duckman.demo_standup --out results/standup_demo.mp4
# showcase round for the video: held-out seed 2 (learned and planner both evaluated, unedited)
.venv/bin/python -m duckman.eval --policy learned --seed 2 --checkpoint "$CK" --out results/learned_seed2.json \
    --video results/final_seed2.mp4 --speed 2 --label "Final trained strategy, seed 2"
.venv/bin/python -m duckman.eval --policy planner --seed 2 --out results/planner_seed2.json
.venv/bin/python -m duckman.eval --policy learned --seed 0 --checkpoint checkpoints/strategy_gen0.npz \
    --out results/gen0_seed0.json --video results/gen0_seed0.mp4 --speed 2 --max-t 40 --label "Generation 0 (untrained), seed 0"
# cold open: the first power window and ghost catch of the seed-2 round (video seconds 37-47 = sim 74-94 s)
.venv/bin/python -m duckman.cut results/final_seed2.mp4 results/coldopen.mp4 --start 37 --end 47
.venv/bin/python -m duckman.make_video \
    --cold-open "results/coldopen.mp4:Power window and ghost catch, seed 2 (2x)" \
    --clips "results/realtime_seed2.mp4:Real time, unedited" "results/gen0_seed0.mp4:Generation 0 - untrained" \
            "results/neutral_clip.mp4:Neutral baseline - never moves" "results/final_seed2.mp4:Final policy - full round, seed 2" \
            "results/standup_demo.mp4:Stand-up policy demo (trained here)" \
    --results results/learned_seed2.json results/planner_seed2.json results/learned_seed0.json results/planner_seed0.json results/neutral_seed0.json results/frozen_seed0.json \
    --curve checkpoints/curve.csv --images training/strategy/curves.png training/standup/mean_reward.png --out result.mp4
echo "done: result.mp4 + results/*.json in $(( $(date +%s) - START )) s"
