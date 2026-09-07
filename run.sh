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
.venv/bin/python -m pytest -q tests
mkdir -p results
CK=checkpoints/strategy_final.npz
.venv/bin/python -m duckman.eval --policy learned --seed 0 --checkpoint "$CK" --out results/learned_seed0.json \
    --video results/final_seed0.mp4 --speed 2 --label "Final trained strategy, seed 0"
.venv/bin/python -m duckman.eval --policy planner --seed 0 --out results/planner_seed0.json
.venv/bin/python -m duckman.eval --policy neutral --seed 0 --out results/neutral_seed0.json --max-t 60
.venv/bin/python -m duckman.eval --policy learned --seed 0 --checkpoint checkpoints/strategy_gen0.npz \
    --out results/gen0_seed0.json --video results/gen0_seed0.mp4 --speed 2 --max-t 60 --label "Generation 0 (untrained), seed 0"
.venv/bin/python -m duckman.make_video \
    --clips "results/gen0_seed0.mp4:Generation 0 - untrained" "results/final_seed0.mp4:Final policy - full round" \
    --results results/learned_seed0.json results/planner_seed0.json results/neutral_seed0.json \
    --curve checkpoints/curve.csv --images training/standup/mean_reward.png --out result.mp4
echo "done: result.mp4 + results/*.json"
