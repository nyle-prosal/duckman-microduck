"""Robustness to Pollen's battery-voltage domain-randomisation range (BAM vin 6.5-8.2 V; nominal 7.4).
python -m duckman.robustness --seeds 10 --workers 6"""
import argparse
import json
from multiprocessing import Pool
import numpy as np
from .constants import ROOT
from .eval import run_eval

VINS = [6.5, 7.4, 8.2]


def _one(args):
    policy, seed, vin = args
    r = run_eval(policy, seed, str(ROOT / "checkpoints/strategy_final.npz") if policy == "learned" else None, vin=vin)
    return {"policy": policy, "seed": seed, "vin": vin, "score": r["score"], "coins": r["coins"], "lives_lost": r["lives_lost"], "falls": r["falls"], "end": r["end"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = [(p, s, v) for v in VINS for p in ("learned", "planner") for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    json.dump(rows, open(ROOT / "results/robustness.json", "w"), indent=1)
    lines = ["| Battery (BAM vin) | Learned mean score | Planner mean score | Learned lives lost | Planner lives lost | Falls (all ducks, learned / planner) |", "|---|---|---|---|---|---|"]
    for v in VINS:
        L = [r for r in rows if r["vin"] == v and r["policy"] == "learned"]
        P = [r for r in rows if r["vin"] == v and r["policy"] == "planner"]
        lines.append(f"| {v} V{' (nominal)' if v == 7.4 else ''} | **{np.mean([r['score'] for r in L]):.0f}** | {np.mean([r['score'] for r in P]):.0f} | "
                     f"{np.mean([r['lives_lost'] for r in L]):.2f} | {np.mean([r['lives_lost'] for r in P]):.2f} | {sum(r['falls'] for r in L)} / {sum(r['falls'] for r in P)} |")
    t = "\n".join(lines) + f"\n\nSeeds 0-{a.seeds - 1} per cell. 6.5-8.2 V is the per-env battery range Pollen randomises during gait training; every duck's actuators run at the given voltage.\n"
    (ROOT / "results/robustness.md").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
