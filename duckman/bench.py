"""Held-out benchmark: learned vs planner vs neutral vs frozen over N seeds. Writes results/bench.json and a markdown table.
python -m duckman.bench --seeds 60 --workers 4 --checkpoint checkpoints/strategy_final.npz
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from .constants import ROOT
from .eval import run_eval


def _one(args):
    policy, seed, ck = args
    r = run_eval(policy, seed, ck)          # every policy gets the full 240 s clock
    return {k: r[k] for k in ("policy", "seed", "score", "coins", "pellets", "ghosts", "lives_lost", "falls", "end", "t")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/strategy_final.npz"))
    ap.add_argument("--out", default=str(ROOT / "results/bench.json"))
    a = ap.parse_args()
    jobs = [(p, s, a.checkpoint if p == "learned" else None) for p in ("learned", "planner", "neutral", "frozen") for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    Path(a.out).parent.mkdir(exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    lines = ["| Policy | Seeds | Mean score | Std | Mean coins | Ghosts caught | Lives lost | Rounds to the clock | Falls |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in ("learned", "planner", "neutral", "frozen"):
        rs = [r for r in rows if r["policy"] == p]
        sc = np.array([r["score"] for r in rs])
        lines.append(f"| {p} | {len(rs)} | **{sc.mean():.0f}** | {sc.std():.0f} | {np.mean([r['coins'] for r in rs]):.1f} | "
                     f"{np.mean([r['ghosts'] for r in rs]):.2f} | {np.mean([r['lives_lost'] for r in rs]):.2f} | "
                     f"{sum(r['end'] == 'timeout' for r in rs)}/{len(rs)} | {sum(r['falls'] for r in rs)} |")
    learned = [r["score"] for r in rows if r["policy"] == "learned"]
    planner = [r["score"] for r in rows if r["policy"] == "planner"]
    diff = np.array(learned) - np.array(planner)
    wins = int((diff > 0).sum()); ties = int((diff == 0).sum())
    # paired statistics: mean difference with a 95% bootstrap CI, and an exact two-sided sign test on wins vs losses
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(20000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    from math import comb
    n_eff = len(diff) - ties
    k = min(wins, n_eff - wins)
    p_sign = min(1.0, 2 * sum(comb(n_eff, i) for i in range(k + 1)) / 2 ** n_eff) if n_eff else 1.0
    lines.append(f"\nPaired per-seed difference (learned - planner): mean {diff.mean():+.0f} points, 95% bootstrap CI "
                 f"[{lo:+.0f}, {hi:+.0f}]; learned wins {wins}/{a.seeds} seeds ({ties} ties), two-sided sign test p = {p_sign:.2f}. "
                 f"The mean difference is {'' if not (lo <= 0 <= hi) else 'not '}statistically significant (bootstrap CI {'excludes' if not (lo <= 0 <= hi) else 'includes'} 0); "
                 f"the win rate is {'not ' if p_sign > 0.05 else ''}significant (sign test).")
    table = "\n".join(lines)
    Path(a.out).with_suffix(".md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
