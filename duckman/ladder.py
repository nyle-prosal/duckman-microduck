"""Generation ladder on held-out seeds: random init -> imitation only -> early ES -> final elite.
python -m duckman.ladder --seeds 20 --workers 6"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from .constants import ROOT
from .eval import run_eval

LADDER = [("random init (generation 0)", "checkpoints/ladder/00_random_init.npz"),
          ("imitation of the planner only", "checkpoints/ladder/01_imitation_only.npz"),
          ("evolution, generation 5", "checkpoints/ladder/02_es_gen05.npz"),
          ("evolution, generation 20 (shipped)", "checkpoints/ladder/03_es_gen20_final.npz")]


def _one(args):
    label, ck, seed = args
    r = run_eval("learned", seed, str(ROOT / ck))
    return {"stage": label, "seed": seed, "score": r["score"], "coins": r["coins"], "ghosts": r["ghosts"], "lives_lost": r["lives_lost"], "end": r["end"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = [(l, ck, s) for l, ck in LADDER for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    json.dump(rows, open(ROOT / "results/ladder.json", "w"), indent=1)
    lines = ["| Stage | Mean score | Std | Mean coins | Ghosts caught | Lives lost |", "|---|---|---|---|---|---|"]
    for l, _ in LADDER:
        rs = [r for r in rows if r["stage"] == l]
        sc = np.array([r["score"] for r in rs])
        lines.append(f"| {l} | **{sc.mean():.0f}** | {sc.std():.0f} | {np.mean([r['coins'] for r in rs]):.1f} | "
                     f"{np.mean([r['ghosts'] for r in rs]):.2f} | {np.mean([r['lives_lost'] for r in rs]):.2f} |")
    t = "\n".join(lines) + f"\n\nHeld-out seeds 0-{a.seeds - 1}, full 240 s rounds, identical ghosts and maze per seed.\n"
    (ROOT / "results/ladder.md").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
