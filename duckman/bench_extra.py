"""Score an arbitrary Strategy class on held-out seeds (used for the plug-in example baseline).
python -m duckman.bench_extra examples_api.my_strategy:GreedyCoinStrategy --seeds 60 --workers 6"""
import argparse
import importlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from .constants import ROOT
from .maze import Maze
from .game import Game
from .policies import DuckManPolicy
from .ghosts import default_ghosts
from .eval import resolve_recovery


def _load(spec):
    mod, cls = spec.split(":")
    sys.path.insert(0, str(ROOT))
    return getattr(importlib.import_module(mod), cls)


def _one(args):
    spec, seed = args
    rec = resolve_recovery("auto")
    g = Game(Maze(), seed, DuckManPolicy(_load(spec)(), rec), default_ghosts("standup" if rec == "standup" else "none"))
    g.reset()
    r = g.run()
    return {"seed": seed, "score": r["score"], "coins": r["coins"], "ghosts": r["ghosts"], "lives_lost": r["lives_lost"], "end": r["end"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        rows = pool.map(_one, [(a.spec, s) for s in range(a.seeds)], chunksize=1)
    sc = np.array([r["score"] for r in rows])
    out = a.out or str(ROOT / f"results/extra_{a.spec.split(':')[1]}.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"{a.spec}: mean {sc.mean():.0f} std {sc.std():.0f} coins {np.mean([r['coins'] for r in rows]):.1f} "
          f"ghosts {np.mean([r['ghosts'] for r in rows]):.2f} lives {np.mean([r['lives_lost'] for r in rows]):.2f} over {a.seeds} seeds -> {out}")


if __name__ == "__main__":
    main()
