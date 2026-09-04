"""Score checkpoints on the training seed pool. python -m duckman.compare a.npz b.npz [--workers 6]"""
import argparse
from multiprocessing import Pool
import numpy as np
from .train_es import rollout, fitness, SEED_POOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEED_POOL)
    a = ap.parse_args()
    jobs = [(np.load(p)["flat"].astype(np.float32), s, 240.0) for p in a.paths for s in a.seeds]
    with Pool(a.workers) as pool:
        res = pool.map(rollout, jobs, chunksize=1)
    i = 0
    for p in a.paths:
        rs = res[i:i + len(a.seeds)]
        i += len(a.seeds)
        print(f"{p:50s} fitness {np.mean([fitness(r) for r in rs]):7.1f}  score {np.mean([r['score'] for r in rs]):6.1f}"
              f"  lives lost {np.mean([r['lives_lost'] for r in rs]):.2f}  scores {[r['score'] for r in rs]}", flush=True)


if __name__ == "__main__":
    main()
