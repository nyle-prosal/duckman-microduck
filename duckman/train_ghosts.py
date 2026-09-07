"""Co-evolution step: evolve a shared ghost network against the frozen learned Duck-Man.
Fitness = -(Duck-Man score) + 30 per tag. python -m duckman.train_ghosts --run ghosts1 --generations 200"""
import argparse
import csv
import os
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from .constants import CLOCK_S, ROOT
from .maze import Maze
from .game import Game
from .policies import DuckManPolicy
from .strategy_net import NetStrategy, MLP
from .ghost_net import NetGhostStrategy, N_GHOST_FEATURES, learned_ghosts
from .eval import resolve_recovery
from .train_es import SEED_POOL

PAC_CK = str(ROOT / "checkpoints/strategy_final.npz")
_GAMES = {}


def _game(seed):
    if seed not in _GAMES:
        if len(_GAMES) >= 6:
            _GAMES.clear()
        rec = resolve_recovery("auto")
        st = NetGhostStrategy(MLP(N_GHOST_FEATURES))
        _GAMES[seed] = Game(Maze(), seed, DuckManPolicy(NetStrategy.load(PAC_CK), rec),
                            learned_ghosts(st, "standup" if rec == "standup" else "none"))
    return _GAMES[seed]


def rollout(args):
    flat, seed, max_t = args
    g = _game(seed)
    g.policies["G0_"].p.mlp.set_flat(flat)          # shared object across the four ghosts
    g.reset()
    r = g.run(max_t=max_t)
    return dict(score=r["score"], coins=r["coins"], lives_lost=r["lives_lost"], ghosts_caught=r["ghosts"], t=r["t"], seed=seed)


def fitness(res):
    return -res["score"] + 30.0 * res["lives_lost"]


def train(run_name, generations, pop=48, sigma=0.03, lr=0.005, seeds_per_gen=3, workers=8, max_t=CLOCK_S, root=ROOT / "runs", resume=None, master_seed=0):
    out = Path(root) / run_name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(master_seed)
    theta = MLP(N_GHOST_FEATURES, seed=master_seed).get_flat() if resume is None else np.load(resume)["flat"].astype(np.float32)
    np.savez(out / "gen_0000.npz", flat=theta, n_in=N_GHOST_FEATURES)
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    best_fit, best_theta, stale = -np.inf, theta.copy(), 0
    half = pop // 2
    with open(out / "curve.csv", "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["gen", "fit_mean", "fit_max", "theta_fit", "theta_pac_score", "theta_tags", "best_fit", "wall_s"])
        with Pool(workers) as pool:
            for gen in range(1, generations + 1):
                t0 = time.time()
                eps = rng.standard_normal((half, theta.size)).astype(np.float32)
                eps = np.concatenate([eps, -eps])
                seeds = [int(s) for s in rng.choice(SEED_POOL, size=seeds_per_gen, replace=False)]
                jobs = [(theta + sigma * eps[i], s, max_t) for i in range(pop) for s in seeds] + [(theta, s, max_t) for s in SEED_POOL]
                res = pool.map(rollout, jobs, chunksize=1)
                cand = res[:pop * seeds_per_gen]
                per = np.array([fitness(r) for r in cand]).reshape(pop, seeds_per_gen).mean(1)
                base = res[pop * seeds_per_gen:]
                theta_fit = float(np.mean([fitness(r) for r in base]))
                pac = float(np.mean([r["score"] for r in base]))
                tags = float(np.mean([r["lives_lost"] for r in base]))
                top = int(per.argmax())
                top_theta = (theta + sigma * eps[top]).astype(np.float32)
                top_fit = float(np.mean([fitness(r) for r in pool.map(rollout, [(top_theta, s, max_t) for s in SEED_POOL], chunksize=1)]))
                improved = False
                for cf, ct, tag in ((theta_fit, theta, "theta"), (top_fit, top_theta, "top")):
                    if cf > best_fit:
                        best_fit, best_theta, stale, improved = cf, ct.copy(), 0, True
                        np.savez(out / "best.npz", flat=ct, n_in=N_GHOST_FEATURES, fit=cf, gen=gen)
                        print(f"gen {gen}: new best ghosts from {tag}: fitness {cf:.1f}", flush=True)
                if not improved:
                    stale += 1
                ranks = per.argsort().argsort().astype(np.float32)
                util = ranks / (pop - 1) - 0.5
                grad = (eps * util[:, None]).sum(0) / (pop * sigma)
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad ** 2
                theta = (theta + lr * m / (np.sqrt(v) + 1e-8)).astype(np.float32)
                if stale >= 3:
                    theta, stale = best_theta.copy(), 0
                    m[:] = 0
                    v[:] = 0
                w.writerow([gen, per.mean(), per.max(), theta_fit, pac, tags, best_fit, round(time.time() - t0, 1)])
                f.flush()
                np.savez(out / f"gen_{gen:04d}.npz", flat=theta, n_in=N_GHOST_FEATURES)
                print(f"gen {gen}: ghost fit mean {per.mean():.1f} max {per.max():.1f} | theta: Duck-Man scores {pac:.1f}, tags {tags:.2f} | best {best_fit:.1f} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--resume")
    a = ap.parse_args()
    train(a.run, a.generations, a.pop, a.sigma, a.lr, a.seeds, a.workers, resume=a.resume)


if __name__ == "__main__":
    main()
