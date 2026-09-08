"""Evolution strategies (OpenAI-ES, antithetic, rank-shaped, Adam) for the Duck-Man strategy network.

Every rollout is a full Duck-Man round in the MuJoCo simulation. Rollouts run in a process pool.
python -m duckman.train_es --run run1 --generations 300 [--pop 64 --workers 10 --max-t 240 --seeds 4]
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

# one physics/inference thread per worker process; the pool provides the parallelism
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import numpy as np
from .constants import CLOCK_S, ROOT
from .maze import Maze
from .game import Game
from .policies import DuckManPolicy
from .strategy_net import MLP, NetStrategy, N_FEATURES

_GAMES = {}


def _game(seed, kind="scripted"):
    key = (seed, kind)
    if key not in _GAMES:
        if len(_GAMES) >= 12:
            _GAMES.clear()
        from .eval import resolve_recovery, make_ghosts
        rec = resolve_recovery("auto")
        _GAMES[key] = Game(Maze(), seed, DuckManPolicy(NetStrategy(MLP(N_FEATURES)), rec), make_ghosts(kind, rec))
    return _GAMES[key]


def rollout(args):
    flat, seed, max_t = args[:3]
    kind = args[3] if len(args) > 3 else "scripted"
    g = _game(seed, kind)
    g.policies["P_"].strategy.mlp.set_flat(flat)
    g.reset()
    r = g.run(max_t=max_t)
    return dict(score=r["score"], coins=r["coins"], lives_lost=r["lives_lost"], t=r["t"], seed=seed, end=r["end"])


def fitness(res):
    return res["score"] - 100.0 * res["lives_lost"]


SEED_POOL = [1000, 1001, 1002, 1003, 1004, 1005]   # training layouts; evaluation uses seeds 0-999


def train(run_name, generations, pop=64, sigma=0.1, lr=0.02, seeds_per_gen=4, workers=10, max_t=CLOCK_S,
          root=ROOT / "runs", resume=None, master_seed=0, ghosts="scripted", fresh_bar=False):
    kinds = ["scripted", "learned"] if ghosts == "mixed" else [ghosts]
    out = Path(root) / run_name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(master_seed)
    mlp = MLP(N_FEATURES, seed=master_seed)
    theta = mlp.get_flat() if resume is None else np.load(resume)["flat"].astype(np.float32)
    if not (out / "gen_0000.npz").exists():
        np.savez(out / "gen_0000.npz", flat=theta, n_in=N_FEATURES)
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    best_fit, best_theta, stale = -np.inf, theta.copy(), 0
    if resume is not None and not fresh_bar and "fit" in np.load(resume):
        best_fit = float(np.load(resume)["fit"])      # resuming from an elite: only a better policy may replace it
    half = pop // 2
    start_gen = 1
    if (out / "curve.csv").exists():
        rows = list(csv.DictReader(open(out / "curve.csv")))
        if rows:
            start_gen = int(rows[-1]["gen"]) + 1
    with open(out / "curve.csv", "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["gen", "fit_mean", "fit_max", "fit_min", "score_mean", "coins_mean", "lives_lost_mean",
                        "theta_fit", "theta_score", "wall_s"])
        with Pool(workers) as pool:
            for gen in range(start_gen, start_gen + generations):
                t0 = time.time()
                eps = rng.standard_normal((half, theta.size)).astype(np.float32)
                eps = np.concatenate([eps, -eps])
                seeds = [int(s) for s in rng.choice(SEED_POOL, size=min(seeds_per_gen, len(SEED_POOL)), replace=False)]
                cand_jobs = [(theta + sigma * eps[i], s, max_t, k) for i in range(pop) for s in seeds for k in kinds]
                base_jobs = [(theta, s, max_t, k) for s in SEED_POOL for k in kinds]
                res = pool.map(rollout, cand_jobs + base_jobs, chunksize=1)
                ncand = pop * seeds_per_gen * len(kinds)
                cand = res[:ncand]
                per = np.array([fitness(r) for r in cand]).reshape(pop, seeds_per_gen * len(kinds)).mean(1)
                base = res[ncand:]                         # the unperturbed policy on the WHOLE pool (x ghost kinds)
                theta_fit = float(np.mean([fitness(r) for r in base]))
                theta_score = float(np.mean([r["score"] for r in base]))
                # (1+lambda)-style elitism: re-score the top individual on the whole pool and adopt it if it wins
                top = int(per.argmax())
                top_theta = (theta + sigma * eps[top]).astype(np.float32)
                top_res = pool.map(rollout, [(top_theta, s, max_t, k) for s in SEED_POOL for k in kinds], chunksize=1)
                top_fit = float(np.mean([fitness(r) for r in top_res]))
                improved = False
                for cand_fit, cand_theta, tag in ((theta_fit, theta, "theta"), (top_fit, top_theta, "top individual")):
                    if cand_fit > best_fit:
                        best_fit, best_theta, stale, improved = cand_fit, cand_theta.copy(), 0, True
                        np.savez(out / "best.npz", flat=cand_theta, n_in=N_FEATURES, fit=cand_fit, gen=gen)
                        print(f"gen {gen}: new best from {tag}: pool fitness {cand_fit:.1f}", flush=True)
                if not improved:
                    stale += 1
                ranks = per.argsort().argsort().astype(np.float32)
                util = ranks / (pop - 1) - 0.5
                grad = (eps * util[:, None]).sum(0) / (pop * sigma)
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad ** 2
                theta = (theta + lr * m / (np.sqrt(v) + 1e-8)).astype(np.float32)
                if stale >= 3:                      # elitism: go back to the best known policy and explore from there
                    theta, stale = best_theta.copy(), 0
                    m[:] = 0
                    v[:] = 0
                    print(f"gen {gen}: reverted to best (fit {best_fit:.1f})", flush=True)
                sc = np.array([r["score"] for r in cand])
                co = np.array([r["coins"] for r in cand])
                ll = np.array([r["lives_lost"] for r in cand])
                w.writerow([gen, per.mean(), per.max(), per.min(), sc.mean(), co.mean(), ll.mean(), theta_fit, theta_score,
                            round(time.time() - t0, 1)])
                f.flush()
                np.savez(out / f"gen_{gen:04d}.npz", flat=theta, n_in=N_FEATURES)
                print(f"gen {gen}: fit mean {per.mean():.1f} max {per.max():.1f} (top on pool {top_fit:.1f}) | theta fit {theta_fit:.1f} score {theta_score:.1f} | best {best_fit:.1f}"
                      f" | pop score {sc.mean():.1f} coins {co.mean():.1f} lives {ll.mean():.2f} ({time.time() - t0:.0f}s)", flush=True)
    return {"generations": generations, "theta": theta, "out": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-t", type=float, default=CLOCK_S)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--resume")
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--ghosts", default="scripted", help="scripted | mixed (scripted and learned ghosts, both count)")
    ap.add_argument("--fresh-bar", action="store_true", help="do not inherit the resumed elite's fitness bar (new objective)")
    a = ap.parse_args()
    train(a.run, a.generations, a.pop, a.sigma, a.lr, a.seeds, a.workers, a.max_t, resume=a.resume, master_seed=a.master_seed,
          ghosts=a.ghosts, fresh_bar=a.fresh_bar)


if __name__ == "__main__":
    main()
