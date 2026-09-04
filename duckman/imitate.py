"""Warm-start the strategy network by imitating the scripted planner (behaviour cloning), then hand
the weights to evolution strategies. python -m duckman.imitate --rounds 40 --workers 4 --out runs/bc/init.npz
"""
import argparse
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from .constants import ROOT
from .maze import Maze
from .game import Game
from .policies import DuckManPolicy
from .ghosts import default_ghosts
from .planner import PlannerStrategy
from .strategy_net import features, ACTIONS, MLP, N_FEATURES, NetStrategy
from .eval import resolve_recovery


class Recording(PlannerStrategy):
    def __init__(self):
        self.X, self.y = [], []

    def choose(self, view, cell, options):
        c = super().choose(view, cell, options)
        a = 4 if c is None else ACTIONS.index((c[0] - cell[0], c[1] - cell[1]))
        self.X.append(features(view, cell))
        self.y.append(a)
        return c


def collect(seed):
    rec = resolve_recovery("auto")
    st = Recording()
    g = Game(Maze(), seed, DuckManPolicy(st, rec), default_ghosts("standup" if rec == "standup" else "none"))
    g.reset()
    r = g.run()
    return np.array(st.X, np.float32), np.array(st.y, np.int64), r["score"]


def fit(X, y, epochs=300, lr=1e-3, seed=0, l2=1e-4):
    rng = np.random.default_rng(seed)
    m = MLP(N_FEATURES, seed=seed)
    n = len(X)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, 64):
            b = idx[s:s + 64]
            xb, yb = X[b], y[b]
            # forward
            h1 = np.tanh(xb @ m.W[0] + m.b[0])
            h2 = np.tanh(h1 @ m.W[1] + m.b[1])
            logits = h2 @ m.W[2] + m.b[2]
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            # backward (cross-entropy)
            d = p.copy()
            d[np.arange(len(b)), yb] -= 1
            d /= len(b)
            gW2, gb2 = h2.T @ d, d.sum(0)
            d1 = (d @ m.W[2].T) * (1 - h2 ** 2)
            gW1, gb1 = h1.T @ d1, d1.sum(0)
            d0 = (d1 @ m.W[1].T) * (1 - h1 ** 2)
            gW0, gb0 = xb.T @ d0, d0.sum(0)
            for W, gW in ((m.W[0], gW0), (m.W[1], gW1), (m.W[2], gW2)):
                W -= lr * (gW + l2 * W)
            m.b[0] -= lr * gb0
            m.b[1] -= lr * gb1
            m.b[2] -= lr * gb2
        if ep % 50 == 0 or ep == epochs - 1:
            logits = np.tanh(np.tanh(X @ m.W[0] + m.b[0]) @ m.W[1] + m.b[1]) @ m.W[2] + m.b[2]
            acc = float((logits.argmax(1) == y).mean())
            print(f"epoch {ep}: train accuracy {acc:.3f}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "runs/bc/init.npz"))
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    seeds = [2000 + i for i in range(a.rounds)]          # disjoint from ES pool (1000s) and eval seeds (0-999)
    with Pool(a.workers) as pool:
        parts = pool.map(collect, seeds)
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    scores = [p[2] for p in parts]
    print(f"collected {len(X)} decisions from {a.rounds} planner rounds; planner mean score {np.mean(scores):.1f}; "
          f"action histogram {np.bincount(y, minlength=5).tolist()}")
    m = fit(X, y, epochs=a.epochs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    NetStrategy(m).save(a.out)
    np.savez(Path(a.out).with_name("dataset.npz"), X=X, y=y, seeds=seeds, scores=scores)
    print("saved", a.out)


if __name__ == "__main__":
    main()
