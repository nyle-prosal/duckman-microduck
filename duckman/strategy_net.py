"""Duck-Man strategy network: maze-relative features -> 5 logits (E, W, S, N, stay), wall-masked argmax."""
import numpy as np

ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
R = 2   # local window radius -> 5x5


def features(view, cell):
    m = view.maze
    n = (2 * R + 1) ** 2
    walls, coins = np.zeros(n, np.float32), np.zeros(n, np.float32)
    i = 0
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            c = (cell[0] + dc, cell[1] + dr)
            walls[i] = 0.0 if c in m.corridor else 1.0
            coins[i] = 1.0 if view.coins_alive.get(c) else (0.5 if view.pellets_alive.get(c) else 0.0)
            i += 1
    gh = []
    for k in range(4):
        g = f"G{k}_"
        gc = view.duck_cell[g]
        mode = view.ghost_mode[g]
        gh += [np.clip((gc[0] - cell[0]) / 5, -1, 1), np.clip((gc[1] - cell[1]) / 5, -1, 1),
               1.0 if mode == "frightened" else 0.0, 1.0 if mode in ("eaten", "home") else 0.0]
    pel = [c for c, a in view.pellets_alive.items() if a]
    if pel:
        p = min(pel, key=lambda c: abs(c[0] - cell[0]) + abs(c[1] - cell[1]))
        pv = [np.clip((p[0] - cell[0]) / 6, -1, 1), np.clip((p[1] - cell[1]) / 6, -1, 1), 1.0]
    else:
        pv = [0.0, 0.0, 0.0]
    yaw = view.duck_yaw["P_"]
    head = [np.cos(yaw), np.sin(yaw)]
    misc = [view.power_left / 10.0, view.lives / 3.0, view.clock_left / 240.0,
            sum(view.coins_alive.values()) / max(1, len(view.coins_alive))]
    return np.concatenate([walls, coins, np.array(gh, np.float32), np.array(pv, np.float32),
                           np.array(head, np.float32), np.array(misc, np.float32)]).astype(np.float32)


N_FEATURES = 2 * (2 * R + 1) ** 2 + 16 + 3 + 2 + 4


class MLP:
    def __init__(self, n_in, hidden=(64, 64), n_out=5, seed=0):
        rng = np.random.default_rng(seed)
        dims = [n_in, *hidden, n_out]
        self.shapes = [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.W = [rng.standard_normal(s).astype(np.float32) * np.sqrt(2 / s[0]) for s in self.shapes]
        self.b = [np.zeros(s[1], np.float32) for s in self.shapes]
        self.n_params = sum(a * b + b for a, b in self.shapes)

    def get_flat(self):
        return np.concatenate([w.ravel() for w in self.W] + list(self.b))

    def set_flat(self, flat):
        flat = np.asarray(flat, np.float32)
        i = 0
        for k, (a, b) in enumerate(self.shapes):
            self.W[k] = flat[i:i + a * b].reshape(a, b).copy()
            i += a * b
        for k, (a, b) in enumerate(self.shapes):
            self.b[k] = flat[i:i + b].copy()
            i += b

    def forward(self, x):
        h = x
        for k in range(len(self.W)):
            h = h @ self.W[k] + self.b[k]
            if k < len(self.W) - 1:
                h = np.tanh(h)
        return h


class NetStrategy:
    label = "trained strategy network (ES)"

    def __init__(self, mlp):
        self.mlp = mlp

    def reset(self, seed):
        pass

    def choose(self, view, cell, options):
        logits = self.mlp.forward(features(view, cell))
        best, bestv = None, -np.inf
        for i, (dc, dr) in enumerate(ACTIONS):
            c = (cell[0] + dc, cell[1] + dr)
            if (dc, dr) == (0, 0):
                cand = None
            elif c in options:
                cand = c
            else:
                continue
            if logits[i] > bestv:
                best, bestv = cand, logits[i]
        return best

    def save(self, path):
        np.savez(path, flat=self.mlp.get_flat(), n_in=N_FEATURES)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = MLP(int(z["n_in"]))
        m.set_flat(z["flat"])
        return cls(m)


def random_net(seed):
    return NetStrategy(MLP(N_FEATURES, seed=seed))
