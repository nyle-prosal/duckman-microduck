"""Learned ghost strategy: one network shared by all four ghosts, deciding the next cell while hunting.
Frightened/eaten/home behaviour stays scripted (flee / walk home), exactly as for the personality ghosts."""
import numpy as np
from .strategy_net import MLP, ACTIONS
from .ghosts import _away

R = 2


def ghost_features(view, prefix, cell):
    m = view.maze
    n = (2 * R + 1) ** 2
    walls = np.zeros(n, np.float32)
    i = 0
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            c = (cell[0] + dc, cell[1] + dr)
            walls[i] = 0.0 if m.walkable(c, True) else 1.0
            i += 1
    p = view.duck_cell["P_"]
    yaw = view.duck_yaw["P_"]
    rel = [np.clip((p[0] - cell[0]) / 6, -1, 1), np.clip((p[1] - cell[1]) / 6, -1, 1), np.cos(yaw), np.sin(yaw),
           min(1.0, (abs(p[0] - cell[0]) + abs(p[1] - cell[1])) / 12)]
    others = []
    for k in range(4):
        g = f"G{k}_"
        if g == prefix:
            continue
        gc = view.duck_cell[g]
        others += [np.clip((gc[0] - cell[0]) / 6, -1, 1), np.clip((gc[1] - cell[1]) / 6, -1, 1)]
    ident = [1.0 if prefix == f"G{k}_" else 0.0 for k in range(4)]
    misc = [1.0 if getattr(view, "scatter", False) else 0.0, view.power_left / 12.0,
            sum(view.coins_alive.values()) / max(1, len(view.coins_alive)), view.clock_left / 240.0]
    return np.concatenate([walls, np.array(rel, np.float32), np.array(others, np.float32),
                           np.array(ident, np.float32), np.array(misc, np.float32)]).astype(np.float32)


N_GHOST_FEATURES = (2 * R + 1) ** 2 + 5 + 6 + 4 + 4


class NetGhostStrategy:
    label = "learned ghost (shared network, ES)"

    def __init__(self, mlp):
        self.mlp = mlp

    def reset(self, seed):
        pass

    def choose(self, view, prefix, cell, mode):
        if mode == "frightened":
            return _away(view, cell, view.duck_cell["P_"])
        logits = self.mlp.forward(ghost_features(view, prefix, cell))
        opts = view.maze.neighbors(cell, ghost=True)
        best, bestv = None, -np.inf
        for i, (dc, dr) in enumerate(ACTIONS):
            c = (cell[0] + dc, cell[1] + dr)
            cand = None if (dc, dr) == (0, 0) else (c if c in opts else "skip")
            if cand == "skip":
                continue
            if logits[i] > bestv:
                best, bestv = cand, logits[i]
        return best

    def save(self, path):
        np.savez(path, flat=self.mlp.get_flat(), n_in=N_GHOST_FEATURES)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = MLP(int(z["n_in"]))
        m.set_flat(z["flat"])
        return cls(m)


def learned_ghosts(path, recovery="none"):
    from .policies import GhostPolicy
    st = NetGhostStrategy.load(path) if isinstance(path, str) else path
    return [GhostPolicy(k, st, recovery) for k in range(4)]
