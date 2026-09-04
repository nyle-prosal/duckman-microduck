"""Scripted ghost personalities (decisions on cell arrival, BFS on the grid)."""
import numpy as np


def _heading_cell(view, prefix="P_"):
    c = view.duck_cell[prefix]
    yaw = view.duck_yaw[prefix]
    d = (int(round(np.cos(yaw))), -int(round(np.sin(yaw))))   # row grows downward
    return (c[0] + 2 * d[0], c[1] + 2 * d[1])


def _nearest_walkable(view, target):
    return min(view.maze.corridor, key=lambda c: abs(c[0] - target[0]) + abs(c[1] - target[1]))


def _away(view, cell, threat):
    opts = view.maze.neighbors(cell, ghost=False) or view.maze.neighbors(cell, ghost=True)
    if not opts:
        return None
    return max(opts, key=lambda c: abs(c[0] - threat[0]) + abs(c[1] - threat[1]))


class _G:
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def choose(self, view, prefix, cell, mode):
        if mode == "frightened":
            return _away(view, cell, view.duck_cell["P_"])
        t = _nearest_walkable(view, self.target(view, prefix, cell))
        if t == cell:
            opts = view.maze.neighbors(cell, ghost=True)
            return tuple(int(x) for x in self.rng.choice(opts)) if opts else None
        return view.maze.bfs_next(cell, t, ghost=True)


class Blinky(_G):
    label = "Blinky (chase)"

    def target(self, view, prefix, cell):
        return view.duck_cell["P_"]


class Pinky(_G):
    label = "Pinky (ambush 2 ahead)"

    def target(self, view, prefix, cell):
        return _heading_cell(view)


class Inky(_G):
    label = "Inky (mirror of Blinky)"

    def target(self, view, prefix, cell):
        p, b = view.duck_cell["P_"], view.duck_cell["G0_"]
        return (2 * p[0] - b[0], 2 * p[1] - b[1])


class Clyde(_G):
    label = "Clyde (shy)"

    def target(self, view, prefix, cell):
        p = view.duck_cell["P_"]
        far = abs(p[0] - cell[0]) + abs(p[1] - cell[1]) > 3
        return p if far else view.maze.ghost_corners[3]


def default_ghosts(recovery="none"):
    from .policies import GhostPolicy
    return [GhostPolicy(0, Blinky(), recovery), GhostPolicy(1, Pinky(), recovery),
            GhostPolicy(2, Inky(), recovery), GhostPolicy(3, Clyde(), recovery)]
