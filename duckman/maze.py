"""ASCII maze -> grid graph. Cell = (col, row); row 0 is the TOP line of the map; y grows upward."""
from collections import deque
from .constants import CELL

MAP = [
    "o.....o",
    ".#.#.#.",
    "...D...",
    ".#HHH#.",
    ".......",
    ".#.#.#.",
    "...P...",
]


class Maze:
    def __init__(self, rows=MAP):
        self.rows, self.cols = len(rows), len(rows[0])
        self.corridor, self.house = set(), set()
        self.wall_cells, self.coins, self.pellets = [], [], []
        self.pac_start = self.door = None
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                cell = (c, r)
                if ch == "#":
                    self.wall_cells.append(cell)
                    continue
                if ch == "H":
                    self.house.add(cell)
                    continue
                self.corridor.add(cell)
                if ch == ".":
                    self.coins.append(cell)
                elif ch == "o":
                    self.pellets.append(cell)
                elif ch == "P":
                    self.pac_start = cell
                elif ch == "D":
                    self.door = cell
        hs = sorted(self.house)
        self.ghost_starts = [self.door, hs[0], hs[1], hs[2]]
        self.ghost_corners = [(self.cols - 1, 0), (0, 0), (self.cols - 1, self.rows - 1), (0, self.rows - 1)]
        self._bfs_cache = {}

    def xy(self, cell):
        return (cell[0] * CELL, (self.rows - 1 - cell[1]) * CELL)

    def cell(self, x, y):
        return (int(round(x / CELL)), int(round(self.rows - 1 - y / CELL)))

    def walkable(self, cell, ghost):
        return cell in self.corridor or (ghost and cell in self.house)

    def neighbors(self, cell, ghost):
        c, r = cell
        out = []
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c + d[0], r + d[1])
            if self.walkable(n, ghost):
                out.append(n)
        return out

    def _bfs(self, src, ghost):
        key = (src, ghost)
        if key in self._bfs_cache:
            return self._bfs_cache[key]
        prev = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.neighbors(u, ghost):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        self._bfs_cache[key] = prev
        return prev

    def dist(self, a, b, ghost):
        prev = self._bfs(a, ghost)
        if b not in prev:
            return None
        d, x = 0, b
        while prev[x] is not None:
            x = prev[x]
            d += 1
        return d

    def bfs_next(self, src, dst, ghost):
        if src == dst:
            return None
        prev = self._bfs(src, ghost)
        if dst not in prev:
            return None
        x = dst
        while prev[x] != src:
            x = prev[x]
        return x
