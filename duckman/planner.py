"""Scripted Duck-Man baseline: BFS to the nearest coin with a ghost-danger cost."""


class PlannerStrategy:
    label = "scripted planner"
    DANGER = 2

    def reset(self, seed):
        pass

    def choose(self, view, cell, options):
        m = view.maze
        ghosts = [view.duck_cell[g] for g, mode in view.ghost_mode.items() if mode == "chase"]
        fr = [view.duck_cell[g] for g, mode in view.ghost_mode.items() if mode == "frightened"]

        def danger(c):
            return any(abs(c[0] - g[0]) + abs(c[1] - g[1]) <= self.DANGER for g in ghosts)

        if view.power_left > 0 and fr:
            t = min(fr, key=lambda g: m.dist(cell, g, False) or 99)
            return m.bfs_next(cell, t, False)
        near = any(abs(cell[0] - g[0]) + abs(cell[1] - g[1]) <= 3 for g in ghosts)
        pellets = [c for c, a in view.pellets_alive.items() if a]
        coins = [c for c, a in view.coins_alive.items() if a]
        targets = pellets if (near and pellets) else coins + pellets
        if not targets:
            return None
        best, bestd = None, None
        for t in targets:
            d = m.dist(cell, t, False)
            if d is None:
                continue
            d += 6 * sum(1 for g in ghosts if abs(t[0] - g[0]) + abs(t[1] - g[1]) <= self.DANGER)
            if bestd is None or d < bestd:
                best, bestd = t, d
        nxt = m.bfs_next(cell, best, False)
        if nxt is not None and danger(nxt):
            safe = [o for o in options if not danger(o)]
            if safe:
                nxt = min(safe, key=lambda o: m.dist(o, best, False) or 99)
        return nxt
