"""Plug your own Duck-Man strategy into the arena in ~20 lines.

    PYTHONPATH=. python examples_api/my_strategy.py --seed 0

A strategy is any object with reset(seed) and choose(view, cell, options) -> next cell or None (stay).
`view` is a duckman.game.GameView (positions, cells, coin/pellet states, ghost modes, power timer, lives,
clock); `options` are the walkable neighbour cells. The scripted navigator and Pollen's learned gait do
the rest, so your code never touches joints or physics, which is also what the causality tests assert.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from duckman.maze import Maze                      # noqa: E402
from duckman.game import Game                      # noqa: E402
from duckman.policies import DuckManPolicy         # noqa: E402
from duckman.ghosts import default_ghosts          # noqa: E402
from duckman.eval import resolve_recovery          # noqa: E402


class GreedyCoinStrategy:
    """Walk to the nearest live coin; if a ghost is within two cells, walk to the nearest pellet instead."""
    label = "example: greedy coins"

    def reset(self, seed):
        pass

    def choose(self, view, cell, options):
        m = view.maze
        ghosts = [view.duck_cell[g] for g, mode in view.ghost_mode.items() if mode == "chase"]
        threatened = any(abs(g[0] - cell[0]) + abs(g[1] - cell[1]) <= 2 for g in ghosts)
        pellets = [c for c, alive in view.pellets_alive.items() if alive]
        coins = [c for c, alive in view.coins_alive.items() if alive and c != cell]
        targets = pellets if (threatened and pellets) else coins
        if not targets:
            return None
        target = min(targets, key=lambda c: m.dist(cell, c, False) or 999)
        return m.bfs_next(cell, target, False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rec = resolve_recovery("auto")
    game = Game(Maze(), a.seed, DuckManPolicy(GreedyCoinStrategy(), rec), default_ghosts("standup" if rec == "standup" else "none"))
    game.reset()
    r = game.run()
    print({k: r[k] for k in ("score", "coins", "pellets", "ghosts", "lives_lost", "end", "t")})
