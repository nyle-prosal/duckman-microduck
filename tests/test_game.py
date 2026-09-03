import numpy as np
from duckman.maze import Maze
from duckman.game import Game
from duckman.policies import DuckManPolicy, GhostPolicy, StaticGhost, ScriptedWalk


def ghosts():
    return [GhostPolicy(k, StaticGhost()) for k in range(4)]


def test_coin_scores_when_toppled_and_no_teleport():
    m = Maze()
    path = [(m.pac_start[0] - 1, m.pac_start[1]), (m.pac_start[0] - 2, m.pac_start[1])]
    g = Game(m, 0, DuckManPolicy(ScriptedWalk(path)), ghosts(), log_positions=True)
    g.reset()
    r = g.run(max_t=30.0)
    assert r["coins"] >= 1 and r["score"] == 10 * r["coins"]
    prev = None
    for snap in g.positions_log:
        if prev:
            assert max(np.linalg.norm(snap[k] - prev[k]) for k in snap) < 0.05
        prev = snap


def test_determinism():
    def run():
        m = Maze()
        g = Game(m, 5, DuckManPolicy(ScriptedWalk([(m.pac_start[0] + 1, m.pac_start[1])])), ghosts())
        g.reset()
        g.run(max_t=15.0)
        return g.events, g.sim.data.qpos.copy()
    e1, q1 = run()
    e2, q2 = run()
    assert e1 == e2 and np.array_equal(q1, q2)
