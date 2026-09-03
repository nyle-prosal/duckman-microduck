import numpy as np
from duckman.maze import Maze
from duckman.game import Game
from duckman.policies import DuckManPolicy, GhostPolicy, NeutralStrategy, StaticGhost


def make_game(seed=0):
    return Game(Maze(), seed, DuckManPolicy(NeutralStrategy()), [GhostPolicy(k, StaticGhost()) for k in range(4)])


def test_policy_interface_and_call_counts():
    g = make_game()
    g.reset()
    for _ in range(100):
        g.step()
    for p, pol in g.policies.items():
        assert pol.calls == 100
        assert sum(pol.gait_calls().values()) == 100   # exactly one gait network call per control step
    r = g.result()
    lo, hi = np.array(r["action_bounds"]["P_"]["min"]), np.array(r["action_bounds"]["P_"]["max"])
    assert np.isfinite(lo).all() and (hi - lo < 4.0).all()


def test_neutral_collects_nothing_in_20s():
    g = make_game()
    g.reset()
    r = g.run(max_t=20.0)
    assert r["coins"] == 0 and r["score"] == 0
