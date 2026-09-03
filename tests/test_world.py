import numpy as np
from duckman.maze import Maze
from duckman.world import Sim


def test_world_builds_five_ducks_and_tokens():
    s = Sim(Maze(), seed=0)
    assert set(s.ducks) == {"P_", "G0_", "G1_", "G2_", "G3_"}
    assert len(s.info["coins"]) == len(s.maze.coins) and len(s.info["pellets"]) == 2
    assert all(s.token_upright(n) for n in s.info["coins"])


def test_ducks_start_at_cells_and_obs_is_48():
    s = Sim(Maze(), seed=0)
    p = s.ducks["P_"]
    assert np.allclose(p.pos()[:2], s.maze.xy(s.maze.pac_start), atol=0.03)
    assert p.proprio(np.zeros(14, np.float32)).shape == (48,)


def test_reset_is_deterministic():
    s = Sim(Maze(), seed=3)
    for _ in range(20):
        s.step_physics()
    a = s.root_positions()
    s.reset(3)
    for _ in range(20):
        s.step_physics()
    b = s.root_positions()
    assert all(np.allclose(a[k], b[k]) for k in a)
