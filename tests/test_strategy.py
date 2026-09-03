import numpy as np
from duckman.maze import Maze
from duckman.game import Game
from duckman.policies import DuckManPolicy
from duckman.ghosts import default_ghosts
from duckman.strategy_net import features, N_FEATURES, MLP, NetStrategy, random_net


def test_feature_shape_and_range():
    g = Game(Maze(), 0, DuckManPolicy(random_net(0)), default_ghosts())
    v = g.reset()
    f = features(v, v.duck_cell["P_"])
    assert f.shape == (N_FEATURES,) and np.abs(f).max() <= 1.0


def test_mlp_roundtrip_and_masking():
    m = MLP(N_FEATURES)
    m.set_flat(m.get_flat() * 0 + 0.1)
    assert np.allclose(m.get_flat(), 0.1)
    g = Game(Maze(), 0, DuckManPolicy(NetStrategy(m)), default_ghosts())
    v = g.reset()
    cell = v.duck_cell["P_"]
    opts = v.maze.neighbors(cell, False)
    c = NetStrategy(m).choose(v, cell, opts)
    assert c is None or c in opts


def test_save_load(tmp_path):
    n = random_net(1)
    p = tmp_path / "w.npz"
    n.save(p)
    m = NetStrategy.load(p)
    assert np.allclose(n.mlp.get_flat(), m.mlp.get_flat())
