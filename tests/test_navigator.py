import numpy as np
from duckman.maze import Maze
from duckman.world import Sim
from duckman.navigator import Navigator
from duckman.gait import Gait
from duckman.constants import POLICY_DIR


def test_walks_to_adjacent_cell():
    m = Maze()
    s = Sim(m, 0)
    p = s.ducks["P_"]
    g = Gait(POLICY_DIR / "alpha_walking.onnx")
    nav = Navigator(m, 0.4)
    nav.set_target((m.pac_start[0] + 1, m.pac_start[1]))
    a = np.zeros(14, np.float32)
    for _ in range(int(12 / 0.02)):
        cmd = np.zeros(13, np.float32)
        cmd[:3] = nav.twist(p.pos()[:2], p.yaw())
        a = g.run(np.concatenate([p.proprio(a), cmd]))
        p.set_targets(g.targets(a))
        s.step_physics()
        if nav.arrived(p.pos()[:2]):
            break
    assert nav.arrived(p.pos()[:2]) and p.upright() > 0.9 and s.t < 12
