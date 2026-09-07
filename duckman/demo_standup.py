"""Stand-up policy demo (labelled, not a scored round): spawn the Duck-Man face-down in the maze and let the
trained stand-up policy alone bring it back to its feet. python -m duckman.demo_standup --out results/standup_demo.mp4
"""
import argparse
import math
import numpy as np
import mujoco
from .maze import Maze
from .game import Game
from .policies import DuckManPolicy, NeutralStrategy
from .ghosts import default_ghosts
from .constants import POLICY_DIR
from .gait import Gait
from .render import Recorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/standup_demo.mp4")
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--pose", default="face_down", choices=["face_down", "face_up", "sit"])
    a = ap.parse_args()
    g = Game(Maze(), 0, DuckManPolicy(NeutralStrategy(), "standup"), default_ghosts("standup"))
    g.reset()
    sim, p = g.sim, g.sim.ducks["P_"]
    su = Gait(POLICY_DIR / "standup.onnx")
    # spawn: Pollen's face-down/face-up quaternion at low height, then settle 1 s with the actuators holding the default pose
    r2 = 2 ** -0.5
    q = {"face_down": [r2, 0, r2, 0], "face_up": [r2, 0, -r2, 0], "sit": [1, 0, 0, 0]}[a.pose]
    x, y = g.maze.xy(g.maze.pac_start)
    p.place([x, y, 0.07], q)                      # reset path (allow-listed in tests/test_no_sim_writes.py)
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(50):
        sim.step_physics()
    try:
        rec = Recorder(g, f"Stand-up demo: {a.pose.replace('_', '-')} spawn, trained policy", speed=1)
    except Exception as e:
        print(f"[warn] video disabled: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return
    rec.cam.distance, rec.cam.elevation, rec.cam.azimuth = 1.1, -28, 135
    rec.cam.lookat[:] = [x, y, 0.06]
    rec.chase.distance = 0.6
    last = np.zeros(14, np.float32)
    calls = 0
    for _ in range(int(1.5 / 0.02)):          # show the starting pose for 1.5 s before the policy is switched on
        sim.step_physics()
        g.view = g._view([])
        rec.maybe_capture()
    for k in range(int(a.seconds / 0.02)):
        cmd = np.zeros(13, np.float32)
        act = su.run(np.concatenate([p.proprio(last), cmd]))
        last = act
        calls += 1
        p.set_targets(su.targets(act))
        for gp in ["G0_", "G1_", "G2_", "G3_"]:                      # ghosts just hold their stand
            gh = sim.ducks[gp]
            gh.set_targets(g.policies[gp].gaits["stand"].targets(np.zeros(14, np.float32)))
        sim.step_physics()
        g.view = g._view([])
        rec.maybe_capture()
    rec.save(a.out)
    print(f"stand-up demo: pose {a.pose}, policy calls {calls}, final upright {p.upright():.2f}, height {p.pos()[2]:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
