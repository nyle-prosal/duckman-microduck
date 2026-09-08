"""Play Duck-Man yourself against the ghosts, in the same physics. Arrow keys / WASD in THIS terminal choose the
next corridor; the MuJoCo viewer window shows the game. Not part of the scored evaluation.
python -m duckman.play [--seed 0] [--ghosts scripted|learned]"""
import argparse
import select
import sys
import termios
import time
import tty
import mujoco
import mujoco.viewer
from .constants import ROOT, CTRL_DT
from .maze import Maze
from .game import Game
from .policies import DuckManPolicy
from .ghosts import default_ghosts
from .eval import resolve_recovery

KEYS = {"w": (0, -1), "s": (0, 1), "a": (-1, 0), "d": (1, 0), "\x1b[A": (0, -1), "\x1b[B": (0, 1), "\x1b[D": (-1, 0), "\x1b[C": (1, 0)}


class HumanStrategy:
    label = "human (keyboard)"

    def __init__(self):
        self.want = None

    def reset(self, seed):
        self.want = None

    def choose(self, view, cell, options):
        if self.want is None:
            return None
        c = (cell[0] + self.want[0], cell[1] + self.want[1])
        return c if c in options else None


def read_key(timeout=0.0):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        ch += sys.stdin.read(2)
    return ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ghosts", default="scripted", choices=["scripted", "learned"])
    a = ap.parse_args()
    rec = resolve_recovery("auto")
    if a.ghosts == "learned" and (ROOT / "checkpoints/ghosts_final.npz").exists():
        from .ghost_net import learned_ghosts
        ghosts = learned_ghosts(str(ROOT / "checkpoints/ghosts_final.npz"), "standup" if rec == "standup" else "none")
    else:
        ghosts = default_ghosts("standup" if rec == "standup" else "none")
    human = HumanStrategy()
    g = Game(Maze(), a.seed, DuckManPolicy(human, rec), ghosts)
    g.reset()
    print("Duck-Man: steer with WASD or arrow keys (in this terminal), q quits. The Duck-Man commits to one cell at a time.")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        with mujoco.viewer.launch_passive(g.sim.model, g.sim.data, show_left_ui=False, show_right_ui=False) as v:
            v.cam.lookat[:] = [1.35, 1.35, 0]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 4.2, 90, -62
            last = time.time()
            while v.is_running() and not g.done:
                k = read_key(0.0)
                if k == "q":
                    break
                if k in KEYS:
                    human.want = KEYS[k]
                g.step()
                v.sync()
                dt = CTRL_DT - (time.time() - last)
                if dt > 0:
                    time.sleep(dt)
                last = time.time()
                if int(g.sim.t * 50) % 50 == 0:
                    vw = g.view
                    print(f"\rscore {vw.score:4d}  lives {vw.lives}  coins left {sum(vw.coins_alive.values()):2d}  clock {vw.clock_left:5.1f}s  power {vw.power_left:4.1f}   ", end="", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    r = g.result()
    print(f"\nfinal: score {r['score']}  coins {r['coins']}  ghosts caught {r['ghosts']}  lives lost {r['lives_lost']}  end {r['end']}")


if __name__ == "__main__":
    main()
