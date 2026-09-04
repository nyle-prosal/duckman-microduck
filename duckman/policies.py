"""Policy objects. Every duck is driven by one of these: reset(seed) / act(obs) -> 14 joint targets.

act() is called at every 50 Hz control step and its return value is the only thing written to the
actuators. Layering: strategy (learned or scripted) -> scripted navigator -> Pollen's learned gait.
"""
import numpy as np
from .constants import POLICY_DIR, SPEED, CTRL_DT, DEFAULT_POSE
from .gait import Gait
from .navigator import Navigator


class NeutralStrategy:
    """Disabled Duck-Man: never leaves its cell (used by the causality test)."""
    label = "neutral (always stay)"

    def reset(self, seed):
        pass

    def choose(self, view, cell, options):
        return None


class ScriptedWalk:
    """Test helper: visit a fixed list of cells, then stay."""
    label = "scripted test path"

    def __init__(self, cells):
        self.cells = list(cells)
        self.i = 0

    def reset(self, seed):
        self.i = 0

    def choose(self, view, cell, options):
        if self.i < len(self.cells):
            c = self.cells[self.i]
            self.i += 1
            return c
        return None


class _Base:
    def __init__(self):
        self.gaits = {"walk": Gait(POLICY_DIR / "alpha_walking.onnx"),
                      "stand": Gait(POLICY_DIR / "alpha_stand.onnx")}
        self.calls = 0
        self.last = np.zeros(14, np.float32)
        self.active = "stand"

    def gait_calls(self):
        return {k: g.calls for k, g in self.gaits.items()}

    def action_bounds(self):
        used = [g for g in self.gaits.values() if g.calls]
        if not used:
            return {"min": None, "max": None}
        return {"min": np.min([g.action_min for g in used], axis=0).tolist(),
                "max": np.max([g.action_max for g in used], axis=0).tolist()}

    def _run(self, obs, cmd13, which):
        self.active = which
        g = self.gaits[which]
        a = g.run(np.concatenate([obs["proprio"], np.asarray(cmd13, np.float32)]))
        self.last = a
        return g.targets(a)

    @staticmethod
    def _twist_cmd(twist):
        cmd = np.zeros(13, np.float32)
        cmd[:3] = twist
        return cmd, ("walk" if abs(twist[0]) + abs(twist[2]) > 0.05 else "stand")


class DuckManPolicy(_Base):
    REPLAN_S = 8.0   # a target not reached within this long (blocked by a ghost) is re-decided

    def __init__(self, strategy, recovery="sitstand"):
        super().__init__()
        self.strategy, self.recovery = strategy, recovery
        self.gaits["sitstand"] = Gait(POLICY_DIR / "alpha_sitstand.onnx")
        if recovery == "standup":
            self.gaits["standup"] = Gait(POLICY_DIR / "standup.onnx")
        elif recovery != "sitstand":
            raise ValueError(recovery)
        self.up_t = 0.0
        self.label = f"Duck-Man[{strategy.label}]"
        self.nav = None
        self.mode, self.mode_t = "play", 0.0

    def reset(self, seed):
        self.strategy.reset(seed)
        self.nav = Navigator(None, SPEED["pacman"])
        self.mode, self.mode_t = "play", 0.0
        self.calls = 0
        self.last = np.zeros(14, np.float32)
        self.strategy_calls = 0
        self.target_t = 0.0

    def ready(self):
        return self.mode == "play"

    def act(self, obs):
        self.calls += 1
        view = obs["view"]
        self.nav.maze = view.maze
        pos, yaw = view.duck_pos["P_"], view.duck_yaw["P_"]
        if view.tagged and self.mode == "play":
            self.mode, self.mode_t = ("stunned" if self.recovery == "sitstand" else "down"), 0.0
            self.nav.target = None
        if self.recovery == "standup" and self.mode == "play" and view.upright["P_"] < 0.5:
            # knocked over during play (no tag): get up with the stand-up policy; ghosts keep chasing
            self.mode, self.mode_t = "recover", 0.0
            self.nav.target = None
        if self.mode == "stunned":
            # tagged: hold a balanced stand while the ghosts turn away (a sitting duck is easy to topple),
            # then sit down for the rest of the reset.
            self.mode_t += CTRL_DT
            t = self._run(obs, np.zeros(13, np.float32), "stand")
            pc = view.duck_cell["P_"]
            clear = all(abs(view.duck_cell[g][0] - pc[0]) + abs(view.duck_cell[g][1] - pc[1]) >= 2 for g in view.ghost_mode)
            if clear or self.mode_t > 8.0 or view.phase == "reset_done":
                self.mode, self.mode_t = "down", 0.0
            return t
        if self.mode == "down":
            self.mode_t += CTRL_DT
            if self.recovery == "sitstand":
                cmd = np.zeros(13, np.float32)
                cmd[0] = 1.0                      # posture flag: 1 = sit
                t = self._run(obs, cmd, "sitstand")
                if view.phase == "reset_done" or self.mode_t > 30.0:
                    self.mode, self.mode_t = "recover", 0.0
                return t
            # stand-up recovery: the tagged duck drops into the sit posture ("knocked down"); if the shove
            # actually toppled it, the stand-up policy takes over right away (it is trained from
            # face-down / face-up / sitting starts). Play resumes only after the ghosts are home.
            if view.upright["P_"] > 0.5:
                cmd = np.zeros(13, np.float32)
                cmd[0] = 1.0
                t = self._run(obs, cmd, "sitstand")
            else:
                t = self._run(obs, np.zeros(13, np.float32), "standup")
            if view.phase == "reset_done" or self.mode_t > 30.0:
                self.mode, self.mode_t = "recover", 0.0
            return t
        if self.mode == "recover":
            self.mode_t += CTRL_DT
            if self.recovery == "standup":
                t = self._run(obs, np.zeros(13, np.float32), "standup")
                self.up_t = self.up_t + CTRL_DT if view.upright["P_"] > 0.9 else 0.0
                if self.up_t >= 1.0 or self.mode_t > 10.0:
                    self.mode = "play"
                    self.nav.target = None
                    self.up_t = 0.0
                return t
            t = self._run(obs, np.zeros(13, np.float32), "sitstand")   # flag 0 = stand
            if (self.mode_t > 2.0 and view.upright["P_"] > 0.75) or self.mode_t > 6.0:
                self.mode = "play"
                self.nav.target = None
            return t
        cell = view.duck_cell["P_"]
        self.target_t += CTRL_DT
        if self.nav.target is None or self.nav.arrived(pos) or self.target_t > self.REPLAN_S:
            nxt = self.strategy.choose(view, cell, view.maze.neighbors(cell, ghost=False))
            self.strategy_calls += 1
            self.nav.set_target(nxt)
            self.target_t = 0.0
        cmd, which = self._twist_cmd(self.nav.twist(pos, yaw))
        return self._run(obs, cmd, which)


class StaticGhost:
    label = "static"

    def reset(self, seed):
        pass

    def choose(self, view, prefix, cell, mode):
        return None


class GhostPolicy(_Base):
    def __init__(self, index, personality):
        super().__init__()
        self.k, self.p = index, personality
        self.prefix = f"G{index}_"
        self.label = f"ghost{index}[{personality.label}]"
        self.nav = None

    REPLAN_S = 6.0

    def reset(self, seed):
        self.p.reset(seed + self.k)
        self.nav = Navigator(None, SPEED["ghost"])
        self.calls = 0
        self.last = np.zeros(14, np.float32)
        self.last_mode = "chase"
        self.target_t = 0.0

    def act(self, obs):
        self.calls += 1
        view = obs["view"]
        self.nav.maze = view.maze
        pos, yaw = view.duck_pos[self.prefix], view.duck_yaw[self.prefix]
        mode = view.ghost_mode[self.prefix]
        cell = view.duck_cell[self.prefix]
        self.nav.speed = (SPEED["eaten"] if mode in ("eaten", "home") else
                          SPEED["frightened"] if mode == "frightened" else SPEED["ghost"])
        self.target_t += CTRL_DT
        if self.nav.target is None or self.nav.arrived(pos) or mode != self.last_mode or self.target_t > self.REPLAN_S:
            if mode in ("eaten", "home"):
                nxt = view.maze.bfs_next(cell, view.ghost_home[self.prefix], ghost=True, blocked=(view.duck_cell["P_"],))
            else:
                nxt = self.p.choose(view, self.prefix, cell, mode)
            self.nav.set_target(nxt)
            self.last_mode = mode
            self.target_t = 0.0
        cmd, which = self._twist_cmd(self.nav.twist(pos, yaw))
        return self._run(obs, cmd, which)
