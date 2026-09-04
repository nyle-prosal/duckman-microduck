"""Scripted cell navigator: turn in place toward the target cell, then walk. Emits a twist command.

Measured quirks of Pollen's gait that this encodes: a pure yaw command from standstill does not
start stepping, so a turn begins with a 0.4 s forward kick-start; sideways/backward commands do not
move the robot at all, so the duck always faces where it walks.
"""
import math
import numpy as np
from .constants import CTRL_DT


class Navigator:
    ARRIVE = 0.06
    FACE_TOL = 0.35
    KP = 2.5
    KICK_S = 0.4
    KICK_VX = 0.3

    def __init__(self, maze, speed):
        self.maze, self.speed, self.target = maze, speed, None
        self.turn_t = 0.0

    def set_target(self, cell):
        self.target = cell

    def stop(self):
        self.turn_t = 0.0
        return (0.0, 0.0, 0.0)

    def arrived(self, pos):
        if self.target is None:
            return True
        tx, ty = self.maze.xy(self.target)
        return math.hypot(tx - pos[0], ty - pos[1]) < self.ARRIVE

    def twist(self, pos, yaw):
        if self.target is None or self.arrived(pos):
            return self.stop()
        tx, ty = self.maze.xy(self.target)
        err = (math.atan2(ty - pos[1], tx - pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(err) < self.FACE_TOL:
            self.turn_t = 0.0
            return (self.speed, 0.0, float(np.clip(self.KP * err, -1.0, 1.0)))
        self.turn_t += CTRL_DT
        vx = self.KICK_VX if self.turn_t <= self.KICK_S else 0.0
        return (vx, 0.0, 1.0 if err > 0 else -1.0)
