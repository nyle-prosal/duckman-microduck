"""Shared constants. Robot/actuator values mirror pollen-robotics/microduck_rl (Apache-2.0)."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROBOT_XML = ROOT / "assets/microduck/robot_groundcontact.xml"
POLICY_DIR = ROOT / "assets/policies"

CTRL_DT, PHYS_DT, DECIMATION = 0.02, 0.005, 4
JOINT_ORDER = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
# STAND2 / HOME_FRAME from microduck_rl (scripts/infer_policy.py DEFAULT_POSE)
DEFAULT_POSE = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                         0.0, 0.0873, 0.4579, 0.0049, -0.4530], dtype=np.float32)
OBS_DIM, ACT_DIM = 61, 14

# BAM M6 defaults — mirror _BAM_ACTUATOR_KWARGS in microduck_rl/robot/microduck_constants.py
BAM = dict(motor_name="xl330", model="m6", kp_fw=200.0, vin=7.4, vin_drop_gain=0.1, vin_min=6.0,
           max_current=None, solref_friction=(-5.0e4, -2.0e2),
           solimp_friction=(0.99, 0.9999, 0.001, 0.5, 2.0))

# game
CELL = 0.45
WALL_H = 0.12
DUCK_Z = 0.12
COIN_R, COIN_H, COIN_MASS = 0.035, 0.03, 0.004
PELLET_R, PELLET_H, PELLET_MASS = 0.05, 0.05, 0.008
CLOCK_S, LIVES, POWER_S = 240.0, 3, 10.0
SCORE = dict(coin=10, pellet=50, ghost=200, clear=500)
SPEED = dict(pacman=0.40, ghost=0.30, frightened=0.30, eaten=0.35)
COLORS = {"P_": (1.0, 0.85, 0.1, 1), "G0_": (0.95, 0.15, 0.1, 1), "G1_": (1.0, 0.55, 0.8, 1),
          "G2_": (0.2, 0.9, 0.95, 1), "G3_": (1.0, 0.6, 0.15, 1)}
