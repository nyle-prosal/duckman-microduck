"""One Microduck inside a multi-duck model: joint/sensor indices, proprioceptive obs, target writes."""
import math
import numpy as np
import mujoco
from .constants import JOINT_ORDER, DEFAULT_POSE, DUCK_Z


def quat_yaw(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


class Duck:
    def __init__(self, model, data, bam, prefix):
        self.m, self.d, self.bam, self.prefix = model, data, bam, prefix
        names = [prefix + j for j in JOINT_ORDER]
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        assert min(jids) >= 0, names
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.qt = np.array([bam.dof_to_q_target[n] for n in names])
        free = [j for j in range(model.njnt)
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(prefix)][0]
        self.rq, self.rv = model.jnt_qposadr[free], model.jnt_dofadr[free]
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "trunk_base")
        self.gyro = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + "imu_ang_vel")]

    def set_pose(self, xy, yaw):
        self.d.qpos[self.rq:self.rq + 3] = [xy[0], xy[1], DUCK_Z]
        self.d.qpos[self.rq + 3:self.rq + 7] = quat_yaw(yaw)
        self.d.qpos[self.qadr] = DEFAULT_POSE
        self.d.qvel[self.rv:self.rv + 6] = 0
        self.d.qvel[self.vadr] = 0
        self.bam.q_target[self.qt] = DEFAULT_POSE

    def pos(self):
        return self.d.xpos[self.trunk].copy()

    def quat(self):
        return self.d.xquat[self.trunk].copy()

    def yaw(self):
        w, x, y, z = self.quat()
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def proj_grav(self):
        q = self.quat().astype(np.float32)
        w, xyz = q[0], q[1:4]
        v = np.array([0, 0, -1], np.float32)
        t = np.cross(xyz, v) * 2
        return v - w * t + np.cross(xyz, t)

    def upright(self):
        return float(-self.proj_grav()[2])

    def fallen(self):
        return self.upright() < 0.5

    def proprio(self, last_action):
        g = self.d.sensordata[self.gyro:self.gyro + 3].astype(np.float32)
        return np.concatenate([g, self.proj_grav(),
                               self.d.qpos[self.qadr].astype(np.float32) - DEFAULT_POSE,
                               self.d.qvel[self.vadr].astype(np.float32),
                               np.asarray(last_action, np.float32)])

    def set_targets(self, targets):
        self.bam.q_target[self.qt] = targets
