"""BAM actuator conversion for a MuJoCo MjSpec.

Adapted from pollen-robotics/microduck_rl scripts/infer_policy.py (Apache-2.0):
mirrors bam.mjlab.BamActuator.edit_spec, which is what the policies were trained against.
Position actuators -> torque motors with the voltage-bounded force range; joint damping and
frictionloss zeroed (BAM rewrites them every step); stiff friction constraint.
"""
import numpy as np
import mujoco
from .constants import BAM, PHYS_DT


def load_bam_model():
    from bam.model import load_model
    m = load_model(motor_name=BAM["motor_name"], model=BAM["model"])
    m.actuator.kp = BAM["kp_fw"]
    m.actuator.vin = BAM["vin"]
    m.actuator.max_current = BAM["max_current"]
    return m


def compile_with_bam(spec: mujoco.MjSpec, groups=None):
    """Compile the spec with BAM actuators. `groups` maps a name -> actuator-name prefix; one
    MujocoController is built per group so the load-dependent voltage sag is computed per robot
    (as in training, one robot per env). Returns (model, data, {group: controller})."""
    from bam.mujoco import MujocoController
    bm = load_bam_model()
    kt, R = bm.kt.value, bm.R.value
    flim = bm.actuator.vin * kt / R
    names = []
    for act in spec.actuators:
        tgt = act.target
        tname = tgt.name if hasattr(tgt, "name") else str(tgt)
        if "passive_" in tname:
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-flim, flim)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for j in spec.joints:
            if j.name == tname:
                j.damping = np.zeros((3, 1))
                j.frictionloss = 0.0
                j.solref_friction = BAM["solref_friction"]
                j.solimp_friction = BAM["solimp_friction"]
                break
    model = spec.compile()
    model.opt.timestep = PHYS_DT
    data = mujoco.MjData(model)
    groups = groups or {"all": ""}
    ctrls = {}
    for gname, prefix in groups.items():
        sub = [n for n in names if n.startswith(prefix)]
        ctrls[gname] = MujocoController(load_bam_model(), sub, model, data,
                                        vin_drop_gain=BAM["vin_drop_gain"], vin_min=BAM["vin_min"])
    return model, data, ctrls
