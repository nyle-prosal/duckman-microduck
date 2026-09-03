"""Builds the Duck-Man arena as an MjSpec and wraps the compiled model."""
import numpy as np
import mujoco
from .constants import (ROBOT_XML, CELL, WALL_H, DUCK_Z, COIN_R, COIN_H, COIN_MASS, PELLET_R, PELLET_H,
                        PELLET_MASS, COLORS, DECIMATION, CTRL_DT, DEFAULT_POSE)
from .bam_loader import compile_with_bam
from .duck import Duck, quat_yaw
from .maze import Maze

SHELLS = ["top_head_shell_material", "bottom_head_shell_material", "left_shell_material", "right_shell_material"]


def _base_spec():
    s = mujoco.MjSpec()
    s.modelname = "duckman"
    s.compiler.autolimits = True
    s.meshdir = str(ROBOT_XML.parent / "assets")
    s.visual.global_.offwidth, s.visual.global_.offheight = 1920, 1080
    s.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    s.visual.headlight.ambient = [0.35, 0.35, 0.35]
    tex = s.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                        mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.08, 0.08, 0.16], rgb2=[0.06, 0.06, 0.12],
                        markrgb=[0.2, 0.2, 0.35], width=300, height=300)
    mat = s.add_material(name="floor", texuniform=True, texrepeat=[6, 6], reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
    s.worldbody.add_light(pos=[1.35, 1.35, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor")
    return s


def _add_duck(s, prefix, xy, yaw):
    child = mujoco.MjSpec.from_file(str(ROBOT_XML))
    child.meshdir = str(ROBOT_XML.parent / "assets")
    for m in child.materials:
        if m.name in SHELLS:
            m.rgba = list(COLORS[prefix])
    f = s.worldbody.add_frame(pos=[xy[0], xy[1], DUCK_Z], quat=quat_yaw(yaw))
    s.attach(child, prefix=prefix, frame=f)


def _add_token(s, name, xy, r, h, mass, rgba):
    b = s.worldbody.add_body(name=name, pos=[xy[0], xy[1], h])
    b.add_freejoint(name=name + "_free")
    b.mass = mass
    b.ipos = [0, 0, 0]
    ixx = mass * (3 * r * r + 4 * h * h) / 12
    b.inertia = [ixx, ixx, mass * r * r / 2]
    b.explicitinertial = True
    b.add_geom(name=name + "_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h, 0], rgba=list(rgba),
               friction=[0.6, 0.005, 0.0001])


def build_spec(maze: Maze, seed: int):
    rng = np.random.default_rng(seed)
    s = _base_spec()
    wall = [0.15, 0.2, 0.9, 1]
    for c in maze.wall_cells:
        x, y = maze.xy(c)
        s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, WALL_H / 2],
                             size=[CELL / 2, CELL / 2, WALL_H / 2], rgba=wall)
    W, Hh = maze.cols * CELL / 2, maze.rows * CELL / 2
    cx, cy = (maze.cols - 1) * CELL / 2, (maze.rows - 1) * CELL / 2
    t = 0.03
    for px, py, sx, sy in [(cx - W - t, cy, t, Hh + 2 * t), (cx + W + t, cy, t, Hh + 2 * t),
                           (cx, cy - Hh - t, W + 2 * t, t), (cx, cy + Hh + t, W + 2 * t, t)]:
        s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[px, py, WALL_H / 2], size=[sx, sy, WALL_H / 2], rgba=wall)
    for h in maze.house:
        x, y = maze.xy(h)
        s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, 0.001], size=[CELL / 2, CELL / 2, 0.001],
                             rgba=[0.3, 0.3, 0.4, 1], contype=0, conaffinity=0)
    info = {"coins": [], "pellets": [], "ducks": [], "starts": {}}
    for c in maze.coins:
        x, y = maze.xy(c)
        j = rng.uniform(-0.01, 0.01, 2)
        n = f"coin_{c[0]}_{c[1]}"
        _add_token(s, n, (x + j[0], y + j[1]), COIN_R, COIN_H, COIN_MASS, (1, 0.85, 0.1, 1))
        info["coins"].append(n)
    for c in maze.pellets:
        x, y = maze.xy(c)
        n = f"pellet_{c[0]}_{c[1]}"
        _add_token(s, n, (x, y), PELLET_R, PELLET_H, PELLET_MASS, (0.8, 0.3, 1, 1))
        info["pellets"].append(n)
    starts = [("P_", maze.pac_start, np.pi / 2)] + [(f"G{k}_", maze.ghost_starts[k], -np.pi / 2) for k in range(4)]
    for prefix, cell, yaw in starts:
        x, y = maze.xy(cell)
        j = rng.uniform(-0.01, 0.01, 2)
        yw = yaw + rng.uniform(-0.05, 0.05)
        _add_duck(s, prefix, (x + j[0], y + j[1]), yw)
        info["ducks"].append(prefix)
        info["starts"][prefix] = ((x + j[0], y + j[1]), yw)
    return s, info


class Sim:
    def __init__(self, maze: Maze, seed: int):
        self.maze, self.seed = maze, seed
        spec, self.info = build_spec(maze, seed)
        self.model, self.data, self.bams = compile_with_bam(spec, {p: p for p in self.info["ducks"]})
        self.ducks = {p: Duck(self.model, self.data, self.bams[p], p) for p in self.info["ducks"]}
        self._tok = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                     for n in self.info["coins"] + self.info["pellets"]}
        self._collision_groups()
        self._owner = np.full(self.model.ngeom, -1, dtype=np.int64)
        self._code = {}
        for i, p in enumerate(self.info["ducks"]):
            self._code[p] = i
        for i, n in enumerate(self._tok):
            self._code[n] = 100 + i
        for g in range(self.model.ngeom):
            b = self.model.body(self.model.geom_bodyid[g]).name
            for p, c in self._code.items():
                if b == p or (p.endswith("_") and b.startswith(p)):
                    self._owner[g] = c
                    break
        self._tok_ids = np.array(list(self._tok.values()))
        self._tok_names = list(self._tok)
        self._con = set()
        self.touched = {n: False for n in self._tok}
        self.t = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._tok0 = {n: self.data.xpos[b][:2].copy() for n, b in self._tok.items()}
        self.reset(seed)

    def _collision_groups(self):
        """Arcade semantics via MuJoCo collision bitmasks (a pair collides if contype_a & conaffinity_b or
        contype_b & conaffinity_a):
          Duck-Man collision geoms  contype 3 / conaffinity 3
          tokens                    contype 2 / conaffinity 2   -> only the Duck-Man (and floor/walls) touch them
          ghosts                    contype 8 / conaffinity 1   -> collide with Duck-Man, floor, walls; pass through
                                                                   other ghosts and tokens, as in the arcade
          floor and walls           contype 3 / conaffinity 3
        Physics inside each duck (joints, actuators, self-collision geoms) is untouched."""
        m = self.model
        for g in range(m.ngeom):
            body = m.body(m.geom_bodyid[g]).name
            plain = m.geom_contype[g] == 1 and m.geom_conaffinity[g] == 1
            if body.startswith("P_") and plain:
                m.geom_contype[g], m.geom_conaffinity[g] = 3, 3
            elif body.startswith("G") and plain:
                m.geom_contype[g], m.geom_conaffinity[g] = 8, 1
            elif body in self._tok:
                m.geom_contype[g], m.geom_conaffinity[g] = 2, 2
            elif body == "world" and plain:
                m.geom_contype[g], m.geom_conaffinity[g] = 3, 3

    def reset(self, seed=None):
        assert seed is None or seed == self.seed, "layout is baked at build time; make a new Sim for another seed"
        mujoco.mj_resetData(self.model, self.data)
        for p, d in self.ducks.items():
            d.set_pose(*self.info["starts"][p])
        for b in self.bams.values():
            b.q_target[:] = self.data.qpos[b.qpos_indexes]
            if hasattr(b, "last_ts"):
                b.last_ts = 0.0
            if hasattr(b, "_prev_motor_torque"):
                b._prev_motor_torque = np.zeros_like(b._prev_motor_torque)
        self.touched = {n: False for n in self._tok}
        self.t = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step_physics(self):
        for _ in range(DECIMATION):
            for b in self.bams.values():
                b.update()
            mujoco.mj_step(self.model, self.data)
        self.t += CTRL_DT
        n = self.data.ncon
        if n:
            a = self._owner[self.data.contact.geom1[:n]]
            b = self._owner[self.data.contact.geom2[:n]]
            self._con = set(zip(a.tolist(), b.tolist())) | set(zip(b.tolist(), a.tolist()))
            for (x, y) in self._con:
                if x == 0 and y >= 100:
                    self.touched[self._tok_names[y - 100]] = True
        else:
            self._con = set()

    def body_pos(self, name):
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

    def token_upright(self, name):
        return bool(self.data.xmat[self._tok[name]][8] > 0.7)

    def token_collected(self, name, displace=0.06):
        """A token counts once the Duck-Man has touched it AND it is toppled or >= `displace` m from its spawn."""
        if not self.touched[name]:
            return False
        b = self._tok[name]
        return (self.data.xmat[b][8] <= 0.7) or (np.linalg.norm(self.data.xpos[b][:2] - self._tok0[name]) >= displace)

    def _pairs(self):
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            yield m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name

    def contact(self, a, b):
        return (self._code[a], self._code[b]) in self._con

    def contact_token(self, prefix, token):
        return (self._code[prefix], self._code[token]) in self._con

    def root_positions(self):
        out = {p: d.pos() for p, d in self.ducks.items()}
        out.update({n: self.body_pos(n) for n in self._tok})
        return out
