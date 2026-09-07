"""Broadcast-style renderer (render-only, no physics): overhead arena at left, side panel at right with
score, lives, power bar, event ticker, ghost legend and a chase-cam inset. Plus a Pillow learning-curve chart."""
import csv
import math
import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont
from .constants import CTRL_DT, CELL, COLORS

W, H = 1280, 720
MAIN_W = 940
PANEL_X = MAIN_W
INSET_W, INSET_H = 300, 170
GHOST_NAMES = {"G0_": ("Blinky", "chases you"), "G1_": ("Pinky", "ambushes ahead"),
               "G2_": ("Inky", "mirrors Blinky"), "G3_": ("Clyde", "shy, patrols")}
EVENT_TEXT = {"coin": None, "pellet": ("POWER PELLET  +50", (255, 120, 255)), "ghost_caught": ("GHOST CAUGHT  +200", (120, 255, 160)),
              "tag": ("TAGGED  -1 life", (255, 90, 90)), "resume": ("PLAY RESUMES", (200, 200, 220)),
              "cleared": ("MAZE CLEARED  +500", (255, 230, 120)), "end": None, "fell": ("KNOCKED DOWN", (255, 170, 90))}


def _font(size):
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _rgb(rgba):
    return tuple(int(255 * c) for c in rgba[:3])


class Recorder:
    def __init__(self, game, label, fps=25, size=(W, H), speed=1):
        self.g, self.label, self.fps, self.speed = game, label, fps, speed
        self.frames, self.k = [], 0
        self.every = max(1, int(round(1 / (fps * CTRL_DT)))) * speed
        self.main = mujoco.Renderer(game.sim.model, H, MAIN_W)
        self.inset = mujoco.Renderer(game.sim.model, INSET_H, INSET_W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        m = game.maze
        self.cam.lookat[:] = [(m.cols - 1) * CELL / 2, (m.rows - 1) * CELL / 2 - 0.05, 0]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 4.35, 90, -64
        self.chase = mujoco.MjvCamera()
        self.chase.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.chase.distance, self.chase.elevation = 0.9, -22
        self.f_big, self.f_mid, self.f_small, self.f_tiny = _font(34), _font(24), _font(19), _font(15)
        self.ticker = []          # (time, text, colour)
        self.duck_names = {"P_": "Duck-Man", **{k: v[0] for k, v in GHOST_NAMES.items()}}

    def maybe_capture(self):
        self.k += 1
        v = self.g.view
        for e in v.events_this_step:
            t = EVENT_TEXT.get(e)
            if t:
                self.ticker.append((v.t, t[0], t[1]))
        if self.k % self.every == 0:
            self.frames.append(self.frame())

    def frame(self):
        v = self.g.view
        self.main.update_scene(self.g.sim.data, self.cam)
        canvas = Image.new("RGB", (W, H), (12, 12, 20))
        canvas.paste(Image.fromarray(self.main.render()), (0, 0))
        px, py = v.duck_pos["P_"]
        yaw = v.duck_yaw["P_"]
        self.chase.lookat[:] = [px, py, 0.12]
        self.chase.azimuth = math.degrees(yaw) + 180
        self.inset.update_scene(self.g.sim.data, self.chase)
        inset = Image.fromarray(self.inset.render())
        d = ImageDraw.Draw(canvas, "RGBA")
        x0 = PANEL_X + 18
        d.rectangle([PANEL_X, 0, W, H], fill=(16, 16, 26, 255))
        d.text((x0, 16), "DUCK-MAN", fill=(255, 215, 80), font=self.f_big)
        d.text((x0, 56), "maze tag for Microduck", fill=(170, 170, 190), font=self.f_small)
        d.text((x0, 96), f"{v.score:5d}", fill=(255, 255, 255), font=_font(52))
        d.text((x0 + 150, 118), "score", fill=(150, 150, 170), font=self.f_small)
        d.text((x0, 160), f"clock {v.clock_left:5.1f} s", fill=(220, 220, 230), font=self.f_mid)
        d.text((x0, 190), "lives", fill=(255, 120, 120), font=self.f_mid)
        for i in range(3):
            cx = x0 + 78 + i * 26
            d.ellipse([cx, 196, cx + 16, 212], fill=(255, 90, 90) if i < v.lives else (60, 40, 50), outline=(255, 120, 120))
        d.text((x0, 220), f"coins left {sum(v.coins_alive.values()):2d}   pellets {sum(v.pellets_alive.values())}", fill=(220, 220, 230), font=self.f_small)
        # power bar
        d.text((x0, 250), "power", fill=(150, 150, 170), font=self.f_tiny)
        d.rectangle([x0, 268, W - 18, 280], outline=(90, 90, 120), fill=(30, 30, 45))
        if v.power_left > 0:
            d.rectangle([x0, 268, x0 + (W - 18 - x0) * v.power_left / 12.0, 280], fill=(255, 120, 255))
        phase = {"play": "chase" if not getattr(v, "scatter", False) else "scatter wave", "down": "reset: ghosts back off",
                 "reset_done": "reset: standing up"}.get(v.phase, v.phase)
        d.text((x0, 288), f"phase: {phase}", fill=(170, 170, 190), font=self.f_tiny)
        # ticker (last 3 events, fading)
        y = 312
        for (t, text, col) in self.ticker[-2:][::-1]:
            age = v.t - t
            if age < 6.0:
                a = int(255 * max(0.25, 1 - age / 6.0))
                d.text((x0, y), text, fill=col + (a,), font=self.f_small)
                y += 24
        # legend
        ly = 378
        d.text((x0, ly - 24), "who's who", fill=(150, 150, 170), font=self.f_tiny)
        for p in ["P_", "G0_", "G1_", "G2_", "G3_"]:
            col = _rgb(COLORS[p])
            mode = v.ghost_mode.get(p)
            if mode == "frightened":
                col = (90, 110, 255)
            elif mode in ("eaten",):
                col = (150, 150, 160)
            d.rectangle([x0, ly + 3, x0 + 14, ly + 17], fill=col)
            name = self.duck_names[p]
            role = "the player" if p == "P_" else GHOST_NAMES[p][1]
            if mode == "frightened":
                role = "frightened!"
            elif mode == "eaten":
                role = "caught, walking home"
            d.text((x0 + 22, ly), f"{name:9s} {role}", fill=(220, 220, 230), font=self.f_tiny)
            ly += 21
        # inset
        canvas.paste(inset, (PANEL_X + 20, H - INSET_H - 44))
        d.rectangle([PANEL_X + 20, H - INSET_H - 44, PANEL_X + 20 + INSET_W, H - 44], outline=(90, 90, 120))
        d.text((PANEL_X + 22, H - 40), "chase cam (Duck-Man)", fill=(150, 150, 170), font=self.f_tiny)
        d.text((PANEL_X + 22, H - 22), self.label + (f"  |  {self.speed}x speed" if self.speed != 1 else ""), fill=(255, 215, 80), font=self.f_tiny)
        # arena footer
        d.rectangle([0, H - 26, MAIN_W, H], fill=(0, 0, 0, 140))
        d.text((10, H - 22), "MuJoCo  |  real Microduck model x5  |  BAM actuators  |  Pollen's learned gait on every joint  |  simulation only",
               fill=(200, 200, 215), font=self.f_tiny)
        return np.asarray(canvas)

    def save(self, path):
        w = imageio.get_writer(path, fps=self.fps, codec="libx264", quality=6, pixelformat="yuv420p", macro_block_size=1)
        for f in self.frames:
            w.append_data(f)
        w.close()


def curve_png(csv_path, out_png, w=1280, h=720):
    rows = list(csv.DictReader(open(csv_path)))
    xs = [int(r["gen"]) for r in rows]
    ys = [float(r["theta_fit"]) for r in rows]
    sc = [float(r["score_mean"]) for r in rows]
    img = Image.new("RGB", (w, h), (18, 18, 28))
    d = ImageDraw.Draw(img)
    f = _font(24)
    pad = 80
    lo, hi = min(min(ys), min(sc)), max(max(ys), max(sc))
    span = max(1e-6, hi - lo)

    def pts(vals):
        return [(pad + (x - xs[0]) / max(1, xs[-1] - xs[0]) * (w - 2 * pad), h - pad - (y - lo) / span * (h - 2 * pad))
                for x, y in zip(xs, vals)]
    d.line([(pad, h - pad), (w - pad, h - pad)], fill=(120, 120, 140), width=2)
    d.line([(pad, pad), (pad, h - pad)], fill=(120, 120, 140), width=2)
    if len(xs) > 1:
        d.line(pts(sc), fill=(120, 160, 255), width=3)
        d.line(pts(ys), fill=(255, 210, 60), width=4)
    d.text((pad, 20), "Duck-Man strategy: evolution strategies inside the MuJoCo sim (laptop CPU), after imitation of the planner",
           fill=(235, 235, 235), font=f)
    d.text((pad, 50), f"yellow: fitness of the unperturbed policy   blue: mean score of the population   generations {xs[0]}-{xs[-1]}",
           fill=(200, 200, 210), font=f)
    d.text((pad, h - pad + 12), f"generation ->    (y range {lo:.0f} to {hi:.0f})", fill=(200, 200, 210), font=f)
    img.save(out_png)
