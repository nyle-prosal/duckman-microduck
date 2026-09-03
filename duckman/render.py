"""Overhead renderer with HUD, and a Pillow learning-curve chart."""
import csv
import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont
from .constants import CTRL_DT, CELL


def _font(size):
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


class Recorder:
    def __init__(self, game, label, fps=25, size=(1280, 720), speed=1):
        self.g, self.label, self.fps, self.size, self.speed = game, label, fps, size, speed
        self.frames, self.k = [], 0
        self.every = max(1, int(round(1 / (fps * CTRL_DT)))) * speed
        self.r = mujoco.Renderer(game.sim.model, size[1], size[0])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        m = game.maze
        self.cam.lookat[:] = [(m.cols - 1) * CELL / 2, (m.rows - 1) * CELL / 2 - 0.1, 0]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 4.1, 90, -62
        self.font = _font(26)

    def maybe_capture(self):
        self.k += 1
        if self.k % self.every == 0:
            self.frames.append(self.frame())

    def frame(self):
        self.r.update_scene(self.g.sim.data, self.cam)
        img = Image.fromarray(self.r.render())
        draw_hud(img, self.g.view, self.label, self.speed, self.font)
        return np.asarray(img)

    def save(self, path):
        w = imageio.get_writer(path, fps=self.fps, codec="libx264", quality=6, pixelformat="yuv420p", macro_block_size=1)
        for f in self.frames:
            w.append_data(f)
        w.close()


def draw_hud(img, view, label, speed=1, font=None):
    font = font or _font(26)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, img.width, 44], fill=(0, 0, 0, 160))
    txt = (f"DUCK-MAN   score {view.score:5d}   lives {'*' * view.lives:<3}   coins left {sum(view.coins_alive.values()):2d}"
           f"   clock {view.clock_left:5.1f}s")
    if view.power_left > 0:
        txt += f"   POWER {view.power_left:4.1f}"
    if view.phase != "play":
        txt += "   [reset]"
    d.text((12, 8), txt, fill=(255, 255, 255, 255), font=font)
    d.rectangle([0, img.height - 40, img.width, img.height], fill=(0, 0, 0, 140))
    d.text((12, img.height - 34), label + (f"   {speed}x speed" if speed != 1 else ""), fill=(255, 230, 120, 255), font=font)


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
    d.text((pad, 20), "Duck-Man strategy: evolution strategies inside the MuJoCo sim (laptop CPU)", fill=(235, 235, 235), font=f)
    d.text((pad, 50), f"yellow: fitness of the unperturbed policy   blue: mean score of the population   generations {xs[0]}-{xs[-1]}",
           fill=(200, 200, 210), font=f)
    d.text((pad, h - pad + 12), f"generation ->    (y range {lo:.0f} to {hi:.0f})", fill=(200, 200, 210), font=f)
    img.save(out_png)
