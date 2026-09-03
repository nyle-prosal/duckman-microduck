"""Assemble result.mp4: title card -> labelled clips -> learning curve -> end card with measured numbers.

python -m duckman.make_video --clips "a.mp4:Generation 0" "b.mp4:Final policy (2x speed)" \
    --results results/learned_seed0.json results/planner_seed0.json results/neutral_seed0.json \
    --curve checkpoints/curve.csv --out result.mp4
No frame is edited; clips are the unmodified evaluation renders (speed-ups are burned in by the renderer).
"""
import argparse
import json
import os
import imageio
import numpy as np
from PIL import Image, ImageDraw
from .render import _font, curve_png

W, H, FPS = 1280, 720, 25


def card(lines, seconds, bg=(14, 14, 24)):
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    y = 90
    for text, size, color in lines:
        f = _font(size)
        d.text((80, y), text, fill=color, font=f)
        y += int(size * 1.6)
    return [np.asarray(img)] * int(seconds * FPS)


def label_frames(path, label):
    f = _font(28)
    for fr in imageio.get_reader(path):
        img = Image.fromarray(fr)
        if img.size != (W, H):
            img = img.resize((W, H))
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([W - 560, 52, W - 12, 96], fill=(0, 0, 0, 150))
        d.text((W - 548, 60), label, fill=(255, 255, 255, 255), font=f)
        yield np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="path:label")
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--curve")
    ap.add_argument("--out", default="result.mp4")
    a = ap.parse_args()
    w = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=6, pixelformat="yuv420p", macro_block_size=1)
    white, gold, grey = (240, 240, 240), (255, 215, 80), (170, 170, 185)
    for fr in card([("DUCK-MAN", 72, gold), ("Maze tag for Microduck", 40, white),
                    ("Five real Microducks in MuJoCo. Every duck walks with Pollen Robotics' learned gait", 26, grey),
                    ("through the BAM actuator model. Coins count only when the Duck-Man knocks them over.", 26, grey),
                    ("The Duck-Man's strategy network was trained by evolution strategies", 26, grey),
                    ("inside this simulation on a laptop CPU. Ghosts and navigation are scripted.", 26, grey),
                    ("Simulation only. No hardware claims.", 26, grey)], 4.0):
        w.append_data(fr)
    for spec in a.clips:
        path, label = spec.split(":", 1)
        for fr in label_frames(path, label):
            w.append_data(fr)
    if a.curve and os.path.exists(a.curve):
        png = a.out + ".curve.png"
        curve_png(a.curve, png)
        img = np.asarray(Image.open(png).convert("RGB").resize((W, H)))
        for _ in range(int(4 * FPS)):
            w.append_data(img)
    lines = [("Measured results (same seed, same maze)", 40, gold)]
    for rp in a.results:
        r = json.load(open(rp))
        lines.append((f"{r['labels']['P_']:<44s} score {r['score']:>5d}   coins {r['coins']:>2d}   lives lost {r['lives_lost']}"
                      f"   end: {r['end']}   seed {r['seed']}", 24, white))
    lines.append(("Every number above comes from results/*.json written by duckman/eval.py; ./run.sh reproduces them.", 22, grey))
    for fr in card(lines, 6.0):
        w.append_data(fr)
    w.close()
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
