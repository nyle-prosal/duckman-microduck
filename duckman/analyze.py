"""What did the strategy learn? Behaviour statistics and cell-occupancy heat maps, learned vs planner.
python -m duckman.analyze --seeds 10 --workers 4"""
import argparse
import json
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .constants import ROOT
from .maze import Maze
from .game import Game
from .eval import make_policy, resolve_recovery
from .ghosts import default_ghosts


def _one(args):
    policy, seed = args
    rec = resolve_recovery("auto")
    ck = str(ROOT / "checkpoints/strategy_final.npz") if policy == "learned" else None
    g = Game(Maze(), seed, make_policy(policy, ck, rec), default_ghosts("standup" if rec == "standup" else "none"), log_positions=True)
    g.reset()
    r = g.run()
    ev = r["events"]
    pellets = [e[0] for e in ev if e[1] == "pellet"]
    catches = [e[0] for e in ev if e[1] == "ghost_caught"]
    tags = [e[0] for e in ev if e[1] == "tag"]
    # catches per power window: a catch belongs to the latest pellet before it
    per_window = [sum(1 for c in catches if p <= c < p + 12.0) for p in pellets]
    occ = np.zeros((g.maze.rows, g.maze.cols))
    for snap in g.positions_log:
        c = g.maze.cell(*snap["P_"][:2])
        if 0 <= c[0] < g.maze.cols and 0 <= c[1] < g.maze.rows:
            occ[c[1], c[0]] += 1
    return {"policy": policy, "seed": seed, "score": r["score"], "first_pellet_s": pellets[0] if pellets else None,
            "pellets": len(pellets), "catches": len(catches), "catch_windows": sum(1 for w in per_window if w),
            "tags_during_power": sum(1 for t in tags if any(p <= t < p + 12.0 for p in pellets)), "tags": len(tags),
            "time_to_first_tag_s": tags[0] if tags else None, "occ": occ.tolist()}


def heatmap(occ_l, occ_p, maze, path):
    W, H, pad, cw = 1280, 720, 60, 70
    img = Image.new("RGB", (W, H), (18, 18, 28))
    d = ImageDraw.Draw(img)
    f = ImageFont.load_default(22)
    fs = ImageFont.load_default(16)
    for k, (occ, title) in enumerate(((occ_l, "learned strategy"), (occ_p, "scripted planner"))):
        ox = pad + k * 620
        oy = 110
        d.text((ox, 70), f"Where the Duck-Man spends its time: {title}", fill=(235, 235, 235), font=f)
        m = occ / max(1e-9, occ.max())
        for r in range(maze.rows):
            for c in range(maze.cols):
                x0, y0 = ox + c * cw, oy + r * cw
                if (c, r) in maze.wall_cells:
                    d.rectangle([x0, y0, x0 + cw - 2, y0 + cw - 2], fill=(35, 40, 100))
                elif (c, r) in maze.house:
                    d.rectangle([x0, y0, x0 + cw - 2, y0 + cw - 2], fill=(60, 60, 75))
                else:
                    v = m[r, c]
                    d.rectangle([x0, y0, x0 + cw - 2, y0 + cw - 2], fill=(int(30 + 225 * v), int(30 + 180 * v), int(40 + 40 * (1 - v))))
                    d.text((x0 + 6, y0 + 24), f"{100 * occ[r, c] / occ.sum():.0f}%", fill=(20, 20, 30) if v > 0.5 else (200, 200, 210), font=fs)
                if (c, r) in maze.pellets:
                    d.ellipse([x0 + cw - 22, y0 + 4, x0 + cw - 6, y0 + 20], fill=(255, 120, 255))
    d.text((pad, 20), "Cell occupancy over 10 held-out rounds each (percent of control steps); pink dots = power pellets", fill=(200, 200, 210), font=f)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(p, s) for p in ("learned", "planner") for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    maze = Maze()
    occ = {p: np.sum([np.array(r["occ"]) for r in rows if r["policy"] == p], axis=0) for p in ("learned", "planner")}
    heatmap(occ["learned"], occ["planner"], maze, ROOT / "training/strategy/heatmap.png")
    for r in rows:
        del r["occ"]
    json.dump(rows, open(ROOT / "results/analysis.json", "w"), indent=1)
    lines = ["| Behaviour (mean per round, 10 held-out seeds) | Learned | Planner |", "|---|---|---|"]

    def stat(key, fmt="{:.2f}", none_ok=False):
        out = []
        for p in ("learned", "planner"):
            vals = [r[key] for r in rows if r["policy"] == p and r[key] is not None]
            out.append(fmt.format(np.mean(vals)) if vals else "never")
        return out
    for label, key, fmt in [("pellets collected", "pellets", "{:.2f}"), ("time to first pellet (s)", "first_pellet_s", "{:.0f}"),
                            ("ghosts caught", "catches", "{:.2f}"), ("power windows with at least one catch", "catch_windows", "{:.2f}"),
                            ("tags suffered", "tags", "{:.2f}"), ("of which during own power window", "tags_during_power", "{:.2f}"),
                            ("time to first tag (s)", "time_to_first_tag_s", "{:.0f}"), ("score", "score", "{:.0f}")]:
        l, p = stat(key, fmt)
        lines.append(f"| {label} | {l} | {p} |")
    t = "\n".join(lines) + "\n"
    (ROOT / "results/analysis.md").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
