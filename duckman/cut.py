"""Cut a time window out of an already-rendered evaluation video (re-encode only; no frame is altered).
python -m duckman.cut in.mp4 out.mp4 --start 50 --end 70"""
import argparse
import imageio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    a = ap.parse_args()
    r = imageio.get_reader(a.src)
    fps = r.get_meta_data().get("fps", 25)
    w = imageio.get_writer(a.dst, fps=fps, codec="libx264", quality=6, pixelformat="yuv420p", macro_block_size=1)
    n = 0
    for i, fr in enumerate(r):
        t = i / fps
        if a.start <= t < a.end:
            w.append_data(fr)
            n += 1
    w.close()
    print(f"wrote {a.dst}: {n} frames ({n / fps:.1f} s) from {a.src} [{a.start}, {a.end}) s")


if __name__ == "__main__":
    main()
