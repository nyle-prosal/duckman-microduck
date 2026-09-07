"""Duck-Man: maze tag for Microduck.

Headless Linux: MuJoCo's default GL backend needs a display. If none is present, pick EGL or OSMesa
before mujoco is imported anywhere, so ./run.sh can render result.mp4 on a server. If neither library is
installed, evaluation still runs; only the video step is skipped (see duckman/eval.py).
"""
import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ:
    from ctypes.util import find_library
    # OSMesa (software) first: it works on any headless box; EGL needs a working GPU driver
    if find_library("OSMesa"):
        os.environ["MUJOCO_GL"] = "osmesa"
    elif find_library("EGL"):
        os.environ["MUJOCO_GL"] = "egl"
