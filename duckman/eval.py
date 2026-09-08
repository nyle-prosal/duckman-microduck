"""Evaluate one Duck-Man policy on one seed and write the evidence JSON.

python -m duckman.eval --policy learned|planner|neutral|frozen --seed 0 [--checkpoint x.npz] [--out r.json] [--video v.mp4]
    [--ghosts scripted|learned] [--recovery sitstand|standup] [--max-t S] [--speed N] [--label TEXT]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from .constants import CLOCK_S, POLICY_DIR, ROOT
from .maze import Maze
from .game import Game
from .policies import DuckManPolicy, NeutralStrategy, FrozenDuckMan
from .ghosts import default_ghosts
from .planner import PlannerStrategy
from .strategy_net import NetStrategy


def make_policy(name, checkpoint=None, recovery="sitstand"):
    if name == "learned":
        return DuckManPolicy(NetStrategy.load(checkpoint), recovery)
    if name == "planner":
        return DuckManPolicy(PlannerStrategy(), recovery)
    if name == "neutral":
        return DuckManPolicy(NeutralStrategy(), recovery)
    if name == "frozen":
        return FrozenDuckMan()
    raise ValueError(name)


def resolve_recovery(recovery):
    """'auto' -> stand-up policy if assets/policies/standup.onnx exists, else sit-and-stand."""
    if recovery == "auto":
        return "standup" if (POLICY_DIR / "standup.onnx").exists() else "sitstand"
    return recovery


def make_ghosts(kind, recovery):
    rec = "standup" if recovery == "standup" else "none"
    if kind == "learned":
        from .ghost_net import learned_ghosts
        return learned_ghosts(str(ROOT / "checkpoints/ghosts_final.npz"), rec)
    return default_ghosts(rec)


def run_eval(policy, seed, checkpoint=None, max_t=CLOCK_S, video=None, log_positions=False, recovery="auto",
             speed=1, label=None, vin=None, ghosts="scripted"):
    recovery = resolve_recovery(recovery)
    ghost_kind = ghosts
    ghosts = make_ghosts(ghost_kind, recovery)
    g = Game(Maze(), seed, make_policy(policy, checkpoint, recovery), ghosts, log_positions=log_positions, vin=vin)
    g.reset()
    rec = None
    if video:
        from .render import Recorder
        try:
            rec = Recorder(g, label or f"{policy}  seed {seed}", speed=speed)
        except Exception as e:                     # headless machine without EGL/OSMesa: evaluate anyway, skip the video
            print(f"[warn] video disabled: {type(e).__name__}: {str(e)[:120]} -- on headless Linux install libegl1 or libosmesa6", flush=True)
            rec = None
    t0 = time.time()
    while not g.done and g.sim.t < max_t - 1e-9:
        g.step()
        if rec:
            rec.maybe_capture()
    r = g.result()
    r.update(policy=policy, wall_s=round(time.time() - t0, 1), recovery=recovery, checkpoint=checkpoint, battery_vin=vin, ghost_kind=ghost_kind,
             checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
             max_t=max_t, control_dt=0.02)
    if rec:
        rec.save(video)
        r["video"] = video
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint")
    ap.add_argument("--out")
    ap.add_argument("--video")
    ap.add_argument("--speed", type=int, default=1, help="video speed-up factor (labelled on screen)")
    ap.add_argument("--label")
    ap.add_argument("--max-t", type=float, default=CLOCK_S)
    ap.add_argument("--recovery", default="auto", help="auto|standup|sitstand")
    ap.add_argument("--ghosts", default="scripted", help="scripted|learned (learned needs checkpoints/ghosts_final.npz)")
    a = ap.parse_args()
    r = run_eval(a.policy, a.seed, a.checkpoint, a.max_t, a.video, recovery=a.recovery, speed=a.speed, label=a.label, ghosts=a.ghosts)
    print(json.dumps({k: v for k, v in r.items() if k != "events"}, indent=1, default=str))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
