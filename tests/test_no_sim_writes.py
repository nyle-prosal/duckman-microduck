"""Mechanical trust-contract check: no rollout code ever writes simulator state.

Walks the AST of every function in duckman/ that runs during an episode and fails if any of them assigns
into qpos, qvel, ctrl, xfrc_applied, qfrc_applied, mocap_pos/quat or calls mj_resetData. The only places
allowed to touch state are the reset paths (Sim.reset, Duck.set_pose) and the BAM controller, which writes
ctrl torques from the policy's joint targets. Adapted from an idea in another entrant's submission.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "duckman"
FORBIDDEN_ATTRS = {"qpos", "qvel", "ctrl", "xfrc_applied", "qfrc_applied", "mocap_pos", "mocap_quat", "act", "qacc"}
ALLOWED_FUNCS = {("duck.py", "set_pose"), ("duck.py", "place"), ("world.py", "reset"), ("world.py", "set_ball")}


def _writes(fn: ast.FunctionDef):
    found = []
    for node in ast.walk(fn):
        targets = []
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            base = t
            while isinstance(base, ast.Subscript):
                base = base.value
            if isinstance(base, ast.Attribute) and base.attr in FORBIDDEN_ATTRS:
                found.append(f"{fn.name}: writes .{base.attr} (line {node.lineno})")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("mj_resetData", "mj_resetDataKeyframe"):
            found.append(f"{fn.name}: calls {node.func.attr} (line {node.lineno})")
    return found


def test_rollout_code_never_writes_simulator_state():
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if (path.name, node.name) in ALLOWED_FUNCS:
                    continue
                offenders += [f"{path.name}::{w}" for w in _writes(node)]
    assert not offenders, "simulator state written outside reset paths:\n" + "\n".join(offenders)


def test_allowed_writers_are_only_reset_paths():
    # sanity: the allowlist itself must be small and named
    assert ALLOWED_FUNCS == {("duck.py", "set_pose"), ("duck.py", "place"), ("world.py", "reset"), ("world.py", "set_ball")}
