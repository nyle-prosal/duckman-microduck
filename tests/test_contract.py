import numpy as np
import onnxruntime as ort
from duckman.constants import POLICY_DIR, DEFAULT_POSE, OBS_DIM, ACT_DIM, ROBOT_XML
from duckman.gait import Gait


def test_onnx_contract():
    for n in ["alpha_walking", "alpha_stand", "alpha_sitstand"]:
        g = Gait(POLICY_DIR / f"{n}.onnx")
        a = g.run(np.zeros(OBS_DIM, np.float32))
        assert a.shape == (ACT_DIM,) and np.isfinite(a).all() and g.calls == 1


def test_default_pose_matches_onnx_metadata():
    s = ort.InferenceSession(str(POLICY_DIR / "alpha_walking.onnx"))
    meta = s.get_modelmeta().custom_metadata_map
    if "default_joint_pos" in meta:
        ref = np.array([float(x) for x in meta["default_joint_pos"].strip("[]").split(",")])
        assert np.allclose(ref, DEFAULT_POSE, atol=1e-3)


def test_assets_present():
    assert ROBOT_XML.exists() and (ROBOT_XML.parent / "assets").is_dir()
