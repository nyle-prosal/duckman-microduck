"""ONNX policy wrapper. Counts calls and tracks the action range for the evidence report."""
import numpy as np
import onnxruntime as ort
from .constants import OBS_DIM, ACT_DIM, DEFAULT_POSE


class Gait:
    def __init__(self, path, scale=1.0):
        self.path, self.scale = str(path), scale
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        self.i, self.o = self.s.get_inputs()[0].name, self.s.get_outputs()[0].name
        shape = tuple(self.s.get_inputs()[0].shape[1:])
        assert shape == (OBS_DIM,), shape
        self.calls = 0
        self.action_min = np.full(ACT_DIM, np.inf)
        self.action_max = np.full(ACT_DIM, -np.inf)

    def run(self, obs61: np.ndarray) -> np.ndarray:
        a = self.s.run([self.o], {self.i: obs61.astype(np.float32)[None]})[0][0].astype(np.float32)
        self.calls += 1
        self.action_min = np.minimum(self.action_min, a)
        self.action_max = np.maximum(self.action_max, a)
        return a

    def targets(self, action: np.ndarray) -> np.ndarray:
        return DEFAULT_POSE + action * self.scale
