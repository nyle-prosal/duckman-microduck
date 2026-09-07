import numpy as np
import pytest
from duckman.eval import run_eval
from duckman.constants import ROOT

CK = ROOT / "checkpoints/strategy_final.npz"


def test_neutral_policy_fails():
    r = run_eval("neutral", 0, None)                     # full 240 s clock, same as every other policy
    assert r["coins"] == 0 and r["score"] == 0
    assert r["calls"]["P_"]["policy"] == int(round(r["t"] / 0.02))


def test_frozen_policy_fails():
    r = run_eval("frozen", 0, None)                      # no network at all: default pose held by the actuators
    assert r["coins"] == 0 and r["score"] == 0
    assert r["calls"]["P_"]["policy"] == int(round(r["t"] / 0.02))


@pytest.mark.skipif(not CK.exists(), reason="no trained checkpoint yet")
def test_learned_policy_succeeds_on_seed_0():
    r = run_eval("learned", 0, str(CK))
    assert r["score"] >= 200 and r["coins"] >= 12, {k: r[k] for k in ("score", "coins", "lives_lost", "end")}
    lo, hi = np.array(r["action_bounds"]["P_"]["min"]), np.array(r["action_bounds"]["P_"]["max"])
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    n = run_eval("neutral", 0, None)
    assert n["coins"] == 0 and n["score"] < r["score"]
