from duckman.train_es import train


def test_es_two_generations(tmp_path):
    out = train("smoke", generations=2, pop=4, seeds_per_gen=1, workers=2, max_t=8.0, root=tmp_path)
    assert (tmp_path / "smoke/curve.csv").exists() and (tmp_path / "smoke/gen_0002.npz").exists()
    assert out["generations"] == 2
