from duckman.maze import Maze


def test_all_corridor_reachable():
    m = Maze()
    for c in m.corridor:
        assert m.dist(m.pac_start, c, ghost=False) is not None, c


def test_tokens_on_corridor_and_unique():
    m = Maze()
    t = m.coins + m.pellets
    assert set(t) <= m.corridor and len(set(t)) == len(t) and len(m.pellets) == 4
    assert m.pac_start not in t and m.door not in t


def test_house_only_for_ghosts():
    m = Maze()
    assert all(h not in m.neighbors(m.door, ghost=False) for h in m.house)
    assert any(h in m.neighbors(m.door, ghost=True) for h in m.house)


def test_bfs_step_moves_closer():
    m = Maze()
    a, b = m.pac_start, m.pellets[0]
    n = m.bfs_next(a, b, ghost=False)
    assert n in m.neighbors(a, False) and m.dist(n, b, False) == m.dist(a, b, False) - 1


def test_xy_roundtrip():
    m = Maze()
    for c in m.corridor:
        assert m.cell(*m.xy(c)) == c
