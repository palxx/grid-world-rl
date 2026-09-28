import numpy as np

from rtgrid import config as C
from rtgrid.env import CardinalEnv
from rtgrid.generator import make_layout
from rtgrid.layout import Layout, _transforms, canonical_hash


def test_make_layout_deterministic_and_sized(spec):
    a, _ = make_layout(spec, 5)
    b, _ = make_layout(spec, 5)
    assert a.rows() == b.rows() and a.start == b.start
    assert a.grid.shape == (spec.size, spec.size)
    assert len(a.cells("G")) == 1


def test_hash_dihedral_invariant(spec):
    lay, _ = make_layout(spec, 7)
    h = canonical_hash(lay)
    for g, s in _transforms(lay.grid, lay.start):
        assert canonical_hash(Layout(np.array(g), s)) == h
    other, _ = make_layout(spec, 8)
    assert canonical_hash(other) != h


def test_env_rewards():
    g = np.array([list(r) for r in ["#####", "#.cG#", "#L..#", "#####", "#####"]], dtype="<U1")
    lay = Layout(g, (1, 1))
    env = CardinalEnv(lay, horizon=10)
    r = env.step(C.UP)  # wall bump
    assert r.bump and r.reward == C.STEP + C.BUMP and env.pos == (1, 1)
    r = env.step(C.RIGHT)  # coin
    assert r.reward == C.STEP + C.COIN and env.mask == 1
    r = env.step(C.LEFT)
    r = env.step(C.RIGHT)  # coin already collected
    assert r.reward == C.STEP and env.mask == 1
    r = env.step(C.RIGHT)  # goal
    assert r.terminated and r.reward == C.STEP + C.GOAL and env.done
    env2 = CardinalEnv(lay, 10)
    r = env2.step(C.DOWN)  # lava
    assert r.terminated and r.reward == C.STEP + C.LAVA


def test_truncation():
    g = np.array([list(r) for r in ["####", "#.G#", "####", "####"]], dtype="<U1")
    env = CardinalEnv(Layout(g, (1, 1)), horizon=3)
    for _ in range(3):
        r = env.step(C.UP)
    assert r.truncated and not r.terminated and env.done
