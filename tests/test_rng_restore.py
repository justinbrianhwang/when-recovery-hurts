"""Regression: environment RNG save/restore must work with ManiSkill's BatchedRNG.

Found on the first real GPU host (smoke_pick_001 / smoke_stack_001): copy.deepcopy(BatchedRNG)
raises because BatchedRNG.__getattribute__ forwards __reduce_ex__ to its member generators.
"""
import copy
import numpy as np
import pytest
from hci.simulation import rng_state, set_rng_state


class BatchedRNG(np.random.RandomState):
    """Local replica of the forwarding behaviour of mani_skill BatchedRNG at the pinned commit."""
    def __init__(self, rngs):
        self.rngs = rngs
    def __getitem__(self, i):
        return self.rngs[i]
    def __getattribute__(self, item):
        if item in ['rngs', '__class__', '__dict__', '__getattribute__', '__getitem__']:
            return object.__getattribute__(self, item)
        if callable(getattr(self.rngs[0], item)):
            def method(*args, **kwargs):
                return np.array([object.__getattribute__(r, item)(*args, **kwargs) for r in self.rngs])
            return method
        return np.array([getattr(r, item) for r in self.rngs])


class Env:
    pass


def make(cls):
    e = Env()
    e._batched_episode_rng = cls([np.random.RandomState(5)])
    e._episode_rng = e._batched_episode_rng[0]
    e._main_rng = np.random.RandomState(9)
    return e


def check(cls):
    e = make(cls)
    names = ('_batched_episode_rng', '_episode_rng', '_main_rng')
    saved = {n: rng_state(getattr(e, n)) for n in names}
    first = [e._batched_episode_rng.uniform(size=3), e._episode_rng.uniform(size=3), e._main_rng.uniform(size=3)]
    for n in names:
        set_rng_state(e, n, saved[n])
    assert e._episode_rng is e._batched_episode_rng[0]  # aliasing preserved
    second = [e._batched_episode_rng.uniform(size=3), e._episode_rng.uniform(size=3), e._main_rng.uniform(size=3)]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_replica_reproduces_deepcopy_failure_and_state_restore_works():
    with pytest.raises(Exception):
        copy.deepcopy(BatchedRNG([np.random.RandomState(1)]))
    check(BatchedRNG)


def test_real_maniskill_batched_rng_restore():
    mod = pytest.importorskip('mani_skill.envs.utils.randomization.batched_rng')
    check(mod.BatchedRNG)


def test_layout_change_is_rejected_and_other_objects_still_copy():
    e = make(BatchedRNG)
    saved = rng_state(e._batched_episode_rng)
    e._batched_episode_rng = BatchedRNG([np.random.RandomState(1), np.random.RandomState(2)])
    with pytest.raises(ValueError):
        set_rng_state(e, '_batched_episode_rng', saved)
    g = np.random.default_rng(3)
    e.gen = g
    s = rng_state(g)
    a = e.gen.random()
    set_rng_state(e, 'gen', s)
    assert e.gen.random() == a
