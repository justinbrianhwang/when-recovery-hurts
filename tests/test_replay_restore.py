"""Regression: restore must reproduce simulator state that is NOT in get_state_dict.

Found on the first real GPU host (smoke_pick_002 / smoke_stack_002): set_state_dict on rigid actors loses
PhysX contact/solver state, so restored rollouts diverged from the unrestored one (up to 6e-3 > 1e-6).
Runtime.restore now replays the logged action prefix from reset(seed) and verifies the snapshot state.
Mock physics only: this checks the control flow, not ManiSkill itself.
"""
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from test_runtime_mock import MockEnv, MockPolicy
from hci.simulation import Runtime, trace, compare_traces

CFG = {'env': {'horizon': 40}, 'protocol': {'min_intervention_step': 12, 'decision_stride': 4, 'minimum_post_retreat_steps': 4},
       'retreat': {'max_steps': 6, 'arrival_tolerance_m': .005, 'mode': 'up', 'distance_m': .03, 'max_translation_per_step_m': .005, 'lookback_steps': 8},
       'audit': {'state_atol': 1e-6}}


class HiddenStateEnv(MockEnv):
    """Momentum `h` influences every step but is absent from get_state_dict (like a solver cache)."""
    def __init__(self):
        super().__init__(); self.h = torch.zeros((1, 3)); self.resets = 0; self.jitter = 0.
    def reset(self, seed):
        self.h = torch.zeros((1, 3)); self.resets += 1
        out = super().reset(seed); self.p = self.p + self.jitter * self.resets
        return out
    def set_state_dict(self, s):
        raise AssertionError('restore must not rely on set_state_dict')
    def step(self, a):
        self.h = .9 * self.h + a[:, :3]
        self.p = self.p + self.h * .01; self._elapsed_steps += 1
        return self.obs(), 0., torch.tensor([False]), torch.tensor([False]), {'success': torch.tensor([False])}


def test_restore_reproduces_hidden_state_and_survives_other_branch():
    rt = Runtime(HiddenStateEnv(), MockPolicy(), CFG).reset(1)
    for _ in range(12): rt.step_base()
    s = rt.snapshot(); a = trace(rt, 16)
    rt.restore(s); b = trace(rt, 16)
    rt.restore(s); rt.force(2); rt.complete()
    rt.restore(s); c = trace(rt, 16)
    for other in (b, c):
        r = compare_traces(a, other)
        assert r['state_max_abs'] == 0 and r['action_max_abs'] == 0 and r['outcome_equal']
    assert rt.t == 28 and len(rt.actions) == 28 and rt.replay_error == 0


def test_restore_fails_loudly_when_replay_does_not_match():
    env = HiddenStateEnv(); rt = Runtime(env, MockPolicy(), CFG).reset(1)
    for _ in range(12): rt.step_base()
    s = rt.snapshot(); env.jitter = 1e-3
    with pytest.raises(RuntimeError):
        rt.restore(s)


def test_snapshot_from_other_episode_is_rejected():
    rt = Runtime(HiddenStateEnv(), MockPolicy(), CFG).reset(1)
    for _ in range(12): rt.step_base()
    s = rt.snapshot(); rt.reset(2)
    with pytest.raises(ValueError):
        rt.restore(s)
