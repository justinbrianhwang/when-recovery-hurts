"""Recovery-V1 oracle XY pilot: trigger, the four arms, the matched duration, privileged-information isolation, accounting.

Mock physics only. These tests say nothing about ManiSkill, the GPU, the real controller tracking or the research policy.
"""
import ast, copy, inspect, itertools, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from test_runtime_mock import MockEnv
from hci.policies import QueuePolicy
from hci.simulation import Runtime
from hci.trees import clone
from hci.io import seed_for, read_json
from hci import recovery_v1_oracle as O
from hci import pilot

CFG = {'env': {'horizon': 90}, 'protocol': {'min_intervention_step': 12, 'decision_stride': 4, 'minimum_post_retreat_steps': 4},
       'retreat': {'max_steps': 40, 'arrival_tolerance_m': .005, 'mode': 'up', 'distance_m': .05, 'max_translation_per_step_m': .005, 'lookback_steps': 8},
       'audit': {'state_atol': 1e-6, 'action_atol': 1e-6, 'observation_atol': 1.0, 'replay_steps': 10}}
A0 = np.array([.10, .00]); B0 = np.array([.00, .10]); OPEN_W = .08; CUBE_W = .04; GAIN = .4; PLACE_TOL = .015


class PlaceEnv(MockEnv):
    """A gripper that tracks 40 % of each commanded increment, a cube that is carried while the gripper is closed on it,
    and success when the cube is released within PLACE_TOL of the target centre. `reads` counts accesses to cube poses."""
    gain = GAIN

    def __init__(self):
        super().__init__(); self.reads = 0; self._boot()

    def _boot(self):
        self.p = torch.tensor([[0., 0., .1]]); self.w = OPEN_W; self.A = torch.tensor([[A0[0], A0[1], .02]]); self.B = torch.tensor([[B0[0], B0[1], .02]])
        self.held = False; self.ok = False; self._elapsed_steps = torch.zeros(1, dtype=torch.int32)

    @property
    def cubeA(self):
        self.reads += 1; return SimpleNamespace(pose=SimpleNamespace(p=self.A.clone()))

    @property
    def cubeB(self):
        self.reads += 1; return SimpleNamespace(pose=SimpleNamespace(p=self.B.clone()))

    def obs(self):
        q = torch.zeros((1, 9)); q[0, -2:] = self.w / 2
        return {'agent': {'qpos': q, 'qvel': torch.zeros((1, 9))}, 'extra': {'tcp_pose': torch.cat([self.p, torch.tensor([[1., 0., 0., 0.]])], -1)}}

    def _info(self):
        return {'success': torch.tensor([bool(self.ok)])}

    def reset(self, seed):
        self._boot(); return self.obs(), self._info()

    def get_state_dict(self):
        return {'p': self.p.clone(), 'A': self.A.clone(), 'B': self.B.clone(), 'w': torch.tensor([self.w]), 'held': torch.tensor([self.held]), 'ok': torch.tensor([self.ok])}

    def step(self, a):
        self.p = self.p + a[:, :3] * .1 * self.gain; self._elapsed_steps += 1; g = float(a[0, 3])
        if g < 0:
            if self.held or float(torch.linalg.norm(self.p[0, :2] - self.A[0, :2])) < .012: self.held = True; self.w = CUBE_W
            else: self.w = 0.
        else:
            if self.held: self.ok = self.ok or float(torch.linalg.norm(self.A[0, :2] - self.B[0, :2])) <= PLACE_TOL
            self.held = False; self.w = OPEN_W
        if self.held: self.A[0, :2] = self.p[0, :2]
        return self.obs(), 0., torch.tensor([False]), torch.tensor([False]), self._info()


class PlacePolicy(QueuePolicy):
    """Sees only its own proprioception (TCP, finger opening). It believes the target is at B0 + bias(scene), so it
    releases off-centre by |bias|; it opens when it is within `at` of the believed target."""
    at = .012

    def __init__(self, bias_scale=.035):
        super().__init__(8, 2); self.bias_scale = bias_scale

    def convert(self, raw): return clone(raw)

    def bias(self, episode):
        g = np.random.default_rng(seed_for(episode, 0, 0, 'bias')); ang = g.uniform(0, 2 * math.pi); return self.bias_scale * g.uniform(0, 1) * np.array([math.cos(ang), math.sin(ang)])

    def generate(self, episode, t, repeat=0):
        self.last_seed = seed_for(episode, t, repeat, 'dp'); o = self.history[-1]
        p = o['extra']['tcp_pose'][0, :2].numpy().astype(float); w = float(o['agent']['qpos'][0, -2:].sum())
        noise = np.random.default_rng(self.last_seed).normal(0, 2e-4, 2); out = np.zeros((8, 4), np.float32)
        if w > .06: goal, grip = A0, (1. if np.linalg.norm(p - A0) > .008 else -1.)
        elif w < .02: goal, grip = p, -1.
        else:
            goal = B0 + self.bias(episode); grip = 1. if np.linalg.norm(p - goal) <= self.at else -1.
        for i in range(8):
            d = goal - p; n = np.linalg.norm(d); step = d * min(1., .02 / max(n, 1e-12)) + noise
            out[i, :2] = np.clip(step / .1, -1, 1); out[i, 3] = grip; p = p + step * GAIN
            if grip > 0 and w <= .06: out[i, :2] = 0
        return out


def make(env=None, policy=None, cfg=CFG):
    return Runtime(env or PlaceEnv(), policy or PlacePolicy(), cfg)


def probe(rt, z):
    e = rt.env; A = e.A[0].tolist(); B = e.B[0].tolist(); T = e.p[0].tolist()
    return {'t': rt.t, 'tcp': T, 'cubeA': [A[0], A[1], A[2] + (.05 if e.held else 0.)], 'cubeB': B, 'tcp_to_cubeA': 0., 'cubeA_to_cubeB_xy': math.hypot(A[0] - B[0], A[1] - B[1]),
            'is_cubeA_grasped': bool(e.held), 'is_cubeA_on_cubeB': False, 'is_cubeA_static': True, 'success': bool(rt.y), 'action': None if z is None else [float(x) for x in z]}


def scene(seed, **kw):
    return O.collect_scene(make(kw.pop('env', None), kw.pop('policy', None)), seed, CFG, probe=kw.pop('probe', probe), **kw)


@pytest.fixture(scope='module')
def recs():
    return [scene(s)[0] for s in range(1, 41)]


def cell(rec, arm):
    return next(x for x in rec['cells'] if x['arm'] == arm)


def at_trigger(seed=3, env=None):
    r = make(env).reset(seed)
    while not r.done:
        if O.trigger_reading(r)['fires']: return r, r.snapshot()
        r.step_base()
    raise AssertionError('no trigger in this mock scene')


# ---------------------------------------------------------------- trigger
def test_trigger_rule_uses_the_declared_thresholds():
    assert O.trigger_rule(1., 3, .04) and not O.trigger_rule(-1., 30, .04) and not O.trigger_rule(1., 2, .04)
    assert not O.trigger_rule(1., 30, 0.) and not O.trigger_rule(1., 30, .08) and not O.trigger_rule(0., 30, .04)
    assert O.close_run([[0, 0, 0, 1.], [0, 0, 0, -1.], [0, 0, 0, -.2]]) == 2 and O.close_run([]) == 0 and O.close_run([[0, 0, 0, -1.], [0, 0, 0, .5]]) == 0


def test_gripper_sign_matches_the_existing_diagnostics_and_the_pinned_panda_limits():
    """Open = positive: hci.diagnostics counts a close command as action[-1] < 0, and the pinned gripper controller maps
    the normalised command linearly onto [-0.01, 0.04] m per finger (lower = closed)."""
    from hci import diagnostics, recovery_v1
    assert 'r[\'action\'][-1] < 0' in inspect.getsource(diagnostics.stage_events)
    assert 'grip[i] > 0' in inspect.getsource(recovery_v1.classify)
    lo, hi = -.01, .04; width = lambda a: 2 * (lo + (a + 1) / 2 * (hi - lo))
    assert width(-1) < O.WIDTH_MIN_M < .04 < O.WIDTH_MAX_M < width(1) == pytest.approx(.08)


def test_trigger_inputs_are_observable_only():
    """The trigger reads the pending action, the executed actions and agent/qpos. With cube poses poisoned it still works."""
    class Poisoned(PlaceEnv):
        cubeA = property(lambda self: (_ for _ in ()).throw(AssertionError('ground truth read')))
        cubeB = cubeA
    r, _ = at_trigger(env=Poisoned()); rd = O.trigger_reading(r)
    assert rd['fires'] and rd['pending_gripper_command'] > 0 and rd['close_run'] >= O.CLOSE_RUN_MIN and O.WIDTH_MIN_M <= rd['finger_width_m'] <= O.WIDTH_MAX_M


def test_looking_at_the_pending_action_changes_nothing():
    a = make().reset(5); plain = a.complete(); ref = pilot._ref(a)
    b = make().reset(5)
    while not b.done: O.trigger_reading(b); b.step_base()
    assert pilot._same_run(ref, b, CFG['audit'])['match'] and b.policy.inference_calls == plain['policy_calls']


def test_no_trigger_scene_keeps_every_arm_at_the_trunk_outcome():
    class NeverCloses(PlacePolicy):
        def generate(self, episode, t, repeat=0):
            a = super().generate(episode, t, repeat); a[:, 3] = 1.; return a
    rec, traces = scene(2, policy=NeverCloses())
    assert not rec['triggered'] and rec['trigger'] is None and rec['counts']['continuations'] == 0 and rec['oracle_reads'] == 0
    assert all(x['no_trigger'] and x['y'] == rec['trunk']['success'] and x['steps'] == rec['trunk']['steps'] for x in rec['cells']) and set(traces) == {'trunk'}


def test_one_trigger_per_scene_before_the_pending_open_is_executed(recs):
    assert sum(r['triggered'] for r in recs) >= 35
    for r in [r for r in recs if r['triggered']]:
        t = r['trigger']['step']; calls = [k['t'] for k in r['trunk']['inference_calls']]
        assert r['trigger']['fires'] and r['trigger']['held_gripper_command'] < 0 and r['counts']['continuations'] == 4 and r['counts']['reference_control_runs'] == 1
        assert max(c for c in calls if c <= t) > t - 8                      # the pending action comes from a chunk inferred at or before the trigger step
        assert not r['trigger']['gt_for_diagnosis_only']['false_trigger']


def test_false_trigger_is_recorded_not_censored():
    class GhostEnv(PlaceEnv):
        """Fingers stop at cube width although nothing is held (a jammed gripper in the mock)."""
        def step(self, a):
            out = super().step(a)
            if float(a[0, 3]) < 0 and not self.held: self.w = CUBE_W
            return (self.obs(), *out[1:])

    class ClosesEarly(PlacePolicy):
        def generate(self, episode, t, repeat=0):
            a = super().generate(episode, t, repeat)
            if t == 0: a[:, :2] = 0; a[:4, 3] = -1.; a[4:, 3] = 1.
            return a
    rec, _ = scene(4, env=GhostEnv(), policy=ClosesEarly())
    assert rec['triggered'] and rec['trigger']['step'] == 4 and rec['trigger']['gt_for_diagnosis_only']['false_trigger'] and len(rec['cells']) == 4
    assert O.summarize([rec], [4], True)['denominators']['false_triggers_by_ground_truth'] == 1


# ---------------------------------------------------------------- privileged information
def test_ground_truth_reaches_a_controller_only_inside_center_oracle():
    gate = O.OracleGate(); env = PlaceEnv()
    with pytest.raises(PermissionError): O.oracle_offset(env, gate)
    assert env.reads == 0 and gate.reads == 0
    for arm, expect in (('C0', False), ('R5', False), ('HOLD_MATCHED', False), ('CENTER_ORACLE', True)):
        r, s = at_trigger(); r.env.reads = 0; g = O.OracleGate(); O.apply_arm(r, arm, 5, g); r.complete()
        assert (r.env.reads > 0) == expect == (g.reads > 0) and not g.open


def test_cells_record_oracle_reads_only_for_center(recs):
    for r in [r for r in recs if r['triggered']]:
        assert [cell(r, a)['oracle_reads'] > 0 for a in O.ARMS] == [False, False, False, True] and r['reference']['oracle_reads'] > 0


def test_source_has_one_privileged_control_read():
    """cubeA / cubeB appear in exactly one function of the module; the policy classes are not touched by it."""
    src = Path(O.__file__).read_text(encoding='utf-8'); tree = ast.parse(src); where = set()
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for n in ast.walk(fn):
            if isinstance(n, ast.Attribute) and n.attr in ('cubeA', 'cubeB', 'evaluate', 'get_state_dict') and n.attr != 'get_state_dict': where.add(fn.name)
    assert where == {'oracle_offset'}
    assert 'policies' not in [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names] and 'convert' not in src and 'state_parts' not in src


def test_policy_observation_is_the_same_whitelist_in_every_arm():
    """What the policy holds after any arm has the keys of the whitelisted observation and nothing else."""
    for arm in O.ARMS:
        r, s = at_trigger(); O.apply_arm(r, arm, 4, O.OracleGate())
        if not r.done: r.step_base()
        for h in r.policy.history: assert set(h) == {'agent', 'extra'} and set(h['agent']) == {'qpos', 'qvel'} and set(h['extra']) == {'tcp_pose'}


# ---------------------------------------------------------------- CENTER_ORACLE bounds, direction, units
@pytest.mark.parametrize('offset', [(.02, 0.), (-.02, 0.), (0., .025), (0., -.025), (.015, -.015)])
def test_center_moves_the_gripper_towards_removing_the_offset(offset):
    r, s = at_trigger(); e = r.env; e.B[0, :2] = e.A[0, :2] + torch.tensor(offset); start = e.p[0, :2].clone()
    O.center_oracle(r, O.OracleGate()); moved = (e.p[0, :2] - start).numpy(); info = r.retreat_info
    assert moved @ np.array(offset) > 0 and abs(moved[0] * offset[1] - moved[1] * offset[0]) < 1e-6                                   # along the offset, not against it and not sideways
    assert info['offset_after_norm_m'] < info['offset_before_norm_m'] and info['offset_before_norm_m'] == pytest.approx(math.hypot(*offset), abs=1e-6)
    assert e.p[0, 2] == pytest.approx(.1) and info['tcp_dz_m'] == 0                                                 # no vertical command


def test_center_command_units_and_bounds():
    """5 mm commanded = 0.05 in normalised action (0.1 m per unit); at most 12 steps; the target never leaves 3 cm; the gripper command is held."""
    r, s = at_trigger(); e = r.env; e.B[0, :2] = e.A[0, :2] + torch.tensor([.06, 0.]); n0 = len(r.actions); grip = float(r.policy.last_action[-1])
    O.center_oracle(r, O.OracleGate()); acts = np.array(r.actions[n0:]); info = r.retreat_info
    assert len(acts) == O.MAX_STEPS == info['steps_used'] and info['end_reason'] == 'step_budget'
    assert np.allclose(np.linalg.norm(acts[:, :2], axis=1), .05, atol=1e-6) and (acts[:, 2] == 0).all() and (acts[:, 3] == grip).all() and grip < 0
    assert info['tcp_xy_displacement_m'] == pytest.approx(12 * .005 * GAIN, abs=1e-6) and 12 * .005 * GAIN < O.TARGET_CAP_M      # the mock arm achieves 40 % of each command
    assert info['commanded_path_m'] == pytest.approx(.06) and not info['reached'] and len(r.policy.queue) == 0


def test_center_target_cap_binds_when_the_arm_tracks_fully():
    r, s = at_trigger(); e = r.env; e.gain = 1.; e.B[0, :2] = e.A[0, :2] + torch.tensor([.06, 0.])
    O.center_oracle(r, O.OracleGate()); info = r.retreat_info
    assert info['end_reason'] == 'target_cap' and info['steps_used'] == 6 and info['tcp_xy_displacement_m'] == pytest.approx(O.TARGET_CAP_M, abs=1e-6)
    assert info['offset_after_norm_m'] == pytest.approx(.03, abs=1e-6)


def test_center_never_commands_a_target_outside_the_cap_even_from_a_drifted_tcp():
    """Review finding: a measured TCP outside the cap could have been given a 5 mm step whose target is still outside."""
    class Pushed(PlaceEnv):
        def step(self, a):
            out = super().step(a); self.p[0, 0] += .012; return (self.obs(), *out[1:])     # an external push of 12 mm per step along +x
    r, s = at_trigger(); e = r.env; e.__class__ = Pushed; e.B[0, :2] = e.A[0, :2] + torch.tensor([.08, 0.])
    O.center_oracle(r, O.OracleGate()); info = r.retreat_info
    assert info['end_reason'] == 'target_cap' and info['tcp_xy_displacement_m'] > O.TARGET_CAP_M and 0 < info['steps_used'] < O.MAX_STEPS
    assert all(x['target_distance_from_start_m'] <= O.TARGET_CAP_M + 1e-9 for x in info['steps'])


def test_center_stops_at_the_residual_tolerance():
    r, s = at_trigger(); e = r.env; e.B[0, :2] = e.A[0, :2] + torch.tensor([.018, 0.])
    O.center_oracle(r, O.OracleGate()); info = r.retreat_info
    assert info['end_reason'] == 'within_tolerance' and info['offset_after_norm_m'] <= O.RESIDUAL_TOL_M and info['steps_used'] == 4      # 18 mm - 4 x 2 mm = 10 mm


def test_zero_step_case_clears_the_queue_and_replans_identically_in_both_arms():
    r, s = at_trigger(); e = r.env; e.B[0, :2] = e.A[0, :2] + torch.tensor([.008, 0.]); t = r.t; n0 = len(r.actions)
    O.center_oracle(r, O.OracleGate()); assert r.retreat_info['steps_used'] == 0 and r.t == t and len(r.actions) == n0 and not r.policy.queue and r.intervention == 'center_oracle_xy'
    a = r.step_base(); seed_a = r.policy.last_seed
    r2, _ = at_trigger(); r2.env.B[0, :2] = r2.env.A[0, :2] + torch.tensor([.008, 0.]); O.hold_matched(r2, 0); assert r2.t == t and not r2.policy.queue
    b = r2.step_base(); assert np.array_equal(a, b) and seed_a == r2.policy.last_seed == seed_for(r.episode, t, 0, 'dp')


def test_hold_matched_holds_xy_for_the_given_steps_without_reading_poses():
    r, s = at_trigger(); start = r.env.p.clone(); t = r.t; r.env.reads = 0; grip = float(r.policy.last_action[-1])
    O.hold_matched(r, 7); info = r.retreat_info
    assert r.t == t + 7 and info['steps_used'] == 7 and info['xy_drift_max_m'] == 0 and torch.equal(r.env.p, start) and r.env.reads == 0 and info['gripper_command'] == grip
    with pytest.raises(ValueError): O.hold_matched(at_trigger()[0], O.MAX_STEPS + 1)


def test_arms_refuse_a_state_whose_held_gripper_command_is_open():
    r = make().reset(1); r.step_base()                                           # the first executed command of this mock policy is "open"
    assert r.policy.last_action[-1] > 0
    for f in (lambda: O.retreat_r5(r), lambda: O.hold_matched(r, 2), lambda: O.center_oracle(r, O.OracleGate())):
        with pytest.raises(RuntimeError, match='not a close command'): f()
    assert r.intervention is None


def test_r5_is_the_existing_retreat_unchanged():
    """On a state that is on the Stage-2 eligibility grid, retreat_r5 and Runtime.force(2) give the same run."""
    r = make().reset(3)
    while not (r.eligible()[2] and r.policy.last_action is not None and r.policy.last_action[-1] < 0): r.step_base()
    s = r.snapshot(); r.force(2); a = r.complete(); ref = pilot._ref(r); info = clone(r.retreat_info)
    r.restore(s); O.retreat_r5(r); b = r.complete()
    drop = lambda d: {k: v for k, v in d.items() if k != 'policy_inference_seconds'}                      # wall-clock time
    assert pilot._same_run(ref, r, CFG['audit'])['match'] and drop(a) == drop(b) and info == r.retreat_info and info['mode'] == 'up' and info['commanded_distance_m'] == pytest.approx(.05)
    assert 'self.retreat()' in inspect.getsource(Runtime.force) and 'r.retreat()' in inspect.getsource(O.retreat_r5)


# ---------------------------------------------------------------- same state, same noise, handoff
def test_c0_from_the_snapshot_reproduces_the_trunk(recs):
    for r in [r for r in recs if r['triggered']]:
        c0 = cell(r, 'C0'); assert r['checks']['c0_reproduces_trunk']['match'] and (c0['y'], c0['steps']) == (r['trunk']['success'], r['trunk']['steps'])
        assert c0['intervention'] is None and c0['control'] is None


def test_hold_and_center_return_at_the_same_step_with_the_same_seed(recs):
    seen = 0
    for r in [r for r in recs if r['triggered']]:
        t = r['trigger']['step']; T = r['reference']['t_center']; h = cell(r, 'HOLD_MATCHED'); c = cell(r, 'CENTER_ORACLE'); assert r['handoff']['pass']
        assert h['control']['steps_planned'] == T == c['control_steps'] and h['control_inference_calls'] == c['control_inference_calls'] == 0
        if h['policy_resumed'] and c['policy_resumed']:
            seen += 1; fh = h['inference_calls'][0]; fc = c['inference_calls'][0]
            assert fh['t'] == fc['t'] == t + T and fh['seed'] == fc['seed'] == seed_for(r['scene'], t + T, 0, 'dp') and h['queue_len_at_return'] == c['queue_len_at_return'] == 0
            assert h['history_len_at_return'] == c['history_len_at_return']
    assert seen >= 30


def test_every_inference_uses_repeat_zero_and_the_step_key(recs):
    for r in recs:
        for calls in [r['trunk']['inference_calls']] + [x.get('inference_calls', []) for x in r['cells']]:
            assert all(k['policy_noise_repeat_id'] == 0 and k['seed'] == seed_for(r['scene'], k['t'], 0, 'dp') for k in calls)


def test_arm_order_does_not_change_any_result():
    base, _ = scene(7); assert base['triggered']
    for order in list(itertools.permutations(O.ARMS))[1::5]:
        rec, _ = scene(7, order=order)
        for a in O.ARMS:
            x, y = cell(base, a), cell(rec, a); assert (x['y'], x['steps'], x['control_steps'], x['inference_calls']) == (y['y'], y['steps'], y['control_steps'], y['inference_calls'])


def test_restore_is_prefix_replay_never_a_state_write():
    class Spy(PlaceEnv):
        writes = 0
        def set_state_dict(self, s): Spy.writes += 1
    rec, _ = scene(7, env=Spy()); assert rec['triggered'] and rec['counts']['restores'] == 5 and Spy.writes == 0


def test_a_faulty_restore_stops_the_scene():
    class Forgetful(PlacePolicy):
        def restore(self, s): super().restore(s); self.queue = []
    with pytest.raises(RuntimeError, match='restored runtime differs'): scene(7, policy=Forgetful())


def test_center_that_differs_from_its_reference_stops_the_scene(monkeypatch, recs):
    real = O.center_oracle; n = {'k': 0}; moved = next(r['scene'] for r in recs if r['triggered'] and r['reference']['t_center'] > 0)
    still = next(r['scene'] for r in recs if r['triggered'] and r['reference']['t_center'] == 0)

    def flaky(r, gate):
        n['k'] += 1
        if n['k'] == 2: r.env.B[0, 0] += .004                                      # the second run sees another target: a technical fault
        real(r, gate)
    monkeypatch.setattr(O, 'center_oracle', flaky)
    with pytest.raises(RuntimeError, match='differs from its control-only reference'): scene(moved)
    n['k'] = 0
    with pytest.raises(RuntimeError, match='differs from its control-only reference'): scene(still)       # also when no step was taken


def test_no_arm_gets_bonus_steps(recs):
    assert all(x['steps'] <= CFG['env']['horizon'] for r in recs for x in r['cells'])
    late = [r for r in recs if r['triggered'] and any(x['reason'] == 'horizon' for x in r['cells'])]
    assert all(x['steps'] == CFG['env']['horizon'] for r in late for x in r['cells'] if x['reason'] == 'horizon')


# ---------------------------------------------------------------- the mock shows the intended contrast and its limit
def test_mock_contrast_center_rescues_some_scenes_and_hold_does_not(recs):
    s = O.summarize(recs, range(1, 41), True); p = s['primary_CENTER_ORACLE_vs_C0']
    assert p['rescue'] >= 1 and s['secondary']['HOLD_MATCHED_vs_C0']['rescue'] == 0
    undone = [r['scene'] for r in recs if r['triggered'] and r['reference']['offset_after_norm_m'] <= O.RESIDUAL_TOL_M and r['reference']['t_center'] > 0 and not cell(r, 'CENTER_ORACLE')['y']]
    assert undone                                   # after a correction the re-planning policy can move back before it opens: alignment is not success


# ---------------------------------------------------------------- accounting and analysis
def test_validate_scene_rejects_tampering(recs):
    r0 = next(r for r in recs if r['triggered'] and r['reference']['t_center'] > 0)
    def bad(f):
        r = copy.deepcopy(r0); f(r)
        with pytest.raises(ValueError): O.validate_scene(r, CFG)
    bad(lambda r: r['cells'].pop())
    bad(lambda r: r['cells'][0].update(y=1 - r['cells'][0]['y']))
    bad(lambda r: r['cells'][2].update(oracle_reads=3))
    bad(lambda r: r['cells'][3].update(oracle_reads=0))
    bad(lambda r: r['cells'][3]['control'].update(gripper_command=1.))
    bad(lambda r: r['cells'][2]['control'].update(steps_planned=r['reference']['t_center'] + 1))
    bad(lambda r: r['cells'][3]['control']['steps'][0].update(commanded_xy_m=[.006, 0.]))
    bad(lambda r: r['cells'][3]['control']['steps'][0].update(target_distance_from_start_m=.031))
    bad(lambda r: r['cells'][3]['control']['steps'].pop())
    bad(lambda r: r['cells'][1].update(steps=CFG['env']['horizon'] + 1))
    bad(lambda r: r['handoff'].update({'pass': False}))


def test_paired_table_and_denominators():
    ref = {1: 1, 2: 1, 3: 0, 4: 0, 5: 0}; arm = {1: 1, 2: 0, 3: 1, 4: 1, 5: 0}; p = O.paired(ref, arm)
    assert (p['both_succeed'], p['harm'], p['rescue'], p['both_fail'], p['net']) == (1, 1, 2, 1, 1) and p['harm_given_reference_success'] == .5
    assert p['rescue_scenes'] == [3, 4] and p['harm_scenes'] == [2] and O.paired({1: 0}, {1: 0})['harm_given_reference_success'] == 'NA'
    with pytest.raises(ValueError): O.paired({1: 1}, {2: 1})
    lo, hi = O.wilson(79, 100); assert lo == pytest.approx(.7002, abs=2e-3) and hi == pytest.approx(.8583, abs=2e-3)


def test_summary_keeps_every_scene_and_refuses_a_partial_run(recs):
    s = O.summarize(recs, range(1, 41), True); d = s['denominators']
    assert d['all'] == 40 == d['no_trigger'] + d['triggered'] and d['triggered'] == d['actually_moved_T_gt_0'] + d['already_aligned_T_eq_0']
    assert all(v['n'] == 40 for v in s['success'].values()) and s['primary_CENTER_ORACLE_vs_C0']['n'] == 40
    with pytest.raises(ValueError): O.summarize(recs[:-1], range(1, 41), True)
    with pytest.raises(ValueError): O.summarize(recs + [recs[0]], range(1, 41), True)


def test_followup_gate_needs_all_five():
    ok = {'rescue': 5, 'net': 4, 'harm': 1}
    assert O.followup_gate(ok, 84, 80, True)['label'] == 'ORACLE_GATE_PASS'
    for p, c, h, a in (({**ok, 'rescue': 3}, 84, 80, True), ({**ok, 'net': 2}, 84, 80, True), ({**ok, 'harm': 4}, 84, 80, True), (ok, 80, 80, True), (ok, 84, 80, False)):
        assert O.followup_gate(p, c, h, a)['label'] == 'NO_ORACLE_GATE_PASS'


# ---------------------------------------------------------------- tracking audit, seeds, cost
def test_tracking_probe_measures_gain_direction_and_shortfall():
    out = []
    for target in O.PROBE_TARGETS_M:
        r, s = at_trigger(); r.env.reads = 0; x = O.tracking_probe(r, target); assert r.env.reads == 0; out.append({'scene': 3, **x})
        assert x['direction_cosine'] == pytest.approx(1.) and x['gain_median'] == pytest.approx(GAIN, abs=1e-6) and x['tcp_dz_m'] == 0
        assert .02 - 1e-6 <= x['achieved_norm_m'] <= .022 + 1e-6 and x['steps_used'] in (10, 11)           # 3 cm target, 1 cm tolerance, 2 mm per step
    assert O.tracking_verdict(out, 5)['pass'] and O.tracking_verdict(out, 2)['status'] == 'TECHNICAL_BLOCKED'
    wrong = [{**out[0], 'direction_cosine': -1.}]; assert not O.tracking_verdict(wrong, 5)['pass']
    weak = [{**out[0], 'achieved_norm_m': .004}]; assert not O.tracking_verdict(weak, 5)['pass']
    sag = [{**out[0], 'tcp_dz_m': -.008}]; assert not O.tracking_verdict(sag, 5)['pass']


def test_pilot_seeds_are_the_frozen_unused_range():
    m = read_json('configs/recovery_v1_splits.json'); ids = O.pilot_seeds(m, {'x': [1, 2], 'diag': m['splits']['recovery_v1_diag']})
    assert ids == list(range(50700, 50800))
    with pytest.raises(ValueError, match='recorded use'): O.pilot_seeds(m, {'x': [50750]})
    bad = copy.deepcopy(m); bad['splits'][O.SET] = bad['splits'][O.SET][:-1]
    with pytest.raises(ValueError, match='frozen range'): O.pilot_seeds(bad, {})


def test_cost_projection_uses_the_worst_case_and_refuses_above_the_cap():
    assert O.STEPS_PER_SCENE_BOUND == 1212
    d = O.projected_cost(seconds_per_env_step=.08, usd_per_hour=.42, spent_usd=.3, cap_usd=2.5); hours = (100 * 1212 * .08 + 600) * 1.25 / 3600
    assert d['projected_remaining_hours'] == pytest.approx(hours) and d['projected_total_usd'] == pytest.approx(.3 + hours * .42 + .2) and d['fits_cap']
    assert not O.projected_cost(seconds_per_env_step=.2, usd_per_hour=.6, spent_usd=.3, cap_usd=2.5)['fits_cap']


def test_declared_constants_are_the_requested_bounds():
    assert (O.RESIDUAL_TOL_M, O.STEP_CMD_M, O.TARGET_CAP_M, O.MAX_STEPS, O.REPEAT) == (.010, .005, .030, 12, 0)
    assert O.ARMS == ('C0', 'R5', 'HOLD_MATCHED', 'CENTER_ORACLE') and O.RANGE == (50700, 50799) and O.GATE == {'rescue_min': 4, 'net_min': 3, 'harm_max': 3}


# ---------------------------------------------------------------- driver: approval gate, ledger, files, partial runs
import argparse, gzip, importlib.util, json, re


def driver():
    spec = importlib.util.spec_from_file_location('oxy_driver', 'scripts/31_recovery_v1_oracle_xy.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture(scope='module')
def X():
    return driver()


def test_every_server_command_refuses_without_an_explicit_authorization(X, tmp_path):
    plan = tmp_path / 'plan.json'; plan.write_text('{"code": {}}'); run = tmp_path / 'run'; run.mkdir()
    ns = lambda: argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu', usd_per_hour=.4, spent_usd=0.)
    had_ledger = Path(X.LEDGER).exists()                         # since the pilot was executed (2026-11-17) the machine that ran it holds the ledger; a refused command must not create or touch one
    for cmd in (X.cmd_scenes, X.cmd_audit, X.cmd_budget, X.cmd_run):
        with pytest.raises(SystemExit, match='no authorization'): cmd(ns())
    auth = {'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 2.5, 'plan_sha256': '0' * 64}
    for bad in (auth, {**auth, 'plan_sha256': X.sha256(plan), 'cap_usd': None}, {**auth, 'plan_sha256': X.sha256(plan), 'approved_by': 'agent'}, {**auth, 'plan_sha256': X.sha256(plan), 'paid_compute_authorized': 'yes'}):
        (run / 'authorization.json').write_text(json.dumps(bad))
        for cmd in (X.cmd_scenes, X.cmd_audit, X.cmd_budget, X.cmd_run):
            with pytest.raises(SystemExit, match='Refused'): cmd(ns())
    assert not [p for p in run.iterdir() if p.is_dir()] and Path(X.LEDGER).exists() == had_ledger      # iterdir, not glob('*/'): before Python 3.13 that pattern also returns files (server run, attempt 1)


def test_a_plan_that_failed_its_preparation_is_refused_even_with_an_authorization(X, tmp_path):
    for body in ({'code': {}, 'pass': False, 'problems': ['x'], 'work_id': X.WORK}, {'code': {}, 'pass': True, 'problems': [], 'work_id': 'other'}, {'code': {}}):
        plan = tmp_path / 'plan.json'; plan.write_text(json.dumps(body)); run = tmp_path / 'run'; run.mkdir(exist_ok=True)
        (run / 'authorization.json').write_text(json.dumps({'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 2.5, 'plan_sha256': X.sha256(plan)}))
        for cmd in (X.cmd_scenes, X.cmd_audit, X.cmd_run):
            with pytest.raises(SystemExit, match='did not pass its own preparation checks'): cmd(argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu', usd_per_hour=.4, spent_usd=0.))


def test_the_pilot_re_evaluates_the_budget_with_its_own_figures(X):
    run = inspect.getsource(X.cmd_run)
    assert run.index('O.projected_cost(') < run.index('write_json(LEDGER') and "a.usd_per_hour < bg['inputs']['usd_per_hour']" in run and "a.spent_usd < bg['inputs']['spent_usd']" in run


def test_no_code_writes_an_authorization_and_the_ledger_precedes_the_first_rollout(X):
    for f in ('scripts/31_recovery_v1_oracle_xy.py', 'src/hci/recovery_v1_oracle.py', 'scripts/recovery_v1_oracle/run.sh'):
        src = Path(f).read_text(encoding='utf-8')
        assert not re.search(r"(write_json|open|write_text)\([^)\n]*authorization", src) and 'RUN_FROZEN_TEST' not in src and '|| true' not in src and 'set_state' not in src.replace('set_state is never used', '')
    run = inspect.getsource(X.cmd_run); assert run.index('require_authorization') < run.index('write_json(LEDGER') < run.index('V.runtime(') < run.index('O.collect_scene')
    assert "'x'" not in run and 'exist_ok=True' not in run                       # exclusive creation: write_json refuses to overwrite


def test_server_file_list_is_complete_and_present(X):
    assert all(Path(f).exists() for f in X.SERVER_READS)
    for f in ('src/hci/recovery_v1_oracle.py', 'src/hci/pilot2b.py', 'src/hci/pilot2d.py', 'src/hci/pilot.py', 'scripts/30_recovery_v1_diag.py', 'scripts/recovery_v1_oracle/run.sh', X.AUDIT_REFERENCE, X.KEYS): assert f in X.SERVER_READS
    assert set(X.D.SERVER_READS) <= set(X.SERVER_READS) | {'scripts/recovery_v1/run.sh'}
    imports = set(re.findall(r'^from \.(\w+) import', Path('src/hci/recovery_v1_oracle.py').read_text(encoding='utf-8'), re.M))
    assert {f'src/hci/{m}.py' for m in imports} <= set(X.CODE)


def test_shell_driver_matches_the_script_and_needs_the_gates():
    sh = Path('scripts/recovery_v1_oracle/run.sh').read_text(encoding='utf-8'); assert '\r' not in sh
    for word in ('dry-run --plan', 'scenes --plan', 'audit --plan', 'budget --plan', 'run --plan'): assert word in sh
    assert sh.index('authorization.json ]] ||') < sh.index('case "$1"') and 'grep -q "^BUDGET_OK"' in sh and 'grep -q "^AUDIT_DONE"' in sh and 'grep -q "^SCENES_DONE"' in sh and 'grep -q "^ENV_DONE"' in sh
    assert sh.count('--usd-per-hour "${USD_PER_HOUR:?}" --spent-usd "${SPENT_USD:?}"') == 2


def test_audit_seeds_are_used_development_scenes_not_pilot_scenes(X):
    v = read_json('configs/vision_v1_splits.json')['splits']['vision_v1_select']; assert set(X.AUDIT_SEEDS) <= set(v) and not set(X.AUDIT_SEEDS) & set(range(50700, 50800))
    assert 'recovery_v1/recovery_v1_diag' in X.used_ledger() and O.pilot_seeds(read_json(X.MANIFEST), X.used_ledger()) == list(range(50700, 50800))
    scan = X.recorded_use_scan(); assert not {f for f in scan['files_naming_a_pilot_seed'] if not f.startswith(('results/vision_center_001/run2/', 'results/vision_center_v2_001/preparation/'))}      # later work on these scenes as used development data: the Vision-CENTER shadow pass (run2, stage B1) and the offline v2 observability audit of its records
    if scan['pilot_ledger_exists']: assert read_json(X.LEDGER)['plan_sha256'] == 'b7078afbe705a5ffd8a7128951f9c9e7618bb7c2fed5c044a5474c2fc895730b'      # the one executed pass; before it, no ledger existed
           # the server holds only part of results/, so no file count is asserted


def test_summary_from_raw_records_and_refusal_of_a_partial_run(X, tmp_path):
    d = tmp_path / 'pilot'; (d / 'scenes').mkdir(parents=True); plan = tmp_path / 'plan.json'; plan.write_text('{}'); audit = tmp_path / 'audit.json'; audit.write_text('{"pass": true}')
    for s in range(50700, 50800):
        rec, traces = scene(s); tp = X.save_traces(d, s, traces); rec['trace'] = tp.relative_to(d).as_posix(); rec['trace_sha256'] = X.sha256(tp); X.write_json(d / 'scenes' / f'{s}.json', rec)
    ns = argparse.Namespace(pilot=str(d), plan=str(plan), audit=str(audit), out=str(tmp_path / 'result.json')); X.cmd_summarize(ns); res = read_json(tmp_path / 'result.json')
    assert res['scenes'] == 100 == res['denominators']['all'] and res['oracle_reads_outside_center'] == 0 and res['primary_CENTER_ORACLE_vs_C0']['n'] == 100
    with gzip.open(d / 'traces' / 'trace_50700.json.gz', 'rt', encoding='utf-8') as f: assert set(json.load(f)['traces']) <= {'trunk', 'reference', *O.ARMS}
    ns.out = str(tmp_path / 'again.json'); (d / 'scenes' / '50750.error.json').write_text('{}')
    with pytest.raises(SystemExit, match='INCOMPLETE'): X.cmd_summarize(ns)
    (d / 'scenes' / '50750.error.json').unlink(); (d / 'scenes' / '50799.json').unlink()
    with pytest.raises(SystemExit, match='INCOMPLETE'): X.cmd_summarize(ns)
    assert not (tmp_path / 'again.json').exists()
