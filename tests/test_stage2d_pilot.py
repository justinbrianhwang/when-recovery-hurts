"""Stage 2D: the two matched control arms, the T_ref reference, the handoff, accounting and tables.

Mock physics only. These tests say nothing about ManiSkill, the GPU or the research policy.
"""
import copy, inspect, itertools
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from test_runtime_mock import MockPolicy
from test_stage2a_pilot import GoalEnv, StaleObsEnv, ForgetfulQueuePolicy, CFG, CPS, REPS
from hci.simulation import Runtime
from hci import pilot, pilot2d as p2
from hci.io import read_json, seed_for

R5, R2M, H0M = p2.CONTROL


class SeedPolicy(MockPolicy):
    def generate(self, episode, t, repeat=0):
        self.last_seed = seed_for(episode, t, repeat); return super().generate(episode, t, repeat)


class NoSuccessEnv(GoalEnv):
    goal = 1e9


class LiftEnv(GoalEnv):
    """Success as soon as z rises `rise` above `base` (set by the test): the episode ends inside a control part."""
    goal = 1e9; base = None; rise = .012

    def _info(self):
        return {'success': torch.tensor([bool(self.base is not None and self.p[0, 2] > self.base + self.rise)])}


class LagEnv(NoSuccessEnv):
    """The arm achieves only 40 % of each commanded increment."""
    def step(self, a):
        b = a.clone(); b[:, :3] *= .4; return super().step(b)


class DriftEnv(NoSuccessEnv):
    """A constant pull in -z on every step (a carried load in the mock): holding needs feedback."""
    def step(self, a):
        out = super().step(a); self.p[0, 2] -= .001; return (self.obs(), *out[1:])


class SpyEnv(NoSuccessEnv):
    writes = 0

    def set_state_dict(self, s):
        SpyEnv.writes += 1; super().set_state_dict(s)


def make(env=None, policy=None, cfg=CFG):
    return Runtime(env or GoalEnv(), policy or SeedPolicy(), cfg)


def probe(rt, z):
    p = rt.env.p[0].tolist()
    return {'t': rt.t, 'tcp': p, 'cubeA': [0, 0, .02], 'cubeB': [.1, 0, .02], 'tcp_to_cubeA': 1., 'cubeA_to_cubeB_xy': .1, 'is_cubeA_grasped': False,
            'is_cubeA_on_cubeB': False, 'is_cubeA_static': True, 'success': bool(rt.y), 'action': None if z is None else [float(x) for x in z],
            'gripper_qpos': [.04, .04]}


def scene(seed, **kw):
    return p2.collect_scene(make(kw.pop('env', None), kw.pop('policy', None), kw.pop('cfg', CFG)), seed, CFG, CPS, REPS, **kw)[0]


@pytest.fixture(scope='module')
def recs():
    return [scene(s) for s in range(1, 13)]


def at_checkpoint(seed=1, t=12, env=None, cfg=CFG):
    r = make(env, cfg=cfg).reset(seed)
    for _ in range(t): r.step_base()
    return r, r.snapshot()


def cells_of(rec):
    return [(b, {c['arm']: c for c in b['cells']}) for b in rec['blocks']]


# 1
def test_r5_is_the_existing_retreat_unchanged(recs):
    rec = recs[0]; r = make().reset(rec['scene']); n = 0
    snaps = {}
    while not r.done:
        if r.t in CPS and r.eligible().all(): snaps[r.t] = r.snapshot()
        r.step_base()
    for b, cell in cells_of(rec):
        r.restore(snaps[b['checkpoint']]); r.repeat = b['repeat']; r.force(2); info = copy.deepcopy(r.retreat_info); res = r.complete(); x = cell['R5']; n += 1
        assert (x['y'], x['steps'], x['reason'], x['intervention']) == (res['success'], res['steps'], res['reason'], 2)
        assert x['control_steps'] == info['steps_used'] == b['t_ref'] and x['control']['end_reason'] == info['end_reason']
        assert x['control']['achieved_displacement_m'] == info['achieved_displacement_m'] and x['same_as_reference']['pass']
    assert n == len(rec['blocks']) > 0


# 2, 5
def test_r2m_holds_after_an_early_arrival_and_all_three_arms_resume_at_the_same_step_with_the_same_seed(recs):
    n = 0
    for rec in recs:
        for b, cell in cells_of(rec):
            t, T = b['checkpoint'], b['t_ref']; x = cell['R2M']; n += 1
            assert x['control']['reached'] and x['control']['first_arrival_step'] < T and x['control']['hold_steps'] == T - x['control']['first_arrival_step'] > 0
            firsts = {(cell[k]['inference_calls'][0]['t'], cell[k]['inference_calls'][0]['seed']) for k in ('R5', 'R2M', 'H0M') if cell[k]['policy_resumed']}
            assert firsts == {(t + T, seed_for(rec['scene'], t + T, b['repeat']))} and b['handoff']['matched_handoff'] is True
            for k in ('R5', 'R2M', 'H0M'):
                assert cell[k]['control_inference_calls'] == 0 and cell[k]['first_policy_step'] == t + T and cell[k]['queue_len_at_return'] == 0
                assert all(c['t'] >= t + T for c in cell[k]['inference_calls'])
            assert cell['C0']['first_inference_step'] in (None, t + 4) and cell['C0']['control'] is None
    assert n > 0


# 3
def test_h0m_follows_the_fixed_snapshot_position_and_never_writes_state():
    r, s = at_checkpoint(env=SpyEnv()); SpyEnv.writes = 0; start = r.tcp_history[-1].copy(); n0 = len(r.actions)
    p2.matched_control(r, H0M, 5); i = r.retreat_info
    assert SpyEnv.writes == 0 and i['anchor_tcp'] == start.tolist() and i['move_steps'] == 0 and i['hold_steps'] == i['steps_used'] == 5 and r.t == 17
    assert all(np.all(a[:3] == 0) for a in r.actions[n0:]) and np.array_equal(r.tcp_history[-1], start) and i['hold_drift_max_m'] == 0
    # with a pull on the arm the target stays the snapshot position: the commands push back and the measured drift is reported, not hidden
    r, s = at_checkpoint(env=DriftEnv()); start = r.tcp_history[-1].copy(); n0 = len(r.actions); p2.matched_control(r, H0M, 5); i = r.retreat_info
    acts = r.actions[n0:]; assert np.all(acts[0][:3] == 0) and all(a[2] > 0 for a in acts[1:]) and i['anchor_tcp'] == start.tolist()
    assert 0 < i['hold_drift_max_m'] <= .002 + 1e-6 and i['achieved_displacement_m'] > 0
    # a zero command is not a hold: the same pull without feedback moves five times as far
    r, s = at_checkpoint(env=DriftEnv()); z = float(r.tcp_history[-1][2])
    for _ in range(5): r.advance(np.zeros(4, np.float32))
    assert abs((z - float(r.tcp_history[-1][2])) - .005) < 1e-6


# 4
def test_the_gripper_command_of_the_snapshot_is_held_in_every_control_step():
    rec, traces, _, _ = p2.collect_scene(make(), 1, CFG, CPS, REPS, probe=probe); col = p2.TRACE_COLUMNS.index('a_gripper'); n = 0
    snap = {s['checkpoint']: s for s in rec['snapshots']}
    for b, cell in cells_of(rec):
        for k in ('R5', 'R2M', 'H0M'):
            rows = traces[f"b_{b['checkpoint']}_{b['repeat']}_{k}"]; g = np.float32(snap[b['checkpoint']]['last_gripper_command']); n += 1
            T = b['t_ref']; assert cell[k]['control_steps'] == T >= 5 and np.all(rows[:T, col] == g) and cell[k]['control']['gripper_command'] == float(g)
    assert n > 0 and len({s['last_gripper_command'] for s in snap.values()}) > 1


# 6
def test_a_target_that_is_not_reached_is_reported_and_gets_no_extra_time():
    cfg = {**CFG, 'retreat': {**CFG['retreat'], 'max_steps': 6}}; r, s = at_checkpoint(env=LagEnv(), cfg=cfg)
    r.force(2); ref = copy.deepcopy(r.retreat_info); assert ref['end_reason'] == 'budget_exhausted' and ref['steps_used'] == 6 and not ref['reached']
    r.restore(s); calls = r.policy.inference_calls; p2.matched_control(r, R2M, 6); i = r.retreat_info
    assert not i['reached'] and i['first_arrival_step'] is None and i['anchor_tcp'] is None and i['steps_used'] == i['move_steps'] == 6 and i['hold_steps'] == 0
    assert r.t == 18 and i['end_reason'] == 'matched_return' and i['achieved_displacement_m'] < i['commanded_distance_m'] and r.policy.inference_calls == calls
    rec = scene(1, env=LagEnv(), cfg=cfg)
    for b, cell in cells_of(rec):
        assert b['t_ref'] == 6 and not cell['R2M']['control']['reached'] and cell['R2M']['control_steps'] == 6 and cell['R2M']['first_policy_step'] == b['checkpoint'] + 6
        assert not cell['R5']['control']['reached'] and cell['R5']['control']['end_reason'] == 'budget_exhausted'
    with pytest.raises(ValueError): r.restore(s); p2.matched_control(r, R2M, 7)          # more than the Retreat budget is refused


def lift_seed():
    """A mock scene whose trunk reaches step 12 without having risen 1.2 cm above its height at step 12."""
    for seed in range(1, 200):
        r = make(env=NoSuccessEnv()).reset(seed); zs = [0.]
        for _ in range(12): r.step_base(); zs.append(float(r.tcp_history[-1][2]))
        if max(zs) < zs[-1] + .011: return seed, zs[-1]
    raise AssertionError('no usable mock scene')


# 7
def test_an_episode_that_ends_during_control_stops_at_once_and_is_kept():
    seed, z = lift_seed(); env = LiftEnv(); env.base = z; rec = p2.collect_scene(make(env=env), seed, CFG, [12], REPS)[0]
    s = rec['snapshots'][0]; ref = s['reference']; assert ref['reference_early_absorbing'] and ref['end_reason'] == 'episode_ended' and ref['t_ref'] == 3
    assert ref['ended_as']['y'] == 1 and len(rec['blocks']) == 4
    for b, cell in cells_of(rec):
        for k in ('R5', 'R2M'):          # both rise 1.5 cm in three steps: success inside the control part, nothing is executed afterwards
            x = cell[k]; assert x['early_absorbing'] and not x['policy_resumed'] and x['first_policy_step'] is None and not x['inference_calls']
            assert (x['y'], x['steps'], x['success_phase'], x['post_return_steps']) == (1, 15, 'control', 0)
        h = cell['H0M']; assert h['policy_resumed'] and h['control_steps'] == 3 and h['first_policy_step'] == 15      # same T_ref, own outcome
        assert b['handoff']['matched_handoff'] == 'NA' and b['handoff']['pass'] and b['reference_early_absorbing']
        assert b['y_R5'] == 1 and b['y_R2M'] == 1                                                                    # counted, not removed
    rows = p2.block_rows([rec]); assert len(rows) == 4 and sum(x['y_R5'] for x in rows) == 4
    assert p2.summarize([rec])['handoff']['matched_handoff_NA_reference_ended_early'] == 4


# 8
def test_control_steps_are_charged_to_the_same_horizon(recs):
    rec = scene(1, env=NoSuccessEnv())
    for b, cell in cells_of(rec):
        for x in cell.values(): assert x['steps'] == CFG['env']['horizon'] and x['reason'] == 'horizon'
        for k in ('R5', 'R2M', 'H0M'):
            x = cell[k]; assert x['control_steps'] == b['t_ref'] and x['post_return_steps'] == 60 - b['checkpoint'] - b['t_ref'] == cell['R5']['post_return_steps']
    for rec in recs:
        for b, cell in cells_of(rec):
            assert all(x['steps'] <= 60 and x['steps_after_checkpoint'] == x['steps'] - b['checkpoint'] for x in cell.values())


def strip(rec):
    return [[(c['experimental_arm'], c['y'], c['steps'], c['reason'], c['control_steps'], [(k['t'], k['seed'], k['chunk_sha256_16']) for k in c['inference_calls']])
             for c in b['cells']] for b in rec['blocks']] + [[s['reference']['t_ref'], s['reference']['commands_sha256']] for s in rec['snapshots']]


# 9
def test_results_do_not_depend_on_the_order_of_the_arms_and_reruns_are_identical():
    base = strip(scene(1)); assert len(base) == 15
    for order in list(itertools.permutations(p2.ARMS))[1::5]: assert strip(scene(1, order=order)) == base
    rec = scene(5, recheck=True); arms = [d['experimental_arm'] for d in rec['rechecks']]
    assert rec['reproducibility_ok'] and arms.count(R2M) == arms.count(H0M) == rec['counts']['snapshots'] > 0
    assert all(d['match'] and d['action_max_abs'] == 0 and d['inference_and_control_record_equal'] and d['repeat'] == 0 for d in rec['rechecks'])
    assert len(rec['no_intervention_duplicates']) == rec['counts']['snapshots'] and all(d['match'] for d in rec['no_intervention_duplicates'])
    with pytest.raises(RuntimeError): scene(1, policy=type('Leaky', (ForgetfulQueuePolicy, SeedPolicy), {})())


def test_every_arm_starts_from_the_same_restored_snapshot_and_the_reference_leaves_no_trace(monkeypatch):
    seen = {}; real = p2.apply_arm

    def spy(r, arm, rep, t_ref, on_phase=None):
        seen.setdefault((r.t, rep), {})[arm] = pilot.runtime_tree(r); real(r, arm, rep, t_ref, on_phase)
    monkeypatch.setattr(p2, 'apply_arm', spy); rec = scene(1)
    assert len(seen) == rec['counts']['blocks'] > 0 and rec['counts']['reference_control_runs'] == rec['counts']['snapshots']
    for trees in seen.values():
        assert set(trees) == set(p2.ARMS) and trees['continue']['discrete']['intervention'] is None and trees['continue']['discrete']['retreat_info'] is None
        for arm in p2.ARMS[1:]: assert pilot.compare_tree(trees['continue'], trees[arm], {'state_atol': 0., 'action_atol': 0., 'observation_atol': 0.})['pass']


# 10
def test_no_outcome_or_object_state_reaches_the_controller_or_the_saved_inputs():
    # the mock environment has no cube, no task flag and no success before the horizon: the control parts run on TCP positions alone
    r, s = at_checkpoint(env=NoSuccessEnv()); p2.matched_control(r, R2M, 5); assert not hasattr(r.env, 'cubeA')
    assert set(inspect.signature(p2.matched_control).parameters) == {'r', 'arm', 't_ref', 'on_phase'}
    src = inspect.getsource(p2.matched_control) + inspect.getsource(p2.apply_arm)
    for word in ('cube', 'success', 'evaluate', 'probe', 'set_state', '.y'): assert word not in src
    rec, _, _, inputs = p2.collect_scene(make(), 1, CFG, CPS, REPS, probe=probe)
    assert sorted(inputs) == [x['checkpoint'] for x in rec['snapshots']]
    for x in inputs.values(): assert set(x) == {'state_history', 'last_action', 'queue', 't', 'remaining_budget'}
    # T_ref is fixed before any branch of the snapshot exists and is the same for every repeat
    for s in rec['snapshots']: assert {b['t_ref'] for b in rec['blocks'] if b['checkpoint'] == s['checkpoint']} == {s['reference']['t_ref']}


# 11
def test_the_policy_resumes_on_the_real_history_with_an_empty_queue():
    r, s = at_checkpoint(); p2.apply_arm(r, R2M, 2, 5); a = r.actions[-1]
    assert r.repeat == 2 and r.policy.queue == [] and np.array_equal(r.policy.last_action, a) and len(r.policy.history) == 2 and r.intervention == 'matched_2cm'
    assert np.array_equal(pilot.as_numpy(r.policy.history[-1]['extra']['tcp_pose'])[0, :3], r.tcp_history[-1])          # the last real observation, not the old one
    assert not np.array_equal(pilot.as_numpy(r.policy.history[-1]['extra']['tcp_pose']), pilot.as_numpy(s['policy']['history'][-1]['extra']['tcp_pose']))
    calls = r.policy.inference_calls; r.step_base(); assert r.policy.inference_calls == calls + 1 and len(r.policy.queue) == 7
    with pytest.raises(ValueError): p2.matched_control(r, H0M, 5)                      # one intervention per episode, as before
    r.restore(s); assert r.intervention is None and r.retreat_info is None and len(r.policy.queue) == 4
    rec, traces, _, _ = p2.collect_scene(make(), 1, CFG, CPS, REPS, probe=probe); ph = p2.TRACE_COLUMNS.index('phase'); P = p2.PHASES.index
    for b, cell in cells_of(rec):
        t, T, rep = b['checkpoint'], b['t_ref'], b['repeat']; x = cell['R2M']['control']
        assert np.all(traces[f'b_{t}_{rep}_C0'][:, ph] == P('continue')) and np.all(traces[f'b_{t}_{rep}_R5'][:T, ph] == P('intervention_move'))
        assert np.all(traces[f'b_{t}_{rep}_H0M'][:T, ph] == P('intervention_hold')) and np.all(traces[f'b_{t}_{rep}_H0M'][T:, ph] == P('policy_after_return'))
        want = [P('intervention_move')] * x['move_steps'] + [P('intervention_hold')] * x['hold_steps']; assert list(traces[f'b_{t}_{rep}_R2M'][:T, ph]) == want
        assert np.array_equal(traces[f'ref_{t}'], traces[f'b_{t}_{rep}_R5'][:T]) and traces[f'ref_{t}'].shape == (T, len(p2.TRACE_COLUMNS))
    assert np.all(traces['trunk'][:, ph] == P('continue')) and np.all(traces['trunk'][:, -3:-1] == np.float32(.04))


# 12
def test_distance_duration_and_sample_are_not_parameters_of_the_run():
    assert p2.DISTANCE_M == {R2M: 0.02, H0M: 0.0} and p2.ARMS == ('continue', 'retreat_5cm', 'retreat_2cm_matched', 'hold_matched')
    names = set(inspect.signature(p2.collect_scene).parameters)
    assert not names & {'distance', 'distance_m', 't_ref', 'duration', 'tolerance', 'gate', 'outcome'}
    with pytest.raises(ValueError): p2.collect_scene(make(), 1, CFG, CPS, REPS, order=p2.ARMS[:3])
    with pytest.raises(ValueError): p2.matched_control(at_checkpoint()[0], 'retreat_5cm', 5)
    r, _ = at_checkpoint(t=13)                                                        # off the decision stride: eligibility is not loosened for the new arms
    with pytest.raises(ValueError): p2.matched_control(r, H0M, 5)


@pytest.mark.parametrize('bad', ['early_policy', 'longer_hold', 'other_seed', 'r5_changed'])
def test_a_wrong_control_arm_stops_the_scene(monkeypatch, bad):
    real = p2.apply_arm

    def wrong(r, arm, rep, t_ref, on_phase=None):
        if bad == 'longer_hold' and arm == H0M: return real(r, arm, rep, t_ref + 1, on_phase)
        if bad == 'r5_changed' and arm == R5 and rep == 2: r.c = {**r.c, 'retreat': {**r.c['retreat'], 'distance_m': .02}}
        real(r, arm, rep, t_ref, on_phase)
        if bad == 'r5_changed': r.c = CFG
        if bad == 'early_policy' and arm == R2M: r.policy.queue = [x.copy() for x in r.policy.generate(r.episode, r.t, r.repeat)]
        if bad == 'other_seed' and arm == H0M: r.repeat = rep + 1
    monkeypatch.setattr(p2, 'apply_arm', wrong)
    with pytest.raises((RuntimeError, ValueError)): scene(1)


def test_accounting_rejects_missing_duplicated_or_relabelled_cells(recs):
    rec = recs[0]; assert rec['counts']['continuations'] == 4 * rec['counts']['blocks'] == 16 * rec['counts']['snapshots']
    for i, change in ((1, {'intervention': None}), (2, {'t_ref': 4}), (2, {'control_inference_calls': 1}), (3, {'first_policy_step': 12}),
                      (0, {'queue_len_after_choice': 0}), (2, {'policy_noise_repeat_id': 1}), (3, {'control_steps': 6}), (1, {'steps': 61})):
        bad = copy.deepcopy(rec); bad['blocks'][0]['cells'][i].update(change)
        with pytest.raises(ValueError): p2.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); bad['blocks'][0]['cells'][2]['control']['commanded_distance_m'] = .03
    with pytest.raises(ValueError): p2.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); bad['blocks'][0]['cells'][2]['control']['gripper_command'] = 9.
    with pytest.raises(ValueError): p2.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); del bad['blocks'][-1]
    with pytest.raises(ValueError): p2.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); bad['snapshots'][0]['reference']['policy_inference_calls'] = 1
    with pytest.raises(ValueError): p2.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); bad['blocks'][0]['cells'][1], bad['blocks'][0]['cells'][2] = bad['blocks'][0]['cells'][2], bad['blocks'][0]['cells'][1]
    with pytest.raises(ValueError): p2.validate_scene(bad, CFG)


def test_a_failed_or_missing_scene_never_marks_the_data_complete(tmp_path):
    def run_scene(seed):
        if seed == 3: raise RuntimeError('simulator error')
        return p2.collect_scene(make(), seed, CFG, CPS, REPS, recheck=seed == 1)[:3]
    with pytest.raises(RuntimeError): p2.collect_scenes([1, 5, 3, 4], tmp_path, run_scene)
    assert sorted(p.name for p in (tmp_path / 'scenes').iterdir()) == ['1.json', '3.error.json', '5.json']
    assert p2.finalize(tmp_path, [1, 5, 3, 4], CFG, rerun_scenes=[1])[0]['completed'] is False
    assert p2.finalize(tmp_path, [1, 5], CFG, rerun_scenes=[1])[0]['completed'] is False          # the error file stays
    ok = tmp_path / 'ok'; p2.collect_scenes([1, 5], ok, run_scene); meta, got = p2.finalize(ok, [1, 5], CFG, rerun_scenes=[1])
    assert meta['completed'] is True and meta['continuations'] == 4 * meta['blocks'] and meta['reference_control_runs'] == meta['snapshots']
    assert p2.finalize(ok, [1, 5], CFG, rerun_scenes=[1, 5])[0]['completed'] is False             # scene 5 lacks the declared reruns
    assert p2.finalize(ok, [1, 5], CFG)[0]['completed'] is False                                  # scene 1 carries reruns nobody declared
    assert p2.finalize(ok, [1, 5, 6], CFG, rerun_scenes=[1])[0]['completed'] is False
    rec = read_json(ok / 'scenes' / '1.json'); lied = dict(rec, rechecks=[dict(d, match=False) for d in rec['rechecks']]); assert len(p2.repro_problems(lied, True)) == 2


def test_audit_runs_the_new_arms_checks_the_handoff_and_detects_a_bad_restore():
    res = p2.audit_scene(make(), 1, CFG, CPS); rep = p2.audit_report([res]); first = res['checkpoints'][0]
    assert res['pass'] and rep['pass'] and all(v > 0 for v in rep['branches_executed'].values())
    assert set(first['branches']) == {'refresh', 'reference_r5_control', R5, R2M, H0M} and first['branches'][R5]['same_as_reference']['pass']
    assert first['handoff']['pass'] and first['handoff']['matched_handoff'] is True and first['handoff']['R2M']['first_policy_step'] == 12 + first['branches']['reference_r5_control']['t_ref']
    for name in ('after_reference_r5_control', 'after_' + R5, 'after_' + R2M, 'after_' + H0M): assert first['checks'][name + '_trace']['pass'] and first['checks'][name + '_restored']['pass']
    assert first['input_saving_changes_nothing']['pass']
    assert not p2.audit_scene(make(policy=type('Leaky', (ForgetfulQueuePolicy, SeedPolicy), {})()), 1, CFG, CPS)['pass']
    bad = p2.audit_scene(make(env=StaleObsEnv()), 1, CFG, CPS); assert not bad['pass']
    with pytest.raises(RuntimeError): scene(1, env=StaleObsEnv())


def test_tables_follow_the_declared_definitions(recs):
    s = p2.summarize(recs); rows = p2.block_rows(recs); n = len(rows); d = {(x['arm'], x['reference']): x for x in s['pairs'] if x['checkpoint'] == 'all'}
    assert set(d) == set(p2.PAIRS) and [x['comparison'] for x in s['pairs'] if x['checkpoint'] == 'all'] == ['primary'] * 3 + ['secondary'] * 3
    for x in d.values(): assert x['blocks'] == n == x['0to1'] + x['1to0'] + x['1to1'] + x['0to0']
    keys = [(x['scene'], x['checkpoint'], x['block_repeat_id']) for x in rows]; assert len(keys) == len(set(keys)) == sum(r['counts']['blocks'] for r in recs)
    cells = {(b['scene'], b['checkpoint'], b['repeat']): {c['arm']: c['y'] for c in b['cells']} for r in recs for b in r['blocks']}
    for x in rows:
        y = cells[(x['scene'], x['checkpoint'], x['block_repeat_id'])]
        for a, ref in p2.PAIRS: assert x[f'{a}_vs_{ref}'] == f'{y[ref]}to{y[a]}'
    scenes = p2.scene_rows(recs); used = [x for x in scenes if x['in_conditional_summary']]; one = used[0]; mine = [x for x in rows if x['scene'] == one['scene']]
    assert abs(one['d_R2M_R5'] - np.mean([x['y_R2M'] - x['y_R5'] for x in mine])) < 1e-12 and one['n_0to1_R2M_R5'] == sum(x['R2M_vs_R5'] == '0to1' for x in mine)
    assert abs(s['scene_equal_weight']['d_H0M_R5']['mean_over_scenes'] - np.mean([x['d_H0M_R5'] for x in used])) < 1e-12
    q = s['scene_equal_weight']['d_R2M_H0M']; assert q['scenes_positive'] + q['scenes_negative'] + q['scenes_zero'] == len(used)
    empty = p2.scene_rows([dict(recs[0], blocks=[], snapshots=[], counts={**recs[0]['counts'], 'snapshots': 0})], [])[0]
    assert empty['d_R2M_R5'] == 'NA' and not empty['in_conditional_summary']
    refs = p2.reference_rows(recs); assert len(refs) == s['snapshots'] == s['t_ref']['snapshots'] and all(x['r5_runs_equal_to_reference'] == x['r5_runs_compared'] == 4 for x in refs)
    ho = p2.handoff_rows(recs); assert len(ho) == n and s['handoff']['pass'] == n and s['handoff']['control_inference_calls'] == 0
    assert s['control_part_by_arm']['H0M']['hold_steps']['n'] == n and s['control_part_by_arm']['R5']['hold_steps']['max'] == 0
    text = str(s).lower(); assert 'wilson' not in text and 'p_value' not in text and 'confidence' not in text
    assert len(p2.call_rows(recs)) > 0 and {x['cells_not_run'] for x in p2.skip_rows(recs)} <= {16}


def test_event_reading_keeps_missing_events_and_uses_the_grasped_blocks_as_the_denominator():
    C = {n: i for i, n in enumerate(p2.TRACE_COLUMNS)}

    def rows(grasp, grip, ax, lift=None, phase=None):
        a = np.zeros((len(grasp), len(p2.TRACE_COLUMNS)), np.float32); a[:, C['t']] = 12 + np.arange(len(grasp)); a[:, C['is_cubeA_grasped']] = grasp
        a[:, C['a_gripper']] = grip; a[:, C['cubeA_x']] = ax; a[:, C['cubeA_z']] = .02 + (np.zeros(len(grasp)) if lift is None else np.asarray(lift))
        a[:, C['gripper_qpos_1']] = a[:, C['gripper_qpos_2']] = .01; return a
    # checkpoint row, 3 control steps (cube lifted with the hand), then the policy opens 6 cm from cube B and the flag turns off one step later
    d = p2.derived(rows([1, 1, 1, 1, 1, 1, 0, 0], [-1, -1, -1, -1, -1, 1, 1, 1], [.06] * 8, lift=[.02, .03, .04, .05, .05, .05, .0, .0]), .02); e = p2.event_reading(d, 3)
    assert e['grasped_at_checkpoint'] and e['grasp_flag_kept_through_control'] is True and e['lift_lost_in_control'] is False and e['cubeB_moved_in_control'] is False
    assert (e['open_command_found'], e['open_command_steps_after_return'], e['open_command_within_align_xy']) == (True, 2, False) and abs(e['open_command_cubeA_to_cubeB_xy_m'] - .06) < 1e-6
    assert (e['grasp_release_found'], e['grasp_release_steps_after_return'], e['grasp_release_within_align_xy']) == (True, 3, False) and abs(e['cube_lift_change_in_control_m'] - .03) < 1e-6
    # no open command and no release: recorded as not found
    d = p2.derived(rows([1] * 6, [-1] * 6, [.0] * 6), .02); e = p2.event_reading(d, 0)
    assert e['grasp_flag_kept_through_control'] == 'NA' and not e['open_command_found'] and not e['grasp_release_found'] and e['open_command_within_align_xy'] is None
    # the flag turns off inside the control part: reported there, and not counted as a release after the return
    d = p2.derived(rows([1, 1, 0, 0, 0], [-1] * 5, [.0] * 5), .02); e = p2.event_reading(d, 3)
    assert e['grasp_flag_kept_through_control'] is False and e['grasp_flag_off_in_control_step'] == 2 and not e['grasp_release_found'] and not e['grasped_at_return']
    # the branch ended during control
    d = p2.derived(rows([1, 1, 1], [-1] * 3, [.0] * 3), .02); e = p2.event_reading(d, 3); assert e['ended_during_control'] and not e['open_command_found']
    ev = [{'scene': 1, 'arm': a, **p2.event_reading(d, 0 if a == 'C0' else 3), 'grasped_at_checkpoint': g} for a in p2.SHORT.values() for g in (True, False)]
    s = p2.event_summary(ev); assert len(s) == 8 and all(x['blocks'] == 1 for x in s) and s[0]['grasp_flag_kept_through_control'] == 'NA' and s[0]['open_command_none'] == 1
