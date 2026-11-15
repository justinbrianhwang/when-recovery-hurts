"""Handoff H1: one-shot latest-observation padding after R5. Control flow, isolation of the manipulation, accounting.

Mock physics only. These tests say nothing about ManiSkill, the GPU, the research policy or whether padding helps.
"""
import copy, importlib.util, inspect, itertools, json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from test_stage2a_pilot import GoalEnv, StaleObsEnv, ForgetfulQueuePolicy, CFG, CPS, REPS
from test_stage2d_pilot import SeedPolicy, NoSuccessEnv, LiftEnv, lift_seed, probe
from hci.simulation import Runtime
from hci import pilot, handoff_h1 as H
from hci.io import read_json, write_json, seed_for, sha256

_spec = importlib.util.spec_from_file_location('h1script', Path(__file__).resolve().parents[1] / 'scripts' / '26_handoff_h1.py')
S = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S)


class VelEnv(GoalEnv):
    """qvel carries the last commanded motion, so a wrong 'stop' (zeroed velocity) would be visible."""
    def __init__(self):
        super().__init__(); self.v = torch.zeros((1, 9))

    def reset(self, seed):
        self.v = torch.zeros((1, 9)); return super().reset(seed)

    def step(self, a):
        self.v = torch.zeros((1, 9)); self.v[:, :3] = a[:, :3] + .5; return super().step(a)

    def obs(self):
        o = super().obs(); o['agent']['qvel'] = self.v.clone(); return o


def make(env=None, policy=None, cfg=CFG):
    return Runtime(env or GoalEnv(), policy or SeedPolicy(), cfg)


def scene(seed, **kw):
    return H.collect_scene(make(kw.pop('env', None), kw.pop('policy', None)), seed, CFG, CPS, REPS, **kw)[0]


@pytest.fixture(scope='module')
def recs():
    return [scene(s) for s in range(1, 13)]


def at_return(seed=1, t=12, env=None):
    r = make(env).reset(seed)
    for _ in range(t): r.step_base()
    s = r.snapshot(); r.force(2); return r, s


def cells_of(rec):
    return [(b, {c['arm']: c for c in b['cells']}) for b in rec['blocks']]


# 1
def test_r_keep_is_the_existing_r5_and_leaves_the_history_alone(recs):
    rec = recs[0]; r = make().reset(rec['scene']); snaps = {}; n = 0
    while not r.done:
        if r.t in CPS and r.eligible().all(): snaps[r.t] = r.snapshot()
        r.step_base()
    for b, cell in cells_of(rec):
        r.restore(snaps[b['checkpoint']]); r.repeat = b['repeat']; r.force(2); hist = H.entry_hashes(r.policy.history); res = r.complete(); x = cell['R_KEEP']; n += 1
        assert (x['y'], x['steps'], x['reason'], x['intervention']) == (res['success'], res['steps'], res['reason'], 2) and x['retreat'] == pilot.jsonable(res['retreat'])
        assert not x['padding_applied'] and x['padding'] is None and x['policy_input_history_sha256'] == hist and x['return_state_equals_reference']['pass']
    assert n == len(rec['blocks']) > 0
    r, s = at_return(); before = pilot.runtime_tree(r); assert H.apply_arm(make().reset(1), 'continue', 0) is None
    r.restore(s); assert H.apply_arm(r, 'r_keep', 3) is None and pilot.compare_tree({**before, 'discrete': {**before['discrete'], 'repeat': 3}}, pilot.runtime_tree(r), H.EXACT)['pass']


# 2, 3, 4
def test_padding_copies_the_latest_entry_without_aliasing_and_keeps_schema():
    r, s = at_return(env=VelEnv()); hist = list(r.policy.history); prev, latest = hist; before = copy.deepcopy(hist); schema = H.entry_schema(latest)
    assert not np.array_equal(pilot.as_numpy(prev['extra']['tcp_pose']), pilot.as_numpy(latest['extra']['tcp_pose']))        # the retreat moved between the two
    pad = H.pad_history_once(r.policy); new = list(r.policy.history)
    assert new[1] is latest and new[0] is not latest and new[0] is not prev                                                # latest untouched, earlier one a new object
    assert pilot.maxdiff(new[1], before[1]) == 0 and pilot.maxdiff(new[0], before[1]) == 0 and pilot.maxdiff(new[0], before[0]) > 0
    assert pad['latest_entry_unchanged'] and pad['earlier_entries_equal_latest'] and pad['input_changed'] and not pad['no_op'] and pad['entries'] == 2
    assert H.entry_schema(new[0]) == H.entry_schema(new[1]) == schema == pad['entry_schema'] and list(schema) == sorted(schema)
    for path in (('agent', 'qpos'), ('agent', 'qvel'), ('extra', 'tcp_pose')):                                              # no shared memory between the two entries
        a, b = new[0][path[0]][path[1]], new[1][path[0]][path[1]]; assert a is not b; a += 1.; assert pilot.maxdiff(new[1], before[1]) == 0; a -= 1.
    assert r.policy.history.maxlen == 2 and len(r.policy.history) == 2
    r2 = make().reset(1)
    with pytest.raises(ValueError): r2.policy.history = type(r2.policy.history)([r2.policy.history[-1]], maxlen=2); H.pad_history_once(r2.policy)


# 5, 6
def test_padding_changes_nothing_but_the_earlier_history_entry_and_does_not_zero_velocity():
    r, s = at_return(env=VelEnv()); before = pilot.runtime_tree(r); raw = copy.deepcopy(r.raw); state = copy.deepcopy(r.env.get_state_dict()); t = r.t; n = len(r.actions)
    v_latest = pilot.as_numpy(r.policy.history[-1]['agent']['qvel']).copy(); assert np.abs(v_latest).max() > 0
    pad = H.pad_history_once(r.policy); after = pilot.runtime_tree(r); cmp = pilot.compare_tree(before, after, H.EXACT)
    assert cmp['failed'] == ['history_state'] and H.only_history_changed(before, r, pad)['pass']
    assert pilot.maxdiff(r.raw, raw) == 0 and pilot.maxdiff(r.env.get_state_dict(), state) == 0 and (r.t, len(r.actions), r.policy.queue, r.intervention) == (t, n, [], 2)
    assert after['discrete'] == before['discrete'] and np.array_equal(after['last_action'], before['last_action'])           # time, counters, RNG digests, retreat record
    for h in r.policy.history: assert np.array_equal(pilot.as_numpy(h['agent']['qvel']), v_latest)                          # the copied velocity is the measured one, not zero
    assert np.abs(pilot.as_numpy(r.raw['agent']['qvel'])).max() > 0
    r.policy.queue = [np.zeros(4, np.float32)]; assert not H.only_history_changed(before, r, pad)['pass']                    # anything else that moves is caught
    src = inspect.getsource(H.pad_history_once); assert 'env' not in src and 'reset(' not in src.replace('policy.reset()', '') and 'zeros' not in src and 'advance' not in src


# 7
def test_padding_is_applied_once_and_the_fifo_continues_with_real_observations(monkeypatch):
    n = []; real = H.pad_history_once
    monkeypatch.setattr(H, 'pad_history_once', lambda p: n.append(1) or real(p)); rec = scene(1); blocks = rec['counts']['blocks']
    assert len(n) == blocks > 0                                                           # one call per R_PAD_ONCE branch, none for the other arms
    monkeypatch.undo(); r, s = at_return(); H.pad_history_once(r.policy); padded = r.policy.history[0]
    a = r.step_base(); assert r.policy.history[0] is not padded and pilot.maxdiff(r.policy.history[0], padded) == 0        # padded pair shifted: [latest, new real observation]
    assert np.array_equal(pilot.as_numpy(r.policy.history[-1]['extra']['tcp_pose'])[0, :3], r.tcp_history[-1])
    r.step_base(); assert all(np.array_equal(pilot.as_numpy(h['extra']['tcp_pose'])[0, :3], x) for h, x in zip(r.policy.history, r.tcp_history[-2:]))   # two real ones again


# 8
def test_keep_and_pad_resume_at_the_same_step_with_the_same_seed_and_no_inference_during_r5(recs):
    n = 0
    for rec in recs:
        for b, cell in cells_of(rec):
            t = b['checkpoint']; ref = next(s for s in rec['snapshots'] if s['checkpoint'] == t)['reference']; tau = t + ref['r5_steps']; k, p = cell['R_KEEP'], cell['R_PAD_ONCE']; n += 1
            assert k['first_inference_step'] == p['first_inference_step'] == tau == ref['return_step'] and k['first_inference_seed'] == p['first_inference_seed'] == seed_for(rec['scene'], tau, b['repeat'])
            assert all(c['t'] >= tau for c in k['inference_calls'] + p['inference_calls']) and ref['policy_inference_calls'] == 0 and k['r5_steps'] == p['r5_steps'] == ref['r5_steps']
            assert k['policy_input_history_sha256'] == ref['actual_history_sha256'] and p['policy_input_history_sha256'] == ref['padded_history_sha256'] and len(set(ref['padded_history_sha256'])) == 1
            d = b['handoff']['first_chunk_difference']; assert d['identical'] == (k['first_chunk'] == p['first_chunk']) and d['units'] == 'normalised controller input'
            assert b['handoff']['input_changed'] and p['padding']['input_changed'] == ref['padding_would_change_the_input']
    assert n > 0


def test_a_block_whose_last_two_observations_are_equal_is_a_kept_no_op():
    class Still(NoSuccessEnv):                                                             # the arm does not respond: every observation is the same
        def step(self, a): return super().step(a * 0)
    rec = scene(1, env=Still()); assert rec['blocks']
    for b, cell in cells_of(rec):
        p = cell['R_PAD_ONCE']; assert p['padding_applied'] and p['padding']['no_op'] and not b['handoff']['input_changed'] and b['handoff']['first_chunk_difference']['identical']
        assert (p['y'], p['steps'], p['inference_calls']) == (cell['R_KEEP']['y'], cell['R_KEEP']['steps'], cell['R_KEEP']['inference_calls'])       # same seed, same input, same rollout
    s = H.summarize([rec]); assert s['handoff']['no_op_blocks'] == s['blocks'] and s['handoff']['input_changed_blocks'] == 0


# 9
def test_an_episode_that_ends_during_r5_is_neither_padded_nor_resumed_and_is_kept():
    seed, z = lift_seed(); env = LiftEnv(); env.base = z; rec, _, _, _, handoffs = H.collect_scene(make(env=env), seed, CFG, [12], REPS); ref = rec['snapshots'][0]['reference']
    assert ref['ended_during_control'] and ref['r5_steps'] == 3 and handoffs[12] == {} and len(rec['blocks']) == 4
    for b, cell in cells_of(rec):
        for k in ('R_KEEP', 'R_PAD_ONCE'):
            x = cell[k]; assert not x['padding_applied'] and not x['policy_resumed'] and x['ended_during_control'] and not x['inference_calls'] and x['first_chunk'] is None
            assert (x['y'], x['steps']) == (1, 15)
        assert b['y_R_KEEP'] == b['y_R_PAD_ONCE'] == 1 and b['handoff']['pass'] and not b['handoff']['padding_applied']
    s = H.summarize([rec]); assert s['handoff']['ended_during_r5_blocks'] == 4 and s['handoff']['policy_resumed_blocks'] == 0 and s['blocks'] == 4 and s['resumed_blocks_only']['pairs'] == []
    assert s['success_blocks_by_arm']['R_PAD_ONCE'] == 4                                     # counted in the full allocation


def strip(rec):
    return [[(c['experimental_arm'], c['y'], c['steps'], c['reason'], c['first_chunk'], [(k['t'], k['seed'], k['chunk_sha256_16']) for k in c['inference_calls']]) for c in b['cells']]
            for b in rec['blocks']] + [s['reference']['actual_history_sha256'] for s in rec['snapshots']]


# 10
def test_padding_does_not_leak_into_other_branches_or_depend_on_their_order():
    base = strip(scene(1)); assert len(base) == 15
    for order in itertools.permutations(H.ARMS): assert strip(scene(1, order=order)) == base
    r, s = at_return(); H.pad_history_once(r.policy); r.complete(); r.restore(s); tree = pilot.runtime_tree(r)
    r2 = make().reset(1)
    for _ in range(12): r2.step_base()
    assert pilot.compare_tree(pilot.runtime_tree(r2), tree, H.EXACT)['pass'] and r.intervention is None and len(r.policy.queue) == 4
    with pytest.raises(RuntimeError): scene(1, policy=type('Leaky', (ForgetfulQueuePolicy, SeedPolicy), {})())
    with pytest.raises(RuntimeError): scene(1, env=StaleObsEnv())


# 11
def test_reproducibility_checks_have_the_declared_keys_and_targets():
    rec = scene(5, recheck=True); snaps = [s['checkpoint'] for s in rec['snapshots']]
    assert sorted((d['checkpoint'], d['repeat'], d['against']) for d in rec['no_intervention_duplicates']) == [(t, 0, 'trunk') for t in snaps]
    assert sorted((d['checkpoint'], d['repeat'], d['experimental_arm']) for d in rec['rechecks']) == sorted((t, 0, a) for t in snaps for a in H.R5_ARMS)
    assert all(d['match'] and d['inference_and_first_chunk_equal'] and d['action_max_abs'] == 0 for d in rec['rechecks']) and not H.repro_problems(rec, True)
    assert H.repro_problems(rec, False) and H.repro_problems(dict(rec, rechecks=rec['rechecks'][1:]), True)
    assert len(H.repro_problems(dict(rec, rechecks=[dict(d, match=False) for d in rec['rechecks']]), True)) == 2


# 12
def test_nothing_about_outcomes_or_objects_decides_the_padding():
    src = inspect.getsource(H.pad_history_once) + inspect.getsource(H.apply_arm)
    for word in ('cube', 'success', 'grasp', 'probe', 'evaluate', '.y', 'stage', 'scene'): assert word not in src
    assert set(inspect.signature(H.apply_arm).parameters) == {'r', 'arm', 'rep'} and set(inspect.signature(H.pad_history_once).parameters) == {'policy'}
    assert not set(inspect.signature(H.collect_scene).parameters) & {'pad', 'padding', 'history_length', 'wait', 'hold', 'outcome', 'threshold'}
    with pytest.raises(ValueError): H.collect_scene(make(), 1, CFG, CPS, REPS, order=H.ARMS[:2])
    rec, traces, _, inputs, handoffs = H.collect_scene(make(), 1, CFG, CPS, REPS, probe=probe)
    for x in inputs.values(): assert set(x) == {'state_history', 'last_action', 'queue', 't', 'remaining_budget'}
    assert sorted(handoffs) == [s['checkpoint'] for s in rec['snapshots']] and all(s['reference']['handoff_inputs_sha256'] == pilot.tree_hash(handoffs[s['checkpoint']]) for s in rec['snapshots'])
    assert all(b['handoff']['padding_applied'] for b in rec['blocks'])                    # every block of every snapshot, whatever its outcome
    ph = H.TRACE_COLUMNS.index('phase'); P = H.PHASES.index
    for b in rec['blocks']:
        t, rep = b['checkpoint'], b['repeat']; T = b['cells'][1]['r5_steps']
        for k in ('R_KEEP', 'R_PAD_ONCE'):
            rows = traces[f'b_{t}_{rep}_{k}']; assert np.all(rows[:T, ph] == P('intervention_move')) and np.all(rows[T:, ph] == P('policy_after_return'))
            assert np.array_equal(rows[:T], traces[f'ref_{t}'])                           # no extra step, no waiting: the control part is the reference in both arms
        assert np.array_equal(traces[f'b_{t}_{rep}_R_KEEP'][:T], traces[f'b_{t}_{rep}_R_PAD_ONCE'][:T])


@pytest.mark.parametrize('bad', ['inference_count', 'pad_keep', 'extra_step', 'other_seed', 'clear_last_action'])
def test_a_wrong_manipulation_stops_the_scene(monkeypatch, bad):
    real = H.pad_history_once; force = Runtime.force

    def wrong(policy):
        out = real(policy)
        if bad == 'clear_last_action': policy.last_action = None
        return out

    def wrong_force(self, action):
        force(self, action)
        if bad == 'extra_step' and action == 2 and self.repeat == 1: self.advance(np.zeros(4, np.float32))
        if bad == 'pad_keep' and action == 2 and self.repeat == 2: real(self.policy)
        if bad == 'other_seed' and action == 2 and self.repeat == 3: self.repeat = 0
        if bad == 'inference_count' and action == 2 and self.repeat == 1: self.policy.inference_calls += 1
    monkeypatch.setattr(H, 'pad_history_once', wrong); monkeypatch.setattr(Runtime, 'force', wrong_force)
    with pytest.raises((RuntimeError, ValueError)): scene(1)


def test_accounting_rejects_missing_duplicated_or_relabelled_cells(recs):
    rec = recs[0]; assert rec['counts']['continuations'] == 3 * rec['counts']['blocks'] == 12 * rec['counts']['snapshots'] and H.validate_scene(rec, CFG)
    for i, change in ((1, {'padding_applied': True}), (2, {'padding_applied': False}), (2, {'first_inference_seed': 1}), (1, {'intervention': None}), (0, {'padding_applied': True}),
                      (2, {'r5_steps': 99}), (1, {'policy_noise_repeat_id': 2}), (2, {'steps': 61}), (2, {'policy_input_history_sha256': ['x', 'y']}), (0, {'queue_len_after_choice': 0})):
        bad = copy.deepcopy(rec); bad['blocks'][0]['cells'][i].update(change)
        with pytest.raises(ValueError): H.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); del bad['blocks'][-1]
    with pytest.raises(ValueError): H.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); bad['blocks'].append(copy.deepcopy(bad['blocks'][0]))
    with pytest.raises(ValueError): H.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); bad['blocks'][0]['cells'][2]['padding']['latest_entry_unchanged'] = False
    with pytest.raises(ValueError): H.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); bad['blocks'][0]['cells'][1], bad['blocks'][0]['cells'][2] = bad['blocks'][0]['cells'][2], bad['blocks'][0]['cells'][1]
    with pytest.raises(ValueError): H.validate_scene(bad, CFG)


def test_audit_runs_both_r5_arms_and_detects_a_bad_restore():
    res = H.audit_scene(make(), 1, CFG, CPS); rep = H.audit_report([res]); first = res['checkpoints'][0]
    assert res['pass'] and rep['pass'] and all(v > 0 for v in rep['branches_executed'].values()) and set(first['branches']) == {'refresh', 'reference_r5_control', 'r_keep', 'r_pad_once'}
    assert first['branches']['r_pad_once']['only_history_changed']['pass'] and first['branches']['r_keep']['padding'] is None and first['handoff']['pass']
    assert first['handoff']['first_inference']['R_KEEP'] == first['handoff']['first_inference']['R_PAD_ONCE'] and first['handoff']['first_inference']['R_KEEP'][0] == first['handoff']['expected_first_inference_step']
    for name in ('after_reference_r5_control', 'after_r_keep', 'after_r_pad_once'): assert first['checks'][name + '_restored']['pass'] and first['checks'][name + '_trace']['pass']
    assert not H.audit_scene(make(policy=type('Leaky', (ForgetfulQueuePolicy, SeedPolicy), {})()), 1, CFG, CPS)['pass'] and not H.audit_scene(make(env=StaleObsEnv()), 1, CFG, CPS)['pass']


def test_tables_follow_the_declared_definitions(recs):
    s = H.summarize(recs); rows = H.block_rows(recs); n = len(rows); d = {(x['arm'], x['reference']): x for x in s['pairs'] if x['checkpoint'] == 'all'}
    assert set(d) == set(H.PAIRS) and [x['comparison'] for x in s['pairs'] if x['checkpoint'] == 'all'] == ['primary', 'secondary', 'secondary']
    for x in d.values(): assert x['blocks'] == n == x['0to1'] + x['1to0'] + x['1to1'] + x['0to0']
    p = d[('R_PAD_ONCE', 'R_KEEP')]; assert p['0to1'] == sum(x['y_R_KEEP'] == 0 and x['y_R_PAD_ONCE'] == 1 for x in rows) and p['1to0'] == sum(x['y_R_KEEP'] == 1 and x['y_R_PAD_ONCE'] == 0 for x in rows)
    scenes = H.scene_rows(recs); used = [x for x in scenes if x['in_conditional_summary']]; one = used[0]; mine = [x for x in rows if x['scene'] == one['scene']]
    assert abs(one['d_R_PAD_ONCE_R_KEEP'] - np.mean([x['y_R_PAD_ONCE'] - x['y_R_KEEP'] for x in mine])) < 1e-12
    assert abs(s['scene_equal_weight']['d_R_PAD_ONCE_R_KEEP']['mean_over_scenes'] - np.mean([x['d_R_PAD_ONCE_R_KEEP'] for x in used])) < 1e-12
    h = s['handoff']; assert h['blocks'] == n == h['policy_resumed_blocks'] + h['ended_during_r5_blocks'] and h['input_changed_blocks'] + h['no_op_blocks'] == h['policy_resumed_blocks']
    assert h['outcome_differs_blocks'] == p['0to1'] + p['1to0'] and h['first_chunk_arm_mean_abs']['n'] == h['policy_resumed_blocks']
    empty = H.scene_rows([dict(recs[0], blocks=[], snapshots=[], counts={**recs[0]['counts'], 'snapshots': 0})], [])[0]; assert empty['d_R_PAD_ONCE_R_KEEP'] == 'NA'
    text = str(s).lower(); assert 'wilson' not in text and 'p_value' not in text and 'confidence' not in text
    c = H.chunk_difference([[0, 0, 0, 1]] * 8, [[.1, 0, -.3, -1]] * 8); assert abs(c['arm_max_abs'] - .3) < 1e-12 and c['gripper_sign_differs_steps'] == 8 and not c['identical'] and H.chunk_difference(None, None) is None
    json.dumps(s)


# 13
def test_execution_is_refused_without_a_user_authorization_for_this_plan(tmp_path):
    run = tmp_path / 'runs' / 'handoff_h1_001'; run.mkdir(parents=True); plan = run / 'scope_plan.json'; write_json(plan, {'x': 1})
    with pytest.raises(SystemExit): S.require_authorization(run, plan)                                      # no file
    for bad in ({'approved_by': 'agent', 'paid_compute_authorized': True, 'cap_usd': 1.0, 'plan_sha256': sha256(plan)},
                {'approved_by': 'user', 'paid_compute_authorized': False, 'cap_usd': 1.0, 'plan_sha256': sha256(plan)},
                {'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 0, 'plan_sha256': sha256(plan)},
                {'approved_by': 'user', 'paid_compute_authorized': True, 'plan_sha256': sha256(plan)},
                {'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': True, 'plan_sha256': sha256(plan)},
                {'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 1.0, 'plan_sha256': 'another plan'}):
        write_json(run / 'authorization.json', bad, overwrite=True)
        with pytest.raises(SystemExit): S.require_authorization(run, plan)
    write_json(run / 'authorization.json', {'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 1.0, 'plan_sha256': sha256(plan)}, overwrite=True)
    assert S.require_authorization(run, plan)['cap_usd'] == 1.0
    src = inspect.getsource(S)
    for name in S.EXECUTION: assert 'require_authorization(' in inspect.getsource(getattr(S, 'cmd_' + name))   # every execution command asks first
    assert "'authorization.json'" in src and src.count("write_json(Path(run) / 'authorization.json'") == 0 and 'authorization.json\', {' not in src   # nothing here writes an approval
    for name in ('plan', 'inspect_inputs', 'dry_run'):
        body = inspect.getsource(getattr(S, 'cmd_' + name)); assert 'runtime(' not in body and 'make_env' not in body and 'require_authorization' not in body
    assert S.STATUS['paid_compute'] == 'NOT_AUTHORIZED' and S.STATUS['handoff_h1'] == 'PREPARATION_ONLY'


def test_incomplete_or_foreign_records_never_mark_the_data_complete(tmp_path):
    def run_scene(seed):
        if seed == 3: raise RuntimeError('simulator error')
        return H.collect_scene(make(), seed, CFG, CPS, REPS, recheck=seed == 1)[:3]
    with pytest.raises(RuntimeError): H.collect_scenes([1, 5, 3, 4], tmp_path, run_scene)
    assert sorted(p.name for p in (tmp_path / 'scenes').iterdir()) == ['1.json', '3.error.json', '5.json'] and H.finalize(tmp_path, [1, 5], CFG, rerun_scenes=[1])[0]['completed'] is False
    ok = tmp_path / 'ok'; H.collect_scenes([1, 5], ok, run_scene); meta, got = H.finalize(ok, [1, 5], CFG, rerun_scenes=[1])
    assert meta['completed'] is True and meta['continuations'] == 3 * meta['blocks'] and meta['reference_control_runs'] == meta['snapshots']
    assert H.finalize(ok, [1, 5, 6], CFG, rerun_scenes=[1])[0]['completed'] is False and H.finalize(ok, [1], CFG, rerun_scenes=[1])[0]['completed'] is False
    with pytest.raises(FileExistsError): H.collect_scenes([1], ok, run_scene)                              # an existing record is never overwritten
    rec = read_json(ok / 'scenes' / '1.json'); cut = dict(rec, blocks=rec['blocks'][:-1]); write_json(ok / 'scenes' / '1.json', cut, overwrite=True)
    assert H.finalize(ok, [1, 5], CFG, rerun_scenes=[1])[0]['completed'] is False                         # a missing block
