"""Stage 2B four-arm control: the alternative-noise Continue, its parity with Continue, accounting and tables.

Mock physics only. These tests say nothing about ManiSkill, the GPU or the research policy.
"""
import itertools
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from test_runtime_mock import MockPolicy
from test_stage2a_pilot import GoalEnv, StaleObsEnv, ForgetfulQueuePolicy, CFG, CPS, REPS
from hci.simulation import Runtime
from hci import pilot, pilot2b as p2
from hci.io import read_json, seed_for


class SeedPolicy(MockPolicy):
    def generate(self, episode, t, repeat=0):
        self.last_seed = seed_for(episode, t, repeat); return super().generate(episode, t, repeat)


def make(env=None, policy=None):
    return Runtime(env or GoalEnv(), policy or SeedPolicy(), CFG)


def scene(seed, **kw):
    return p2.collect_scene(make(kw.pop('env', None), kw.pop('policy', None)), seed, CFG, CPS, REPS, **kw)[0]


@pytest.fixture(scope='module')
def recs():
    return [scene(s) for s in range(1, 13)]


def at_checkpoint(seed=1, t=12):
    r = make().reset(seed)
    for _ in range(t): r.step_base()
    return r, r.snapshot()


def test_the_four_arms_start_from_the_same_restored_snapshot(monkeypatch):
    seen = {}; real = p2.apply_arm

    def spy(r, arm, rep):
        seen.setdefault((r.t, rep), {})[arm] = pilot.runtime_tree(r); real(r, arm, rep)
    monkeypatch.setattr(p2, 'apply_arm', spy); rec = scene(1)
    assert len(seen) == rec['counts']['blocks'] > 0
    for trees in seen.values():
        assert set(trees) == set(p2.ARMS)
        for arm in p2.ARMS[1:]: assert pilot.compare_tree(trees['continue'], trees[arm], p2.EXACT)['pass']


def test_alt_noise_continue_keeps_queue_and_history_and_follows_continue_until_the_natural_inference():
    r, s = at_checkpoint(); q = [x.copy() for x in r.policy.queue]; before = pilot.runtime_tree(r); runs = {}
    for arm in ('continue', 'continue_alt_noise'):
        r.restore(s); p2.apply_arm(r, arm, 2); after = pilot.runtime_tree(r)
        assert r.intervention is None and len(r.policy.queue) == 4 and all(np.array_equal(a, b) for a, b in zip(q, r.policy.queue))
        want = {**before, 'discrete': {**before['discrete'], 'repeat': p2.noise_id(arm, 2)}}       # the noise key is the only allowed change
        assert pilot.compare_tree(want, after, p2.EXACT)['pass']
        calls = r.policy.inference_calls; runs[arm] = [r.step_base() for _ in range(5)]; assert r.policy.inference_calls == calls + 1
    a, b = runs['continue'], runs['continue_alt_noise']
    assert all(np.array_equal(x, y) for x, y in zip(a[:4], b[:4])) and not np.allclose(a[4], b[4])


def test_block_repeat_and_policy_noise_repeat_are_separate_fields(recs):
    n = 0
    for rec in recs:
        for b in rec['blocks']:
            ids = {c['experimental_arm']: (c['block_repeat_id'], c['policy_noise_repeat_id']) for c in b['cells']}; r = b['repeat']
            assert ids == {'continue': (r, r), 'continue_alt_noise': (r, r + 4), 'refresh': (r, r), 'retreat_retry': (r, r)}
            for c in b['cells']:
                assert all(k['policy_noise_repeat_id'] == c['policy_noise_repeat_id'] for k in c['inference_calls'])
                assert all(k['seed'] == seed_for(rec['scene'], k['t'], c['policy_noise_repeat_id']) for k in c['inference_calls']); n += len(c['inference_calls'])
    assert n > 0
    # alternative keys never fall on a key that another block's Continue already used
    assert {p2.noise_id('continue_alt_noise', r) for r in REPS}.isdisjoint(REPS)
    with pytest.raises(ValueError): p2.collect_scene(make(), 1, CFG, CPS, [0, 1, 2, 4])


@pytest.mark.parametrize('bad', ['clear_queue', 'infer_now', 'reset_history', 'extra_action'])
def test_a_wrong_alt_noise_arm_is_detected(monkeypatch, bad):
    real = p2.apply_arm

    def wrong(r, arm, rep):
        real(r, arm, rep)
        if arm != 'continue_alt_noise': return
        if bad == 'clear_queue': r.policy.clear()
        elif bad == 'infer_now': r.policy.queue = [x.copy() for x in r.policy.generate(r.episode, r.t, r.repeat)][:len(r.policy.queue)]
        elif bad == 'reset_history': r.policy.queue[0] = r.policy.queue[0] + 1e-3
        elif bad == 'extra_action': r.advance(np.zeros(4, np.float32))
    monkeypatch.setattr(p2, 'apply_arm', wrong)
    with pytest.raises((RuntimeError, ValueError)): scene(1)


def test_alt_noise_rerun_and_continue_rerun_are_identical():
    rec = scene(5, recheck=True); arms = [d['experimental_arm'] for d in rec['rechecks']]
    assert rec['reproducibility_ok'] and arms.count('continue_alt_noise') == arms.count('continue') == rec['counts']['snapshots'] > 0
    assert all(d['match'] and d['action_max_abs'] == 0 and d['inference_steps_seeds_chunks_equal'] for d in rec['rechecks'])
    assert {(d['repeat'], d['experimental_arm']) for d in rec['rechecks']} == {(0, 'continue_alt_noise'), (1, 'continue')}
    assert len(rec['no_intervention_duplicates']) == rec['counts']['snapshots'] and all(d['match'] for d in rec['no_intervention_duplicates'])


def strip(rec):
    return [[(c['experimental_arm'], c['y'], c['steps'], c['reason'], [(k['t'], k['seed'], k['chunk_sha256_16']) for k in c['inference_calls']])
             for c in b['cells']] for b in rec['blocks']]


def test_results_do_not_depend_on_the_order_in_which_the_arms_are_run():
    base = strip(scene(1)); assert len(base) == 12
    for order in list(itertools.permutations(p2.ARMS))[1::5]:
        assert strip(scene(1, order=order)) == base


def test_a_branch_that_leaks_into_the_next_restore_stops_the_scene():
    with pytest.raises(RuntimeError): scene(1, policy=type('Leaky', (ForgetfulQueuePolicy, SeedPolicy), {})())


def test_ending_while_the_old_queue_is_executed_is_not_an_error(recs):
    hits = [b for rec in recs for b in rec['blocks'] if b['c0_c1_queue_prefix_parity']['first_inference_step_C0'] is None]
    assert hits                                        # the mock has blocks that succeed before any new inference
    for b in hits:
        c0, c1 = b['cells'][:2]; p = b['c0_c1_queue_prefix_parity']
        assert p['pass'] and p['first_inference_seeds_differ'] is None and (c0['y'], c0['steps']) == (c1['y'], c1['steps']) and not c1['inference_calls']


def test_pairs_are_joined_on_scene_checkpoint_and_block_repeat(recs):
    rows = p2.block_rows(recs); keys = [(x['scene'], x['checkpoint'], x['block_repeat_id']) for x in rows]
    assert len(keys) == len(set(keys)) == sum(r['counts']['blocks'] for r in recs)
    cells = {(b['scene'], b['checkpoint'], b['repeat']): {c['arm']: c['y'] for c in b['cells']} for r in recs for b in r['blocks']}
    for x in rows:
        y = cells[(x['scene'], x['checkpoint'], x['block_repeat_id'])]
        assert [x[f'y_{k}'] for k in ('C0', 'C1', 'F', 'R')] == [y[k] for k in ('C0', 'C1', 'F', 'R')] and x['policy_noise_repeat_id_C1'] == x['block_repeat_id'] + 4
        for a, ref in p2.PAIRS: assert x[f'{a}_vs_{ref}'] == f'{y[ref]}to{y[a]}'
    assert {x['C1_vs_C0'] for x in rows} >= {'0to1', '1to0'}        # the control arm does change outcomes in the mock, so the join matters


def test_the_control_arm_is_neither_an_intervention_nor_a_block(recs):
    rec = recs[0]; assert rec['counts']['blocks'] == rec['counts']['snapshots'] * len(REPS) and rec['counts']['continuations'] == 4 * rec['counts']['blocks']
    for b in rec['blocks']:
        c1 = b['cells'][1]; assert c1['intervention'] is None and c1['retreat'] is None and c1['queue_len_after_choice'] == 4
    import copy
    for change in ({'intervention': 1}, {'queue_len_after_choice': 0}, {'policy_noise_repeat_id': 1}, {'first_inference_step': 12}):
        bad = copy.deepcopy(rec); bad['blocks'][0]['cells'][1].update(change)
        with pytest.raises(ValueError): p2.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); extra = copy.deepcopy(bad['blocks'][0]); extra['repeat'] = 4; bad['blocks'].append(extra)
    with pytest.raises(ValueError): p2.validate_scene(bad, CFG)
    bad = copy.deepcopy(rec); del bad['blocks'][0]['cells'][1]
    with pytest.raises(ValueError): p2.validate_scene(bad, CFG)


def test_a_failed_or_missing_scene_never_marks_the_data_complete(tmp_path):
    def run_scene(seed):
        if seed == 3: raise RuntimeError('simulator error')
        return p2.collect_scene(make(), seed, CFG, CPS, REPS)[:3]
    with pytest.raises(RuntimeError): p2.collect_scenes([1, 2, 3, 4], tmp_path, run_scene)
    assert sorted(p.name for p in (tmp_path / 'scenes').iterdir()) == ['1.json', '2.json', '3.error.json']
    meta, got = p2.finalize(tmp_path, [1, 2, 3, 4], CFG); assert meta['completed'] is False and len(got) == 2
    assert p2.finalize(tmp_path, [1, 2], CFG)[0]['completed'] is False
    ok = tmp_path / 'ok'; p2.collect_scenes([1, 2], ok, run_scene); meta, got = p2.finalize(ok, [1, 2], CFG)
    assert meta['completed'] is True and meta['continuations'] == 4 * meta['blocks']
    assert p2.finalize(ok, [1, 2, 5], CFG)[0]['completed'] is False


def test_a_restore_that_changes_the_observation_is_detected():
    res = p2.audit_scene(make(env=StaleObsEnv()), 1, CFG, CPS)
    assert not res['pass'] and not res['checkpoints'][0]['checks']['repeat_restored']['live_observation_pass']
    with pytest.raises(RuntimeError): scene(1, env=StaleObsEnv())


def test_audit_runs_the_alt_noise_branch_and_passes_on_a_correct_runtime():
    res = p2.audit_scene(make(), 1, CFG, CPS); rep = p2.audit_report([res]); first = res['checkpoints'][0]
    assert res['pass'] and rep['pass'] and all(v > 0 for v in rep['branches_executed'].values())
    assert first['checks']['after_continue_alt_noise_trace']['pass'] and first['branches']['continue_alt_noise']['policy_noise_repeat_id'] == 4
    assert first['input_saving_changes_nothing']['pass']
    bad = p2.audit_scene(make(policy=type('Leaky', (ForgetfulQueuePolicy, SeedPolicy), {})()), 1, CFG, CPS); assert not bad['pass']


def test_saved_policy_inputs_hold_no_label_and_saving_changes_nothing():
    r, _ = at_checkpoint(); x, before, cmp = p2.inputs_without_side_effect(r)
    assert cmp['pass'] and set(x) == {'state_history', 'last_action', 'queue', 't', 'remaining_budget'}
    assert x['queue'].shape == (4, 4) and x['state_history'].shape[0] == 2 and int(x['t']) == 12 and int(x['remaining_budget']) == 48
    rec, _, _, inputs = p2.collect_scene(make(), 1, CFG, CPS, REPS)
    assert sorted(inputs) == [s['checkpoint'] for s in rec['snapshots']]
    assert all(s['policy_inputs_sha256'] == pilot.tree_hash(inputs[s['checkpoint']]) for s in rec['snapshots'])


def test_tables_follow_the_declared_definitions(recs):
    s = p2.summarize(recs); rows = p2.block_rows(recs); n = len(rows); d = {(x['arm'], x['reference']): x for x in s['pairs'] if x['checkpoint'] == 'all'}
    for x in d.values():
        assert x['blocks'] == n == x['0to1'] + x['1to0'] + x['1to1'] + x['0to0'] and x['reference_success_blocks'] + x['reference_failure_blocks'] == n
    assert abs((d[('F', 'C0')]['delta_mean_outcome'] - d[('C1', 'C0')]['delta_mean_outcome']) - d[('F', 'C1')]['delta_mean_outcome']) < 1e-12
    scenes = p2.scene_rows(recs); used = [x for x in scenes if x['in_conditional_summary']]
    assert len(scenes) == len(recs) and s['scene_equal_weight']['scenes_in_summary'] == len(used)
    one = used[0]; mine = [x for x in rows if x['scene'] == one['scene']]
    assert abs(one['d_F_C0'] - np.mean([x['y_F'] - x['y_C0'] for x in mine])) < 1e-12
    assert abs(s['scene_equal_weight']['d_F_C0']['mean_over_scenes'] - np.mean([x['d_F_C0'] for x in used])) < 1e-12
    q = s['scene_equal_weight']['d_R_F']; assert q['scenes_positive'] + q['scenes_negative'] + q['scenes_zero'] == len(used)
    comp = [x for x in p2.complementarity_rows(rows) if x['checkpoint'] == 'all']
    assert sum(x['blocks'] for x in comp if x['pattern'].startswith('F=')) == n == sum(x['blocks'] for x in comp if x['pattern'].startswith('C1='))
    for x in comp: assert sum(k * x[f'snapshots_with_{k}_repeats'] for k in (1, 2, 3, 4)) == x['blocks']
    assert p2.pair_table([dict(rows[0], F_vs_C0='0to0')], 'F', 'C0')['rate_1to0'] == 'NA'
    empty = p2.scene_rows([dict(recs[0], blocks=[], snapshots=[], counts={**recs[0]['counts'], 'snapshots': 0})], [])[0]
    assert empty['d_F_C0'] == 'NA' and not empty['in_conditional_summary']
    text = str(s).lower(); assert 'wilson' not in text and 'p_value' not in text and 'confidence' not in text
    calls = p2.call_rows(recs); assert sum(1 for k in calls if k['experimental_arm'] == 'continue_alt_noise') > 0 and len(p2.parity_rows(recs)) == n


def test_reproducibility_failure_keeps_rows_but_blocks_completion(tmp_path):
    class Drift(SeedPolicy):
        n = 0

        def generate(self, episode, t, repeat=0):
            Drift.n += 1; return super().generate(episode, t + Drift.n, repeat)
    p2.collect_scenes([1], tmp_path, lambda s: p2.collect_scene(make(policy=Drift()), s, CFG, CPS, REPS)[:3])
    meta, got = p2.finalize(tmp_path, [1], CFG)
    assert len(got) == 1 and meta['completed'] is False and meta['reproducibility_ok'] is False and read_json(tmp_path / 'scenes' / '1.json')['blocks']


def test_completion_is_derived_from_the_recorded_checks_not_from_a_flag(tmp_path):
    p2.collect_scenes([1, 5], tmp_path, lambda s: p2.collect_scene(make(), s, CFG, CPS, REPS, recheck=s == 1)[:3])
    assert p2.finalize(tmp_path, [1, 5], CFG, rerun_scenes=[1])[0]['completed'] is True
    assert p2.finalize(tmp_path, [1, 5], CFG, rerun_scenes=[1, 5])[0]['completed'] is False      # scene 5 lacks the declared reruns
    assert p2.finalize(tmp_path, [1, 5], CFG)[0]['completed'] is False                           # scene 1 carries reruns nobody declared
    rec = read_json(tmp_path / 'scenes' / '1.json'); assert not p2.repro_problems(rec, True)
    cut = dict(rec, no_intervention_duplicates=rec['no_intervention_duplicates'][1:]); assert p2.repro_problems(cut, True)
    lied = dict(rec, rechecks=[dict(d, match=False) for d in rec['rechecks']]); assert len(p2.repro_problems(lied, True)) == 2
