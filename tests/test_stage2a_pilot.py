"""Stage 2A pilot: control flow of the restore audit, the paired-branch collector and the tables.

Mock physics only. These tests say nothing about ManiSkill, the GPU or the research policy.
"""
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from test_runtime_mock import MockEnv, MockPolicy
from hci.simulation import Runtime
from hci import pilot
from hci.io import read_json

CFG = {'env': {'horizon': 60}, 'protocol': {'min_intervention_step': 12, 'decision_stride': 4, 'minimum_post_retreat_steps': 4},
       'retreat': {'max_steps': 6, 'arrival_tolerance_m': .005, 'mode': 'up', 'distance_m': .03, 'max_translation_per_step_m': .005, 'lookback_steps': 8},
       'audit': {'state_atol': 1e-6, 'action_atol': 1e-6, 'observation_atol': 1.0, 'replay_steps': 10}}
CPS = [12, 28, 44]; REPS = [0, 1, 2, 3]


class GoalEnv(MockEnv):
    """Success once the first coordinate passes `goal` (a random walk under MockPolicy, so repeats differ)."""
    goal = .012

    def _info(self):
        return {'success': torch.tensor([bool(self.p[0, 0] > self.goal)])}

    def reset(self, seed):
        obs, _ = super().reset(seed); return obs, self._info()

    def step(self, a):
        obs = super().step(a)[0]; return obs, 0., torch.tensor([False]), torch.tensor([False]), self._info()


def make(env=None, policy=None):
    return Runtime(env or GoalEnv(), policy or MockPolicy(), CFG)


def scene(seed, **kw):
    return pilot.collect_scene(make(kw.pop('env', None), kw.pop('policy', None)), seed, CFG, CPS, REPS, **kw)[0]


def some_scenes(n=12):
    return [scene(s) for s in range(1, n + 1)]


def test_every_block_is_compared_with_its_own_continue():
    recs = some_scenes(); rows = pilot.pair_rows(recs); mixed = 0
    for rec in recs:
        cont = {(b['checkpoint'], b['repeat']): b['cells'][0]['y'] for b in rec['blocks']}
        for x in [x for x in rows if x['scene'] == rec['scene']]:
            assert x['y_continue'] == cont[(x['checkpoint'], x['repeat'])]
            assert x['outcome'] == pilot.CLASSES[(x['y_continue'], x['y_action'])]
        for t in {k[0] for k in cont}: mixed += len({cont[(t, r)] for r in REPS}) > 1
    assert mixed > 0                      # the mock has snapshots whose Continue differs between repeats, so the link matters
    assert {x['outcome'] for x in rows} >= {'rescue', 'harm'}


def test_continue_keeps_queue_and_history_and_interventions_clear_the_queue():
    r = make().reset(1)
    for _ in range(12): r.step_base()
    s = r.snapshot(); q = [x.copy() for x in r.policy.queue]; h = pilot.runtime_tree(r)
    r.restore(s); r.repeat = 2; r.force(0); after = pilot.runtime_tree(r)
    assert len(r.policy.queue) == len(q) == 4 and all(np.array_equal(a, b) for a, b in zip(q, r.policy.queue))
    assert pilot.compare_tree({**h, 'discrete': {**h['discrete'], 'repeat': 2}}, after, CFG['audit'])['pass'] and r.intervention is None
    rec = scene(1)
    for b in rec['blocks']:
        assert b['cells'][0]['queue_len_after_choice'] == 4 and [c['queue_len_after_choice'] for c in b['cells'][1:]] == [0, 0]
        assert b['cells'][0]['intervention'] is None and [c['intervention_step'] for c in b['cells'][1:]] == [b['checkpoint']] * 2


def test_every_planned_checkpoint_is_a_snapshot_or_a_listed_skip_whatever_the_outcome():
    recs = some_scenes(); early = 0
    for rec in recs:
        used = {x['checkpoint'] for x in rec['snapshots']}; skipped = {x['checkpoint'] for x in rec['skips']}
        assert used | skipped == set(CPS) and not used & skipped
        assert rec['counts']['blocks'] == len(used) * len(REPS) and rec['counts']['continuations'] == 3 * len(rec['blocks'])
        for x in rec['skips']:
            assert x['checkpoint'] >= rec['trunk']['steps'] and x['reason'] == 'trunk_succeeded_before_checkpoint'; early += 1
        # snapshots exist for failing and for succeeding continuations alike
    ys = [b['y_continue'] for rec in recs for b in rec['blocks']]
    assert early > 0 and 0 in ys and 1 in ys
    rec = recs[0]; del rec['blocks'][-1]
    with pytest.raises(ValueError): pilot.validate_scene(rec, CFG)        # a dropped block cannot pass the accounting


def test_no_intervention_after_success():
    r = make().reset(2); r.complete(); assert r.y and r.t < 12
    assert not r.eligible()[1] and not r.eligible()[2]
    with pytest.raises(ValueError): r.force(1)
    with pytest.raises(RuntimeError): r.step_base()


def test_a_scene_has_one_role_and_other_declared_scenes_are_never_selected():
    recs = some_scenes(4)
    for rec in recs: assert {b['role'] for b in rec['blocks']} <= {pilot.ROLE} and rec['role'] == pilot.ROLE
    assert {(x['scene'], x['role']) for x in pilot.pair_rows(recs)} == {(r['scene'], pilot.ROLE) for r in recs if r['blocks']}
    train = list(range(10000, 10030)); declared = {'splits/train': train, 'splits/dev': [10002], 'policy_eval/policy_qualify_B': [50300]}
    chosen, skipped = pilot.select_scenes(train, 24, declared, demo_seeds=[10005, 7])
    assert chosen == [s for s in train if s not in (10002, 10005)][:24] and [x['seed'] for x in skipped] == [10002, 10005]
    with pytest.raises(ValueError): pilot.select_scenes(train[:10], 24, declared)       # nothing is borrowed from another split


def test_a_failed_scene_stops_the_worker_and_never_marks_the_data_complete(tmp_path):
    def run_scene(seed):
        if seed == 3: raise RuntimeError('simulator error')
        return pilot.collect_scene(make(), seed, CFG, CPS, REPS)
    with pytest.raises(RuntimeError): pilot.collect_scenes([1, 2, 3, 4], tmp_path, run_scene)
    names = sorted(p.name for p in (tmp_path / 'scenes').iterdir())
    assert names == ['1.json', '2.json', '3.error.json'] and read_json(tmp_path / 'scenes' / '3.error.json')['completed'] is False
    meta, recs = pilot.finalize(tmp_path, [1, 2, 3, 4], CFG)
    assert meta['completed'] is False and len(recs) == 2 and len(meta['problems']) == 3
    ok, _ = pilot.finalize(tmp_path, [1, 2], CFG)
    assert ok['completed'] is False                                           # the error file of scene 3 is still an unscheduled leftover
    with pytest.raises(FileExistsError): pilot.collect_scenes([1], tmp_path, run_scene)   # no overwrite


def test_complete_run_is_marked_complete(tmp_path):
    pilot.collect_scenes([1, 2], tmp_path, lambda s: pilot.collect_scene(make(), s, CFG, CPS, REPS))
    meta, recs = pilot.finalize(tmp_path, [1, 2], CFG)
    assert meta['completed'] is True and meta['continuations'] == sum(r['counts']['continuations'] for r in recs)


# ---------------------------------------------------------------- restore audit
def test_audit_passes_on_a_correct_runtime():
    res = pilot.audit_scene(make(), 1, CFG, CPS); rep = pilot.audit_report([res])
    assert res['pass'] and rep['pass'] and rep['branches_executed']['retreat_retry'] > 0 and rep['branches_executed']['refresh'] > 0


class ForgetfulQueuePolicy(MockPolicy):
    """restore() puts back the history (the policy INPUT) but leaves the unexecuted queue of the previous branch."""
    def restore(self, s):
        q = self.queue; super().restore(s); self.queue = q


def test_audit_catches_a_wrong_queue_restore_even_though_the_inputs_are_identical():
    res = pilot.audit_scene(make(policy=ForgetfulQueuePolicy()), 1, CFG, CPS); first = res['checkpoints'][0]['checks']
    assert not res['pass'] and not pilot.audit_report([res])['pass']
    bad = first['after_retreat_retry_restored']
    assert 'queue' in bad['failed'] and bad['numeric_max_abs']['env_state'] == 0 and bad['numeric_max_abs']['history_state'] == 0
    import json; json.dumps(res, allow_nan=False)                             # a failing audit must still be writable as a report
    assert bad['live_observation_pass']                                       # observation and simulator state agree; only the queue is wrong
    with pytest.raises(RuntimeError): scene(1, policy=ForgetfulQueuePolicy()) # the collector refuses to branch from such a restore


class StaleObsEnv(MockEnv):
    """The environment's observation drifts with the number of resets while get_state_dict stays identical."""
    def __init__(self): super().__init__(); self.resets = 0

    def reset(self, seed): self.resets += 1; return super().reset(seed)

    def obs(self):
        o = super().obs(); o['agent']['qpos'] = o['agent']['qpos'] + 1e-3 * max(0, self.resets - 1); return o


def test_live_observation_is_compared_not_only_the_cached_copy():
    res = pilot.audit_scene(make(env=StaleObsEnv()), 1, CFG, CPS); c = res['checkpoints'][0]['checks']['repeat_restored']
    assert not res['pass'] and not c['live_observation_pass'] and c['numeric_max_abs']['env_state'] == 0
    with pytest.raises(RuntimeError): scene(1, env=StaleObsEnv())


class DriftingPolicy(MockPolicy):
    """Action chunks depend on how often the policy was called: a rerun of the same Continue differs."""
    n = 0

    def generate(self, episode, t, repeat=0):
        DriftingPolicy.n += 1; return super().generate(episode, t + DriftingPolicy.n, repeat)


def test_a_changed_no_intervention_duplicate_is_flagged_and_never_subtracted():
    DriftingPolicy.n = 0; rec = scene(1, policy=DriftingPolicy(), recheck_repeat=1)
    assert not rec['reproducibility_ok'] and any(not d['match'] for d in rec['no_intervention_duplicates'] + rec['continue_rechecks'])
    s = pilot.summarize([rec])
    assert s['reproducibility']['interpretation_withheld'] and s['reproducibility']['duplicate_mismatches'] + s['reproducibility']['recheck_mismatches'] > 0
    assert s['paired_blocks'] == rec['counts']['blocks']                       # the counts are reported as measured


def test_a_reproducibility_failure_keeps_the_rows_but_the_data_set_is_not_marked_complete(tmp_path):
    DriftingPolicy.n = 0
    pilot.collect_scenes([1], tmp_path, lambda s: pilot.collect_scene(make(policy=DriftingPolicy()), s, CFG, CPS, REPS))
    meta, recs = pilot.finalize(tmp_path, [1], CFG)
    assert len(recs) == 1 and recs[0]['completed'] and meta['completed'] is False and meta['reproducibility_ok'] is False


def test_clean_runs_reproduce_the_trunk_and_the_rerun():
    rec = scene(5, recheck_repeat=1)
    assert rec['reproducibility_ok'] and len(rec['continue_rechecks']) == rec['counts']['snapshots'] == len(rec['no_intervention_duplicates'])
    assert all(d['action_max_abs'] == 0 and d['final_state_max_abs'] == 0 for d in rec['no_intervention_duplicates'])


# ---------------------------------------------------------------- tables and scale
def test_tables_count_each_block_once_per_action_and_use_na_for_empty_denominators():
    recs = some_scenes(); s = pilot.summarize(recs); blocks = sum(r['counts']['blocks'] for r in recs)
    for a in ('refresh', 'retreat_retry'):
        f = s['by_action_and_checkpoint'][a]['all_checkpoints']
        assert f['blocks'] == blocks == f['continue_success_blocks'] + f['continue_failure_blocks']
        assert f['rescue'] + f['harm'] + f['both_succeed'] + f['both_fail'] == blocks
        assert sum(s['by_action_and_checkpoint'][a][f't{t}']['blocks'] for t in CPS if f't{t}' in s['by_action_and_checkpoint'][a]) == blocks
    assert s['continue_success_blocks'] == sum(b['y_continue'] for r in recs for b in r['blocks']) and s['paired_blocks'] == blocks
    assert len(pilot.scene_rows(recs)) == len(recs) and s['cells_not_run'] == 12 * sum(len(r['skips']) for r in recs)
    only_fail = [dict(x, outcome='both_fail') for x in pilot.pair_rows(recs)[:5]]
    assert pilot._four(only_fail)['H_branch'] == 'NA' and pilot._four(only_fail)['R_branch'] == 0
    assert 'interval' not in str(s['by_action_and_checkpoint']).lower() and 'wilson' not in str(s).lower()


def test_scale_uses_time_and_money_only_and_takes_a_prefix_size():
    kw = dict(usd_per_hour=.36, spent_usd=.2, cap_usd=1.5, reserve_usd=.25, workers=2, efficiency=.7)
    assert pilot.scale_decision([300, 300], 98, **kw)['chosen_scenes'] == 24
    slow = pilot.scale_decision([1000, 1000], 98, **kw); slower = pilot.scale_decision([9000, 9000], 98, **kw)
    assert slow['chosen_scenes'] == 12 and slower['chosen_scenes'] == 2 and not slower['options'][0]['projected_total_usd'] <= 1.5
    assert pilot.scale_decision([300, 300], 98, **{**kw, 'spent_usd': 1.4})['fits_cap'] is False
    with pytest.raises(ValueError): pilot.scale_decision([300, 300], 0, **kw)
