"""Stage 2C offline review: join, recount, leave-one-repeat-out choice, retreat events. Synthetic records only.

Passing these tests says the analysis code does what its rules say. It says nothing about the robot, the policy or any hypothesis.
"""
import copy, importlib.util, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location('stage2c', Path(__file__).resolve().parents[1] / 'scripts' / '21_stage2c_offline_review.py')
s2c = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s2c)
ARMS = s2c.ARMS


def make(spec, trunk=None):
    """spec: {scene: {checkpoint: {arm: [y of repeat 0..3]}}} -> (csv rows, scene records)."""
    rows = []; recs = []
    for scene, cps in spec.items():
        rec = {'scene': scene, 'trunk': {'success': (trunk or {}).get(scene, 0)}, 'snapshots': [], 'blocks': [], 'skips': [], 'no_intervention_duplicates': [], 'rechecks': [],
               'initial_state': {'cubeA': [0., 0., .02]}}
        for t, ys in cps.items():
            sid = f'{scene}:{t}:abc'; rec['snapshots'].append({'checkpoint': t, 'snapshot_id': sid})
            rec['no_intervention_duplicates'].append({'checkpoint': t, 'repeat': 0, 'against': 'trunk', 'match': True})
            for r in range(4):
                y = {a: ys[a][r] for a in ARMS}
                cells = [{'arm': a, 'experimental_arm': s2c.ARM_NAMES[a], 'block_repeat_id': r, 'policy_noise_repeat_id': r + 4 if a == 'C1' else r, 'y': y[a]} for a in ARMS]
                rec['blocks'].append({'scene': scene, 'checkpoint': t, 'block_repeat_id': r, 'snapshot_id': sid, **{f'y_{a}': y[a] for a in ARMS}, 'cells': cells,
                                      'c0_c1_queue_prefix_parity': {'pass': True}})
                rows.append({'scene': str(scene), 'checkpoint': str(t), 'block_repeat_id': str(r), 'snapshot_id': sid, **{f'y_{a}': str(y[a]) for a in ARMS},
                             'policy_noise_repeat_id_C1': str(r + 4), 'trunk_success': str(rec['trunk']['success']), 'grasped_at_checkpoint': 'False'})
        recs.append(rec)
    return rows, recs


Z = [0, 0, 0, 0]; O = [1, 1, 1, 1]
BASE = {1: {12: dict(C0=Z, C1=Z, F=[1, 0, 0, 0], R=O), 52: dict(C0=O, C1=O, F=O, R=Z)},
        2: {12: dict(C0=[1, 0, 1, 0], C1=[0, 1, 0, 0], F=[1, 1, 0, 0], R=[0, 0, 1, 1])}}


def blocks_of(spec, **kw):
    return s2c.join_blocks(*make(spec, **kw))


# ---------------------------------------------------------------- join and strict values
def test_join_accepts_consistent_records_and_keeps_every_block():
    b = blocks_of(BASE); assert len(b) == 12 and {x['repeat'] for x in b} == {0, 1, 2, 3}


def test_join_fails_when_a_block_is_tied_to_another_scene_or_checkpoint():
    rows, recs = make(BASE); rows[0]['checkpoint'] = '52'
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)                # collides with the real (1, 52, 0)
    rows, recs = make(BASE); rows[0]['scene'] = '2'
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)
    rows, recs = make(BASE); rows[0]['snapshot_id'] = '1:52:abc'
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)
    rows, recs = make(BASE); recs[0]['blocks'][0]['snapshot_id'] = '1:52:abc'
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)
    rows, recs = make(BASE); recs[0]['snapshots'][0]['snapshot_id'] = '2:12:abc'; recs[0]['blocks'][0]['snapshot_id'] = '2:12:abc'
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)


def test_join_fails_on_missing_or_duplicate_repeats_and_conflicting_values():
    rows, recs = make(BASE); del rows[3]
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)
    rows, recs = make(BASE); rows.append(dict(rows[0]))
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)
    rows, recs = make(BASE); del rows[3]; del recs[0]['blocks'][3]              # the same repeat missing on both sides
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)
    rows, recs = make(BASE); rows[0]['y_R'] = '0'
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)
    rows, recs = make(BASE); recs[0]['blocks'][0]['cells'][1]['policy_noise_repeat_id'] = 0
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)


@pytest.mark.parametrize('bad', ['False', 'True', '', '2', '0.0', 'yes', None, True, 2, 0.5])
def test_outcome_values_other_than_0_or_1_are_errors(bad):
    with pytest.raises(ValueError): s2c.bit(bad)
    rows, recs = make(BASE); rows[0]['y_C0'] = bad
    with pytest.raises(ValueError): s2c.join_blocks(rows, recs)


def test_boolean_text_is_read_strictly():
    assert s2c.boolean('False') is False and s2c.boolean('True') is True and s2c.bit('0') == 0 and s2c.bit(1) == 1
    for bad in ('false', '0', '', 1, None):
        with pytest.raises(ValueError): s2c.boolean(bad)


# ---------------------------------------------------------------- recount
def test_recount_sets_and_subsets_and_trunk_cross_table():
    rows, recs = make(BASE, trunk={1: 1}); b = s2c.join_blocks(rows, recs); rc = s2c.recount(b, recs)
    assert rc['success_blocks'] == {'C0': 6, 'C1': 5, 'F': 7, 'R': 6} and rc['blocks'] == 12 and rc['snapshots'] == 3
    assert rc['r_sets']['C0=1,R=0']['blocks'] == 5 and rc['r_sets']['C0=1,F=1,R=0']['blocks'] == 5 and rc['r_sets']['C0=0,R=1']['blocks'] == 5
    assert rc['r_sets']['C0=0,F=0,R=1']['blocks'] == 4 and rc['r_sets']['C0=C1=F=0,R=1']['blocks'] == 4 and all(rc['subset_checks'].values())
    x = rc['trunk_success_cross']['C0=1,R=0']; assert x['in_trunk_success_scenes']['blocks'] == 4 and x['in_trunk_failure_scenes']['blocks'] == 1
    assert rc['trunk_success_cross']['blocks_of_trunk_success_scenes']['of_which_C0=0'] == 4       # a trunk success does not make every C0 a success
    p = {(q['arm'], q['reference']): q for q in rc['pairs']}[('R', 'C0')]
    assert (p['0to1'], p['1to0'], p['1to1'], p['0to0']) == (5, 5, 1, 1) and p['reference_failure_blocks'] == 6


def test_scene_equal_weighting_with_different_numbers_of_checkpoints():
    spec = {1: {12: dict(C0=Z, C1=Z, F=O, R=Z), 52: dict(C0=Z, C1=Z, F=O, R=Z), 92: dict(C0=Z, C1=Z, F=O, R=Z)}, 2: {12: dict(C0=Z, C1=Z, F=Z, R=Z)}}
    rows, recs = make(spec); rc = s2c.recount(s2c.join_blocks(rows, recs), recs); p = {(q['arm'], q['reference']): q for q in rc['pairs']}[('F', 'C0')]
    assert p['delta_mean_outcome'] == 12 / 16 and p['scene_equal_delta'] == 0.5 and (p['scenes_positive'], p['scenes_zero']) == (1, 1)
    tr = s2c.transfer_rows(s2c.join_blocks(rows, recs)); summ, table = s2c.transfer_summary(tr)
    assert summ['policies']['always_F']['block_mean'] == 0.75 and summ['policies']['always_F']['scene_equal_mean'] == 0.5
    assert [t['rows_per_rule'] for t in table] == [12, 4]


# ---------------------------------------------------------------- leave-one-repeat-out choice
def test_first_max_ties_go_to_c0_then_f_then_r_and_c1_is_never_a_candidate():
    assert s2c.first_max({'C0': 2, 'F': 2, 'R': 2}) == 'C0' and s2c.first_max({'C0': 1, 'F': 2, 'R': 2}) == 'F' and s2c.first_max({'C0': 0, 'R': 1}) == 'R'
    assert s2c.first_max({'C0': 0, 'F': 0}) == 'C0'
    assert all('C1' not in v for v in s2c.RULES.values()) and 'C1' not in s2c.PRIORITY
    spec = {1: {12: dict(C0=Z, C1=O, F=Z, R=Z)}}                                # C1 always succeeds and must still never be chosen
    tr = s2c.transfer_rows(blocks_of(spec)); assert {r['chosen_arm'] for r in tr} == {'C0'} and all(r['y_C1_held_out'] == 1 for r in tr)


def test_changing_the_held_out_outcome_never_changes_the_choice_of_that_fold():
    b = blocks_of(BASE); base = {(r['rule'], r['scene'], r['checkpoint'], r['held_out_repeat']): r for r in s2c.transfer_rows(b)}
    for i in range(len(b)):
        for arm in ARMS:
            alt = copy.deepcopy(b); alt[i]['y'][arm] = 1 - alt[i]['y'][arm]
            for r in s2c.transfer_rows(alt):
                if (r['scene'], r['checkpoint'], r['held_out_repeat']) == (b[i]['scene'], b[i]['checkpoint'], b[i]['repeat']):
                    old = base[(r['rule'], r['scene'], r['checkpoint'], r['held_out_repeat'])]
                    assert r['chosen_arm'] == old['chosen_arm'] and [r[f'training_successes_{a}'] for a in 'C0 F R'.split()] == [old[f'training_successes_{a}'] for a in 'C0 F R'.split()]


def test_identical_outcomes_give_no_gain_and_every_repeat_is_held_out_once():
    spec = {1: {12: dict(C0=O, C1=O, F=O, R=O)}, 2: {12: dict(C0=Z, C1=Z, F=Z, R=Z), 52: dict(C0=[1, 0, 1, 0], C1=[1, 0, 1, 0], F=[1, 0, 1, 0], R=[1, 0, 1, 0])}}
    tr = s2c.transfer_rows(blocks_of(spec)); summ, _ = s2c.transfer_summary(tr)
    assert len(tr) == 3 * 3 * 4 and sorted(r['held_out_repeat'] for r in tr if r['rule'] == 'G_CF' and r['scene'] == 1) == [0, 1, 2, 3]
    for rule in s2c.RULES:
        assert summ['policies'][rule]['scene_equal_minus_always_C0'] == 0 and summ['policies'][rule]['block_mean'] == summ['policies']['always_C0']['block_mean']
    assert summ['G_CFR_minus_G_CF']['sum_of_differences'] == 0


def test_a_consistent_advantage_transfers_and_a_single_lucky_repeat_does_not():
    spec = {1: {12: dict(C0=Z, C1=Z, F=Z, R=O)},                               # R better in all four repeats
            2: {12: dict(C0=Z, C1=Z, F=Z, R=[0, 0, 1, 0])}}                    # R better in one repeat only
    b = blocks_of(spec); tr = [r for r in s2c.transfer_rows(b) if r['rule'] == 'G_CR']
    a = [r for r in tr if r['scene'] == 1]; c = [r for r in tr if r['scene'] == 2]
    assert [r['chosen_arm'] for r in a] == ['R'] * 4 and sum(r['y_chosen'] for r in a) == 4
    assert [r['chosen_arm'] for r in c] == ['R', 'R', 'C0', 'R'] and sum(r['y_chosen'] for r in c) == 0      # chosen from the lucky repeat, never confirmed
    snaps = {s['scene']: s for s in s2c.snapshot_rows(b)}
    assert (snaps[1]['k_R'], snaps[1]['n_R_only_vs_C0_F'], snaps[1]['repeats_R_only_vs_C0_F']) == (4, 4, '0 1 2 3')
    assert (snaps[2]['k_R'], snaps[2]['n_R_only_vs_C0_F'], snaps[2]['repeats_R_only_vs_C0_F']) == (1, 1, '2') and snaps[2]['mean_R_minus_C0'] == 0.25


def test_cfr_minus_cf_lists_only_the_rows_where_the_choice_differs():
    spec = {1: {12: dict(C0=Z, C1=Z, F=Z, R=O)}, 2: {12: dict(C0=Z, C1=Z, F=O, R=Z)}}
    summ, _ = s2c.transfer_summary(s2c.transfer_rows(blocks_of(spec))); g = summ['G_CFR_minus_G_CF']
    assert g['rows_where_the_choice_differs'] == 4 and g['scenes'] == [1] and g['sum_of_differences'] == 4
    assert summ['choices']['G_CFR']['R']['rows'] == 4 and summ['choices']['G_CFR']['F']['rows'] == 4 and summ['choices']['G_CF']['F']['held_out_vs_C0']['0to1'] == 4


# ---------------------------------------------------------------- retreat events
def trace(n, t0=0, grasp=0, z=.02, tcp=(0., 0., .02), A=(0., 0.), B=(.2, 0.), on=0, success=0, grip=-1.):
    r = np.zeros((n, 18), np.float32); r[:, 0] = np.arange(t0 + 1, t0 + n + 1); r[:, 1:4] = tcp; r[:, 4:6] = A; r[:, 6] = z; r[:, 7:9] = B; r[:, 9] = .02
    r[:, 10] = grasp; r[:, 11] = on; r[:, 13] = success; r[:, 17] = grip
    return r


def test_time_axis_and_retreat_end_come_from_the_record_not_from_a_fixed_number():
    trunk = trace(200); trunk[11, 10] = 1                                       # grasped exactly at step 12 (row 11)
    br = trace(188, t0=12, grasp=1); ev = s2c.retreat_events(trunk, br, 12, 7, .02, 0, 200, 'horizon', 19)
    assert ev['cp_grasped'] and ev['return_step'] == 19 and ev['remaining_steps_at_return'] == 181 and ev['retreat_steps_used'] == 7 and not ev['insufficient_trace']
    assert s2c.retreat_events(trunk, br, 12, 7, .02, 0, 200, 'horizon', 37)['insufficient_trace']            # inference log disagrees with steps_used
    assert s2c.retreat_events(trunk, br[:-1], 12, 7, .02, 0, 200, 'horizon', 19)['insufficient_trace']       # a row is missing
    assert s2c.retreat_events(trunk, trace(188, t0=13), 12, 7, .02, 0, 200, 'horizon', 19)['insufficient_trace']


def test_grasp_loss_is_reported_as_an_observation_without_a_cause_label():
    trunk = trace(12, grasp=1, z=.05); br = trace(188, t0=12, grasp=1, z=.05); br[3:, 10] = 0; br[3:, 6] = .02        # flag off from the 4th retreat step
    ev = s2c.retreat_events(trunk, br, 12, 10, .02, 0, 200, 'horizon', 22)
    assert ev['grasp_loss_during_retreat'] and ev['first_grasp_loss_step_during_retreat'] == 16 and ev['lift_lost_during_retreat'] and ev['never_grasped_after_return']
    assert not ev['grasp_kept_through_retreat'] and ev['ended_at_horizon']
    words = ' '.join(list(ev) + list(s2c.EVENT_FLAGS)).lower()
    for cause in ('drop', 'time_only', 'time_cause', 'ood', 'off_manifold', 'collision', 'because'): assert cause not in words
    assert len(s2c.NOT_CLAIMED) >= 4


def test_grasp_release_on_the_target_cube_is_kept_apart_from_other_grasp_losses():
    trunk = trace(12, grasp=1, z=.05); br = trace(60, t0=12, grasp=1, z=.06, A=(.2, 0.)); br[40:, 10] = 0; br[39:, 11] = 1; br[-1, 13] = 1
    ev = s2c.retreat_events(trunk, br, 12, 5, .02, 1, 72, 'success', 17)
    assert ev['success_after_return'] and ev['grasp_loss_after_return'] and ev['grasp_loss_after_return_while_on_cubeB'] and ev['first_grasp_loss_step_after_return'] == 53
    assert ev['grasp_kept_through_retreat'] and not ev['retained_grasp_no_success_by_horizon'] and ev['first_aligned_step_after_return'] == 18


def test_cube_b_displacement_and_reference_arm_at_the_same_physical_step():
    trunk = trace(12); br = trace(188, t0=12); br[2:, 7] = .21                  # cube B shifted by 1 cm during the retreat
    ev = s2c.retreat_events(trunk, br, 12, 6, .02, 0, 200, 'horizon', 18)
    assert ev['cubeB_moved_during_retreat'] and not ev['cubeB_moved_after_return'] and ev['no_reapproach_before_horizon'] is False       # tcp is at the cube
    c0 = trace(30, t0=12); c0[5:, 10] = 1
    assert s2c.reference_at(c0, 12, 18, .02, {})['grasped'] is True and s2c.reference_at(c0, 12, 17, .02, {})['grasped'] is False
    assert s2c.reference_at(c0, 12, 100, .02, {})['ended_before'] is True
    assert s2c.reference_at(c0, 12, 12, .02, {'grasped': False}) == {'grasped': False, 'ended_before': False}


# ---------------------------------------------------------------- read-only on real-shaped inputs
def test_the_analysis_does_not_change_any_input_file(tmp_path, monkeypatch):
    res = tmp_path / 'results'; run = tmp_path / 'run'; (res / 'scenes').mkdir(parents=True); (run / 'branches' / 'traces').mkdir(parents=True)
    (run / 'branches' / 'inputs').mkdir(); (run / 'branches' / 'scenes').mkdir()
    rows, recs = make({1: {12: dict(C0=Z, C1=Z, F=[1, 0, 0, 0], R=O)}})
    for rec in recs:
        for b in rec['blocks']:
            for c in b['cells']: c.update(steps=20, reason='horizon', first_inference_step=15 if c['arm'] == 'R' else 16, retreat={'steps_used': 3} if c['arm'] == 'R' else None)
        text = json.dumps(rec); (res / 'scenes' / '1.json').write_text(text); (run / 'branches' / 'scenes' / '1.json').write_text(text)
        np.savez(run / 'branches' / 'traces' / '1.npz', columns=np.array(list(s2c.COL)), trunk=trace(20),
                 **{f'b_12_{r}_{a}': trace(8, t0=12) for r in range(4) for a in ARMS})
    fields = {'online_inputs': {'queue': ''}}; np.savez(run / 'branches' / 'inputs' / '1.npz', t12_queue=np.zeros((4, 4), np.float32))
    import csv
    with (res / 'pair_outcomes_four_arm.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for name, value in (('summary.json', {}), ('plan.json', {'code': {}, 'repository': {'head': 'x'}}), ('observation_fields.json', fields)):
        (res / name).write_text(json.dumps(value))
    for name in ('scene_equal_weight_summary.csv', 'refresh_retreat_complementarity.csv', 'pair_summary.csv', 'skip_reasons.csv'): (res / name).write_text('a\n')
    lines = [f"{s2c.sha256(p)}  {p.as_posix()}" for p in sorted(run.rglob('*')) if p.is_file()]; (res / 'backup_sha256_stage2b_001.txt').write_text('\n'.join(lines) + '\n')
    a = SimpleNamespace(out=str(tmp_path / 'out'), results=res.as_posix(), run=run.as_posix(), backup=str(tmp_path / 'bk' / 'none.tgz'), label='after')
    monkeypatch.setattr(s2c, 'recount', lambda b, r, *x: {**s2c.__dict__['_recount'](b, r), 'all_checks_pass': True})
    before = s2c.hashes(s2c.input_files(a)); s2c.cmd_manifest(a); s2c.cmd_run(a); rep = s2c.cmd_verify(a)
    assert rep['unchanged'] and s2c.hashes(s2c.input_files(a)) == before
    out = tmp_path / 'out'; assert json.loads((out / 'run_status.json').read_text())['stage2c'] == 'COMPLETED_POSTHOC_OFFLINE_DIAGNOSTIC'
    assert len(s2c.read_csv(out / 'retreat_event_rows.csv')) == 4 and len(s2c.read_csv(out / 'repeat_transfer_rows.csv')) == 12
    (res / 'summary.json').write_text('{"x": 1}')
    with pytest.raises(SystemExit): s2c.cmd_run(a)                              # a changed input refuses the run
    a.label = 'again'
    with pytest.raises(SystemExit): s2c.cmd_verify(a)


def test_a_branch_that_ends_during_the_retreat_is_not_called_a_success_after_return():
    trunk = trace(12, grasp=1, z=.05); br = trace(4, t0=12, grasp=1, z=.06); br[-1, 13] = 1
    ev = s2c.retreat_events(trunk, br, 12, 4, .02, 1, 16, 'success', None)        # ended on the 4th retreat step, no inference afterwards
    assert not ev['insufficient_trace'] and ev['ended_during_retreat'] and not ev['success_after_return'] and ev['steps_from_return_to_end'] == 0
    later = trace(9, t0=12, grasp=1, z=.06); later[-1, 13] = 1
    ev = s2c.retreat_events(trunk, later, 12, 4, .02, 1, 21, 'success', 16); assert ev['success_after_return'] and not ev['ended_during_retreat']


def test_cases_are_only_taken_from_blocks_with_a_usable_trace():
    row = lambda rep, bad: {'scene': 1, 'checkpoint': 12, 'repeat': rep, 'snapshot_id': 's', 'R_vs_C0': '0to1', 'in_C0=1,F=1,R=0': 0, 'in_C0=0,F=0,R=1': 1,
                            'in_C0=C1=F=0,R=1': 1, 'trunk_success': 0, 'trace_file': 'f', 'trace_key': f'b_12_{rep}_R', 'insufficient_trace': bad,
                            **({} if bad else {k: False for k in s2c.EVENT_FLAGS[:-1]})}
    snap = [{'scene': 1, 'checkpoint': 12, 'n_R_only_vs_C0_F': 2, 'repeats_R_only_vs_C0_F': '0 1', 'n_C0success_Rfail': 0, 'repeats_C0success_Rfail': ''}]
    cases = s2c.case_index([row(0, True), row(1, False)], snap); assert [c['repeat'] for c in cases] == [1]
    assert s2c.case_index([row(0, True), row(1, True)], snap) == []


def test_outputs_cannot_be_written_into_an_input_location(tmp_path):
    for inside in ('results/x', 'run/branches/y', 'backup/z'):
        a = SimpleNamespace(out=str(tmp_path / inside), results=str(tmp_path / 'results'), run=str(tmp_path / 'run'), backup=str(tmp_path / 'backup' / 'b.tgz'), label='v')
        with pytest.raises(SystemExit): s2c.cmd_manifest(a)
        assert not (tmp_path / inside).exists()


s2c._recount = s2c.recount
