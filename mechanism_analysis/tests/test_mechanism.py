"""Synthetic checks of the mechanism analysis. Synthetic scene IDs (9_000_000+) in isolated temp folders only; no C1 ledger,
no reserved seed, no simulator."""
import gzip, hashlib, json, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(HERE), str(HERE.parent / 'src')]
import mechanism as M
import s6_validity as S6

SYN = 9_000_000


def row(t, tcp, A, B, grip=-1.0, grasped=True, success=False):
    return {'t': t, 'tcp': list(tcp), 'cubeA': list(A), 'cubeB': list(B), 'is_cubeA_grasped': grasped, 'success': success, 'action': [0, 0, 0, grip], 'finger_qpos': [0.02, 0.02]}


def scene(seed, T=5, K=3, move=0.004, y=(0, 1, 1), push=0.0, decision='ACT', why=None, memory=None, completed=True):
    """Trunk rows 1..T; arms C0, VISION_CENTER_V2, CENTER_ORACLE; v2 executes K steps of `move` in +x."""
    A0 = np.array([0.0, 0.0, 0.06]); B0 = np.array([0.02, 0.0, 0.02]); tcp0 = np.array([0.0, 0.0, 0.05])
    trunk = [row(t, tcp0, A0, B0) for t in range(1, T + 1)]
    def branch(k, mv, pushv, succ_at=None):
        out = []
        for i in range(1, 12):
            d = min(i, k) * mv; pb = min(i, k) * pushv
            out.append(row(T + i, tcp0 + [d, 0, 0], A0 + [d, 0, 0], B0 - [pb, 0, 0], grip=(1.0 if i > k + 2 else -1.0), grasped=i <= k + 3, success=bool(succ_at and i >= succ_at)))
        return out
    tr = {'trunk': trunk, 'C0': branch(0, 0, 0, 9 if y[0] else None), 'VISION_CENTER_V2': branch(K if decision == 'ACT' else 0, move, push, 9 if y[1] else None), 'CENTER_ORACLE': branch(3, 0.005, 0, 9 if y[2] else None)}
    steps = [{'commanded_xy_m': [move, 0.0], 'target_distance_from_start_m': (i + 1) * move} for i in range(K)] if decision == 'ACT' else []
    rd = {'t': T, 'why': why, 'target_src': 'memory_hand' if memory else ('frame' if why is None else None), 'hold_src': 'frame', 'u': 0.004, 'offset_norm_m': 0.02, 'gap_m': 0.01,
          'memory_reason': None if why != 'NO_TARGET_ESTIMATE' else 'never_seen', 'target_xy': None if why else [0.021, 0.0], 'held_xyz': None if why == 'NO_HOLD_ESTIMATE' else [0.0, 0.0, 0.06]}
    if memory: rd['memory_target'] = {'xy': [0.021, 0.0], 'u': 0.007, 'src': 'hand', 't_last': memory, 'exposed_steps': 0, 'confirmations': 4}
    cells = [{'arm': 'C0', 'y': y[0], 'acted': False, 'control': None, 'control_steps': 0, 'steps': 16, 'reason': 'x', 'first_inference_step': T + 1, 'gt_release_after_trigger': {}, 'gt_reading_error': None},
             {'arm': 'VISION_CENTER_V2', 'y': y[1], 'acted': decision == 'ACT', 'control_steps': K if decision == 'ACT' else 0, 'steps': 16, 'reason': 'x', 'first_inference_step': T + K + 1,
              'control': {'decision': decision, 'readings': [rd], 'steps': steps, 'end_reason': 'within_tolerance' if decision == 'ACT' else None},
              'gt_reading_error': {'true_offset_m': 0.02, 'offset_error_m': 0.001, 'bound_m': 0.004, 'error_within_bound': True}, 'gt_release_after_trigger': {'release_step': T + K + 4}},
             {'arm': 'CENTER_ORACLE', 'y': y[2], 'acted': True, 'control_steps': 3, 'steps': 16, 'reason': 'x', 'first_inference_step': T + 4, 'control': {'steps': [{'commanded_xy_m': [0.005, 0]}] * 3, 'end_reason': 'within_tolerance'}, 'gt_release_after_trigger': {}}]
    rec = {'scene': seed, 'role': 'synthetic', 'arms': ['C0', 'VISION_CENTER_V2', 'CENTER_ORACLE'], 'triggered': True, 'trigger': {'step': T}, 'initial_state': row(0, tcp0, A0, B0), 'cells': cells, 'memory': {'hold_steps': [1, 3]}, 'completed': completed}
    return rec, {'seed': seed, 'traces': tr}


MOCK_PLAN = 'ab' * 32                                     # a synthetic plan hash: never the frozen C1 plan


def write_run(root, scenes, stage='pass_train_seed_1'):
    d = Path(root) / stage; (d / 'scenes').mkdir(parents=True, exist_ok=True); (d / 'traces').mkdir(exist_ok=True)
    for rec, tr in scenes:
        p = d / 'traces' / f'trace_{rec["scene"]}.json.gz'
        with gzip.open(p, 'wt', encoding='utf-8') as f: json.dump(tr, f)
        (d / 'frames').mkdir(exist_ok=True); fp = d / 'frames' / f'frames_{rec["scene"]}.npz'; fp.write_bytes(b'synthetic frames ' + str(rec['scene']).encode())
        rec = {'policy_id': stage.replace('pass_', '', 1), 'plan_sha256': MOCK_PLAN, 'frames': f'frames/{fp.name}', 'frames_sha256': hashlib.sha256(fp.read_bytes()).hexdigest(), **rec, 'trace': f'traces/{p.name}', 'trace_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        (d / 'scenes' / f'{rec["scene"]}.json').write_text(json.dumps(rec), encoding='utf-8')
    return d


def mock_manifest(tmp, scenes, arms=('C0', 'VISION_CENTER_V2', 'CENTER_ORACLE'), policies=('train_seed_1', 'train_seed_2'), name='mock_expected.json', **kw):
    """A MOCK expected manifest (review F2): the small synthetic set the gate compares the run with."""
    p = Path(tmp) / name; p.write_text(json.dumps({'kind': 'MOCK_EXPECTED_MANIFEST', 'policies': list(policies), 'scenes': list(scenes), 'arms': list(arms), 'record_plan_sha256': MOCK_PLAN, **kw}), encoding='utf-8'); return p


def analyse(tmp_path, scenes):
    write_run(tmp_path, scenes); run = M.Run(tmp_path, 'MOCK', 'syn', {'C1_train_seed_1': 'pass_train_seed_1'}, 'train_seed_1'); return M.analyse_run(run)


def test_source_mixing_duplicate_keys_and_wrong_arm_or_policy_are_detected(tmp_path):
    res = analyse(tmp_path, [scene(SYN + 1)]); rows = res['rows']
    assert M.check_keys(rows) == []
    assert any('duplicate' in p for p in M.check_keys(rows + rows[:1]))
    assert any('mixed' in p for p in M.check_keys(rows + [{**rows[0], 'source_kind': 'PHASE_B_ORIGINAL'}]))
    assert any('unknown arm' in p for p in M.check_keys([{**rows[0], 'arm': 'V2S'}]))
    assert any('no policy' in p for p in M.check_keys([{**rows[0], 'policy': None}]))
    with pytest.raises(ValueError): M.Run(tmp_path, 'REPLAY_AS_ORIGINAL', 'x', {}, 'p')


def test_totals_are_conserved_and_act_differs_from_movement(tmp_path):
    sc = [scene(SYN + 1), scene(SYN + 2, move=0.0002), scene(SYN + 3, decision='ALIGNED', why=None), scene(SYN + 4, decision='NO_HOLD_ESTIMATE', why='NO_HOLD_ESTIMATE')]
    res = analyse(tmp_path, sc); f = M.funnel(res['rows'], ['C0', 'VISION_CENTER_V2', 'CENTER_ORACLE']); d = {(r['arm'], r['step']): r for r in f}
    assert M.check_funnel(f) == []
    assert d[('VISION_CENTER_V2', 'decision_ACT')]['numerator'] == 2 and d[('VISION_CENTER_V2', 'control_steps_executed')]['numerator'] == 2
    assert d[('VISION_CENTER_V2', 'actual_movement')]['numerator'] == 1          # 3 x 0.2 mm is an ACT, not a movement
    assert d[('VISION_CENTER_V2', 'decision_ACT')]['numerator'] + d[('VISION_CENTER_V2', 'abstained_or_no_action')]['numerator'] == d[('VISION_CENTER_V2', 'triggered')]['numerator']
    miss = [r for r in f if r['arm'] == 'VISION_CENTER_V2' and r['step'].startswith('oracle_rescue_missed:')]
    assert sum(r['numerator'] for r in miss) == d[('VISION_CENTER_V2', 'oracle_rescue_missed')]['numerator']


def test_missing_abstention_and_errors_are_never_success_or_zero(tmp_path):
    rec, tr = scene(SYN + 1); rec['cells'][1]['y'] = None
    res = analyse(tmp_path, [(rec, tr), scene(SYN + 2, completed=False)])
    v2 = [r for r in res['rows'] if r['arm'] == 'VISION_CENTER_V2' and r['scene'] == SYN + 1][0]
    assert v2['y_arm'] is None and v2['outcome_vs_c0'] is None
    assert any(e['status'] == 'INCOMPLETE_RECORD' for e in res['errors'])
    d = Path(tmp_path) / 'pass_train_seed_1' / 'scenes'; (d / f'{SYN + 3}.error.json').write_text('{}', encoding='utf-8')
    res2 = M.analyse_run(M.Run(tmp_path, 'MOCK', 'syn', {'C1_train_seed_1': 'pass_train_seed_1'}, 'train_seed_1'))
    assert any(e['status'] == 'ERROR_RECORD' for e in res2['errors'])


def test_time_order_and_pre_post_step_alignment(tmp_path):
    res = analyse(tmp_path, [scene(SYN + 1)]); v2 = [r for r in res['rows'] if r['arm'] == 'VISION_CENTER_V2'][0]
    assert (v2['t_trigger'], v2['t_control_start'], v2['t_control_end']) == (5, 5, 8) and M.check_time_order(v2) == []
    assert v2['t_first_open'] == 11 and v2['steps_control_end_to_open'] == 3          # rows T+i with i > K+2 carry the open command
    assert abs(v2['tcp_xy_net_m'] - 0.012) < 1e-9                                       # trunk row T vs branch row T+K
    assert M.check_time_order({**v2, 't_policy_resume': 6}) == ['policy resumes before control ended']
    assert 'control starts before the trigger' in M.check_time_order({**v2, 't_control_start': 4})


def test_zero_denominators_are_na():
    assert M.ratio(0, 0) is None and M.ratio(3, 0) is None and M.ratio(1, 4) == 0.25
    assert M.cos(np.array([0.0005, 0]), np.array([1.0, 0])) is None
    assert M.dist([]) == {'n': 0, 'median': None, 'p90': None, 'max': None}


def test_memory_identity_is_a_vector_identity_and_magnitudes_do_not_add(tmp_path):
    rec, tr = scene(SYN + 1, memory=2); tr = tr['traces']
    for x in tr['trunk']:
        if x['t'] > 2: x['cubeB'] = [0.02 - 0.003, 0.004, 0.02]       # the target moved after the memory's last update
    rec['initial_state']['cubeB'] = [0.02, 0.0, 0.02]
    m = M.memory_decomposition(rec, rec['cells'][1], tr)
    assert m['status'] == 'ESTIMATED' and m['identity_residual_m'] < 1e-12 and m['age_steps'] == 3
    assert m['err_now_m'] < m['err_at_store_m'] + m['target_move_since_store_m'] - 1e-6      # triangle: not a sum


def test_push_assisted_alignment_is_flagged(tmp_path):
    res = analyse(tmp_path, [scene(SYN + 1, move=0.0005, push=0.004)]); v2 = [r for r in res['rows'] if r['arm'] == 'VISION_CENTER_V2'][0]
    assert v2['target_moved'] and v2['push_assisted'] and v2['reduction_from_target_move_m'] > v2['reduction_from_held_move_m']


def test_miss_classification_follows_code_order():
    base = {'arm': 'VISION_CENTER_V2', 'acted': False, 'moved': None, 'memory_reason': None}
    assert M.classify_miss({**base, 'decision': 'NO_HOLD_ESTIMATE'}) == 'HOLD_ESTIMATE_NOT_ACCEPTED'
    assert M.classify_miss({**base, 'decision': 'NO_TARGET_ESTIMATE', 'memory_reason': 'never_seen'}) == 'NO_OBSERVATION'
    assert M.classify_miss({**base, 'decision': 'NO_TARGET_ESTIMATE', 'memory_reason': 'stale'}) == 'MEMORY_UNUSABLE_OR_INVALIDATED'
    assert M.classify_miss({**base, 'decision': 'LOW_CARRY'}) == 'ABSTAIN_HEIGHT'
    assert M.classify_miss({**base, 'decision': 'ALIGNED_WITHIN_UNCERTAINTY'}) == 'JUDGED_ALIGNED'
    assert M.classify_miss({**base, 'decision': 'ACT', 'acted': True, 'moved': False}) == 'ACT_WITHOUT_MOVEMENT'
    assert M.classify_miss({**base, 'decision': 'ACT', 'acted': True, 'moved': True, 'offset_after_m': 0.02}) == 'MOVED_ALIGNMENT_INSUFFICIENT'
    assert M.classify_miss({**base, 'decision': 'ACT', 'acted': True, 'moved': True, 'offset_after_m': 0.004}) == 'ALIGNED_BETTER_BUT_FAILED_AFTER_RESUME'
    assert M.classify_miss({**base, 'arm': 'VISION_CENTER_V1', 'decision': 'NO_TARGET_ESTIMATE', 'target_fit_present': None}) == 'UNCLASSIFIABLE_MISSING_RECORD'


def test_trace_hash_mismatch_is_refused(tmp_path):
    d = write_run(tmp_path, [scene(SYN + 1)]); p = d / 'scenes' / f'{SYN + 1}.json'; r = json.loads(p.read_text()); r['trace_sha256'] = '0' * 64; p.write_text(json.dumps(r))
    res = M.analyse_run(M.Run(tmp_path, 'MOCK', 'syn', {'s': 'pass_train_seed_1'}, 'train_seed_1'))
    assert res['rows'] == [] and [e['status'] for e in res['errors']] == ['TRACE_SHA_MISMATCH']          # never used; recorded (review F2), not a traceback


def test_s6_fail_is_preserved_and_shape_matching_is_exact():
    rows = [{'diff_m': 0.01 * w, 'vrel_dot_n_mps': -w, 'dt_sub_s': 0.01, 'kind': 'k', 'phase': 'p', 'speed_rel_mps': w, 'normal_z_abs': 1.0, 'ang_speed_a_rps': 0.0} for w in (0.15, 0.2, 0.3, 0.4, 0.5)]
    s = S6.summarise(rows)
    assert s['official_S6']['status'] == 'FAIL' and s['explanatory_fit_diff_vs_vreln_dtsub']['status'] == 'EXPLANATORY_ONLY'
    assert s['explanatory_fit_diff_vs_vreln_dtsub']['residual_over_tol_after_fit'] == 0          # a perfect explanation still leaves S6 FAIL
    c = [{'a': 'x', 'shape_a': 0, 'b': 'y', 'shape_b': 1, 'points': [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0.002], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0.001]]}]
    n, sep = S6.min_point(c, 'x', 0, 'y', 1); assert sep == 0.001 and list(n) == [1, 0, 0]
    n2, _ = S6.min_point(c, 'y', 1, 'x', 0); assert list(n2) == [-1, 0, 0]                   # swapped orientation flips the normal
    assert S6.min_point(c, 'x', 1, 'y', 1) == (None, None)                                   # another shape index never matches
    assert S6.summarise([])['official_S6']['status'] == 'NOT_OBSERVABLE'


def test_offline_replay_arms_have_no_outcome(tmp_path):
    p = tmp_path / 'off.json'
    p.write_text(json.dumps({'rows': [{'stage': 'B1', 'scene': SYN + 1, 'triggered': True, 'arms': {'VISION_CENTER_V2': {'decision': 'ACT', 'act': True, 'target_src': 'frame'}, 'V2_NO_HAND_MEMORY': {'decision': 'ALIGNED', 'act': False, 'target_src': 'memory_base'}}}]}))
    rows = M.offline_rows(p); r = [x for x in rows if x['arm'] == 'V2_NO_HAND_MEMORY'][0]
    assert r['outcome'] == 'NOT_RUN' and r['motion'] == 'NOT_RUN' and r['act_differs_from_v2'] and r['source_kind'] == 'OFFLINE_DECISION_REPLAY'
    t = M.ablation_table(rows); assert t[0]['outcome_difference'] == 'NOT_RUN' and t[0]['act_lost_vs_v2'] == 1


def test_c1_command_reads_a_run_keeps_inputs_and_refuses_overwrite(tmp_path):
    run = tmp_path / 'run'; sc = [scene(SYN + 1), scene(SYN + 2, y=(1, 0, 1))]; write_run(run, sc); write_run(run, sc, stage='pass_train_seed_2')
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    cmd = [sys.executable, str(HERE / 'run_analysis.py'), 'c1', '--run', str(run), '--out', str(tmp_path / 'out'), '--source-kind', 'MOCK', '--expected', str(mock_manifest(tmp_path, [SYN + 1, SYN + 2]))]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace'); assert r.returncode == 3, (r.stderr or '') + (r.stdout or '')
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}; assert before == after
    man = json.loads((tmp_path / 'out' / 'analysis_manifest.json').read_text(encoding='utf-8')); assert man['status'] == 'INCOMPLETE' and man['source_kind'] == 'MOCK'
    assert man['problems'] and all(p.startswith('MOCK_MANIFEST_REDUCED') for p in man['problems'])      # a 3-arm synthetic set is never a complete C1-format result (review f1)
    assert subprocess.run(cmd, capture_output=True, text=True).returncode != 0                 # refuses an existing output folder
    r2 = subprocess.run([sys.executable, str(HERE / 'run_analysis.py'), 'c1', '--run', str(tmp_path / 'nothing'), '--out', str(tmp_path / 'o2')], capture_output=True, text=True); assert r2.returncode != 0


def test_review_m1_missing_outcomes_leave_the_denominators_and_fail_the_check(tmp_path):
    rec, tr = scene(SYN + 1, y=(1, 1, 1)); rec['cells'][1]['y'] = None
    res = analyse(tmp_path, [(rec, tr), scene(SYN + 2, y=(1, 1, 1))]); f = M.funnel(res['rows'], ['C0', 'VISION_CENTER_V2', 'CENTER_ORACLE']); d = {(r['arm'], r['step']): r for r in f}
    assert d[('VISION_CENTER_V2', 'overall_harm_of_c0_success')]['denominator'] == 1 and d[('VISION_CENTER_V2', 'triggered_harm_of_c0_success')]['denominator'] == 1    # the scene without an outcome is not a harm-free success
    assert d[('VISION_CENTER_V2', 'outcome_missing_completeness_error')]['numerator'] == 1
    assert any('outcomes missing' in p for p in M.check_funnel(f))
    assert any('no recorded outcome' in p for r in res['rows'] for p in M.check_integrity(r))


def test_review_m1_a_missing_motion_row_is_na_not_no_movement(tmp_path):
    rec, tr = scene(SYN + 1); tr['traces']['VISION_CENTER_V2'] = [x for x in tr['traces']['VISION_CENTER_V2'] if x['t'] != 8]     # row T+K removed
    res = analyse(tmp_path, [(rec, tr)]); v2 = [r for r in res['rows'] if r['arm'] == 'VISION_CENTER_V2'][0]
    assert v2['moved'] is None and v2['motion_record_missing'] and M.check_integrity(v2)


def test_review_m2_a_missing_intermediate_control_row_and_an_untriggered_missing_outcome_are_errors(tmp_path):
    rec, tr = scene(SYN + 1); tr['traces']['VISION_CENTER_V2'] = [x for x in tr['traces']['VISION_CENTER_V2'] if x['t'] != 7]     # row T+2 of K=3 removed
    rec2, tr2 = scene(SYN + 2); rec2['triggered'] = False; rec2['cells'][0]['y'] = None
    res = analyse(tmp_path, [(rec, tr), (rec2, tr2)])
    v2 = [r for r in res['rows'] if r['arm'] == 'VISION_CENTER_V2' and r['scene'] == SYN + 1][0]; c0 = [r for r in res['rows'] if r['arm'] == 'C0' and r['scene'] == SYN + 2][0]
    assert v2['moved'] is None and v2['motion_record_missing']
    assert 'no recorded outcome' in M.check_integrity(c0)
    import run_analysis as RA
    s = RA.summaries(res, []); assert s['motion_VISION_CENTER_V2']['moved'] == 0          # review m3: the summaries do not crash on a missing motion row


def test_review_m1_s6_with_excluded_rows_is_never_pass():
    rows = [{'diff_m': 0.0001 * w, 'vrel_dot_n_mps': -w, 'dt_sub_s': 0.01, 'kind': 'k', 'phase': 'p', 'speed_rel_mps': w, 'normal_z_abs': 1.0, 'ang_speed_a_rps': 0.0} for w in (0.1, 0.2, 0.3)]
    assert S6.summarise(rows)['official_S6']['status'] == 'PASS'
    assert S6.summarise(rows, excluded={'rows_without_contacts_or_link_poses': 1})['official_S6']['status'] == 'NOT_OBSERVABLE'
    assert S6.summarise(rows + [{**rows[0], 'diff_m': 0.002}], excluded={'separated_pairs_not_reconstructed': 3})['official_S6']['status'] == 'FAIL'


def test_review_m1_mock_c1_outputs_are_labelled_not_a_result(tmp_path):
    run = tmp_path / 'run'; write_run(run, [scene(SYN + 1)])
    r = subprocess.run([sys.executable, str(HERE / 'run_analysis.py'), 'c1', '--run', str(run), '--out', str(tmp_path / 'out'), '--source-kind', 'MOCK', '--expected', str(mock_manifest(tmp_path, [SYN + 1]))], capture_output=True, text=True, encoding='utf-8')
    assert r.returncode == 3                                                                     # pass train_seed_2 missing: INCOMPLETE
    assert (tmp_path / 'out' / 'MOCK_NOT_A_RESULT.txt').exists()
    man = json.loads((tmp_path / 'out' / 'analysis_manifest.json').read_text(encoding='utf-8')); assert man['is_research_result'] is False and man['status'] == 'INCOMPLETE'
    import csv
    assert any(p.startswith('MISSING_POLICY_PASS') for p in man['problems']) and not (tmp_path / 'out' / 'C1_MECHANISM_REPORT.md').exists()


def test_review_m4_a_row_without_poses_or_grasp_flag_is_na_never_a_value(tmp_path):
    rec, tr = scene(SYN + 1)
    for x in tr['traces']['VISION_CENTER_V2']:
        if x['t'] == 7: del x['cubeB']
        if x['t'] == 8: del x['is_cubeA_grasped']
    res = analyse(tmp_path, [(rec, tr)]); v2 = [r for r in res['rows'] if r['arm'] == 'VISION_CENTER_V2'][0]
    assert v2['moved'] is None and v2['motion_record_missing'] and v2.get('grasped_after') is None
    rec2, tr2 = scene(SYN + 2)
    for x in tr2['traces']['VISION_CENTER_V2']:
        if x['t'] == 8: del x['is_cubeA_grasped']
    d = tmp_path / 'b'; res2 = analyse(d, [(rec2, tr2)]); w = [r for r in res2['rows'] if r['arm'] == 'VISION_CENTER_V2'][0]
    assert w['moved'] is True and w['grasped_before'] is True and w['grasped_after'] is None
    import run_analysis as RA
    assert RA.summaries(res2, [])['motion_VISION_CENTER_V2']['grasp_lost_during_control'] == 0      # an absent flag is not a grasp loss


def test_review_m5_malformed_memory_and_hold_rows_are_na_or_errors_never_values(tmp_path):
    rec, tr = scene(SYN + 1, memory=2); del rec['cells'][1]['control']['readings'][0]['memory_target']['t_last']
    rec2, tr2 = scene(SYN + 2, memory=2); del rec2['cells'][1]['control']['readings'][0]['memory_target']['u']
    rec3, tr3 = scene(SYN + 3); del tr3['traces']['trunk'][2]['tcp']                    # the trunk row of the last hold fit (t=3) without the TCP
    rec4, tr4 = scene(SYN + 4); rec4['cells'][1]['control']['readings'][0]['target_xy'] = 'not a vector'
    res = analyse(tmp_path, [(rec, tr), (rec2, tr2), (rec3, tr3), (rec4, tr4)])
    m = {x['scene']: x for x in res['memory']}; h = {x['scene']: x for x in res['hold']}
    assert m[SYN + 1]['status'] == 'ESTIMATED_NO_MEMORY_T_LAST'
    assert m[SYN + 2]['status'] == 'ESTIMATED' and m[SYN + 2]['err_now_within_memory_u'] is None
    assert h[SYN + 3]['status'] == 'ESTIMATED' and 'held_rel_tcp_change_since_fit_m' not in h[SYN + 3]
    assert any(e['status'] == 'MALFORMED_BRANCH' and e['scene'] == SYN + 4 for e in res['errors'])     # recorded, never dropped silently
    assert SYN + 4 not in m


def test_review_m6_malformed_scenes_and_cells_give_an_incomplete_status_and_exit_3(tmp_path):
    rec1, tr1 = scene(SYN + 1); rec1['cells'] = 'not a list'
    rec2, tr2 = scene(SYN + 2); rec2['cells'][2] = ['not', 'a', 'mapping']
    run = tmp_path / 'run'; write_run(run, [(rec1, tr1), (rec2, tr2), scene(SYN + 3)]); write_run(run, [scene(SYN + 3)], stage='pass_train_seed_2')
    res = M.analyse_run(M.Run(run, 'MOCK', 'syn', {'s': 'pass_train_seed_1'}, 'train_seed_1')); st = {(e.get('scene'), e['status']) for e in res['errors']}
    assert (SYN + 1, 'MALFORMED_RECORD') in st and (SYN + 2, 'MALFORMED_RECORD') in st     # a non-mapping cell also breaks the arm-cell check (review m9)
    assert {r['scene'] for r in res['rows']} == {SYN + 3}                                  # the valid scene is kept; the bad ones are listed
    r = subprocess.run([sys.executable, str(HERE / 'run_analysis.py'), 'c1', '--run', str(run), '--out', str(tmp_path / 'out'), '--source-kind', 'MOCK', '--expected', str(mock_manifest(tmp_path, [SYN + 1, SYN + 2, SYN + 3]))], capture_output=True, text=True, encoding='utf-8', errors='replace')
    man = json.loads((tmp_path / 'out' / 'analysis_manifest.json').read_text(encoding='utf-8'))
    assert r.returncode == 3 and man['status'] == 'INCOMPLETE' and any(p.startswith('CELLS_DIFFER_FROM_REGISTRY') for p in man['problems'])      # the gate sees it first (review F2)


def test_review_m7_an_unreadable_scene_file_is_a_recorded_error(tmp_path):
    run = tmp_path / 'run'; d = write_run(run, [scene(SYN + 1)]); write_run(run, [scene(SYN + 1)], stage='pass_train_seed_2')
    (d / 'scenes' / f'{SYN + 2}.json').write_text('{not json', encoding='utf-8'); (d / 'scenes' / f'{SYN + 3}.json').write_text('[1, 2]', encoding='utf-8')
    res = M.analyse_run(M.Run(run, 'MOCK', 'syn', {'s': 'pass_train_seed_1'}, 'train_seed_1'))
    assert sum(e['status'] == 'UNREADABLE_RECORD' for e in res['errors']) == 2 and {r['scene'] for r in res['rows']} == {SYN + 1}
    r = subprocess.run([sys.executable, str(HERE / 'run_analysis.py'), 'c1', '--run', str(run), '--out', str(tmp_path / 'out'), '--source-kind', 'MOCK', '--expected', str(mock_manifest(tmp_path, [SYN + 1]))], capture_output=True, text=True, encoding='utf-8', errors='replace')
    man = json.loads((tmp_path / 'out' / 'analysis_manifest.json').read_text(encoding='utf-8'))
    assert r.returncode == 3 and man['status'] == 'INCOMPLETE' and any(p.startswith('UNREADABLE_RECORD') for p in man['problems']) and any(p.startswith('MALFORMED_RECORD') for p in man['problems'])


def test_review_m8_bad_outcomes_bad_scene_ids_and_absent_commands(tmp_path):
    rec1, tr1 = scene(SYN + 1); rec1['cells'][1]['y'] = 'unknown'
    rec2, tr2 = scene(SYN + 2); rec2['scene'] = [SYN + 2]
    rec3, tr3 = scene(SYN + 3); rec3['cells'][1]['control']['steps'] = []
    d = Path(tmp_path) / 'run'; dd = d / 'pass_train_seed_1'; (dd / 'scenes').mkdir(parents=True); (dd / 'traces').mkdir()
    for i, (rec, tr) in enumerate([(rec1, tr1), (rec2, tr2), (rec3, tr3)]):
        p = dd / 'traces' / f'trace_{SYN + 1 + i}.json.gz'
        with gzip.open(p, 'wt', encoding='utf-8') as f: json.dump(tr, f)
        rec = {**rec, 'trace': f'traces/{p.name}', 'trace_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        (dd / 'scenes' / f'{SYN + 1 + i}.json').write_text(json.dumps(rec), encoding='utf-8')
    res = M.analyse_run(M.Run(d, 'MOCK', 'syn', {'s': 'pass_train_seed_1'}, 'train_seed_1')); st = {(e.get('arm'), e['status']) for e in res['errors']}
    assert ('VISION_CENTER_V2', 'MALFORMED_BRANCH') in st and any(e['status'] == 'MALFORMED_RECORD' for e in res['errors'])
    v2 = [r for r in res['rows'] if r['arm'] == 'VISION_CENTER_V2' and r['scene'] == SYN + 3][0]
    assert v2['cmd_path_m'] is None and v2['moved'] is True                               # no recorded command is NA, the measured motion stays


def test_review_m9_missing_arm_cells_bad_control_steps_and_missing_motion_class(tmp_path):
    rec1, tr1 = scene(SYN + 1); rec1['arms'] = ['C0', 'VISION_CENTER_V2', 'CENTER_ORACLE']; rec1['cells'] = rec1['cells'][:2]          # the oracle cell is missing
    rec2, tr2 = scene(SYN + 2); rec2['cells'][1]['control_steps'] = True
    rec3, tr3 = scene(SYN + 3); rec3['cells'][1]['control_steps'] = -2
    res = analyse(tmp_path, [(rec1, tr1), (rec2, tr2), (rec3, tr3)]); st = {(e.get('scene'), e['status']) for e in res['errors']}
    assert (SYN + 1, 'MALFORMED_RECORD') in st and (SYN + 2, 'MALFORMED_BRANCH') in st and (SYN + 3, 'MALFORMED_BRANCH') in st
    assert M.classify_miss({'arm': 'VISION_CENTER_V2', 'decision': 'ACT', 'acted': True, 'moved': None}) == 'UNCLASSIFIABLE_MISSING_RECORD'


def test_review_m10_required_control_steps_and_arms_list(tmp_path):
    rec1, tr1 = scene(SYN + 1); del rec1['cells'][1]['control_steps']
    rec2, tr2 = scene(SYN + 2); del rec2['arms']
    rec3, tr3 = scene(SYN + 3); rec3['triggered'] = False
    for c in rec3['cells']: c.pop('control_steps', None)
    res = analyse(tmp_path, [(rec1, tr1), (rec2, tr2), (rec3, tr3)]); st = {(e.get('scene'), e['status']) for e in res['errors']}
    assert (SYN + 1, 'MALFORMED_BRANCH') in st and (SYN + 2, 'MALFORMED_RECORD') in st
    assert {r['arm'] for r in res['rows'] if r['scene'] == SYN + 3} == {'C0', 'VISION_CENTER_V2', 'CENTER_ORACLE'}     # an untriggered record without control_steps is valid


def test_review_m11_none_steps_on_a_trigger_and_float_zero_without_one(tmp_path):
    rec1, tr1 = scene(SYN + 1); rec1['cells'][1]['control_steps'] = None
    rec2, tr2 = scene(SYN + 2); rec2['triggered'] = False
    for c in rec2['cells']: c['control_steps'] = 0.0
    res = analyse(tmp_path, [(rec1, tr1), (rec2, tr2)]); st = [(e.get('scene'), e['status']) for e in res['errors']]
    assert (SYN + 1, 'MALFORMED_BRANCH') in st and st.count((SYN + 2, 'MALFORMED_BRANCH')) == 3


def test_v1_readings_are_read_with_their_own_fields_and_oracle_zero_steps_are_counted(tmp_path):
    rec, tr = scene(SYN + 1); rec['arms'] = ['C0', 'VISION_CENTER_V1', 'CENTER_ORACLE']
    rd = {'t': 5, 'target_fit': {'xy': [0.021, 0.0], 'valid': True}, 'hold_fit': {'valid': True}, 'target_fresh': True, 'target_xy': [0.021, 0.0], 'held_xyz': [0.0, 0.0, 0.06], 'offset_norm_m': 0.02, 'offset_xy': [0.02, 0.0]}
    rec['cells'][1] = {**rec['cells'][1], 'arm': 'VISION_CENTER_V1', 'control': {**rec['cells'][1]['control'], 'readings': [rd]}}
    tr['traces']['VISION_CENTER_V1'] = tr['traces'].pop('VISION_CENTER_V2'); rec['cells'][2]['control_steps'] = 0
    res = analyse(tmp_path, [(rec, tr)]); m = res['memory'][0]; h = res['hold'][0]
    assert (m['target_src'], m['status'], h['hold_src'], h['status']) == ('frame', 'ESTIMATED', 'frame', 'ESTIMATED')
    f = {(x['arm'], x['step']): x for x in M.funnel(res['rows'], ['C0', 'VISION_CENTER_V1', 'CENTER_ORACLE'])}
    assert f[('CENTER_ORACLE', 'oracle_within_tolerance_at_start')]['numerator'] == 1 and f[('CENTER_ORACLE', 'decision_ACT')]['numerator'] == 0


def test_review_m12_oracle_zero_steps_with_another_end_reason_is_not_within_tolerance(tmp_path):
    rec, tr = scene(SYN + 1); rec['cells'][2]['control_steps'] = 0; rec['cells'][2]['control']['end_reason'] = 'episode_ended'
    res = analyse(tmp_path, [(rec, tr)]); o = [r for r in res['rows'] if r['arm'] == 'CENTER_ORACLE'][0]
    assert o['decision'] == 'ORACLE_ZERO_STEPS:episode_ended'
    f = {(x['arm'], x['step']): x for x in M.funnel(res['rows'], ['C0', 'VISION_CENTER_V2', 'CENTER_ORACLE'])}
    assert f[('CENTER_ORACLE', 'oracle_within_tolerance_at_start')]['numerator'] == 0 and f[('CENTER_ORACLE', 'oracle_zero_steps_other_end_reason')]['numerator'] == 1
