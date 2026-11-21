"""Regression tests of the independent review's three findings (C1 analysis targeted fix v1.0).

F1  overall vs trigger-conditional vs move-conditional denominators (untriggered C0 failures and successes included).
F2  the expected-set gate: empty, partial, duplicate, extra, wrong-plan, wrong-arm, error and changed inputs never pass,
    and a complete input still passes.
F3  the c1 command produces the whole chain (memory, S6, executed ablation, cases, report, claims, cross-check).

Synthetic scene IDs (9_000_000+) and the real collector on the mock runtime with seeds 7 and 8; no C1 ledger, no research
seed, no simulator. Every blocked case is checked for its own problem code, an INCOMPLETE status, exit code 3 and the
absence of a traceback, so an unrelated crash is never taken for a block.
"""
import csv, gzip, hashlib, importlib.util, json, shutil, subprocess, sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]; REPO = HERE.parent; sys.path[:0] = [str(HERE), str(REPO / 'src'), str(REPO / 'tests'), str(HERE / 'tests')]
import mechanism as M
from test_mechanism import SYN, MOCK_PLAN, scene, write_run, mock_manifest

CLI = [sys.executable, str(HERE / 'run_analysis.py'), 'c1']
FROZEN = '6b3f04fea1ea5346d6f0725b6e49ae9ce04c4fb132f37450243a807024e5dbf7'


def run_c1(run, out, *extra, kind='MOCK'):
    r = subprocess.run(CLI + ['--run', str(run), '--out', str(out), '--source-kind', kind, *extra], capture_output=True, text=True, encoding='utf-8', errors='replace')
    man = json.loads((Path(out) / 'analysis_manifest.json').read_text(encoding='utf-8')) if (Path(out) / 'analysis_manifest.json').exists() else None
    return r, man


def blocked(r, man, out, code):
    assert 'Traceback' not in r.stderr and 'Error' not in r.stderr.split('\n')[0], r.stderr          # a crash is not a block
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    assert man is not None and man['status'] == 'INCOMPLETE', man
    assert any(p.startswith(code) or f' {code}' in p for p in man['problems']), (code, man['problems'])
    assert not (Path(out) / 'C1_MECHANISM_REPORT.md').exists() and not (Path(out) / 'claims_to_evidence.csv').exists() and (Path(out) / 'INCOMPLETE_NOT_A_RESULT.txt').exists()


# ---------------------------------------------------------------- F1
def untriggered(seed, y):
    rec, tr = scene(seed, y=y); rec['triggered'] = False; rec['trigger'] = None
    for c in rec['cells']: c['control_steps'] = 0; c['acted'] = False; c['control'] = None
    return rec, tr


def test_f1_overall_and_trigger_conditional_denominators_are_separate(tmp_path):
    sc = [scene(SYN + 1, y=(0, 1, 1)), scene(SYN + 2, y=(1, 0, 1)), scene(SYN + 3, y=(1, 1, 1)), untriggered(SYN + 4, (0, 0, 0)), untriggered(SYN + 5, (1, 1, 1))]
    write_run(tmp_path, sc); res = M.analyse_run(M.Run(tmp_path, 'MOCK', 'syn', {'s': 'pass_train_seed_1'}, 'train_seed_1'))
    assert res['errors'] == []
    f = M.funnel(res['rows'], ['C0', 'VISION_CENTER_V2', 'CENTER_ORACLE']); d = {(x['arm'], x['step']): (x['numerator'], x['denominator']) for x in f}
    assert M.check_funnel(f) == []
    v = lambda s: d[('VISION_CENTER_V2', s)]
    assert v('scenes')[0] == 5 and v('triggered') == (3, 5) and v('not_triggered') == (2, 5)
    assert v('overall_rescue_of_c0_failure') == (1, 2) and v('triggered_rescue_of_c0_failure') == (1, 1)            # an untriggered C0 failure stays in the overall denominator
    assert v('overall_harm_of_c0_success') == (1, 3) and v('triggered_harm_of_c0_success') == (1, 2)                # so does an untriggered C0 success
    assert v('overall_success') == (3, 5) and v('overall_success')[0] == v('overall_c0_success')[0] + v('overall_rescue_of_c0_failure')[0] - v('overall_harm_of_c0_success')[0]
    assert v('overall_net_rescue_minus_harm') == (0, 5) and v('not_triggered_rescue_of_c0_failure') == (0, 1) and v('not_triggered_harm_of_c0_success') == (0, 1)
    assert v('moved_harm_of_c0_success_where_arm_moved') == (1, 2)                                                 # C0 successes in which v2 actually moved
    assert d[('C0', 'overall_success')] == (3, 5) and d[('CENTER_ORACLE', 'overall_rescue_of_c0_failure')] == (1, 2)       # the oracle did not run in the untriggered C0 failure
    bad = [dict(x) for x in f]
    for x in bad:
        if x['arm'] == 'VISION_CENTER_V2' and x['step'] == 'overall_rescue_of_c0_failure': x['numerator'] += 1
    assert any('success != C0 success + rescue - harm' in p for p in M.check_funnel(bad)) and any('overall rescue != triggered' in p for p in M.check_funnel(bad))


def test_f1_a_missing_untriggered_outcome_is_a_completeness_error_not_a_failure(tmp_path):
    rec, tr = untriggered(SYN + 4, (0, 0, 0)); rec['cells'][1]['y'] = None
    write_run(tmp_path, [scene(SYN + 1, y=(0, 1, 1)), (rec, tr)]); res = M.analyse_run(M.Run(tmp_path, 'MOCK', 'syn', {'s': 'pass_train_seed_1'}, 'train_seed_1'))
    f = M.funnel(res['rows'], ['C0', 'VISION_CENTER_V2', 'CENTER_ORACLE']); d = {(x['arm'], x['step']): (x['numerator'], x['denominator']) for x in f}
    assert d[('VISION_CENTER_V2', 'outcome_missing_completeness_error')] == (1, 2) and d[('VISION_CENTER_V2', 'overall_rescue_of_c0_failure')] == (1, 1)
    assert any('outcomes missing' in p for p in M.check_funnel(f))


# ---------------------------------------------------------------- F2 on synthetic records
def two_pass(tmp, seeds=(SYN + 1, SYN + 2)):
    run = tmp / 'run'; sc = [scene(s) for s in seeds]; write_run(run, sc); write_run(run, sc, stage='pass_train_seed_2'); return run


def test_f2_reproduction_empty_scene_folders_are_blocked(tmp_path):
    run = tmp_path / 'run'
    for pid in ('train_seed_1', 'train_seed_2'): (run / f'pass_{pid}' / 'scenes').mkdir(parents=True)
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(mock_manifest(tmp_path, [SYN + 1, SYN + 2]))); blocked(r, man, tmp_path / 'out', 'EMPTY_PASS')
    assert any(p.startswith('MISSING_SCENES') for p in man['problems'])
    r2 = subprocess.run(CLI + ['--run', str(run), '--out', str(tmp_path / 'o2'), '--source-kind', 'MOCK'], capture_output=True, text=True, encoding='utf-8', errors='replace')
    assert r2.returncode != 0 and 'needs --expected' in r2.stderr and 'Traceback' not in r2.stderr and not (tmp_path / 'o2').exists()      # no manifest: refused, never skipped


def test_f2_a_reduced_synthetic_manifest_is_never_complete(tmp_path):
    # the synthetic set is otherwise consistent: its only problems are the reductions of the manifest (3 arms, no collector summary, no spatial fields)
    run = two_pass(tmp_path); r, man = run_c1(run, tmp_path / 'out', '--expected', str(mock_manifest(tmp_path, [SYN + 1, SYN + 2]))); blocked(r, man, tmp_path / 'out', 'MOCK_MANIFEST_REDUCED')
    assert all(p.startswith('MOCK_MANIFEST_REDUCED') for p in man['problems'])
    r2 = subprocess.run(CLI + ['--run', str(run), '--out', str(tmp_path / 'o2'), '--source-kind', 'MOCK', '--expected', 'x', '--c1-plan-sha256', 'ab' * 32], capture_output=True, text=True)
    assert r2.returncode == 2 and 'unrecognized arguments' in r2.stderr                         # the frozen hash is not a CLI option (review f1)


def _drop(run, pid, s): (run / f'pass_{pid}' / 'scenes' / f'{s}.json').unlink()


def _edit(run, pid, s, fn):
    p = run / f'pass_{pid}' / 'scenes' / f'{s}.json'; rec = json.loads(p.read_text(encoding='utf-8')); fn(rec); p.write_text(json.dumps(rec), encoding='utf-8')


def _copy_as(run, pid, s, new, keep_id=False):
    src = run / f'pass_{pid}' / 'scenes' / f'{s}.json'; rec = json.loads(src.read_text(encoding='utf-8'))
    if not keep_id: rec['scene'] = new
    (run / f'pass_{pid}' / 'scenes' / f'{new}.json').write_text(json.dumps(rec), encoding='utf-8')


def _drop_arm(rec): rec['arms'] = [a for a in rec['arms'] if a != 'CENTER_ORACLE']; rec['cells'] = [c for c in rec['cells'] if c['arm'] != 'CENTER_ORACLE']


CASES = {
    'one_policy_only': (lambda run: shutil.rmtree(run / 'pass_train_seed_2'), 'MISSING_POLICY_PASS'),
    'one_scene_missing': (lambda run: _drop(run, 'train_seed_2', SYN + 2), 'MISSING_SCENES'),
    'extra_scene': (lambda run: _copy_as(run, 'train_seed_1', SYN + 2, SYN + 9), 'EXTRA_SCENES'),
    'duplicate_scene_key': (lambda run: _copy_as(run, 'train_seed_1', SYN + 2, f'{SYN + 2}_copy', keep_id=True), 'DUPLICATE_SCENE'),
    'extra_policy_pass': (lambda run: shutil.copytree(run / 'pass_train_seed_2', run / 'pass_train_seed_3'), 'EXTRA_POLICY_PASS'),
    'same_arm_missing_from_arms_and_cells': (lambda run: _edit(run, 'train_seed_1', SYN + 1, _drop_arm), 'ARMS_DIFFER_FROM_REGISTRY'),
    'error_record': (lambda run: (run / 'pass_train_seed_2' / 'scenes' / f'{SYN + 2}.error.json').write_text('{"completed": false}', encoding='utf-8'), 'ERROR_RECORD'),
    'record_of_another_plan': (lambda run: _edit(run, 'train_seed_2', SYN + 1, lambda r: r.update(plan_sha256='cd' * 32)), 'WRONG_PLAN_IN_RECORD'),
    'record_of_another_policy': (lambda run: _edit(run, 'train_seed_2', SYN + 1, lambda r: r.update(policy_id='train_seed_1')), 'WRONG_POLICY'),
    'incomplete_record': (lambda run: _edit(run, 'train_seed_1', SYN + 2, lambda r: r.update(completed=False)), 'INCOMPLETE_RECORD'),
    'trace_missing': (lambda run: next((run / 'pass_train_seed_1' / 'traces').glob('*.gz')).unlink(), 'TRACE_MISSING'),
    'trace_changed': (lambda run: next((run / 'pass_train_seed_2' / 'traces').glob('*.gz')).write_bytes(gzip.compress(b'{"traces": {}}')), 'TRACE_SHA_MISMATCH'),
    'same_count_other_scenes': (lambda run: (_copy_as(run, 'train_seed_2', SYN + 2, SYN + 9), _drop(run, 'train_seed_2', SYN + 2)), 'POLICY_SCENE_SETS_DIFFER'),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_f2_partial_or_inconsistent_inputs_are_blocked(tmp_path, name):
    run = two_pass(tmp_path); damage, code = CASES[name]; damage(run)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(mock_manifest(tmp_path, [SYN + 1, SYN + 2]))); blocked(r, man, tmp_path / 'out', code)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}        # the gate never writes into the run


def test_f2_a_missing_trace_and_changed_frames_are_both_reported(tmp_path):
    run = two_pass(tmp_path); next((run / 'pass_train_seed_1' / 'traces').glob('*.gz')).unlink(); next((run / 'pass_train_seed_2' / 'frames').glob('*.npz')).write_bytes(b'changed')
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(mock_manifest(tmp_path, [SYN + 1, SYN + 2]))); blocked(r, man, tmp_path / 'out', 'TRACE_MISSING')
    assert any(p.startswith('FRAMES_SHA_MISMATCH') for p in man['problems'])                     # the two checks are independent (found by the final test run)


@pytest.mark.parametrize('kw, code', [({'scenes': [51100, 51101]}, 'MOCK_USES_RESEARCH_SEEDS'), ({'record_plan_sha256': FROZEN}, 'MOCK_USES_RESEARCH_PLAN'), ({'kind': 'C1_PLAN'}, 'MOCK_MANIFEST_INVALID'), ({'arms': ['C0', 'V2S']}, 'MOCK_MANIFEST_INVALID')])
def test_f2_a_mock_manifest_cannot_borrow_research_seeds_or_the_plan(tmp_path, kw, code):
    run = two_pass(tmp_path); m = json.loads(mock_manifest(tmp_path, [SYN + 1, SYN + 2]).read_text(encoding='utf-8')); m.update(kw); p = tmp_path / 'm.json'; p.write_text(json.dumps(m), encoding='utf-8')
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(p)); blocked(r, man, tmp_path / 'out', code)


def test_f2_a_mock_run_with_a_ledger_is_blocked(tmp_path):
    run = two_pass(tmp_path); (run / 'ledger.json').write_text('{}', encoding='utf-8')
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(mock_manifest(tmp_path, [SYN + 1, SYN + 2]))); blocked(r, man, tmp_path / 'out', 'MOCK_USES_LEDGER')


def test_f2_c1_original_reads_the_plan_file_and_its_registries(tmp_path):
    run = tmp_path / 'run'
    for pid in ('train_seed_1', 'train_seed_2'): (run / f'pass_{pid}' / 'scenes').mkdir(parents=True)
    r, man = run_c1(run, tmp_path / 'out', kind='C1_ORIGINAL'); blocked(r, man, tmp_path / 'out', 'MISSING_SCENES')
    assert man['expected']['plan_sha256'] == FROZEN and man['expected']['scenes_n'] == 200 and man['expected']['scenes_first_last'] == [51100, 51299] and len(man['expected']['arms']) == 9
    assert 'VISION_CENTER_V2S' not in man['expected']['arms'] and any('(200)' in p for p in man['problems']) and any(p.startswith('PROVENANCE_MISSING') for p in man['problems'])
    plan = REPO / 'results/vision_center_v2_c1_001/preparation_v3/plan.json'; wrong = tmp_path / 'plan_copy.json'; wrong.write_bytes(plan.read_bytes() + b'\n')
    r, man = run_c1(run, tmp_path / 'out2', '--plan', str(wrong), kind='C1_ORIGINAL'); blocked(r, man, tmp_path / 'out2', 'WRONG_PLAN')
    r = subprocess.run(CLI + ['--run', str(run), '--out', str(tmp_path / 'o3'), '--source-kind', 'C1_ORIGINAL', '--expected', str(mock_manifest(tmp_path, [1]))], capture_output=True, text=True, encoding='utf-8', errors='replace')
    assert r.returncode != 0 and 'only for MOCK' in r.stderr and 'Traceback' not in r.stderr
    assert M.sha256(plan) == FROZEN                                                                                   # the plan file itself is unchanged


# ---------------------------------------------------------------- F3: the real collector on the mock runtime, both policies, nine arms
@pytest.fixture(scope='module')
def collector_run(tmp_path_factory):
    pytest.importorskip('torch')
    from test_recovery_v1_oracle import CFG
    from test_vision_center import BiasPolicy
    from test_vision_center_v2 import VisionEnv2
    import test_spatial_links as SLT
    from hci.simulation import Runtime
    from hci import vision_center_c1 as C1
    from hci.io import write_json
    spec = importlib.util.spec_from_file_location('c1_driver_fix', REPO / 'scripts/34_vision_center_c1.py'); X34 = importlib.util.module_from_spec(spec); spec.loader.exec_module(X34)
    def probe(rt, z):                                     # the spatial mock of tests/test_spatial_links.py plus one listed, separated shape pair (the S6 population)
        r = SLT.spatial_mock(rt, z); r['contacts'] = list(r['contacts']) + [SLT.c('panda_link3', 'table', sep=.006)]; return r
    bias = {('train_seed_1', 7): (.028, .012), ('train_seed_1', 8): (-.004, .0), ('train_seed_2', 7): (-.028, .0), ('train_seed_2', 8): (.028, .012)}
    root = tmp_path_factory.mktemp('c1fix'); run = root / 'run'; run.mkdir(); recs = {}
    for pid in C1.POLICIES:
        for s in (7, 8):
            r = Runtime(VisionEnv2(), BiasPolicy(bias[(pid, s)]), CFG); rec, tr, fr = C1.collect_scene(r, s, CFG, probe=probe, arms=C1.BASE_ARMS, policy_id=pid)
            rec['plan_sha256'] = MOCK_PLAN; X34.Z.save_scene(run / f'pass_{pid}', s, rec, tr, fr); recs.setdefault(pid, []).append(rec)
    write_json(run / 'result.json', {**C1.summarize(recs, [7, 8], C1.BASE_ARMS), 'plan_sha256': MOCK_PLAN})     # cmd_summarize adds the plan hash the same way                                       # the collector's own summary code (no simulator)
    man = mock_manifest(root, [7, 8], arms=C1.BASE_ARMS, role=C1.ROLE, collector_result='result.json', spatial_fields=True)
    return run, man, root, C1


def test_f3_the_c1_command_produces_the_whole_chain(collector_run, tmp_path):
    run, man_p, _, C1 = collector_run; out = tmp_path / 'out'
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    r, man = run_c1(run, out, '--expected', str(man_p))
    assert r.returncode == 0 and man['status'] == 'ANALYSIS_COMPLETE', r.stdout + r.stderr
    import run_analysis as RA
    assert set(man['required_outputs']) == set(RA.C1_REQUIRED) and all(man['required_outputs'].values()) and man['inputs']['input_sha256_unchanged']
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()} and man['is_research_result'] is False and (out / 'MOCK_NOT_A_RESULT.txt').exists()
    res = json.loads((out / 'results.json').read_text(encoding='utf-8')); cr = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    # independent recount: overall success per policy and arm equals the collector summary
    f = {(x['policy'], x['arm'], x['step']): x for x in res['funnel']}
    for pid in C1.POLICIES:
        for arm in C1.BASE_ARMS: assert (f[(pid, arm, 'overall_success')]['numerator'], f[(pid, arm, 'overall_scenes_with_complete_outcomes')]['numerator']) == (cr['per_policy'][pid]['success'][arm]['k'], cr['per_policy'][pid]['success'][arm]['n'])
    cc = list(csv.DictReader(open(out / 'tables' / 'crosscheck_against_collector_summary.csv', encoding='utf-8'))); assert cc and all(x['match'] == 'True' for x in cc)
    for pid in C1.POLICIES:
        o = res['s6'][pid]['official_S6']; assert o['status'] in ('FAIL', 'PASS') and o['pairs'] > 0 and o['tolerance_m'] == 0.001          # the 1 mm tolerance is unchanged
        assert res['s6'][pid]['contact_list_vs_impulse']['separated_pairs_listed_in_contact_list'] == o['pairs']
    abl = list(csv.DictReader(open(out / 'tables' / 'executed_ablation_chain.csv', encoding='utf-8'))); assert {(x['policy'], x['arm']) for x in abl} == {(p, a) for p in C1.POLICIES for a in M.ABLATION}
    rows = list(csv.DictReader(open(out / 'scene_arm_summary.csv', encoding='utf-8'))); assert len(rows) == 2 * 2 * len(C1.BASE_ARMS) and {x['source_kind'] for x in rows} == {'MOCK'}
    mem = list(csv.DictReader(open(out / 'tables' / 'target_estimate_and_memory_decomposition.csv', encoding='utf-8')))
    assert len(mem) == sum(1 for x in rows if x['triggered'] == 'True' and x['arm'] in M.VISION_ARMS)                  # one memory row per triggered vision branch
    claims = list(csv.DictReader(open(out / 'claims_to_evidence.csv', encoding='utf-8'))); assert any(c['id'] == 'K03' and c['value'].startswith(f'{len(cc)} of {len(cc)}') for c in claims)
    assert 'MOCK records' in (out / 'C1_MECHANISM_REPORT.md').read_text(encoding='utf-8')
    cases = list(csv.DictReader(open(out / 'case_index.csv', encoding='utf-8'))); assert all(c['status'] == 'INDEXED' for c in cases)
    assert (out / 'tables' / 's6_pair_rows.csv.gz').stat().st_size > 0 and 'shared scenes are not independent samples' in (out / 'policy_paired_outcomes.csv').read_text(encoding='utf-8')


def _copy(collector_run, tmp_path):
    run = tmp_path / 'run'; shutil.copytree(collector_run[0], run); return run


def test_f3_a_collector_summary_that_disagrees_is_blocked(collector_run, tmp_path):
    run = _copy(collector_run, tmp_path); cr = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    cr['per_policy']['train_seed_2']['against_C0']['VISION_CENTER_V2']['rescue'] += 1; (run / 'result.json').write_text(json.dumps(cr), encoding='utf-8')
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(collector_run[1])); blocked(r, man, tmp_path / 'out', 'CROSSCHECK_MISMATCH')


def test_f3_a_collector_summary_of_another_plan_is_blocked(collector_run, tmp_path):
    run = _copy(collector_run, tmp_path); cr = json.loads((run / 'result.json').read_text(encoding='utf-8')); cr['plan_sha256'] = 'cd' * 32; (run / 'result.json').write_text(json.dumps(cr), encoding='utf-8')
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(collector_run[1])); blocked(r, man, tmp_path / 'out', 'COLLECTOR_RESULT_OTHER_PLAN')


def test_f3_changed_frames_are_blocked(collector_run, tmp_path):
    run = _copy(collector_run, tmp_path); f = next((run / 'pass_train_seed_2' / 'frames').glob('*.npz')); f.write_bytes(f.read_bytes() + b'x')
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(collector_run[1])); blocked(r, man, tmp_path / 'out', 'FRAMES_SHA_MISMATCH')


@pytest.mark.parametrize('field, code', [('trace', 'TRACE_OUTSIDE_RUN'), ('frames', 'FRAMES_OUTSIDE_RUN')])
def test_f3_record_paths_outside_the_run_are_blocked(collector_run, tmp_path, field, code):
    run = _copy(collector_run, tmp_path); p = run / 'pass_train_seed_1' / 'scenes' / '7.json'; rec = json.loads(p.read_text(encoding='utf-8'))
    ext = tmp_path / 'outside.bin'; src = run / 'pass_train_seed_1' / rec[field]; ext.write_bytes(src.read_bytes()); rec[field] = str(ext); p.write_text(json.dumps(rec), encoding='utf-8')   # same bytes, outside the run
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(collector_run[1])); blocked(r, man, tmp_path / 'out', code)


def test_f3_a_collector_summary_without_one_policy_is_blocked(collector_run, tmp_path):
    run = _copy(collector_run, tmp_path); cr = json.loads((run / 'result.json').read_text(encoding='utf-8')); del cr['per_policy']['train_seed_2']; (run / 'result.json').write_text(json.dumps(cr), encoding='utf-8')
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(collector_run[1])); blocked(r, man, tmp_path / 'out', 'CROSSCHECK_MISSING')


def test_f3_a_missing_collector_summary_is_blocked(collector_run, tmp_path):
    run = _copy(collector_run, tmp_path); (run / 'result.json').unlink()
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(collector_run[1])); blocked(r, man, tmp_path / 'out', 'COLLECTOR_RESULT_MISSING')


def test_f3_spatial_fields_expected_but_absent_is_blocked(collector_run, tmp_path):
    run = _copy(collector_run, tmp_path); p = run / 'pass_train_seed_1' / 'scenes' / '8.json'; rec = json.loads(p.read_text(encoding='utf-8'))
    tp = run / 'pass_train_seed_1' / rec['trace']; t = json.loads(gzip.decompress(tp.read_bytes()))
    for rows in t['traces'].values():
        for x in rows: x.pop('contacts', None)
    tp.write_bytes(gzip.compress(json.dumps(t).encode())); rec['trace_sha256'] = M.sha256(tp); p.write_text(json.dumps(rec), encoding='utf-8')
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(collector_run[1])); blocked(r, man, tmp_path / 'out', 'SPATIAL_FIELDS_MISSING')


@pytest.mark.parametrize('name', ['upper_case_scene', 'upper_case_error', 'subfolder'])
def test_f3_unexpected_files_in_scenes_are_blocked(collector_run, tmp_path, name):
    run = _copy(collector_run, tmp_path); d = run / 'pass_train_seed_1' / 'scenes'
    if name == 'upper_case_scene': (d / '9.JSON').write_text((d / '8.json').read_text(encoding='utf-8').replace('"scene": 8', '"scene": 9'), encoding='utf-8')
    elif name == 'upper_case_error': (d / '8.error.JSON').write_text('{"completed": false}', encoding='utf-8')
    else: (d / 'more').mkdir()
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(collector_run[1])); blocked(r, man, tmp_path / 'out', 'UNEXPECTED_FILE_IN_SCENES')


def test_f3_a_collector_summary_outside_the_run_is_blocked(collector_run, tmp_path):
    (tmp_path / 'external.json').write_bytes((collector_run[0] / 'result.json').read_bytes())
    m = json.loads(Path(collector_run[1]).read_text(encoding='utf-8')); m['collector_result'] = '../external.json'; p = tmp_path / 'm.json'; p.write_text(json.dumps(m), encoding='utf-8')
    run = tmp_path / 'run'; shutil.copytree(collector_run[0], run)
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(p)); blocked(r, man, tmp_path / 'out', 'MOCK_MANIFEST_INVALID')


@pytest.mark.parametrize('change', [{'arms': ['C0', 'VISION_CENTER_V2']}, {'collector_result': None}, {'spatial_fields': False}, {'policies': ['train_seed_1']}, {'require_policy_id': False}])
def test_f3_a_reduced_manifest_cannot_define_away_arms_or_checks(collector_run, tmp_path, change):
    m = json.loads(Path(collector_run[1]).read_text(encoding='utf-8')); m.update(change); p = tmp_path / 'm.json'; p.write_text(json.dumps(m), encoding='utf-8')
    r, man = run_c1(collector_run[0], tmp_path / 'out', '--expected', str(p)); blocked(r, man, tmp_path / 'out', 'MOCK_MANIFEST_REDUCED')


def test_f3_a_missing_arm_in_the_collector_run_is_blocked(collector_run, tmp_path):
    run = _copy(collector_run, tmp_path); p = run / 'pass_train_seed_2' / 'scenes' / '7.json'; rec = json.loads(p.read_text(encoding='utf-8'))
    rec['arms'] = [a for a in rec['arms'] if a != 'V2_RULES_OFF']; rec['cells'] = [c for c in rec['cells'] if c['arm'] != 'V2_RULES_OFF']; p.write_text(json.dumps(rec), encoding='utf-8')
    r, man = run_c1(run, tmp_path / 'out', '--expected', str(collector_run[1])); blocked(r, man, tmp_path / 'out', 'ARMS_DIFFER_FROM_REGISTRY')
