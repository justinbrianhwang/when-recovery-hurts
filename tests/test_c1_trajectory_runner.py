"""The C1 trajectory audit runner (scripts/35_c1_trajectory_audit.py) on a mock C1 run folder.

The folder is written exactly as the run writes it (C1.collect_scene on the mock runtime, the driver's save_scene, the plan
hash stamped into every record, run/ledger.json), for both policies, all nine arms and synthetic seeds 7 and 8. No C1
scene, no simulator. Every fault below is introduced on purpose into a copy of the folder; the original is checked to be
unchanged."""
import copy, gzip, importlib.util, json, shutil, subprocess, sys
from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
from test_recovery_v1_oracle import CFG
from test_vision_center import BiasPolicy
from test_vision_center_v2 import VisionEnv2
from test_trajectory_audit import probe2
from hci.simulation import Runtime
from hci import vision_center_c1 as C1
from hci import trajectory_audit as T

SEEDS = [7, 8]
BIAS = {('train_seed_1', 7): (.028, .012), ('train_seed_1', 8): (-.004, .0), ('train_seed_2', 7): (-.028, .0), ('train_seed_2', 8): (.028, .012)}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture(scope='module')
def R(): return load_module('c1_traj_runner', 'scripts/35_c1_trajectory_audit.py')


@pytest.fixture(scope='module')
def mock(tmp_path_factory):
    X34 = load_module('c1_driver_tr', 'scripts/34_vision_center_c1.py'); root = tmp_path_factory.mktemp('c1mock'); plan = root / 'plan.json'
    pv = json.loads(Path('results/vision_center_v2_c1_001/preparation/plan_v2.json').read_text(encoding='utf-8'))
    pv['seed_registry'] = {**pv['seed_registry'], 'range': [SEEDS[0], SEEDS[-1]], 'n': len(SEEDS), 'note': 'MOCK: the seed range is the synthetic seeds of this test'}
    plan.write_text(json.dumps(pv), encoding='utf-8'); h = X34.sha256(plan); run = root / 'run'; run.mkdir()
    for pid in C1.POLICIES:
        for s in SEEDS:
            r = Runtime(VisionEnv2(), BiasPolicy(BIAS[(pid, s)]), CFG); rec, tr, fr = C1.collect_scene(r, s, CFG, probe=probe2, arms=C1.BASE_ARMS, policy_id=pid)
            rec['plan_sha256'] = h; X34.Z.save_scene(run / f'pass_{pid}', s, rec, tr, fr)
    (run / 'ledger.json').write_text(json.dumps({'set': C1.SET, 'plan_sha256': h, 'note': 'mock'}))
    return {'run': run, 'plan': plan, 'h': h, 'root': root, 'tree': {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()}}


def faulty(mock, tmp_path, edit):
    d = tmp_path / 'run'; shutil.copytree(mock['run'], d); edit(d); return d


def rec_path(d, pid, s): return d / f'pass_{pid}' / 'scenes' / f'{s}.json'
def read(p): return json.loads(Path(p).read_text())
def write(p, v): Path(p).write_text(json.dumps(v))
def fails(summary, pid, check): return {s for v in summary['fail_scenes'].get(pid, {}).values() for k, ss in v.items() if k == check for s in ss}


def test_complete_mock_run_is_recognised_and_passes_every_record_check(R, mock, tmp_path):
    pytest.importorskip('matplotlib'); pytest.importorskip('PIL')
    s = R.audit(mock['run'], mock['plan'], tmp_path / 'out', None, figure_scenes=1)
    assert s['run_status'] == {'status': 'COMPLETE', 'reasons': []} and s['policies'] == list(C1.POLICIES) and s['arms'] == list(C1.BASE_ARMS) and s['plan_sha256'] == mock['h']
    assert s['records_audited'] == {pid: 2 for pid in C1.POLICIES} and all(set(s['check_counts'][pid]) == set(C1.BASE_ARMS) for pid in C1.POLICIES if s['triggered'][pid])
    for pid in C1.POLICIES:
        for k, v in s['record_check_counts'][pid].items(): assert v['FAIL'] == 0, (pid, k, v)
    out = tmp_path / 'out'; man = read(out / 'trajectory_manifest.json')
    assert (out / 'per_scene_checks.csv').exists() and man['run_folder_unchanged_by_audit'] and man['figures'] and all(f['outputs'] for f in man['figures'])
    names = set(man['figures'][0]['outputs']); assert {'3d_trajectory.png', 'xy_alignment.png', 'timeline_actions_gripper.png', 'rgb_recorded.gif', 'rgb_with_posthoc_ground_truth.gif'} <= names
    rows = (out / 'per_scene_checks.csv').read_text(encoding='utf-8').splitlines(); assert len({r.split(',')[2] for r in rows[1:]} - {'*'}) == 9 and any(',R_frame_index,PASS,' in r for r in rows)


def test_equal_inputs_give_an_identical_summary_and_the_inputs_are_not_changed(R, mock, tmp_path):
    a = R.audit(mock['run'], mock['plan'], tmp_path / 'a', None, make_figures=False); b = R.audit(mock['run'], mock['plan'], tmp_path / 'b', None, make_figures=False)
    assert (tmp_path / 'a' / 'audit_summary.json').read_bytes() == (tmp_path / 'b' / 'audit_summary.json').read_bytes() and (tmp_path / 'a' / 'per_scene_checks.csv').read_bytes() == (tmp_path / 'b' / 'per_scene_checks.csv').read_bytes()
    assert {p.relative_to(mock['run']).as_posix(): p.read_bytes() for p in mock['run'].rglob('*') if p.is_file()} == mock['tree']
    with pytest.raises(SystemExit, match='Refused overwrite'): R.audit(mock['run'], mock['plan'], tmp_path / 'a', None, make_figures=False)
    for inside in (mock['run'] / 'audit', mock['run'] / 'pass_train_seed_1' / 'x', mock['run']):
        with pytest.raises(SystemExit, match='outside the run folder'): R.audit(mock['run'], mock['plan'], inside, None, make_figures=False)
    sub = R.audit(mock['run'], mock['plan'], tmp_path / 'sub', [7], make_figures=False); assert sub['run_status']['status'] == 'SUBSET_ONLY'               # a narrower --seeds is never COMPLETE
    assert {p.relative_to(mock['run']).as_posix(): p.read_bytes() for p in mock['run'].rglob('*') if p.is_file()} == mock['tree']


def test_a_missing_arm_or_scene_is_found(R, mock, tmp_path):
    def drop_arm(d):
        p = rec_path(d, 'train_seed_1', 7); v = read(p); v['cells'] = [c for c in v['cells'] if c['arm'] != 'V2_RULES_OFF']; v['arms'] = [a for a in v['arms'] if a != 'V2_RULES_OFF']; write(p, v)
    s = R.audit(faulty(mock, tmp_path / 'x', drop_arm), mock['plan'], tmp_path / 'o1', None, make_figures=False)
    assert s['run_status']['status'] == 'INCOMPLETE' and 7 in fails(s, 'train_seed_1', 'R_record_identity')
    s = R.audit(faulty(mock, tmp_path / 'y', lambda d: rec_path(d, 'train_seed_2', 8).unlink()), mock['plan'], tmp_path / 'o2', None, make_figures=False)
    assert s['run_status']['status'] == 'INCOMPLETE' and any('pass_train_seed_2: 1 planned scene records missing' in x for x in s['run_status']['reasons']) and s['records_audited']['train_seed_2'] == 1
    s = R.audit(faulty(mock, tmp_path / 'z', lambda d: shutil.rmtree(d / 'pass_train_seed_2')), mock['plan'], tmp_path / 'o3', None, make_figures=False)
    assert s['run_status']['status'] == 'INCOMPLETE' and 'pass folder missing: pass_train_seed_2' in s['run_status']['reasons']
    def lose_trace(d): next((d / 'pass_train_seed_1' / 'traces').glob('trace_8.json.gz')).unlink()
    s = R.audit(faulty(mock, tmp_path / 'w', lose_trace), mock['plan'], tmp_path / 'o4', None, make_figures=False)
    assert s['run_status']['status'] == 'INCOMPLETE' and 8 in fails(s, 'train_seed_1', 'R_files_and_hashes')


def test_a_malformed_record_is_a_failed_check_not_a_crash(R, mock, tmp_path):
    for name, edit in {'no_cells': lambda v: v.pop('cells'), 'no_trigger_step': lambda v: v['trigger'].pop('step'), 'cells_not_a_list': lambda v: v.__setitem__('cells', 'x')}.items():
        def f(d, edit=edit):
            p = rec_path(d, 'train_seed_1', 7); v = read(p); edit(v); write(p, v)
        s = R.audit(faulty(mock, tmp_path / name, f), mock['plan'], tmp_path / f'o_{name}', None, make_figures=False)
        assert s['run_status']['status'] == 'INCOMPLETE' and 7 in fails(s, 'train_seed_1', 'R_record_identity'), name


def test_a_missing_frames_file_or_a_broken_ledger_is_never_complete(R, mock, tmp_path):
    def strip(d):
        p = rec_path(d, 'train_seed_2', 8); v = read(p); v.pop('frames'); v.pop('frames_sha256'); write(p, v)
    s = R.audit(faulty(mock, tmp_path / 'f', strip), mock['plan'], tmp_path / 'of', None, make_figures=False)
    assert s['run_status']['status'] == 'INCOMPLETE' and 8 in fails(s, 'train_seed_2', 'R_files_and_hashes')
    for name, text in (('not_json', '{broken'), ('a_list', '[1, 2]'), ('no_plan', '{}')):
        s = R.audit(faulty(mock, tmp_path / name, lambda d, text=text: (d / 'ledger.json').write_text(text)), mock['plan'], tmp_path / f'o_{name}', None, make_figures=False)
        assert s['run_status']['status'] == 'INCOMPLETE' and any('ledger.json' in x for x in s['run_status']['reasons']), name


def test_a_record_that_breaks_its_figures_still_gives_an_incomplete_summary(R, mock, tmp_path):
    pytest.importorskip('matplotlib'); pytest.importorskip('PIL')
    def drop_key(d):                                               # remove one stored frame array; keep the record's hash consistent so only the frame checks can see it
        import numpy as np
        p = rec_path(d, 'train_seed_1', 7); v = read(p); fp = d / 'pass_train_seed_1' / v['frames']
        with np.load(fp) as z: fr = {k: z[k] for k in z.files if k != 'after_C0'}
        fp.unlink(); np.savez_compressed(fp, **fr); v['frames_sha256'] = R.sha(fp); write(p, v)
    s = R.audit(faulty(mock, tmp_path / 'x', drop_key), mock['plan'], tmp_path / 'o', None, figure_scenes=6)
    assert s['run_status']['status'] == 'INCOMPLETE' and 7 in fails(s, 'train_seed_1', 'R_frame_index') and (tmp_path / 'o' / 'audit_summary.json').exists()
    man = read(tmp_path / 'o' / 'trajectory_manifest.json'); assert set(s['figure_errors']) == {f"{f['policy']}:{f['seed']}" for f in man['figures'] if 'error' in f}


def test_a_figure_error_or_a_change_of_the_run_folder_on_valid_records_is_not_complete(R, mock, tmp_path, monkeypatch):
    pytest.importorskip('matplotlib'); pytest.importorskip('PIL')
    d = faulty(mock, tmp_path / 'v', lambda d: None)                                       # a valid copy: every record check passes
    def broken(*a, **k): raise RuntimeError('injected figure failure')
    monkeypatch.setattr(R.T, 'fig_xy', broken)
    s = R.audit(d, mock['plan'], tmp_path / 'o1', None, figure_scenes=1)
    assert s['run_status']['status'] == 'INCOMPLETE' and s['figure_errors'] and not any(v['FAIL'] for pid in s['record_check_counts'] for v in s['record_check_counts'][pid].values())
    def touch(rec, tr, out, policy=None): (d / 'touched_during_audit.txt').write_text('x'); out.write_bytes(b'')        # a run folder that changes while it is audited
    monkeypatch.setattr(R.T, 'fig_xy', touch)
    with pytest.raises(SystemExit, match='changed during the audit'): R.audit(d, mock['plan'], tmp_path / 'o2', None, figure_scenes=1)
    s = read(tmp_path / 'o2' / 'audit_summary.json'); m = read(tmp_path / 'o2' / 'trajectory_manifest.json')
    assert s['run_status']['status'] == 'INCOMPLETE' and m['run_folder_changed_files'] == ['touched_during_audit.txt'] and not m['run_folder_unchanged_by_audit']


def test_an_unreadable_file_of_the_run_folder_gives_an_incomplete_summary(R, mock, tmp_path, monkeypatch):
    d = faulty(mock, tmp_path / 'v', lambda d: (d / 'notes.txt').write_text('x')); real = R.sha
    def sha(p):
        if Path(p).name == 'notes.txt': raise PermissionError('injected: unreadable')
        return real(p)
    monkeypatch.setattr(R, 'sha', sha)
    s = R.audit(d, mock['plan'], tmp_path / 'o', None, make_figures=False)
    assert s['run_status']['status'] == 'INCOMPLETE' and any('could not be read' in x for x in s['run_status']['reasons']) and (tmp_path / 'o' / 'audit_summary.json').exists()


def test_a_listing_error_in_the_run_folder_gives_an_incomplete_summary(R, mock, tmp_path, monkeypatch):
    import pathlib
    d = faulty(mock, tmp_path / 'v', lambda d: None); real = pathlib.Path.rglob
    def rglob(self, pattern, *a, **k):
        if pattern == 'budget_stop.json': raise PermissionError('injected: the folder cannot be listed')
        return real(self, pattern, *a, **k)
    monkeypatch.setattr(pathlib.Path, 'rglob', rglob)
    s = R.audit(d, mock['plan'], tmp_path / 'o', None, make_figures=False)
    assert s['run_status']['status'] == 'INCOMPLETE' and any('could not be inspected' in x for x in s['run_status']['reasons']) and (tmp_path / 'o' / 'audit_summary.json').exists()


def test_a_missing_run_folder_gives_an_incomplete_summary(R, mock, tmp_path):
    s = R.audit(tmp_path / 'no_such_run', mock['plan'], tmp_path / 'o', None, make_figures=False)
    assert s['run_status']['status'] == 'INCOMPLETE' and 'the run folder is missing or is not a folder' in s['run_status']['reasons'] and (tmp_path / 'o' / 'audit_summary.json').exists()
    assert not (tmp_path / 'no_such_run').exists()


def test_a_wrong_snapshot_digest_or_branch_id_is_found(R, mock, tmp_path):
    def edit(d):
        p = rec_path(d, 'train_seed_1', 7); v = read(p); v['cells'][3]['snapshot_digest'] = '0' * 64; v['cells'][4]['branch_id'] = 'x'; write(p, v)
    s = R.audit(faulty(mock, tmp_path / 'x', edit), mock['plan'], tmp_path / 'o', None, make_figures=False)
    assert 7 in fails(s, 'train_seed_1', 'R_branch_and_snapshot') and 7 in s['cases']['train_seed_1']['record_check_failed']
    row = next(l for l in (tmp_path / 'o' / 'per_scene_checks.csv').read_text(encoding='utf-8').splitlines() if l.startswith('train_seed_1,7,*,') and 'R_branch_and_snapshot' in l)
    assert 'FAIL' in row and 'VISION_CENTER_V2' in row and 'CENTER_ORACLE' in row


def test_a_frame_index_mismatch_is_found(R, mock, tmp_path):
    def edit(d):
        p = rec_path(d, 'train_seed_1', 7); v = read(p); c0 = next(c for c in v['cells'] if c['arm'] == 'C0'); c0['frames_after_trigger']['steps'] = c0['frames_after_trigger']['steps'][:-1]; write(p, v)
    s = R.audit(faulty(mock, tmp_path / 'x', edit), mock['plan'], tmp_path / 'o', None, make_figures=False)
    assert 7 in fails(s, 'train_seed_1', 'R_frame_index')
    def dup(d):
        p = rec_path(d, 'train_seed_2', 8); v = read(p); c0 = next(c for c in v['cells'] if c['arm'] == 'C0'); st = c0['frames_after_trigger']['steps']; st[1] = st[0]; write(p, v)
    s = R.audit(faulty(mock, tmp_path / 'y', dup), mock['plan'], tmp_path / 'o2', None, make_figures=False)
    assert 8 in fails(s, 'train_seed_2', 'R_frame_index')


def test_an_abstaining_arm_that_differs_from_C0_is_found(R, mock, tmp_path):
    rec = read(rec_path(mock['run'], 'train_seed_1', 7)); arm = next(c['arm'] for c in rec['cells'] if c['arm'] != 'C0' and not c.get('acted'))
    def edit(d):
        tp = d / 'pass_train_seed_1' / rec['trace']
        with gzip.open(tp, 'rt', encoding='utf-8') as f: v = json.load(f)
        row = v['traces'][arm][-3]; row['cubeB'] = [row['cubeB'][0] + .004, row['cubeB'][1], row['cubeB'][2]]
        tp.unlink()
        with gzip.open(tp, 'wt', encoding='utf-8') as f: json.dump(v, f)
        p = rec_path(d, 'train_seed_1', 7); r = read(p); r['trace_sha256'] = R.sha(tp); write(p, r)                 # consistent hash: only the identity check can see it
    s = R.audit(faulty(mock, tmp_path / 'x', edit), mock['plan'], tmp_path / 'o', None, make_figures=False)
    assert 7 in fails(s, 'train_seed_1', 'R_abstain_pointer_to_C0') and 7 in s['fail_scenes']['train_seed_1'][arm]['9_abstain_keeps_policy'] and s['run_status']['status'] == 'INCOMPLETE'


def test_a_wrong_hand_back_or_control_frame_count_is_found(R, mock, tmp_path):
    def late(d):
        p = rec_path(d, 'train_seed_2', 7); v = read(p); c = next(c for c in v['cells'] if c['acted'] and c['arm'] != 'CENTER_ORACLE'); c['first_inference_step'] += 1; write(p, v)
    s = R.audit(faulty(mock, tmp_path / 'x', late), mock['plan'], tmp_path / 'o', None, make_figures=False)
    assert 7 in fails(s, 'train_seed_2', 'R_handback') and s['run_status']['status'] == 'INCOMPLETE'
    def reading(d):
        p = rec_path(d, 'train_seed_1', 7); v = read(p); c = next(c for c in v['cells'] if c['arm'] == 'VISION_CENTER_V2'); c['control']['readings'] = c['control']['readings'][:-1]; write(p, v)
    s = R.audit(faulty(mock, tmp_path / 'y', reading), mock['plan'], tmp_path / 'o2', None, make_figures=False)
    assert 7 in fails(s, 'train_seed_1', 'R_frame_index')


def test_an_interrupted_run_is_never_taken_for_a_complete_one(R, mock, tmp_path):
    cases = {'budget_stop': lambda d: write(d / 'budget_stop.json', {'kind': 'deadline', 'scene': 8}),
             'pass_budget_stop': lambda d: write(d / 'pass_train_seed_2' / 'budget_stop.json', {'before_scene': 8}),
             'error_file': lambda d: write(d / 'pass_train_seed_1' / 'scenes' / '8.error.json', {'completed': False}),
             'no_ledger': lambda d: (d / 'ledger.json').unlink(),
             'foreign_plan_record': lambda d: write(rec_path(d, 'train_seed_1', 8), {**read(rec_path(d, 'train_seed_1', 8)), 'plan_sha256': '0' * 64})}
    for name, edit in cases.items():
        s = R.audit(faulty(mock, tmp_path / name, edit), mock['plan'], tmp_path / f'o_{name}', None, make_figures=False)
        assert s['run_status']['status'] == 'INCOMPLETE' and s['run_status']['reasons'], name


def test_the_command_line_returns_0_complete_3_incomplete_4_subset(R, mock, tmp_path):
    cmd = [sys.executable, 'scripts/35_c1_trajectory_audit.py', '--run', str(mock['run']), '--plan', str(mock['plan']), '--no-figures']
    a = subprocess.run(cmd + ['--out', str(tmp_path / 'a')], capture_output=True, text=True); assert a.returncode == 0 and '"COMPLETE"' in a.stdout, a.stderr[-2000:]
    b = subprocess.run(cmd + ['--seeds', '7-9', '--out', str(tmp_path / 'b')], capture_output=True, text=True); assert b.returncode == 3 and 'INCOMPLETE' in b.stdout
    c = subprocess.run(cmd + ['--seeds', '8', '--out', str(tmp_path / 'c')], capture_output=True, text=True); assert c.returncode == 4 and 'SUBSET_ONLY' in c.stdout
