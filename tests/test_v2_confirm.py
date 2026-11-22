"""Tests of the confirmatory Calibration / Final Test runner for Vision-CENTER V2 (hci.v2_confirm, scripts/40_v2_confirm.py).

Unit tests use synthetic outcomes. The end-to-end tests run the real driver commands (freeze, run, analyze) with the mock
runtime of the C1 tests (mock physics, synthetic renderer, scripted policy) on synthetic seeds 7 and 8: no calibration or
test seed is reset, rendered or rolled out, and the real ledger folder is never touched (checked).
"""
import argparse, hashlib, importlib.util, json
from pathlib import Path

import pytest

from hci import v2_confirm as V

REAL_LEDGER_DIR = Path('artifacts/ledger')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- seeds
def test_split_seeds_are_exactly_the_declared_ranges_and_disjoint_from_development():
    cal = V.split_seeds('configs/splits.json', 'calibration'); test = V.split_seeds('configs/splits.json', 'test')
    assert cal == list(range(30000, 30400)) and test == list(range(90000, 90400))
    c1 = json.loads(Path('configs/vision_center_c1_splits.json').read_text(encoding='utf-8'))['splits']['vision_center_v2_c1_dev']
    sp = json.loads(Path('configs/splits.json').read_text(encoding='utf-8'))['splits']
    assert V.seeds_disjoint({'cal': cal, 'test': test, 'c1': c1, 'dev': sp['dev'], 'train': sp['train'], 'audit': sp['audit'], 'phaseB': list(range(50000, 51100))}) == []
    with pytest.raises(ValueError): V.split_seeds('configs/splits.json', 'dev')


# ---------------------------------------------------------------- certificate and efficacy
def outc(n_plus, k, n_fail=0, rescues=0, moved=0):
    o = {}; s = 0
    for i in range(n_plus): o[s] = (1, 0 if i < k else 1, i < max(k, moved)); s += 1
    for i in range(n_fail): o[s] = (0, 1 if i < rescues else 0, i < rescues); s += 1
    return o


def test_certificate_matches_the_binomial_boundaries_of_the_readiness_audit():
    for fam_size, kmax in ((1, 9), (2, 8), (4, 7)):                             # V2_CALIBRATION_READINESS.md section 5, n_plus = 330
        assert V.power_table(330, fam_size)['largest_k_that_passes'] == kmax
    fam = V.family(['train_seed_1'])
    assert V.certificate({'train_seed_1': outc(330, 9)}, fam)['members'][0]['certified']
    assert not V.certificate({'train_seed_1': outc(330, 10)}, fam)['members'][0]['certified']


def test_holm_family_joint_statement_and_zero_denominator():
    fam = V.family(['train_seed_1', 'train_seed_2'])
    r = V.certificate({'train_seed_1': outc(330, 8), 'train_seed_2': outc(330, 8)}, fam)
    assert all(m['certified'] for m in r['members']) and r['joint_statement_all_policies']
    r = V.certificate({'train_seed_1': outc(330, 0), 'train_seed_2': outc(330, 12)}, fam)
    assert [m['certified'] for m in r['members']] == [True, False] and not r['joint_statement_all_policies']
    r = V.certificate({'train_seed_1': outc(0, 0, n_fail=50), 'train_seed_2': outc(330, 0)}, fam)
    assert r['members'][0]['n_plus'] == 0 and not r['members'][0]['certified'] and not r['joint_statement_all_policies']
    with pytest.raises(ValueError): V.family(['train_seed_1', 'train_seed_1'])
    with pytest.raises(ValueError): V.family(['train_seed_3'])


def test_efficacy_is_exact_paired_and_separate_from_the_certificate():
    o = outc(330, 1, n_fail=70, rescues=16)
    e = V.efficacy({'train_seed_1': o}, ['train_seed_1'])['per_policy'][0]
    assert (e['v2_only'], e['c0_only']) == (16, 1) and e['p_one_sided'] == pytest.approx(sum(__import__('math').comb(17, j) for j in range(16, 18)) / 2 ** 17) and e['rejected_null']
    assert V.efficacy({'p': {0: (1, 1, False)}}, ['p'])['per_policy'][0]['p_one_sided'] == 1.0


def test_outcomes_refuse_missing_arms_and_duplicates():
    good = {'scene': 1, 'cells': [{'arm': 'C0', 'y': 1}, {'arm': V.CANDIDATE, 'y': 1, 'acted': False}]}
    assert V.outcomes([good]) == {1: (1, 1, False)}
    with pytest.raises(ValueError): V.outcomes([{'scene': 1, 'cells': [{'arm': 'C0', 'y': 1}]}])
    with pytest.raises(ValueError): V.outcomes([good, good])


# ---------------------------------------------------------------- refusals
def test_partial_runs_approval_records_markers_and_token(tmp_path):
    d = tmp_path / 'pass'; (d / 'scenes').mkdir(parents=True); (d / 'scenes' / '7.json').write_text('{}')
    with pytest.raises(SystemExit, match='INCOMPLETE'): V.check_complete(d, [7, 8])
    (d / 'scenes' / '8.json').write_text('{}'); assert V.check_complete(d, [7, 8]) == [7, 8]
    (d / 'scenes' / '8.error.json').write_text('{}')
    with pytest.raises(SystemExit, match='INCOMPLETE'): V.check_complete(d, [7, 8])
    am = tmp_path / 'amendment.md'; am.write_text('amendment'); rec = tmp_path / 'approval.json'
    with pytest.raises(SystemExit, match='no amendment approval'): V.require_amendment_approval(rec, am)
    rec.write_text(json.dumps({'approved_by': 'user', 'amendment_sha256': '0' * 64}))
    with pytest.raises(SystemExit, match='does not name'): V.require_amendment_approval(rec, am)
    rec.write_text(json.dumps({'approved_by': 'user', 'amendment_sha256': sha(am)})); assert V.require_amendment_approval(rec, am)['approved_by'] == 'user'
    V.create_exclusive(tmp_path / 'l.json', {'x': 1})
    with pytest.raises(FileExistsError): V.create_exclusive(tmp_path / 'l.json', {'x': 2})
    with pytest.raises(PermissionError): V.require_test_approval('run_frozen_test')
    assert V.require_test_approval(V.APPROVAL)
    man = V.manifest([str(am)], {}); assert V.verify_manifest(man)
    am.write_text('changed')
    with pytest.raises(SystemExit, match='changed'): V.verify_manifest(man)


# ---------------------------------------------------------------- end to end with the mock runtime (torch needed by the mock policy)
@pytest.fixture
def X40():
    spec = importlib.util.spec_from_file_location('v2_confirm_driver', Path('scripts/40_v2_confirm.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture
def rig(X40, tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from test_recovery_v1_oracle import CFG
    from test_vision_center import BiasPolicy
    from test_vision_center_v2 import VisionEnv2
    from test_trajectory_audit import probe2
    from hci.simulation import Runtime
    before = sorted(p.name for p in REAL_LEDGER_DIR.glob('*')) if REAL_LEDGER_DIR.exists() else []
    class Env(VisionEnv2):
        def close(self): pass
    monkeypatch.setattr(X40, 'make_runtime', lambda c, policy_path, upstream, device: Runtime(Env(), BiasPolicy((.028, .012) if 's1' in policy_path else (-.028, .0)), CFG))
    monkeypatch.setattr(X40, 'load_config', lambda: CFG); monkeypatch.setattr(X40, 'probe', lambda: probe2)
    am = tmp_path / 'AMENDMENT_017_FINAL.md'; am.write_text('the amendment text'); ap = tmp_path / 'approval_017.json'
    ap.write_text(json.dumps({'approved_by': 'user', 'amendment_sha256': sha(am), 'scope': ['test fixture']}))
    yield {'tmp': tmp_path, 'am': am, 'ap': ap, 'ledgers': tmp_path / 'ledger'}
    assert (sorted(p.name for p in REAL_LEDGER_DIR.glob('*')) if REAL_LEDGER_DIR.exists() else []) == before, 'the test touched the real ledger folder'


def ns(**k): return argparse.Namespace(**k)


def test_end_to_end_calibration_then_test_with_every_refusal(X40, rig):
    t = rig['tmp']; F = t / 'frozen_cal'
    X40.cmd_freeze(ns(stage='calibration', out=str(F), amendment=str(rig['am']), approval_record=str(rig['ap']), policies=['train_seed_1', 'train_seed_2'], upstream_commit='a4a4f92', pip_freeze=None, calibration_freeze=None, calibration_analysis=None))
    with pytest.raises(SystemExit, match='exists'):
        X40.cmd_freeze(ns(stage='calibration', out=str(F), amendment=str(rig['am']), approval_record=str(rig['ap']), policies=['train_seed_1'], upstream_commit='x', pip_freeze=None, calibration_freeze=None, calibration_analysis=None))
    run = t / 'cal_run'
    with pytest.raises(SystemExit, match='calibration or test seeds'):
        X40.cmd_run(ns(frozen=str(F), policy='train_seed_1', out=str(run), approval=None, upstream='x', device='cpu', seeds=['30000'], ledger_dir=str(rig['ledgers'])))
    with pytest.raises(SystemExit, match='Final Test only'):
        X40.cmd_run(ns(frozen=str(F), policy='train_seed_1', out=str(run), approval=V.APPROVAL, upstream='x', device='cpu', seeds=['7', '8'], ledger_dir=str(rig['ledgers'])))
    for pid in ('train_seed_1', 'train_seed_2'):
        X40.cmd_run(ns(frozen=str(F), policy=pid, out=str(run), approval=None, upstream='x', device='cpu', seeds=['7', '8'], ledger_dir=str(rig['ledgers'])))
    with pytest.raises(SystemExit, match='never resumed'):
        X40.cmd_run(ns(frozen=str(F), policy='train_seed_1', out=str(run), approval=None, upstream='x', device='cpu', seeds=['7', '8'], ledger_dir=str(t / 'other_ledger')))
    with pytest.raises(FileExistsError):                                       # a fresh output folder cannot bypass the single-use ledger
        X40.cmd_run(ns(frozen=str(F), policy='train_seed_1', out=str(t / 'cal_run_2'), approval=None, upstream='x', device='cpu', seeds=['7', '8'], ledger_dir=str(rig['ledgers'])))
    ana = t / 'cal_analysis.json'
    X40.cmd_analyze(ns(frozen=str(F), run=str(run), out=str(ana), seeds=['7', '8']))
    res = json.loads(ana.read_text()); assert [m['policy'] for m in res['certificate']['members']] == ['train_seed_1', 'train_seed_2'] and all(m['scenes'] == 2 for m in res['certificate']['members'])
    assert 'efficacy' not in res and len(res['record_sha256']) == 4
    with pytest.raises(SystemExit, match='INCOMPLETE'): X40.cmd_analyze(ns(frozen=str(F), run=str(run), out=str(t / 'a2.json'), seeds=['7', '8', '9']))
    # a changed frozen file blocks every later command
    tf = t / 'frozen_test'
    X40.cmd_freeze(ns(stage='test', out=str(tf), amendment=str(rig['am']), approval_record=str(rig['ap']), policies=['train_seed_1', 'train_seed_2'], upstream_commit='a4a4f92', pip_freeze=None,
                      calibration_freeze=str(F / 'manifest.json'), calibration_analysis=str(ana)))
    trun = t / 'test_run'
    with pytest.raises(PermissionError): X40.cmd_run(ns(frozen=str(tf), policy='train_seed_1', out=str(trun), approval=None, upstream='x', device='cpu', seeds=['7', '8'], ledger_dir=str(rig['ledgers'])))
    assert not (tf / 'TEST_STARTED_train_seed_1.json').exists()
    X40.cmd_run(ns(frozen=str(tf), policy='train_seed_1', out=str(trun), approval=V.APPROVAL, upstream='x', device='cpu', seeds=['7', '8'], ledger_dir=str(rig['ledgers'])))
    assert (tf / 'TEST_STARTED_train_seed_1.json').exists()
    with pytest.raises(FileExistsError):                                       # the started marker blocks a second Test of the same policy
        X40.cmd_run(ns(frozen=str(tf), policy='train_seed_1', out=str(t / 'test_run_2'), approval=V.APPROVAL, upstream='x', device='cpu', seeds=['7', '8'], ledger_dir=str(t / 'fresh_ledger')))
    ana.write_text('{"changed": true}')
    with pytest.raises(SystemExit, match='frozen inputs changed'): X40.cmd_verify(ns(frozen=str(tf)))
