"""Tests of the generalization runner gen_arch_plainconv_001 (scripts/41_gen_arch_plainconv.py) with the mock runtime of
the C1 tests on synthetic seeds 7, 8: no new development, calibration or test scene is used; the real ledger is untouched."""
import argparse, importlib.util, json
from pathlib import Path

import pytest

from hci import v2_confirm as V

REAL_LEDGER = Path('artifacts/ledger/gen_arch_plainconv_001.json')


@pytest.fixture
def X41(monkeypatch):
    pytest.importorskip('torch')
    from test_recovery_v1_oracle import CFG
    from test_vision_center import BiasPolicy
    from test_vision_center_v2 import VisionEnv2
    from test_trajectory_audit import probe2
    from hci.simulation import Runtime
    spec = importlib.util.spec_from_file_location('gen41', Path('scripts/41_gen_arch_plainconv.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    class Env(VisionEnv2):
        def close(self): pass
    monkeypatch.setattr(m, 'make_runtime', lambda c, p, u, d: Runtime(Env(), BiasPolicy((.028, .012)), CFG)); monkeypatch.setattr(m, 'load_config', lambda: CFG); monkeypatch.setattr(m, 'probe', lambda: probe2)
    existed = REAL_LEDGER.exists(); yield m; assert REAL_LEDGER.exists() == existed


def ns(**k): return argparse.Namespace(**k)


def test_constants_are_the_declared_design(X41):
    assert X41.SEEDS == (51300, 51499) and X41.ARMS[0] == 'C0' and 'VISION_CENTER_V2' in X41.ARMS and X41.POLICY.endswith('stack_1d_s1_d500_clean_v1_i080000')
    assert not set(X41.seeds()) & (set(range(30000, 30400)) | set(range(90000, 90400)) | set(range(50000, 51300)))
    meta = json.loads(Path(X41.POLICY, 'policy_meta.json').read_text(encoding='utf-8')); assert meta['args']['obs_horizon'] == 2 and meta['env_id'] == 'StackCube-v1'


def test_plan_authorization_ledger_run_analyze_and_refusals(X41, tmp_path):
    run = tmp_path / 'run'; X41.cmd_plan(ns(run=str(run)))
    with pytest.raises(SystemExit, match='exists'): X41.cmd_plan(ns(run=str(run)))
    led = tmp_path / 'ledger.json'; args = dict(run=str(run), upstream='x', device='cpu', seeds=['7', '8'], ledger=str(led))
    with pytest.raises(SystemExit, match='no authorization'): X41.cmd_run(ns(**args))
    (run / 'authorization.json').write_text(json.dumps({'approved_by': 'user', 'plan_sha256': '0' * 64, 'cap_usd': 1.5}))
    with pytest.raises(SystemExit, match='does not name'): X41.cmd_run(ns(**args))
    (run / 'authorization.json').write_text(json.dumps({'approved_by': 'user', 'plan_sha256': V.sha256(run / 'plan.json'), 'cap_usd': 1.5}))
    with pytest.raises(SystemExit, match='calibration or test'): X41.cmd_run(ns(**{**args, 'seeds': ['90000']}))
    X41.cmd_run(ns(**args))
    with pytest.raises(SystemExit, match='never resumed'): X41.cmd_run(ns(**args))
    X41.cmd_analyze(ns(run=str(run), out=str(tmp_path / 'a.json'), seeds=['7', '8']))
    res = json.loads((tmp_path / 'a.json').read_text()); assert res['scenes'] == 2 and set(res['arms']) == set(X41.ARMS) and 'p_one_sided' in res['H_arch']
    with pytest.raises(SystemExit, match='INCOMPLETE'): X41.cmd_analyze(ns(run=str(run), out=str(tmp_path / 'b.json'), seeds=['7', '8', '9']))
