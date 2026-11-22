"""Regression tests of the diagnostic loaders, denominators and row mapping (POST_HOC_EXPLORATORY code).

They need the local C1 run folder and the synced Drive copy of the C1 analysis outputs; without them they are skipped
(never passed silently: pytest reports the skip)."""
from __future__ import annotations

import csv, gzip, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

needs_data = pytest.mark.skipif(not (C.RUN.exists() and (C.MECH / 'tables' / 's6_pair_rows.csv.gz').exists() and (C.TRAJ / 'per_scene_checks.csv').exists()),
                                reason='C1 run folder or analysis outputs not available')


@needs_data
def test_s6_denominators_reproduce_the_official_counts():
    import diagnose_s6_offline as D
    d = D.load_rows(C.MECH / 'tables' / 's6_pair_rows.csv.gz')
    got = {p: (int((d.policy == p).sum()), int(d[d.policy == p].over_tol.sum())) for p in C.POLICIES}
    assert got == {'train_seed_1': (293614, 27376), 'train_seed_2': (286925, 26891)}
    assert set(d.impulse_group) <= {'nonzero_impulse', 'zero_impulse', 'missing'} and (d.impulse_group == 'missing').sum() == 0
    assert (d.physx_sep_m > 0).all()                      # the S6 population: pairs PhysX reports separated


@needs_data
def test_s6_row_mapping_to_raw_trace_and_points():
    """A stored pair row maps back to its raw trace row: the existing reconstruction recomputed there equals the stored
    value, and the minimum separation over the pair's PhysX points equals the stored physx_sep."""
    import diagnose_s6_offline as D
    d = D.load_rows(C.MECH / 'tables' / 's6_pair_rows.csv.gz')
    rows = d[(d.policy == 'train_seed_1') & (d.scene == 51100)].head(25).assign(selection='test').to_dict('records')
    out = D.recompute(('train_seed_1', 51100, rows))
    assert len(out) == 25
    for o in out:
        assert o['recon_mapping_abs_err_m'] <= 1e-12
        assert abs(o['point_sep_min_m'] - o['physx_sep_m']) <= 1e-12 and o['n_points'] >= 1


@needs_data
def test_trace_hash_is_checked(monkeypatch):
    rec, h = C.record('train_seed_1', 51100); bad = dict(rec, trace_sha256='0' * 64)
    monkeypatch.setattr(C, 'record', lambda p, s: (bad, h))
    with pytest.raises(ValueError):
        C.traces('train_seed_1', 51100)


@needs_data
def test_check4_denominators_and_one_known_failure():
    import diagnose_check4_offline as B
    audit = B.audit_rows(C.TRAJ / 'per_scene_checks.csv')
    v2 = {k: r['status'] for k, r in audit.items() if k[2] == 'VISION_CENTER_V2' and r['status'] in ('PASS', 'FAIL')}
    assert sum(1 for k in v2 if k[0] == 'train_seed_1') == 41 and sum(1 for k in v2 if k[0] == 'train_seed_2') == 39
    assert sum(1 for k, s in v2.items() if k[0] == 'train_seed_1' and s == 'FAIL') == 9 and sum(1 for k, s in v2.items() if k[0] == 'train_seed_2' and s == 'FAIL') == 5
    m, steps = B.analyse_move('train_seed_1', 51185, 'VISION_CENTER_V2', audit)
    assert m['audit_csv_status'] == m['rerun_audit_status'] == m['recomputed_status'] == 'FAIL' and m['failed_steps'] == '1'
    assert m['POST_HOC_STEP2PLUS_status'] == 'PASS' and steps[0]['cos_fail'] and not steps[0]['progress_fail']
    assert abs(steps[0]['cos_act_cmd'] - 0.6625799224621143) < 1e-9          # the value the audit recorded


@needs_data
def test_check4_command_and_step_alignment():
    """Command j is the delta target set at row t0+j-1 and the actual step is row t0+j-1 -> t0+j: the recorded
    tcp_after of command j is the TCP of row t0+j."""
    rec, _ = C.record('train_seed_1', 51157); tr, _ = C.traces('train_seed_1', 51157); c = C.cell(rec, 'VISION_CENTER_V2'); t0 = rec['trigger']['step']
    rows = {int(r['t']): r for r in tr['VISION_CENTER_V2']}
    for j, s in enumerate(c['control']['steps']):
        assert np.allclose(s['tcp_after'], rows[t0 + j + 1]['tcp'], atol=0)
        assert np.allclose(np.asarray(rows[t0 + j + 1]['action'][:2]) * 0.1, s['commanded_xy_m'], atol=1e-8)
