"""Independent recount of every headline number of the manuscript from the hash-checked raw scene records (read-only).

Two implementations per quantity:
  A  the project code (hci.v2_confirm.certificate / efficacy, hci.vision_center_c1.exact_one_sided / primary_tests)
  B  a hand-written count from the raw cells + scipy.stats.binom + a hand-written exact paired test and Holm
Any disagreement raises. Post-hoc DESCRIPTIVE quantities (paired bootstrap of the success difference, one-sided
Clopper-Pearson upper bounds of conditional harm) are added and labelled; they change no pre-registered test.

usage: python -P paper/ras_when_recovery_hurts_001/figures/source_code/recount_all.py --out paper/ras_when_recovery_hurts_001/data/recount.json
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'src'))
from hci import v2_confirm as V  # noqa: E402
from hci import vision_center_c1 as C1  # noqa: E402

SRC = {
    'C1': REPO / 'runs/vision_center_v2_c1_001_server/extracted/runs/vision_center_v2_c1_001',
    'CAL': REPO / 'runs/ras_exec_001/download_final/runs/ras_v2_confirm_001/calibration',
    'TEST': REPO / 'runs/ras_exec_001/download_test/runs/ras_v2_confirm_001/test',
    'B3': REPO / 'runs/ras_exec_001/download_phase1/runs/gen_arch_plainconv_001',
}
POL = ('train_seed_1', 'train_seed_2')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(dirpass, seeds):
    recs = []; d = Path(dirpass) / 'scenes'
    for s in seeds:
        p = d / f'{s}.json'; r = json.loads(p.read_text(encoding='utf-8'))
        if sha(Path(dirpass) / r['trace']) != r['trace_sha256']: raise ValueError(f'trace hash mismatch {p}')
        recs.append(r)
    if list((d).glob('*.error.json')): raise ValueError(f'error records in {d}')
    return recs


def cells(recs, arm): return {r['scene']: next(c for c in r['cells'] if c['arm'] == arm) for r in recs}


def exact_b(b, c):  # implementation B: P(X >= b), X ~ Bin(b + c, 1/2)
    n = b + c
    return 1.0 if n == 0 else float(binom.sf(b - 1, n, 0.5))


def holm_b(ps, level):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [False] * len(ps)
    for j, i in enumerate(order):
        if ps[i] <= level / (len(ps) - j): out[i] = True
        else: break
    return out


def boot_diff(y0, y1, reps=20000, seed=20261123):
    d = np.asarray(y1, float) - np.asarray(y0, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d))); m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cp_upper(k, n, conf=0.95): return 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))


def paired(recs, ref, arm):
    a, b = cells(recs, ref), cells(recs, arm); S = sorted(a)
    y0 = [int(a[s]['y']) for s in S]; y1 = [int(b[s]['y']) for s in S]
    resc = sum(1 for p, q in zip(y0, y1) if p == 0 and q == 1); harm = sum(1 for p, q in zip(y0, y1) if p == 1 and q == 0)
    return {'n': len(S), 'success_ref': sum(y0), 'success_arm': sum(y1), 'rescue': resc, 'harm': harm, 'net': resc - harm,
            'ref_failures': len(S) - sum(y0), 'ref_successes': sum(y0), 'acted': sum(1 for s in S if b[s].get('acted'))}, y0, y1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(); out = Path(a.out)
    res = {'what': 'independent recount (implementations A and B) from hash-checked raw records', 'sources': {k: str(v.relative_to(REPO)).replace('\\', '/') for k, v in SRC.items()}}
    # ---------------- C1 development
    c1 = {}
    for pid in POL:
        recs = load(SRC['C1'] / f'pass_{pid}', range(51100, 51300)); e = {'triggered': sum(1 for r in recs if r['triggered'])}
        for arm in C1.BASE_ARMS:
            if arm == 'C0': e['C0'] = {'success': sum(int(c['y']) for c in cells(recs, 'C0').values())}; continue
            st, y0, y1 = paired(recs, 'C0', arm); st['p_exact_vs_C0'] = exact_b(st['rescue'], st['harm']); e[arm] = st
        st, y0, y1 = paired(recs, 'V2_SINGLE_FRAME', 'VISION_CENTER_V2'); e['V2_vs_SINGLE_FRAME'] = {'v2_only': st['rescue'], 'single_only': st['harm'], 'p_B': exact_b(st['rescue'], st['harm']), 'p_A': C1.exact_one_sided(st['rescue'], st['harm'])}
        c1[pid] = e
    p_gen = exact_b(c1['train_seed_2']['VISION_CENTER_V2']['rescue'], c1['train_seed_2']['VISION_CENTER_V2']['harm'])
    p_mem = max(c1[p]['V2_vs_SINGLE_FRAME']['p_B'] for p in POL)
    hb = holm_b([p_gen, p_mem], 0.05)
    c1['H_gen'] = {'p': p_gen, 'rejected': hb[0]}; c1['H_mem'] = {'p_intersection_union': p_mem, 'rejected': hb[1]}
    assert all(abs(c1[p]['V2_vs_SINGLE_FRAME']['p_A'] - c1[p]['V2_vs_SINGLE_FRAME']['p_B']) < 1e-12 for p in POL)
    res['C1_development'] = c1
    # ---------------- calibration and test
    for stage, key, seeds in (('calibration', 'CAL', range(30000, 30400)), ('test', 'TEST', range(90000, 90400))):
        by = {}; sect = {}
        for pid in POL:
            recs = load(SRC[key] / f'pass_{pid}', seeds); st, y0, y1 = paired(recs, 'C0', 'VISION_CENTER_V2')
            st['triggered'] = sum(1 for r in recs if r['triggered']); st['harm_p_B'] = float(binom.cdf(st['harm'], st['ref_successes'], 0.05))
            st['harm_rate'] = st['harm'] / st['ref_successes']; st['harm_cp95_upper_POSTHOC'] = cp_upper(st['harm'], st['ref_successes'])
            st['success_diff'] = (st['success_arm'] - st['success_ref']) / st['n']; st['success_diff_boot95_POSTHOC'] = boot_diff(y0, y1)
            st['eff_p_B'] = exact_b(st['rescue'], st['harm'])
            moved = cells(recs, 'VISION_CENTER_V2'); base = cells(recs, 'C0')
            st['moved_among_baseline_success'] = sum(1 for s in base if base[s]['y'] == 1 and moved[s].get('acted'))
            sect[pid] = st; by[pid] = V.outcomes(recs)
        cert = V.certificate(by, V.family(POL)); eff = V.efficacy(by, list(POL))
        hc = holm_b([sect[p]['harm_p_B'] for p in POL], 0.05); he = holm_b([sect[p]['eff_p_B'] for p in POL], 0.05)
        for i, pid in enumerate(POL):
            m = next(x for x in cert['members'] if x['policy'] == pid); e = next(x for x in eff['per_policy'] if x['policy'] == pid)
            assert (m['n_plus'], m['k_harm']) == (sect[pid]['ref_successes'], sect[pid]['harm']) and abs(m['p_value'] - sect[pid]['harm_p_B']) < 1e-12 and m['certified'] == hc[i]
            assert (e['v2_only'], e['c0_only']) == (sect[pid]['rescue'], sect[pid]['harm']) and abs(e['p_one_sided'] - sect[pid]['eff_p_B']) < 1e-12 and e['rejected_null'] == he[i]
            sect[pid]['harm_certified'] = hc[i]; sect[pid]['efficacy_rejected'] = he[i]
        sect['joint_harm_statement'] = all(hc)
        res[stage] = sect
    # ---------------- B3
    recs = load(SRC['B3'] / 'pass', range(51300, 51500)); b3 = {'triggered': sum(1 for r in recs if r['triggered']), 'C0': sum(int(c['y']) for c in cells(recs, 'C0').values())}
    for arm in ('V2_SINGLE_FRAME', 'VISION_CENTER_V2', 'CENTER_ORACLE'):
        st, y0, y1 = paired(recs, 'C0', arm); st['p_exact_vs_C0'] = exact_b(st['rescue'], st['harm']); b3[arm] = st
    b3['VISION_CENTER_V2']['success_diff_boot95_POSTHOC'] = boot_diff(*paired(recs, 'C0', 'VISION_CENTER_V2')[1:])
    srv = json.loads((SRC['B3'] / 'analysis.json').read_text(encoding='utf-8')) if (SRC['B3'] / 'analysis.json').exists() else json.loads((REPO / 'results/ras_exec_001/b3_generalization/analysis.json').read_text(encoding='utf-8'))
    assert srv['H_arch']['v2_only'] == b3['VISION_CENTER_V2']['rescue'] and srv['H_arch']['c0_only'] == b3['VISION_CENTER_V2']['harm'] and abs(srv['H_arch']['p_one_sided'] - b3['VISION_CENTER_V2']['p_exact_vs_C0']) < 1e-12
    oracle_rescue = {r['scene'] for r in recs if cells([r], 'C0')[r['scene']]['y'] == 0 and cells([r], 'CENTER_ORACLE')[r['scene']]['y'] == 1}
    b3['oracle_rescues_recovered_by_v2'] = sum(1 for r in recs if r['scene'] in oracle_rescue and cells([r], 'VISION_CENTER_V2')[r['scene']]['y'] == 1); b3['oracle_rescues'] = len(oracle_rescue)
    res['B3_development'] = b3
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({'C1': {p: {k: (c1[p][k]['success_arm'], c1[p][k]['rescue'], c1[p][k]['harm']) for k in ('VISION_CENTER_V2', 'V2_SINGLE_FRAME', 'CENTER_ORACLE')} for p in POL}, 'H_gen': c1['H_gen'], 'H_mem': c1['H_mem'],
                      'cal': {p: (res['calibration'][p]['ref_successes'], res['calibration'][p]['harm'], res['calibration'][p]['harm_p_B'], res['calibration'][p]['harm_cp95_upper_POSTHOC']) for p in POL},
                      'test': {p: (res['test'][p]['success_ref'], res['test'][p]['success_arm'], res['test'][p]['rescue'], res['test'][p]['harm'], res['test'][p]['eff_p_B'], res['test'][p]['success_diff_boot95_POSTHOC']) for p in POL},
                      'B3': (b3['C0'], b3['VISION_CENTER_V2']['success_arm'], b3['VISION_CENTER_V2']['rescue'], b3['VISION_CENTER_V2']['harm'], b3['VISION_CENTER_V2']['p_exact_vs_C0'])}, indent=1))


if __name__ == '__main__':
    main()
