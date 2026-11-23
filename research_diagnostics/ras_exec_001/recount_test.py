"""Independent local recount of the Final Test analysis from the downloaded raw scene records (counting without hci; scipy
binomial; hand-written exact paired test and Holm), compared with the server's test_analysis.json. Read-only.

usage: python -P research_diagnostics/ras_exec_001/recount_test.py --run <test folder> --server <test_analysis.json> --out <json>
"""
import argparse, hashlib, json, math
from pathlib import Path
from scipy.stats import binom

EPS = DELTA = ALPHA = 0.05


def exact_one_sided(b, c):
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, j) for j in range(b, n + 1)) / 2 ** n


def holm(items, key, level):
    ps = sorted(items, key=lambda e: e[key]); m = len(ps); stop = False
    for j, e in enumerate(ps):
        e[f'{key}_holm_level'] = level / (m - j); ok = (not stop) and e[key] <= e[f'{key}_holm_level']
        e[f'{key}_rejected'] = ok
        if not ok: stop = True


def main():
    a = argparse.ArgumentParser(); a.add_argument('--run', required=True); a.add_argument('--server', required=True); a.add_argument('--out', required=True); a = a.parse_args()
    res = {}
    for pid in ('train_seed_1', 'train_seed_2'):
        d = Path(a.run) / f'pass_{pid}' / 'scenes'; recs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(d.glob('*.json')) if p.stem.isdigit()]
        assert sorted(r['scene'] for r in recs) == list(range(90000, 90400)), (pid, len(recs))
        assert not list(d.glob('*.error.json')) and (d.parent / 'PASS_DONE.json').is_file()
        for r in recs: assert hashlib.sha256((d.parent / r['trace']).read_bytes()).hexdigest() == r['trace_sha256']
        y = {r['scene']: {c['arm']: c for c in r['cells']} for r in recs}
        n_plus = sum(1 for s in y if y[s]['C0']['y'] == 1); k = sum(1 for s in y if y[s]['C0']['y'] == 1 and y[s]['VISION_CENTER_V2']['y'] == 0)
        b = sum(1 for s in y if y[s]['C0']['y'] == 0 and y[s]['VISION_CENTER_V2']['y'] == 1)
        res[pid] = {'scenes': len(recs), 'triggered': sum(1 for r in recs if r['triggered']), 'v2_acted': sum(1 for s in y if y[s]['VISION_CENTER_V2'].get('acted')),
                    'c0_success': n_plus, 'v2_success': sum(1 for s in y if y[s]['VISION_CENTER_V2']['y'] == 1), 'n_plus': n_plus, 'k_harm': k, 'v2_only': b, 'c0_only': k,
                    'harm_p': float(binom.cdf(k, n_plus, EPS)), 'efficacy_p': exact_one_sided(b, k)}
    items = list(res.values()); holm(items, 'harm_p', DELTA); holm(items, 'efficacy_p', ALPHA)
    srv = json.loads(Path(a.server).read_text(encoding='utf-8')); ok = True
    for m in srv['certificate']['members']:
        e = res[m['policy']]; same = (m['n_plus'], m['k_harm']) == (e['n_plus'], e['k_harm']) and abs(m['p_value'] - e['harm_p']) < 1e-12 and m['certified'] == (e['harm_p_rejected'] and e['n_plus'] > 0)
        e['certificate_equals_server'] = same; ok &= same
    for m in srv['efficacy']['per_policy']:
        e = res[m['policy']]; same = (m['v2_only'], m['c0_only']) == (e['v2_only'], e['c0_only']) and abs(m['p_one_sided'] - e['efficacy_p']) < 1e-12 and m['rejected_null'] == e['efficacy_p_rejected']
        e['efficacy_equals_server'] = same; ok &= same
    out = {'recount': res, 'all_equal_server': ok}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding='utf-8'); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
