"""Independent local recount of the calibration certificate from the downloaded raw scene records (no hci import for the
counting; scipy for the binomial and a hand-written Holm), compared with the server's calibration_analysis.json.

usage: python -P research_diagnostics/ras_exec_001/recount_calibration.py --run <calibration folder> --server <calibration_analysis.json> --out <json>
"""
import argparse, gzip, hashlib, json
from pathlib import Path
from scipy.stats import binom

EPS = DELTA = 0.05


def main():
    a = argparse.ArgumentParser(); a.add_argument('--run', required=True); a.add_argument('--server', required=True); a.add_argument('--out', required=True); a = a.parse_args()
    res = {}; ok = True
    for pid in ('train_seed_1', 'train_seed_2'):
        d = Path(a.run) / f'pass_{pid}' / 'scenes'; recs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(d.glob('*.json')) if p.stem.isdigit()]
        seeds = sorted(r['scene'] for r in recs); assert seeds == list(range(30000, 30400)), (pid, len(seeds))
        assert not list(d.glob('*.error.json'))
        for r in recs:
            tp = d.parent / r['trace']; assert hashlib.sha256(tp.read_bytes()).hexdigest() == r['trace_sha256'], (pid, r['scene'])
        y = {r['scene']: {c['arm']: c for c in r['cells']} for r in recs}
        n_plus = sum(1 for s in y if y[s]['C0']['y'] == 1); k = sum(1 for s in y if y[s]['C0']['y'] == 1 and y[s]['VISION_CENTER_V2']['y'] == 0)
        resc = sum(1 for s in y if y[s]['C0']['y'] == 0 and y[s]['VISION_CENTER_V2']['y'] == 1); acts = sum(1 for s in y if y[s]['VISION_CENTER_V2'].get('acted'))
        trig = sum(1 for r in recs if r['triggered'])
        res[pid] = {'scenes': len(recs), 'triggered': trig, 'v2_acted': acts, 'c0_success': n_plus, 'v2_success': sum(1 for s in y if y[s]['VISION_CENTER_V2']['y'] == 1),
                    'n_plus': n_plus, 'k_harm': k, 'rescue': resc, 'p_value': float(binom.cdf(k, n_plus, EPS))}
    ps = sorted(res.items(), key=lambda kv: kv[1]['p_value']); m = len(ps); stop = False
    for j, (pid, e) in enumerate(ps):
        e['holm_level'] = DELTA / (m - j); e['certified'] = (not stop) and e['p_value'] <= e['holm_level'] and e['n_plus'] > 0
        if not e['certified']: stop = True
    srv = json.loads(Path(a.server).read_text(encoding='utf-8'))['certificate']['members']
    for mm in srv:
        e = res[mm['policy']]; same = (mm['n_plus'], mm['k_harm'], mm['certified']) == (e['n_plus'], e['k_harm'], e['certified']) and abs(mm['p_value'] - e['p_value']) < 1e-12
        e['equals_server'] = same; ok &= same
    out = {'recount': res, 'joint_statement_all_policies': all(e['certified'] for e in res.values()), 'all_equal_server': ok}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding='utf-8'); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
