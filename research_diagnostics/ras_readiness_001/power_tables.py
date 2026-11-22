"""Calibration power for the V2 harm certificate (exact binomial, Holm). Expected n_plus from the C1 development C0
success rates (171/200, 164/200) scaled to 400 calibration scenes; family size 1 (one policy) or 2 (both policies).

usage: python -P research_diagnostics/ras_readiness_001/power_tables.py --out <json>
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from hci import v2_confirm as V
from hci.statistics import zero_event_requirement

RATES = (0.0, 0.003, 0.006, 0.01, 0.02, 0.03, 0.04)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', required=True); a = a.parse_args(); out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    rows = []
    for npl, lab in ((342, 'train_seed_1: C1 C0 rate 171/200 x 400'), (328, 'train_seed_2: 164/200 x 400'), (300, 'conservative'), (250, 'low')):
        for fam in (1, 2):
            t = V.power_table(npl, fam, true_rates=RATES); t['label'] = lab; rows.append(t)
    res = {'epsilon': V.EPSILON, 'delta': V.DELTA, 'rows': rows, 'zero_event_n_plus': {'family_1': zero_event_requirement(.05, .05, 1), 'family_2': zero_event_requirement(.05, .05, 2)},
           'c1_development_reference': {'train_seed_1': {'harm': 1, 'n_plus': 171}, 'train_seed_2': {'harm': 0, 'n_plus': 164}, 'note': 'development data; not a calibration'}}
    out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    for t in rows: print(t['label'], t['n_plus'], 'family', t['family_size'], 'k_max', t['largest_k_that_passes'], {k: round(v, 3) for k, v in t['pass_probability'].items()})
    print('zero-event n_plus', res['zero_event_n_plus'])


if __name__ == '__main__':
    main()
