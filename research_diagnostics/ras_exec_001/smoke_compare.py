"""Real-simulator smoke check of the calibration runner (scripts/40 run on USED C1 development scenes 51103, 51157, 51253
with both policies) against the ORIGINAL C1 records: for every scene-run, the trunk, C0 and VISION_CENTER_V2 trace rows
(JSON canonical, exact) and the C0 / V2 cells (outcome, steps, acted, control steps, decision, control readings).
A difference is reported; the calibration must not start unless every comparison is identical.

usage: python -P research_diagnostics/ras_exec_001/smoke_compare.py --smoke <runs/ras_v2_confirm_001/smoke> --c1 <C1 extracted run folder> --out <json>
"""
import argparse, gzip, json
from pathlib import Path

KEYS = ('y', 'steps', 'reason', 'acted', 'control_steps', 'intervention', 'intervention_step')


def canon(x): return json.dumps(x, sort_keys=True)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--smoke', required=True); a.add_argument('--c1', required=True); a.add_argument('--out', required=True); a = a.parse_args()
    out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    rows = []; ok = True
    for pid in ('train_seed_1', 'train_seed_2'):
        for s in (51103, 51157, 51253):
            sd = Path(a.smoke) / f'pass_{pid}'; cd = Path(a.c1) / f'pass_{pid}'
            r1 = json.loads((sd / 'scenes' / f'{s}.json').read_text(encoding='utf-8')); r0 = json.loads((cd / 'scenes' / f'{s}.json').read_text(encoding='utf-8'))
            with gzip.open(sd / r1['trace'], 'rt', encoding='utf-8') as f: t1 = json.load(f)['traces']
            with gzip.open(cd / r0['trace'], 'rt', encoding='utf-8') as f: t0 = json.load(f)['traces']
            e = {'policy': pid, 'scene': s, 'trigger_equal': r0.get('trigger', {}) and r1.get('trigger', {}) and r0['trigger']['step'] == r1['trigger']['step']}
            for k in ('trunk', 'C0', 'VISION_CENTER_V2'):
                e[f'trace_{k}_identical'] = canon(t0.get(k)) == canon(t1.get(k)); e[f'trace_{k}_rows'] = [len(t0.get(k) or []), len(t1.get(k) or [])]
            c0 = {c['arm']: c for c in r0['cells']}; c1 = {c['arm']: c for c in r1['cells']}
            for arm in ('C0', 'VISION_CENTER_V2'):
                e[f'{arm}_cell_equal'] = all(c0[arm].get(k) == c1[arm].get(k) for k in KEYS) and canon((c0[arm].get('control') or {}).get('readings')) == canon((c1[arm].get('control') or {}).get('readings'))
                e[f'{arm}_y'] = [c0[arm]['y'], c1[arm]['y']]
            e['identical'] = bool(e['trigger_equal']) and all(v for k, v in e.items() if k.endswith(('_identical', '_equal')))
            ok &= e['identical']; rows.append(e); print(json.dumps(e))
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps({'all_identical': bool(ok), 'scene_runs': rows}, indent=1), encoding='utf-8')
    print('ALL_IDENTICAL' if ok else 'DIFFERENCES')


if __name__ == '__main__':
    main()
