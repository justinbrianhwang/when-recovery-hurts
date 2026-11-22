"""Runner faithfulness on the SAME host: the calibration runner's smoke output (arms C0, V2) against the unchanged C1
collector run on the same host by scripts/42 (all nine C1 arms, logger OFF == ON) for the scene-runs both cover.
Separates a runner defect from a host difference: identical here + different from the original C1 = host effect.

usage: python -P research_diagnostics/ras_exec_001/runner_vs_samehost.py --smoke <smoke folder> --b4 <runs/s6_substep_001> --out <json>
"""
import argparse, gzip, json
from pathlib import Path


def canon(x): return json.dumps(x, sort_keys=True)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--smoke', required=True); a.add_argument('--b4', required=True); a.add_argument('--out', required=True); a = a.parse_args()
    rows = []; ok = True
    for f in sorted(Path(a.b4).glob('trace_on_*.json.gz')):
        parts = f.name.replace('.json.gz', '').split('_'); pid = '_'.join(parts[2:5]); s = int(parts[5])
        sp = Path(a.smoke) / f'pass_{pid}' / 'scenes' / f'{s}.json'
        if not sp.exists(): continue
        b4 = json.load(gzip.open(f, 'rt', encoding='utf-8')); r1 = json.loads(sp.read_text(encoding='utf-8'))
        with gzip.open(sp.parent.parent / r1['trace'], 'rt', encoding='utf-8') as g: t1 = json.load(g)['traces']
        t0 = json.loads(json.dumps(b4['traces'], default=float)); c0 = {c['arm']: c for c in b4['record']['cells']}; c1 = {c['arm']: c for c in r1['cells']}
        e = {'policy': pid, 'scene': s}
        for k in ('trunk', 'C0', 'VISION_CENTER_V2'): e[f'trace_{k}_identical'] = canon(t0.get(k)) == canon(t1.get(k))
        for arm in ('C0', 'VISION_CENTER_V2'): e[f'{arm}_y_steps_equal'] = (c0[arm]['y'], c0[arm]['steps']) == (c1[arm]['y'], c1[arm]['steps'])
        e['identical'] = all(v for k, v in e.items() if k.endswith(('_identical', '_equal'))); ok &= e['identical']; rows.append(e); print(json.dumps(e))
    Path(a.out).write_text(json.dumps({'all_identical': bool(ok) and bool(rows), 'scene_runs': rows}, indent=1), encoding='utf-8'); print('ALL_IDENTICAL' if ok and rows else 'DIFFERENCES_OR_NONE')


if __name__ == '__main__':
    main()
