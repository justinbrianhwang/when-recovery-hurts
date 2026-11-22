"""Shape-index check for the residual of the S6 sub-step analysis: every residual row involves a finger (4 collision boxes
per finger; every other body has 1 shape). For each finger pair of the LAST sub-step, the PhysX separation is compared with
the reconstructed distance at the PRE pose to EACH of the four boxes of that finger. If another index matches, the
contact's shape index and the logged geometry order disagree.

usage: python -P research_diagnostics/ras_exec_001/finger_shape_index_check.py --in <s6_substep_001_try2> --out <json>
"""
import argparse, collections, glob, gzip, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(Path(__file__).resolve().parent))
from hci import spatial_links as L  # noqa: E402
import s6_substep_analysis as A  # noqa: E402


def main():
    a = argparse.ArgumentParser(); a.add_argument('--in', dest='inp', required=True); a.add_argument('--out', required=True); a = a.parse_args()
    tally = collections.Counter(); per_scene = collections.Counter(); examples = []; n = 0
    for f in sorted(glob.glob(str(Path(a.inp) / 'substeps_*.json.gz'))):
        d = json.load(gzip.open(f, 'rt', encoding='utf-8')); pol, sc = d['policy'], d['scene']
        tr = json.load(gzip.open(Path(f).with_name(f'trace_on_{pol}_{sc}.json.gz'), 'rt', encoding='utf-8')); geo = tr['record']['initial_state']['collision_geometry']
        for seg in d['segments']:
            last = {}
            for x in seg['substeps']:
                if x['k'] >= last.get(x['control_step'], {'k': -1})['k']: last[x['control_step']] = x
            for x in last.values():
                W = None
                for c in x['contacts']:
                    if not c['points']: continue
                    fa = 'finger' in c['a']; fb = 'finger' in c['b']
                    if not (fa or fb) or (fa and fb): continue
                    sep = min(p[9] for p in c['points'])
                    if sep <= 0: continue
                    W = W or L.world(A.row_of(x['pre'], geo), geo)
                    fn, fi, on, oi = (c['a'], c['shape_a'], c['b'], c['shape_b']) if fa else (c['b'], c['shape_b'], c['a'], c['shape_a'])
                    O = [s for s in W.get(on, []) if s.name == f'{on}#{oi}']
                    if not O: continue
                    d_all = {j: L.gap([s], O) for j, s in enumerate(W[fn]) if s.name == f'{fn}#{j}'}
                    err = {j: abs(sep - v) for j, v in d_all.items()}; best = min(err, key=err.get); n += 1
                    own_ok = err.get(fi, 1) <= 1e-3
                    key = ('own_index_within_1mm' if own_ok else ('other_index_matches' if err[best] <= 1e-3 else 'no_index_matches'))
                    tally[key] += 1
                    if key != 'own_index_within_1mm':
                        per_scene[(pol, sc)] += 1
                        if len(examples) < 30: examples.append({'policy': pol, 'scene': sc, 'step': x['control_step'], 'finger': fn, 'logged_index': fi, 'best_index': best, 'other': f'{on}#{oi}',
                                                                'physx_sep_mm': sep * 1e3, 'recon_by_index_mm': {j: v * 1e3 for j, v in d_all.items()}})
                        tally[f'map {fi}->{best}' if err[best] <= 1e-3 else f'unmatched from {fi}'] += 1
    res = {'finger_pairs_compared': n, 'tally': dict(tally), 'per_scene_not_own_index': {f'{k[0]}:{k[1]}': v for k, v in per_scene.items()}, 'examples': examples}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps({k: res[k] for k in ('finger_pairs_compared', 'tally')}, indent=1)); print(json.dumps(examples[:3], indent=1))


if __name__ == '__main__':
    main()
