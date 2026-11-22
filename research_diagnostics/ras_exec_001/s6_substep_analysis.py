"""S6 sub-step analysis (POST_HOC_VALIDATION; the official S6 FAIL is unchanged).

Input: scripts/42 output (substeps_<policy>_<scene>.json.gz, trace_on_*.json.gz with the record and its collision
geometry). For the LAST physics sub-step of every logged control step (the contact list C1 recorded), every listed
shape pair with PhysX separation > 0 (the S6 population) is compared with the reconstructed signed distance of the same
two logged shapes (hci.spatial_links, unchanged geometry) at
  POST  the pose after the sub-step (= the pose C1 logged), and
  PRE   the pose before the sub-step (newly observed here).
Also for every sub-step k of a fixed subset (the trunk of each scene-run): PRE/POST of that sub-step.
Reported by pair kind and relative-speed class: share within 1 mm, median |difference|, and the rows that stay above 1 mm
at PRE (unexplained by timing).

usage: python -P research_diagnostics/ras_exec_001/s6_substep_analysis.py --in <runs/s6_substep_001> --out <folder> [--workers 4]
"""
from __future__ import annotations

import argparse, collections, csv, glob, gzip, json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from hci import spatial_links as L  # noqa: E402

TOL = 0.001
SPEED_CLASSES = ((0, 0.001, 'static <1 mm/s'), (0.001, 0.01, 'slow 1-10 mm/s'), (0.01, 0.1, 'moving 10-100 mm/s'), (0.1, 1e9, 'fast >100 mm/s'))


def row_of(st, geo):
    return {'link_pose': st['link_pose'], 'cubeA': st['cubeA'][:3], 'cubeA_q': st['cubeA'][3:7], 'cubeB': st['cubeB'][:3], 'cubeB_q': st['cubeB'][3:7]}


def body_pos(st, n):
    if n in ('cubeA', 'cubeB'): return np.asarray(st[n][:3])
    if n in st['link_pose']: return np.asarray(st['link_pose'][n][:3])
    return np.zeros(3)


def one(path):
    d = json.load(gzip.open(path, 'rt', encoding='utf-8')); pol, sc = d['policy'], d['scene']
    tr = json.load(gzip.open(Path(path).with_name(f'trace_on_{pol}_{sc}.json.gz'), 'rt', encoding='utf-8'))
    geo = tr['record']['initial_state']['collision_geometry']; rows = []
    for si, seg in enumerate(d['segments']):
        last = {}
        for x in seg['substeps']:
            if x['k'] >= last.get(x['control_step'], {'k': -1})['k']: last[x['control_step']] = x
        targets = [(x, 'last') for x in last.values()] + ([(x, 'every_k_trunk') for x in seg['substeps'] if x['k'] < 4] if si == 0 else [])
        for x, which in targets:
            Wpre = L.world(row_of(x['pre'], geo), geo); Wpost = L.world(row_of(x['post'], geo), geo)
            for c in x['contacts']:
                if not c['points']: continue
                sep = min(p[9] for p in c['points'])
                if sep <= 0: continue
                na, ia, nb, ib = c['a'], c['shape_a'], c['b'], c['shape_b']
                A0 = [s for s in Wpre.get(na, []) if s.name == f'{na}#{ia}']; B0 = [s for s in Wpre.get(nb, []) if s.name == f'{nb}#{ib}']
                A1 = [s for s in Wpost.get(na, []) if s.name == f'{na}#{ia}']; B1 = [s for s in Wpost.get(nb, []) if s.name == f'{nb}#{ib}']
                if not (A0 and B0 and A1 and B1): continue
                rp = L.gap(A0, B0); rq = L.gap(A1, B1)
                va = (body_pos(x['post'], na) - body_pos(x['pre'], na)) / 0.01; vb = (body_pos(x['post'], nb) - body_pos(x['pre'], nb)) / 0.01
                imp = any(any(v != 0 for v in p[6:9]) for p in c['points'])
                rows.append({'policy': pol, 'scene': sc, 'segment': si, 'segment_kind': seg['kind'], 'control_step': x['control_step'], 'k': x['k'], 'which': which, 'pair': f'{na}#{ia}|{nb}#{ib}',
                             'kind': L.category(na, nb) or 'other', 'physx_sep_m': sep, 'recon_pre_m': rp, 'recon_post_m': rq, 'rel_speed_m_s': float(np.linalg.norm(va - vb)), 'impulse': imp})
    return rows


def cls(v):
    return next(n for lo, hi, n in SPEED_CLASSES if lo <= v < hi)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--in', dest='inp', required=True); a.add_argument('--out', required=True); a.add_argument('--workers', type=int, default=4); a = a.parse_args()
    out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    files = sorted(glob.glob(str(Path(a.inp) / 'substeps_*.json.gz')))
    with ProcessPoolExecutor(a.workers) as ex: rows = [r for part in ex.map(one, files) for r in part]
    out.mkdir(parents=True)
    with gzip.open(out / 'substep_pair_rows.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    summ = {'label': 'POST_HOC_VALIDATION', 'scene_run_files': len(files), 'rows': len(rows), 'groups': {}}
    for which in ('last', 'every_k_trunk'):
        R = [r for r in rows if r['which'] == which]; g = collections.defaultdict(list)
        for r in R: g[(r['kind'], cls(r['rel_speed_m_s']))].append(r); g[('ALL', cls(r['rel_speed_m_s']))].append(r); g[('ALL', 'ALL')].append(r)
        out_g = {}
        for (k, s), rr in sorted(g.items()):
            dpre = np.array([abs(r['physx_sep_m'] - r['recon_pre_m']) for r in rr]); dpost = np.array([abs(r['physx_sep_m'] - r['recon_post_m']) for r in rr])
            out_g[f'{k} | {s}'] = {'n': len(rr), 'over_1mm_post': int((dpost > TOL).sum()), 'over_1mm_pre': int((dpre > TOL).sum()), 'median_abs_post_mm': float(np.median(dpost) * 1e3), 'median_abs_pre_mm': float(np.median(dpre) * 1e3),
                                   'p99_abs_pre_mm': float(np.percentile(dpre, 99) * 1e3), 'max_abs_pre_mm': float(dpre.max() * 1e3)}
        summ['groups'][which] = out_g
    resid = sorted([r for r in rows if r['which'] == 'last' and abs(r['physx_sep_m'] - r['recon_pre_m']) > TOL], key=lambda r: -abs(r['physx_sep_m'] - r['recon_pre_m']))
    summ['residual_rows_over_1mm_at_pre'] = len(resid); summ['residual_examples'] = resid[:40]
    (out / 'S6_SUBSTEP_SUMMARY.json').write_text(json.dumps(summ, indent=1), encoding='utf-8')
    a_ = summ['groups']['last']['ALL | ALL']; print(json.dumps({'rows': len(rows), 'last_substep_all': a_, 'residual_over_1mm_at_pre': len(resid)}, indent=1))


if __name__ == '__main__':
    main()
