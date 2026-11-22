"""Aggregate the per-scene results of run_all.py into the tables, figures and summary of the S4/S5 hull sensitivity.

usage: python -P research_diagnostics/c1_hull_sensitivity_001/aggregate.py --work runs/c1_hull_sensitivity_001/work --out results/vision_center_v2_c1_001/posthoc_hull_sensitivity_001
"""
from __future__ import annotations

import argparse, collections, glob, gzip, json, math, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hull_s45 as H  # noqa: E402
C = H.C; L = H.L
S4, S5 = H.S4_KEY, H.S5_KEY
CASES = C.DRIVE_RUN / 'figures' / 'case_figures_001'
MM = 1e3


def load(work):
    done = json.loads((work / 'RUN_DONE.json').read_text(encoding='utf-8'))
    errs = sorted(p.name for p in work.glob('*.error.json'))
    res = []
    for p in sorted(work.glob('*.json.gz')):
        with gzip.open(p, 'rt', encoding='utf-8') as f: res.append(json.load(f))
    return done, errs, res


def pair_kind(a, b):
    k = L.category(a, b) or 'other'
    return k


def main():
    p = argparse.ArgumentParser(); p.add_argument('--work', required=True); p.add_argument('--out', required=True); a = p.parse_args()
    work, out = Path(a.work), Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists: never overwritten')
    done, errs, res = load(work)
    expected = {(pol, int(f.stem)) for pol in C.POLICIES for f in (C.pass_dir(pol) / 'scenes').glob('*.json') if f.stem.isdigit()}
    got = {(r['policy'], r['scene']) for r in res}
    complete = bool(done['complete'] and not errs and got == expected)
    out.mkdir(parents=True); (out / 'figures').mkdir()

    # ------------------------------------------------ geometry: one mesh table, identical across records?
    geo_hashes = collections.Counter(r['geometry_sha256'] for r in res)
    mesh_sets = {json.dumps(r['mesh_stats'], sort_keys=True) for r in res}
    mesh = res[0]['mesh_stats']
    C.write_csv(out / 'MESH_SHAPE_STATS.csv', mesh)

    # ------------------------------------------------ branches
    br = []
    for r in res:
        for b in r['branches']:
            lo, hu = b['items']['LOGGED'], b['items']['HULL']
            row = {'policy': b['policy'], 'scene': b['scene'], 'arm': b['arm'], 'acted': b['acted'], 'control_steps': b['control_steps'], 'trigger': b['trigger'], 'rows': b['rows'], 'y': b['y']}
            for tag, it in (('logged', lo), ('hull', hu)):
                row[f'S4_status_{tag}'] = it[S4]['status']; row[f'S4_min_distance_mm_{tag}'] = it[S4]['min_distance_m'] * MM if 'min_distance_m' in it[S4] else None
                row[f'S5_status_{tag}'] = it[S5]['status']; row[f'S5_n_pairs_{tag}'] = it[S5].get('n_pairs'); row[f'S5_n_violations_{tag}'] = it[S5].get('n_violations')
                row[f'S5_n_not_reconstructed_{tag}'] = it[S5].get('n_not_reconstructed'); row[f'S5_max_penetration_mm_{tag}'] = it[S5].get('max_penetration_m', 0.) * MM
                row[f'S5_n_overlapping_{tag}'] = it[S5].get('n_overlapping')
            row['S4_change_mm'] = None if row['S4_min_distance_mm_logged'] is None else row['S4_min_distance_mm_hull'] - row['S4_min_distance_mm_logged']
            row['S4_status_changed'] = row['S4_status_logged'] != row['S4_status_hull']; row['S5_status_changed'] = row['S5_status_logged'] != row['S5_status_hull']
            row['S5_items_identical'] = json.dumps(lo[S5], sort_keys=True) == json.dumps(hu[S5], sort_keys=True)
            br.append(row)
    C.write_csv(out / 'S45_BRANCH_RESULTS.csv', br)

    def xtab(item):
        t = collections.Counter((x[f'{item}_status_logged'], x[f'{item}_status_hull']) for x in br)
        return {f'{a_} -> {b_}': n for (a_, b_), n in sorted(t.items())}
    status = {'branches': len(br), 'S4_logged_to_hull': xtab('S4'), 'S5_logged_to_hull': xtab('S5'),
              'S4_status_changes': [x for x in br if x['S4_status_changed']], 'S5_status_changes': [x for x in br if x['S5_status_changed']],
              'S5_items_identical_in_every_branch': all(x['S5_items_identical'] for x in br)}
    by_arm = {}
    for x in br:
        e = by_arm.setdefault(x['arm'], collections.Counter())
        for item in ('S4', 'S5'):
            for tag in ('logged', 'hull'): e[f'{item}_{tag}_{x[f"{item}_status_{tag}"]}'] += 1
    status['by_arm'] = {k: dict(v) for k, v in by_arm.items()}

    # ------------------------------------------------ S4: margins and per-row change
    acted = [x for x in br if x['S4_min_distance_mm_logged'] is not None]
    ch = np.array([x['S4_change_mm'] for x in acted]); lo_ = np.array([x['S4_min_distance_mm_logged'] for x in acted]); hu_ = np.array([x['S4_min_distance_mm_hull'] for x in acted])
    mv = [m for r in res for m in r['move_rows']]
    C.write_csv(out / 'S4_MOVE_ROWS.csv', [{**m, **{f'd_{g}_cubeB_change_mm': None if m[f'd_{g}_cubeB_logged_m'] is None else (m[f'd_{g}_cubeB_hull_m'] - m[f'd_{g}_cubeB_logged_m']) * MM for g in (*L.GROUPS, 'robot')}} for m in mv])
    grp = {}
    for g in (*L.GROUPS, 'robot'):
        d = np.array([(m[f'd_{g}_cubeB_hull_m'] - m[f'd_{g}_cubeB_logged_m']) * MM for m in mv if m[f'd_{g}_cubeB_logged_m'] is not None])
        grp[g] = {'rows': int(len(d)), 'changed_rows': int((np.abs(d) > 1e-9).sum()), 'min_change_mm': float(d.min()), 'max_change_mm': float(d.max()), 'max_abs_change_mm': float(np.abs(d).max()),
                  'median_abs_change_of_changed_rows_mm': float(np.median(np.abs(d[np.abs(d) > 1e-9]))) if (np.abs(d) > 1e-9).any() else 0.,
                  'min_distance_logged_mm': float(min(m[f'd_{g}_cubeB_logged_m'] for m in mv if m[f'd_{g}_cubeB_logged_m'] is not None) * MM),
                  'min_distance_hull_mm': float(min(m[f'd_{g}_cubeB_hull_m'] for m in mv if m[f'd_{g}_cubeB_hull_m'] is not None) * MM)}
    nearest = collections.Counter((m['nearest_group_logged'], m['nearest_group_hull']) for m in mv)
    closest = sorted(acted, key=lambda x: x['S4_min_distance_mm_hull'])[:15]
    s4 = {'acted_branches': len(acted), 'move_rows': len(mv), 'threshold_mm': 0.0,
          'min_distance_logged_mm': {'min': float(lo_.min()), 'p05': float(np.percentile(lo_, 5)), 'median': float(np.median(lo_)), 'max': float(lo_.max())},
          'min_distance_hull_mm': {'min': float(hu_.min()), 'p05': float(np.percentile(hu_, 5)), 'median': float(np.median(hu_)), 'max': float(hu_.max())},
          'branch_change_mm': {'min': float(ch.min()), 'max': float(ch.max()), 'n_changed': int((np.abs(ch) > 1e-9).sum()), 'n_decreased': int((ch < -1e-9).sum()), 'n_increased': int((ch > 1e-9).sum())},
          'smallest_margin_hull_mm': float(hu_.min()), 'margin_over_largest_branch_change': float(hu_.min() / max(1e-12, np.abs(ch).max())),
          'per_group_move_rows': grp, 'nearest_group_logged_to_hull': {f'{a_} -> {b_}': n for (a_, b_), n in sorted(nearest.items())},
          'closest_15_branches_hull': [{k: x[k] for k in ('policy', 'scene', 'arm', 'control_steps', 'S4_min_distance_mm_logged', 'S4_min_distance_mm_hull', 'S4_change_mm')} for x in closest]}

    # ------------------------------------------------ S5: observed pairs
    pr = [q for r in res for q in r['pair_rows']]
    kinds = {}
    for q in pr:
        k = pair_kind(q['a'], q['b']); e = kinds.setdefault(k, {'rows': 0, 'mesh_rows': 0, 'not_reconstructed': 0, 'lg': [], 'hu': []}); e['rows'] += 1; e['mesh_rows'] += q['involves_mesh']
        if q['reconstructed_logged_m'] is None: e['not_reconstructed'] += 1; continue
        e['lg'].append(q['reconstructed_logged_m']); e['hu'].append(q['reconstructed_hull_m'])
    kt = []
    for k, e in sorted(kinds.items()):
        lg, hu = np.array(e['lg']) * MM, np.array(e['hu']) * MM; d = hu - lg
        kt.append({'pair_kind': k, 'observed_pair_rows': e['rows'], 'rows_involving_mesh': e['mesh_rows'], 'not_reconstructed': e['not_reconstructed'],
                   'max_reconstructed_logged_mm': float(lg.max()) if len(lg) else None, 'max_reconstructed_hull_mm': float(hu.max()) if len(hu) else None,
                   'rows_above_1mm_logged': int((lg > 1).sum()), 'rows_above_1mm_hull': int((hu > 1).sum()),
                   'max_penetration_logged_mm': float(max(0., -lg.min())) if len(lg) else None, 'max_penetration_hull_mm': float(max(0., -hu.min())) if len(hu) else None,
                   'rows_changed': int((np.abs(d) > 1e-9).sum()), 'max_abs_change_mm': float(np.abs(d).max()) if len(d) else None})
    C.write_csv(out / 'S5_OBSERVED_PAIRS_BY_KIND.csv', kt)
    meshp = [q for q in pr if q['involves_mesh']]
    C.write_csv(out / 'S5_OBSERVED_MESH_PAIR_ROWS.csv', [{**q, 'change_mm': None if q['reconstructed_logged_m'] is None else (q['reconstructed_hull_m'] - q['reconstructed_logged_m']) * MM} for q in meshp],
                fields=['policy', 'scene', 'trace', 't', 'a', 'shape_a', 'b', 'shape_b', 'involves_mesh', 'physx_min_separation_m', 'reconstructed_logged_m', 'reconstructed_hull_m', 'change_mm'])
    allr = np.array([q['reconstructed_logged_m'] for q in pr if q['reconstructed_logged_m'] is not None]) * MM
    allh = np.array([q['reconstructed_hull_m'] for q in pr if q['reconstructed_hull_m'] is not None]) * MM
    d_all = allh - allr
    s5 = {'observed_pair_rows_unique': len(pr), 'involving_mesh': len(meshp), 'not_reconstructed': sum(q['reconstructed_logged_m'] is None for q in pr),
          'max_reconstructed_logged_mm': float(allr.max()), 'max_reconstructed_hull_mm': float(allh.max()), 'threshold_mm': L.CONTACT_GEOM_TOL_M * MM,
          'margin_logged_mm': float(L.CONTACT_GEOM_TOL_M * MM - allr.max()), 'margin_hull_mm': float(L.CONTACT_GEOM_TOL_M * MM - allh.max()),
          'rows_changed': int((np.abs(d_all) > 1e-9).sum()), 'max_abs_change_mm': float(np.abs(d_all).max()),
          'max_penetration_logged_mm': float(max(0., -allr.min())), 'max_penetration_hull_mm': float(max(0., -allh.min())), 'by_kind': kt,
          'note': 'unique (trace, step, shape pair) rows; a row before the trigger is shared by every arm branch and counted once here (branch counts in S45_BRANCH_RESULTS.csv count it per branch, as spatial_items does)'}

    # ------------------------------------------------ the 14 recorded case figures (28 arm entries): reproduction
    repro = []
    lookup = {(x['policy'], x['scene'], x['arm']): x for x in br}
    resmap = {(r['policy'], r['scene']): r for r in res}
    for pj in sorted(glob.glob(str(CASES / '*' / '*' / 'plotted.json'))):
        d = json.loads(Path(pj).read_text(encoding='utf-8')); pol = Path(pj).parts[-3]; case = Path(pj).parts[-2]; sc = int(case.split('_')[0])
        for arm, v in d['arms'].items():
            si = v['summary']['spatial_items']; b = next(bb for bb in resmap[(pol, sc)]['branches'] if bb['arm'] == arm)
            same_logged = {k: json.dumps(si[k], sort_keys=True, default=float) == json.dumps(b['items']['LOGGED'][k], sort_keys=True, default=float) for k in (S4, S5)}
            repro.append({'policy': pol, 'case': case, 'arm': arm, 'S4_recorded': si[S4]['status'], 'S4_recorded_min_mm': si[S4].get('min_distance_m', math.nan) * MM if 'min_distance_m' in si[S4] else None,
                          'S4_logged_identical_to_recorded': same_logged[S4], 'S4_hull_status': b['items']['HULL'][S4]['status'],
                          'S4_hull_min_mm': b['items']['HULL'][S4]['min_distance_m'] * MM if 'min_distance_m' in b['items']['HULL'][S4] else None,
                          'S5_recorded': si[S5]['status'], 'S5_recorded_n_pairs': si[S5].get('n_pairs'), 'S5_logged_identical_to_recorded': same_logged[S5],
                          'S5_hull_status': b['items']['HULL'][S5]['status'], 'S5_hull_identical_to_recorded': json.dumps(si[S5], sort_keys=True, default=float) == json.dumps(b['items']['HULL'][S5], sort_keys=True, default=float),
                          'plotted_json': C.rel(pj), 'plotted_json_sha256': C.sha256(pj)})
    C.write_csv(out / 'RECORDED_CASES_REPRODUCTION.csv', repro)

    # ------------------------------------------------ figures
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(lo_, hu_, s=8, c='#2a6fb0'); m_ = max(lo_.max(), hu_.max()) * 1.05; ax[0].plot([0, m_], [0, m_], color='#888', lw=.8)
    ax[0].axvline(0, color='#b03030', lw=1); ax[0].axhline(0, color='#b03030', lw=1); ax[0].set_xlabel('S4 min distance, logged mesh [mm]'); ax[0].set_ylabel('S4 min distance, convex hull [mm]')
    ax[0].set_title(f'S4 per acted branch (n={len(acted)}); red = threshold 0', fontsize=9)
    ax[1].hist(ch, bins=40, color='#2a6fb0'); ax[1].set_xlabel('hull - logged, S4 min distance [mm]'); ax[1].set_ylabel('branches'); ax[1].set_title(f'S4 change per branch (min {ch.min():.3f}, max {ch.max():.3f} mm)', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'figures' / 's4_logged_vs_hull.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(allr, bins=80, color='#555', alpha=.7, label='logged'); ax[0].hist(allh, bins=80, histtype='step', color='#c0392b', label='hull'); ax[0].axvline(1, color='#b03030', ls='--', lw=1)
    ax[0].set_yscale('log'); ax[0].set_xlabel('reconstructed distance of observed contact pairs [mm]'); ax[0].set_ylabel('pair rows'); ax[0].legend(fontsize=8); ax[0].set_title('S5: all observed pairs; dashed = 1 mm rule', fontsize=9)
    md = np.array([r_['change_mm'] for r_ in [{**q, 'change_mm': None if q['reconstructed_logged_m'] is None else (q['reconstructed_hull_m'] - q['reconstructed_logged_m']) * MM} for q in meshp] if r_['change_mm'] is not None])
    if len(md): ax[1].hist(md, bins=40, color='#c0392b')
    ax[1].set_xlabel('hull - logged [mm], observed pairs with a mesh shape'); ax[1].set_ylabel('pair rows'); ax[1].set_title(f'S5 mesh pairs (n={len(meshp)})', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'figures' / 's5_observed_pairs.png', dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6)); xs = np.arange(len(mesh))
    ax.bar(xs - .2, [m['max_logged_surface_depth_inside_hull_m'] * MM for m in mesh], .4, label='max depth of logged surface inside hull', color='#2a6fb0')
    ax.bar(xs + .2, [(1 - m['volume_ratio_logged_to_hull']) * 100 for m in mesh], .4, label='volume missing vs hull [%]', color='#e08a2c')
    ax.set_xticks(xs); ax.set_xticklabels([m['body'].replace('panda_', '') for m in mesh], fontsize=8); ax.legend(fontsize=8); ax.set_title('Logged PhysX meshes vs the convex hull of their vertices', fontsize=9)
    ax.set_ylabel('mm  |  %'); fig.tight_layout(); fig.savefig(out / 'figures' / 'mesh_shapes_vs_hull.png', dpi=130); plt.close(fig)

    summ = {'label': C.LABEL, 'what': 'S4/S5 recomputed with the logged PhysX meshes replaced by the convex hull of the same vertices; everything else identical',
            'complete': complete, 'run': done, 'error_files': errs, 'scene_runs': len(res), 'expected_scene_runs': len(expected), 'missing': sorted(map(list, expected - got)),
            'geometry_variants_in_records': dict(geo_hashes), 'mesh_shapes_identical_in_every_record': len(mesh_sets) == 1,
            'status': status, 'S4': s4, 'S5': s5,
            'recorded_cases': {'arm_entries': len(repro), 'logged_identical_to_recorded_S4': sum(x['S4_logged_identical_to_recorded'] for x in repro), 'logged_identical_to_recorded_S5': sum(x['S5_logged_identical_to_recorded'] for x in repro),
                               'hull_status_equal_to_recorded_S4': sum(x['S4_hull_status'] == x['S4_recorded'] for x in repro), 'hull_status_equal_to_recorded_S5': sum(x['S5_hull_status'] == x['S5_recorded'] for x in repro)}}
    C.jdump(out / 'S45_HULL_SENSITIVITY_SUMMARY.json', summ)
    print(json.dumps({'complete': complete, 'scene_runs': len(res), 'branches': len(br), 'S4': status['S4_logged_to_hull'], 'S5': status['S5_logged_to_hull'],
                      'S4_branch_change_mm': s4['branch_change_mm'], 'S4_min_hull_mm': s4['smallest_margin_hull_mm'], 'S5_rows_changed': s5['rows_changed'], 'S5_max_change_mm': s5['max_abs_change_mm'],
                      'S5_max_recon_mm': [s5['max_reconstructed_logged_mm'], s5['max_reconstructed_hull_mm']], 'recorded_cases': summ['recorded_cases']}, indent=1))
    if not complete: raise SystemExit(3)


if __name__ == '__main__':
    main()
