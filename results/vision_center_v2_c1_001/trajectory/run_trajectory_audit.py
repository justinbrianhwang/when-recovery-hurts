"""Phase C1 preparation: trajectory and behaviour audit of the stored Vision-CENTER v2 Phase B records (500 used scenes,
train seed 1). No simulator, no policy, no new scene; the records are read only.

    python results/vision_center_v2_c1_001/trajectory/run_trajectory_audit.py <phase B root> <evidence out dir> [processes]

<phase B root>: the folder holding shadow/ (stage B1, 50600-50899: C0, v1, v2) and pilot/ (stage B2, 50900-51099: C0, v1,
v2, oracle). Writes, refusing to overwrite:
  results/vision_center_v2_c1_001/trajectory/audit_summary.json   counts per arm and check, sync, case list, provenance
  results/vision_center_v2_c1_001/trajectory/per_scene_checks.csv one row per scene, arm and check
  <evidence out dir>/<case>/...                                    figures and GIFs for the selected cases, MANIFEST.json
Case selection is fixed in CASE_RULES below before any figure was looked at. Every rule's case list holds all matching
scenes. Figures are made for all scenes of every risk rule; for the two broad lists ('any_check_failed',
'zero_step_intervention_without_need') and the representative rules only the lowest seeds get figures (MANIFEST.json,
figure_selection).
"""
import os
for _k in ('OMP', 'OPENBLAS', 'MKL', 'NUMEXPR', 'BLIS'): os.environ[_k + '_NUM_THREADS'] = '1'
import csv, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, 'src')
from hci import trajectory_audit as T

HERE = Path('results/vision_center_v2_c1_001/trajectory'); STAGES = (('B1', 'shadow'), ('B2', 'pilot'))
NAMED = (51009, 51004, 50959, 50892)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(job):
    stage, folder, root, seed = job; d = Path(root) / folder
    rec, tr, fr = T.load_scene(d, seed)
    m = T.scene_metrics(rec, tr)
    sync = T.sync_check(rec, tr, fr) if rec['triggered'] else None
    ctl = T.control_tcp_sync(rec, tr) if rec['triggered'] else {}
    cf = {a: T.control_frame_check(rec, tr, fr, a) for a in ('VISION_CENTER_V2',) if rec['triggered'] and T.cell(rec, a) and T.cell(rec, a).get('acted')}
    end = {a: (T.cell(rec, a).get('control') or {}).get('end_reason') for a in rec['arms'] if rec['triggered'] and T.cell(rec, a) and T.cell(rec, a).get('acted')}
    mem = {a: {'used_memory': bool((T.cell(rec, a).get('control') or {}).get('used_memory')), 'gt': T.cell(rec, a).get('gt_reading_error')} for a in rec['arms'] if a.startswith('VISION') and rec['triggered']}
    return {'stage': stage, 'folder': folder, 'seed': seed, 'metrics': m, 'sync': sync, 'control_tcp_sync': ctl, 'control_frames': cf, 'end_reason': end, 'memory': mem,
            'files': {k: (str(Path(folder) / rec[k]), rec[k + '_sha256']) for k in ('trace', 'frames') if rec.get(k)} | {'scene': (str(Path(folder) / 'scenes' / f'{seed}.json'), sha(d / 'scenes' / f'{seed}.json'))}}


def chk(r, arm, name): return (((r['metrics']['arms'].get(arm) or {}).get('checks') or {}).get(name) or {})


CASE_RULES = {
    'harm_any_arm': lambda r: any(v['harm'] for v in r['metrics']['arms'].values()),
    'target_pushed_during_move': lambda r: any(chk(r, a, '7_no_slip_no_push').get('target_xy_displacement_m', 0) > T.PUSH_TOL_M for a in r['metrics']['arms']),
    'memory_error_move_v2': lambda r: bool(r['memory'].get('VISION_CENTER_V2', {}).get('used_memory') and (r['memory']['VISION_CENTER_V2']['gt'] or {}).get('offset_error_m') is not None
                                           and (r['memory']['VISION_CENTER_V2']['gt']['offset_error_m'] > T.MEMORY_ERROR_M or r['memory']['VISION_CENTER_V2']['gt'].get('error_within_bound') is False)
                                           and r['metrics']['arms']['VISION_CENTER_V2']['acted']),
    'unnecessary_move': lambda r: any(chk(r, a, '2a_acted_only_when_needed').get('status') == 'FAIL' and chk(r, a, '2a_acted_only_when_needed').get('move_steps', 0) > 0 for a in r['metrics']['arms']),
    'zero_step_intervention_without_need': lambda r: any(chk(r, a, '2a_acted_only_when_needed').get('status') == 'FAIL' and chk(r, a, '2a_acted_only_when_needed').get('move_steps') == 0 for a in r['metrics']['arms']),
    'v2_abstained_oracle_rescued': lambda r: 'CENTER_ORACLE' in r['metrics']['arms'] and r['metrics']['arms']['CENTER_ORACLE']['rescue'] and not r['metrics']['arms']['VISION_CENTER_V2']['acted'],
    'correction_undone_after_resume': lambda r: any(chk(r, a, '8_policy_keeps_correction').get('status') == 'FAIL' for a in r['metrics']['arms']),
    'step_budget_reached': lambda r: any(v == 'step_budget' for v in r['end_reason'].values()),
    'any_check_failed': lambda r: any(x.get('status') == 'FAIL' and k != '2b_acted_when_needed' for v in r['metrics']['arms'].values() for k, x in v['checks'].items()),
}
REPRESENTATIVE = {   # lowest seeds, three per rule: normal and failure behaviour next to the risk cases
    'v2_rescue': lambda r: r['metrics']['arms']['VISION_CENTER_V2']['rescue'],
    'v2_moved_both_succeed': lambda r: r['metrics']['arms']['VISION_CENTER_V2']['acted'] and r['metrics']['arms']['C0']['y'] == 1 and r['metrics']['arms']['VISION_CENTER_V2']['y'] == 1,
    'v2_abstained_both_succeed': lambda r: not r['metrics']['arms']['VISION_CENTER_V2']['acted'] and r['metrics']['arms']['C0']['y'] == 1,
    'v2_abstained_both_fail': lambda r: not r['metrics']['arms']['VISION_CENTER_V2']['acted'] and r['metrics']['arms']['C0']['y'] == 0,
    'v2_moved_both_fail': lambda r: r['metrics']['arms']['VISION_CENTER_V2']['acted'] and r['metrics']['arms']['C0']['y'] == 0 and r['metrics']['arms']['VISION_CENTER_V2']['y'] == 0,
}


def figures(job):
    folder, root, seed, out = job; d = Path(root) / folder; rec, tr, fr = T.load_scene(d, seed); out = Path(out); out.mkdir(parents=True, exist_ok=True); made = []
    for name, fn in (('3d_trajectory.png', T.fig_3d), ('xy_alignment.png', T.fig_xy), ('timeline.png', T.fig_timeline), ('after_trigger_outcome.png', T.outcome_strip)):
        p = out / name
        if p.exists(): raise FileExistsError(p)
        fn(rec, tr, p); made.append(p)
    if fr is not None:
        t0 = rec['trigger']['step']; first = max(0, t0 - 10); k_end = max([0] + [T.cell(rec, a)['control_steps'] for a in rec['arms'] if T.cell(rec, a).get('acted')])
        for name, ov in (('rgb_recorded', False), ('rgb_with_posthoc_ground_truth', True)):
            p = out / f'{name}.gif'
            if p.exists(): raise FileExistsError(p)
            pan = T.rgb_panels(rec, tr, fr, overlay=ov); T.save_gif(pan, p); made.append(p)
            for tag, t in (('trigger', t0), ('end_of_longest_move', t0 + k_end)):           # lossless key frames
                q = out / f'{name}_step{t}_{tag}.png'
                if q.exists(): raise FileExistsError(q)
                pan[t - first].save(q); made.append(q)
    src = {k: {'path': str(Path(folder) / rec[k]), 'sha256': rec[k + '_sha256']} for k in ('trace', 'frames') if rec.get(k)}
    src['scene'] = {'path': str(Path(folder) / 'scenes' / f'{seed}.json'), 'sha256': sha(d / 'scenes' / f'{seed}.json')}
    return {'seed': seed, 'outputs': {p.name: sha(p) for p in made}, 'sources': src}


def main():
    root, evid = sys.argv[1], Path(sys.argv[2]); procs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for p in (HERE / 'audit_summary.json', HERE / 'per_scene_checks.csv'):
        if p.exists(): raise SystemExit(f'Refused overwrite: {p}')
    jobs = [(st, f, root, int(p.stem)) for st, f in STAGES for p in sorted((Path(root) / f / 'scenes').glob('*.json'))]
    from multiprocessing import Pool
    with Pool(procs) as pool: rows = pool.map(one, jobs, chunksize=4)
    trig = [r for r in rows if r['metrics']['triggered']]
    counts = {}
    with open(HERE / 'per_scene_checks.csv', 'x', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['stage', 'scene', 'arm', 'y', 'acted', 'harm', 'rescue', 'check', 'status', 'evidence_json'])
        for r in trig:
            for arm, v in r['metrics']['arms'].items():
                for k, x in v['checks'].items():
                    counts.setdefault(arm, {}).setdefault(k, {s: 0 for s in T.STATUS})[x['status']] += 1
                    w.writerow([r['stage'], r['seed'], arm, v['y'], int(v['acted']), int(v['harm']), int(v['rescue']), k, x['status'], json.dumps({kk: vv for kk, vv in x.items() if kk != 'status'})])
    fails = {}
    for r in trig:
        for arm, v in r['metrics']['arms'].items():
            for k, x in v['checks'].items():
                if x['status'] == 'FAIL': fails.setdefault(arm, {}).setdefault(k, []).append(r['seed'])
    sy = [r['sync'] for r in trig if r['sync']]
    import numpy as np
    lagmean = {l: float(np.mean([s['mean_px_distance_by_lag'][l] for s in sy if s['mean_px_distance_by_lag'][l] is not None])) for l in (-2, -1, 0, 1, 2)}
    ctl = [v['max_difference_m'] for r in trig for v in r['control_tcp_sync'].values()]
    cf = [x for r in trig for x in r['control_frames'].values() if x]
    sync = {'scenes_with_frames': len(sy), 'coordinates_px_median_over_scenes': {k: float(np.median([s['coordinates_px_median'][k] for s in sy if s['coordinates_px_median'][k] is not None])) for k in ('cubeA', 'cubeB')},
            'coordinates_px_p95_over_scenes': {k: float(np.percentile([s['coordinates_px_median'][k] for s in sy if s['coordinates_px_median'][k] is not None], 95)) for k in ('cubeA', 'cubeB')},
            'image_lag_mean_px_by_lag': lagmean, 'image_lag_best_by_scene': {str(l): sum(1 for s in sy if s['best_lag'] == l) for l in (-2, -1, 0, 1, 2, None)},
            'control_steps_tcp_observation_vs_trace': {'moves': len(ctl), 'max_difference_m': max(ctl) if ctl else None},
            'v2_control_frames_held_cube_visible_at_projection': {'frames_in_view': sum(x['projected_in_view'] for x in cf), 'red_at_projection': sum(x['red_at_projection_base_camera'] for x in cf)}}
    cases = {k: sorted(r['seed'] for r in trig if rule(r)) for k, rule in CASE_RULES.items()}
    cases.update({f'representative_{k}': sorted(r['seed'] for r in trig if rule(r))[:3] for k, rule in REPRESENTATIVE.items()})
    cases['named_in_the_request'] = list(NAMED)
    pick = sorted({s for k, v in cases.items() if k not in ('any_check_failed', 'zero_step_intervention_without_need') for s in v} | set(cases['any_check_failed'][:12]) | set(cases['zero_step_intervention_without_need'][:3]))     # all scenes of every risk rule; the two broad lists contribute their lowest seeds
    by = {r['seed']: r for r in rows}; man = []
    with Pool(procs) as pool: man = pool.map(figures, [(by[s]['folder'], root, s, str(evid / f'scene_{s}')) for s in pick])
    code = {f: sha(f) for f in ('src/hci/trajectory_audit.py', 'src/hci/vision_center.py', 'results/vision_center_v2_c1_001/trajectory/run_trajectory_audit.py')}
    figure_selection = {k: ('lowest 12 seeds' if k == 'any_check_failed' else 'lowest 3 seeds' if k == 'zero_step_intervention_without_need' else 'lowest 3 seeds (rule is a representative sample)' if k.startswith('representative_') else 'all matching scenes') for k in cases}
    (evid / 'MANIFEST.json').write_text(json.dumps({'cases': cases, 'figure_selection': figure_selection, 'scenes': man, 'code_sha256': code, 'phase_b_root': 'runs/vision_center_v2_001/server_backup/extracted/runs/vision_center_v2_001 (git-ignored server backup; sources listed per scene with sha256)'}, indent=1), encoding='utf-8')
    summary = {'what': 'trajectory and behaviour audit of the stored Phase B records (used development scenes, train seed 1); no rollout', 'scenes': len(rows), 'triggered': len(trig),
               'by_stage': {st: sum(1 for r in trig if r['stage'] == st) for st, _ in STAGES}, 'check_counts': counts, 'fail_scenes': fails, 'sync': sync, 'cases': cases, 'figure_scenes': pick,
               'thresholds': {k: getattr(T, k) for k in ('RESIDUAL_TOL_M', 'STEP_CMD_M', 'TARGET_CAP_M', 'MAX_STEPS', 'DIR_COS_MIN', 'PROGRESS_MIN_M', 'DZ_TOL_M', 'WIDTH_TOL_M', 'SLIP_TOL_M', 'PUSH_TOL_M', 'REVERT_FRACTION_MAX', 'SAME_TOL', 'MEMORY_ERROR_M')},
               'code_sha256': code, 'evidence_manifest_sha256': sha(evid / 'MANIFEST.json')}
    with open(HERE / 'audit_summary.json', 'x', encoding='utf-8') as f: json.dump(summary, f, indent=1)
    print(json.dumps({'scenes': len(rows), 'triggered': len(trig), 'figure_scenes': len(pick), 'sync': sync}, indent=1))


if __name__ == '__main__':
    main()
