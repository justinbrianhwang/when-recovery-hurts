"""S6 measurement-validity analysis on the stored local CPU replay (LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS).

The official S6 (|PhysX separation - reconstructed gap| <= 1 mm for every pair PhysX reports separated) is FAIL and stays
FAIL. This module only decomposes the stored differences. No simulation, no sub-step instrumentation.

Compared values, per listed shape pair and logged row t:
  physx_sep  = min over the pair's contact points of the PhysX point separation (signed, metres) from the stored contact
               list, which ManiSkill reads after the last of the control step's physics sub-steps;
  recon      = signed gap between the two convex shapes posed at the logged link / actor poses of row t (after the step),
               computed by hci.spatial_links.gap in the world frame;
  diff       = physx_sep - recon.
Explanatory term (post hoc, NOT a correction of S6):
  pred       = -(v_a - v_b) . n * dt_sub, with v from the finite difference of the logged positions between rows t-1 and t
               (an average over the control step, an approximation), n the stored normal of the minimum-separation point,
               dt_sub the stored sim_timestep_s. The sign convention of n is taken from the data, not assumed: both signs
               are fitted and reported.
"""
from __future__ import annotations

import gzip, json, math
from pathlib import Path

import numpy as np

from hci import spatial_links as SL


def body_pos(row, name):
    if name in ('cubeA', 'cubeB'): return np.asarray(row[name], float)
    if name == 'table' or name.startswith('ground'): return np.zeros(3)
    lp = row.get('link_pose') or {}
    return np.asarray(lp[name][:3], float) if name in lp else None


def body_quat(row, name):
    if name in ('cubeA', 'cubeB'): return np.asarray(row[f'{name}_q'], float)
    lp = row.get('link_pose') or {}
    return np.asarray(lp[name][3:7], float) if name in lp else None


def ang_speed(q0, q1, dt):
    if q0 is None or q1 is None: return None
    d = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)))); return 2 * math.acos(min(1.0, d)) / dt


def min_point(contacts, a, sa, b, sb):
    for c in contacts:
        if (c['a'], c['shape_a'], c['b'], c['shape_b']) == (a, sa, b, sb):
            P = min(c['points'], key=lambda p: p[9]); return np.asarray(P[3:6], float), float(P[9])
        if (c['b'], c['shape_b'], c['a'], c['shape_a']) == (a, sa, b, sb):
            P = min(c['points'], key=lambda p: p[9]); return -np.asarray(P[3:6], float), float(P[9])
    return None, None


def phase_of(arm, t, trigger, K):
    if arm == 'trunk': return 'trunk_before_trigger' if trigger is None or t <= trigger else 'trunk_after_trigger'
    if K and trigger is not None and trigger < t <= trigger + K: return 'control'
    return 'after_control' if trigger is not None and t > trigger else 'before_trigger'


def collect(replay_dir, records):
    """records: {scene: phase-B record} for the control step counts. Returns one row per (scene, arm, t, shape pair).
    Layout of the stored local replay: <replay_dir>/traces/trace_<seed>.json.gz with seed, trigger_step and the collision
    geometry in the trunk's initial row."""
    out = []; collect.excluded = {'rows_without_contacts_or_link_poses': 0, 'separated_pairs_not_reconstructed': 0}
    for f in sorted(Path(replay_dir, 'traces').glob('trace_*.json.gz')):
        t = json.load(gzip.open(f, 'rt', encoding='utf-8')); seed = int(t['seed']); tr = t['traces']
        out += scene_rows(seed, t.get('trigger_step'), tr, tr['trunk'][0]['collision_geometry'], records.get(seed) or {}, collect.excluded)
    return out


def _c1_scene(item):
    """One C1 scene: (rows, excluded counts). Pure function of the record and its traces."""
    rec, tr = item
    if isinstance(tr, (str, Path)): tr = json.load(gzip.open(tr, 'rt', encoding='utf-8'))['traces']     # a trace path (hash checked by the gate before)
    ex = {'rows_without_contacts_or_link_poses': 0, 'separated_pairs_not_reconstructed': 0, 'scenes_without_collision_geometry': 0}
    geo = (rec.get('initial_state') or {}).get('collision_geometry') or (tr.get('trunk') or [{}])[0].get('collision_geometry')
    if geo is None: ex['scenes_without_collision_geometry'] += 1; return [], ex
    return scene_rows(rec['scene'], rec['trigger']['step'] if rec.get('triggered') else None, tr, geo, rec, ex), ex


def collect_c1(items, workers=1):
    """The same rows for C1 records (review F3): items = [(record, traces)] of one pass. The C1 collector stores the collision
    geometry in record.initial_state (hci.spatial_probe); the trigger step comes from the record. workers > 1 computes the
    scenes in a process pool: the same per-scene function, results kept in the input order (speed only)."""
    items = list(items)
    if workers > 1 and len(items) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex: res = list(ex.map(_c1_scene, items, chunksize=1))
    else: res = [_c1_scene(i) for i in items]
    out = []; collect_c1.excluded = {'rows_without_contacts_or_link_poses': 0, 'separated_pairs_not_reconstructed': 0, 'scenes_without_collision_geometry': 0}
    for rows, ex_ in res:
        out += rows
        for k, v in ex_.items(): collect_c1.excluded[k] += v
    return out


def scene_rows(seed, T, tr, geo, rec, excluded):
    out = []; Ks = {c['arm']: int(c.get('control_steps') or 0) for c in rec.get('cells', [])}
    for arm, rows in tr.items():
        for r0, r1 in zip(rows, rows[1:]):
            if arm != 'trunk' and T is not None and int(r1['t']) <= T: continue      # an arm trace repeats the trunk prefix up to the trigger: count it once (in the trunk)
            if 'contacts' not in r1 or r1.get('link_pose') is None: excluded['rows_without_contacts_or_link_poses'] += 1; continue
            dt_c = float(r1['sim_timestep_s']) * float(r1['sim_substeps_per_control']); dt_s = float(r1['sim_timestep_s'])
            m = SL.row_metrics(r1, geo)
            excluded['separated_pairs_not_reconstructed'] += sum(1 for s in m['contact_shape_pairs_not_reconstructed'] if s['physx_min_separation_m'] > 0)
            for s in m['separation_vs_reconstruction']:
                if s['physx_min_separation_m'] <= 0: continue      # the S6 population: pairs PhysX reports separated
                n, sep = min_point(r1['contacts'], s['a'], s['shape_a'], s['b'], s['shape_b'])
                pa0, pa1, pb0, pb1 = body_pos(r0, s['a']), body_pos(r1, s['a']), body_pos(r0, s['b']), body_pos(r1, s['b'])
                vrel = None if any(x is None for x in (pa0, pa1, pb0, pb1)) else ((pa1 - pa0) - (pb1 - pb0)) / dt_c
                va = None if pa0 is None or pa1 is None else (pa1 - pa0) / dt_c
                out.append({'scene': seed, 'arm': arm, 't': int(r1['t']), 'phase': phase_of(arm, int(r1['t']), T, Ks.get(arm, 0)),
                            'a': s['a'], 'shape_a': s['shape_a'], 'b': s['b'], 'shape_b': s['shape_b'], 'kind': SL.category(s['a'], s['b']),
                            'physx_sep_m': s['physx_min_separation_m'], 'recon_m': s['reconstructed_m'], 'diff_m': s['physx_min_separation_m'] - s['reconstructed_m'],
                            'listed_in_contact_list': True, 'nonzero_impulse_recorded': s.get('observed'),      # listed pair vs a recorded nonzero impulse: kept apart
                            'normal': None if n is None else n.tolist(), 'normal_z_abs': None if n is None else abs(float(n[2])),
                            'vrel_dot_n_mps': None if vrel is None or n is None else float(np.dot(vrel, n)), 'speed_rel_mps': None if vrel is None else float(np.linalg.norm(vrel)),
                            'vrel_z_mps': None if vrel is None else float(vrel[2]), 'v_a_z_mps': None if va is None else float(va[2]),
                            'ang_speed_a_rps': ang_speed(body_quat(r0, s['a']), body_quat(r1, s['a']), dt_c), 'dt_sub_s': dt_s, 'dt_control_s': dt_c})
    return out


def fit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return {'n': int(len(x)), 'r': None, 'slope': None}
    r = float(np.corrcoef(x, y)[0, 1]); slope = float(np.dot(x, y) / np.dot(x, x)); return {'n': int(len(x)), 'r': r, 'slope_through_origin': slope}


def q(a, p): return None if len(a) == 0 else float(np.percentile(np.abs(np.asarray(a, float)), p))


def summarise(rows, tol=SL.SEPARATION_TOL_M, excluded=None):
    d = [r['diff_m'] for r in rows]; ok = [r for r in rows if r['vrel_dot_n_mps'] is not None]
    x = [r['vrel_dot_n_mps'] * r['dt_sub_s'] for r in ok]; y = [r['diff_m'] for r in ok]
    f = fit(x, y); s = f.get('slope_through_origin')
    resid = [yy - (s * xx if s is not None else 0.0) for xx, yy in zip(x, y)]
    def by(key, bins=None):
        g = {}
        for r in rows:
            k = r[key] if bins is None else bins(r)
            g.setdefault(str(k), []).append(r['diff_m'])
        return {k: {'n': len(v), 'median_abs_mm': q(v, 50) * 1e3, 'p99_abs_mm': q(v, 99) * 1e3, 'max_abs_mm': q(v, 100) * 1e3, 'over_tol': int(sum(abs(x) > tol for x in v))} for k, v in sorted(g.items())}
    speed_bin = lambda r: 'NA' if r['speed_rel_mps'] is None else ('<0.01' if r['speed_rel_mps'] < 0.01 else '0.01-0.05' if r['speed_rel_mps'] < 0.05 else '0.05-0.1' if r['speed_rel_mps'] < 0.1 else '>=0.1')
    nz_bin = lambda r: 'NA' if r['normal_z_abs'] is None else ('vertical |n_z|>=0.9' if r['normal_z_abs'] >= 0.9 else 'horizontal |n_z|<0.1' if r['normal_z_abs'] < 0.1 else 'oblique')
    rot_bin = lambda r: 'NA' if r['ang_speed_a_rps'] is None else ('<0.1 rad/s' if r['ang_speed_a_rps'] < 0.1 else '0.1-1 rad/s' if r['ang_speed_a_rps'] < 1 else '>=1 rad/s')
    excluded = excluded or {}; incomplete = any(excluded.values())
    return {'official_S6': {'tolerance_m': tol, 'pairs': len(rows), 'over_tolerance': int(sum(abs(x) > tol for x in d)), 'excluded': excluded,
                            'status': 'FAIL' if any(abs(x) > tol for x in d) else ('NOT_OBSERVABLE' if not rows or incomplete else 'PASS'),
                            'note': 'unchanged; the explanatory fit below never changes it'},
            'abs_diff_mm': {'median': q(d, 50) * 1e3 if d else None, 'p90': q(d, 90) * 1e3 if d else None, 'p99': q(d, 99) * 1e3 if d else None, 'max': q(d, 100) * 1e3 if d else None},
            'signed_diff_mm_median': float(np.median(d)) * 1e3 if d else None,
            'contact_list_vs_impulse': {'separated_pairs_listed_in_contact_list': len(rows), 'of_which_nonzero_impulse_recorded': int(sum(1 for r in rows if r.get('nonzero_impulse_recorded') is True)),
                                        'of_which_impulse_flag_absent': int(sum(1 for r in rows if r.get('nonzero_impulse_recorded') is None)),
                                        'note': 'a listed pair is a PhysX contact-list entry (proximity); a recorded nonzero impulse is an observed contact; the two are never merged'},
            'explanatory_fit_diff_vs_vreln_dtsub': {**f, 'pairs_without_velocity': len(rows) - len(ok),
                                                    'residual_abs_mm_after_fit': {'median': q(resid, 50) * 1e3 if resid else None, 'p99': q(resid, 99) * 1e3 if resid else None, 'max': q(resid, 100) * 1e3 if resid else None},
                                                    'residual_over_tol_after_fit': int(sum(abs(x) > tol for x in resid)), 'status': 'EXPLANATORY_ONLY'},
            'by_pair_kind': by('kind'), 'by_phase': by('phase'), 'by_relative_speed': by('speed_rel_mps', speed_bin), 'by_normal_direction': by('normal_z_abs', nz_bin),
            'by_rotation_of_body_a': by('ang_speed_a_rps', rot_bin),
            'fit_by_normal_direction': {k: fit([r['vrel_dot_n_mps'] * r['dt_sub_s'] for r in ok if nz_bin(r) == k], [r['diff_m'] for r in ok if nz_bin(r) == k]) for k in ('vertical |n_z|>=0.9', 'horizontal |n_z|<0.1', 'oblique')},
            'fit_by_pair_kind': {k: fit([r['vrel_dot_n_mps'] * r['dt_sub_s'] for r in ok if r['kind'] == k], [r['diff_m'] for r in ok if r['kind'] == k]) for k in sorted({r['kind'] for r in ok})}}
