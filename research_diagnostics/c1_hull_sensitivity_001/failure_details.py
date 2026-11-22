"""Details of every S4 / S5 FAIL found by the full recomputation (the official C1 outputs computed S1-S6 only for the 14
case figures). Read-only; EXPLANATORY_ONLY where marked; no status is changed.

For each S5 violation row (observed contact pair reconstructed > 1 mm apart): impulse, PhysX separation, body speeds,
the trigger phase, and the reconstructed distance of the same two shapes at poses interpolated toward the previous logged
row (fractions f of the control step; f = 0.8 is one physics sub-step before the logged pose; linear position, slerp
orientation - an ASSUMPTION, as in the post-hoc physics diagnostics). For the S4 FAIL: the per-step distances, the
finger-target contact forces and the ground-truth grasp before and during the move.

usage: python -P research_diagnostics/c1_hull_sensitivity_001/failure_details.py --summary <S45 summary json> --out <json>
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hull_s45 as H  # noqa: E402
C = H.C; L = H.L; G = H.G
FRACS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)


def interp_row(r0, r1, f):
    """Row whose body poses are (1 - f) * r0 + f * r1 (position linear, orientation slerp); contacts are not used."""
    r = dict(r1); r['link_pose'] = {}
    for n, p in r1['link_pose'].items():
        q = r0['link_pose'][n]; r['link_pose'][n] = list((1 - f) * np.asarray(q[:3]) + f * np.asarray(p[:3])) + list(G.slerp(q[3:], p[3:], f))
    for c in ('cubeA', 'cubeB'):
        r[c] = list((1 - f) * np.asarray(r0[c]) + f * np.asarray(r1[c])); r[c + '_q'] = list(G.slerp(r0[c + '_q'], r1[c + '_q'], f))
    return r


def pair_gap(row, geo, a, ia, b, ib):
    W = L.world(row, geo); A = [x for x in W[a] if x.name == f'{a}#{ia}']; B = [x for x in W[b] if x.name == f'{b}#{ib}']; return L.gap(A, B)


def rows_of(rec, tr, trace):
    """The continuous row sequence of a trace (trunk rows up to the trigger first for an arm)."""
    if trace == 'trunk': return list(tr['trunk'])
    t0 = rec['trigger']['step']; return [r for r in tr['trunk'] if r['t'] <= t0] + list(tr[trace])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); p.add_argument('--branches', required=True); a = p.parse_args()
    out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    br = list(csv.DictReader(open(a.branches, encoding='utf-8')))
    s5_fail = [b for b in br if b['S5_status_logged'] == 'FAIL']; s4_fail = [b for b in br if b['S4_status_logged'] == 'FAIL']
    res = {'label': C.LABEL, 's5_fail_branches': [{k: b[k] for k in ('policy', 'scene', 'arm', 'acted', 'trigger', 'y')} for b in s5_fail], 's5_violation_rows': [], 's4_fail': []}
    seen = set()
    for b in s5_fail:
        pol, sc, arm = b['policy'], int(b['scene']), b['arm']
        rec, _ = C.record(pol, sc); tr, _ = C.traces(pol, sc); geo = rec['initial_state']['collision_geometry']
        rows, src, t0, k, c = H.branch_rows(rec, tr, arm)
        full = {'trunk': rows_of(rec, tr, 'trunk')}
        for r, s in zip(rows, src):
            for pr in H.s5_pairs(r, L.world(r, geo)):
                if pr['reconstructed_m'] is None or pr['reconstructed_m'] <= L.CONTACT_GEOM_TOL_M: continue
                key = (pol, sc, s, int(r['t']), pr['a'], pr['shape_a'], pr['b'], pr['shape_b'])
                if key in seen: continue
                seen.add(key)
                seq = full.setdefault(s, rows_of(rec, tr, s)); prev = next(x for x in seq if x['t'] == r['t'] - 1)
                imp = [np.asarray(q[6:9]) for cc in r['contacts'] if {(cc['a'], cc['shape_a']), (cc['b'], cc['shape_b'])} == {(pr['a'], pr['shape_a']), (pr['b'], pr['shape_b'])} for q in cc['points']]
                shifted = {f'f={f:.1f}': pair_gap(interp_row(prev, r, f), geo, pr['a'], pr['shape_a'], pr['b'], pr['shape_b']) * 1e3 for f in FRACS}
                rel = np.linalg.norm(np.asarray(r['cubeA_v']) - np.asarray(r['cubeB_v'])) if {pr['a'], pr['b']} == {'cubeA', 'cubeB'} else None
                tt = rec['trigger']['step'] if rec.get('triggered') else None; kk = (C.cell(rec, s) or {}).get('control_steps', 0) if s != 'trunk' else 0
                res['s5_violation_rows'].append({'policy': pol, 'scene': sc, 'trace': s, 't': int(r['t']), 'trigger': tt, 'control_steps_of_trace': kk,
                    'phase': 'before_or_at_trigger' if tt is None or r['t'] <= tt else ('move' if r['t'] <= tt + kk else 'after_trigger_policy_acting'),
                    'pair': f"{pr['a']}#{pr['shape_a']} - {pr['b']}#{pr['shape_b']}", 'reconstructed_mm': pr['reconstructed_m'] * 1e3, 'physx_min_separation_mm': pr['physx_min_separation_m'] * 1e3,
                    'contact_points': len(imp), 'impulse_norm_sum_Ns': float(sum(np.linalg.norm(x) for x in imp)), 'force_N': float(np.linalg.norm(np.sum(imp, 0)) / r['sim_timestep_s']) if imp else 0.,
                    'cubeA_speed_mm_s': float(np.linalg.norm(r['cubeA_v']) * 1e3), 'cubeB_speed_mm_s': float(np.linalg.norm(r['cubeB_v']) * 1e3),
                    'cubeA_minus_cubeB_speed_times_substep_mm': None if rel is None else float(rel * r['sim_timestep_s'] * 1e3),
                    'is_cubeA_grasped': r['is_cubeA_grasped'], 'is_cubeA_on_cubeB': r['is_cubeA_on_cubeB'], 'reconstructed_at_interpolated_pose_mm_EXPLANATORY_ONLY': shifted,
                    'branches_sharing_this_row': sorted({bb['arm'] for bb in s5_fail if bb['policy'] == pol and int(bb['scene']) == sc})})
    for b in s4_fail:
        pol, sc, arm = b['policy'], int(b['scene']), b['arm']
        rec, _ = C.record(pol, sc); tr, _ = C.traces(pol, sc); geo = rec['initial_state']['collision_geometry']; c = C.cell(rec, arm); t0 = rec['trigger']['step']
        seq = rows_of(rec, tr, arm); steps = []
        for r in seq:
            if r['t'] < t0 - 3 or r['t'] > t0 + c['control_steps'] + 2: continue
            W = L.world(r, geo); d = H.s4_row(W); cp = L.contact_pairs(r)
            fb = {f'{x}-{y}': {'observed': v['observed'], 'force_N': v['force_N'], 'min_separation_mm': v['min_separation_m'] * 1e3} for (x, y), v in cp.items() if 'cubeB' in (x, y) and any('finger' in z for z in (x, y))}
            steps.append({'t': int(r['t']), 'phase': 'move' if t0 < r['t'] <= t0 + c['control_steps'] else ('trigger_or_before' if r['t'] <= t0 else 'after_move'), 'd_fingers_cubeB_mm': d['fingers'] * 1e3, 'd_hand_cubeB_mm': d['hand'] * 1e3,
                          'd_robot_cubeB_mm': d['robot'] * 1e3, 'finger_cubeB_contacts': fb, 'is_cubeA_grasped': r['is_cubeA_grasped'], 'cubeB_xy_mm': [v * 1e3 for v in r['cubeB'][:2]], 'tcp_xy_mm': [v * 1e3 for v in r['tcp'][:2]]})
        c0 = C.cell(rec, 'C0')
        res['s4_fail'].append({'policy': pol, 'scene': sc, 'arm': arm, 'trigger': t0, 'control_steps': c['control_steps'], 'y': c['y'], 'C0_y': c0['y'], 'harm_vs_C0': bool(c0['y'] and not c['y']),
                               'trigger_gt': rec['trigger'].get('gt_for_diagnosis_only'), 'gt_effect_of_control': c.get('gt_effect_of_control'), 'control_end_reason': (c.get('control') or {}).get('end_reason'), 'steps': steps})
    C.jdump(out, res)
    print(json.dumps({'s5_unique_violation_rows': len(res['s5_violation_rows']), 's4_fail': len(res['s4_fail'])}))
    for v in res['s5_violation_rows']: print(v['policy'], v['scene'], v['trace'], v['t'], v['pair'], round(v['reconstructed_mm'], 3), {k: round(x, 3) for k, x in v['reconstructed_at_interpolated_pose_mm_EXPLANATORY_ONLY'].items()})


if __name__ == '__main__':
    main()
