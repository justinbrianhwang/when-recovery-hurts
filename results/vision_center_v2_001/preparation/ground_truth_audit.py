"""Vision-CENTER v2, Phase A: what the stored records of the USED scenes 50600-50899 (Vision-CENTER run2) say about
observability. No simulator, no policy, no new scene. Ground truth from the logged traces is the LABEL here; nothing in this
file feeds a controller.

  1  does the target cube move before the trigger, and when (per-step logged poses of all 289 triggered scenes)
  2  does the held cube move in the gripper between the grasp and the trigger
  3  does the 'gripper near the remembered target' rule of hci.vision_center_v2 cover the steps in which the target moved
  4  carry height and offset against a pushed target, for every centring move that was executed (v1 and the oracle)
  5  the held-cube fit on the 289 real trigger frames: the three candidate solutions of the unchanged v1 fit against truth
  6  the v2 decision rules applied to the estimates v1 recorded at the trigger (frame-only part of v2; no memory)
Usage (repository root): python results/vision_center_v2_001/preparation/ground_truth_audit.py <out.json> [processes]"""
import os
for _k in ('OMP', 'OPENBLAS', 'MKL', 'NUMEXPR', 'BLIS'): os.environ[_k + '_NUM_THREADS'] = '1'
import glob, gzip, json, sys
import numpy as np
sys.path.insert(0, 'src')
from hci import vision_center as V1
from hci import vision_center_v2 as V2
R = 'results/vision_center_001/run2/'; H = V1.HOLD_FIT; NAMED = (50813, 50872, 50817, 50846, 50892)
pct = lambda a, q=(50, 90, 99, 100): None if len(a) == 0 else {f'p{k}': round(float(np.percentile(a, k)) * 1000, 2) for k in q}


def candidates(img):
    """The v1 held-cube fit with its three candidate solutions exposed (same search, same scores)."""
    k = V1.classes(img); lit = k['red_lit']; n = int(lit.sum())
    if n < H['pixels_min']: return None
    red = k['red']; known = ~k['grey']; nred = int(red.sum()); z0, z1 = H['z_step_m']; st = H['stages']

    def outline(h, yaw):
        sil = V1.hold_silhouette(h, yaw); return 0.0 if sil is None else int((sil & red).sum()) / max(nred + int((sil & ~red & known).sum()), 1)
    coarse = [V1._hold_at(lit, hz, (0.0, 0.0), 0.0, np.pi / 4, st[:1], 2) for hz in np.arange(-H['z_span_m'], H['z_span_m'] + 1e-9, z0)]; out = []
    for c in sorted(coarse, key=lambda r: -(r[0] + outline(r[1], r[2])))[:3]:
        best = c
        for hz in c[1][2] + np.arange(-z0 + z1, z0 - 1e-9, z1):
            r = V1._hold_at(lit, hz, c[1][:2], c[2], np.pi / 10, st[1:2], 2)
            if r[0] > best[0]: best = r
        best = V1._hold_at(lit, best[1][2], best[1][:2], best[2], np.pi / 60, st[2:], 1); so = outline(best[1], best[2])
        out.append({'h': [float(v) for v in best[1]], 'iou': float(best[0]), 'outline': float(so), 'total': float(best[0] + so)})
    out.sort(key=lambda x: -x['total']); return out


def scene(a):
    stage, p = a; r = json.load(open(p, encoding='utf-8'))
    if not r['triggered']: return {'scene': r['scene'], 'stage': stage, 'triggered': False, 'y0': r['cells'][0]['y']}
    s = r['scene']; T = r['trigger']['step']; cells = {c['arm']: c for c in r['cells']}; v = cells['VISION_CENTER']; g = r['trigger']['gt_for_diagnosis_only']; rd = v['control']['readings'][0]
    with gzip.open(R + stage + f'/traces/trace_{s}.json.gz', 'rt', encoding='utf-8') as fh: st = [r['initial_state']] + json.load(fh)['traces']['trunk']
    B = np.array([x['cubeB'][:2] for x in st[:T + 1]]); P = np.array([x['tcp'] for x in st[:T + 1]]); hold = lambda x: V1.quat_R(x['tcp_q']).T @ (np.array(x['cubeA']) - np.array(x['tcp']))
    step = np.r_[0, np.linalg.norm(np.diff(B, axis=0), axis=1)]; moved = step > 0.0003; near = (np.linalg.norm(P[:, :2] - B, axis=1) < V2.CONTACT_RADIUS_M) & (P[:, 2] < V2.CONTACT_TCP_Z_M)
    grasp = [i for i in range(T + 1) if st[i]['is_cubeA_grasped'] and st[i]['cubeA'][2] > 0.03]; g0 = grasp[0] if grasp else None; hT = hold(st[T]); width = [float(sum(x['finger_qpos'])) for x in st[:T + 1]]
    drift = None if g0 is None else float(max(np.linalg.norm((hold(st[i]) - hT)[:2]) for i in range(min(g0 + 8, T), T + 1)))
    img = np.load(R + stage + f'/frames/hand_{s}.npz')['frames'][0]; c = candidates(img); top = None if not c else c[0]
    near_tie = [] if not c else [q for q in c if q['total'] >= top['total'] - H['margin_min']]
    out = {'scene': s, 'stage': stage, 'triggered': True, 'trigger_step': T, 'y0': cells['C0']['y'], 'y_v1': v['y'], 'y_oracle': cells['CENTER_ORACLE']['y'] if 'CENTER_ORACLE' in cells else None, 'v1_decision': v['control']['decision'],
           'true_offset_m': g['offset_xy_m'], 'true_gap_m': g['height_gap_m'], 'target_moved_before_trigger_m': float(np.linalg.norm(B[T] - B[0])), 'target_first_moved_step': int(np.argmax(moved)) if moved.any() else None,
           'target_moved_in_last_10_steps_m': float(np.linalg.norm(B[T] - B[max(0, T - 10)])), 'steps_target_moved': int(moved.sum()), 'steps_target_moved_while_gripper_not_near': int((moved & ~near).sum()),
           'target_displacement_while_gripper_not_near_m': float(step[~near].sum()), 'steps_gripper_near': int(near.sum()), 'first_step_gripper_near_after_grasp': next((i for i in range(g0 or 0, T + 1) if near[i]), None),
           'contact_force_A_B_before_trigger': bool(max(x.get('f_A_B', 0.) for x in st[:T + 1]) > 0), 'first_lifted_grasp_step': g0, 'hold_xy_change_from_grasp_to_trigger_m': drift,
           'finger_width_range_last_10_steps_m': float(max(width[max(0, T - 10):]) - min(width[max(0, T - 10):])), 'true_hold': hT.tolist(),
           'v1_hold_valid': bool(rd['hold_fit'] and rd['hold_fit']['valid']), 'hold_top_error_xy_m': None if top is None else float(np.linalg.norm(np.array(top['h'][:2]) - hT[:2])), 'hold_top_error_z_m': None if top is None else float(abs(top['h'][2] - hT[2])),
           'hold_near_tie_xy_spread_m': None if top is None else float(max(np.linalg.norm(np.array(q['h'][:2]) - np.array(top['h'][:2])) for q in near_tie)), 'hold_candidates': c,
           'v1_estimated_offset_m': rd['offset_norm_m'], 'v1_estimated_held_z_m': None if rd['held_xyz'] is None else rd['held_xyz'][2]}
    for arm, key in (('VISION_CENTER', 'v1'), ('CENTER_ORACLE', 'oracle')):
        if arm in cells and cells[arm]['acted']: out[f'{key}_pushed_target_m'] = cells[arm]['gt_effect_of_control']['cubeB_xy_displacement_m']; out[f'{key}_end'] = cells[arm]['control']['end_reason']
    if rd['offset_norm_m'] is not None:                       # the frame-only part of v2 on the estimate v1 recorded: bound u = hand + hold
        est = {'why': None, 'offset_norm_m': rd['offset_norm_m'], 'u': V2.U_HAND_M + V2.U_HOLD_M, 'gap_m': rd['held_xyz'][2] - V1.CUBE_HALF_M - V1.TARGET_TOP_Z_M, 'target_src': 'frame', 'held_xyz': np.array(rd['held_xyz'])}
        out['v2_rules_on_v1_estimate'] = V2.decide(est, 200 - T)['reason']
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    OUT = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    jobs = [(st, p) for st in ('shadow', 'pilot') for p in sorted(glob.glob(R + st + '/scenes/*.json'))]
    with Pool(n) as p: rows = p.map(scene, jobs, chunksize=4)
    T = [x for x in rows if x['triggered']]; fail = [x for x in T if not x['y0']]; hid = [x for x in fail if x['v1_decision'] == 'NO_TARGET_ESTIMATE']; mv = [x for x in T if x['target_moved_before_trigger_m'] > 0.001]
    ok = [x for x in T if x['v1_hold_valid']]; rej = [x for x in T if not x['v1_hold_valid'] and x['hold_candidates']]; acted = [x for x in T if 'v1_pushed_target_m' in x]; orc = [x for x in T if 'oracle_pushed_target_m' in x]
    band = lambda L, key: [{'gap_from_mm': lo, 'gap_to_mm': hi, 'moves': sum(1 for x in L if lo <= x['true_gap_m'] * 1000 < hi), 'target_pushed_over_1mm': sum(1 for x in L if lo <= x['true_gap_m'] * 1000 < hi and x[key] > 0.001)}
                           for lo, hi in ((-99, -1), (-1, 0), (0, 2), (2, 4), (4, 99))]
    changed = [x for x in T if x['v1_decision'] == 'ACT' and x.get('v2_rules_on_v1_estimate') != 'ACT']
    summary = {
        'scenes': len(rows), 'triggered': len(T), 'c0_failures_with_trigger': len(fail),
        'target': {'moved_over_1mm_before_trigger': len(mv), 'moved_over_5mm': sum(1 for x in mv if x['target_moved_before_trigger_m'] > 0.005), 'of_which_in_the_last_10_steps_only': sum(1 for x in mv if abs(x['target_moved_in_last_10_steps_m'] - x['target_moved_before_trigger_m']) < 0.0005),
                   'cube_contact_before_trigger': sum(1 for x in T if x['contact_force_A_B_before_trigger']), 'moved_scenes': [[x['scene'], round(x['target_moved_before_trigger_m'] * 1000, 1), x['trigger_step'] - x['target_first_moved_step'], x['y0']] for x in mv],
                   'hidden_target_c0_failures': len(hid), 'hidden_target_c0_failures_where_target_moved_over_1mm': [x['scene'] for x in hid if x['target_moved_before_trigger_m'] > 0.001]},
        'contact_rule': {'radius_m': V2.CONTACT_RADIUS_M, 'tcp_z_m': V2.CONTACT_TCP_Z_M, 'steps_flagged_per_scene_median': float(np.median([x['steps_gripper_near'] for x in T])),
                         'scenes_where_target_moved_outside_flagged_steps': [[x['scene'], round(x['target_displacement_while_gripper_not_near_m'] * 1000, 1)] for x in T if x['target_displacement_while_gripper_not_near_m'] > 0.0005]},
        'hold': {'xy_change_from_grasp_to_trigger_mm': pct([x['hold_xy_change_from_grasp_to_trigger_m'] for x in T if x['hold_xy_change_from_grasp_to_trigger_m'] is not None]),
                 'v1_accepted': len(ok), 'accepted_error_xy_mm': pct([x['hold_top_error_xy_m'] for x in ok]), 'accepted_error_z_mm': pct([x['hold_top_error_z_m'] for x in ok]),
                 'v1_rejected': len(rej), 'rejected_error_xy_mm': pct([x['hold_top_error_xy_m'] for x in rej]), 'rejected_error_z_mm': pct([x['hold_top_error_z_m'] for x in rej]),
                 'rejected_whose_near_tie_candidates_agree_within_2mm': sum(1 for x in rej if x['hold_near_tie_xy_spread_m'] <= 0.002), 'rejected_c0_failures': [x['scene'] for x in rej if not x['y0']]},
        'pushed_target_by_carry_height': {'v1_moves': band(acted, 'v1_pushed_target_m'), 'oracle_moves': band(orc, 'oracle_pushed_target_m'),
                                          'pushed_scenes': [[x['scene'], round(x['true_offset_m'] * 1000, 1), round(x['true_gap_m'] * 1000, 1), round(x.get('v1_pushed_target_m', 0) * 1000, 1), round(x.get('oracle_pushed_target_m', 0) * 1000, 1)] for x in T
                                                            if x.get('v1_pushed_target_m', 0) > 0.001 or x.get('oracle_pushed_target_m', 0) > 0.001],
                                          'rescued_with_true_offset_over_30mm_and_gap_under_3mm': [[x['scene'], round(x['true_offset_m'] * 1000, 1), round(x['true_gap_m'] * 1000, 1)] for x in acted if not x['y0'] and x['y_v1'] and x['true_offset_m'] > 0.03 and x['true_gap_m'] < 0.003]},
        'v2_rules_on_v1_acts': {'v1_acts': sum(1 for x in T if x['v1_decision'] == 'ACT'), 'no_longer_acts': [[x['scene'], x['v2_rules_on_v1_estimate'], round(x['v1_estimated_offset_m'] * 1000, 1), x['y0'], x['y_v1']] for x in changed],
                                'rescues_lost': sum(1 for x in changed if not x['y0'] and x['y_v1']), 'note': 'frame-only part of v2 (uncertainty margin, side-contact rule) applied to the estimates v1 recorded; these are the scenes the rules were designed on'},
        'named_scenes': {str(x['scene']): {k: x[k] for k in ('trigger_step', 'v1_decision', 'y0', 'y_oracle', 'true_offset_m', 'true_gap_m', 'target_moved_before_trigger_m', 'first_lifted_grasp_step', 'hold_xy_change_from_grasp_to_trigger_m',
                                                              'finger_width_range_last_10_steps_m', 'v1_hold_valid', 'hold_top_error_xy_m', 'hold_top_error_z_m', 'hold_near_tie_xy_spread_m', 'contact_force_A_B_before_trigger')} for x in T if x['scene'] in NAMED}}
    with open(OUT, 'x', encoding='utf-8') as fo: json.dump({'status': 'OFFLINE, used scenes only; ground truth is a label', 'summary': summary, 'scenes': rows}, fo, indent=1)
    print(json.dumps(summary, indent=1))
