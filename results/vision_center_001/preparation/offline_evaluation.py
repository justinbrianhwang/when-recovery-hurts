"""Vision-CENTER, Phase A: offline evaluation of the perception and of the act/abstain decision on STORED data only.
No simulator, no policy, no new scene. Ground truth (cube poses in the stored traces and demonstration states) is used as the
label to score against and never as an input to hci.vision_center.

Data:
  F  camera frames of the 21 FAILED episodes of the consumed diagnostic set (50600-50699) with their per-step ground truth
     (the only policy rollouts with frames on this machine; successes of 50600-50699 and all of 50700-50799 have no frames)
  D  the 500 training demonstrations (motion planner; images, camera parameters, cube poses) - development data
  G  ground-truth traces of all 200 used scenes (no images): only for counts that need no perception
Usage (repository root): python results/vision_center_001/preparation/offline_evaluation.py <frames dir> <demo .h5> <out.json>
"""
import glob, gzip, json, re, sys, time
import numpy as np
sys.path.insert(0, 'src')
from hci import vision_center as VC
from hci import recovery_v1_oracle as O

FRAMES, DEMOS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DIAG = 'results/recovery_v1_diagnostic/run1'; PILOT = 'results/recovery_v1_oracle_xy_pilot_001/run1'
mm = lambda v: None if len(v) == 0 else {'n': int(len(v)), 'median': round(float(np.median(v)) * 1000, 2), 'p90': round(float(np.percentile(v, 90)) * 1000, 2), 'max': round(float(np.max(v)) * 1000, 2)}
say = lambda *a: print(*a, flush=True)


def used_scenes():
    out = []; eps = {json.loads(l)['seed']: json.loads(l) for l in open(DIAG + '/episodes.jsonl', encoding='utf-8')}
    for p in sorted(glob.glob(DIAG + '/traces/trace_*.json.gz')):
        with gzip.open(p, 'rt', encoding='utf-8') as f: t = json.load(f)
        st = [t['initial']] + t['rows']; acts = [r['action'] for r in t['rows']]; trig = None
        for i in range(1, len(acts)):
            if O.trigger_rule(acts[i][3], O.close_run(acts[:i]), sum(st[i]['finger_qpos'])): trig = i; break
        out.append({'seed': t['seed'], 'st': st, 'acts': acts, 'trig': trig, 'y': eps[t['seed']]['success_once'], 'set': 'recovery_v1_diag'})
    for p in sorted(glob.glob(PILOT + '/scenes/5*.json')):
        r = json.load(open(p, encoding='utf-8'))
        with gzip.open(PILOT + '/' + r['trace'], 'rt', encoding='utf-8') as f: t = json.load(f)['traces']
        out.append({'seed': r['scene'], 'st': [r['initial_state']] + t['trunk'], 'acts': [x['action'] for x in t['trunk']], 'trig': (r['trigger'] or {}).get('step'), 'y': r['trunk']['success'], 'set': 'recovery_v1_pilot_reserved'})
    return out


def truth(st):
    A, B, P = np.array(st['cubeA']), np.array(st['cubeB']), np.array(st['tcp']); R = VC.quat_R(st['tcp_q'])
    return {'A': A, 'B': B, 'offset_xy': B[:2] - A[:2], 'hold': R.T @ (A - P), 'offset_tcp_frame': (R.T @ (B - A))[:2]}


def case(img, pose, tr, steps_left):
    """One reading scored against the truth `tr`."""
    rd = VC.reading(img, pose); d = VC.decide(rd, steps_left); tf, hf = rd['target_fit'], rd['hold_fit']; off = float(np.linalg.norm(tr['offset_xy']))
    row = {'true_offset_m': off, 'true_offset_in_tcp_frame_m': tr['offset_tcp_frame'].tolist(), 'true_held_z_m': float(tr['A'][2]), 'decision': d['reason'], 'reading': VC.summary(rd),
           'target_error_m': None if not rd['target_fresh'] else float(np.linalg.norm(tf['xy'] - tr['B'][:2])), 'lit_target_pixels': int(VC.classes(img)['green_lit'].sum()),
           'hold_error_xy_m': None if hf is None else float(np.linalg.norm((hf['h'] - tr['hold'])[:2])), 'hold_error_z_m': None if hf is None else float(hf['h'][2] - tr['hold'][2]),
           'tcp_only_error_m': float(np.linalg.norm(tr['hold'][:2])), 'offset_error_m': None if rd['offset_xy'] is None else float(np.linalg.norm(rd['offset_xy'] - tr['offset_xy'])),
           'estimated_offset_m': rd['offset_norm_m']}
    row['unnecessary_act'] = bool(d['act'] and off <= VC.RESIDUAL_TOL_M); row['act_would_start_within_c0_safe_zone'] = bool(d['act'] and off <= 0.015)
    return row


def main():
    res = {'status': 'OFFLINE, stored data only; ground truth is a label, never an input', 'decision_constants': VC.DECISION, 'full_view': VC.FULL_VIEW, 'partial_fit': {k: v for k, v in VC.PARTIAL_FIT.items()},
           'hold_fit': {k: v for k, v in VC.HOLD_FIT.items()}}
    scenes = used_scenes(); by = {e['seed']: e for e in scenes}
    # ---- inventory
    have = sorted(int(re.findall(r'(\d+)\.npz', p)[0]) for p in glob.glob(FRAMES + '/frames_*.npz'))
    trig = [e for e in scenes if e['trig'] is not None]
    res['inventory'] = {'used_scenes': len(scenes), 'triggered': len(trig), 'scenes_with_frames': len(have), 'scenes_with_frames_list': have,
                        'with_frames_by_set': {s: sum(1 for x in have if by[x]['set'] == s) for s in ('recovery_v1_diag', 'recovery_v1_pilot_reserved')},
                        'with_frames_and_trigger': sum(1 for x in have if by[x]['trig'] is not None), 'with_frames_and_success': sum(1 for x in have if by[x]['y']),
                        'triggered_without_frames': len(trig) - sum(1 for x in have if by[x]['trig'] is not None),
                        'note': 'frames were stored for failed diagnostic episodes only; the pilot stored no frames'}
    say(json.dumps(res['inventory']))
    # ---- G: counts that need no perception (all 189 triggered used scenes)
    G = []
    for e in trig:
        s = e['st'][e['trig']]; tr = truth(s); B0 = np.array(e['st'][0]['cubeB'])
        G.append({'seed': e['seed'], 'y': int(e['y']), 'offset': float(np.linalg.norm(tr['offset_xy'])), 'hold_xy': float(np.linalg.norm(tr['hold'][:2])), 'held_z': float(tr['A'][2]), 'steps_left': 200 - e['trig'],
                  'target_moved_before_trigger': float(np.linalg.norm(tr['B'][:2] - B0[:2]))})
    off = np.array([g['offset'] for g in G]); y = np.array([g['y'] for g in G])
    res['ground_truth_counts'] = {
        'triggered': len(G), 'tcp_to_held_cube_xy_m': mm([g['hold_xy'] for g in G]), 'tcp_only_error_above_5mm': int(sum(g['hold_xy'] > 0.005 for g in G)),
        'held_cube_below_carry_limit': sorted(g['seed'] for g in G if g['held_z'] < VC.CARRY_Z_MIN_M), 'steps_left_below_limit': sorted(g['seed'] for g in G if g['steps_left'] < VC.STEPS_LEFT_MIN),
        'target_moved_before_trigger_above_2mm': int(sum(g['target_moved_before_trigger'] > 0.002 for g in G)),
        'by_true_offset': [{'from_m': lo, 'to_m': hi, 'scenes': int(((off > lo) & (off <= hi)).sum()), 'uninterrupted_success': int(y[(off > lo) & (off <= hi)].sum())} for lo, hi in ((0, .01), (.01, .015), (.015, .02), (.02, .03), (.03, .06))],
        'true_offset_at_least_act_min': int((off >= VC.ACT_MIN_M).sum()), 'of_which_uninterrupted_failures': int(((off >= VC.ACT_MIN_M) & (y == 0)).sum()),
        'uninterrupted_failures_below_act_min': int(((off < VC.ACT_MIN_M) & (y == 0)).sum())}
    say(json.dumps(res['ground_truth_counts']))
    # ---- F: failed diagnostic episodes with frames
    F = []; holds = []
    for s in have:
        e = by[s]; fr = np.load(FRAMES + '/frames_%d.npz' % s)['frames']; t = e['trig']
        if t is None: F.append({'seed': s, 'triggered': False}); continue
        st = e['st'][t]; row = case(fr[t][:, VC.IMG:], st['tcp'] + st['tcp_q'], truth(st), 200 - t); row.update(seed=s, triggered=True, trigger_step=t); F.append(row)
        say('F', s, row['decision'], 'true', round(row['true_offset_m'] * 1000, 1), 'est', None if row['estimated_offset_m'] is None else round(row['estimated_offset_m'] * 1000, 1), 'err', None if row['offset_error_m'] is None else round(row['offset_error_m'] * 1000, 2))
        for i in range(max(1, t - 36), t, 6):                                    # more held-cube frames of the same episodes
            si = e['st'][i]
            if not si['is_cubeA_grasped'] or si['cubeA'][2] - e['st'][0]['cubeA'][2] < 0.01: continue
            hf = VC.hold_fit(fr[i][:, VC.IMG:]); tr = truth(si)
            if hf is not None: holds.append({'seed': s, 't': i, 'iou': hf['iou'], 'valid': hf['valid'], 'error_xy_m': float(np.linalg.norm((hf['h'] - tr['hold'])[:2])), 'error_z_m': float(hf['h'][2] - tr['hold'][2])})
    T = [x for x in F if x['triggered']]; cov = [x for x in T if x['offset_error_m'] is not None]
    res['failed_episodes'] = {'episodes': len(F), 'triggered': len(T), 'decisions': {k: sum(1 for x in T if x['decision'] == k) for k in ('ACT',) + VC.ABSTAIN},
                              'offset_error_m': mm([x['offset_error_m'] for x in cov]), 'target_error_m': mm([x['target_error_m'] for x in cov]), 'hold_error_xy_m': mm([x['hold_error_xy_m'] for x in T if x['hold_error_xy_m'] is not None]),
                              'tcp_only_error_m': mm([x['tcp_only_error_m'] for x in T]), 'unnecessary_acts': sum(x['unnecessary_act'] for x in T), 'rows': F,
                              'hold_fit_other_frames': {'n': len(holds), 'valid': sum(h['valid'] for h in holds), 'error_xy_m_valid': mm([h['error_xy_m'] for h in holds if h['valid']]),
                                                        'abs_error_z_m_valid': mm([abs(h['error_z_m']) for h in holds if h['valid']]), 'rows': holds}}
    say(json.dumps({k: v for k, v in res['failed_episodes'].items() if k != 'rows' and k != 'hold_fit_other_frames'}), json.dumps({k: v for k, v in res['failed_episodes']['hold_fit_other_frames'].items() if k != 'rows'}))
    # ---- D: demonstrations
    import h5py
    f = h5py.File(DEMOS, 'r'); ks = sorted(f.keys(), key=lambda k: int(k.split('_')[1])); cal = {'base_extrinsic': 0., 'intrinsic': 0., 'hand_rotation': 0., 'hand_position_m': 0.}
    K = np.array([[VC.FOCAL_PX, 0, VC.CENTRE_PX], [0, VC.FOCAL_PX, VC.CENTRE_PX], [0, 0, 1.]])
    for k in ks[::10]:
        g = f[k]['obs']; cal['base_extrinsic'] = max(cal['base_extrinsic'], float(np.abs(g['sensor_param/base_camera/extrinsic_cv'][:] - VC.BASE_EXTRINSIC).max()))
        cal['intrinsic'] = max(cal['intrinsic'], float(np.abs(g['sensor_param/base_camera/intrinsic_cv'][:] - K).max()), float(np.abs(g['sensor_param/hand_camera/intrinsic_cv'][:] - K).max()))
        E = g['sensor_param/hand_camera/extrinsic_cv'][:]; tp = g['extra/tcp_pose'][:]
        for i in range(0, len(E), 9):
            R, p = VC.cam_hand(tp[i]); cal['hand_rotation'] = max(cal['hand_rotation'], float(np.abs(R - E[i][:, :3].T).max())); cal['hand_position_m'] = max(cal['hand_position_m'], float(np.abs(p + E[i][:, :3].T @ E[i][:, 3]).max()))
    res['calibration_check'] = {'max_abs_difference_to_stored_sensor_param': cal, 'demonstrations_checked': len(ks[::10]), 'meaning': 'the constants in hci.vision_center reproduce the stored camera parameters; the hand camera pose follows from the TCP pose alone'}
    say(json.dumps(res['calibration_check']))
    tv = {'hand_full': [], 'hand_partial_all': [], 'hand_partial_valid': [], 'base_valid': [], 'base_frames': 0, 'hand_frames': 0, 'hand_lit': 0}; cb = VC.cam_base()
    for k in ks[1::12]:                                                           # target estimate, every 4th frame of 42 demonstrations
        g = f[k]; hi = g['obs/sensor_data/hand_camera/rgb']; bi = g['obs/sensor_data/base_camera/rgb']; tp = g['obs/extra/tcp_pose'][:]; Bs = g['env_states/actors/cubeB'][:, :3]
        for i in range(0, len(tp), 4):
            B = Bs[i][:2]; tv['base_frames'] += 1; tv['hand_frames'] += 1; b = VC.target_full_view(bi[i], cb, 'base_camera')
            if b and b['valid']: tv['base_valid'].append(float(np.linalg.norm(b['xy'] - B)))
            cam = VC.cam_hand(tp[i]); fv = VC.target_full_view(hi[i], cam, 'hand_camera')
            if fv is None: continue
            tv['hand_lit'] += 1
            if fv['valid']: tv['hand_full'].append(float(np.linalg.norm(fv['xy'] - B)))
            else:
                r = VC.target_partial_fit(hi[i], cam, 'hand_camera')
                if r is not None:
                    e_ = float(np.linalg.norm(r['xy'] - B)); tv['hand_partial_all'].append(e_)
                    if r['valid']: tv['hand_partial_valid'].append(e_)
    res['target_estimate_on_demonstrations'] = {'frames': tv['hand_frames'], 'hand_frames_with_lit_target_pixels': tv['hand_lit'], 'hand_full_view_error_m': mm(tv['hand_full']),
                                                'hand_partial_view_fits': len(tv['hand_partial_all']), 'hand_partial_view_accepted_error_m': mm(tv['hand_partial_valid']),
                                                'hand_partial_view_all_error_m': mm(tv['hand_partial_all']), 'base_full_view_error_m': mm(tv['base_valid']),
                                                'base_full_view_error_above_5mm': int(sum(x > 0.005 for x in tv['base_valid'])), 'base_full_view_error_above_10mm': int(sum(x > 0.010 for x in tv['base_valid']))}
    say(json.dumps(res['target_estimate_on_demonstrations']))
    D = []; t0 = time.time()
    for k in ks[::3]:                                                             # the act/abstain decision at the demonstration's own release
        g = f[k]; acts = g['actions'][:]; q = g['obs/agent/qpos'][:]; trig = None
        for i in range(1, len(acts)):
            if O.trigger_rule(float(acts[i][3]), O.close_run([a for a in acts[:i]]), float(q[i][-2] + q[i][-1])): trig = i; break
        if trig is None: D.append({'demo': k, 'triggered': False}); continue
        tp = g['obs/extra/tcp_pose'][trig]; A = g['env_states/actors/cubeA'][trig][:3].astype(float); B = g['env_states/actors/cubeB'][trig][:3].astype(float); R = VC.quat_R(tp[3:7])
        tr = {'A': A, 'B': B, 'offset_xy': B[:2] - A[:2], 'hold': R.T @ (A - tp[:3]), 'offset_tcp_frame': (R.T @ (B - A))[:2]}
        row = case(g['obs/sensor_data/hand_camera/rgb'][trig], tp, tr, len(acts) - trig + 100); row.update(demo=k, triggered=True, trigger_step=trig); D.append(row)
    T = [x for x in D if x['triggered']]
    res['demonstration_releases'] = {'demonstrations': len(D), 'triggered': len(T), 'true_offset_m': mm([x['true_offset_m'] for x in T]), 'true_offset_above_10mm': sum(x['true_offset_m'] > 0.010 for x in T),
                                     'decisions': {k: sum(1 for x in T if x['decision'] == k) for k in ('ACT',) + VC.ABSTAIN}, 'unnecessary_acts': sum(x['unnecessary_act'] for x in T),
                                     'hold_fit_valid': sum(1 for x in T if x['reading']['hold_fit'] and x['reading']['hold_fit']['valid']), 'hold_error_xy_m_valid': mm([x['hold_error_xy_m'] for x in T if x['reading']['hold_fit'] and x['reading']['hold_fit']['valid']]),
                                     'abs_hold_error_z_m_valid': mm([abs(x['hold_error_z_m']) for x in T if x['reading']['hold_fit'] and x['reading']['hold_fit']['valid']]),
                                     'tcp_only_error_m': mm([x['tcp_only_error_m'] for x in T]), 'target_estimates': sum(1 for x in T if x['target_error_m'] is not None), 'target_error_m': mm([x['target_error_m'] for x in T if x['target_error_m'] is not None]),
                                     'offset_error_m': mm([x['offset_error_m'] for x in T if x['offset_error_m'] is not None]), 'seconds_per_reading': (time.time() - t0) / max(len(T), 1), 'rows': D,
                                     'limit': 'motion-planner states, not policy states: the cube is placed centred, so these mostly test abstaining and the held-cube fit'}
    say(json.dumps({k: v for k, v in res['demonstration_releases'].items() if k != 'rows'}))
    with open(OUT, 'x', encoding='utf-8') as fo: json.dump(res, fo, indent=1)


if __name__ == '__main__':
    main()
