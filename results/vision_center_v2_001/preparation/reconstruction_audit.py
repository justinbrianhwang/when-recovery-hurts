"""Vision-CENTER v2, Phase A: when was the target cube observable by the HAND camera before the trigger, in used scenes
that have no stored video? Hand images are reconstructed from the logged states (reconstruct.py, ground truth, offline
only) and passed to the unchanged v1 target fit. Part 1 checks the reconstruction against the 289 real trigger frames of
run2. Part 2 gives the timeline for every used scene of run2 in which v1 had no target estimate at the trigger, plus
50817, 50846 and 50892. The base camera cannot be reconstructed (the arm is not modelled).
Usage (repository root): python results/vision_center_v2_001/preparation/reconstruction_audit.py <diagnostic frames dir> <out.json> [processes]"""
import os
for _k in ('OMP', 'OPENBLAS', 'MKL', 'NUMEXPR', 'BLIS'): os.environ[_k + '_NUM_THREADS'] = '1'
import glob, gzip, json, re, sys
import numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hci import vision_center as V1
import reconstruct as RC
R = 'results/vision_center_001/run2/'; DIAG = 'results/recovery_v1_diagnostic/run1'; EXTRA = (50817, 50846, 50892); _RM = None


def robot_mask(frames_dir):
    global _RM
    if _RM is None:
        _RM = RC.RobotMask()
        for p in sorted(glob.glob(frames_dir + '/frames_*.npz')):
            s = int(re.findall(r'(\d+)\.npz', p)[0])
            with gzip.open(DIAG + f'/traces/trace_{s}.json.gz', 'rt', encoding='utf-8') as fh: t = json.load(fh)
            st = [t['initial']] + t['rows']; fr = np.load(p)['frames']
            for i in range(len(st)): _RM.add(fr[i][:, V1.IMG:], sum(st[i]['finger_qpos']))
    return _RM


def scene(a):
    stage, s, frames_dir = a; rm = robot_mask(frames_dir); r = json.load(open(R + stage + f'/scenes/{s}.json', encoding='utf-8')); T = r['trigger']['step']
    with gzip.open(R + stage + f'/traces/trace_{s}.json.gz', 'rt', encoding='utf-8') as fh: st = [r['initial_state']] + json.load(fh)['traces']['trunk']
    syn = lambda x: RC.render_hand(x['tcp'], x['tcp_q'], x['cubeA'], x['cubeA_q'], x['cubeB'], x['cubeB_q'], rm.get(sum(x['finger_qpos'])))
    pose = lambda x: np.array(x['tcp'] + x['tcp_q'], float); fit = lambda img, x: V1.target_partial_fit(img, V1.cam_hand(pose(x)), 'hand_camera')
    real = np.load(R + stage + f'/frames/hand_{s}.npz')['frames'][0]; fr = fit(real, st[T]); fs = fit(syn(st[T]), st[T]); vr = bool(fr and fr['valid']); vs = bool(fs and fs['valid'])
    v = [c for c in r['cells'] if c['arm'] == 'VISION_CENTER'][0]['control']; cells = {c['arm']: c['y'] for c in r['cells']}; g = r['trigger']['gt_for_diagnosis_only']
    out = {'scene': s, 'stage': stage, 'trigger_step': T, 'v1_decision': v['decision'], 'y': cells, 'true_offset_m': g['offset_xy_m'], 'true_gap_m': g['height_gap_m'],
           'check_at_trigger': {'real_valid': vr, 'reconstructed_valid': vs, 'xy_difference_m': float(np.linalg.norm(fr['xy'] - fs['xy'])) if (vr and vs) else None,
                                'real_lit_pixels': int(V1.classes(real)['green_lit'].sum()), 'reconstructed_lit_pixels': int(V1.classes(syn(st[T]))['green_lit'].sum())}}
    if v['decision'] != 'NO_TARGET_ESTIMATE' and s not in EXTRA: return out
    BT = np.array(st[T]['cubeB'][:2]); tl = []; grasp = next((i for i in range(T + 1) if st[i]['is_cubeA_grasped']), None)
    for i in range(0, T + 1, 2):
        ft = fit(syn(st[i]), st[i])
        if ft is not None and ft['valid']: tl.append({'t': i, 'error_m': float(np.linalg.norm(ft['xy'] - np.array(st[i]['cubeB'][:2]))), 'error_against_trigger_m': float(np.linalg.norm(ft['xy'] - BT)), 'pixels': ft['pixels']})
    out['timeline'] = {'steps_checked': len(range(0, T + 1, 2)), 'valid_steps': [x['t'] for x in tl], 'first_grasp_step': grasp, 'valid_before_grasp': sum(1 for x in tl if grasp is None or x['t'] < grasp),
                       'valid_while_holding': sum(1 for x in tl if grasp is not None and x['t'] >= grasp), 'last_valid': tl[-1] if tl else None,
                       'max_error_against_trigger_m': max((x['error_against_trigger_m'] for x in tl), default=None),
                       'target_moved_after_last_valid_m': float(np.linalg.norm(np.array(st[tl[-1]['t']]['cubeB'][:2]) - BT)) if tl else None}
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    FR, OUT = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 4; jobs = []
    for stage in ('shadow', 'pilot'):
        for p in sorted(glob.glob(R + stage + '/scenes/*.json')):
            r = json.load(open(p, encoding='utf-8'))
            if r['triggered']: jobs.append((stage, r['scene'], FR))
    with Pool(n) as p: res = p.map(scene, jobs, chunksize=2)
    def cls(x):
        tl = x['timeline']; return 'never' if not tl['valid_steps'] else ('while_holding' if tl['valid_while_holding'] else 'before_grasp_only')
    chk = [x['check_at_trigger'] for x in res]; nt = [x for x in res if 'timeline' in x and x['v1_decision'] == 'NO_TARGET_ESTIMATE']; fail = [x for x in nt if not x['y']['C0']]; count = lambda L: {k: sum(1 for x in L if cls(x) == k) for k in ('while_holding', 'before_grasp_only', 'never')}
    dx = [c['xy_difference_m'] for c in chk if c['xy_difference_m'] is not None]; seen = [x for x in nt if x['timeline']['last_valid']]
    summary = {'check_against_real_trigger_frames': {'frames': len(chk), 'target_fit_agrees': sum(1 for c in chk if c['real_valid'] == c['reconstructed_valid']), 'both_accepted': len(dx), 'xy_difference_mm_median': round(float(np.median(dx)) * 1000, 2), 'xy_difference_mm_max': round(float(np.max(dx)) * 1000, 2)},
               'scenes_where_v1_had_no_target_at_the_trigger': {'n': len(nt), 'hand_camera_saw_the_target_earlier': count(nt), 'steps_between_last_view_and_trigger_median': float(np.median([x['trigger_step'] - x['timeline']['last_valid']['t'] for x in seen])),
                                                                'target_moved_after_last_view_over_2mm': sum(1 for x in seen if x['timeline']['target_moved_after_last_valid_m'] > 0.002),
                                                                'target_moved_after_last_view_mm_max': round(max(x['timeline']['target_moved_after_last_valid_m'] for x in seen) * 1000, 2)},
               'of_which_uninterrupted_failures': {'n': len(fail), 'hand_camera_saw_the_target_earlier': count(fail), 'by_scene': {str(x['scene']): cls(x) for x in fail}},
               'named_scenes': {str(x['scene']): {'v1_decision': x['v1_decision'], 'trigger_step': x['trigger_step'], 'first_grasp_step': x['timeline']['first_grasp_step'], 'class': cls(x), 'valid_steps_of_those_checked': [len(x['timeline']['valid_steps']), x['timeline']['steps_checked']],
                                                 'first_valid_step': x['timeline']['valid_steps'][0] if x['timeline']['valid_steps'] else None, 'last_valid_step': x['timeline']['valid_steps'][-1] if x['timeline']['valid_steps'] else None,
                                                 'last_view_error_against_trigger_mm': None if not x['timeline']['last_valid'] else round(x['timeline']['last_valid']['error_against_trigger_m'] * 1000, 2),
                                                 'target_moved_after_last_view_mm': None if not x['timeline']['last_valid'] else round(x['timeline']['target_moved_after_last_valid_m'] * 1000, 2)} for x in res if 'timeline' in x and x['scene'] in (50813, 50872, 50817, 50846, 50892)},
               'limits': 'reconstructed hand images (ground-truth states), every second step; the base camera and the held-cube fit are not reconstructed'}
    with open(OUT, 'x', encoding='utf-8') as fo: json.dump({'status': 'OFFLINE; hand images RECONSTRUCTED from ground-truth states, used scenes only; not a perception result on real images', 'summary': summary, 'scenes': res}, fo, indent=1)
    print('written', OUT)
