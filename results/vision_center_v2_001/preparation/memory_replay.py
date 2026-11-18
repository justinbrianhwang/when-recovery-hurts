"""Vision-CENTER v2, Phase A: the memory and the decision of hci.vision_center_v2 replayed on STORED full-episode frames.
No simulator, no policy, no new scene. Ground truth (cube poses in stored traces / demonstration states) is the label
that results are scored against and is never passed to the module.

  V  the 21 failed episodes of the consumed diagnostic set (50600-50699) that have full video (both cameras)
  D  every third of the 500 training demonstrations (motion planner; both cameras)
Usage (repository root): python results/vision_center_v2_001/preparation/memory_replay.py <frames dir> <demo .h5> <out.json> [processes]
"""
import os
for _k in ('OMP', 'OPENBLAS', 'MKL', 'NUMEXPR', 'BLIS'): os.environ[_k + '_NUM_THREADS'] = '1'
import glob, gzip, json, re, sys
import numpy as np
sys.path.insert(0, 'src')
from hci import vision_center as V1
from hci import vision_center_v2 as V2
from hci import recovery_v1_oracle as O
DIAG = 'results/recovery_v1_diagnostic/run1'
f = lambda x: None if x is None else [float(v) for v in np.asarray(x).reshape(-1)]


def replay(hand, base, poses, widths, grips, T, steps_left, truth):
    """Memory from step 0 to the trigger step T, then both decisions at T. `truth(i)` -> (A, B) is used for labels only."""
    mem = V2.Memory(); tl = []
    for t in range(T + 1):
        last = None if t == 0 else float(grips[t - 1]); o = mem.update(t, hand[t], base[t], poses[t], widths[t], last); tr = mem.target
        tl.append({'t': t, 'hand': o['hand'], 'base': o['base'], 'hold': o['hold'], 'mem_n': None if tr is None else tr['n'], 'mem_src': None if tr is None else tr['src'], 'exposed': None if tr is None else tr['exposed'],
                   'mem_error_now_m': None if tr is None else float(np.linalg.norm(tr['xy'] - truth(t)[1][:2])), 'hold_mem': mem.hold is not None})
    last = float(grips[T - 1]); rd = V1.reading(hand[T], poses[T]); d1 = V1.decide(rd, steps_left); mem.note_hold(T, rd['hold_fit'], widths[T])
    est = V2.estimate(rd, mem, hand[T], poses[T], widths[T], last); d2 = V2.decide(est, steps_left); A, B = truth(T); true = B[:2] - A[:2]; n = float(np.linalg.norm(true))
    R = V1.quat_R(poses[T][3:7]); hold_true = R.T @ (A - poses[T][:3])
    row = {'trigger_step': T, 'v1': d1['reason'], 'v2': d2['reason'], 'v2_est': V2.summary(est), 'memory_log': dict(mem.log), 'true_offset_m': n, 'true_gap_m': float(A[2] - 0.06),
           'v2_offset_error_m': None if est['offset_xy'] is None else float(np.linalg.norm(est['offset_xy'] - true)), 'v2_target_error_m': None if est['target_xy'] is None else float(np.linalg.norm(est['target_xy'] - B[:2])),
           'v2_hold_error_xy_m': None if est['h'] is None else float(np.linalg.norm((est['h'] - hold_true)[:2])), 'v2_error_within_bound': None if est['offset_xy'] is None else bool(np.linalg.norm(est['offset_xy'] - true) <= est['u']),
           'v2_unnecessary_act': bool(d2['act'] and n <= V2.RESIDUAL_TOL_M), 'v1_offset_error_m': None if rd['offset_xy'] is None else float(np.linalg.norm(rd['offset_xy'] - true)),
           'target_moved_since_memory_m': None if mem.target is None else float(np.linalg.norm(truth(mem.target['t_last'])[1][:2] - B[:2])), 'timeline': tl}
    return row


def video(seed):
    with gzip.open(DIAG + f'/traces/trace_{seed}.json.gz', 'rt', encoding='utf-8') as fh: t = json.load(fh)
    st = [t['initial']] + t['rows']; acts = [r['action'] for r in t['rows']]; T = None
    for i in range(1, len(acts)):
        if O.trigger_rule(acts[i][3], O.close_run(acts[:i]), sum(st[i]['finger_qpos'])): T = i; break
    if T is None: return {'seed': seed, 'triggered': False}
    fr = np.load(FRAMES + f'/frames_{seed}.npz')['frames']; poses = [np.array(s['tcp'] + s['tcp_q'], float) for s in st]; widths = [float(sum(s['finger_qpos'])) for s in st]
    row = replay(fr[:, :, V1.IMG:], fr[:, :, :V1.IMG], poses, widths, [a[3] for a in acts], T, 200 - T, lambda i: (np.array(st[i]['cubeA']), np.array(st[i]['cubeB'])))
    row.update(seed=seed, triggered=True); return row


def demo(k):
    import h5py
    with h5py.File(DEMOS, 'r') as fh:
        g = fh[k]; acts = g['actions'][:]; q = g['obs/agent/qpos'][:]; T = None
        for i in range(1, len(acts)):
            if O.trigger_rule(float(acts[i][3]), O.close_run([a for a in acts[:i]]), float(q[i][-2] + q[i][-1])): T = i; break
        if T is None: return {'demo': k, 'triggered': False}
        tp = g['obs/extra/tcp_pose'][:].astype(float); A = g['env_states/actors/cubeA'][:, :3].astype(float); B = g['env_states/actors/cubeB'][:, :3].astype(float)
        row = replay(g['obs/sensor_data/hand_camera/rgb'][:T + 1], g['obs/sensor_data/base_camera/rgb'][:T + 1], tp, (q[:, -2] + q[:, -1]).astype(float), acts[:, 3], T, len(acts) - T + 100, lambda i: (A[i], B[i]))
    row.update(demo=k, triggered=True); del row['timeline']; return row


def job(a):
    global FRAMES, DEMOS
    FRAMES, DEMOS = a[2], a[3]; return video(a[1]) if a[0] == 'V' else demo(a[1])


if __name__ == '__main__':
    FRAMES, DEMOS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    import h5py
    from multiprocessing import Pool
    with h5py.File(DEMOS, 'r') as fh: ks = sorted(fh.keys(), key=lambda k: int(k.split('_')[1]))[::3]
    seeds = sorted(int(re.findall(r'(\d+)\.npz', p)[0]) for p in glob.glob(FRAMES + '/frames_*.npz'))
    with Pool(n) as p: res = p.map(job, [('V', s, FRAMES, DEMOS) for s in seeds] + [('D', k, FRAMES, DEMOS) for k in ks], chunksize=1)
    mm = lambda a: None if len(a) == 0 else {'n': len(a), 'median': round(float(np.median(a)) * 1000, 2), 'p90': round(float(np.percentile(a, 90)) * 1000, 2), 'max': round(float(np.max(a)) * 1000, 2)}
    count = lambda L, k: {v: sum(1 for x in L if x[k] == v) for v in sorted({x[k] for x in L})}

    def part(L):
        T = [x for x in L if x['triggered']]; src = lambda x: f"target_{x['v2_est']['target_src']}/hold_{x['v2_est']['hold_src']}"; est = [x for x in T if x['v2_offset_error_m'] is not None]
        return {'episodes': len(L), 'triggered': len(T), 'v1_decisions': count(T, 'v1'), 'v2_decisions': count(T, 'v2'), 'v2_estimate_sources': {k: sum(1 for x in est if src(x) == k) for k in sorted({src(x) for x in est})},
                'v2_offset_error_mm': mm([x['v2_offset_error_m'] for x in est]), 'v2_offset_error_mm_remembered_target': mm([x['v2_offset_error_m'] for x in est if x['v2_est']['target_src'] != 'frame']),
                'v2_estimates_outside_their_bound': sum(1 for x in est if not x['v2_error_within_bound']), 'v2_unnecessary_acts': sum(x['v2_unnecessary_act'] for x in T), 'v2_acts': sum(1 for x in T if x['v2'] == 'ACT'), 'v1_acts': sum(1 for x in T if x['v1'] == 'ACT'),
                'v1_no_target_then_v2': count([x for x in T if x['v1'] == 'NO_TARGET_ESTIMATE'], 'v2'), 'v1_no_hold_then_v2': count([x for x in T if x['v1'] == 'NO_HOLD_ESTIMATE'], 'v2'), 'memory_conflicts': sum(x['memory_log']['conflicts'] for x in T)}
    summary = {'video_failed_diagnostic_episodes': part(res[:len(seeds)]), 'demonstration_releases': part(res[len(seeds):]),
               'note': 'development data: every constant of v2 was chosen with these frames in view. The video episodes are all uninterrupted FAILURES; no outcome of a v2 move exists (no simulator was run)'}
    with open(OUT, 'x', encoding='utf-8') as fo: json.dump({'status': 'OFFLINE, stored frames only; ground truth is a label, never an input', 'memory': V2.MEMORY, 'decision': V2.DECISION, 'summary': summary,
                                                           'video_failed_diagnostic_episodes': res[:len(seeds)], 'demonstration_releases': res[len(seeds):]}, fo, indent=1)
    print('written', OUT)
else:
    FRAMES = DEMOS = None
