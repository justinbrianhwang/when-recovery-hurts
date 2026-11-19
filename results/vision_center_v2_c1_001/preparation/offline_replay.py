"""Vision-CENTER v2 Phase C1, free offline step: the DECISION of every arm replayed on the stored frames and traces of the
500 used scenes 50600-51099 (stage B1: 50600-50899, stage B2: 50900-51099). No simulator, no policy, no new scene.

For each triggered scene: every memory is rebuilt from step 0 to the trigger from the stored images of both cameras, the
TCP pose, the finger opening and the executed gripper commands of the stored trace; then each arm decides at the trigger.
For scenes in which v2 moved, v2s is also replayed along the frames of v2's own control steps.
Ground truth (cube positions in the trace) is used only for after-the-fact labels and never reaches a memory or a decision.
A replayed decision is NOT a rollout: it gives no outcome, no rescue and no harm.

    python results/vision_center_v2_c1_001/preparation/offline_replay.py <frames root> <cache dir> <out.json> [processes]
frames root: the folder that holds shadow/frames and pilot/frames (git-ignored server backup of Phase B).
"""
import os
for _k in ('OMP', 'OPENBLAS', 'MKL', 'NUMEXPR', 'BLIS'): os.environ[_k + '_NUM_THREADS'] = '1'
import gzip, json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from hci import vision_center as V1, vision_center_v2 as V2, vision_center_c1 as C1, vision_center_v2s as S

EXE = Path('results/vision_center_v2_001/execution'); STAGES = (('B1', 'shadow', 50600, 50899), ('B2', 'pilot', 50900, 51099))
ARMS = [a for a in C1.ARMS if a not in ('C0', 'CENTER_ORACLE')]
f = lambda x: None if x is None else [float(v) for v in np.asarray(x).reshape(-1)]
cell = lambda r, a: next((x for x in r['cells'] if x['arm'] == a), None)


def obs_at(st, t, hand, base, grip=None):
    s = st; g = (None if s['action'] is None else float(s['action'][-1])) if grip is None else grip
    return C1.StepObs(t, hand, base, np.array(s['tcp'] + s['tcp_q'], float), float(sum(s['finger_qpos'])), g)


def decision_row(est, d, truth):
    A, B = truth; true = B[:2] - A[:2]; e = est.get('offset_xy'); tm = est.get('memory_target')
    return {'decision': d['reason'], 'act': bool(d['act']), 'target_src': est['target_src'], 'hold_src': est['hold_src'], 'u': est['u'], 'offset_norm_m': est['offset_norm_m'], 'gap_m': est['gap_m'], 'target_xy': f(est['target_xy']),
            'memory_reason': est.get('memory_reason'), 'memory_t_last': None if tm is None else tm['t_last'], 'memory_exposed': None if tm is None else tm['exposed_steps'],
            'offset_error_m': None if e is None else float(np.linalg.norm(np.asarray(e) - true)), 'target_error_m': None if est['target_xy'] is None else float(np.linalg.norm(np.asarray(est['target_xy']) - B[:2])),
            'within_bound': None if e is None else bool(np.linalg.norm(np.asarray(e) - true) <= est['u'])}


def scene(job):
    stage, folder, seed, frames_root, cache_dir = job
    rec = json.loads((EXE / folder / 'scenes' / f'{seed}.json').read_text(encoding='utf-8'))
    if not rec['triggered']: return {'stage': stage, 'scene': seed, 'triggered': False}
    with gzip.open(EXE / folder / 'traces' / f'trace_{seed}.json.gz', 'rt', encoding='utf-8') as fh: tr = json.load(fh)['traces']
    st = [rec['initial_state']] + tr['trunk']; T = rec['trigger']['step']; fr = np.load(Path(frames_root) / folder / 'frames' / f'frames_{seed}.npz'); trunk = fr['trunk']; I = V1.IMG
    assert trunk.shape[0] == T + 1 and all(st[t]['t'] == t for t in range(T + 1))
    cp = Path(cache_dir) / f'{seed}.pkl'; cached = pickle.loads(cp.read_bytes()) if cp.exists() else None
    have_orig = bool(cached and 'orig' in cached); orig = pickle.loads(cached['orig']) if have_orig else V2.Memory(); mems = {'FULL': C1.Mem(), **{a: C1.Mem(**kw) for a, kw in C1.MEMORY_VARIANTS.items()}, C1.SAFETY_ARM: S.MemS()}; store = {'trunk': [], 'control': [], 'orig': None}
    truth = lambda t, rows=st: (np.array(rows[t]['cubeA'], float), np.array(rows[t]['cubeB'], float)); B0 = None; so = None
    for t in range(T + 1):
        so = obs_at(st[t], t, trunk[t][:, I:], trunk[t][:, :I])
        if cached is not None: so._c = dict(cached['trunk'][t])                         # fits computed in an earlier run of this script (same images, same functions)
        if not have_orig: orig.update(t, so.hand, so.base, so.pose, so.width, so.grip)   # the original memory computes its own fits (kept in the cache afterwards)
        for m in mems.values(): m.update_obs(so)
        store['trunk'].append(dict(so._c))
    store['orig'] = pickle.dumps(orig)
    out = {'stage': stage, 'scene': seed, 'triggered': True, 'trigger_step': T, 'y': {x['arm']: x['y'] for x in rec['cells']}, 'mirror_equals_original_memory': bool(C1.same_memory(orig, mems['FULL'])),
           'memory_log_equals_record': bool(all(mems['FULL'].log[k] == rec['memory']['log'][k] for k in rec['memory']['log'])),
           'held_grip_equals_record': bool(abs(so.grip - rec['trigger']['held_gripper_command']) < 1e-9), 'width_equals_record': bool(abs(so.width - rec['trigger']['finger_width_m']) < 1e-7)}
    A, B = truth(T); steps_left = rec['trigger']['remaining_budget']; rd = C1.reading_from(so); arms = {}
    gt = {'true_offset_m': float(np.linalg.norm((B - A)[:2])), 'true_gap_m': float(A[2] - V1.CUBE_HALF_M - V1.TARGET_TOP_Z_M)}
    d1 = V1.decide(rd, steps_left); arms['VISION_CENTER_V1'] = {'decision': d1['reason'], 'act': bool(d1['act']), 'offset_norm_m': rd['offset_norm_m'], 'target_src': 'frame' if rd['target_fresh'] else None, 'hold_src': 'frame' if (rd['hold_fit'] is not None and rd['hold_fit']['valid']) else None}
    at = {'VISION_CENTER_V2': orig.copy(), 'V2_SINGLE_FRAME': C1.single_frame_memory(so), **{a: mems[a].copy() for a in C1.MEMORY_VARIANTS}, 'V2_RULES_OFF': mems['FULL'].copy(), C1.SAFETY_ARM: mems[C1.SAFETY_ARM].copy()}
    ests = {}
    for a, m in at.items():
        if a == C1.SAFETY_ARM: S.at_trigger(m, so)
        m.note_hold(T, rd['hold_fit'], so.width); est = V2.estimate(rd, m, so.hand, so.pose, so.width, so.grip); ests[a] = est
        d = C1.decide_rules_off(est, steps_left) if a == 'V2_RULES_OFF' else S.decide(est, steps_left) if a == C1.SAFETY_ARM else V2.decide(est, steps_left); arms[a] = decision_row(est, d, (A, B))
    out['arms'] = arms; out['gt'] = gt; ms = mems[C1.SAFETY_ARM]
    out['v2s_invalidations'] = [{**x, 'target_moved_since_t_last_until_trigger_m': float(np.linalg.norm(truth(T)[1][:2] - truth(x['t_last'])[1][:2])), 'target_moved_since_t_last_until_then_m': float(np.linalg.norm(truth(x['t'])[1][:2] - truth(x['t_last'])[1][:2]))} for x in ms.log['invalidations']]
    # the record of the executed run, for the exactness check
    v1c = cell(rec, 'VISION_CENTER_V1')['control']; v2c = cell(rec, 'VISION_CENTER_V2'); r0 = v2c['control']['readings'][0]; a2 = arms['VISION_CENTER_V2']
    num = lambda x, y: (x is None and y is None) or (x is not None and y is not None and abs(x - y) < 1e-9)
    out['check'] = {'v1_decision_equal': v1c['decision'] == d1['reason'], 'v2_decision_equal': v2c['control']['decision'] == a2['decision'], 'v2_sources_equal': (r0['target_src'], r0['hold_src']) == (a2['target_src'], a2['hold_src']),
                    'v2_offset_equal': num(r0['offset_norm_m'], a2['offset_norm_m']), 'v2_bound_equal': num(r0['u'], a2['u']), 'v2_gap_equal': num(r0['gap_m'], a2['gap_m']), 'recorded': {'v1': v1c['decision'], 'v2': v2c['control']['decision']}}
    out['v2_recorded'] = {'acted': v2c['acted'], 'used_memory': v2c['control']['used_memory'], 'end_reason': v2c['control']['end_reason'], 'steps_used': v2c['control']['steps_used'],
                          'pushed_m': None if not v2c['gt_effect_of_control'] else v2c['gt_effect_of_control']['cubeB_xy_displacement_m'], 'offset_error_m': (v2c['gt_reading_error'] or {}).get('offset_error_m')}
    # v2s along the frames of v2's own control steps (only where both moved): would v2s have stopped earlier, would its command have differed
    if v2c['acted'] and arms[C1.SAFETY_ARM]['act'] and 'VISION_CENTER_V2' in fr.files:
        cf = fr['VISION_CENTER_V2']; rows = tr['VISION_CENTER_V2']; logs = v2c['control']['readings']; K = v2c['control']['steps_used']; grip = rec['trigger']['held_gripper_command']
        m = at[C1.SAFETY_ARM]; mo = at['VISION_CENTER_V2']; est = ests[C1.SAFETY_ARM]; target0 = est['target_xy'].copy(); hold0 = est['h'].copy(); stop = None; dmax = 0.; repl_ok = True
        for i in range(1, min(K, cf.shape[0] - 1) + 1):
            sc = obs_at(rows[i - 1], T + i, cf[i][:, I:], cf[i][:, :I], grip=grip)
            if cached is not None and i - 1 < len(cached['control']): sc._c = dict(cached['control'][i - 1])
            prev_src = est['target_src']; prev_u = est['u']; m.update_obs(sc); rdc = C1.reading_from(sc); m.note_hold(T + i, rdc['hold_fit'], sc.width); mo.note_hold(T + i, rdc['hold_fit'], sc.width)
            eo = V2.estimate(rdc, mo, sc.hand, sc.pose, sc.width, grip); lo = logs[i] if i < len(logs) else None
            if lo is not None and not num(lo['offset_norm_m'], eo['offset_norm_m']): repl_ok = False
            if stop is None:
                why = None
                if rdc['target_fresh'] and float(np.linalg.norm(rdc['target_fit']['xy'] - target0)) > V2.TARGET_JUMP_MAX_M + (0.0 if prev_src == 'frame' else prev_u): why = 'target_moved'
                est = V2.estimate(rdc, m, sc.hand, sc.pose, sc.width, grip)
                if why is None:
                    if est['why'] is not None: why = 'estimate_lost:' + est['why'] + ':' + str(est.get('memory_reason'))
                    elif est['held_xyz'][2] < V2.CARRY_Z_MIN_M: why = 'low_carry'
                    elif float(np.linalg.norm((est['h'] - hold0)[:2])) > V2.HOLD_SHIFT_MAX_M: why = 'hold_shifted'
                if why is not None: stop = {'control_step': i, 'reason': why, 'v2_went_on': bool(i < K or v2c['control']['end_reason'] in ('within_tolerance', 'step_budget', 'target_cap'))}
                elif eo['offset_xy'] is not None and est['offset_xy'] is not None: dmax = max(dmax, float(np.linalg.norm(est['offset_xy'] - eo['offset_xy'])))
            store['control'].append(dict(sc._c))
        v2_stop = v2c['control']['end_reason']
        out['v2s_control'] = {'steps_replayed': min(K, cf.shape[0] - 1), 'v2_control_log_reproduced': repl_ok, 'v2s_stop': stop, 'v2_end_reason': v2_stop, 'max_offset_difference_to_v2_m': dmax,
                              'v2s_stops_where_v2_did_not': bool(stop is not None and not (stop['control_step'] == K and v2_stop == stop['reason'].split(':')[0]))}
    cp.parent.mkdir(parents=True, exist_ok=True); cp.write_bytes(pickle.dumps(store))
    return out


def main():
    frames_root, cache_dir, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); procs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    if out.exists(): raise SystemExit(f'Refused overwrite: {out}')
    jobs = [(st, folder, s, frames_root, cache_dir) for st, folder, lo, hi in STAGES for s in range(lo, hi + 1)]
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as p: rows = p.map(scene, jobs, chunksize=4)
    else: rows = [scene(j) for j in jobs]
    sha = lambda p: __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()
    res = {'work_id': C1.WORK, 'what': 'offline decision replay on used scenes; development data; no rollout, no outcome', 'scenes': len(rows), 'triggered': sum(r['triggered'] for r in rows),
           'code_sha256': {p: sha(p) for p in ('src/hci/vision_center.py', 'src/hci/vision_center_v2.py', 'src/hci/vision_center_c1.py', 'src/hci/vision_center_v2s.py', 'results/vision_center_v2_c1_001/preparation/offline_replay.py')},
           'constants': {'v2_memory': V2.MEMORY, 'v2_decision': V2.DECISION, 'v2s': S.SAFETY, 'single_frame': C1.SINGLE_FRAME}, 'rows': rows}
    out.write_text(json.dumps(res), encoding='utf-8', newline='\n')
    t = [r for r in rows if r['triggered']]; print('triggered', len(t), 'checks', {k: sum(1 for r in t if r['check'][k] is True) for k in t[0]['check'] if k != 'recorded'}, 'mirror', sum(1 for r in t if r['mirror_equals_original_memory'] is True),
                                                   'memlog', sum(r['memory_log_equals_record'] for r in t), 'grip', sum(r['held_grip_equals_record'] for r in t), 'width', sum(r['width_equals_record'] for r in t))


if __name__ == '__main__': main()
