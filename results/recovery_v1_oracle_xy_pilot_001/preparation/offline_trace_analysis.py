"""Offline evidence for the oracle XY pilot, from the 100 stored traces of the CONSUMED diagnostic set (50600-50699).
No simulator, no policy. Development data: the trigger thresholds were chosen with this file's trigger section in view.
Usage (repository root): python results/recovery_v1_oracle_xy_pilot_001/preparation/offline_trace_analysis.py results/recovery_v1_diagnostic/run1 <out.json>
"""
import gzip, json, glob, math, sys
import numpy as np
sys.path.insert(0, 'src')
from hci import recovery_v1_oracle as O
D, OUT = sys.argv[1], sys.argv[2]
eps = {}
for line in open(D + '/episodes.jsonl', encoding='utf-8'):
    e = json.loads(line); eps[e['seed']] = e
track = []; trig = []
for p in sorted(glob.glob(D + '/traces/trace_*.json.gz')):
    with gzip.open(p, 'rt', encoding='utf-8') as f: t = json.load(f)
    st = [t['initial']] + t['rows']; acts = [r['action'] for r in t['rows']]; z0 = t['initial']['cubeA'][2]
    for i, a in enumerate(acts):                                   # commanded increment (0.1 m per unit) against the TCP displacement of that step
        cmd = np.array(a[:3]) * 0.1; got = np.array(st[i + 1]['tcp']) - np.array(st[i]['tcp']); carrying = st[i]['is_cubeA_grasped'] and st[i + 1]['is_cubeA_grasped']
        track.append((*cmd, *got, carrying))
    fired = None; events = 0
    for i in range(1, len(acts)):                                  # state i, pending action acts[i], executed actions acts[:i]
        if acts[i][3] > 0 and acts[i - 1][3] < 0: events += 1
        if fired is None and O.trigger_rule(acts[i][3], O.close_run(acts[:i]), sum(st[i]['finger_qpos'])):
            A, B = st[i]['cubeA'], st[i]['cubeB']
            fired = {'step': i, 'close_run': O.close_run(acts[:i]), 'finger_width_m': sum(st[i]['finger_qpos']), 'gt_grasped': bool(st[i]['is_cubeA_grasped']), 'gt_lift_m': A[2] - z0,
                     'gt_offset_xy_m': math.hypot(A[0] - B[0], A[1] - B[1]), 'on_stage2_grid': i % 4 == 0 and i % 8 != 0, 'steps_left': 200 - i}
    trig.append({'seed': t['seed'], 'success_once': eps[t['seed']]['success_once'], 'label': eps[t['seed']]['label'], 'steps': len(acts), 'open_after_close_events': events, 'trigger': fired})
T = np.array(track, float)


def gain(mask, ax):
    c = T[mask, ax]; g = T[mask, 3 + ax]; big = np.abs(c) > 0.002; r = g[big] / c[big]
    return {'n': int(big.sum()), 'least_squares': float(c[big] @ g[big] / (c[big] @ c[big])), 'median': float(np.median(r)), 'p10': float(np.percentile(r, 10)), 'p90': float(np.percentile(r, 90))}


carry = T[:, 6] == 1
fired = [x for x in trig if x['trigger']]; off = np.array([x['trigger']['gt_offset_xy_m'] for x in fired]); y = np.array([x['success_once'] for x in fired])


def predict(e0, g, reading):
    """Linear tracking model: each step achieves g x the commanded increment; the carried cube moves with the gripper; cubeB does not move."""
    e = e0; moved = 0.; cmd = 0.; n = 0
    while n < O.MAX_STEPS and e > O.RESIDUAL_TOL_M:
        c = min(O.STEP_CMD_M, e); c = min(c, O.TARGET_CAP_M - moved) if reading == 'target' else min(c, O.TARGET_CAP_M - cmd)
        if c <= 1e-9: break
        cmd += c; e -= g * c; moved += g * c; n += 1
    return e, n


pred = {}
for reading in ('target', 'sum_of_commands'):
    for g in (0.31, 0.39, 0.47):
        res = [predict(x, g, reading) for x in off]; r = np.array([a for a, _ in res]); n = np.array([b for _, b in res]); f = y == 0
        most = O.MAX_STEPS * O.STEP_CMD_M * g if reading == 'target' else O.TARGET_CAP_M * g
        pred[f'{reading}_gain_{g}'] = {'scenes_moved': int((n > 0).sum()), 'scenes_already_within_tolerance': int((n == 0).sum()), 'residual_within_1.0cm': int((r <= 0.0101).sum()),
                                       'residual_within_1.5cm': int((r <= 0.015).sum()), 'c0_failures_among_triggered': int(f.sum()), 'c0_failures_residual_within_1.0cm': int((r[f] <= 0.0101).sum()),
                                       'c0_failures_residual_within_1.5cm': int((r[f] <= 0.015).sum()), 'steps_median_when_moved': float(np.median(n[n > 0])), 'steps_max': int(n.max()), 'max_correction_m': float(min(most, O.TARGET_CAP_M))}
bins = [(0, .010), (.010, .015), (.015, .020), (.020, .025), (.025, .033), (.033, 9)]
steps = [x['trigger']['step'] for x in fired]
out = {'source': 'results/recovery_v1_diagnostic/run1 (100 stored traces of the consumed diagnostic set; development data)', 'not': 'no simulator step and no policy inference were run for this file',
       'tracking': {'what': "achieved TCP displacement of one control step divided by the commanded increment (action x 0.1 m), steps with |command| > 2 mm, the policy's own motions",
                    'all_steps': {a: gain(np.ones(len(T), bool), i) for i, a in enumerate('xyz')}, 'while_carrying': {a: gain(carry, i) for i, a in enumerate('xyz')},
                    'reading': 'about 0.39 of each commanded increment is achieved per control step (the controller re-targets from the measured pose every step). A 5 mm command moves the gripper about 2 mm. '
                               'This agrees with the existing 5 cm retreat needing 23-27 steps.',
                    'consequence': {'target_cap_reading': 'at most 12 x 5 mm x 0.39 = about 2.3 cm of real correction; the 12-step limit binds before the 3 cm target cap',
                                    'sum_of_commands_reading': 'at most 3 cm x 0.39 = about 1.2 cm of real correction, after 6 steps'}},
       'trigger_on_consumed_traces': {'rule': O.TRIGGER, 'fires_in': len(fired), 'of': len(trig), 'not_grasped_at_trigger': sum(1 for x in fired if not x['trigger']['gt_grasped']),
                                      'not_lifted_1cm_at_trigger': sum(1 for x in fired if x['trigger']['gt_lift_m'] < 0.01), 'no_trigger': [{'seed': x['seed'], 'label': x['label']} for x in trig if not x['trigger']],
                                      'c0_success_among_triggered': int(y.sum()), 'c0_failure_among_triggered': int((1 - y).sum()),
                                      'trigger_step': {'min': int(min(steps)), 'median': float(np.median(steps)), 'max': int(max(steps))},
                                      'on_stage2_eligibility_grid': sum(1 for x in fired if x['trigger']['on_stage2_grid']), 'fewer_than_48_steps_left': sum(1 for x in fired if x['trigger']['steps_left'] < 48),
                                      'fewer_than_12_steps_left': sum(1 for x in fired if x['trigger']['steps_left'] < 12),
                                      'offset_at_trigger_vs_outcome': [{'offset_cm': [lo * 100, hi * 100 if hi < 1 else None], 'scenes': int(((off > lo) & (off <= hi)).sum()), 'c0_failures': int((1 - y[(off > lo) & (off <= hi)]).sum())} for lo, hi in bins],
                                      'without_the_width_and_run_conditions': 'the first open-after-close command comes with empty fingers in 2 episodes (50610, 50626) and after a 1-step close in 1 (50636); the width window and the run length exclude these'},
       'predicted_correction': {'model': predict.__doc__, 'by_cap_reading_and_gain': pred, 'caveat': 'a prediction about the controller, not about task success; the policy re-plans afterwards and may move again before it opens'},
       'episode_length': {'mean_steps': float(np.mean([x['steps'] for x in trig])), 'mean_trigger_step': float(np.mean(steps))},
       'per_scene': trig}
json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'per_scene'}, indent=1))
