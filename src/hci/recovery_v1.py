"""Recovery-V1 diagnostic: why does the Vision-V1 base policy still fail? (StackCube-v1 only, no intervention)

One fixed policy is run without any intervention. After every control step the simulator's ground truth (cube poses and
velocities, finger opening, contact forces, the task's own flags) is written to a trace. From the trace each failed
episode gets ONE mechanism label by the fixed rule list below, together with the numbers the label rests on.

Everything read here is LOGGING ONLY. Nothing in this module may be passed to a policy or a monitor.
The rules and thresholds were written before any Recovery-V1 episode was run. Where no rule applies the label is UNKNOWN.
"""
from __future__ import annotations
import math
import numpy as np
from .diagnostics import LIFT_M, NEAR_M, stage_events

CUBE_M = 0.04                                   # cube edge; cubeA rests on cubeB when its centre is 0.04 m higher
XY_TOL_M = math.hypot(0.02, 0.02) + 0.005       # the task's own tolerance (StackCube-v1 evaluate), 0.0333 m
Z_TOL_M = 0.005                                 # the task's own tolerance on the height difference
STATIC_LIN = 1e-2; STATIC_ANG = 0.5             # the task's own "static" thresholds (m/s, rad/s)
RELEASE_GAP_STEPS = 3                           # a grasp counts as ended after this many consecutive steps without the grasp flag
OPEN_WINDOW_STEPS = 5                           # an open command within this many steps up to the loss of the grasp makes it a commanded release
IN_PLACE_STEPS = 3                              # steps with the in-place flag while grasped that make "was in place but did not let go"
CUBE_B_MOVED_M = 0.01                           # secondary flag: cubeB pushed more than this from its initial xy
SETS = {'recovery_v1_diag': (50600, 50699), 'recovery_v1_pilot_reserved': (50700, 50799)}
RUNNABLE = ('recovery_v1_diag',)                # the reserved pilot range is declared so that nothing else takes it; this work never runs it

LABELS = {
    'SUCCESS': 'task success',
    'NO_APPROACH': 'the gripper never came within 3 cm of cubeA',
    'NO_GRASP': 'near cubeA, grasp flag never true',
    'GRASP_NOT_LIFTED': 'grasp flag true, cubeA never lifted 1 cm while grasped',
    'NOT_RELEASED_AT_PLACE': 'cubeA was in the success pose on cubeB while still grasped and the episode ended with the grasp still on (did not let go)',
    'HOLDING_NOT_PLACED': 'episode ended still holding cubeA, which was never in the success pose during that hold',
    'GRASP_LOST_WITHOUT_OPEN_COMMAND': 'the lifted cube left the grasp while the gripper command was still "close" (slip or knock)',
    'RELEASE_OFF_TARGET_XY': 'commanded release with cubeA outside the horizontal tolerance over cubeB (position error)',
    'EARLY_RELEASE_ABOVE': 'commanded release horizontally over cubeB but higher than the resting height (opened too early, dropped)',
    'RELEASE_BELOW_REST_HEIGHT': 'commanded release horizontally over cubeB but lower than the resting height (pressed in or tilted)',
    'POST_RELEASE_UNSTABLE': 'released in the success pose, then cubeA left that pose (instability after contact)',
    'IN_PLACE_NOT_STATIC': 'released in the success pose and stayed there, never static at a released step',
    'UNKNOWN': 'no rule applies; the cause is not established by the log',
}
RULE_ORDER = ['SUCCESS', 'NO_APPROACH', 'NO_GRASP', 'GRASP_NOT_LIFTED', 'NOT_RELEASED_AT_PLACE', 'HOLDING_NOT_PLACED', 'GRASP_LOST_WITHOUT_OPEN_COMMAND', 'RELEASE_OFF_TARGET_XY',
              'EARLY_RELEASE_ABOVE', 'RELEASE_BELOW_REST_HEIGHT', 'POST_RELEASE_UNSTABLE', 'IN_PLACE_NOT_STATIC', 'UNKNOWN']
THRESHOLDS = {'near_m': NEAR_M, 'lift_m': LIFT_M, 'xy_tol_m': XY_TOL_M, 'z_tol_m': Z_TOL_M, 'release_gap_steps': RELEASE_GAP_STEPS, 'open_window_steps': OPEN_WINDOW_STEPS,
              'in_place_steps': IN_PLACE_STEPS, 'cube_b_moved_m': CUBE_B_MOVED_M, 'static_lin_m_s': STATIC_LIN, 'static_ang_rad_s': STATIC_ANG}


# ---------------------------------------------------------------- seeds
def diag_seeds(manifest, name, used):
    """Seeds of the diagnostic set after checking the frozen ranges and that no seed has a recorded use. The reserved pilot range is refused."""
    if name not in RUNNABLE: raise KeyError(f'{name} is not runnable in this work (the pilot range 50700-50799 is reserved and unused)')
    table = manifest['splits']
    if set(table) != set(SETS): raise ValueError('the Recovery-V1 manifest must hold exactly the declared sets')
    for k, (lo, hi) in SETS.items():
        if list(table[k]) != list(range(lo, hi + 1)): raise ValueError(f'{k} is not the frozen range {lo}-{hi}')
    own = {s for v in table.values() for s in v}
    if len(own) != sum(len(v) for v in table.values()): raise ValueError('the declared sets overlap')
    clash = {src: sorted(set(ids) & own)[:10] for src, ids in used.items()}; clash = {k: v for k, v in clash.items() if v}
    if clash: raise ValueError(f'new seeds have a recorded use: {clash}. The manifest is not patched silently')
    return list(table[name])


# ---------------------------------------------------------------- trace (simulator side)
def probe(u, t, action=None):
    """Ground-truth snapshot after control step t. Extends hci.diagnostics.step_state. Logging only."""
    import torch
    from .diagnostics import step_state
    from .trees import as_numpy
    row = step_state(u, t, action); v = lambda x: [float(z) for z in as_numpy(x).reshape(-1)]
    force = lambda a, b: float(torch.linalg.norm(u.scene.get_pairwise_contact_forces(a, b), dim=-1).reshape(-1)[0])
    ag = u.agent; A = u.cubeA; B = u.cubeB
    row.update(tcp_q=v(ag.tcp.pose.q), finger_qpos=v(ag.robot.get_qpos()[0, -2:]), cubeA_q=v(A.pose.q), cubeB_q=v(B.pose.q),
               cubeA_v=v(A.linear_velocity), cubeA_w=v(A.angular_velocity), cubeB_v=v(B.linear_velocity), cubeB_w=v(B.angular_velocity),
               f_A_B=force(A, B), f_A_left=force(ag.finger1_link, A), f_A_right=force(ag.finger2_link, A), f_A_table=force(A, u.table_scene.table),
               f_B_left=force(ag.finger1_link, B), f_B_right=force(ag.finger2_link, B))
    return row


# ---------------------------------------------------------------- derived quantities (pure)
def tilt_deg(q):
    """Angle between the nearest cube face normal and the vertical (0 = flat on a face), quaternion w, x, y, z."""
    w, x, y, z = q; n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0; w, x, y, z = w / n, x / n, y / n, z / n
    up = max(abs(2 * (x * z - w * y)), abs(2 * (y * z + w * x)), abs(1 - 2 * (x * x + y * y)))
    return math.degrees(math.acos(min(1.0, up)))


def derived(r):
    A, B = r['cubeA'], r['cubeB']
    return {'xy': math.hypot(A[0] - B[0], A[1] - B[1]), 'h': A[2] - B[2] - CUBE_M, 'grip': None if r['action'] is None else r['action'][-1],
            'width': sum(r['finger_qpos']) if 'finger_qpos' in r else None}


def holds(rows):
    """Maximal runs of the grasp flag, merged over gaps shorter than RELEASE_GAP_STEPS. [(first index, last index)]"""
    out = []
    for i, r in enumerate(rows):
        if not r['is_cubeA_grasped']: continue
        if out and i - out[-1][1] - 1 < RELEASE_GAP_STEPS: out[-1][1] = i
        else: out.append([i, i])
    return [tuple(x) for x in out]


def classify(initial, rows):
    """One label for one episode and the evidence for it. rows: probe() outputs after each executed step, in order."""
    ev = stage_events(initial, rows) if rows else {'furthest_stage': 'unreadable'}
    if not rows: return {'label': 'UNKNOWN', 'evidence': {'why': 'empty trace'}, 'events': ev, 'flags': {}}
    z0 = initial['cubeA'][2]; d = [derived(r) for r in rows]; lifted = [r['cubeA'][2] - z0 >= LIFT_M for r in rows]; n = len(rows); T = lambda i: rows[i]['t']
    seg = holds(rows); carried = [(a, b) for a, b in seg if any(lifted[i] for i in range(a, b + 1))]
    grip = [x['grip'] for x in d]; first_grasp = seg[0][0] if seg else None
    open_after = [T(i) for i in range(first_grasp or 0, n) if first_grasp is not None and grip[i] is not None and grip[i] > 0]
    b_moved = max(math.hypot(r['cubeB'][0] - initial['cubeB'][0], r['cubeB'][1] - initial['cubeB'][1]) for r in rows)
    ev.update(first_open_command_after_grasp_step=open_after[0] if open_after else None, open_command_steps_after_grasp=len(open_after), holds=[[T(a), T(b)] for a, b in seg],
              carried_holds=len(carried), first_in_place_step=next((r['t'] for r in rows if r['is_cubeA_on_cubeB']), None),
              in_place_steps=sum(1 for r in rows if r['is_cubeA_on_cubeB']), final_xy_offset_m=d[-1]['xy'], final_height_gap_m=d[-1]['h'],
              final_finger_width_m=d[-1]['width'], final_tilt_deg=tilt_deg(rows[-1]['cubeA_q']) if 'cubeA_q' in rows[-1] else None)
    flags = {'cubeB_moved': b_moved > CUBE_B_MOVED_M, 'cubeB_max_xy_displacement_m': b_moved, 'holds': len(seg), 'carried_holds': len(carried)}

    def done(label, **evidence):
        return {'label': label, 'meaning': LABELS[label], 'evidence': evidence, 'events': ev, 'flags': flags}
    if ev.get('first_success_step') is not None: return done('SUCCESS', success_step=ev['first_success_step'])
    if not seg:
        if ev['first_near_step'] is None: return done('NO_APPROACH', min_tcp_to_cubeA_m=ev['min_tcp_to_cubeA_m'])
        return done('NO_GRASP', first_near_step=ev['first_near_step'], close_command_steps=ev['close_command_steps'], min_tcp_to_cubeA_m=ev['min_tcp_to_cubeA_m'])
    if not carried:
        return done('GRASP_NOT_LIFTED', first_grasp_step=T(seg[0][0]), grasp_steps=ev['grasp_steps'], max_cubeA_lift_m=ev['max_cubeA_lift_m'], final_is_grasped=ev['final_is_grasped'])
    a, b = seg[-1]
    if b < n - 1 and n - 1 - b < RELEASE_GAP_STEPS:                       # the grasp flag went off in the last steps: neither a hold nor a release is established
        return done('UNKNOWN', why='the grasp flag was lost fewer than RELEASE_GAP_STEPS steps before the end of the episode', last_grasped_step=T(b), steps_without_grasp_at_end=n - 1 - b,
                    gripper_commands_at_end=grip[b:], xy_offset_at_last_grasped_step_m=d[b]['xy'], height_gap_at_last_grasped_step_m=d[b]['h'])
    if b == n - 1:                                                        # the episode ended inside a hold
        idx = range(a, b + 1); place = [i for i in idx if rows[i]['is_cubeA_on_cubeB']]; opens = [T(i) for i in idx if grip[i] is not None and grip[i] > 0]
        common = dict(hold_steps=[T(a), T(b)], hold_included_a_lift=any(lifted[i] for i in idx), open_command_steps_in_hold=len(opens), first_open_command_step_in_hold=opens[0] if opens else None,
                      min_xy_offset_m=min(d[i]['xy'] for i in idx), final_xy_offset_m=d[b]['xy'], final_height_gap_m=d[b]['h'], final_finger_width_m=d[b]['width'], final_gripper_command=grip[b])
        if len(place) >= IN_PLACE_STEPS:
            return done('NOT_RELEASED_AT_PLACE', in_place_steps_while_grasped=len(place), first_in_place_step=T(place[0]), last_in_place_step=T(place[-1]),
                        static_steps_in_place=sum(1 for i in place if rows[i]['is_cubeA_static']), **common)
        return done('HOLDING_NOT_PLACED', in_place_steps_while_grasped=len(place), min_abs_height_gap_while_within_xy_m=min([abs(d[i]['h']) for i in idx if d[i]['xy'] <= XY_TOL_M], default=None),
                    steps_within_xy_tol=sum(1 for i in idx if d[i]['xy'] <= XY_TOL_M), **common)
    a, b = carried[-1]                                                    # last hold that carried the cube; the grasp ended after step index b
    if seg[-1] != (a, b): flags['regrasp_without_lift_after_last_carry'] = True
    win = range(max(0, b - OPEN_WINDOW_STEPS + 2), min(n, b + 2)); opened = [T(i) for i in win if grip[i] is not None and grip[i] > 0]
    # the release pose is measured at the LAST GRASPED step b (where the cube was when it was still held); the first step without the grasp flag is reported next to it
    at = dict(release_step=T(b), release_measured_at='last grasped step', xy_offset_at_release_m=d[b]['xy'], height_gap_at_release_m=d[b]['h'], cubeA_lift_at_release_m=rows[b]['cubeA'][2] - z0,
              in_place_at_release=bool(rows[b]['is_cubeA_on_cubeB']), first_step_without_grasp=T(b + 1), xy_offset_at_first_step_without_grasp_m=d[b + 1]['xy'],
              height_gap_at_first_step_without_grasp_m=d[b + 1]['h'], in_place_at_first_step_without_grasp=bool(rows[b + 1]['is_cubeA_on_cubeB']),
              gripper_commands_around_release=[grip[i] for i in win], f_A_B_at_release=rows[b].get('f_A_B'),
              tilt_at_release_deg=tilt_deg(rows[b]['cubeA_q']) if 'cubeA_q' in rows[b] else None, final_xy_offset_m=d[-1]['xy'], final_height_gap_m=d[-1]['h'])
    if not opened: return done('GRASP_LOST_WITHOUT_OPEN_COMMAND', **at)
    if d[b]['xy'] > XY_TOL_M: return done('RELEASE_OFF_TARGET_XY', **at)
    if d[b]['h'] > Z_TOL_M: return done('EARLY_RELEASE_ABOVE', **at)
    if d[b]['h'] < -Z_TOL_M: return done('RELEASE_BELOW_REST_HEIGHT', **at)
    after = range(b + 1, n); left = [i for i in after if not rows[i]['is_cubeA_on_cubeB']]
    post = dict(steps_after_release=len(after), first_step_out_of_place=T(left[0]) if left else None, out_of_place_steps=len(left),
                finger_contact_steps_after_release=sum(1 for i in after if max(rows[i].get('f_A_left', 0.), rows[i].get('f_A_right', 0.)) > 0),
                max_cubeA_speed_after_release_m_s=max((float(np.linalg.norm(rows[i]['cubeA_v'])) for i in after if 'cubeA_v' in rows[i]), default=None),
                static_released_in_place_steps=sum(1 for i in after if rows[i]['is_cubeA_on_cubeB'] and rows[i]['is_cubeA_static'] and not rows[i]['is_cubeA_grasped']))
    if left: return done('POST_RELEASE_UNSTABLE', **at, **post)
    if post['static_released_in_place_steps'] == 0 and len(after) > 0: return done('IN_PLACE_NOT_STATIC', **at, **post)
    return done('UNKNOWN', why='released in the success pose and stayed there; no rule explains the missing success flag', **at, **post)


def summarize(episodes):
    """episodes: [{'seed', 'success_once', 'label', 'furthest_stage', ...}]. Counts only; no rule is applied here."""
    n = len(episodes); fails = [e for e in episodes if not e['success_once']]; by = {}
    for e in fails: by.setdefault(e['label'], []).append(e['seed'])
    order = sorted(by, key=lambda k: (-len(by[k]), RULE_ORDER.index(k)))
    return {'n': n, 'success_once': n - len(fails), 'failures': len(fails), 'failure_labels': {k: {'count': len(by[k]), 'seeds': by[k], 'meaning': LABELS[k]} for k in order},
            'largest_label': order[0] if order else None, 'largest_is_tied': len(order) > 1 and len(by[order[0]]) == len(by[order[1]]),
            'unknown': len(by.get('UNKNOWN', [])), 'thresholds': THRESHOLDS}


def projected_cost(*, seconds_per_episode, usd_per_hour, spent_usd, cap_usd, episodes=100, fixed_seconds=900, reserve_usd=0.20, margin=1.25):
    """All-in projection for the one diagnostic pass. Inputs are time and money only."""
    hours = (episodes * seconds_per_episode + fixed_seconds) * margin / 3600; total = spent_usd + hours * usd_per_hour + reserve_usd
    return {'projected_remaining_hours': hours, 'projected_total_usd': total, 'cap_usd': cap_usd, 'fits_cap': bool(total <= cap_usd),
            'inputs': {'seconds_per_episode': seconds_per_episode, 'usd_per_hour': usd_per_hour, 'spent_usd': spent_usd, 'episodes': episodes, 'fixed_seconds': fixed_seconds,
                       'reserve_usd': reserve_usd, 'margin': margin}, 'rule': 'if the projection exceeds the cap the diagnostic pass does not start; the number of scenes is not reduced'}
