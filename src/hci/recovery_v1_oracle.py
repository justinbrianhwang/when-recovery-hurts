"""Recovery-V1 oracle XY pilot (work id recovery_v1_oracle_xy_pilot_001): centre before release, as a diagnostic reference.

Four continuations from ONE runtime state, captured just before the frozen policy's first pending release command:

  C0             the pending action is executed, queue and history untouched (must reproduce the uninterrupted trunk)
  R5             the existing 5 cm upward retreat (Runtime.retreat, unchanged), queue cleared, policy re-plans
  HOLD_MATCHED   queue cleared, gripper command held, TCP held at its XY for exactly the number of control steps
                 CENTER_ORACLE used on this state, then the policy re-plans
  CENTER_ORACLE  queue cleared, gripper command held, bounded horizontal move that removes the offset between the two
                 cube centres, then the policy re-plans

PRIVILEGED INFORMATION. `oracle_offset` is the only function in this module whose result reaches a controller. It reads
the simulator's exact cube positions and may be called only while an OracleGate is open, which happens only inside the
control part of the CENTER_ORACLE arm. The policy, the trigger, C0, R5 and HOLD_MATCHED never receive object poses.
CENTER_ORACLE is therefore NOT deployable; it is a reference arm (amendments/AMENDMENT_014, draft). Everything passed
in as `probe` is ground truth for LOGGING ONLY, as in hci.recovery_v1.

The trigger, every bound and every pass criterion below were fixed before any pilot scene was run. The trigger
thresholds were chosen on the stored traces of the consumed diagnostic set (50600-50699), which is development data.
"""
from __future__ import annotations
import math, time
import numpy as np
from .trees import clone, as_numpy
from .pilot import runtime_tree, compare_tree, obs_diff, live_obs_ok, jsonable, tree_hash, _same_run, _ref
from .pilot2b import Hooks
from .pilot2d import prefix_equal

WORK = 'recovery_v1_oracle_xy_pilot_001'
ROLE = 'recovery_v1_oracle_xy_pilot_dev'
ARMS = ('C0', 'R5', 'HOLD_MATCHED', 'CENTER_ORACLE')
CONTROL = ARMS[1:]
INTERVENTION = {'C0': None, 'R5': 2, 'HOLD_MATCHED': 'hold_matched_xy', 'CENTER_ORACLE': 'center_oracle_xy', 'TRACKING_PROBE': 'tracking_probe_xy'}
SET = 'recovery_v1_pilot_reserved'; RANGE = (50700, 50799)
REPEAT = 0                                      # one policy-noise repeat per scene

# --- trigger: the policy's own pending action, its past gripper commands and the measured finger opening (all observable)
OPEN_CMD = 0.0                                  # gripper command > 0 = open, < 0 = close (pinned Panda config: -1 -> -0.01 m, +1 -> 0.04 m per finger)
CLOSE_RUN_MIN = 3                               # at least this many consecutive executed close commands immediately before
WIDTH_MIN_M = 0.02; WIDTH_MAX_M = 0.06          # finger opening (sum of both finger joints): closed on something, neither empty (0) nor open (0.08)
# --- CENTER_ORACLE bounds
RESIDUAL_TOL_M = 0.010                          # stop when the horizontal offset between the cube centres is at most this
STEP_CMD_M = 0.005                              # commanded horizontal increment per control step, at most
TARGET_CAP_M = 0.030                            # the commanded TCP target never leaves this horizontal distance from the TCP at the trigger
MAX_STEPS = 12                                  # control steps, at most; every step is charged to the 200-step episode
MIN_CMD_M = 1e-6
BOUNDS = {'residual_tol_m': RESIDUAL_TOL_M, 'step_cmd_m': STEP_CMD_M, 'target_cap_m': TARGET_CAP_M, 'max_steps': MAX_STEPS}
TRIGGER = {'open_cmd_above': OPEN_CMD, 'close_run_min': CLOSE_RUN_MIN, 'width_min_m': WIDTH_MIN_M, 'width_max_m': WIDTH_MAX_M}
# --- tracking audit on declared, already-used scenes (technical criteria, fixed now; no task outcome enters)
PROBE_TARGETS_M = ((0.03, 0.0), (-0.03, 0.0), (0.0, 0.03), (0.0, -0.03))
AUDIT_PASS = {'direction_cosine_min': 0.95, 'gain_min': 0.2, 'gain_max': 1.2, 'abs_dz_max_m': 0.005, 'useful_xy_min_m': 0.010, 'triggered_scenes_min': 3}
# --- follow-up gate (engineering thresholds, not a significance test)
GATE = {'rescue_min': 4, 'net_min': 3, 'harm_max': 3}
STEPS_PER_SCENE_BOUND = 200 + 5 * 200 + MAX_STEPS   # trunk + five restored branches (prefix replay + continuation) + the control-only reference


# ---------------------------------------------------------------- seeds
def pilot_seeds(manifest, used):
    """The reserved pilot set, after checking the frozen range and that no seed has a recorded use."""
    ids = list(manifest['splits'][SET])
    if ids != list(range(RANGE[0], RANGE[1] + 1)): raise ValueError(f'{SET} is not the frozen range {RANGE[0]}-{RANGE[1]}')
    clash = {src: sorted(set(v) & set(ids))[:10] for src, v in used.items()}; clash = {k: v for k, v in clash.items() if v}
    if clash: raise ValueError(f'pilot seeds have a recorded use: {clash}. The manifest is not patched silently')
    return ids


# ---------------------------------------------------------------- trigger (observable quantities only)
def finger_width(raw):
    """Finger opening from the policy's own proprioception (agent/qpos, last two joints). Not ground-truth object state."""
    q = as_numpy(raw['agent']['qpos']).reshape(-1); return float(q[-2] + q[-1])


def close_run(actions):
    n = 0
    for a in reversed(actions):
        if float(a[-1]) < OPEN_CMD: n += 1
        else: break
    return n


def trigger_rule(pending_grip, run, width):
    """Pure rule: an open command is pending, directly after a run of close commands, with the fingers closed on something."""
    return bool(pending_grip > OPEN_CMD and run >= CLOSE_RUN_MIN and WIDTH_MIN_M <= width <= WIDTH_MAX_M)


def pending(r):
    """The action the policy will execute next, WITHOUT executing it. If the queue is empty the chunk is inferred here
    exactly as QueuePolicy.action would infer it at this step (same noise key, same counters), so looking changes nothing."""
    p = r.policy
    if r.done: raise RuntimeError('no pending action after the end of the episode')
    if not p.queue:
        started = time.perf_counter(); a = p.generate(r.episode, r.t, r.repeat)
        p.inference_seconds += time.perf_counter() - started; p.inference_calls += 1
        if a.shape[0] != p.act_horizon or not np.isfinite(a).all(): raise ValueError('Invalid policy action chunk')
        p.queue = [x.copy() for x in a]
    return np.asarray(p.queue[0], np.float32).copy()


def trigger_reading(r):
    a = pending(r); run = close_run(r.actions); w = finger_width(r.raw)
    return {'step': int(r.t), 'pending_gripper_command': float(a[-1]), 'close_run': int(run), 'finger_width_m': w, 'fires': trigger_rule(float(a[-1]), run, w)}


# ---------------------------------------------------------------- the one privileged read that reaches a controller
class OracleGate:
    """Open only inside the control part of CENTER_ORACLE. Every read is counted."""
    def __init__(self): self.open = False; self.reads = 0


def oracle_offset(env, gate):
    """Horizontal vector from the cubeA centre to the cubeB centre (simulator ground truth). DIAGNOSTIC ORACLE ONLY."""
    if not gate.open: raise PermissionError('ground-truth cube poses may be read for control only inside the CENTER_ORACLE control part')
    gate.reads += 1
    return (as_numpy(env.cubeB.pose.p).reshape(-1)[:2] - as_numpy(env.cubeA.pose.p).reshape(-1)[:2]).astype(float)


# ---------------------------------------------------------------- controllers
def arm_scale(r):
    """Metres per unit of normalised arm action. Same checks as Runtime.retreat (root frame, normalised, no virtual target)."""
    k = r.env.agent.controller.controllers['arm'].config
    if not k.frame.startswith('root_translation') or not k.normalize_action or k.use_target: raise ValueError('Unsupported controller frame/normalization/target semantics')
    if not np.allclose(np.abs(as_numpy(r.env.agent.robot.pose.q).reshape(-1)), [1, 0, 0, 0], atol=1e-5): raise ValueError('This adapter requires an unrotated Panda base')
    scale = np.broadcast_to(np.asarray(k.pos_upper, float), (3,))
    if not np.allclose(np.broadcast_to(np.asarray(k.pos_lower, float), (3,)), -scale): raise ValueError('Symmetric normalized translation limits required')
    return scale


def held_grip(r):
    """The last executed gripper command, which every control arm keeps. It must be a close command."""
    g = None if r.policy.last_action is None else float(r.policy.last_action[-1])
    if g is None or not g < OPEN_CMD: raise RuntimeError(f'the gripper command held at the trigger is not a close command ({g}); the arms are not run on this state')
    return g


def _begin(r, arm):
    if r.intervention is not None: raise ValueError('Invalid/repeated intervention')
    grip = held_grip(r); scale = arm_scale(r)
    r.intervention = INTERVENTION[arm]; r.intervention_t = r.t; r.policy.clear()
    return grip, scale, np.asarray(r.tcp_history[-1], float).copy()


def _xy_action(r, step_xy, scale, grip):
    a = np.zeros(r.adim, np.float32); a[:2] = np.clip(step_xy / scale[:2], -1, 1); a[2] = 0.; a[-1] = grip      # no vertical command, gripper command held
    return a


def bounded_xy_move(r, arm, offset_fn):
    """Closed loop on `offset_fn()` (the horizontal vector still to be removed). Bounds: STEP_CMD_M per step, the
    commanded TCP target stays within TARGET_CAP_M of the starting TCP, at most MAX_STEPS steps. Stops as soon as the
    offset is within RESIDUAL_TOL_M. The move is a controller target; the achieved motion is measured, not assumed.
    Zero steps (already within tolerance) is a valid result: the queue is still cleared and the policy re-plans."""
    grip, scale, start = _begin(r, arm); e0 = np.asarray(offset_fn(), float); used = 0; path = 0.; why = None; trail = []
    while why is None:
        e = np.asarray(offset_fn(), float); n = float(np.linalg.norm(e))
        if n <= RESIDUAL_TOL_M: why = 'within_tolerance'
        elif r.done: why = 'episode_ended'
        elif used >= MAX_STEPS: why = 'step_budget'
        else:
            cur = np.asarray(r.tcp_history[-1], float)[:2]; goal = cur + e * min(1., STEP_CMD_M / n); off = goal - start[:2]; m = float(np.linalg.norm(off))
            if m > TARGET_CAP_M: goal = start[:2] + off * (TARGET_CAP_M / m)
            step = goal - cur; s = float(np.linalg.norm(step))
            if s > STEP_CMD_M: step = step * (STEP_CMD_M / s); s = STEP_CMD_M
            reach = float(np.linalg.norm(cur + step - start[:2]))                 # where this command points, measured from the TCP at the trigger
            if s < MIN_CMD_M or reach > TARGET_CAP_M + 1e-9: why = 'target_cap'   # also when the measured TCP is itself outside the cap: no command is sent
            else:
                r.advance(_xy_action(r, step, scale, grip)); used += 1; path += s
                trail.append({'commanded_xy_m': step.tolist(), 'target_distance_from_start_m': reach, 'offset_before_step_m': n, 'tcp_after': np.asarray(r.tcp_history[-1], float).tolist()})
    final = np.asarray(r.tcp_history[-1], float); e1 = np.asarray(offset_fn(), float)
    r.retreat_info = {'mode': arm, 'start_tcp': start.tolist(), 'final_tcp': final.tolist(), 'offset_before_m': e0.tolist(), 'offset_before_norm_m': float(np.linalg.norm(e0)),
                      'offset_after_m': e1.tolist(), 'offset_after_norm_m': float(np.linalg.norm(e1)), 'steps_used': used, 'end_reason': why,
                      'commanded_path_m': path, 'tcp_xy_displacement_m': float(np.linalg.norm(final[:2] - start[:2])), 'tcp_dz_m': float(final[2] - start[2]),
                      'gripper_command': grip, 'steps': trail, 'bounds': dict(BOUNDS), 'reached': why == 'within_tolerance'}
    r.policy.clear()


def center_oracle(r, gate):
    gate.open = True
    try: bounded_xy_move(r, 'CENTER_ORACLE', lambda: oracle_offset(r.env, gate))
    finally: gate.open = False


def hold_matched(r, steps):
    """The TCP's horizontal position at the trigger is held for `steps` control steps with the same per-step bound, the
    same held gripper command and the same zero vertical command as CENTER_ORACLE. No object pose is read."""
    steps = int(steps)
    if not 0 <= steps <= MAX_STEPS: raise ValueError('hold duration outside the CENTER_ORACLE step bound')
    grip, scale, start = _begin(r, 'HOLD_MATCHED'); used = 0; drift = 0.
    while used < steps and not r.done:
        step = start[:2] - np.asarray(r.tcp_history[-1], float)[:2]; s = float(np.linalg.norm(step))
        if s > STEP_CMD_M: step = step * (STEP_CMD_M / s)
        r.advance(_xy_action(r, step, scale, grip)); used += 1; drift = max(drift, float(np.linalg.norm(np.asarray(r.tcp_history[-1], float)[:2] - start[:2])))
    final = np.asarray(r.tcp_history[-1], float)
    r.retreat_info = {'mode': 'HOLD_MATCHED', 'start_tcp': start.tolist(), 'final_tcp': final.tolist(), 'steps_planned': steps, 'steps_used': used,
                      'end_reason': 'episode_ended' if (r.done and used < steps) else 'matched_return', 'xy_drift_max_m': drift,
                      'tcp_xy_displacement_m': float(np.linalg.norm(final[:2] - start[:2])), 'tcp_dz_m': float(final[2] - start[2]), 'gripper_command': grip}
    r.policy.clear()


def retreat_r5(r):
    """The existing 5 cm retreat. Runtime.force(2) is Runtime.retreat() behind the Stage-2 eligibility grid (step multiple
    of 4, inside a chunk, 48 steps left); the trigger step is not on that grid, so the same three statements of force()
    are executed here without the grid. Runtime.retreat itself is called unchanged and holds the last executed gripper
    command, which is checked to be a close command first."""
    if r.intervention is not None: raise ValueError('Invalid/repeated intervention')
    held_grip(r); r.intervention = INTERVENTION['R5']; r.intervention_t = r.t; r.policy.clear(); r.retreat()


def apply_arm(r, arm, t_center, gate):
    r.repeat = REPEAT
    if arm == 'C0': return
    if arm == 'R5': retreat_r5(r)
    elif arm == 'HOLD_MATCHED': hold_matched(r, t_center)
    elif arm == 'CENTER_ORACLE': center_oracle(r, gate)
    else: raise ValueError(arm)


# ---------------------------------------------------------------- ground-truth readings (logging only)
def gt_reading(row, z0=None):
    """row: a probe() row. Returns the numbers used to describe a state after the fact; never an input to anything."""
    if row is None: return None
    A, B = row['cubeA'], row['cubeB']
    return {'t': row['t'], 'grasped': bool(row['is_cubeA_grasped']), 'offset_xy_m': math.hypot(A[0] - B[0], A[1] - B[1]), 'height_gap_m': A[2] - B[2] - 0.04,
            'cubeA': list(A), 'cubeB': list(B), 'tcp': list(row['tcp']), 'cubeA_lift_m': None if z0 is None else A[2] - z0}


def control_effect(before, after):
    if before is None or after is None: return None
    d = lambda k, i: after[k][i] - before[k][i]
    return {'offset_xy_before_m': before['offset_xy_m'], 'offset_xy_after_m': after['offset_xy_m'], 'cubeA_xy_displacement_m': math.hypot(d('cubeA', 0), d('cubeA', 1)),
            'cubeA_dz_m': d('cubeA', 2), 'cubeB_xy_displacement_m': math.hypot(d('cubeB', 0), d('cubeB', 1)), 'tcp_xy_displacement_m': math.hypot(d('tcp', 0), d('tcp', 1)),
            'tcp_dz_m': d('tcp', 2), 'grasped_before': before['grasped'], 'grasped_after': after['grasped']}


def release_reading(rows, start=0):
    """First step at which the grasp flag goes off after index `start`, with the state one step earlier."""
    for i in range(max(1, start), len(rows)):
        if rows[i - 1]['is_cubeA_grasped'] and not rows[i]['is_cubeA_grasped']:
            g = gt_reading(rows[i - 1]); return {'found': True, 'release_step': rows[i]['t'], 'offset_xy_m': g['offset_xy_m'], 'height_gap_m': g['height_gap_m']}
    return {'found': False, 'release_step': None, 'offset_xy_m': None, 'height_gap_m': None}


# ---------------------------------------------------------------- one scene
def handoff(cells, t, t_center):
    """HOLD_MATCHED and CENTER_ORACLE: no inference during the control part, empty queue at return, the same number of
    control steps, the first inference at trigger + T with one and the same seed. A difference is a technical fault."""
    rows = {}; ok = True; firsts = set(); steps = set(); resumed = 0
    for arm in ('HOLD_MATCHED', 'CENTER_ORACLE'):
        x = cells[arm]; first = x['inference_calls'][0] if x['inference_calls'] else None
        rows[arm] = {'policy_resumed': x['policy_resumed'], 'control_steps': x['control_steps'], 'control_inference_calls': x['control_inference_calls'],
                     'queue_len_at_return': x['queue_len_at_return'], 'history_len_at_return': x['history_len_at_return'],
                     'first_policy_step': None if first is None else first['t'], 'first_policy_seed': None if first is None else first['seed']}
        ok &= x['control_inference_calls'] == 0 and x['queue_len_at_return'] == 0 and x['control_steps'] <= t_center
        if x['policy_resumed']:
            resumed += 1; ok &= first is not None and first['t'] == t + t_center and x['control_steps'] == t_center
            firsts.add((first or {}).get('seed')); steps.add(x['control_steps'])
        else: ok &= first is None
    ok &= len(firsts) <= 1 and len(steps) <= 1
    return {'pass': bool(ok), 'expected_first_policy_step': t + t_center, 't_center': t_center, 'arms_resumed': resumed,
            'matched': bool(ok and resumed == 2) if resumed == 2 else 'NA', **rows}


def collect_scene(r, seed, c, probe=None, order=ARMS):
    """Trunk (never restored), then, if the trigger fired, the control-only reference of CENTER_ORACLE (it fixes the
    duration T) and the four arms from the one snapshot. Returns (record, traces). Any technical difference raises."""
    if sorted(order) != sorted(ARMS): raise ValueError('order must be a permutation of the four arms')
    tol = c['audit']; started = time.perf_counter(); gate = OracleGate(); env_steps = 0
    r.reset(seed); r.repeat = REPEAT; initial = probe(r, None) if probe else None; z0 = initial['cubeA'][2] if initial else None; snap = None; traces = {}
    with Hooks(r, probe) as h:
        h.rows = trunk_rows = []; h.calls = trunk_calls = []
        while not r.done:
            if snap is None:
                rd = trigger_reading(r)
                if rd['fires']:
                    snap = {'s': r.snapshot(), 'tree': runtime_tree(r), 'reading': rd, 'queue_len': len(r.policy.queue), 'state_sha': tree_hash(r.env.get_state_dict()),
                            'gt': gt_reading(trunk_rows[-1] if trunk_rows else initial, z0), 'last_grip': float(r.policy.last_action[-1])}
            r.step_base()
        env_steps += r.t
        trunk = {'policy_noise_repeat_id': REPEAT, 'success': int(r.y), 'steps': r.t, 'reason': r.reason, 'policy_calls': r.policy.inference_calls,
                 'policy_inference_seconds': r.policy.inference_seconds, 'inference_calls': trunk_calls}
        trunk_ref = _ref(r); traces['trunk'] = trunk_rows
        if gate.reads: raise RuntimeError('ground-truth cube poses were read for control during the trunk')
        if snap is None:
            cells = {a: {'arm': a, 'no_trigger': True, 'source': 'uninterrupted trunk (no trigger: every arm equals C0)', 'y': int(r.y), 'steps': r.t, 'reason': r.reason} for a in ARMS}
            record = {'scene': int(seed), 'role': ROLE, 'arms': list(ARMS), 'initial_state': initial, 'trunk': trunk, 'triggered': False, 'trigger': None, 'reference': None,
                      'cells': [cells[a] for a in ARMS], 'handoff': None, 'checks': {}, 'counts': {'continuations': 0, 'reference_control_runs': 0, 'restores': 0, 'env_steps': env_steps},
                      'oracle_reads': 0, 'wall_seconds': time.perf_counter() - started, 'completed': True}
            record = jsonable(record); validate_scene(record, c); return record, traces
        s = snap['s']; t = snap['reading']['step']; before = snap['gt']; restores = 0

        def start():
            nonlocal restores, env_steps
            r.restore(s); restores += 1; env_steps += len(s['actions']); cmp = compare_tree(snap['tree'], runtime_tree(r), tol); live = obs_diff(r.replay_raw, s['raw'])
            if not cmp['pass'] or not live_obs_ok(live, tol): raise RuntimeError(f'scene {seed} trigger {t}: the restored runtime differs from the snapshot ({cmp["failed"]}, live observation {live})')
            if trigger_reading(r) != snap['reading']: raise RuntimeError(f'scene {seed} trigger {t}: the trigger reading is not reproduced after the restore')
            return r.replay_error, live
        # reference: the control part of CENTER_ORACLE alone, once; no policy inference; it only fixes the matched duration T
        start(); h.rows = ref_rows = []; h.calls = ref_calls = []; h.prefix = ref_prefix = []; h.prefix_n = MAX_STEPS; reads0 = gate.reads
        center_oracle(r, gate); h.prefix = None; env_steps += r.t - t
        if ref_calls: raise RuntimeError(f'scene {seed} trigger {t}: the reference control part called the policy')
        ri = clone(r.retreat_info); T = int(ri['steps_used']); ref_end = bool(r.done)
        reference = {'t_center': T, 'end_reason': ri['end_reason'], 'ended_episode': ref_end, 'offset_before_norm_m': ri['offset_before_norm_m'], 'offset_after_norm_m': ri['offset_after_norm_m'],
                     'commanded_path_m': ri['commanded_path_m'], 'tcp_xy_displacement_m': ri['tcp_xy_displacement_m'], 'tcp_dz_m': ri['tcp_dz_m'], 'policy_inference_calls': 0,
                     'oracle_reads': gate.reads - reads0, 'commands_sha256': tree_hash([x['action'] for x in ref_prefix]), 'expected_return_step': t + T,
                     'note': 'controller-only preparation run; not a continuation; no outcome is read from it'}
        traces['reference'] = ref_rows; cells = {}; dup = None

        def branch(arm):
            nonlocal env_steps
            tick = time.perf_counter(); err, live = start(); calls0 = r.policy.inference_calls; reads0 = gate.reads
            h.rows = rows = []; h.calls = calls = []; h.prefix = prefix = [] if arm == 'CENTER_ORACLE' else None; h.prefix_n = MAX_STEPS
            apply_arm(r, arm, T, gate); h.prefix = None; ctl = arm != 'C0'; k = r.t - t; cc = len(calls) if ctl else 0; q = len(r.policy.queue); early = ctl and bool(r.done)
            hl = len(r.policy.history); info = clone(r.retreat_info) if ctl else None; reads = gate.reads - reads0
            if ctl and cc: raise RuntimeError(f'scene {seed} trigger {t} {arm}: the policy was called during the control part')
            if (reads > 0) != (arm == 'CENTER_ORACLE'): raise RuntimeError(f'scene {seed} trigger {t} {arm}: ground-truth cube poses were read {reads} times for control')
            after = gt_reading(rows[k - 1], z0) if (probe and ctl and k) else (before if ctl else None)
            res = r.complete(); env_steps += r.t - t; same = None
            if arm == 'CENTER_ORACLE':
                same = prefix_equal(ref_prefix, prefix, tol)
                same['offset_before_equal'] = bool(abs(info['offset_before_norm_m'] - ri['offset_before_norm_m']) <= tol['state_atol'])
                if not (same['pass'] and same['offset_before_equal'] and k == T and info['end_reason'] == ri['end_reason']):
                    raise RuntimeError(f'scene {seed} trigger {t}: CENTER_ORACLE differs from its control-only reference ({jsonable(same)})')
            if ctl and info['gripper_command'] != snap['last_grip']: raise RuntimeError(f'scene {seed} trigger {t} {arm}: the gripper command of the snapshot was not held')
            cells[arm] = {'arm': arm, 'no_trigger': False, 'policy_noise_repeat_id': REPEAT, 'y': int(res['success']), 'steps': res['steps'], 'reason': res['reason'],
                          'intervention': res['intervention'], 'intervention_step': res['intervention_step'], 'control': info, 'control_steps': k if ctl else 0,
                          'control_inference_calls': cc, 'ended_during_control': bool(early), 'policy_resumed': bool(ctl and not early),
                          'queue_len_at_return': q if ctl else None, 'history_len_at_return': hl if ctl else None, 'oracle_reads': reads,
                          'first_inference_step': calls[0]['t'] if calls else None, 'inference_calls': calls, 'policy_calls_after_trigger': res['policy_calls'] - calls0,
                          'steps_after_trigger': res['steps'] - t, 'same_as_reference': same, 'gt_effect_of_control': control_effect(before, after) if (probe and ctl) else None,
                          'gt_release_after_trigger': release_reading([trunk_rows[t - 1] if t else initial] + rows) if probe else None,
                          'prefix_replay_state_error': err, 'live_observation': live, 'wall_seconds': time.perf_counter() - tick}
            traces[arm] = rows
        for arm in order:
            branch(arm)
            if arm == 'C0':
                dup = _same_run(trunk_ref, r, tol)
                if not dup['match']: raise RuntimeError(f'scene {seed} trigger {t}: C0 from the snapshot does not reproduce the uninterrupted trunk ({jsonable(dup)})')
        ho = handoff(cells, t, T)
        if not ho['pass']: raise RuntimeError(f'scene {seed} trigger {t}: HOLD_MATCHED and CENTER_ORACLE did not hand over identically ({jsonable(ho)})')
    g = snap['gt']
    trig = {**snap['reading'], 'snapshot_id': f"{int(seed)}:{t}:{snap['state_sha'][:16]}", 'queue_len': snap['queue_len'], 'remaining_budget': r.horizon - t,
            'held_gripper_command': snap['last_grip'], 'on_stage2_eligibility_grid': bool(t % c['protocol']['decision_stride'] == 0 and 0 < snap['queue_len'] < r.policy.act_horizon),
            'gt_for_diagnosis_only': None if g is None else {**g, 'false_trigger': not (g['grasped'] and (g['cubeA_lift_m'] or 0.) >= 0.01),
                                                             'false_trigger_rule': 'cubeA not grasped, or not lifted 1 cm, at the trigger'}}
    record = {'scene': int(seed), 'role': ROLE, 'arms': list(ARMS), 'initial_state': initial, 'trunk': trunk, 'triggered': True, 'trigger': trig, 'reference': reference,
              'cells': [cells[a] for a in ARMS], 'handoff': ho, 'checks': {'c0_reproduces_trunk': dup},
              'counts': {'continuations': len(ARMS), 'reference_control_runs': 1, 'restores': restores, 'env_steps': env_steps},
              'oracle_reads': gate.reads, 'wall_seconds': time.perf_counter() - started, 'completed': True}
    record = jsonable(record); validate_scene(record, c)
    return record, traces


def validate_scene(rec, c=None):
    """Accounting of one scene record: four arms, once each; the semantics declared above hold cell by cell."""
    cells = rec['cells']
    if [x['arm'] for x in cells] != list(ARMS): raise ValueError('a scene needs exactly the four arms, once each')
    if c is not None and any(x['steps'] > c['env']['horizon'] for x in cells): raise ValueError('an arm exceeded the episode horizon (no bonus steps)')
    if not rec['triggered']:
        if rec['trigger'] is not None or any(not x['no_trigger'] or (x['y'], x['steps']) != (rec['trunk']['success'], rec['trunk']['steps']) for x in cells):
            raise ValueError('a scene without a trigger must carry the trunk outcome in every arm')
        if rec['counts']['continuations'] or rec['oracle_reads']: raise ValueError('a scene without a trigger has continuations or oracle reads')
        return True
    t = rec['trigger']['step']; T = rec['reference']['t_center']; by = {x['arm']: x for x in cells}
    if not rec['trigger']['fires'] or not rec['trigger']['held_gripper_command'] < OPEN_CMD: raise ValueError('trigger record without a firing rule or a held close command')
    if not 0 <= T <= MAX_STEPS or rec['reference']['policy_inference_calls'] != 0: raise ValueError('the reference is not a control-only run within the step bound')
    x = by['C0']
    if (x['y'], x['steps'], x['reason']) != (rec['trunk']['success'], rec['trunk']['steps'], rec['trunk']['reason']) or not rec['checks']['c0_reproduces_trunk']['match']:
        raise ValueError('C0 does not reproduce the uninterrupted trunk')
    if x['intervention'] is not None or x['control'] is not None or x['oracle_reads']: raise ValueError('C0 was recorded as an intervention')
    for arm in CONTROL:
        x = by[arm]; ctl = x['control']
        if x['intervention'] != INTERVENTION[arm] or x['intervention_step'] != t or x['queue_len_at_return'] != 0 or x['control_inference_calls'] != 0:
            raise ValueError(f'{arm} was not applied at the trigger with a cleared queue and no inference during control')
        if any(k['policy_noise_repeat_id'] != REPEAT for k in x['inference_calls']): raise ValueError('an inference call left the declared noise repeat')
        if ctl['gripper_command'] != rec['trigger']['held_gripper_command'] or ctl['steps_used'] != x['control_steps']: raise ValueError(f'{arm}: gripper command or step count differs from its control record')
        if (x['oracle_reads'] > 0) != (arm == 'CENTER_ORACLE'): raise ValueError('ground truth reached a controller outside CENTER_ORACLE')
    ce = by['CENTER_ORACLE']['control']; ho = by['HOLD_MATCHED']['control']
    if by['CENTER_ORACLE']['control_steps'] != T or ce['commanded_path_m'] > MAX_STEPS * STEP_CMD_M + 1e-9: raise ValueError('CENTER_ORACLE left its step bound or its reference duration')
    if any(math.hypot(*s['commanded_xy_m']) > STEP_CMD_M + 1e-9 for s in ce['steps']): raise ValueError('a CENTER_ORACLE command exceeded the per-step bound')
    if any(s['target_distance_from_start_m'] > TARGET_CAP_M + 1e-9 for s in ce['steps']) or len(ce['steps']) != ce['steps_used'] or ce['steps_used'] > MAX_STEPS:
        raise ValueError('a CENTER_ORACLE command pointed outside the target cap, or the step list does not match the step count')
    if ho['steps_planned'] != T or by['HOLD_MATCHED']['control_steps'] > T: raise ValueError('HOLD_MATCHED was not planned for the CENTER_ORACLE duration')
    if not rec['handoff']['pass']: raise ValueError('handoff check failed')
    if rec['counts']['continuations'] != len(ARMS) or rec['counts']['reference_control_runs'] != 1: raise ValueError('counts do not match the record')
    return True


# ---------------------------------------------------------------- tracking audit (declared used scenes; mechanical, no task outcome)
def tracking_probe(r, target_xy, probe=None):
    """From a restored trigger state: the same bounded move towards a synthetic TCP displacement `target_xy` (no object
    pose is used). Measures what the controller actually achieves within the bounds."""
    start = np.asarray(r.tcp_history[-1], float).copy(); goal = np.asarray(target_xy, float)
    before = probe(r, None) if probe else None
    bounded_xy_move(r, 'TRACKING_PROBE', lambda: goal - (np.asarray(r.tcp_history[-1], float)[:2] - start[:2]))
    info = clone(r.retreat_info); final = np.asarray(r.tcp_history[-1], float); moved = final[:2] - start[:2]; n = float(np.linalg.norm(moved))
    gains = []; prev = start
    for sx in info['steps']:
        cmd = np.asarray(sx['commanded_xy_m'], float); got = np.asarray(sx['tcp_after'], float)[:2] - prev[:2]; prev = np.asarray(sx['tcp_after'], float)
        gains.append(float(got @ cmd / (cmd @ cmd)))
    after = probe(r, None) if probe else None
    return {'target_xy_m': list(map(float, goal)), 'steps_used': info['steps_used'], 'end_reason': info['end_reason'], 'achieved_xy_m': moved.tolist(), 'achieved_norm_m': n,
            'direction_cosine': float(moved @ goal / (n * np.linalg.norm(goal))) if n > 0 else 0., 'tcp_dz_m': float(final[2] - start[2]), 'commanded_path_m': info['commanded_path_m'],
            'gain_per_step': gains, 'gain_median': float(np.median(gains)) if gains else None,
            'gt_effect': control_effect(gt_reading(before), gt_reading(after)) if probe else None}


def tracking_verdict(probes, triggered_scenes):
    p = AUDIT_PASS; why = []
    if triggered_scenes < p['triggered_scenes_min']: why.append(f'only {triggered_scenes} audit scenes reached a trigger')
    for x in probes:
        tag = f"scene {x['scene']} target {x['target_xy_m']}"
        if x['direction_cosine'] < p['direction_cosine_min']: why.append(f'{tag}: direction cosine {x["direction_cosine"]:.3f}')
        if x['gain_median'] is None or not p['gain_min'] <= x['gain_median'] <= p['gain_max']: why.append(f'{tag}: gain {x["gain_median"]}')
        if abs(x['tcp_dz_m']) > p['abs_dz_max_m']: why.append(f'{tag}: vertical drift {x["tcp_dz_m"]:.4f} m')
        if x['achieved_norm_m'] < p['useful_xy_min_m']: why.append(f'{tag}: only {x["achieved_norm_m"]:.4f} m achieved')
    return {'pass': not why, 'status': 'OK' if not why else 'TECHNICAL_BLOCKED', 'problems': why, 'criteria': dict(p), 'probes': len(probes)}


# ---------------------------------------------------------------- analysis (scene is the unit; nothing is filtered)
def wilson(k, n, z=1.959963984540054):
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0., m - h), min(1., m + h)]


def paired(y_ref, y_arm):
    """y_ref, y_arm: {scene: 0/1} over the SAME scenes."""
    if set(y_ref) != set(y_arm): raise ValueError('the two arms are not on the same scenes')
    n = len(y_ref); c = {'both_succeed': 0, 'rescue': 0, 'harm': 0, 'both_fail': 0}
    for s in y_ref: c[('both_succeed' if y_arm[s] else 'harm') if y_ref[s] else ('rescue' if y_arm[s] else 'both_fail')] += 1
    ok = c['both_succeed'] + c['harm']; arm_ok = c['both_succeed'] + c['rescue']
    return {'n': n, **c, 'net': c['rescue'] - c['harm'], 'reference_success': ok, 'arm_success': arm_ok, 'harm_given_reference_success': (c['harm'] / ok) if ok else 'NA',
            'reference_success_wilson95': wilson(ok, n), 'arm_success_wilson95': wilson(arm_ok, n), 'harm_given_reference_success_wilson95': wilson(c['harm'], ok) if ok else 'NA',
            'rescue_scenes': sorted(s for s in y_ref if not y_ref[s] and y_arm[s]), 'harm_scenes': sorted(s for s in y_ref if y_ref[s] and not y_arm[s])}


def followup_gate(primary, center_success, hold_success, audits_pass):
    g = {'rescue_at_least_4': primary['rescue'] >= GATE['rescue_min'], 'net_at_least_3': primary['net'] >= GATE['net_min'], 'harm_at_most_3': primary['harm'] <= GATE['harm_max'],
         'center_exceeds_hold': center_success > hold_success, 'technical_audits_pass': bool(audits_pass)}
    return {'criteria': g, 'label': 'ORACLE_GATE_PASS' if all(g.values()) else 'NO_ORACLE_GATE_PASS', 'thresholds': dict(GATE),
            'scope': 'engineering gate for whether a non-oracle estimator is worth developing; not a significance test and not a guarantee'}


def summarize(records, planned, audits_pass):
    """All planned scenes in every denominator. Raises if a scene is missing: a partial run has no summary."""
    by = {r['scene']: r for r in records}
    if sorted(by) != sorted(int(s) for s in planned) or len(by) != len(records): raise ValueError('the records are not exactly the planned scenes; a partial run is not summarised')
    y = {a: {s: next(x['y'] for x in r['cells'] if x['arm'] == a) for s, r in by.items()} for a in ARMS}
    trig = [r for r in records if r['triggered']]; cell = lambda r, a: next(x for x in r['cells'] if x['arm'] == a)
    false = [r['scene'] for r in trig if (r['trigger']['gt_for_diagnosis_only'] or {}).get('false_trigger')]
    moved = [r for r in trig if r['reference']['t_center'] > 0]; aligned = [r for r in trig if r['reference']['t_center'] == 0]
    stat = lambda v: {'n': len(v), 'min': float(np.min(v)), 'median': float(np.median(v)), 'max': float(np.max(v))} if v else {'n': 0}
    eff = lambda a, k: [cell(r, a)['gt_effect_of_control'][k] for r in moved if cell(r, a)['gt_effect_of_control']]
    primary = paired(y['C0'], y['CENTER_ORACLE'])
    out = {'role': ROLE, 'unit': 'scene', 'scenes': len(records),
           'denominators': {'all': len(records), 'no_trigger': len(records) - len(trig), 'triggered': len(trig), 'false_triggers_by_ground_truth': len(false), 'false_trigger_scenes': false,
                            'actually_moved_T_gt_0': len(moved), 'already_aligned_T_eq_0': len(aligned), 'c0_success': sum(y['C0'].values()), 'c0_failure': len(records) - sum(y['C0'].values())},
           'success': {a: {'k': sum(y[a].values()), 'n': len(records), 'wilson95': wilson(sum(y[a].values()), len(records))} for a in ARMS},
           'primary_CENTER_ORACLE_vs_C0': primary,
           'secondary': {'CENTER_ORACLE_vs_HOLD_MATCHED': paired(y['HOLD_MATCHED'], y['CENTER_ORACLE']), 'HOLD_MATCHED_vs_C0': paired(y['C0'], y['HOLD_MATCHED']),
                         'R5_vs_C0': {**paired(y['C0'], y['R5']), 'note': 'R5 differs in duration and motion; not a duration-matched contrast'}},
           'control': {'t_center_steps': stat([r['reference']['t_center'] for r in trig]), 'center_end_reasons': {k: sum(1 for r in trig if r['reference']['end_reason'] == k) for k in sorted({r['reference']['end_reason'] for r in trig})},
                       'offset_before_m': stat([r['reference']['offset_before_norm_m'] for r in trig]), 'offset_after_center_m_moved': stat([r['reference']['offset_after_norm_m'] for r in moved]),
                       'center_residual_within_tolerance_moved': sum(1 for r in moved if r['reference']['offset_after_norm_m'] <= RESIDUAL_TOL_M),
                       'center_tcp_xy_displacement_m': stat([r['reference']['tcp_xy_displacement_m'] for r in moved]), 'center_tcp_dz_m': stat([r['reference']['tcp_dz_m'] for r in moved]),
                       **{f'{a}_grasp_kept_through_control_moved': sum(1 for r in moved if (cell(r, a)['gt_effect_of_control'] or {}).get('grasped_before') and cell(r, a)['gt_effect_of_control']['grasped_after']) for a in CONTROL},
                       'center_cubeA_xy_displacement_m': stat(eff('CENTER_ORACLE', 'cubeA_xy_displacement_m')), 'hold_tcp_xy_drift_m': stat([cell(r, 'HOLD_MATCHED')['control']['xy_drift_max_m'] for r in moved]),
                       'r5_steps': stat([cell(r, 'R5')['control_steps'] for r in trig]), 'extra_steps_vs_c0': {a: stat([cell(r, a)['steps'] - cell(r, 'C0')['steps'] for r in trig]) for a in CONTROL}},
           'release_after_trigger': {a: {'found': sum(1 for r in trig if (cell(r, a)['gt_release_after_trigger'] or {}).get('found')),
                                         'offset_xy_m': stat([cell(r, a)['gt_release_after_trigger']['offset_xy_m'] for r in trig if (cell(r, a)['gt_release_after_trigger'] or {}).get('found')])} for a in ARMS},
           'followup_gate': followup_gate(primary, sum(y['CENTER_ORACLE'].values()), sum(y['HOLD_MATCHED'].values()), audits_pass),
           'scope': ['CENTER_ORACLE reads simulator ground truth and is not deployable; a gain is feasibility evidence for this controller, trigger and bound only, not evidence about perception.',
                     'One task, one policy, one noise repeat, 100 development scenes. No significance claim; the intervals describe sampling of scenes only.',
                     'If CENTER_ORACLE exceeds C0 but not HOLD_MATCHED, the gain is not attributed to the horizontal correction.',
                     'R5 against C0 is not duration-matched.', 'Pairing is within these 100 scenes; the count is not compared with 79/100 of earlier sets.']}
    return out


def projected_cost(*, seconds_per_env_step, usd_per_hour, spent_usd, cap_usd, scenes=100, fixed_seconds=600, reserve_usd=0.20, margin=1.25):
    """All-in projection for the one pilot pass with every scene at the worst case. Inputs are time and money only."""
    hours = (scenes * STEPS_PER_SCENE_BOUND * seconds_per_env_step + fixed_seconds) * margin / 3600; total = spent_usd + hours * usd_per_hour + reserve_usd
    return {'projected_remaining_hours': hours, 'projected_total_usd': total, 'cap_usd': cap_usd, 'fits_cap': bool(total <= cap_usd),
            'inputs': {'seconds_per_env_step': seconds_per_env_step, 'usd_per_hour': usd_per_hour, 'spent_usd': spent_usd, 'scenes': scenes, 'steps_per_scene_bound': STEPS_PER_SCENE_BOUND,
                       'fixed_seconds': fixed_seconds, 'reserve_usd': reserve_usd, 'margin': margin},
            'rule': 'if the projection exceeds the cap the pilot does not start; scenes, arms and bounds are not reduced'}
