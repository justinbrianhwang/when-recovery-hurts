"""Temporal Vision-CENTER v2 (work id vision_center_v2_001). PREPARATION: the memory and the decision are evaluated offline on
stored frames; the simulator side below has only been mock-tested. Vision-CENTER v1 (hci.vision_center) is imported and
left untouched; it is the single-frame baseline arm.

What v2 adds to v1, using only what the policy itself observes (base_camera and hand_camera RGB, TCP pose, finger joints,
executed gripper commands, step counter):
  target memory   the target cube's position seen EARLIER in the episode (hand camera fit, or the base camera's view of
                  its top face), kept only while it is confirmed and not exposed to an unobserved push;
  hold memory     the last accepted held-cube fit, kept only while the fingers have not moved and no open command ran;
  uncertainty     every remembered quantity carries a bound that enters the act threshold and the stop tolerance;
  safety          no move when the held cube is beside the target and not clearly above its top face (scene 50892);
                  earlier stop when the target or the held cube shifts during the move.
When neither the current frame nor a valid memory gives a position, v2 abstains and the episode is the uninterrupted one.
It never reads a cube pose, a grasp flag, a task flag or an outcome. Ground truth appears only in offline labels.
"""
import numpy as np
from . import vision_center as V1

WORK = 'vision_center_v2_001'
# ---------------------------------------------------------------- uncertainty bounds (metres); see the observability report for the data behind each
U_HAND_M = 0.002                  # hand-camera target fit (accepted fits on development data: max 1.8 mm)
U_BASE_M = 0.005                  # base-camera target estimate below (500 demonstrations: 99.97 % within 5 mm, max 6.3 mm)
U_HOLD_M = 0.002                  # accepted held-cube fit (median 0.8 mm, p90 1.6 mm, max 2.8 mm on 265 used trigger frames)
U_PUSH_M = 0.005                  # added when the gripper was near the remembered target after its last confirmation
# ---------------------------------------------------------------- target from the base camera: largest connected lit-green region
BASE_VIEW = {'ratio_min': 0.90, 'ratio_max': 1.25, 'pixels_min': 9}


def largest_component(mask):
    """Pixel coordinates (rows, cols) of the largest 8-connected region of a boolean mask."""
    pts = set(zip(*(int_arr.tolist() for int_arr in np.nonzero(mask)))); best = []
    while pts:
        stack = [pts.pop()]; comp = []
        while stack:
            p = stack.pop(); comp.append(p)
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    q = (p[0] + dv, p[1] + du)
                    if q in pts: pts.remove(q); stack.append(q)
        if len(comp) > len(best): best = comp
    a = np.array(sorted(best), int).reshape(-1, 2); return a[:, 0], a[:, 1]


def target_base(img_base):
    """Centre of the target cube's top face from the base camera, or None. v1's full-view estimate took the centroid of all
    lit-green pixels and was pulled centimetres away by a few stray pixels; here only the largest connected region counts,
    and it must have the area of an unoccluded 4 cm face at the estimated place."""
    m = V1.classes(img_base)['green_lit']
    if int(m.sum()) < 4: return None
    v, u = largest_component(m); n = len(v); cam = V1.cam_base(); X, s = V1.ray_plane(cam, u.mean() + V1.PIXEL_CENTRE, v.mean() + V1.PIXEL_CENTRE, V1.TARGET_TOP_Z_M)
    if not s > 0: return None
    h = V1.CUBE_HALF_M; quad = np.array([V1.project(cam, [X[0] + a, X[1] + b, V1.TARGET_TOP_Z_M])[0] for a, b in ((-h, -h), (h, -h), (h, h), (-h, h))])
    ratio = n / max(V1.quad_area(quad), 1e-6); border = bool(u.min() == 0 or v.min() == 0 or u.max() == V1.IMG - 1 or v.max() == V1.IMG - 1); k = BASE_VIEW
    return {'xy': X[:2].copy(), 'pixels': n, 'area_ratio': float(ratio), 'touches_border': border, 'valid': bool(not border and n >= k['pixels_min'] and k['ratio_min'] <= ratio <= k['ratio_max'])}


# ---------------------------------------------------------------- memory
CONFIRM_MIN = 3                   # consistent observations before a remembered target may be used
CONTACT_RADIUS_M = 0.10           # gripper within this horizontal distance of the remembered target ...
CONTACT_TCP_Z_M = 0.09            # ... and its TCP below this height: the gripper or the held cube could touch the target
FINGER_TOL_M = 0.001              # a remembered hold is dropped when the finger opening changes by more than this
HOLD_EVERY = 4                    # the held-cube fit runs on every HOLD_EVERY-th step while the gripper is closed on something
HOLD_WIDTH_MAX_M = 0.075          # fingers narrower than fully open (0.08 m)
MEMORY = {'confirm_min': CONFIRM_MIN, 'contact_radius_m': CONTACT_RADIUS_M, 'contact_tcp_z_m': CONTACT_TCP_Z_M, 'finger_tol_m': FINGER_TOL_M, 'hold_every': HOLD_EVERY, 'hold_width_max_m': HOLD_WIDTH_MAX_M,
          'u_hand_m': U_HAND_M, 'u_base_m': U_BASE_M, 'u_hold_m': U_HOLD_M, 'u_push_m': U_PUSH_M, 'base_view': BASE_VIEW}


class Memory:
    """Per-episode record of what was seen. `update` is called once per step with that step's observation, before the
    step's action. It holds no image and reads nothing but its arguments."""

    def __init__(self):
        self.target = None; self.hold = None; self.steps = 0; self.log = {'hand_obs': 0, 'base_obs': 0, 'conflicts': 0, 'hold_obs': 0, 'hold_dropped': 0, 'hold_fits_run': 0}

    def copy(self):
        m = Memory(); m.steps = self.steps; m.log = dict(self.log)
        m.target = None if self.target is None else {**self.target, 'xy': self.target['xy'].copy()}
        m.hold = None if self.hold is None else {**self.hold, 'h': self.hold['h'].copy()}
        return m

    def _observe_target(self, t, xy, u, src):
        tr = self.target; xy = np.asarray(xy, float)
        if tr is not None and float(np.linalg.norm(xy - tr['xy'])) <= tr['u'] + u:                 # consistent with what is remembered
            tr['n'] += 1; tr['t_last'] = t; tr['exposed'] = 0
            if u < tr['u'] or (u == tr['u'] and src == tr['src']): tr.update(xy=xy.copy(), u=u, src=src, t_xy=t)       # the more precise source owns the position
        else:
            if tr is not None: self.log['conflicts'] += 1
            self.target = {'xy': xy.copy(), 'u': u, 'src': src, 'n': 1, 't_first': t, 't_last': t, 't_xy': t, 'exposed': 0}

    def update(self, t, img_hand, img_base, tcp_pose, finger_width, last_grip):
        p = np.asarray(tcp_pose, float).reshape(-1); self.steps += 1; out = {'t': int(t), 'hand': False, 'base': False, 'hold': None}
        tf = V1.target_partial_fit(img_hand, V1.cam_hand(p), 'hand_camera'); tb = target_base(img_base) if img_base is not None else None
        if tf is not None and tf['valid']: self._observe_target(t, tf['xy'], U_HAND_M, 'hand'); self.log['hand_obs'] += 1; out['hand'] = True
        elif tb is not None and tb['valid']: self._observe_target(t, tb['xy'], U_BASE_M, 'base'); self.log['base_obs'] += 1; out['base'] = True
        elif self.target is not None and float(np.linalg.norm(p[:2] - self.target['xy'])) < CONTACT_RADIUS_M and p[2] < CONTACT_TCP_Z_M:
            self.target['exposed'] += 1                                                         # unobserved step with the gripper near the remembered target
        closed = last_grip is not None and float(last_grip) < 0 and float(finger_width) < HOLD_WIDTH_MAX_M
        if self.hold is not None and (not closed or abs(float(finger_width) - self.hold['width']) > FINGER_TOL_M): self.hold = None; self.log['hold_dropped'] += 1
        if closed and t % HOLD_EVERY == 0:
            hf = V1.hold_fit(img_hand); self.log['hold_fits_run'] += 1; out['hold'] = bool(hf is not None and hf['valid'])
            if out['hold']: self.hold = {'h': np.asarray(hf['h'], float).copy(), 'yaw': float(hf['yaw']), 't': int(t), 'width': float(finger_width)}; self.log['hold_obs'] += 1
        return out

    def note_hold(self, t, hf, finger_width):
        """An accepted fit computed elsewhere (the reading at the trigger) refreshes the remembered hold."""
        if hf is not None and hf['valid']: self.hold = {'h': np.asarray(hf['h'], float).copy(), 'yaw': float(hf['yaw']), 't': int(t), 'width': float(finger_width)}

    def target_estimate(self):
        """(estimate, None) or (None, reason). The estimate's bound grows when the gripper was near the target unobserved."""
        tr = self.target
        if tr is None: return None, 'never_seen'
        if tr['n'] < CONFIRM_MIN: return None, 'not_confirmed'
        return {'xy': tr['xy'].copy(), 'u': tr['u'] + (U_PUSH_M if tr['exposed'] > 0 else 0.0), 'src': tr['src'], 't_last': tr['t_last'], 'exposed_steps': tr['exposed'], 'confirmations': tr['n']}, None

    def hold_estimate(self, finger_width, last_grip):
        h = self.hold
        if h is None: return None
        if last_grip is None or float(last_grip) >= 0 or abs(float(finger_width) - h['width']) > FINGER_TOL_M: return None
        return {'h': h['h'].copy(), 'yaw': h['yaw'], 't': h['t'], 'u': U_HOLD_M}


# ---------------------------------------------------------------- the remembered target against the current hand image
CHECK = {'outside_pixels_max': 8, 'inside_pixels_min': 30, 'inside_green_min': 0.7, 'slack_m': 0.004}


def memory_check(img_hand, tcp_pose, xy, u):
    """Does the current hand image contradict a target remembered at `xy` (bound `u`)? Lit-green pixels far outside the
    remembered face, or visible table where the face should be, contradict it. Seeing nothing of it (hidden) does not."""
    cam = V1.cam_hand(np.asarray(tcp_pose, float).reshape(-1)); k = V1.classes(img_hand); X, Y, ahead = V1.plane_points(cam, V1.TARGET_TOP_Z_M)
    d = np.hypot(X - xy[0], Y - xy[1]); lit = k['green_lit']; s = CHECK['slack_m']
    outside = int((lit & ahead & (d > V1.CUBE_HALF_M * np.sqrt(2) + u + s)).sum())
    inner = ahead & (d < V1.CUBE_HALF_M - u - s) & ~(k['red'] | k['grey']); n_in = int(inner.sum()); frac = float((lit & inner).sum() / n_in) if n_in else None
    bad = outside > CHECK['outside_pixels_max'] or (n_in >= CHECK['inside_pixels_min'] and frac < CHECK['inside_green_min'])
    return {'contradicted': bool(bad), 'lit_pixels_outside': outside, 'expected_pixels_unoccluded': n_in, 'lit_fraction_inside': frac}


# ---------------------------------------------------------------- decision
ACT_MIN_M = V1.ACT_MIN_M; RESIDUAL_TOL_M = V1.RESIDUAL_TOL_M; CARRY_Z_MIN_M = V1.CARRY_Z_MIN_M; STEPS_LEFT_MIN = V1.STEPS_LEFT_MIN
MEMORY_STOP_TOL_M = 0.005         # a move driven by a remembered target stops closer to the centre, because that target is less certain
SIDE_OFFSET_M = 0.040             # centres at least one cube width apart (estimate; plus the bound when the target is remembered): the held cube is beside the target, not over it
SIDE_GAP_MIN_M = 0.003            # ... then its bottom must be this far above the target's top face
MEMORY_GAP_MIN_M = 0.000          # a move on a remembered (unseen) target needs the held cube's bottom at or above the target's top face
TARGET_JUMP_MAX_M = 0.003         # during control (v1: 0.005)
HOLD_SHIFT_MAX_M = 0.003          # during control: the held cube moved in the gripper
ABSTAIN = ('NO_HOLD_ESTIMATE', 'NO_TARGET_ESTIMATE', 'MEMORY_CONTRADICTED', 'LOW_CARRY', 'SIDE_CONTACT_RISK', 'TOO_FEW_STEPS', 'ALIGNED', 'ALIGNED_WITHIN_UNCERTAINTY')
DECISION = {'act_min_m': ACT_MIN_M, 'residual_tol_m': RESIDUAL_TOL_M, 'memory_stop_tol_m': MEMORY_STOP_TOL_M, 'carry_z_min_m': CARRY_Z_MIN_M, 'steps_left_min': STEPS_LEFT_MIN, 'side_offset_m': SIDE_OFFSET_M,
            'side_gap_min_m': SIDE_GAP_MIN_M, 'memory_gap_min_m': MEMORY_GAP_MIN_M, 'target_jump_max_m': TARGET_JUMP_MAX_M, 'hold_shift_max_m': HOLD_SHIFT_MAX_M, 'check': CHECK}


def estimate(rd, mem, img_hand, tcp_pose, finger_width, last_grip):
    """Positions for this step: the current frame where v1 accepts it, the memory otherwise. Returns a dict with
    `why` set when no usable estimate exists."""
    p = np.asarray(tcp_pose, float).reshape(-1); out = {'why': None, 'target_src': None, 'hold_src': None, 'u': None, 'target_xy': None, 'held_xyz': None, 'offset_xy': None, 'offset_norm_m': None, 'gap_m': None, 'memory_check': None,
                                                         'memory_reason': None, 'h': None}
    hf = rd['hold_fit']
    if hf is not None and hf['valid']: h = np.asarray(hf['h'], float); uh = U_HOLD_M; out['hold_src'] = 'frame'
    else:
        hm = mem.hold_estimate(finger_width, last_grip)
        if hm is None: out['why'] = 'NO_HOLD_ESTIMATE'; return out
        h = hm['h']; uh = hm['u']; out['hold_src'] = 'memory'
    A = p[:3] + V1.quat_R(p[3:7]) @ h; out['h'] = np.asarray(h, float); out['held_xyz'] = A; out['gap_m'] = float(A[2] - V1.CUBE_HALF_M - V1.TARGET_TOP_Z_M)
    if rd['target_fresh']: txy = np.asarray(rd['target_fit']['xy'], float); ut = U_HAND_M; out['target_src'] = 'frame'
    else:
        tm, why = mem.target_estimate(); out['memory_reason'] = why
        if tm is None: out['why'] = 'NO_TARGET_ESTIMATE'; return out
        chk = memory_check(img_hand, p, tm['xy'], tm['u']); out['memory_check'] = chk; out['memory_target'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in tm.items()}
        if chk['contradicted']: out['why'] = 'MEMORY_CONTRADICTED'; return out
        txy = tm['xy']; ut = tm['u']; out['target_src'] = 'memory_' + tm['src']
    out['target_xy'] = txy; out['u'] = float(ut + uh); out['offset_xy'] = txy - A[:2]; out['offset_norm_m'] = float(np.linalg.norm(out['offset_xy'])); return out


def decide(est, steps_left):
    """Act or abstain at the trigger. Any abstention leaves the policy's queue untouched."""
    if est['why'] is not None: why = est['why']
    else:
        n = est['offset_norm_m']; u = est['u']; gap = est['gap_m']; remembered = est['target_src'] != 'frame'
        if est['held_xyz'][2] < CARRY_Z_MIN_M: why = 'LOW_CARRY'
        elif remembered and gap < MEMORY_GAP_MIN_M: why = 'LOW_CARRY'
        elif n + (u if remembered else 0.0) >= SIDE_OFFSET_M and gap < SIDE_GAP_MIN_M: why = 'SIDE_CONTACT_RISK'
        elif steps_left < STEPS_LEFT_MIN: why = 'TOO_FEW_STEPS'
        elif n < ACT_MIN_M: why = 'ALIGNED'
        elif n - u < RESIDUAL_TOL_M: why = 'ALIGNED_WITHIN_UNCERTAINTY'
        else: why = 'ACT'
    return {'act': why == 'ACT', 'reason': why}


def stop_tol(est):
    return RESIDUAL_TOL_M if est['target_src'] == 'frame' else MEMORY_STOP_TOL_M


def summary(est):
    f = lambda x: None if x is None else [float(v) for v in np.asarray(x).reshape(-1)]
    return {**{k: est.get(k) for k in ('why', 'target_src', 'hold_src', 'u', 'offset_norm_m', 'gap_m', 'memory_check', 'memory_reason', 'memory_target')}, 'target_xy': f(est['target_xy']), 'held_xyz': f(est['held_xyz']), 'offset_xy': f(est['offset_xy'])}


# ================================================================ simulator side (not run in Phase A; mock-tested only)
ARMS = ('C0', 'VISION_CENTER_V1', 'VISION_CENTER_V2', 'CENTER_ORACLE')
ROLE = 'vision_center_v2_dev'; SET = 'vision_center_v2_pilot'; RANGE = (50900, 51099); REPEAT = 0
INTERVENTION = 'vision_center_v2_xy'


def observe(r):
    """The inputs of v2 at one step: both RGB images, the TCP pose and the finger opening of the policy's own observation, and the last executed gripper command."""
    from .trees import as_numpy
    from . import recovery_v1_oracle as O
    hand = np.asarray(as_numpy(r.raw['sensor_data']['hand_camera']['rgb'])).reshape(V1.IMG, V1.IMG, 3); base = np.asarray(as_numpy(r.raw['sensor_data']['base_camera']['rgb'])).reshape(V1.IMG, V1.IMG, 3)
    pose = np.asarray(as_numpy(r.raw['extra']['tcp_pose']), float).reshape(-1)[:7]; last = r.policy.last_action; grip = None if last is None else float(np.asarray(last).reshape(-1)[-1])
    return hand, base, pose, O.finger_width(r.raw), grip


def vision_center_v2(r, mem, frames=None):
    """At the trigger state, with the memory built on the way there. Abstain: nothing is touched. Act: the same bounded
    horizontal move as v1 and the oracle pilot (per-step bound, target cap, step budget, held gripper command, zero vertical
    command), closed on the current frame where it is accepted and on the memory otherwise."""
    from . import recovery_v1_oracle as O
    if r.intervention is not None: raise ValueError('Invalid/repeated intervention')
    hand, base, pose, width, last = observe(r); rd = V1.reading(hand, pose); mem.note_hold(r.t, rd['hold_fit'], width)
    est = estimate(rd, mem, hand, pose, width, last); d = decide(est, r.horizon - r.t); log = [{'t': int(r.t), **summary(est)}]
    if frames is not None: frames.append(np.concatenate([base, hand], 1))
    info = {'mode': 'VISION_CENTER_V2', 'decision': d['reason'], 'acted': bool(d['act']), 'trigger_step': int(r.t), 'readings': log, 'steps_used': 0, 'end_reason': None, 'constants': dict(DECISION), 'memory': dict(MEMORY),
            'memory_log': dict(mem.log), 'used_memory': bool(est['target_src'] not in (None, 'frame') or est['hold_src'] == 'memory')}
    if not d['act']: return info
    grip = O.held_grip(r); scale = O.arm_scale(r); start = np.asarray(r.tcp_history[-1], float).copy(); target0 = est['target_xy'].copy(); hold0 = est['h'].copy()
    r.intervention = INTERVENTION; r.intervention_t = r.t; r.policy.clear(); used = 0; path = 0.; why = None; trail = []
    while why is None:
        e = est['offset_xy']; n = float(np.linalg.norm(e))
        if n <= stop_tol(est): why = 'within_tolerance'
        elif r.done: why = 'episode_ended'
        elif used >= O.MAX_STEPS: why = 'step_budget'
        else:
            cur = np.asarray(r.tcp_history[-1], float)[:2]; goal = cur + e * min(1., O.STEP_CMD_M / n); off = goal - start[:2]; m = float(np.linalg.norm(off))
            if m > O.TARGET_CAP_M: goal = start[:2] + off * (O.TARGET_CAP_M / m)
            step = goal - cur; s = float(np.linalg.norm(step))
            if s > O.STEP_CMD_M: step = step * (O.STEP_CMD_M / s); s = O.STEP_CMD_M
            reach = float(np.linalg.norm(cur + step - start[:2]))
            if s < O.MIN_CMD_M or reach > O.TARGET_CAP_M + 1e-9: why = 'target_cap'
            else:
                r.advance(O._xy_action(r, step, scale, grip)); used += 1; path += s
                trail.append({'commanded_xy_m': step.tolist(), 'target_distance_from_start_m': reach, 'estimated_offset_before_step_m': n, 'tcp_after': np.asarray(r.tcp_history[-1], float).tolist()})
                if r.done: why = 'episode_ended'; break
                hand, base, pose, width, last = observe(r); rd = V1.reading(hand, pose); mem.note_hold(r.t, rd['hold_fit'], width); prev_src = est['target_src']
                if frames is not None: frames.append(np.concatenate([base, hand], 1))
                if rd['target_fresh'] and float(np.linalg.norm(rd['target_fit']['xy'] - target0)) > TARGET_JUMP_MAX_M + (0.0 if prev_src == 'frame' else est['u']): why = 'target_moved'
                est = estimate(rd, mem, hand, pose, width, grip); log.append({'t': int(r.t), **summary(est)})
                if why is not None: break
                if est['why'] is not None: why = 'estimate_lost'
                elif est['held_xyz'][2] < CARRY_Z_MIN_M: why = 'low_carry'
                elif float(np.linalg.norm((est['h'] - hold0)[:2])) > HOLD_SHIFT_MAX_M: why = 'hold_shifted'
    final = np.asarray(r.tcp_history[-1], float)
    info.update(steps_used=used, end_reason=why, start_tcp=start.tolist(), final_tcp=final.tolist(), commanded_path_m=path, tcp_xy_displacement_m=float(np.linalg.norm(final[:2] - start[:2])),
                tcp_dz_m=float(final[2] - start[2]), gripper_command=grip, steps=trail, estimated_offset_before_m=log[0]['offset_norm_m'], estimated_offset_after_m=log[-1]['offset_norm_m'],
                bounds={'step_cmd_m': O.STEP_CMD_M, 'target_cap_m': O.TARGET_CAP_M, 'max_steps': O.MAX_STEPS})
    r.retreat_info = info; r.policy.clear(); return info


def _gt_error(info, gt):
    """After the fact: the estimate at the trigger against the logged ground truth. Diagnosis only."""
    if gt is None or info is None: return None
    rd = info['readings'][0]; true = [gt['cubeB'][0] - gt['cubeA'][0], gt['cubeB'][1] - gt['cubeA'][1]]; n = float(np.hypot(*true)); e = rd.get('offset_xy')
    err = None if e is None else float(np.hypot(e[0] - true[0], e[1] - true[1])); u = rd.get('u')
    return {'true_offset_m': n, 'estimated_offset_m': rd['offset_norm_m'], 'offset_error_m': err, 'bound_m': u, 'error_within_bound': None if (err is None or u is None) else bool(err <= u),
            'target_error_m': None if rd.get('target_xy') is None else float(np.hypot(rd['target_xy'][0] - gt['cubeB'][0], rd['target_xy'][1] - gt['cubeB'][1])),
            'unnecessary_act': bool(info['acted'] and n <= RESIDUAL_TOL_M), 'missed_large_offset': bool(not info['acted'] and n >= 0.02)}


MEMORY_ERROR_M = 0.005            # reporting only: an act on a remembered position whose offset error exceeded this (or its own bound) is counted as a memory error


def collect_scene(r, seed, c, probe=None, arms=ARMS, role=ROLE, keep_trunk_frames=True):
    """Trunk (never restored) with the memory updated at every step up to the trigger; if the trigger fired, every arm in
    `arms` from the one snapshot, each vision arm with its own copy of the memory as it was at the trigger.
    Returns (record, traces, frames). A technical difference raises."""
    from . import recovery_v1_oracle as O
    import time
    arms = tuple(arms)
    if arms[0] != 'C0' or not set(arms) <= set(ARMS) or len(set(arms)) != len(arms): raise ValueError('arms must start with C0 and come from ARMS, once each')
    tol = c['audit']; started = time.perf_counter(); gate = O.OracleGate(); env_steps = 0; mem = Memory(); mem_seconds = 0.; mem_steps = 0; timeline = []
    r.reset(seed); r.repeat = REPEAT; initial = probe(r, None) if probe else None; z0 = initial['cubeA'][2] if initial else None; snap = None; traces = {}; frames = {'trunk': []}
    with O.Hooks(r, probe) as h:
        h.rows = trunk_rows = []; h.calls = trunk_calls = []
        while not r.done:
            if snap is None:
                t0 = time.perf_counter(); hand, base, pose, width, grip = observe(r); timeline.append(mem.update(r.t, hand, base, pose, width, grip)); mem_seconds += time.perf_counter() - t0; mem_steps += 1
                if keep_trunk_frames: frames['trunk'].append(np.concatenate([base, hand], 1))
                rd = O.trigger_reading(r)
                if rd['fires']:
                    snap = {'s': r.snapshot(), 'tree': O.runtime_tree(r), 'reading': rd, 'queue_len': len(r.policy.queue), 'gt': O.gt_reading(trunk_rows[-1] if trunk_rows else initial, z0), 'last_grip': float(r.policy.last_action[-1]), 'mem': mem.copy()}
            r.step_base()
        env_steps += r.t
        trunk = {'policy_noise_repeat_id': REPEAT, 'success': int(r.y), 'steps': r.t, 'reason': r.reason, 'policy_calls': r.policy.inference_calls, 'inference_calls': trunk_calls}
        trunk_ref = O._ref(r); traces['trunk'] = trunk_rows
        msum = {'steps': mem_steps, 'seconds': mem_seconds, 'log': dict(mem.log), 'hand_steps': [x['t'] for x in timeline if x['hand']], 'base_steps': [x['t'] for x in timeline if x['base']], 'hold_steps': [x['t'] for x in timeline if x['hold']]}
        if snap is None:
            cells = [{'arm': a, 'no_trigger': True, 'y': int(r.y), 'steps': r.t, 'reason': r.reason} for a in arms]
            return O.jsonable({'scene': int(seed), 'role': role, 'arms': list(arms), 'initial_state': initial, 'trunk': trunk, 'triggered': False, 'trigger': None, 'cells': cells, 'memory': msum,
                               'counts': {'continuations': 0, 'restores': 0, 'env_steps': env_steps, 'readings': 0, 'memory_steps': mem_steps, 'memory_seconds': mem_seconds}, 'oracle_reads': 0, 'wall_seconds': time.perf_counter() - started, 'completed': True}), traces, frames
        s = snap['s']; t = snap['reading']['step']; before = snap['gt']; cells = []; readings = 0; reading_seconds = 0.
        for arm in arms:
            tick = time.perf_counter(); r.restore(s); env_steps += len(s['actions']); cmp = O.compare_tree(snap['tree'], O.runtime_tree(r), tol); live = O.obs_diff(r.replay_raw, s['raw'])
            if not cmp['pass'] or not O.live_obs_ok(live, tol): raise RuntimeError(f'scene {seed} trigger {t}: the restored runtime differs from the snapshot ({cmp["failed"]}, live observation {live})')
            if O.trigger_reading(r) != snap['reading'] or len(r.policy.queue) != snap['queue_len']: raise RuntimeError(f'scene {seed} trigger {t}: the trigger reading is not reproduced after the restore')
            h.rows = rows = []; h.calls = calls = []; reads0 = gate.reads; info = None; r.repeat = REPEAT; vision = arm in ('VISION_CENTER_V1', 'VISION_CENTER_V2')
            if vision:
                frames[arm] = fr = []; t0 = time.perf_counter()
                info = V1.vision_center(r, fr) if arm == 'VISION_CENTER_V1' else vision_center_v2(r, snap['mem'].copy(), fr)
                reading_seconds += time.perf_counter() - t0; readings += len(info['readings'])
                if not info['acted']:                                           # abstaining must leave the whole runtime as captured
                    kept = O.compare_tree(snap['tree'], O.runtime_tree(r), tol)
                    if not kept['pass'] or r.intervention is not None or len(r.policy.queue) != snap['queue_len'] or r.t != t or calls: raise RuntimeError(f'scene {seed} trigger {t}: an abstaining {arm} changed the runtime ({kept["failed"]})')
            elif arm == 'CENTER_ORACLE':
                O.center_oracle(r, gate); info = O.clone(r.retreat_info)
            k = r.t - t; acted = bool(info and (arm == 'CENTER_ORACLE' or info['acted'])); cc = len(calls); q = len(r.policy.queue); early = acted and bool(r.done); reads = gate.reads - reads0
            if acted and (cc or q): raise RuntimeError(f'scene {seed} trigger {t} {arm}: policy inference during control or a queue left at the hand-back')
            if (reads > 0) != (arm == 'CENTER_ORACLE'): raise RuntimeError(f'scene {seed} trigger {t} {arm}: ground-truth cube poses were read {reads} times for control')
            if acted and info['gripper_command'] != snap['last_grip']: raise RuntimeError(f'scene {seed} trigger {t} {arm}: the gripper command of the snapshot was not held')
            after = O.gt_reading(rows[k - 1], z0) if (probe and acted and k) else (before if acted else None); res = r.complete(); env_steps += r.t - t; same = None
            if not acted:
                same = O._same_run(trunk_ref, r, tol)
                if not same['match']: raise RuntimeError(f'scene {seed} trigger {t}: {arm} without intervention does not reproduce the uninterrupted trunk ({O.jsonable(same)})')
            cells.append({'arm': arm, 'no_trigger': False, 'y': int(res['success']), 'steps': res['steps'], 'reason': res['reason'], 'acted': acted, 'intervention': res['intervention'], 'intervention_step': res['intervention_step'],
                          'control': info, 'control_steps': k if acted else 0, 'ended_during_control': bool(early), 'oracle_reads': reads, 'first_inference_step': calls[0]['t'] if calls else None,
                          'reproduces_trunk': same, 'gt_effect_of_control': O.control_effect(before, after) if (probe and acted) else None,
                          'gt_release_after_trigger': O.release_reading([trunk_rows[t - 1] if t else initial] + rows) if probe else None,
                          'gt_reading_error': (V1._gt_error(info, before) if arm == 'VISION_CENTER_V1' else _gt_error(info, before)) if vision else None, 'wall_seconds': time.perf_counter() - tick})
            traces[arm] = rows
    g = snap['gt']
    trig = {**snap['reading'], 'queue_len': snap['queue_len'], 'remaining_budget': r.horizon - t, 'held_gripper_command': snap['last_grip'],
            'gt_for_diagnosis_only': None if g is None else {**g, 'false_trigger': not (g['grasped'] and (g['cubeA_lift_m'] or 0.) >= 0.01)}}
    rec = {'scene': int(seed), 'role': role, 'arms': list(arms), 'initial_state': initial, 'trunk': trunk, 'triggered': True, 'trigger': trig, 'cells': cells, 'memory': msum,
           'counts': {'continuations': len(arms), 'restores': len(arms), 'env_steps': env_steps, 'readings': readings, 'reading_seconds': reading_seconds, 'memory_steps': mem_steps, 'memory_seconds': mem_seconds}, 'oracle_reads': gate.reads,
           'wall_seconds': time.perf_counter() - started, 'completed': True}
    rec = O.jsonable(rec); validate_scene(rec, c); return rec, traces, frames


def validate_scene(rec, c=None):
    from . import recovery_v1_oracle as O
    cells = rec['cells']; by = {x['arm']: x for x in cells}
    if [x['arm'] for x in cells] != list(rec['arms']) or rec['arms'][0] != 'C0': raise ValueError('a scene needs its declared arms, once each, C0 first')
    if c is not None and any(x['steps'] > c['env']['horizon'] for x in cells): raise ValueError('an arm exceeded the episode horizon')
    if not rec['triggered']:
        if any(not x['no_trigger'] or (x['y'], x['steps']) != (rec['trunk']['success'], rec['trunk']['steps']) for x in cells) or rec['oracle_reads']: raise ValueError('a scene without a trigger must carry the trunk outcome in every arm')
        return True
    x = by['C0']
    if (x['y'], x['steps']) != (rec['trunk']['success'], rec['trunk']['steps']) or x['acted'] or x['oracle_reads'] or not x['reproduces_trunk']['match']: raise ValueError('C0 does not reproduce the uninterrupted trunk')
    for arm, name, act_min in (('VISION_CENTER_V1', V1.INTERVENTION, V1.ACT_MIN_M), ('VISION_CENTER_V2', INTERVENTION, ACT_MIN_M)):
        v = by.get(arm)
        if v is None: continue
        ctl = v['control']
        if v['oracle_reads']: raise ValueError(f'ground truth reached {arm}')
        if ctl['acted'] != v['acted'] or (ctl['decision'] == 'ACT') != v['acted']: raise ValueError(f'{arm} decision and action disagree')
        if not v['acted'] and ((v['y'], v['steps']) != (x['y'], x['steps']) or v['intervention'] is not None): raise ValueError(f'an abstaining {arm} differs from C0')
        if v['acted']:
            if v['intervention'] != name or ctl['steps_used'] > O.MAX_STEPS or len(ctl['steps']) != ctl['steps_used'] or ctl['gripper_command'] != rec['trigger']['held_gripper_command']: raise ValueError(f'{arm} left its step bound or its record is inconsistent')
            if any(np.hypot(*s['commanded_xy_m']) > O.STEP_CMD_M + 1e-9 or s['target_distance_from_start_m'] > O.TARGET_CAP_M + 1e-9 for s in ctl['steps']): raise ValueError(f'a {arm} command exceeded a bound')
            first = ctl['readings'][0]
            if first['offset_norm_m'] < act_min: raise ValueError(f'{arm} acted below its own threshold')
            if arm == 'VISION_CENTER_V2' and first['offset_norm_m'] - first['u'] < RESIDUAL_TOL_M: raise ValueError('VISION_CENTER_V2 acted inside its own uncertainty')
    o = by.get('CENTER_ORACLE')
    if o is not None and (not o['acted'] or o['oracle_reads'] < 1 or o['control']['steps_used'] > O.MAX_STEPS): raise ValueError('CENTER_ORACLE record is inconsistent')
    return True


# ---------------------------------------------------------------- criteria (PROPOSED in Phase A; they become fixed when the plan that carries them is approved by hash, before any rollout)
SHADOW_SETS = (('recovery_v1/recovery_v1_diag', (50600, 50699)), ('recovery_v1/recovery_v1_pilot_reserved', (50700, 50799)), ('vision_center/vision_center_pilot', (50800, 50899)))     # all used; development data
SHADOW_ARMS = ('C0', 'VISION_CENTER_V1', 'VISION_CENTER_V2'); SHADOW_ROLE = 'vision_center_v2_shadow_used_scenes'
SHADOW_PASS = {'v2_unnecessary_acts_max': 2, 'v2_harm_max': 2, 'v2_memory_errors_max': 2, 'v2_target_pushed_max': 1, 'v2_net_minus_v1_net_min': 3, 'v2_memory_acts_min': 5}
GATE = {'rescue_min': 6, 'net_min': 5, 'harm_max': 2, 'memory_errors_max': 1, 'oracle_recovery_min': 0.5, 'v2_rescue_minus_v1_rescue_min': 2}


def _cell(r, a): return next(x for x in r['cells'] if x['arm'] == a)


_SOURCES = (('frame', 'frame'), ('memory_hand', 'frame'), ('memory_base', 'frame'), ('frame', 'memory'), ('memory_hand', 'memory'), ('memory_base', 'memory'))


def v2_stats(records):
    """Counts for the v2 arm with ground truth as the after-the-fact label."""
    trig = [r for r in records if r['triggered']]; pairs = [(r['scene'], _cell(r, 'VISION_CENTER_V2')) for r in trig]; v = [x for _, x in pairs]; acts = [x for x in v if x['acted']]; mem_acts = [(s, x) for s, x in pairs if x['acted'] and x['control']['used_memory']]
    bad = lambda x: bool(x['gt_reading_error'] and x['gt_reading_error']['offset_error_m'] is not None and (x['gt_reading_error']['offset_error_m'] > MEMORY_ERROR_M or x['gt_reading_error']['error_within_bound'] is False))
    err = [x['gt_reading_error']['offset_error_m'] for x in acts if x['gt_reading_error'] and x['gt_reading_error']['offset_error_m'] is not None]
    stat = lambda a: {'n': len(a), 'median': float(np.median(a)), 'max': float(np.max(a))} if a else {'n': 0}; src = lambda x: (x['control']['readings'][0]['target_src'], x['control']['readings'][0]['hold_src'])
    return {'triggered': len(trig), 'decisions': {k: sum(1 for x in v if x['control']['decision'] == k) for k in ('ACT',) + ABSTAIN}, 'acts': len(acts), 'acts_using_memory': len(mem_acts),
            'acts_by_source': {f'target_{a}/hold_{b}': sum(1 for x in acts if src(x) == (a, b)) for a, b in _SOURCES}, 'abstain_rate_of_triggered': (len(v) - len(acts)) / len(v) if v else None,
            'memory_errors': sum(1 for _, x in mem_acts if bad(x)), 'memory_error_scenes': [s for s, x in mem_acts if bad(x)],
            'unnecessary_acts': sum(1 for x in acts if x['gt_reading_error'] and x['gt_reading_error']['unnecessary_act']), 'missed_large_offsets': sum(1 for x in v if x['gt_reading_error'] and x['gt_reading_error']['missed_large_offset']),
            'offset_error_at_act_m': stat(err), 'target_pushed_more_than_1mm': sum(1 for x in acts if x['gt_effect_of_control'] and x['gt_effect_of_control']['cubeB_xy_displacement_m'] > 0.001),
            'end_reasons': {k: sum(1 for x in acts if x['control']['end_reason'] == k) for k in sorted({x['control']['end_reason'] for x in acts})}}


def shadow_verdict(records):
    """Stage B1 on USED scenes (development data): v2 against C0 and against v1. Decides only whether the new scenes are run."""
    from . import recovery_v1_oracle as O
    y = {a: {r['scene']: _cell(r, a)['y'] for r in records} for a in SHADOW_ARMS}; p2 = O.paired(y['C0'], y['VISION_CENTER_V2']); p1 = O.paired(y['C0'], y['VISION_CENTER_V1']); m = v2_stats(records); k = SHADOW_PASS
    crit = {'unnecessary_acts_at_most': m['unnecessary_acts'] <= k['v2_unnecessary_acts_max'], 'harm_at_most': p2['harm'] <= k['v2_harm_max'], 'memory_errors_at_most': m['memory_errors'] <= k['v2_memory_errors_max'],
            'target_pushed_at_most': m['target_pushed_more_than_1mm'] <= k['v2_target_pushed_max'], 'net_above_v1_by_at_least': p2['net'] - p1['net'] >= k['v2_net_minus_v1_net_min'], 'memory_acts_at_least': m['acts_using_memory'] >= k['v2_memory_acts_min']}
    return {'scenes': len(records), **m, 'paired_V2_vs_C0': p2, 'paired_V1_vs_C0': p1, 'paired_V2_vs_V1': O.paired(y['VISION_CENTER_V1'], y['VISION_CENTER_V2']), 'criteria': crit, 'pass': bool(all(crit.values())), 'thresholds': dict(k),
            'label': 'V2_SHADOW_PASS' if all(crit.values()) else 'V2_SHADOW_BLOCKED', 'scope': 'used development scenes on which v2 was designed; decides only whether the new scenes are run. A failure stops the work: no constant is tuned and re-tried under this plan'}


def summarize(records, planned):
    """The pilot on new scenes: all planned scenes in every denominator."""
    from . import recovery_v1_oracle as O
    by = {r['scene']: r for r in records}
    if sorted(by) != sorted(int(s) for s in planned) or len(by) != len(records): raise ValueError('the records are not exactly the planned scenes; a partial run is not summarised')
    y = {a: {s: _cell(r, a)['y'] for s, r in by.items()} for a in ARMS}; n = len(records); p2 = O.paired(y['C0'], y['VISION_CENTER_V2']); p1 = O.paired(y['C0'], y['VISION_CENTER_V1']); po = O.paired(y['C0'], y['CENTER_ORACLE']); m = v2_stats(records)
    both = len(set(p2['rescue_scenes']) & set(po['rescue_scenes'])); rec = (both / po['rescue']) if po['rescue'] else None; trig = [r for r in records if r['triggered']]; v1 = [_cell(r, 'VISION_CENTER_V1') for r in trig]
    g = {'rescue_at_least': p2['rescue'] >= GATE['rescue_min'], 'net_at_least': p2['net'] >= GATE['net_min'], 'harm_at_most': p2['harm'] <= GATE['harm_max'], 'memory_errors_at_most': m['memory_errors'] <= GATE['memory_errors_max'],
         'oracle_recovery_at_least': rec is not None and rec >= GATE['oracle_recovery_min'], 'rescues_above_v1_by_at_least': p2['rescue'] - p1['rescue'] >= GATE['v2_rescue_minus_v1_rescue_min']}
    return {'role': ROLE, 'unit': 'scene', 'scenes': n, 'denominators': {'all': n, 'no_trigger': n - len(trig), 'triggered': len(trig), 'c0_success': sum(y['C0'].values()), 'c0_failure': n - sum(y['C0'].values())},
            'success': {a: {'k': sum(y[a].values()), 'n': n, 'wilson95': O.wilson(sum(y[a].values()), n)} for a in ARMS},
            'primary_V2_vs_C0': p2, 'baseline_V1_vs_C0': p1, 'V2_vs_V1': O.paired(y['VISION_CENTER_V1'], y['VISION_CENTER_V2']), 'reference_CENTER_ORACLE_vs_C0': po,
            'oracle_rescue_recovery': {'v2': both, 'v1': len(set(p1['rescue_scenes']) & set(po['rescue_scenes'])), 'oracle': po['rescue'], 'v2_share': rec},
            'v2': m, 'v1': {'decisions': {k: sum(1 for x in v1 if x['control']['decision'] == k) for k in ('ACT',) + V1.ABSTAIN}, 'abstain_rate_of_triggered': (sum(1 for x in v1 if not x['acted']) / len(v1)) if v1 else None},
            'followup_gate': {'criteria': g, 'label': 'V2_GATE_PASS' if all(g.values()) else 'NO_V2_GATE_PASS', 'thresholds': dict(GATE), 'scope': 'engineering gate, not a significance test'},
            'scope': ['One task, one policy, one noise repeat, 200 development scenes, simulator images with flat colours and exact camera parameters. No significance claim.',
                      'Both vision arms use RGB and robot state only; ground truth appears only in after-the-fact columns.', 'CENTER_ORACLE is the non-deployable reference, unchanged.']}


def pilot_seeds(manifest, used):
    ids = list(manifest['splits'][SET])
    if ids != list(range(RANGE[0], RANGE[1] + 1)): raise ValueError(f'{SET} is not the frozen range {RANGE[0]}-{RANGE[1]}')
    clash = {src: sorted(set(v) & set(ids))[:10] for src, v in used.items()}; clash = {k: v for k, v in clash.items() if v}
    if clash: raise ValueError(f'pilot seeds have a recorded use: {clash}. The manifest is not patched silently')
    return ids


def projected_cost(*, seconds_per_env_step, seconds_per_reading, seconds_per_memory_step, usd_per_hour, spent_usd, cap_usd, scenes, arms, vision_arms=2, fixed_seconds=600, reserve_usd=0.20, margin=1.25):
    """Worst case: every scene at the step bound, the memory updated on all 200 trunk steps, every vision control part at its 12 steps. Time and money only."""
    steps = 200 + arms * 200; per = steps * seconds_per_env_step + 200 * seconds_per_memory_step + vision_arms * 13 * seconds_per_reading; hours = (scenes * per + fixed_seconds) * margin / 3600; total = spent_usd + hours * usd_per_hour + reserve_usd
    return {'projected_remaining_hours': hours, 'projected_total_usd': total, 'cap_usd': cap_usd, 'fits_cap': bool(total <= cap_usd),
            'inputs': {'seconds_per_env_step': seconds_per_env_step, 'seconds_per_reading': seconds_per_reading, 'seconds_per_memory_step': seconds_per_memory_step, 'usd_per_hour': usd_per_hour, 'spent_usd': spent_usd, 'scenes': scenes, 'arms': arms,
                       'vision_arms': vision_arms, 'steps_per_scene_bound': steps, 'fixed_seconds': fixed_seconds, 'reserve_usd': reserve_usd, 'margin': margin},
            'rule': 'if the projection exceeds the cap the stage does not start; scenes, arms and bounds are not reduced'}
