"""Vision-CENTER v2, Phase C1 (work id vision_center_v2_c1_001): component ablation and a second base policy.
PREPARATION: the decisions are replayed offline on stored frames; the simulator side is mock-tested only.

hci.vision_center (v1), hci.vision_center_v2 (v2) and hci.recovery_v1_oracle are imported and NOT edited. The arms
VISION_CENTER_V1, VISION_CENTER_V2 and CENTER_ORACLE call their original functions. The ablation arms call the original
v2 controller (`V2.vision_center_v2`) with a memory in which one component is switched off; only V2_RULES_OFF and the
separate safety candidate (hci.vision_center_v2s) run through `center_move` below, a copy of the v2 control loop that a test
holds equal to the original.

  V2_SINGLE_FRAME     no observation from before the trigger step: the memory is empty until the trigger and receives the
                      trigger step's observation only (hand-camera fit, else the base-camera estimate of that one frame).
                      One frame cannot give three confirmations, so this arm accepts a single base-camera estimate
                      (confirm_min = 1). That is its one rule difference from v2 and it favours this arm.
  V2_NO_HAND_MEMORY   hand-camera target fits never enter the memory (the base camera still does, on every step)
  V2_NO_BASE_MEMORY   base-camera estimates never enter the memory
  V2_NO_HOLD_MEMORY   no remembered held-cube fit: a rejected fit in the current frame means no estimate
  V2_RULES_OFF        full v2 memory, v2-only abstention rules removed as ONE bundle (uncertainty margin, side-contact rule,
                      height rule for a remembered target). The v1 rules stay. Diagnostic: it measures the bundle, it does
                      not identify any single rule.
No arm other than CENTER_ORACLE reads a cube pose, a grasp flag, a task flag or an outcome.
"""
import math, time
import numpy as np
from . import vision_center as V1
from . import vision_center_v2 as V2

WORK = 'vision_center_v2_c1_001'
ROLE = 'vision_center_v2_c1_dev'; SET = 'vision_center_v2_c1_dev'; RANGE = (51100, 51299); REPEAT = 0
BASE_ARMS = ('C0', 'VISION_CENTER_V1', 'V2_SINGLE_FRAME', 'VISION_CENTER_V2', 'CENTER_ORACLE', 'V2_NO_HAND_MEMORY', 'V2_NO_BASE_MEMORY', 'V2_NO_HOLD_MEMORY', 'V2_RULES_OFF')
SAFETY_ARM = 'VISION_CENTER_V2S'
ARMS = BASE_ARMS + (SAFETY_ARM,)
MEMORY_VARIANTS = {'V2_NO_HAND_MEMORY': {'use_hand': False}, 'V2_NO_BASE_MEMORY': {'use_base': False}, 'V2_NO_HOLD_MEMORY': {'use_hold': False}}
SINGLE_FRAME = {'confirm_min': 1}
V2_LIKE = ('V2_SINGLE_FRAME', 'VISION_CENTER_V2', 'V2_NO_HAND_MEMORY', 'V2_NO_BASE_MEMORY', 'V2_NO_HOLD_MEMORY', 'V2_RULES_OFF', SAFETY_ARM)       # arms whose control record has the v2 layout
VISION_ARMS = ('VISION_CENTER_V1',) + V2_LIKE
RULES_OFF_REMOVED = ('ALIGNED_WITHIN_UNCERTAINTY', 'SIDE_CONTACT_RISK', 'LOW_CARRY for a remembered target below the top face')
POLICIES = ('train_seed_1', 'train_seed_2')


# ---------------------------------------------------------------- one step's observation, fits computed at most once
class StepObs:
    """The inputs of v2 at one step (both RGB images, TCP pose, finger opening, last executed gripper command, step) and
    the three image fits, each computed on first use and shared by every memory that is updated on this step."""

    def __init__(self, t, hand, base, pose, width, grip):
        self.t = int(t); self.hand = hand; self.base = base; self.pose = np.asarray(pose, float).reshape(-1); self.width = float(width); self.grip = None if grip is None else float(grip); self._c = {}

    def _get(self, k, f):
        if k not in self._c: self._c[k] = f()
        return self._c[k]

    @property
    def tf(self): return self._get('tf', lambda: V1.target_partial_fit(self.hand, V1.cam_hand(self.pose), 'hand_camera'))

    @property
    def tb(self): return self._get('tb', lambda: V2.target_base(self.base) if self.base is not None else None)

    @property
    def hf(self): return self._get('hf', lambda: V1.hold_fit(self.hand))


def reading_from(so, target=None):
    """hci.vision_center.reading on a StepObs, without computing the fits a second time (a test holds the two equal)."""
    p = so.pose; tf = so.tf; hf = so.hf; fresh = bool(tf is not None and tf['valid']); txy = tf['xy'] if fresh else (None if target is None else np.asarray(target, float))
    out = {'target_fit': tf, 'hold_fit': hf, 'target_fresh': fresh, 'target_xy': txy, 'held_xyz': None, 'offset_xy': None, 'offset_norm_m': None}
    if hf is not None and hf['valid']:
        A = p[:3] + V1.quat_R(p[3:7]) @ hf['h']; out['held_xyz'] = A
        if txy is not None: out['offset_xy'] = txy - A[:2]; out['offset_norm_m'] = float(np.linalg.norm(out['offset_xy']))
    return out


# ---------------------------------------------------------------- memory with switchable components
class Mem(V2.Memory):
    """hci.vision_center_v2.Memory with components that can be switched off. `update_obs` is `Memory.update` on a StepObs;
    with every component on it gives the same memory as the original (held equal by a test, and checked against the
    original memory in every scene of a run)."""

    def __init__(self, use_hand=True, use_base=True, use_hold=True, confirm_min=V2.CONFIRM_MIN, u_hand=V2.U_HAND_M, u_base=V2.U_BASE_M):
        super().__init__(); self.use_hand = bool(use_hand); self.use_base = bool(use_base); self.use_hold = bool(use_hold); self.confirm_min = int(confirm_min); self.u_hand = float(u_hand); self.u_base = float(u_base)

    def _blank(self): return type(self)(self.use_hand, self.use_base, self.use_hold, self.confirm_min, self.u_hand, self.u_base)

    def copy(self):
        m = self._blank(); m.steps = self.steps; m.log = {k: (list(v) if isinstance(v, list) else v) for k, v in self.log.items()}
        m.target = None if self.target is None else {**self.target, 'xy': self.target['xy'].copy()}
        m.hold = None if self.hold is None else {**self.hold, 'h': self.hold['h'].copy()}
        return m

    def update_obs(self, so):
        t = so.t; p = so.pose; self.steps += 1; out = {'t': int(t), 'hand': False, 'base': False, 'hold': None}
        tf = so.tf if self.use_hand else None; tb = so.tb if self.use_base else None
        if tf is not None and tf['valid']: self._observe_target(t, tf['xy'], self.u_hand, 'hand'); self.log['hand_obs'] += 1; out['hand'] = True
        elif tb is not None and tb['valid']: self._observe_target(t, tb['xy'], self.u_base, 'base'); self.log['base_obs'] += 1; out['base'] = True
        elif self.target is not None and float(np.linalg.norm(p[:2] - self.target['xy'])) < V2.CONTACT_RADIUS_M and p[2] < V2.CONTACT_TCP_Z_M:
            self.target['exposed'] += 1
        closed = so.grip is not None and so.grip < 0 and so.width < V2.HOLD_WIDTH_MAX_M
        if self.hold is not None and (not closed or abs(so.width - self.hold['width']) > V2.FINGER_TOL_M): self.hold = None; self.log['hold_dropped'] += 1
        if self.use_hold and closed and t % V2.HOLD_EVERY == 0:
            hf = so.hf; self.log['hold_fits_run'] += 1; out['hold'] = bool(hf is not None and hf['valid'])
            if out['hold']: self.hold = {'h': np.asarray(hf['h'], float).copy(), 'yaw': float(hf['yaw']), 't': int(t), 'width': so.width}; self.log['hold_obs'] += 1
        return out

    def note_hold(self, t, hf, finger_width):
        if self.use_hold: super().note_hold(t, hf, finger_width)

    def hold_estimate(self, finger_width, last_grip):
        return super().hold_estimate(finger_width, last_grip) if self.use_hold else None

    def target_estimate(self):
        tr = self.target
        if tr is None: return None, 'never_seen'
        if tr['n'] < self.confirm_min: return None, 'not_confirmed'
        return {'xy': tr['xy'].copy(), 'u': tr['u'] + (V2.U_PUSH_M if tr['exposed'] > 0 else 0.0), 'src': tr['src'], 't_last': tr['t_last'], 'exposed_steps': tr['exposed'], 'confirmations': tr['n']}, None


def same_memory(a, b):
    """True when two memories hold the same target track, hold and counters (the mirrored update against the original)."""
    def tr(m): return None if m.target is None else {**m.target, 'xy': [float(x) for x in m.target['xy']]}
    def ho(m): return None if m.hold is None else {**m.hold, 'h': [float(x) for x in m.hold['h']]}
    keys = ('hand_obs', 'base_obs', 'conflicts', 'hold_obs', 'hold_dropped', 'hold_fits_run')
    return tr(a) == tr(b) and ho(a) == ho(b) and a.steps == b.steps and all(a.log[k] == b.log[k] for k in keys)


def single_frame_memory(so):
    """The memory of V2_SINGLE_FRAME at the trigger: nothing from before, the trigger step's observation only."""
    m = Mem(**SINGLE_FRAME); m.update_obs(so); return m


# ---------------------------------------------------------------- V2_RULES_OFF: the v1 rules only, on the v2 estimate
def decide_rules_off(est, steps_left):
    if est['why'] is not None: why = est['why']
    elif est['held_xyz'][2] < V2.CARRY_Z_MIN_M: why = 'LOW_CARRY'
    elif steps_left < V2.STEPS_LEFT_MIN: why = 'TOO_FEW_STEPS'
    elif est['offset_norm_m'] < V2.ACT_MIN_M: why = 'ALIGNED'
    else: why = 'ACT'
    return {'act': why == 'ACT', 'reason': why}


# ---------------------------------------------------------------- the v2 control loop with a replaceable decision and an optional memory update
def center_move(r, mem, frames=None, *, mode, decide_fn=V2.decide, estimate_fn=V2.estimate, stop_tol_fn=V2.stop_tol, update_memory=False, intervention=V2.INTERVENTION, at_trigger=None, extra=None):
    """`V2.vision_center_v2` with three openings: the decision function, whether the memory is updated on every control
    step (and the move stopped when the estimate is then refused), and a hook run on the memory at the trigger.
    With the defaults it is the original function (a test compares the two step by step)."""
    from . import recovery_v1_oracle as O
    if r.intervention is not None: raise ValueError('Invalid/repeated intervention')
    hand, base, pose, width, last = V2.observe(r); so = StepObs(r.t, hand, base, pose, width, last)
    if at_trigger is not None: at_trigger(mem, so)
    rd = reading_from(so); mem.note_hold(r.t, rd['hold_fit'], width)
    est = estimate_fn(rd, mem, hand, pose, width, last); d = decide_fn(est, r.horizon - r.t); log = [{'t': int(r.t), **V2.summary(est)}]
    if frames is not None: frames.append(np.concatenate([base, hand], 1))
    info = {'mode': mode, 'decision': d['reason'], 'acted': bool(d['act']), 'trigger_step': int(r.t), 'readings': log, 'steps_used': 0, 'end_reason': None, 'constants': dict(V2.DECISION), 'memory': dict(V2.MEMORY),
            'memory_log': dict(mem.log), 'used_memory': bool(est['target_src'] not in (None, 'frame') or est['hold_src'] == 'memory'), 'memory_updated_during_control': bool(update_memory), **(extra or {})}
    if not d['act']: return info
    grip = O.held_grip(r); scale = O.arm_scale(r); start = np.asarray(r.tcp_history[-1], float).copy(); target0 = est['target_xy'].copy(); hold0 = est['h'].copy()
    r.intervention = intervention; r.intervention_t = r.t; r.policy.clear(); used = 0; path = 0.; why = None; trail = []
    while why is None:
        e = est['offset_xy']; n = float(np.linalg.norm(e))
        if n <= stop_tol_fn(est): why = 'within_tolerance'
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
                hand, base, pose, width, last = V2.observe(r); so = StepObs(r.t, hand, base, pose, width, grip); prev_src = est['target_src']; prev_u = est['u']
                if update_memory: mem.update_obs(so)
                rd = reading_from(so); mem.note_hold(r.t, rd['hold_fit'], width)
                if frames is not None: frames.append(np.concatenate([base, hand], 1))
                if rd['target_fresh'] and float(np.linalg.norm(rd['target_fit']['xy'] - target0)) > V2.TARGET_JUMP_MAX_M + (0.0 if prev_src == 'frame' else prev_u): why = 'target_moved'
                est = estimate_fn(rd, mem, hand, pose, width, grip); log.append({'t': int(r.t), **V2.summary(est)})
                if why is not None: break
                if est['why'] is not None: why = 'estimate_lost'
                elif est['held_xyz'][2] < V2.CARRY_Z_MIN_M: why = 'low_carry'
                elif float(np.linalg.norm((est['h'] - hold0)[:2])) > V2.HOLD_SHIFT_MAX_M: why = 'hold_shifted'
    final = np.asarray(r.tcp_history[-1], float)
    info.update(steps_used=used, end_reason=why, start_tcp=start.tolist(), final_tcp=final.tolist(), commanded_path_m=path, tcp_xy_displacement_m=float(np.linalg.norm(final[:2] - start[:2])),
                tcp_dz_m=float(final[2] - start[2]), gripper_command=grip, steps=trail, estimated_offset_before_m=log[0]['offset_norm_m'], estimated_offset_after_m=log[-1]['offset_norm_m'],
                bounds={'step_cmd_m': O.STEP_CMD_M, 'target_cap_m': O.TARGET_CAP_M, 'max_steps': O.MAX_STEPS}, memory_log_after=dict(mem.log))
    r.retreat_info = info; r.policy.clear(); return info


# ---------------------------------------------------------------- what a vision arm can reach through the runtime
class PrivilegedAccess(RuntimeError):
    """A vision arm asked the runtime for something outside its declared inputs."""


RAW_PATHS = frozenset({('sensor_data', 'hand_camera', 'rgb'), ('sensor_data', 'base_camera', 'rgb'), ('extra', 'tcp_pose'), ('agent', 'qpos')})
ARM_READS = frozenset({'t', 'horizon', 'intervention', 'intervention_t', 'done', 'raw', 'tcp_history', 'adim', 'advance', 'policy', 'env', 'retreat_info'})
ARM_WRITES = frozenset({'intervention', 'intervention_t', 'retreat_info'})
POLICY_READS = frozenset({'last_action', 'clear'})
# `done` is the episode-end flag (success, termination or horizon). A move stops on it because Runtime.advance refuses a step
# after the end; the outcome is absorbing, so stopping cannot change it. `env` reaches only the arm controller's
# configuration and the robot base pose (`recovery_v1_oracle.arm_scale`).


class _Leaf:
    def __init__(self, obj, path, allowed, log, name):
        object.__setattr__(self, '_o', obj); object.__setattr__(self, '_p', path); object.__setattr__(self, '_a', allowed); object.__setattr__(self, '_l', log); object.__setattr__(self, '_n', name)

    def _go(self, k, get):
        """The permission is checked before the object is touched."""
        p = self._p + (k,); full = '.'.join(map(str, (self._n, *p) if self._n else p)); self._l.add(full)
        if p in self._a: return get()
        if any(a[:len(p)] == p for a in self._a): return _Leaf(get(), p, self._a, self._l, self._n)
        raise PrivilegedAccess('vision arm read ' + full)

    def __getitem__(self, k): return self._go(k, lambda: self._o[k])
    def __getattr__(self, k): return self._go(k, lambda: getattr(self._o, k))
    def __setattr__(self, k, v): raise PrivilegedAccess(f'vision arm wrote {".".join(map(str, (self._n, *self._p, k) if self._n else (*self._p, k)))}')


ENV_PATHS = frozenset({('env', 'agent', 'controller', 'controllers', 'arm', 'config'), ('env', 'agent', 'robot', 'pose')})


class _PolicyView:
    def __init__(self, p, log): object.__setattr__(self, '_o', p); object.__setattr__(self, '_l', log)

    def __getattr__(self, k):
        self._l.add('policy.' + k)
        if k not in POLICY_READS: raise PrivilegedAccess('vision arm read policy.' + k)
        return getattr(self._o, k)

    def __setattr__(self, k, v): raise PrivilegedAccess('vision arm wrote policy.' + k)


class ArmView:
    """The runtime as a vision arm sees it: the declared inputs only (both RGB images, TCP pose, finger joints, executed
    gripper command, step counter, TCP history) and the commands it may give. Anything else raises PrivilegedAccess.
    `reads` records every name the arm used."""
    def __init__(self, r):
        object.__setattr__(self, '_r', r); object.__setattr__(self, 'reads', set())

    def __getattr__(self, k):
        r = object.__getattribute__(self, '_r'); log = object.__getattribute__(self, 'reads'); log.add(k)
        if k not in ARM_READS: raise PrivilegedAccess('vision arm read runtime.' + k)
        if k == 'raw': return _Leaf(r.raw, (), RAW_PATHS, log, 'raw')
        if k == 'env': return _Leaf(r.env, ('env',), ENV_PATHS, log, '')
        if k == 'policy': return _PolicyView(r.policy, log)
        if k == 'tcp_history': return tuple(r.tcp_history)
        return getattr(r, k)

    def __setattr__(self, k, v):
        if k not in ARM_WRITES: raise PrivilegedAccess('vision arm wrote runtime.' + k)
        setattr(object.__getattribute__(self, '_r'), k, v)


def run_vision_arm(r, arm, mems, frames):
    """One vision arm at the restored trigger state. `mems`: the memories as they were at the trigger (never mutated here)."""
    if arm == 'VISION_CENTER_V1': return V1.vision_center(r, frames)
    if arm == 'VISION_CENTER_V2': return V2.vision_center_v2(r, mems['VISION_CENTER_V2'].copy(), frames)                 # the original function on the original memory
    if arm in MEMORY_VARIANTS or arm == 'V2_SINGLE_FRAME':
        info = V2.vision_center_v2(r, mems[arm].copy(), frames); info['mode'] = arm; return info                         # the original controller on an ablated memory
    if arm == 'V2_RULES_OFF': return center_move(r, mems['FULL'].copy(), frames, mode=arm, decide_fn=decide_rules_off, extra={'rules_removed': list(RULES_OFF_REMOVED)})
    if arm == SAFETY_ARM:
        from . import vision_center_v2s as S
        return S.vision_center_v2s(r, mems[SAFETY_ARM].copy(), frames)
    raise ValueError(arm)


def intervention_of(arm):
    if arm == 'VISION_CENTER_V1': return V1.INTERVENTION
    if arm == SAFETY_ARM:
        from . import vision_center_v2s as S
        return S.INTERVENTION
    return V2.INTERVENTION


# ---------------------------------------------------------------- one scene
def collect_scene(r, seed, c, probe=None, arms=ARMS, role=ROLE, policy_id=None):
    """Trunk (never restored) with every memory updated on every step up to the trigger from ONE set of image fits; if the
    trigger fired, every arm from the one snapshot. The original v2 memory is updated by its own `Memory.update` as well and
    the mirrored memory must equal it at the trigger. Returns (record, traces, frames). A technical difference raises."""
    from . import recovery_v1_oracle as O
    arms = tuple(arms)
    if arms[0] != 'C0' or not set(arms) <= set(ARMS) or len(set(arms)) != len(arms): raise ValueError('arms must start with C0 and come from ARMS, once each')
    tol = c['audit']; started = time.perf_counter(); gate = O.OracleGate(); env_steps = 0; mem_seconds = 0.; mem_steps = 0; timeline = []
    orig = V2.Memory(); mems = {'FULL': Mem(), **{a: Mem(**kw) for a, kw in MEMORY_VARIANTS.items() if a in arms}}
    if SAFETY_ARM in arms:
        from . import vision_center_v2s as S
        mems[SAFETY_ARM] = S.MemS()
    r.reset(seed); r.repeat = REPEAT; initial = probe(r, None) if probe else None; z0 = initial['cubeA'][2] if initial else None; snap = None; traces = {}; frames = {'trunk': []}
    with O.Hooks(r, probe) as h:
        h.rows = trunk_rows = []; h.calls = trunk_calls = []
        while not r.done:
            if snap is None:
                t0 = time.perf_counter(); hand, base, pose, width, grip = V2.observe(r); so = StepObs(r.t, hand, base, pose, width, grip)
                timeline.append(orig.update(r.t, hand, base, pose, width, grip))
                for m in mems.values(): m.update_obs(so)
                mem_seconds += time.perf_counter() - t0; mem_steps += 1; frames['trunk'].append(np.concatenate([base, hand], 1))
                rd = O.trigger_reading(r)
                if rd['fires']:
                    if not same_memory(orig, mems['FULL']): raise RuntimeError(f'scene {seed} trigger {r.t}: the mirrored memory differs from hci.vision_center_v2.Memory')
                    at = {k: m.copy() for k, m in mems.items()}; at['VISION_CENTER_V2'] = orig.copy()
                    if 'V2_SINGLE_FRAME' in arms: at['V2_SINGLE_FRAME'] = single_frame_memory(so)
                    snap = {'s': r.snapshot(), 'tree': O.runtime_tree(r), 'reading': rd, 'queue_len': len(r.policy.queue), 'gt': O.gt_reading(trunk_rows[-1] if trunk_rows else initial, z0), 'last_grip': float(r.policy.last_action[-1]), 'mems': at}
            r.step_base()
        env_steps += r.t
        trunk = {'policy_noise_repeat_id': REPEAT, 'success': int(r.y), 'steps': r.t, 'reason': r.reason, 'policy_calls': r.policy.inference_calls, 'inference_calls': trunk_calls}
        trunk_ref = O._ref(r); traces['trunk'] = trunk_rows
        msum = {'steps': mem_steps, 'seconds': mem_seconds, 'log': dict(orig.log), 'hand_steps': [x['t'] for x in timeline if x['hand']], 'base_steps': [x['t'] for x in timeline if x['base']], 'hold_steps': [x['t'] for x in timeline if x['hold']],
                'variant_logs': {k: {kk: vv for kk, vv in m.log.items()} for k, m in mems.items()}}
        head = {'scene': int(seed), 'role': role, 'policy_id': policy_id, 'arms': list(arms), 'initial_state': initial, 'trunk': trunk}
        if snap is None:
            cells = [{'arm': a, 'no_trigger': True, 'y': int(r.y), 'steps': r.t, 'reason': r.reason} for a in arms]
            return O.jsonable({**head, 'triggered': False, 'trigger': None, 'cells': cells, 'memory': msum,
                               'counts': {'continuations': 0, 'restores': 0, 'env_steps': env_steps, 'readings': 0, 'memory_steps': mem_steps, 'memory_seconds': mem_seconds}, 'oracle_reads': 0, 'wall_seconds': time.perf_counter() - started, 'completed': True}), traces, frames
        s = snap['s']; t = snap['reading']['step']; before = snap['gt']; cells = []; readings = 0; reading_seconds = 0.
        for arm in arms:
            tick = time.perf_counter(); r.restore(s); env_steps += len(s['actions']); cmp = O.compare_tree(snap['tree'], O.runtime_tree(r), tol); live = O.obs_diff(r.replay_raw, s['raw'])
            if not cmp['pass'] or not O.live_obs_ok(live, tol): raise RuntimeError(f'scene {seed} trigger {t}: the restored runtime differs from the snapshot ({cmp["failed"]}, live observation {live})')
            if O.trigger_reading(r) != snap['reading'] or len(r.policy.queue) != snap['queue_len']: raise RuntimeError(f'scene {seed} trigger {t}: the trigger reading is not reproduced after the restore')
            h.rows = rows = []; h.calls = calls = []; reads0 = gate.reads; info = None; r.repeat = REPEAT; vision = arm in VISION_ARMS
            if vision:
                frames[arm] = fr = []; t0 = time.perf_counter(); view = ArmView(r); info = run_vision_arm(view, arm, snap['mems'], fr); info['runtime_reads'] = sorted(view.reads); reading_seconds += time.perf_counter() - t0; readings += len(info['readings'])
                if not info['acted']:                                           # abstaining must leave the whole runtime as captured
                    kept = O.compare_tree(snap['tree'], O.runtime_tree(r), tol)
                    if not kept['pass'] or r.intervention is not None or len(r.policy.queue) != snap['queue_len'] or r.t != t or calls: raise RuntimeError(f'scene {seed} trigger {t}: an abstaining {arm} changed the runtime ({kept["failed"]})')
                elif r.intervention != intervention_of(arm): raise RuntimeError(f'scene {seed} trigger {t}: {arm} recorded the intervention {r.intervention}')
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
                          'gt_reading_error': (V1._gt_error(info, before) if arm == 'VISION_CENTER_V1' else V2._gt_error(info, before)) if vision else None, 'wall_seconds': time.perf_counter() - tick})
            traces[arm] = rows
    g = snap['gt']
    trig = {**snap['reading'], 'queue_len': snap['queue_len'], 'remaining_budget': r.horizon - t, 'held_gripper_command': snap['last_grip'],
            'gt_for_diagnosis_only': None if g is None else {**g, 'false_trigger': not (g['grasped'] and (g['cubeA_lift_m'] or 0.) >= 0.01)}}
    rec = {**head, 'triggered': True, 'trigger': trig, 'cells': cells, 'memory': msum,
           'counts': {'continuations': len(arms), 'restores': len(arms), 'env_steps': env_steps, 'readings': readings, 'reading_seconds': reading_seconds, 'memory_steps': mem_steps, 'memory_seconds': mem_seconds}, 'oracle_reads': gate.reads,
           'wall_seconds': time.perf_counter() - started, 'completed': True}
    rec = O.jsonable(rec); validate_scene(rec, c); return rec, traces, frames


def validate_scene(rec, c=None):
    from . import recovery_v1_oracle as O
    cells = rec['cells']; by = {x['arm']: x for x in cells}
    if [x['arm'] for x in cells] != list(rec['arms']) or rec['arms'][0] != 'C0' or not set(rec['arms']) <= set(ARMS): raise ValueError('a scene needs its declared arms, once each, C0 first')
    if c is not None and any(x['steps'] > c['env']['horizon'] for x in cells): raise ValueError('an arm exceeded the episode horizon')
    if not rec['triggered']:
        if any(not x['no_trigger'] or (x['y'], x['steps']) != (rec['trunk']['success'], rec['trunk']['steps']) for x in cells) or rec['oracle_reads']: raise ValueError('a scene without a trigger must carry the trunk outcome in every arm')
        return True
    x = by['C0']
    if (x['y'], x['steps']) != (rec['trunk']['success'], rec['trunk']['steps']) or x['acted'] or x['oracle_reads'] or not x['reproduces_trunk']['match']: raise ValueError('C0 does not reproduce the uninterrupted trunk')
    for arm in VISION_ARMS:
        v = by.get(arm)
        if v is None: continue
        ctl = v['control']
        if v['oracle_reads']: raise ValueError(f'ground truth reached {arm}')
        if ctl['acted'] != v['acted'] or (ctl['decision'] == 'ACT') != v['acted']: raise ValueError(f'{arm} decision and action disagree')
        if not v['acted'] and ((v['y'], v['steps']) != (x['y'], x['steps']) or v['intervention'] is not None): raise ValueError(f'an abstaining {arm} differs from C0')
        if v['acted']:
            if v['intervention'] != intervention_of(arm) or ctl['steps_used'] > O.MAX_STEPS or len(ctl['steps']) != ctl['steps_used'] or ctl['gripper_command'] != rec['trigger']['held_gripper_command']: raise ValueError(f'{arm} left its step bound or its record is inconsistent')
            if any(np.hypot(*s['commanded_xy_m']) > O.STEP_CMD_M + 1e-9 or s['target_distance_from_start_m'] > O.TARGET_CAP_M + 1e-9 for s in ctl['steps']): raise ValueError(f'a {arm} command exceeded a bound')
            first = ctl['readings'][0]
            if first['offset_norm_m'] < V2.ACT_MIN_M: raise ValueError(f'{arm} acted below its own threshold')
            if arm in V2_LIKE and arm != 'V2_RULES_OFF' and first['offset_norm_m'] - first['u'] < V2.RESIDUAL_TOL_M: raise ValueError(f'{arm} acted inside its own uncertainty')
    o = by.get('CENTER_ORACLE')
    if o is not None and (not o['acted'] or o['oracle_reads'] < 1 or o['control']['steps_used'] > O.MAX_STEPS): raise ValueError('CENTER_ORACLE record is inconsistent')
    return True


# ---------------------------------------------------------------- statistics (fixed before any C1 rollout)
ALPHA = 0.05                      # familywise, two primary hypotheses, Holm
SAFETY = {'harm_max': 2, 'memory_errors_max': 2, 'target_pushed_over_1mm_max': 1}        # per policy, for VISION_CENTER_V2 and VISION_CENTER_V2S: development criteria, not a calibration
SAFETY_ARMS = ('VISION_CENTER_V2', SAFETY_ARM)


def exact_one_sided(b, c):
    """Exact conditional paired test, one-sided for the arm: P(X >= b), X ~ Binomial(b + c, 1/2); b = scenes only the arm
    solved, c = scenes only the reference solved. No discordant scene: p = 1."""
    b = int(b); c = int(c); n = b + c
    if n == 0: return 1.0
    return float(sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n)


def discordant(y_ref, y_arm):
    s = sorted(y_ref)
    if s != sorted(y_arm): raise ValueError('the two arms do not cover the same scenes')
    return sum(1 for k in s if y_ref[k] == 0 and y_arm[k] == 1), sum(1 for k in s if y_ref[k] == 1 and y_arm[k] == 0)


def holm(ps, alpha=ALPHA):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [False] * len(ps)
    for j, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - j): out[i] = True
        else: break
    return out


def primary_tests(y):
    """y[policy][arm] = {scene: 0/1}. H_gen: v2 above C0 with train seed 2 (exact one-sided paired test over scenes).
    H_mem: v2 above V2_SINGLE_FRAME with BOTH policies: one exact paired test per policy, combined by intersection-union
    (p = the larger of the two), so the 200 shared scenes are never counted as 400. Holm over the two at ALPHA."""
    b, c = discordant(y['train_seed_2']['C0'], y['train_seed_2']['VISION_CENTER_V2']); p_gen = exact_one_sided(b, c)
    per = {}
    for pol in POLICIES:
        bb, cc = discordant(y[pol]['V2_SINGLE_FRAME'], y[pol]['VISION_CENTER_V2']); per[pol] = {'v2_only': bb, 'single_frame_only': cc, 'p_one_sided': exact_one_sided(bb, cc)}
    p_mem = max(v['p_one_sided'] for v in per.values()); rej = holm([p_gen, p_mem])
    return {'alpha_familywise': ALPHA, 'method': 'exact conditional paired test (binomial on the discordant scenes), one-sided; H_mem by intersection-union over the two policies; Holm over the two hypotheses',
            'H_gen': {'statement': 'with train seed 2, VISION_CENTER_V2 succeeds in more scenes than C0', 'v2_only': b, 'c0_only': c, 'p_one_sided': p_gen, 'rejected_null': rej[0]},
            'H_mem': {'statement': 'with each of the two policies, VISION_CENTER_V2 succeeds in more scenes than V2_SINGLE_FRAME', 'per_policy': per, 'p_intersection_union': p_mem, 'rejected_null': rej[1]}}


def _cell(r, a): return next(x for x in r['cells'] if x['arm'] == a)


def arm_stats(records, arm):
    """Counts for one v2-layout arm with ground truth as the after-the-fact label (the definitions of V2.v2_stats)."""
    trig = [r for r in records if r['triggered']]; pairs = [(r['scene'], _cell(r, arm)) for r in trig]; v = [x for _, x in pairs]; acts = [x for x in v if x['acted']]; mem_acts = [(s, x) for s, x in pairs if x['acted'] and x['control']['used_memory']]
    bad = lambda x: bool(x['gt_reading_error'] and x['gt_reading_error']['offset_error_m'] is not None and (x['gt_reading_error']['offset_error_m'] > V2.MEMORY_ERROR_M or x['gt_reading_error']['error_within_bound'] is False))
    err = [x['gt_reading_error']['offset_error_m'] for x in acts if x['gt_reading_error'] and x['gt_reading_error']['offset_error_m'] is not None]
    stat = lambda a: {'n': len(a), 'median': float(np.median(a)), 'max': float(np.max(a))} if a else {'n': 0}; src = lambda x: f"target_{x['control']['readings'][0]['target_src']}/hold_{x['control']['readings'][0]['hold_src']}"
    reasons = sorted({x['control']['decision'] for x in v})
    return {'triggered': len(trig), 'decisions': {k: sum(1 for x in v if x['control']['decision'] == k) for k in reasons}, 'acts': len(acts), 'acts_using_memory': len(mem_acts),
            'acts_by_source': {k: sum(1 for x in acts if src(x) == k) for k in sorted({src(x) for x in acts})}, 'abstain_rate_of_triggered': (len(v) - len(acts)) / len(v) if v else None,
            'memory_errors': sum(1 for _, x in mem_acts if bad(x)), 'memory_error_scenes': [s for s, x in mem_acts if bad(x)],
            'unnecessary_acts': sum(1 for x in acts if x['gt_reading_error'] and x['gt_reading_error']['unnecessary_act']), 'offset_error_at_act_m': stat(err),
            'target_pushed_more_than_1mm': sum(1 for x in acts if x['gt_effect_of_control'] and x['gt_effect_of_control']['cubeB_xy_displacement_m'] > 0.001),
            'target_pushed_scenes': [s for s, x in pairs if x['acted'] and x['gt_effect_of_control'] and x['gt_effect_of_control']['cubeB_xy_displacement_m'] > 0.001],
            'end_reasons': {k: sum(1 for x in acts if x['control']['end_reason'] == k) for k in sorted({x['control']['end_reason'] for x in acts})}}


def summarize_policy(records, planned, arms):
    """One policy: all planned scenes in every denominator."""
    from . import recovery_v1_oracle as O
    by = {r['scene']: r for r in records}
    if sorted(by) != sorted(int(s) for s in planned) or len(by) != len(records): raise ValueError('the records are not exactly the planned scenes; a partial run is not summarised')
    if any(list(r['arms']) != list(arms) for r in records): raise ValueError('a record does not carry the planned arms')
    y = {a: {s: _cell(r, a)['y'] for s, r in by.items()} for a in arms}; n = len(records); trig = [r for r in records if r['triggered']]
    out = {'scenes': n, 'denominators': {'all': n, 'no_trigger': n - len(trig), 'triggered': len(trig), 'c0_success': sum(y['C0'].values()), 'c0_failure': n - sum(y['C0'].values())},
           'success': {a: {'k': sum(y[a].values()), 'n': n, 'wilson95': O.wilson(sum(y[a].values()), n)} for a in arms}, 'against_C0': {a: O.paired(y['C0'], y[a]) for a in arms[1:]}, 'arm_stats': {a: arm_stats(records, a) for a in arms if a in V2_LIKE}}
    pairs = [('VISION_CENTER_V1', 'VISION_CENTER_V2'), ('V2_SINGLE_FRAME', 'VISION_CENTER_V2'), ('VISION_CENTER_V1', 'V2_SINGLE_FRAME'), ('V2_NO_HAND_MEMORY', 'VISION_CENTER_V2'), ('V2_NO_BASE_MEMORY', 'VISION_CENTER_V2'),
             ('V2_NO_HOLD_MEMORY', 'VISION_CENTER_V2'), ('VISION_CENTER_V2', 'V2_RULES_OFF'), ('VISION_CENTER_V2', SAFETY_ARM), ('VISION_CENTER_V2', 'CENTER_ORACLE')]
    out['paired'] = {f'{b}_against_{a}': O.paired(y[a], y[b]) for a, b in pairs if a in arms and b in arms}
    po = out['against_C0'].get('CENTER_ORACLE')
    if po: out['oracle_rescue_recovery'] = {a: {'recovered': len(set(out['against_C0'][a]['rescue_scenes']) & set(po['rescue_scenes'])), 'oracle': po['rescue']} for a in arms[1:] if a != 'CENTER_ORACLE'}
    out['safety'] = {}
    for a in SAFETY_ARMS:
        if a not in arms: continue
        m = out['arm_stats'][a]; p = out['against_C0'][a]; crit = {'harm_at_most': p['harm'] <= SAFETY['harm_max'], 'memory_errors_at_most': m['memory_errors'] <= SAFETY['memory_errors_max'], 'target_pushed_at_most': m['target_pushed_more_than_1mm'] <= SAFETY['target_pushed_over_1mm_max']}
        out['safety'][a] = {'harm': p['harm'], 'memory_errors': m['memory_errors'], 'target_pushed_over_1mm': m['target_pushed_more_than_1mm'], 'criteria': crit, 'met': bool(all(crit.values())), 'thresholds': dict(SAFETY)}
    return out, y


def summarize(records_by_policy, planned, arms):
    """Both policies, reported separately. The 200 scenes are shared; nothing here treats the 400 runs as independent."""
    if tuple(records_by_policy) != POLICIES: raise ValueError('both policies are needed, in the planned order; a partial run is not summarised')
    per = {}; y = {}
    for pol in POLICIES: per[pol], y[pol] = summarize_policy(records_by_policy[pol], planned, arms)
    net = {pol: per[pol]['against_C0']['VISION_CENTER_V2']['net'] for pol in POLICIES}; scenes = sorted(int(s) for s in planned)
    d = np.array([(y['train_seed_2']['VISION_CENTER_V2'][s] - y['train_seed_2']['C0'][s]) - (y['train_seed_1']['VISION_CENTER_V2'][s] - y['train_seed_1']['C0'][s]) for s in scenes], float)
    rng = np.random.default_rng(20261118); boot = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    return {'role': ROLE, 'unit': 'scene, within policy', 'scenes': len(scenes), 'arms': list(arms), 'per_policy': per, 'primary': primary_tests(y),
            'between_policies': {'v2_net_gain': net, 'difference_seed2_minus_seed1_per_scene': float(d.mean()), 'scene_bootstrap95': [float(x) for x in np.quantile(boot, [0.025, 0.975])],
                                 'c0_outcome_differs_between_policies': int(sum(1 for s in scenes if y['train_seed_1']['C0'][s] != y['train_seed_2']['C0'][s])),
                                 'note': 'two policy seeds give one degree of freedom for a between-seed variance: a range is reported, not a variance estimate'},
            'safety_all_met': bool(all(v['met'] for pol in POLICIES for v in per[pol]['safety'].values())),
            'scope': ['Development data. Not calibration, not Test. The primary tests were fixed before the run; everything else is descriptive.', 'One task, two policy seeds trained on the same demonstrations, one noise repeat, simulator images.',
                      'The safety criteria are development criteria for going on; they do not certify a harm rate.', 'V2_RULES_OFF measures a bundle of rules, not any single rule.']}


def pilot_seeds(manifest, used):
    ids = list(manifest['splits'][SET])
    if ids != list(range(RANGE[0], RANGE[1] + 1)): raise ValueError(f'{SET} is not the frozen range {RANGE[0]}-{RANGE[1]}')
    clash = {src: sorted(set(v) & set(ids))[:10] for src, v in used.items()}; clash = {k: v for k, v in clash.items() if v}
    if clash: raise ValueError(f'C1 seeds have a recorded use: {clash}. The manifest is not patched silently')
    return ids


def projected_cost(*, seconds_per_env_step, seconds_per_reading, seconds_per_memory_step, usd_per_hour, spent_usd, cap_usd, scene_runs, arms, vision_arms, fixed_seconds=900, reserve_usd=0.30, margin=1.25):
    """Worst case: every scene at the step bound, every memory updated on all 200 trunk steps, every vision arm at 13 readings. Time and money only."""
    steps = 200 + arms * 200; per = steps * seconds_per_env_step + 200 * seconds_per_memory_step + vision_arms * 13 * seconds_per_reading; hours = (scene_runs * per + fixed_seconds) * margin / 3600; total = spent_usd + hours * usd_per_hour + reserve_usd
    return {'projected_remaining_hours': hours, 'projected_total_usd': total, 'cap_usd': cap_usd, 'fits_cap': bool(total <= cap_usd),
            'inputs': {'seconds_per_env_step': seconds_per_env_step, 'seconds_per_reading': seconds_per_reading, 'seconds_per_memory_step': seconds_per_memory_step, 'usd_per_hour': usd_per_hour, 'spent_usd': spent_usd, 'scene_runs': scene_runs, 'arms': arms,
                       'vision_arms': vision_arms, 'steps_per_scene_bound': steps, 'fixed_seconds': fixed_seconds, 'reserve_usd': reserve_usd, 'margin': margin},
            'rule': 'if the projection exceeds the cap the pass does not start; scenes, arms and bounds are not reduced'}
