"""Vision-CENTER v2s (work id vision_center_v2_c1_001): a separate SAFETY CANDIDATE next to the unchanged v2.
PREPARATION: decisions replayed offline on stored frames; the simulator side is mock-tested only.

hci.vision_center_v2 is imported and not edited. v2s answers one finding of the post-pass audit: in scene 51009 the held
cube pushed the hidden target 38.8 mm after its last view and the memory kept the old position with a 10 mm bound.
v2 has no notion of contact; its "exposed" flag is set in 195 of 196 remembered estimates and only adds 5 mm.

What v2s changes, using the same inputs as v2 (both RGB images, TCP pose, finger opening, executed gripper commands):
  S1 contact invalidation   on every step without a view of the target, the remembered position is DROPPED (not widened)
                            when the estimated held cube, or the gripper itself, was close enough and low enough to have
                            touched the target, or when the gripper is closed on something near the target and no held-cube
                            estimate exists (contact cannot be excluded). A dropped target has to be seen again three times.
  S2 history and trigger    a remembered target needs its three confirmations from steps BEFORE the trigger step. The
                            trigger step's own view may refresh or contradict it, never confirm it.
  S3 memory during control  the memory is updated and re-checked on every control step with the same rules; when the
                            remembered target is dropped or contradicted the move stops and the policy re-plans.
  S4 base-camera bound      6.5 mm instead of 5 mm: the largest error of an accepted base-camera estimate on the 34,882
                            demonstration frames of the Phase A audit was 6.3 mm (the 5 mm of v2 covered 99.97 %).
  S5 height margin          a move on a remembered target needs the held cube's bottom Z_MARGIN_M above the target's top
                            face (v2: 0), the same margin S1 uses.
  S6 wording                an abstention because the offset to a REMEMBERED target is small is labelled
                            ALIGNED_ON_MEMORY, not ALIGNED: the cubes are not known to be aligned.
Every abstention leaves the policy's queue, history and counters untouched, as in v2.
The constants below were fixed from geometry and from data that existed before stage B2, before the offline replay of v2s
was run; they are not tuned on outcomes. No cube pose, grasp flag, task flag or outcome is read.
"""
import numpy as np
from . import vision_center as V1
from . import vision_center_v2 as V2
from . import vision_center_c1 as C1

INTERVENTION = 'vision_center_v2s_xy'
U_BASE_S_M = 0.0065               # S4
Z_MARGIN_M = V2.U_HOLD_M          # S1, S5: the bound of the held-cube estimate (0.002 m)
CUBE_REACH_M = 2 * V1.CUBE_HALF_M * np.sqrt(2)      # two 4 cm cubes can touch when their centres are closer than this horizontally (corner to corner, any yaw): 0.0566 m
FINGER_REACH_M = V1.CUBE_HALF_M * np.sqrt(2) + 0.05  # the gripper (fingers up to 0.04 m from the TCP, 0.01 m pad) against the target's corner: 0.0783 m
FINGER_Z_MARGIN_M = 0.010         # the gripper is "low" when the TCP is less than this above the target's top face
SAFETY = {'u_base_s_m': U_BASE_S_M, 'z_margin_m': Z_MARGIN_M, 'cube_reach_m': float(CUBE_REACH_M), 'finger_reach_m': float(FINGER_REACH_M), 'finger_z_margin_m': FINGER_Z_MARGIN_M,
          'unknown_hold_radius_m': V2.CONTACT_RADIUS_M, 'unknown_hold_tcp_z_m': V2.CONTACT_TCP_Z_M, 'confirm_min_before_trigger': V2.CONFIRM_MIN}
ABSTAIN = V2.ABSTAIN + ('ALIGNED_ON_MEMORY',)


class MemS(C1.Mem):
    """The v2 memory with contact invalidation (S1), history kept apart from the trigger step (S2) and the wider
    base-camera bound (S4). Reads nothing but the StepObs it is given."""

    def __init__(self, use_hand=True, use_base=True, use_hold=True, confirm_min=V2.CONFIRM_MIN, u_hand=V2.U_HAND_M, u_base=U_BASE_S_M):
        super().__init__(use_hand, use_base, use_hold, confirm_min, u_hand, u_base); self.t_trigger = None; self.log['invalidations'] = []

    def copy(self):
        m = super().copy(); m.t_trigger = self.t_trigger; return m

    def _observe_target(self, t, xy, u, src):
        before = self.target; n_prev = 0 if before is None else before['n']; hist = None if before is None else before.get('n_hist')
        super()._observe_target(t, xy, u, src); tr = self.target
        if tr is before: tr['n_prev'] = n_prev; tr['n_hist'] = hist if self.t_trigger is not None else None        # the same track went on
        else: tr['n_prev'] = 0; tr['n_hist'] = 0 if self.t_trigger is not None else None                               # a new track; one started at or after the trigger never counts as history

    def contact(self, so):
        """Could the held cube or the gripper have touched the remembered target on this step? (reason or None)"""
        tr = self.target; p = so.pose; reach_u = tr['u']; d_tcp = float(np.linalg.norm(p[:2] - tr['xy']))
        if d_tcp < FINGER_REACH_M + reach_u and p[2] < V1.TARGET_TOP_Z_M + FINGER_Z_MARGIN_M: return 'gripper_low_near_target'
        closed = so.grip is not None and so.grip < 0 and so.width < V2.HOLD_WIDTH_MAX_M
        if not closed: return None
        h = self.hold_estimate(so.width, so.grip)
        if h is None: return 'held_cube_unknown_near_target' if (d_tcp < V2.CONTACT_RADIUS_M and p[2] < V2.CONTACT_TCP_Z_M) else None
        A = p[:3] + V1.quat_R(p[3:7]) @ h['h']; d = float(np.linalg.norm(A[:2] - tr['xy'])); bottom = float(A[2] - V1.CUBE_HALF_M)
        if d < CUBE_REACH_M + reach_u + V2.U_HOLD_M and bottom < V1.TARGET_TOP_Z_M + Z_MARGIN_M: return 'held_cube_low_near_target'
        return None

    def update_obs(self, so):
        out = super().update_obs(so)
        if self.target is not None and not (out['hand'] or out['base']):
            why = self.contact(so)
            if why is not None:
                self.log['invalidations'].append({'t': so.t, 'reason': why, 'src': self.target['src'], 'n': self.target['n'], 't_last': self.target['t_last']}); self.target = None
        out['invalidated'] = bool(self.log['invalidations'] and self.log['invalidations'][-1]['t'] == so.t); return out

    def freeze_history(self, t):
        """Called once, at the trigger step, after that step's update: fixes how many confirmations came from before it."""
        if self.t_trigger is not None: raise ValueError('the history is frozen once')
        self.t_trigger = int(t); tr = self.target
        if tr is not None: tr['n_hist'] = tr['n_prev'] if tr['t_last'] == int(t) and tr.get('n_prev') is not None and tr['n'] == tr['n_prev'] + 1 else tr['n']

    def target_estimate(self):
        tr = self.target
        if tr is None: return None, ('invalidated_by_contact' if self.log['invalidations'] else 'never_seen')
        n = tr['n'] if self.t_trigger is None else (tr.get('n_hist') or 0)
        if n < self.confirm_min: return None, 'not_confirmed_before_trigger' if self.t_trigger is not None else 'not_confirmed'
        return {'xy': tr['xy'].copy(), 'u': tr['u'] + (V2.U_PUSH_M if tr['exposed'] > 0 else 0.0), 'src': tr['src'], 't_last': tr['t_last'], 'exposed_steps': tr['exposed'], 'confirmations': tr['n'], 'confirmations_before_trigger': n}, None


def decide(est, steps_left):
    """The v2 ladder with the height margin of S5 and the label of S6. For a target seen in the current frame it is V2.decide."""
    if est['why'] is not None: why = est['why']
    else:
        n = est['offset_norm_m']; u = est['u']; gap = est['gap_m']; remembered = est['target_src'] != 'frame'
        if est['held_xyz'][2] < V2.CARRY_Z_MIN_M: why = 'LOW_CARRY'
        elif remembered and gap < Z_MARGIN_M: why = 'LOW_CARRY'
        elif n + (u if remembered else 0.0) >= V2.SIDE_OFFSET_M and gap < V2.SIDE_GAP_MIN_M: why = 'SIDE_CONTACT_RISK'
        elif steps_left < V2.STEPS_LEFT_MIN: why = 'TOO_FEW_STEPS'
        elif n < V2.ACT_MIN_M: why = 'ALIGNED_ON_MEMORY' if remembered else 'ALIGNED'
        elif n - u < V2.RESIDUAL_TOL_M: why = 'ALIGNED_WITHIN_UNCERTAINTY'
        else: why = 'ACT'
    return {'act': why == 'ACT', 'reason': why}


def at_trigger(mem, so):
    """S2: the trigger step's observation has already entered the memory in the trunk; here the history is closed."""
    mem.freeze_history(so.t)


def vision_center_v2s(r, mem, frames=None):
    """v2s at the trigger state, with the MemS built on the way there. Abstain: nothing is touched. Act: the same bounded
    horizontal move as v1, v2 and the oracle, with the memory updated and re-checked on every control step (S3)."""
    return C1.center_move(r, mem, frames, mode='VISION_CENTER_V2S', decide_fn=decide, update_memory=True, intervention=INTERVENTION, at_trigger=at_trigger, extra={'safety': dict(SAFETY)})
