"""C1 mechanism decomposition: offline analysis of stored records only.

Reads stored scene records and traces (Phase B original, a future C1 run), the stored offline decision replay, and
the stored local CPU replay metrics. It never imports a simulator, never steps an environment, never renders and
never writes into an input folder. Every output row carries its source kind, run, stage, policy, scene, arm and the
sha256 of the record it came from.

Time axis (fixed; see ANALYSIS_SPEC.md):
  trace row t        = the state after t environment steps (row 0 is the record's initial_state); its `action` is the
                       action executed at step index t-1.
  t_trigger = T      = the trigger observation is the state row T of the trunk; the decision uses it.
  t_control_start    = T: the first control action is step index T (its result is row T+1).
  t_control_end      = T+K: the state row after the last of the K control actions (K = control_steps); null if K = 0.
  t_policy_resume    = the recorded first_inference_step of the branch (the observation index of the first policy call
                       after the trigger); null if no call was recorded.
  t_first_open       = the first branch row t > T whose executed gripper command is > 0 (opening); action index t-1.
  t_release_gt       = the recorded post-hoc release step (first row whose grasp flag is off after being on).
  t_success          = the first branch row with success true; null if never.
  t_end              = the last branch row.
"""
from __future__ import annotations

import csv, gzip, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SOURCE_KINDS = ('PHASE_B_ORIGINAL', 'LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS', 'OFFLINE_DECISION_REPLAY', 'MOCK', 'C1_ORIGINAL')
VISION_ARMS = ('VISION_CENTER_V1', 'V2_SINGLE_FRAME', 'VISION_CENTER_V2', 'V2_NO_HAND_MEMORY', 'V2_NO_BASE_MEMORY', 'V2_NO_HOLD_MEMORY', 'V2_RULES_OFF')
ORACLE = 'CENTER_ORACLE'; REF = 'C0'; UNREADABLE = '__unreadable_record__'
MOVE_MIN_M = 0.001        # an ACT counts as an actual movement if the TCP moved at least this far in XY during control
PUSH_MIN_M = 0.001        # the target counts as moved during control above this XY displacement (the Phase B summary's own threshold)
NA_NORM_M = 0.001         # direction cosines are NA when either vector is shorter than this
RESIDUAL_TOL_M = 0.010    # the controllers' own residual tolerance (hci.vision_center_v2.RESIDUAL_TOL_M; oracle bound residual_tol_m)
TARGET_CAP_M = 0.03; STEP_CMD_M = 0.005; MAX_STEPS = 12   # three separate bounds, never merged

# single-label classification of an oracle rescue that a vision arm missed: the first block in code order
MISS_ORDER = ('NO_OBSERVATION', 'HOLD_ESTIMATE_NOT_ACCEPTED', 'TARGET_ESTIMATE_NOT_ACCEPTED', 'MEMORY_UNUSABLE_OR_INVALIDATED',
              'ABSTAIN_HEIGHT', 'ABSTAIN_CONTACT_RISK', 'ABSTAIN_TIME', 'JUDGED_ALIGNED', 'ACT_WITHOUT_MOVEMENT',
              'MOVED_ALIGNMENT_INSUFFICIENT', 'ALIGNED_BETTER_BUT_FAILED_AFTER_RESUME', 'UNCLASSIFIABLE_MISSING_RECORD')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def read_json(p):
    p = Path(p)
    if p.suffix == '.gz':
        with gzip.open(p, 'rt', encoding='utf-8') as f: return json.load(f)
    return json.loads(p.read_text(encoding='utf-8'))


def v(x): return None if x is None else np.asarray(x, float)


def norm(x): return None if x is None else float(np.linalg.norm(x))


def cos(a, b):
    if a is None or b is None or norm(a) < NA_NORM_M or norm(b) < NA_NORM_M: return None
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def ratio(n, d): return None if not d else n / d


# ---------------------------------------------------------------- loading
class Run:
    """One stored run folder: <root>/<stage>/scenes/*.json with traces/ and frames/ beside them."""

    def __init__(self, root, source_kind, run_id, stages, policy, host=None):
        if source_kind not in SOURCE_KINDS: raise ValueError(f'unknown source kind {source_kind}')
        if source_kind == 'C1_ORIGINAL' and not Path(root).exists(): raise FileNotFoundError(root)
        self.root = Path(root); self.kind = source_kind; self.run_id = run_id; self.stages = stages; self.policy = policy; self.host = host

    def scenes(self):
        for stage, sub in self.stages.items():
            for f in sorted((self.root / sub / 'scenes').glob('*.json')):
                if f.name.endswith('.error.json'): yield stage, f, None; continue
                try: rec = read_json(f)
                except (ValueError, UnicodeDecodeError) as e: rec = {UNREADABLE: repr(e)}            # recorded by analyse_run (review m7)
                yield stage, f, rec

    def trace(self, stage, rec):
        p = self.root / self.stages[stage] / rec['trace']
        if sha256(p) != rec['trace_sha256']: raise ValueError(f'trace sha mismatch: {p}')
        return read_json(p)['traces'], p


def key(r): return (r['source_kind'], r['run'], r['stage'], r['policy'], r['scene'], r['arm'], r['branch'])


def check_keys(rows):
    """No duplicate key, one source kind and run per call, no arm outside the registry, policy present."""
    seen = set(); kinds = {r['source_kind'] for r in rows}; problems = []
    if len(kinds) > 1: problems.append(f'source kinds mixed in one table: {sorted(kinds)}')
    for r in rows:
        k = key(r)
        if k in seen: problems.append(f'duplicate key {k}')
        seen.add(k)
        if r['policy'] in (None, ''): problems.append(f'no policy for {k}')
        if r['arm'] not in (REF, ORACLE) + VISION_ARMS: problems.append(f'unknown arm {r["arm"]} at {k}')
    return problems


# ---------------------------------------------------------------- per branch
def rows_by_t(rows): return {int(x['t']): x for x in rows}


def first_open(br, T):
    for t in sorted(br):
        if t > T and br[t].get('action') is not None and br[t]['action'][-1] > 0: return t
    return None


def first_success(br):
    for t in sorted(br):
        if br[t].get('success'): return t
    return None


def gt_offset(row):
    """Target minus held cube in XY (the controllers' offset convention: command direction)."""
    return v(row['cubeB'])[:2] - v(row['cubeA'])[:2]


def branch_row(rec, cell, tr, stage, run, sha):
    """One scene_arm_summary row from a stored cell. Ground truth enters only as post-hoc labels."""
    arm = cell['arm']; T = rec['trigger']['step'] if rec.get('triggered') else None; ctl = cell.get('control') or {}
    if not isinstance(arm, str): raise TypeError(f'arm is not a string: {arm!r}')                          # review m8: validated before any use
    if cell.get('y') not in (0, 1, None) or isinstance(cell.get('y'), bool): raise ValueError(f'outcome y is not 0, 1 or absent: {cell.get("y")!r}')
    if T is not None and (not isinstance(T, int) or isinstance(T, bool)): raise TypeError(f'trigger step is not an integer: {T!r}')
    ks = cell.get('control_steps')                 # the collector writes it for every triggered branch and omits it when nothing triggered (reviews m9, m10)
    if rec.get('triggered'):
        if not isinstance(ks, int) or isinstance(ks, bool) or ks < 0: raise TypeError(f'control_steps of a triggered branch is not a non-negative integer: {ks!r}')
    elif not (ks is None or (type(ks) is int and ks == 0)): raise TypeError(f'control_steps recorded without a trigger: {ks!r}')      # integer 0 or absent only (review m11)
    K = int(cell.get('control_steps') or 0); trunk = rows_by_t(tr['trunk']); br = rows_by_t(tr.get(arm) or [])
    rd0 = (ctl.get('readings') or [None])[0] or {}
    out = {'source_kind': run.kind, 'run': run.run_id, 'stage': stage, 'policy': run.policy, 'scene': rec['scene'], 'arm': arm, 'branch': f'{rec["scene"]}/{arm}',
           'record_sha256': sha, 'trace_sha256': rec.get('trace_sha256'), 'completed': rec.get('completed'), 'triggered': bool(rec.get('triggered')),
           't_trigger': T, 'decision': ctl.get('decision') if arm != ORACLE else (None if not cell.get('acted') else ('ACT' if K > 0 else ('ORACLE_WITHIN_TOLERANCE_AT_START' if ctl.get('end_reason') == 'within_tolerance' else f"ORACLE_ZERO_STEPS:{ctl.get('end_reason')}"))),      # review m12
           'abstain_why': rd0.get('why'), 'acted': bool(cell.get('acted')), 'control_steps': K, 'end_reason': ctl.get('end_reason'),
           'y_arm': cell.get('y'), 'steps': cell.get('steps'), 'outcome_reason': cell.get('reason'), 'ended_during_control': cell.get('ended_during_control'),
           'target_src': sources(arm, rd0)[0], 'hold_src': sources(arm, rd0)[1],
           'u_m': rd0.get('u'), 'est_offset_m': rd0.get('offset_norm_m'), 'gap_m': rd0.get('gap_m'), 'memory_reason': rd0.get('memory_reason'),
           'target_fit_present': None if arm not in ('VISION_CENTER_V1',) else rd0.get('target_fit') is not None, 'target_fit_valid': None if arm not in ('VISION_CENTER_V1',) or rd0.get('target_fit') is None else bool(rd0['target_fit'].get('valid'))}
    mt = rd0.get('memory_target') or {}
    out.update({'memory_t_last': mt.get('t_last'), 'memory_age_steps': None if mt.get('t_last') is None or T is None else T - mt['t_last'], 'memory_confirmations': mt.get('confirmations'),
                'memory_exposed_steps': mt.get('exposed_steps'), 'memory_u_m': mt.get('u')})
    ge = cell.get('gt_reading_error') or {}
    out.update({'true_offset_m': ge.get('true_offset_m'), 'offset_error_m': ge.get('offset_error_m'), 'target_error_m': ge.get('target_error_m'),
                'bound_m': ge.get('bound_m'), 'error_within_bound': ge.get('error_within_bound')})
    # motion during control (ground truth, post hoc), from the trunk row T and the branch row T+K
    mo = motion(ctl, trunk, br, T, K, arm) if (T is not None and K > 0 and has_pose(trunk.get(T)) and all(has_pose(br.get(T + i)) for i in range(1, K + 1))) else {}      # every control row present with its poses (reviews m2, m4)
    out.update(mo)
    out['moved'] = None if not out['acted'] or K == 0 or mo.get('tcp_xy_net_m') is None else bool(mo['tcp_xy_net_m'] >= MOVE_MIN_M)
    out['motion_record_missing'] = bool(out['acted'] and K > 0 and not mo)
    rel = cell.get('gt_release_after_trigger') or {}
    t_open = first_open(br, T) if T is not None else None
    out['control_without_trigger'] = bool(K > 0 and T is None)
    out.update({'t_control_start': T if K > 0 and T is not None else None, 't_control_end': (T + K) if K > 0 and T is not None else None, 't_policy_resume': cell.get('first_inference_step'),
                't_first_open': t_open, 't_release_gt': rel.get('release_step'), 't_success': first_success(br) if br else None, 't_end': max(br) if br else None,
                'offset_at_release_m': rel.get('offset_xy_m'), 'height_gap_at_release_m': rel.get('height_gap_m'),
                'offset_at_first_open_m': open_offset(br, trunk, t_open)})
    if out['t_control_end'] is not None and out['t_first_open'] is not None: out['steps_control_end_to_open'] = out['t_first_open'] - out['t_control_end']
    else: out['steps_control_end_to_open'] = None
    if out.get('offset_after_m') is not None and out['offset_at_first_open_m'] is not None: out['offset_change_end_to_open_m'] = out['offset_at_first_open_m'] - out['offset_after_m']
    else: out['offset_change_end_to_open_m'] = None
    return out


def sources(arm, rd0):
    """Target and held-cube sources as recorded. v2-family readings carry target_src / hold_src; the v1 reading carries
    target_fresh / target_xy and hold_fit instead (hci.vision_center), which mean a current-frame estimate."""
    if 'target_src' in rd0 or 'hold_src' in rd0: return rd0.get('target_src'), rd0.get('hold_src')
    t = 'frame' if rd0.get('target_fresh') and rd0.get('target_xy') is not None else None
    h = 'frame' if (rd0.get('hold_fit') or {}).get('valid') and rd0.get('held_xyz') is not None else None
    return t, h


def has_pose(row): return row is not None and all(row.get(k) is not None for k in ('tcp', 'cubeA', 'cubeB'))


def flag(x): return None if x is None else bool(x)


def open_offset(br, trunk, t_open):
    """True offset at the row before the first opening command: the branch row if it exists, else the trunk row; the
    same row is checked and read (review m5). NA if that row has no poses."""
    if not t_open: return None
    row = br[t_open - 1] if (t_open - 1) in br else trunk.get(t_open - 1)
    return norm(gt_offset(row)) if has_pose(row) else None


def motion(ctl, trunk, br, T, K, arm):
    """Command vs actual movement vs object alignment during control. Every bound is kept separate."""
    s, e = trunk[T], br[T + K]; steps = ctl.get('steps') or []
    cmds = [v(x['commanded_xy_m']) for x in steps if x.get('commanded_xy_m') is not None]
    cmd_net = np.sum(cmds, axis=0) if cmds else None
    tcp_path = sum(norm(v(br[t]['tcp'])[:2] - v((br.get(t - 1) or trunk[t - 1])['tcp'])[:2]) for t in range(T + 1, T + K + 1) if t in br and (t - 1 in br or t - 1 in trunk))
    dA = v(e['cubeA'])[:2] - v(s['cubeA'])[:2]; dB = v(e['cubeB'])[:2] - v(s['cubeB'])[:2]; dT = v(e['tcp'])[:2] - v(s['tcp'])[:2]
    o0 = gt_offset(s); o1 = gt_offset(e); u0 = o0 / norm(o0) if norm(o0) >= NA_NORM_M else None
    # cmd_path_m: no recorded command is NA, not zero (review m8)
    out = {'cmd_steps': len(cmds), 'cmd_path_m': float(sum(norm(c) for c in cmds)) if cmds else None, 'cmd_net_m': norm(cmd_net), 'cmd_max_step_m': max((norm(c) for c in cmds), default=None),
           'cmd_target_from_start_m': steps[-1].get('target_distance_from_start_m') if steps else None,
           'tcp_xy_net_m': norm(dT), 'tcp_xy_path_m': float(tcp_path), 'tcp_dz_m': float(e['tcp'][2] - s['tcp'][2]),
           'held_xy_net_m': norm(dA), 'held_dz_m': float(e['cubeA'][2] - s['cubeA'][2]), 'target_xy_net_m': norm(dB), 'target_moved': bool(norm(dB) > PUSH_MIN_M),
           'held_minus_tcp_xy_change_m': norm(dA - dT),
           'offset_before_m': norm(o0), 'offset_after_m': norm(o1), 'offset_reduction_m': norm(o0) - norm(o1),
           'cos_cmd_vs_true_offset': cos(cmd_net, o0), 'cos_tcp_vs_true_offset': cos(dT, o0),
           'reduction_from_held_move_m': None if u0 is None else float(np.dot(dA, u0)),         # held cube moving towards the target
           'reduction_from_target_move_m': None if u0 is None else float(-np.dot(dB, u0)),      # target moving towards the held cube (a push)
           'grasped_before': flag(s.get('is_cubeA_grasped')), 'grasped_after': flag(e.get('is_cubeA_grasped')),        # absent flag stays None (review m4)
           'finger_qpos_change': None if s.get('finger_qpos') is None or e.get('finger_qpos') is None else float(sum(e['finger_qpos']) - sum(s['finger_qpos']))}
    rt, rh = out['reduction_from_target_move_m'], out['reduction_from_held_move_m']
    out['push_assisted'] = None if rt is None else bool(rt > PUSH_MIN_M and rt >= 0.5 * max(rt + max(rh, 0.0), 1e-12))
    out['within_cmd_step_bound'] = None if out['cmd_max_step_m'] is None else bool(out['cmd_max_step_m'] <= STEP_CMD_M + 1e-9)
    out['within_target_cap'] = None if out['cmd_target_from_start_m'] is None else bool(out['cmd_target_from_start_m'] <= TARGET_CAP_M + 1e-9)
    out['within_max_steps'] = bool(len(steps) <= MAX_STEPS)
    out['residual_within_tol_at_end'] = bool(norm(o1) <= RESIDUAL_TOL_M)
    return out


# ---------------------------------------------------------------- memory decomposition (target cube)
def memory_decomposition(rec, cell, tr):
    """current memory error = error at the last memory update + (true position then - true position now).
    A vector identity in world XY; the fused memory value is used as stored, not replaced by one frame."""
    ctl = cell.get('control') or {}; rd0 = (ctl.get('readings') or [None])[0] or {}; T = rec['trigger']['step']; trunk = rows_by_t(tr['trunk'])
    src = sources(cell['arm'], rd0)[0]; row = {'scene': rec['scene'], 'arm': cell['arm'], 'target_src': src, 'why': rd0.get('why') or ctl.get('decision'), 'memory_reason': rd0.get('memory_reason')}
    if src is None: row['status'] = 'NO_TARGET_ESTIMATE'; return row
    if not has_pose(trunk.get(T)): row['status'] = 'ESTIMATED_NO_GT_AT_T'; return row
    est = v(rd0['target_xy']); gt_now = v(trunk[T]['cubeB'])[:2]; e_now = est - gt_now
    row.update({'status': 'ESTIMATED', 'err_now_m': norm(e_now), 'err_now_x_m': float(e_now[0]), 'err_now_y_m': float(e_now[1])})
    if src.startswith('memory_'):
        mt = rd0.get('memory_target') or {}; tl = mt.get('t_last')
        if tl is None: row['status'] = 'ESTIMATED_NO_MEMORY_T_LAST'; return row
        then = trunk.get(tl) if tl in trunk else (rec.get('initial_state') if tl == 0 else None)
        gt_then = v(then['cubeB'])[:2] if has_pose(then) else None
        if gt_then is None: row['status'] = 'ESTIMATED_NO_GT_AT_T_LAST'; return row
        e_store = est - gt_then; move = gt_now - gt_then
        resid = e_now - (e_store - move)
        row.update({'t_last': tl, 'age_steps': T - tl, 'err_at_store_m': norm(e_store), 'target_move_since_store_m': norm(move),
                    'identity_residual_m': norm(resid), 'confirmations': mt.get('confirmations'), 'exposed_steps': mt.get('exposed_steps'),
                    'u_m': mt.get('u'), 'err_now_within_memory_u': None if mt.get('u') is None else bool(norm(e_now) <= mt['u'])})      # no recorded bound: NA (review m5)
    return row


def hold_decomposition(rec, cell, tr):
    """Held cube: estimate error now, and the change of the true held-cube position relative to the TCP between the
    last accepted hold fit and the trigger (world XY, post hoc)."""
    ctl = cell.get('control') or {}; rd0 = (ctl.get('readings') or [None])[0] or {}; T = rec['trigger']['step']; trunk = rows_by_t(tr['trunk'])
    row = {'scene': rec['scene'], 'arm': cell['arm'], 'hold_src': sources(cell['arm'], rd0)[1], 'why': rd0.get('why') or ctl.get('decision')}
    if rd0.get('held_xyz') is None: row['status'] = 'NO_HOLD_ESTIMATE' if row['why'] == 'NO_HOLD_ESTIMATE' else 'NOT_REACHED'; return row
    if not has_pose(trunk.get(T)): row['status'] = 'ESTIMATED_NO_GT_AT_T'; return row
    e = v(rd0['held_xyz']) - v(trunk[T]['cubeA']); row.update({'status': 'ESTIMATED', 'err_now_xy_m': norm(e[:2]), 'err_now_z_m': float(e[2])})
    hs = [t for t in (rec.get('memory') or {}).get('hold_steps') or [] if t <= T]
    if hs:
        tl = max(hs); rel = lambda t: v(trunk[t]['cubeA'])[:2] - v(trunk[t]['tcp'])[:2]
        if has_pose(trunk.get(tl)): row.update({'hold_t_last': tl, 'hold_age_steps': T - tl, 'held_rel_tcp_change_since_fit_m': norm(rel(T) - rel(tl))})
    return row


# ---------------------------------------------------------------- outcome pairing
def outcome_class(y0, y):
    if y0 is None or y is None: return None
    return {(0, 1): 'RESCUE', (1, 0): 'HARM', (1, 1): 'BOTH_SUCCESS', (0, 0): 'BOTH_FAIL'}[(int(y0), int(y))]


def classify_miss(r):
    """First block in code order for a vision branch whose C0 failed while the oracle succeeded and the arm failed."""
    d = r.get('decision'); why = r.get('abstain_why')
    if r.get('decision') is None and not r['acted']: return 'UNCLASSIFIABLE_MISSING_RECORD'
    if d == 'NO_HOLD_ESTIMATE': return 'HOLD_ESTIMATE_NOT_ACCEPTED'
    if d == 'NO_TARGET_ESTIMATE':
        if r['arm'] == 'VISION_CENTER_V1':
            if r.get('target_fit_present') is None: return 'UNCLASSIFIABLE_MISSING_RECORD'
            return 'TARGET_ESTIMATE_NOT_ACCEPTED' if r['target_fit_present'] else 'NO_OBSERVATION'
        if r.get('memory_reason') == 'never_seen': return 'NO_OBSERVATION'
        if r.get('memory_reason') is None: return 'UNCLASSIFIABLE_MISSING_RECORD'
        return 'MEMORY_UNUSABLE_OR_INVALIDATED'
    if d == 'MEMORY_CONTRADICTED': return 'MEMORY_UNUSABLE_OR_INVALIDATED'
    if d == 'LOW_CARRY': return 'ABSTAIN_HEIGHT'
    if d == 'SIDE_CONTACT_RISK': return 'ABSTAIN_CONTACT_RISK'
    if d == 'TOO_FEW_STEPS': return 'ABSTAIN_TIME'
    if d in ('ALIGNED', 'ALIGNED_WITHIN_UNCERTAINTY'): return 'JUDGED_ALIGNED'
    if r['acted']:
        if r.get('moved') is None: return 'UNCLASSIFIABLE_MISSING_RECORD'          # review m9: missing motion is not "no movement"
        if r['moved'] is False: return 'ACT_WITHOUT_MOVEMENT'
        if r.get('offset_after_m') is None: return 'UNCLASSIFIABLE_MISSING_RECORD'
        if r['offset_after_m'] > RESIDUAL_TOL_M: return 'MOVED_ALIGNMENT_INSUFFICIENT'
        return 'ALIGNED_BETTER_BUT_FAILED_AFTER_RESUME'
    return 'UNCLASSIFIABLE_MISSING_RECORD'


def miss_flags(r):
    """Auxiliary flags; several may hold at once. Not a cause."""
    f = []
    if r.get('error_within_bound') is False: f.append('estimate_error_outside_own_bound')
    if r.get('target_src', '') and str(r.get('target_src')).startswith('memory_'): f.append('target_from_memory')
    if r.get('hold_src') == 'memory': f.append('hold_from_memory')
    if r.get('target_moved'): f.append('target_moved_during_control')
    if r.get('push_assisted'): f.append('alignment_reduction_mostly_by_target_motion')
    if r.get('offset_change_end_to_open_m') is not None and r['offset_change_end_to_open_m'] > 0.005: f.append('offset_grew_more_than_5mm_before_opening')
    if r.get('ended_during_control'): f.append('episode_ended_during_control')
    return f


def pair_rows(rows):
    """Attach the C0 and oracle outcome of the same run, stage, policy and scene to each row."""
    by = defaultdict(dict)
    for r in rows: by[(r['run'], r['stage'], r['policy'], r['scene'])][r['arm']] = r
    for r in rows:
        g = by[(r['run'], r['stage'], r['policy'], r['scene'])]
        r['y_c0'] = g[REF]['y_arm'] if REF in g else None; r['y_oracle'] = g[ORACLE]['y_arm'] if ORACLE in g else None
        r['outcome_vs_c0'] = outcome_class(r['y_c0'], r['y_arm']) if r['arm'] != REF else None
        r['oracle_rescue_scene'] = None if r['y_oracle'] is None or r['y_c0'] is None else bool(r['y_c0'] == 0 and r['y_oracle'] == 1)
        r['miss_class'] = classify_miss(r) if (r['arm'] in VISION_ARMS and r['oracle_rescue_scene'] and r['y_arm'] == 0) else None
        r['miss_flags'] = ';'.join(miss_flags(r)) if r['miss_class'] else ''
    return rows


# ---------------------------------------------------------------- whole run
def analyse_run(run, offline=None):
    rows, events, mem, hold, errors = [], [], [], [], []
    BAD = (KeyError, TypeError, IndexError, ValueError, AttributeError)
    for stage, f, rec in run.scenes():
        if rec is None: errors.append({'stage': stage, 'file': f.name, 'status': 'ERROR_RECORD'}); continue
        if not isinstance(rec, dict) or UNREADABLE in rec: errors.append({'stage': stage, 'file': f.name, 'status': 'UNREADABLE_RECORD', 'error': rec.get(UNREADABLE) if isinstance(rec, dict) else 'not an object'}); continue
        sha = sha256(f); scene_id = rec.get('scene') if isinstance(rec, dict) else None
        try:                                       # a malformed scene record is an error record (run INCOMPLETE) (review m6)
            if not rec.get('completed'): errors.append({'stage': stage, 'scene': scene_id, 'status': 'INCOMPLETE_RECORD'})
            if not isinstance(rec.get('cells'), list): raise TypeError('cells is not a list')
            if not isinstance(scene_id, int) or isinstance(scene_id, bool): scene_id = repr(scene_id); raise TypeError('scene id is not an integer')      # review m8
            got = [c.get('arm') if isinstance(c, dict) else None for c in rec['cells']]
            if len(got) != len(set(map(str, got))): raise ValueError(f'duplicate arm cells: {got}')
            if not isinstance(rec.get('arms'), list): raise TypeError('the record has no arms list (review m10)')
            if sorted(map(str, rec['arms'])) != sorted(map(str, got)): raise ValueError(f'arm cells {got} differ from the record arms {rec["arms"]}')      # review m9: no branch silently absent
            tr, _ = run.trace(stage, rec)
        except BAD as e:
            if 'trace sha mismatch' in str(e): raise                                   # a changed input is never tolerated
            errors.append({'stage': stage, 'scene': scene_id, 'file': f.name, 'status': 'MALFORMED_RECORD', 'error': repr(e)}); continue
        for i, cell in enumerate(rec['cells']):
            arm = cell.get('arm') if isinstance(cell, dict) else None
            try:                                   # a malformed branch is an error record (run INCOMPLETE), never dropped silently (reviews m5, m6)
                if not isinstance(cell, dict): raise TypeError(f'cell {i} is not a mapping')
                r = branch_row(rec, cell, tr, stage, run, sha); ev = event_rows(r, tr, cell); mm = hh = None
                if rec.get('triggered') and cell['arm'] in VISION_ARMS and cell.get('control'):
                    mm = {**memory_decomposition(rec, cell, tr), 'stage': stage, 'source_kind': run.kind, 'run': run.run_id, 'policy': run.policy}
                    hh = {**hold_decomposition(rec, cell, tr), 'stage': stage, 'source_kind': run.kind, 'run': run.run_id, 'policy': run.policy}
            except BAD as e:
                errors.append({'stage': stage, 'scene': scene_id, 'arm': arm, 'cell_index': i, 'status': 'MALFORMED_BRANCH', 'error': repr(e)}); continue
            rows.append(r); events.extend(ev)
            if mm is not None: mem.append(mm); hold.append(hh)
    pair_rows(rows)
    return {'rows': rows, 'events': events, 'memory': mem, 'hold': hold, 'errors': errors}


EVENTS = ('t_trigger', 't_control_start', 't_control_end', 't_policy_resume', 't_first_open', 't_release_gt', 't_success', 't_end')


def event_rows(r, tr, cell):
    trunk = rows_by_t(tr['trunk']); br = rows_by_t(tr.get(r['arm']) or []); out = []
    for ev in EVENTS:
        t = r.get(ev)
        row = {k: r[k] for k in ('source_kind', 'run', 'stage', 'policy', 'scene', 'arm', 'branch')}; row['event'] = ev; row['t'] = t
        if t is None: row['null_reason'] = {'t_trigger': 'not triggered', 't_control_start': 'no control step', 't_control_end': 'no control step', 't_policy_resume': 'no policy call recorded after the trigger',
                                            't_first_open': 'no opening command in the branch', 't_release_gt': 'no release found', 't_success': 'never successful', 't_end': 'no branch trace'}[ev]; out.append(row); continue
        src = 'branch' if t in br else ('trunk' if t in trunk else None); x = (br if src == 'branch' else trunk).get(t) if src else None
        row['pose_missing'] = x is None or not has_pose(x)
        if row['pose_missing']: x = None
        row.update({'row_source': src, 'row_index': None if x is None else sorted((br if src == 'branch' else trunk)).index(t),
                    'tcp': None if x is None else json.dumps([round(float(a), 6) for a in x['tcp']]), 'cubeA': None if x is None else json.dumps([round(float(a), 6) for a in x['cubeA']]),
                    'cubeB': None if x is None else json.dumps([round(float(a), 6) for a in x['cubeB']]), 'gt_offset_xy_m': None if x is None else norm(gt_offset(x)),
                    'gripper_cmd': None if x is None or x.get('action') is None else float(x['action'][-1])})
        out.append(row)
    return out


def check_integrity(r):
    """An acted branch with control steps but no motion rows, or a triggered branch without an outcome, is an integrity error."""
    p = []
    if r.get('motion_record_missing'): p.append('acted with control steps but a trace row in T..T+K is missing')
    if r.get('y_arm') is None: p.append('no recorded outcome')                       # every row, triggered or not (review m2)
    if r.get('control_without_trigger'): p.append('control steps recorded without a trigger')
    return p


def check_time_order(r):
    """t_trigger <= t_control_start < t_control_end <= t_first_open; t_policy_resume >= t_control_end when both exist."""
    p = []
    T, s, e, res, o = r.get('t_trigger'), r.get('t_control_start'), r.get('t_control_end'), r.get('t_policy_resume'), r.get('t_first_open')
    if s is not None and (T is None or s < T): p.append('control starts before the trigger')
    if e is not None and (s is None or e <= s): p.append('control ends before it starts')
    if e is not None and res is not None and res < e: p.append('policy resumes before control ended')
    if e is not None and o is not None and o <= e: p.append('opening during control')
    return p


# ---------------------------------------------------------------- funnel and denominators
def funnel(rows, arms):
    """Counts per stage and arm with explicit denominators; every subset is checked against its parent."""
    out = []
    for (run_id, stage, policy), grp in sorted(group(rows, ('run', 'stage', 'policy')).items()):
        scenes = sorted({r['scene'] for r in grp}); trig = sorted({r['scene'] for r in grp if r['triggered']})
        for arm in arms:
            a = [r for r in grp if r['arm'] == arm]
            if not a: continue
            tri = [r for r in a if r['triggered']]; act = [r for r in tri if r['decision'] == 'ACT']
            ctl = [r for r in act if r['control_steps'] > 0]; mov = [r for r in ctl if r['moved']]
            abst = [r for r in tri if not r['acted']]; err = [r for r in a if not r['completed']]
            stop0 = [r for r in act if r['control_steps'] == 0]
            c0 = {r['scene']: r['y_c0'] for r in a}
            missing = [r for r in tri if r['y_arm'] is None or r['y_c0'] is None]
            c0s = [r for r in tri if r['y_c0'] == 1 and r['y_arm'] is not None]; c0f = [r for r in tri if r['y_c0'] == 0 and r['y_arm'] is not None]
            harm = [r for r in c0s if r['outcome_vs_c0'] == 'HARM']; rescue = [r for r in c0f if r['outcome_vs_c0'] == 'RESCUE']
            harm_act = [r for r in c0s if r['moved'] and r['outcome_vs_c0'] == 'HARM']; c0s_act = [r for r in c0s if r['moved']]
            orr = [r for r in tri if r['oracle_rescue_scene'] and r['y_arm'] is not None]; orr_v = [r for r in orr if r['y_arm'] == 1]
            def add(step, num, den, ids, parent=None, note=''):
                out.append({'run': run_id, 'stage': stage, 'policy': policy, 'arm': arm, 'step': step, 'numerator': len(num) if isinstance(num, list) else num,
                            'denominator': den, 'fraction': ratio(len(num) if isinstance(num, list) else num, den), 'parent_step': parent,
                            'scenes': ' '.join(str(s) for s in sorted(ids)), 'note': note})
            add('scenes', a, len(scenes), [r['scene'] for r in a])
            add('triggered', tri, len(a), [r['scene'] for r in tri], 'scenes')
            add('decision_ACT', act, len(tri), [r['scene'] for r in act], 'triggered')
            add('control_steps_executed', ctl, len(act), [r['scene'] for r in ctl], 'decision_ACT')
            add('actual_movement', mov, len(ctl), [r['scene'] for r in mov], 'control_steps_executed', f'TCP XY net displacement >= {MOVE_MIN_M} m')
            add('ACT_with_zero_control_steps', stop0, len(act), [r['scene'] for r in stop0], 'decision_ACT', 'immediate stop')
            if arm == ORACLE:
                wt = [r for r in tri if r['decision'] == 'ORACLE_WITHIN_TOLERANCE_AT_START']
                add('oracle_within_tolerance_at_start', wt, len(tri), [r['scene'] for r in wt], 'triggered', 'engaged, zero control steps: already within its tolerance')
                oz = [r for r in tri if str(r['decision']).startswith('ORACLE_ZERO_STEPS:')]
                add('oracle_zero_steps_other_end_reason', oz, len(tri), [r['scene'] for r in oz], 'triggered', 'engaged, zero control steps, end reason other than within_tolerance (see decision)')
            add('abstained_or_no_action', abst, len(tri), [r['scene'] for r in abst], 'triggered')
            add('technical_error_or_incomplete', err, len(a), [r['scene'] for r in err], 'scenes')
            add('outcome_missing_excluded_from_outcome_rates', missing, len(tri), [r['scene'] for r in missing], 'triggered', 'never counted as success, harm-free or zero; any > 0 makes the analysis INCOMPLETE')
            if arm != REF:
                add('harm_of_all_c0_success', harm, len(c0s), [r['scene'] for r in harm], 'triggered')
                add('harm_of_c0_success_with_movement', harm_act, len(c0s_act), [r['scene'] for r in harm_act], 'triggered')
                add('c0_success_protected_by_abstention', [r for r in c0s if not r['acted'] and r['y_arm'] == 1], len(c0s), [r['scene'] for r in c0s if not r['acted'] and r['y_arm'] == 1], 'triggered')
                add('c0_success_kept_with_movement', [r for r in c0s_act if r['y_arm'] == 1], len(c0s_act), [r['scene'] for r in c0s_act if r['y_arm'] == 1], 'triggered')
                add('rescue_of_all_c0_failure', rescue, len(c0f), [r['scene'] for r in rescue], 'triggered')
                if any(r['y_oracle'] is not None for r in a):
                    add('oracle_rescue_also_rescued', orr_v, len(orr), [r['scene'] for r in orr_v], 'triggered', 'oracle is not an upper bound of all recoveries')
                    vis_not_oracle = [r for r in tri if r['y_c0'] == 0 and r['y_arm'] == 1 and r['y_oracle'] == 0]
                    add('rescued_where_oracle_failed', vis_not_oracle, len(c0f), [r['scene'] for r in vis_not_oracle], 'triggered')
                    miss = [r for r in orr if r['y_arm'] == 0]; cnt = Counter(r['miss_class'] for r in miss)
                    for m in MISS_ORDER: add(f'oracle_rescue_missed:{m}', [r for r in miss if r['miss_class'] == m], len(miss), [r['scene'] for r in miss if r['miss_class'] == m], 'oracle_rescue_missed')
                    assert sum(cnt.values()) == len(miss)
                    add('oracle_rescue_missed', miss, len(orr), [r['scene'] for r in miss], 'triggered')
    return out


def check_funnel(f):
    """Each split sums to its parent; numerators never exceed denominators."""
    p = []
    for r in f:
        if r['denominator'] is not None and r['numerator'] > r['denominator']: p.append(f'numerator above denominator: {r}')
        if r['step'] == 'outcome_missing_excluded_from_outcome_rates' and r['numerator']: p.append(f"outcomes missing for {r['arm']} in {r['stage']}: {r['scenes']}")
    by = group(f, ('run', 'stage', 'policy', 'arm'))
    for k, g in by.items():
        d = {r['step']: r for r in g}
        if 'triggered' in d and 'decision_ACT' in d and 'abstained_or_no_action' in d and d['decision_ACT']['numerator'] + d['abstained_or_no_action']['numerator'] != d['triggered']['numerator']:
            if k[3] not in (ORACLE, REF): p.append(f'ACT + abstain != triggered at {k}')
        miss = [r for r in g if r['step'].startswith('oracle_rescue_missed:')]
        if miss and sum(r['numerator'] for r in miss) != d['oracle_rescue_missed']['numerator']: p.append(f'miss classes do not sum at {k}')
    return p


def group(rows, keys):
    g = defaultdict(list)
    for r in rows: g[tuple(r[k] for k in keys)].append(r)
    return g


# ---------------------------------------------------------------- offline decision replay (decisions only)
def offline_rows(path):
    o = read_json(path); sha = sha256(path); out = []
    for row in o['rows']:
        if not row.get('triggered'): continue
        v2 = row['arms'].get('VISION_CENTER_V2') or {}
        for arm, a in row['arms'].items():
            if arm == 'VISION_CENTER_V2S': continue                        # v2s stays excluded (not a C1 arm); used only to reconcile counts
            out.append({'source_kind': 'OFFLINE_DECISION_REPLAY', 'run': 'offline_replay_r4', 'stage': row['stage'], 'policy': 'train_seed_1', 'scene': row['scene'], 'arm': arm, 'branch': f'{row["scene"]}/{arm}',
                        'record_sha256': sha, 'decision': a.get('decision'), 'act': a.get('act'), 'target_src': a.get('target_src'), 'hold_src': a.get('hold_src'),
                        'v2_decision': v2.get('decision'), 'v2_target_src': v2.get('target_src'), 'decision_differs_from_v2': a.get('decision') != v2.get('decision'),
                        'target_src_differs_from_v2': a.get('target_src') != v2.get('target_src'), 'act_differs_from_v2': a.get('act') != v2.get('act'),
                        'offset_error_m': a.get('offset_error_m'), 'within_bound': a.get('within_bound'),
                        'motion': 'NOT_RUN', 'outcome': 'NOT_RUN' if arm not in ('VISION_CENTER_V1', 'VISION_CENTER_V2') else 'SEE_PHASE_B_ORIGINAL'})
    return out


ABLATION = {'V2_SINGLE_FRAME': 'all memories removed AND confirmation count changed (two changes together)', 'V2_NO_HAND_MEMORY': 'hand-camera target memory removed',
            'V2_NO_BASE_MEMORY': 'base-camera target memory removed', 'V2_NO_HOLD_MEMORY': 'held-cube relative-pose memory removed', 'V2_RULES_OFF': 'v2 decision rules replaced by v1 rules (memories kept)',
            'VISION_CENTER_V1': 'v1 baseline (single frame, v1 rules)'}


def ablation_table(off):
    out = []
    for arm, what in ABLATION.items():
        a = [r for r in off if r['arm'] == arm]
        if not a: continue
        for stage in sorted({r['stage'] for r in a}):
            s = [r for r in a if r['stage'] == stage]
            out.append({'arm': arm, 'stage': stage, 'policy': 'train_seed_1', 'removed_element': what, 'triggered_scenes': len(s),
                        'information_changed_target_src': sum(r['target_src_differs_from_v2'] for r in s), 'decision_changed': sum(r['decision_differs_from_v2'] for r in s),
                        'act_changed': sum(r['act_differs_from_v2'] for r in s), 'act_lost_vs_v2': sum(1 for r in s if r['v2_decision'] == 'ACT' and not r['act']),
                        'act_gained_vs_v2': sum(1 for r in s if r['v2_decision'] != 'ACT' and r['act']),
                        'command_motion_change': 'NOT_RUN', 'outcome_difference': 'NOT_RUN' if arm != 'VISION_CENTER_V1' else 'PHASE_B_ORIGINAL (v1 executed)', 'physical_side_effect_difference': 'NOT_RUN',
                        'source_kind': 'OFFLINE_DECISION_REPLAY'})
    return out


# ---------------------------------------------------------------- statistics helpers
def dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return {'n': 0, 'median': None, 'p90': None, 'max': None}
    a = np.asarray(xs, float); return {'n': len(xs), 'median': float(np.median(a)), 'p90': float(np.percentile(a, 90)), 'max': float(a.max())}


def write_csv(path, rows, cols=None):
    cols = cols or sorted({k for r in rows for k in r})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        for r in rows: w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in cols})
    return cols
