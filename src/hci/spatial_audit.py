"""Spatial occupancy, clearance and contact audit of stored Vision-CENTER records (Phase B now, Phase C1 later).

Post-hoc only: reads the scene record, the per-step trace and the stored RGB frames; runs no simulator, no policy, no
controller, and nothing here is an input of any controller. Positions in the trace are the simulator's logged state
(POST-HOC GROUND TRUTH). Shapes are the simulator's collision geometry (hci.spatial_geometry).

Three different things are kept apart throughout:
  GEOMETRIC PROXIMITY   exact surface distance at a logged pose <= PROXIMITY_M (exploratory threshold)
  POTENTIAL COLLISION   surface distance <= CONTACT_GEOM_TOL_M at a logged pose, or at an interpolated sub-pose between
                        two logged poses (interpolation is an ASSUMPTION: the trace has one pose per control step, PhysX
                        integrates 5 sub-steps)
  OBSERVED CONTACT      a non-zero contact force recorded by the simulator for that body pair (PhysX contacts of the last
                        sub-step of the control step). Only six pairs are recorded; every other pair is NOT_OBSERVABLE,
                        and the absence of a record is never read as the absence of contact.
Continuous-time safety is NOT established by any of this: the poses and contact forces are sampled once per control step.

The arm links (panda_link0..7) are not represented: the trace does not log the arm joint angles. Whole-robot swept volume
and arm-link clearance are therefore NOT_OBSERVABLE; only the hand, the fingers and the held cube are covered.
"""
import math
import numpy as np
from . import spatial_geometry as G
from . import trajectory_audit as T

# Exploratory thresholds, declared before any stored record was processed with this module. They are not part of the
# C1 expectation spec and decide nothing in C1; they only label steps in figures and tables.
PROXIMITY_M = 0.005
CONTACT_GEOM_TOL_M = 0.001
FAR_M = math.inf                # AABB shortcut disabled: every stored distance is exact (review s1, finding 3: a lower bound
                                # in the same column as exact gaps could be read as a clearance)
MOTION_MARGIN = 1.6             # sub-pose screening runs when the closer endpoint distance is within MOTION_MARGIN x the vertex travel
STATUS = ('PASS', 'FAIL', 'NOT_OBSERVABLE', 'NOT_APPLICABLE')
EXPLORATORY = 'EXPLORATORY (threshold declared in hci.spatial_audit, not in the C1 expectation spec)'

# The six body pairs whose contact force the trace records (hci.recovery_v1.probe), and the ones it does not.
RECORDED_PAIRS = {'A_B': ('f_A_B',), 'fingerL_A': ('f_A_left',), 'fingerR_A': ('f_A_right',), 'A_table': ('f_A_table',), 'fingerL_B': ('f_B_left',), 'fingerR_B': ('f_B_right',)}
UNRECORDED_PAIRS = ('hand_A', 'hand_B', 'B_table', 'gripper_table', 'arm_links_any')
DIST_PAIRS = ('A_B', 'fingerL_A', 'fingerR_A', 'hand_A', 'fingerL_B', 'fingerR_B', 'hand_B')
SCREEN_PAIRS = ('A_B', 'fingerL_B', 'fingerR_B', 'hand_B')      # unintended contacts during a centring move


# ---------------------------------------------------------------- shapes of one logged row
def shapes(row):
    A = G.cube(row['cubeA'], row['cubeA_q'], 'A'); B = G.cube(row['cubeB'], row['cubeB_q'], 'B'); g = G.gripper(row['tcp'], row['tcp_q'], row['finger_qpos'])
    return {'A': A, 'B': B, 'hand': g['hand'], 'fingerL': G.finger_shapes(g, 'left'), 'fingerR': G.finger_shapes(g, 'right'), 'gripper': list(g.values())}


def _dist(X, Y):
    """(distance, exact). X, Y: Convex or list of Convex. Exact below FAR_M; otherwise the AABB gap, a lower bound."""
    Xs = X if isinstance(X, list) else [X]; Ys = Y if isinstance(Y, list) else [Y]; best = (math.inf, True)
    for a in Xs:
        for b in Ys:
            gap = G.aabb_gap(a, b)
            if gap > FAR_M: d = (gap, False)
            else: d = (G.signed_distance(a, b), True)
            if d[0] < best[0]: best = d
    return best


def _pair_shapes(S, pair):
    a, b = pair.split('_'); return S[a], S[b]


def row_metrics(row, initial=None):
    """Clearances, footprints and contact records of one logged step. Distances in metres, areas in mm^2."""
    S = shapes(row); out = {'t': int(row['t'])}
    for p in DIST_PAIRS:
        d, ex = _dist(*_pair_shapes(S, p)); out[f'd_{p}'] = d; out[f'exact_{p}'] = ex
    out['d_A_table'] = float(S['A'].V[:, 2].min()); out['d_B_table'] = float(S['B'].V[:, 2].min())
    out['d_gripper_table'] = float(min(s.V[:, 2].min() for s in S['gripper']))
    out['height_gap_Abottom_Btop'] = float(S['A'].V[:, 2].min() - S['B'].V[:, 2].max())
    fa, fb = G.footprint(S['A']), G.footprint(S['B']); fg = [G.footprint(s) for s in S['gripper']]     # one footprint per collision shape, never their hull
    out['fp_overlap_A_B_mm2'] = G.overlap_area(fa, fb) * 1e6; out['fp_gap_A_B'] = G.polygon_gap(fa, fb)
    out['fp_overlap_gripper_B_mm2'] = G.union_overlap_area(fg, fb) * 1e6; out['fp_gap_gripper_B'] = G.union_gap(fg, fb)
    R = G.quat_to_R(row['tcp_q']); out['A_in_tcp'] = (R.T @ (np.subtract(row['cubeA'], row['tcp']))).tolist()
    out['offset_xy_true'] = float(np.hypot(row['cubeB'][0] - row['cubeA'][0], row['cubeB'][1] - row['cubeA'][1]))
    if initial is not None:
        out['B_disp_xy_from_initial'] = float(np.hypot(row['cubeB'][0] - initial['cubeB'][0], row['cubeB'][1] - initial['cubeB'][1]))
        out['B_disp_z_from_initial'] = float(row['cubeB'][2] - initial['cubeB'][2])
        out['B_rot_deg_from_initial'] = G.rotation_angle_deg(initial['cubeB_q'], row['cubeB_q']) if 'cubeB_q' in initial else None
    for p, keys in RECORDED_PAIRS.items():
        f = [row.get(k) for k in keys]
        if any(v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0) for v in f):
            raise ValueError(f'step {row["t"]}: recorded force {keys} = {f} is not a finite non-negative number')   # never read as 'no contact'
        out[f'force_{p}'] = None if any(v is None for v in f) else float(sum(f))
        out[f'observed_contact_{p}'] = None if out[f'force_{p}'] is None else bool(out[f'force_{p}'] > 0)
    for p in UNRECORDED_PAIRS: out[f'observed_contact_{p}'] = None
    for p in DIST_PAIRS: out[f'proximity_{p}'] = bool(out[f'd_{p}'] <= PROXIMITY_M)
    return out


def _interp_row(r0, r1, f):
    q = lambda k: G.slerp(r0[k], r1[k], f).tolist(); lin = lambda k: ((1 - f) * np.asarray(r0[k], float) + f * np.asarray(r1[k], float)).tolist()
    return {'t': r1['t'], 'tcp': lin('tcp'), 'tcp_q': q('tcp_q'), 'cubeA': lin('cubeA'), 'cubeA_q': q('cubeA_q'), 'cubeB': lin('cubeB'), 'cubeB_q': q('cubeB_q'), 'finger_qpos': lin('finger_qpos')}


def screen_interval(r0, r1, m0, m1, pairs=SCREEN_PAIRS, n=G.SIM_STEPS_PER_CONTROL):
    """Potential collision between two logged rows: smallest distance over n interpolated sub-poses, per pair, run only
    where the motion could close the gap (vertex travel bound). Returns {pair: (min distance or None if not screened)}."""
    S0, S1 = shapes(r0), shapes(r1); out = {}
    for p in pairs:
        a0, b0 = _pair_shapes(S0, p); a1, b1 = _pair_shapes(S1, p); L = lambda x: x if isinstance(x, list) else [x]
        travel = max(float(np.linalg.norm(u.V - v.V, axis=1).max()) for u, v in zip(L(a0), L(a1))) + max(float(np.linalg.norm(u.V - v.V, axis=1).max()) for u, v in zip(L(b0), L(b1)))
        lo = min(m0[f'd_{p}'], m1[f'd_{p}'])
        if lo > CONTACT_GEOM_TOL_M + MOTION_MARGIN * travel: out[p] = None; continue
        best = math.inf
        for f in (np.arange(1, n) / n):
            Sx = shapes(_interp_row(r0, r1, f)); best = min(best, _dist(*_pair_shapes(Sx, p))[0])
        out[p] = float(best)
    return out


def branch_metrics(rec, tr, arm, t_from=None):
    """Per-step metrics of one arm from step t_from (default: trigger - 10) to its last step, with potential-collision
    screening between consecutive rows. Rows: trunk up to the trigger, then the arm's own rows (T.series order)."""
    t0 = rec['trigger']['step'] if rec['triggered'] else 0; a = arm if rec['triggered'] else 'trunk'
    rows = [r for r in tr['trunk'] if r['t'] <= t0] + (tr[a] if rec['triggered'] else []) if rec['triggered'] else list(tr['trunk'])
    lo = max(0, t0 - 10) if t_from is None else t_from; rows = [r for r in rows if r['t'] >= lo]
    if any(b['t'] - a_['t'] != 1 for a_, b in zip(rows, rows[1:])): raise ValueError(f'scene {rec["scene"]} {arm}: trace steps are not consecutive')
    init = rec.get('initial_state'); c = T.cell(rec, arm) if rec['triggered'] else None; k = c['control_steps'] if (c and c.get('acted')) else 0
    M = []
    for r in rows:
        m = row_metrics(r, init); m['phase'] = 'before_trigger' if r['t'] <= t0 else ('control' if r['t'] <= t0 + k else 'policy_after'); M.append(m)
    for i in range(len(rows)):
        if i == 0:
            for p in SCREEN_PAIRS: M[i][f'subpose_min_d_{p}'] = None
        else:
            sc = screen_interval(rows[i - 1], rows[i], M[i - 1], M[i])
            for p in SCREEN_PAIRS: M[i][f'subpose_min_d_{p}'] = sc[p]
        for p in SCREEN_PAIRS:
            sub = M[i][f'subpose_min_d_{p}']; M[i][f'potential_collision_{p}'] = bool(M[i][f'd_{p}'] <= CONTACT_GEOM_TOL_M or (sub is not None and sub <= CONTACT_GEOM_TOL_M))
    return M


# ---------------------------------------------------------------- contact vs geometry (validation of the reconstruction)
CONTACT_PAIRS_GEOM = {'A_B': 'd_A_B', 'fingerL_A': 'd_fingerL_A', 'fingerR_A': 'd_fingerR_A', 'A_table': 'd_A_table', 'fingerL_B': 'd_fingerL_B', 'fingerR_B': 'd_fingerR_B'}


def contact_geometry_table(metrics):
    """For each recorded pair: how the reconstructed distance relates to the recorded contact force over all given steps.
    A correct reconstruction puts every recorded contact at distance ~0 and no contact far from zero."""
    out = {}
    for p, dk in CONTACT_PAIRS_GEOM.items():
        d = np.array([m[dk] for m in metrics if m.get(f'force_{p}') is not None], float); f = np.array([m[f'force_{p}'] for m in metrics if m.get(f'force_{p}') is not None], float)
        if not len(d): out[p] = {'n': 0}; continue
        on = f > 0; q = lambda x, v: float(np.quantile(x, v)) if len(x) else None
        out[p] = {'n_steps': int(len(d)), 'n_contact': int(on.sum()),
                  'd_when_contact_m': {'min': q(d[on], 0), 'p01': q(d[on], .01), 'median': q(d[on], .5), 'p99': q(d[on], .99), 'max': q(d[on], 1)},
                  'contact_with_gap_above_tol': int((on & (d > CONTACT_GEOM_TOL_M)).sum()),
                  'no_contact_with_overlap_beyond_tol': int((~on & (d < -CONTACT_GEOM_TOL_M)).sum()),
                  'no_contact_within_tol': int((~on & (np.abs(d) <= CONTACT_GEOM_TOL_M)).sum())}
    return out


# ---------------------------------------------------------------- swept occupancy of a move
def swept_occupancy(rec, tr, arm, voxel=0.002):
    """Voxel occupancy of the hand+fingers, of the held cube, and of both, swept over the control steps with interpolated
    sub-poses; and the XY swept footprint area (1 mm raster). Arm links: NOT_OBSERVABLE."""
    c = T.cell(rec, arm)
    if not (c and c.get('acted') and c['control_steps'] > 0): return {'status': 'NOT_APPLICABLE', 'why': 'no move'}
    t0 = rec['trigger']['step']; k = c['control_steps']; rows = {r['t']: r for r in tr['trunk'] if r['t'] == t0}; rows.update({r['t']: r for r in tr[arm] if r['t'] <= t0 + k})
    seq = [rows[t0]] + [_interp_row(rows[t - 1], rows[t], f) for t in range(t0 + 1, t0 + k + 1) for f in np.arange(1, G.SIM_STEPS_PER_CONTROL + 1) / G.SIM_STEPS_PER_CONTROL]
    Ss = [shapes(r) for r in seq]
    grip, _ = G.swept_voxels([s for S in Ss for s in S['gripper']], voxel); held, _ = G.swept_voxels([S['A'] for S in Ss], voxel)
    fp = set()
    for S in Ss:
        for poly in [G.footprint(s) for s in S['gripper']] + [G.footprint(S['A'])]: fp |= _raster(poly, 0.001)
    Bfp = _raster(G.footprint(Ss[0]['B']), 0.001)
    return {'status': 'APPROXIMATION', 'voxel_m': voxel, 'gripper_swept_volume_cm3': len(grip) * voxel ** 3 * 1e6, 'held_cube_swept_volume_cm3': len(held) * voxel ** 3 * 1e6,
            'union_swept_volume_cm3': len(grip | held) * voxel ** 3 * 1e6, 'xy_swept_footprint_mm2': float(len(fp)), 'xy_swept_overlap_with_target_footprint_mm2': float(len(fp & Bfp)),
            'subposes': len(seq), 'arm_links': 'NOT_OBSERVABLE (arm joint angles not logged)',
            'note': 'interpolated sub-poses (linear position, slerp orientation) between logged control steps; an approximation of the swept space, not observed'}


def _raster(poly, res):
    poly = np.asarray(poly, float); lo = np.floor(poly.min(0) / res).astype(int); hi = np.ceil(poly.max(0) / res).astype(int)
    g = np.stack(np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing='ij'), -1).reshape(-1, 2); X = (g + .5) * res
    e = np.roll(poly, -1, 0) - poly; m = np.all(e[:, 0] * (X[:, None, 1] - poly[:, 1]) - e[:, 1] * (X[:, None, 0] - poly[:, 0]) >= 0, axis=1)
    return set(map(tuple, g[m]))


# ---------------------------------------------------------------- expected vs actual
EXPECTED = (('E1_alignment_error_reduced', 'true XY offset (held cube to target) smaller at the end of the move than at the trigger; every commanded step points into the true offset', 'C1 spec check 3'),
            ('E2_step_command_bound', f'every commanded XY step <= {T.STEP_CMD_M * 1e3:.0f} mm', 'C1 spec check 5'),
            ('E3_total_move_bound', f'commanded target distance from the start <= {T.TARGET_CAP_M * 1e3:.0f} mm', 'C1 spec check 5'),
            ('E4_step_count_bound', f'control steps <= {T.MAX_STEPS}', 'C1 spec check 5'),
            ('E5_no_vertical_motion', f'no vertical command; TCP height drift <= {T.DZ_TOL_M * 1e3:.0f} mm', 'C1 spec check 6'),
            ('E6_ends_before_opening', 'gripper command held closed during the move; the first open command comes after the move', 'C1 spec check 6 and the hand-back rule'),
            ('E7_target_not_moved', f'target XY displacement during the move <= {T.PUSH_TOL_M * 1e3:.0f} mm; held cube shift relative to the TCP <= {T.SLIP_TOL_M * 1e3:.0f} mm', 'C1 spec check 7'),
            ('E8_policy_resumes_cleanly', 'no policy inference and no queued action during the move; first inference at exactly trigger + control steps', 'C1 hand-back rule (driver invariant)'),
            ('E9_abstain_equals_C0', 'an abstaining arm reproduces C0 at every logged step', 'C1 spec check 9'),
            ('E10_policy_keeps_correction', f'the policy undoes at most {T.REVERT_FRACTION_MAX:.0%} of the achieved reduction before it opens the gripper', 'C1 spec check 8'))


def _s(status, **kw): return T._py({'status': status, **kw})


def expected_vs_actual(rec, tr, arm, metrics=None):
    """Each expectation as PASS / FAIL / NOT_OBSERVABLE / NOT_APPLICABLE with the actual numbers. Success is reported,
    never used for a status."""
    c = T.cell(rec, arm); t0 = rec['trigger']['step']; ch = T.check_arm(rec, tr, arm) if arm != 'C0' else {}; acted = bool(c.get('acted')); k = c['control_steps'] if acted else 0
    s = T.series(rec, tr, arm); ev = T.events(rec, s, arm); NA = lambda why: _s('NOT_APPLICABLE', why=why); out = {}
    if arm == 'C0':
        out = {e[0]: NA('reference arm (no intervention by definition)') for e in EXPECTED}
    elif not acted:
        out = {e[0]: NA('the arm abstained') for e in EXPECTED}
        out['E9_abstain_equals_C0'] = same_as_c0(rec, tr, arm)
    elif k == 0:
        out = {e[0]: NA('zero move steps') for e in EXPECTED}
    else:
        c3, c5, c6, c7, c8 = (ch[x] for x in ('3_command_reduces_true_error', '5_move_bounds', '6_no_vertical_or_gripper_change', '7_no_slip_no_push', '8_policy_keeps_correction'))
        e0 = c3.get('true_offset_before_m'); e1 = c3.get('true_offset_after_m')
        if c3['status'] == 'NOT_OBSERVABLE': out['E1_alignment_error_reduced'] = _s('NOT_OBSERVABLE', why=c3.get('why'))
        else: out['E1_alignment_error_reduced'] = _s('PASS' if (c3['status'] == 'PASS' and e1 < e0) else 'FAIL', true_offset_trigger_m=e0, true_offset_end_m=e1, steps_against_true_error=c3.get('steps_against_true_error'))
        if c5['status'] == 'NOT_OBSERVABLE':
            for x in ('E2_step_command_bound', 'E3_total_move_bound', 'E4_step_count_bound'): out[x] = _s('NOT_OBSERVABLE', why=c5.get('why'))
        else:
            out['E2_step_command_bound'] = _s('PASS' if c5['max_step_m'] <= T.STEP_CMD_M + 1e-9 else 'FAIL', max_step_m=c5['max_step_m'])
            out['E3_total_move_bound'] = _s('PASS' if c5['max_target_distance_m'] <= T.TARGET_CAP_M + 1e-9 else 'FAIL', max_target_distance_m=c5['max_target_distance_m'])
            out['E4_step_count_bound'] = _s('PASS' if k <= T.MAX_STEPS else 'FAIL', steps=k)
        out['E5_no_vertical_motion'] = _s('PASS' if (c6['vertical_command_max'] == 0 and c6['tcp_dz_max_m'] <= T.DZ_TOL_M) else 'FAIL', vertical_command_max=c6['vertical_command_max'], tcp_dz_max_m=c6['tcp_dz_max_m'])
        r = ev['release_cmd']
        out['E6_ends_before_opening'] = _s('PASS' if (c6['gripper_command_held_closed'] and (r is None or r > t0 + k)) else 'FAIL', gripper_held_closed=c6['gripper_command_held_closed'], control_end_step=t0 + k, first_open_command_step=r)
        out['E7_target_not_moved'] = _s(c7['status'], target_xy_displacement_m=c7['target_xy_displacement_m'], held_cube_shift_m=c7['held_cube_shift_relative_to_tcp_max_m'], grasped_throughout=c7['grasped_throughout'])
        fi = c.get('first_inference_step')
        if c.get('ended_during_control'): out['E8_policy_resumes_cleanly'] = NA('the episode ended during the move')
        elif fi is None: out['E8_policy_resumes_cleanly'] = _s('NOT_OBSERVABLE', why='first_inference_step not recorded')
        else: out['E8_policy_resumes_cleanly'] = _s('PASS' if fi == t0 + k else 'FAIL', first_inference_step=fi, expected_step=t0 + k)
        out['E9_abstain_equals_C0'] = NA('the arm moved')
        out['E10_policy_keeps_correction'] = _s(c8['status'], **{q: c8.get(q) for q in ('reverted_fraction', 'offset_end_of_move_m', 'offset_at_open_command_m', 'why') if c8.get(q) is not None})
    actual = actual_behaviour(rec, tr, arm, metrics)
    return {'scene': rec['scene'], 'arm': arm, 'acted': acted, 'control_steps': k, 'expected': out, 'actual': actual}


def same_as_c0(rec, tr, arm, tol=T.SAME_TOL):
    """E9: every logged field of every step of the abstaining arm against C0 (positions, orientations, velocities, finger
    joints, executed actions, contact forces, flags). The outcome is reported next to it, not used for the status."""
    if 'C0' not in tr or arm not in tr: return _s('NOT_OBSERVABLE', why='trace of C0 or of the arm missing')
    a, c = tr[arm], tr['C0']
    if len(a) != len(c) or [r['t'] for r in a] != [r['t'] for r in c]: return _s('FAIL', why='different steps', steps=[len(a), len(c)])
    worst = 0.; differ = set(); keys = set()
    for x, y in zip(a, c):
        for k in set(x) | set(y):
            keys.add(k)
            if (k in x) != (k in y): differ.add(k); continue                  # a field logged on one side only
            u, v = x[k], y[k]
            if u is None or v is None or isinstance(u, (bool, str, dict)) or isinstance(v, (bool, str, dict)):
                if u != v: differ.add(k)
                continue
            if np.size(u) != np.size(v): differ.add(k); continue
            d = float(np.max(np.abs(np.subtract(u, v)))) if np.size(u) else 0.
            if not math.isfinite(d) or d > tol: differ.add(k)                 # NaN or inf never passes
            if math.isfinite(d): worst = max(worst, d)
    return _s('PASS' if not differ else 'FAIL', fields_compared=len(keys), steps=len(a), max_abs_difference=worst, differing_fields=sorted(differ),
              outcome_equal_reported_not_judged=T.cell(rec, arm)['y'] == T.cell(rec, 'C0')['y'])


def actual_behaviour(rec, tr, arm, metrics=None):
    """What happened, in numbers (POST-HOC GROUND TRUTH for positions; commanded values from the controller record)."""
    c = T.cell(rec, arm); t0 = rec['trigger']['step']; acted = bool(c.get('acted')); k = c['control_steps'] if acted else 0
    s = T.series(rec, tr, arm); ev = T.events(rec, s, arm); i0 = T.at(s, t0); i1 = T.at(s, t0 + k); out = {'control_end_step': t0 + k if acted else None, 'first_open_command_step': ev['release_cmd'], 'success_step': ev['success'],
                                                                                                    'final_success_reported_not_judged': bool(c['y'])}
    steps = ((c.get('control') or {}).get('steps') or []) if acted else []
    if steps:
        cmd = np.array([x['commanded_xy_m'] for x in steps], float); before = s['tcp'][[T.at(s, t0 + j) for j in range(len(steps))], :2]
        out['commanded_step_xy_m'] = cmd.tolist(); out['commanded_target_xy_m'] = (before + cmd).tolist()
        out['commanded_step_norm_sum_m'] = float(np.linalg.norm(cmd, axis=1).sum())
    out['measured_tcp_displacement_m'] = (s['tcp'][i1] - s['tcp'][i0]).tolist(); out['measured_tcp_xy_displacement_norm_m'] = float(np.linalg.norm(s['tcp'][i1, :2] - s['tcp'][i0, :2]))
    out['held_cube_displacement_m'] = (s['A'][i1] - s['A'][i0]).tolist(); out['target_cube_displacement_m'] = (s['B'][i1] - s['B'][i0]).tolist()
    off = np.linalg.norm(T.offset_xy(s), axis=1); out['true_offset_trigger_m'] = float(off[i0]); out['true_offset_end_of_move_m'] = float(off[i1])
    if ev['release_cmd'] is not None: out['true_offset_at_open_command_m'] = float(off[T.at(s, ev['release_cmd'])])
    out['target_displacement_trigger_to_end_m'] = float(np.linalg.norm(s['B'][-1, :2] - s['B'][i0, :2]))
    if metrics:
        win = lambda ph: [m for m in metrics if m['phase'] == ph]
        for ph in ('before_trigger', 'control', 'policy_after'):
            W = win(ph)
            if not W: continue
            out[f'min_clearance_{ph}'] = {p: float(min(m[f'd_{p}'] for m in W)) for p in ('A_B', 'fingerL_B', 'fingerR_B', 'hand_B')}
            out[f'observed_contacts_{ph}'] = {p: int(sum(1 for m in W if m.get(f'observed_contact_{p}'))) for p in ('A_B', 'fingerL_B', 'fingerR_B')}
            out[f'potential_collision_steps_{ph}'] = {p: int(sum(1 for m in W if m.get(f'potential_collision_{p}'))) for p in SCREEN_PAIRS}
            out[f'proximity_steps_{ph}'] = {p: int(sum(1 for m in W if m.get(f'proximity_{p}'))) for p in ('A_B', 'fingerL_B', 'fingerR_B', 'hand_B')}
        out['unobserved_contact_pairs'] = list(UNRECORDED_PAIRS)
    return T._py(out)


# ---------------------------------------------------------------- what the records hold (C1 record sufficiency)
REQUIREMENTS = [
    # (item, status, source / reason, fix)
    ('cube poses (position + orientation) per step', 'AVAILABLE', 'trace rows cubeA, cubeA_q, cubeB, cubeB_q', None),
    ('cube size and collision shape', 'AVAILABLE', 'upstream constant: half size 0.02 m (stack_cube.py); fixed for the task', None),
    ('cube 3D bounding boxes and XY footprints', 'DERIVABLE_FROM_RECORDED_DATA', 'cube pose + size (exact boxes)', None),
    ('TCP pose per step', 'AVAILABLE', 'trace rows tcp, tcp_q', None),
    ('finger joint values per step', 'AVAILABLE', 'trace rows finger_qpos', None),
    ('finger and hand collision shapes per step', 'DERIVABLE_FROM_RECORDED_DATA', 'TCP pose + finger joints + URDF collision boxes / hand hull (exact; validated against recorded finger contacts)', None),
    ('arm link poses (panda_link0..7) per step', 'MISSING', 'arm joint angles are not logged (only the two finger joints)', 'log robot.get_qpos() (9 values) per step; links then follow by forward kinematics from the URDF'),
    ('arm link clearance to table / cubes, whole-robot swept volume', 'MISSING', 'needs arm link poses', 'as above'),
    ('table and obstacle geometry', 'AVAILABLE', 'upstream constant: table box, top at z = 0; no other obstacle in StackCube-v1', None),
    ('observed contact: held cube - target, held cube - fingers, held cube - table, target - fingers', 'AVAILABLE', 'trace rows f_A_B, f_A_left, f_A_right, f_A_table, f_B_left, f_B_right', None),
    ('observed contact: target - hand, held cube - hand, target - table, gripper - table, arm links - anything', 'MISSING', 'not queried by hci.recovery_v1.probe', 'add get_pairwise_contact_forces for (hand, B), (hand, A), (B, table) and (each robot link, B / table) to the probe'),
    ('contact within a control step (5 PhysX sub-steps)', 'REQUIRES_SIMULATOR_INSTRUMENTATION', 'the probe runs after env.step; PhysX contacts of the earlier sub-steps are not visible', 'not proposed (changes the step loop); continuous-time safety is stated as not established'),
    ('commanded XY step and target distance per control step', 'AVAILABLE', 'cells[].control.steps[].commanded_xy_m, target_distance_from_start_m', None),
    ('measured TCP after each control step', 'AVAILABLE', 'cells[].control.steps[].tcp_after and trace rows', None),
    ('controller online estimates (target centre, held cube, offset)', 'AVAILABLE', 'cells[].control.readings[] (vision arms; the oracle has none)', None),
    ('gripper command and opening per step', 'AVAILABLE', 'trace rows action[-1], finger_qpos', None),
    ('policy hand-back (first inference, queue)', 'AVAILABLE', 'cells[].first_inference_step, control_steps; driver checks queue = 0', None),
    ('frame-to-step alignment', 'AVAILABLE', 'record frame_index; cells[].frames_after_trigger.steps (C1); trunk frame k = step k', None),
    ('RGB after the move', 'AVAILABLE', 'C1: frames after_<ARM> at the listed steps (every step to first open + 20, then every 4th); Phase B: none after the move', None),
    ('abstaining arm trajectory', 'AVAILABLE', 'C1: trace rows of every arm; max difference to C0 in the record', None),
]


def requirements_check(rec, tr):
    """Each AVAILABLE item that names record fields is checked on a real record; returns the table with 'present' flags."""
    rows0 = tr['trunk'][0]; arm_rows = [tr[a][0] for a in tr if a != 'trunk' and tr[a]]; out = []
    fields = {'cubeA_q', 'cubeB_q', 'tcp_q', 'finger_qpos', 'f_A_B', 'f_A_left', 'f_A_right', 'f_A_table', 'f_B_left', 'f_B_right', 'action'}
    have = {f: all(f in r for r in [rows0] + arm_rows) for f in fields}
    acted = [c for c in rec['cells'] if c.get('acted')]
    have['commanded_xy_m'] = bool(acted) and all('commanded_xy_m' in x for c in acted for x in (c.get('control') or {}).get('steps', []))
    have['tcp_after'] = bool(acted) and all('tcp_after' in x for c in acted for x in (c.get('control') or {}).get('steps', []))
    have['first_inference_step'] = all('first_inference_step' in c for c in rec['cells'] if not c.get('no_trigger'))
    have['arm_qpos'] = any('qpos' in r or 'robot_qpos' in r for r in [rows0] + arm_rows)
    have['extra_contacts'] = any(k.startswith('f_hand') or k.startswith('f_B_table') for r in [rows0] + arm_rows for k in r)
    for item, st, src, fix in REQUIREMENTS: out.append({'item': item, 'status': st, 'source': src, 'logging_fix': fix})
    return {'items': out, 'fields_present_in_this_record': have}
