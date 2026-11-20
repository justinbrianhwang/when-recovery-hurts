"""Trajectory and behaviour audit of stored Vision-CENTER runs (Phase B now; Phase C1 records later).

Reads ONLY stored records: the scene record (`scenes/<seed>.json`), the per-step trace (`traces/trace_<seed>.json.gz`) and
the RGB frames (`frames/frames_<seed>.npz`). No simulator, no policy, no new scene. Nothing here feeds a controller: the
cube positions in the traces are ground truth logged for after-the-fact analysis, and every figure that shows them is
labelled POST-HOC GROUND TRUTH.

Time convention (checked by `sync_check`): trace row `t` is the state after the t-th executed action, which is also the
observation of step t. Trunk frame k is the observation of step k, so frame k pairs with trace row t = k (frame 0 with the
initial state). Arm rows start at t = trigger + 1. Control frames of a vision arm: frame j is the observation at step
trigger + j. Coordinates: simulator world frame, metres, z up, table top at z = 0.

Checks (C1_BEHAVIOR_EXPECTATION_SPEC.md): each returns PASS, FAIL or NOT_OBSERVABLE with the numbers it used.
"""
import gzip, json
from pathlib import Path
import numpy as np

CUBE_HALF_M = 0.02
RESIDUAL_TOL_M = 0.01          # recovery_v1_oracle.RESIDUAL_TOL_M: a move is needed above this true offset
STEP_CMD_M = 0.005; TARGET_CAP_M = 0.03; MAX_STEPS = 12
DIR_COS_MIN = 0.9              # each actual TCP xy step must point the way of its command (cosine)
PROGRESS_MIN_M = 0.0002        # and move at least 0.2 mm. The command is a per-step delta target from the current TCP
                               # (at most 5 mm); the controller achieves a fraction of it in one step, so commands do not add up to a path
DZ_TOL_M = 0.005               # TCP height drift during a horizontal move
WIDTH_TOL_M = 0.002            # finger opening change during the move
SLIP_TOL_M = 0.005             # change of held cube relative to the TCP during the move
PUSH_TOL_M = 0.001             # target displacement during the move (as the C1 safety criterion)
REVERT_FRACTION_MAX = 0.5      # the policy may undo at most half of the achieved reduction before the release
SAME_TOL = 1e-6
MOVING_M = 0.002               # sync check: the held cube moved more than this between two frames
MEMORY_ERROR_M = 0.005
STATUS = ('PASS', 'FAIL', 'NOT_OBSERVABLE')
CHECKS = ('1_trigger_condition', '2a_acted_only_when_needed', '2b_acted_when_needed', '3_command_reduces_true_error', '4_motion_follows_command', '5_move_bounds',
          '6_no_vertical_or_gripper_change', '7_no_slip_no_push', '8_policy_keeps_correction', '9_abstain_keeps_policy', '10_no_harmful_act_on_wrong_memory')
POSTHOC = 'POST-HOC GROUND TRUTH (not an input of any vision controller)'


# ---------------------------------------------------------------- loading
def load_scene(root, seed, frames=True):
    root = Path(root); rec = json.loads((root / 'scenes' / f'{seed}.json').read_text(encoding='utf-8'))
    with gzip.open(root / rec['trace'], 'rt', encoding='utf-8') as f: tr = json.load(f)['traces']
    fr = None
    if frames and rec.get('frames') and (root / rec['frames']).exists():
        with np.load(root / rec['frames']) as z: fr = {k: z[k] for k in z.files}
    return rec, tr, fr


def cell(rec, arm): return next((x for x in rec['cells'] if x['arm'] == arm), None)
def _a(rows, k): return np.array([r[k] for r in rows], float)


def series(rec, tr, arm):
    """Full episode of one arm: trunk rows up to the trigger, then the arm's own rows. Without a trigger, the trunk."""
    trunk = tr['trunk']
    if not rec['triggered'] or arm == 'trunk': rows = trunk
    else:
        t0 = rec['trigger']['step']; rows = [r for r in trunk if r['t'] <= t0] + tr[arm]
    t = _a(rows, 't').astype(int)
    if np.any(np.diff(t) != 1): raise ValueError(f'scene {rec["scene"]} {arm}: trace steps are not consecutive')
    return {'t': t, 'tcp': _a(rows, 'tcp'), 'tcp_q': _a(rows, 'tcp_q'), 'A': _a(rows, 'cubeA'), 'B': _a(rows, 'cubeB'), 'action': _a(rows, 'action'),
            'width': _a(rows, 'finger_qpos').sum(1), 'grasped': np.array([bool(r['is_cubeA_grasped']) for r in rows]), 'success': np.array([bool(r['success']) for r in rows]),
            'f_A_B': _a(rows, 'f_A_B'), 'f_B_left': _a(rows, 'f_B_left'), 'f_B_right': _a(rows, 'f_B_right')}


def at(s, t):
    i = np.searchsorted(s['t'], t)
    if i >= len(s['t']) or s['t'][i] != t: raise KeyError(t)
    return i


def phases(rec, arm):
    """(policy_end, recovery_end): policy control up to the trigger step, recovery for the control steps, then the policy again."""
    if not rec['triggered']: return None, None
    t0 = rec['trigger']['step']; c = cell(rec, arm); k = c['control_steps'] if c and c.get('acted') else 0
    return t0, t0 + k


def events(rec, s, arm):
    t0, t1 = phases(rec, arm); c = cell(rec, arm)
    after = s['t'] > (t1 if t1 is not None else -1)
    opens = np.where(after & (s['action'][:, -1] > 0))[0]
    succ = np.where(s['success'])[0]
    return {'trigger': t0, 'control_end': t1, 'release_cmd': int(s['t'][opens[0]]) if len(opens) else None, 'success': int(s['t'][succ[0]]) if len(succ) else None,
            'recorded_release': (c or {}).get('gt_release_after_trigger', {}) and c['gt_release_after_trigger'].get('release_step')}


def offset_xy(s): return s['B'][:, :2] - s['A'][:, :2]              # vector the held cube must still travel (true)


# ---------------------------------------------------------------- the checks
def _py(v):
    if isinstance(v, dict): return {k: _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_py(x) for x in v]
    if isinstance(v, np.generic): return v.item()
    return v


def _st(ok, **kw): return _py({'status': 'PASS' if ok else 'FAIL', **kw})
def _no(why, **kw): return _py({'status': 'NOT_OBSERVABLE', 'why': why, **kw})


def check_arm(rec, tr, arm):
    """The ten checks of the expectation spec for one arm of one triggered scene."""
    out = {}; c = cell(rec, arm); trg = rec['trigger']; t0 = trg['step']; s = series(rec, tr, arm); c0 = series(rec, tr, 'C0') if 'C0' in tr else None
    acted = bool(c.get('acted')); k = c['control_steps'] if acted else 0; i0 = at(s, t0); ev = events(rec, s, arm)
    off = offset_xy(s); e0 = float(np.linalg.norm(off[i0]))
    # 1: the trigger rule on the recorded observables, and the finger width against the logged joints
    rule = trg['pending_gripper_command'] > 0 and trg['close_run'] >= 3 and 0.02 <= trg['finger_width_m'] <= 0.06
    wdiff = abs(float(s['width'][i0]) - trg['finger_width_m'])
    out['1_trigger_condition'] = _st(rule and wdiff < 1e-4, pending=trg['pending_gripper_command'], close_run=trg['close_run'], width_m=trg['finger_width_m'], logged_width_diff_m=wdiff,
                                     carrying_by_ground_truth=None if trg.get('gt_for_diagnosis_only') is None else not trg['gt_for_diagnosis_only']['false_trigger'])
    # 2: need
    need = e0 > RESIDUAL_TOL_M
    out['2a_acted_only_when_needed'] = _st(need, true_offset_m=e0, move_steps=k, note='zero move steps: the queue was cleared and the policy re-planned' if acted and k == 0 else None) if acted else _no('the arm did not intervene', true_offset_m=e0)
    out['2b_acted_when_needed'] = (_st(acted, true_offset_m=e0, decision=(c.get('control') or {}).get('decision')) if need else _no('no move needed (true offset within tolerance)', true_offset_m=e0))
    if not acted:
        for kk in ('3_command_reduces_true_error', '4_motion_follows_command', '5_move_bounds', '6_no_vertical_or_gripper_change', '7_no_slip_no_push', '8_policy_keeps_correction'): out[kk] = _no('the arm did not move')
        if c0 is not None and arm != 'C0':
            n = min(len(s['t']), len(c0['t'])); d = max(float(np.abs(s[q][:n] - c0[q][:n]).max()) for q in ('tcp', 'A', 'B')) if n else 0.
            out['9_abstain_keeps_policy'] = _st(len(s['t']) == len(c0['t']) and d <= SAME_TOL and c['y'] == cell(rec, 'C0')['y'], max_state_difference_to_C0=d, steps=(int(s['t'][-1]), int(c0['t'][-1])))
        else: out['9_abstain_keeps_policy'] = _no('reference arm')
        out['10_no_harmful_act_on_wrong_memory'] = _no('the arm did not move')
        return out
    out['9_abstain_keeps_policy'] = _no('the arm moved')
    ctl = c['control']; steps = ctl['steps']; ii = [at(s, t0 + j) for j in range(k + 1)]
    cmd = np.array([x['commanded_xy_m'] for x in steps], float).reshape(-1, 2)
    if len(cmd) != k: return {**out, **{kk: _no(f'{len(cmd)} commanded steps for {k} control steps') for kk in ('3_command_reduces_true_error', '4_motion_follows_command', '5_move_bounds')}}
    # 3: every commanded step points into the true remaining offset
    dots = [float(cmd[j] @ off[ii[j]]) for j in range(k)]; cos0 = float(cmd[0] @ off[ii[0]] / (np.linalg.norm(cmd[0]) * np.linalg.norm(off[ii[0]]) + 1e-12)) if k else None
    out['3_command_reduces_true_error'] = (_st(all(d > 0 for d in dots), first_step_cosine=cos0, steps_against_true_error=sum(1 for d in dots if d <= 0), failed_steps=[j + 1 for j, d in enumerate(dots) if d <= 0], true_offset_before_m=e0, true_offset_after_m=float(np.linalg.norm(off[ii[-1]])))
                                           if k else _no('zero control steps (already within tolerance)'))
    # 4: each actual TCP xy step goes the way of its command and makes progress (magnitudes reported, not judged)
    act = np.diff(s['tcp'][ii, :2], axis=0); an = np.linalg.norm(act, axis=1); cn = np.linalg.norm(cmd, axis=1); cos = (act * cmd).sum(1) / np.maximum(an * cn, 1e-12)
    out['4_motion_follows_command'] = (_st(bool(np.all(cos >= DIR_COS_MIN) and np.all(an >= PROGRESS_MIN_M)), min_step_cosine=float(cos.min()), min_actual_step_m=float(an.min()), failed_steps=[j + 1 for j in range(k) if cos[j] < DIR_COS_MIN or an[j] < PROGRESS_MIN_M], step_cosines=[round(float(x), 3) for x in cos],
                                           actual_over_commanded_step_median=float(np.median(an / np.maximum(cn, 1e-9))), actual_xy_displacement_m=float(np.linalg.norm(act.sum(0))), commanded_step_sum_m=float(cn.sum()))
                                       if k else _no('zero control steps'))
    # 5: bounds of the move
    reach = [x.get('target_distance_from_start_m', 0.) for x in steps]
    out['5_move_bounds'] = _st(k <= MAX_STEPS and (not k or (np.linalg.norm(cmd, axis=1).max() <= STEP_CMD_M + 1e-9 and max(reach) <= TARGET_CAP_M + 1e-9)),
                               steps=k, max_step_m=float(np.linalg.norm(cmd, axis=1).max()) if k else 0., max_target_distance_m=float(max(reach)) if k else 0.)
    # 6: no vertical command, no vertical drift beyond tolerance, gripper command held, fingers not opening
    seg = slice(ii[0], ii[-1] + 1); acts = s['action'][ii[0] + 1: ii[-1] + 1]
    dz = float(np.abs(s['tcp'][seg, 2] - s['tcp'][ii[0], 2]).max()); dw = float(np.abs(s['width'][seg] - s['width'][ii[0]]).max())
    zcmd = float(np.abs(acts[:, 2]).max()) if len(acts) else 0.; grip_held = bool(len(acts) == 0 or np.all(acts[:, -1] == acts[0, -1])) and (len(acts) == 0 or acts[0, -1] < 0)
    out['6_no_vertical_or_gripper_change'] = _st(zcmd == 0. and dz <= DZ_TOL_M and grip_held and dw <= WIDTH_TOL_M, vertical_command_max=zcmd, tcp_dz_max_m=dz, gripper_command_held_closed=grip_held, finger_width_change_max_m=dw)
    # 7: held cube does not slip, target not pushed
    rel = s['A'][seg] - s['tcp'][seg]; slip = float(np.linalg.norm(rel - rel[0], axis=1).max()); push = float(np.linalg.norm(s['B'][ii[-1], :2] - s['B'][ii[0], :2]))
    out['7_no_slip_no_push'] = _st(slip <= SLIP_TOL_M and push <= PUSH_TOL_M and bool(s['grasped'][seg].all()), held_cube_shift_relative_to_tcp_max_m=slip, target_xy_displacement_m=push, grasped_throughout=bool(s['grasped'][seg].all()))
    # 8: after the hand-back, does the policy undo the correction before it opens the gripper?
    r = ev['release_cmd']
    if r is None: out['8_policy_keeps_correction'] = _no('no open command after the move')
    else:
        e1 = float(np.linalg.norm(off[ii[-1]])); er = float(np.linalg.norm(off[at(s, r)])); red = e0 - e1
        frac = (er - e1) / red if red > 0.001 else None
        out['8_policy_keeps_correction'] = (_st(frac <= REVERT_FRACTION_MAX, offset_trigger_m=e0, offset_end_of_move_m=e1, offset_at_open_command_m=er, reverted_fraction=frac, open_command_step=r)
                                            if frac is not None else _no('the move reduced the true offset by less than 1 mm', offset_trigger_m=e0, offset_end_of_move_m=e1, offset_at_open_command_m=er))
    # 10: a move on a remembered position that was wrong: did it do harm or push the target?
    g = c.get('gt_reading_error') or {}; mem = bool(ctl.get('used_memory'))
    if not mem: out['10_no_harmful_act_on_wrong_memory'] = _no('the move did not use a remembered position')
    else:
        wrong = bool(g.get('offset_error_m') is not None and (g['offset_error_m'] > MEMORY_ERROR_M or g.get('error_within_bound') is False))
        harm = cell(rec, 'C0')['y'] == 1 and c['y'] == 0
        out['10_no_harmful_act_on_wrong_memory'] = _st(not (wrong and (harm or push > PUSH_TOL_M)), memory_wrong=wrong, offset_error_m=g.get('offset_error_m'), bound_m=g.get('bound_m'), harm=harm, target_pushed_m=push)
    return out


def scene_metrics(rec, tr):
    """Every arm of one scene: outcome, events and the ten checks."""
    if not rec['triggered']: return {'scene': rec['scene'], 'triggered': False, 'arms': {}}
    c0y = cell(rec, 'C0')['y']; arms = {}
    for arm in rec['arms']:
        c = cell(rec, arm); s = series(rec, tr, arm)
        arms[arm] = {'y': c['y'], 'acted': bool(c.get('acted')), 'harm': bool(c0y == 1 and c['y'] == 0), 'rescue': bool(c0y == 0 and c['y'] == 1), 'events': events(rec, s, arm),
                     'checks': check_arm(rec, tr, arm) if arm != 'C0' else {'9_abstain_keeps_policy': _no('reference arm')}}
    return {'scene': rec['scene'], 'triggered': True, 'trigger': rec['trigger']['step'], 'arms': arms}


# ---------------------------------------------------------------- synchronisation and coordinates
def sync_check(rec, tr, fr, lags=(-2, -1, 0, 1, 2), window=8, min_pixels=5):
    """Frame/trace alignment and camera coordinates, from the stored trunk frames of the base camera.
    For each frame k the red (held cube) and green (target) pixels within `window` px of the projected logged centre are
    reduced to a centroid. Coordinates: median pixel distance between centroid and projection at lag 0. Alignment: on the
    steps where the held cube moved, the mean distance between the centroid of frame k and the projection of trace row
    k + lag; the smallest must be at lag 0. Occlusion (gripper, arm) removes a step rather than biasing it."""
    from . import vision_center as V1
    if fr is None or 'trunk' not in fr: return None
    img = fr['trunk']; rows = {r['t']: r for r in tr['trunk']}; cb = V1.cam_base(); dist = {l: [] for l in lags}; coord = {'cubeA': [], 'cubeB': []}; moving = 0
    for k in range(len(img)):
        if k not in rows: continue
        C = V1.classes(img[k][:, :V1.IMG])
        for name, col in (('cubeA', 'red'), ('cubeB', 'green')):
            uv, z = V1.project(cb, rows[k][name])
            if z <= 0: continue
            u0, v0 = int(round(uv[0])), int(round(uv[1])); ys, xs = np.nonzero(C[col][max(0, v0 - window):v0 + window + 1, max(0, u0 - window):u0 + window + 1])
            if len(xs) < min_pixels: continue
            cen = np.array([xs.mean() + max(0, u0 - window) + .5, ys.mean() + max(0, v0 - window) + .5]); coord[name].append(float(np.linalg.norm(cen - uv)))
            if name == 'cubeA' and k - 1 in rows and np.linalg.norm(np.subtract(rows[k]['cubeA'], rows[k - 1]['cubeA'])) > MOVING_M:
                moving += 1
                for l in lags:
                    if k + l in rows: dist[l].append(float(np.linalg.norm(cen - V1.project(cb, rows[k + l]['cubeA'])[0])))
    mean = {l: (float(np.mean(v)) if v else None) for l, v in dist.items()}; ok = {l: v for l, v in mean.items() if v is not None}
    best = min(ok, key=ok.get) if ok else None
    return {'scene': rec['scene'], 'coordinates_px_median': {k: (float(np.median(v)) if v else None) for k, v in coord.items()}, 'coordinates_n': {k: len(v) for k, v in coord.items()},
            'moving_steps_used': moving, 'mean_px_distance_by_lag': mean, 'best_lag': best, 'aligned': best == 0}


def control_tcp_sync(rec, tr):
    """Exact numeric alignment: the TCP after each control step comes from the policy observation (`tcp_history`, the same
    observation the frames are cut from); the trace row of that step must hold the same TCP. Returns the largest difference."""
    out = {}
    for c in rec['cells']:
        if not c.get('acted') or not (c.get('control') or {}).get('steps') or c['arm'] not in tr: continue
        s = series(rec, tr, c['arm']); t0 = rec['trigger']['step']
        d = [float(np.linalg.norm(np.subtract(x['tcp_after'], s['tcp'][at(s, t0 + j + 1)]))) for j, x in enumerate(c['control']['steps']) if 'tcp_after' in x]
        if d: out[c['arm']] = {'steps': len(d), 'max_difference_m': max(d)}
    return out


def control_frame_check(rec, tr, fr, arm='VISION_CENTER_V2'):
    """Control frames of a vision arm: frame j against arm row t = trigger + j (row of the trigger from the trunk)."""
    from . import vision_center as V1
    if fr is None or arm not in fr or not rec['triggered']: return None
    s = series(rec, tr, arm); t0 = rec['trigger']['step']; img = fr[arm]; hits = 0; n = 0
    for j in range(len(img)):
        try: i = at(s, t0 + j)
        except KeyError: continue
        if img[j].shape[1] != 2 * V1.IMG: return None
        uv, depth = V1.project(V1.cam_base(), s['A'][i])
        if depth <= 0 or not (1 <= uv[0] < V1.IMG - 1 and 1 <= uv[1] < V1.IMG - 1): continue
        n += 1; hits += int(V1.classes(img[j][:, :V1.IMG][int(uv[1]) - 1:int(uv[1]) + 2, int(uv[0]) - 1:int(uv[0]) + 2])['red'].any())
    return {'scene': rec['scene'], 'arm': arm, 'frames': len(img), 'projected_in_view': n, 'red_at_projection_base_camera': hits}


# ---------------------------------------------------------------- figures (matplotlib, Agg)
ARM_STYLE = {'C0': ('#555555', '-'), 'VISION_CENTER_V1': ('#1f77b4', '-'), 'VISION_CENTER_V2': ('#d62728', '-'), 'CENTER_ORACLE': ('#ff7f0e', '--'), 'V2_SINGLE_FRAME': ('#9467bd', '-'),
             'V2_NO_HAND_MEMORY': ('#8c564b', '-'), 'V2_NO_BASE_MEMORY': ('#e377c2', '-'), 'V2_NO_HOLD_MEMORY': ('#7f7f7f', ':'), 'V2_RULES_OFF': ('#bcbd22', '-'), 'VISION_CENTER_V2S': ('#17becf', '-')}


def _plt():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; return plt


def _title(rec, policy): return f"scene {rec['scene']} | policy {policy} | trigger step {rec['trigger']['step']}"


def fig_3d(rec, tr, out, policy='train_seed_1', window=30):
    plt = _plt(); fig = plt.figure(figsize=(11, 8)); ax = fig.add_subplot(111, projection='3d'); t0 = rec['trigger']['step']
    for arm in rec['arms']:
        c = cell(rec, arm)
        if arm != 'C0' and not c.get('acted'): continue
        s = series(rec, tr, arm); m = s['t'] >= t0 - window; col, ls = ARM_STYLE.get(arm, ('k', '-')); ev = events(rec, s, arm)
        ax.plot(*s['tcp'][m].T, color=col, ls=ls, lw=1.6, label=f'{arm} TCP measured')
        ax.plot(*s['A'][m].T, color=col, ls=':', lw=1.0, label=f'{arm} held cube centre')
        if arm == 'C0': ax.plot(*s['B'][m].T, color='#2a9d2a', lw=2.5, label='target cube centre')
        idx = np.where(m)[0][::10]; ax.scatter(*s['tcp'][idx].T, color=col, s=8)          # every 10 steps: direction of time
        for name, mk in (('trigger', 'o'), ('control_end', 's'), ('release_cmd', '^'), ('success', '*')):
            if ev.get(name) is not None and ev[name] in set(s['t']): ax.scatter(*s['tcp'][at(s, ev[name])], color=col, marker=mk, s=60, edgecolor='k')
        if c.get('acted') and c['control'] and c['control'].get('steps'):
            i0 = at(s, t0); cur = s['tcp'][i0].copy(); pts = [cur]
            for j, x in enumerate(c['control']['steps']):
                b = s['tcp'][at(s, t0 + j)]; pts.append([b[0] + x['commanded_xy_m'][0], b[1] + x['commanded_xy_m'][1], b[2]])
            ax.plot(*np.array(pts).T, color=col, ls='--', lw=1.0, marker='x', ms=4, label=f'{arm} commanded TCP targets')
    ax.set_xlabel('x [m] (world)'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m] (table top 0)')
    ax.set_title(_title(rec, policy) + f'\nsteps {t0 - window}..end; markers: o trigger, s end of move, ^ first open command, * success; dots every 10 steps\n' + POSTHOC, fontsize=8)
    ax.legend(fontsize=6, loc='upper left'); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def _square(ax, xy, color, label, ls='-'):
    import matplotlib.patches as P
    ax.add_patch(P.Rectangle((xy[0] - CUBE_HALF_M, xy[1] - CUBE_HALF_M), 2 * CUBE_HALF_M, 2 * CUBE_HALF_M, fill=False, ec=color, ls=ls, lw=1.2, label=label))


def fig_xy(rec, tr, out, policy='train_seed_1'):
    """Top view at the trigger, end of move and first open command. Squares are the true 40 mm cube footprints (axis-aligned
    for readability; the logged yaw is not drawn). Vision estimates are drawn as crosses and labelled as such."""
    plt = _plt(); arms = [a for a in rec['arms'] if a == 'C0' or cell(rec, a).get('acted')]; fig, axs = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 4.4), squeeze=False); t0 = rec['trigger']['step']
    for ax, arm in zip(axs[0], arms):
        s = series(rec, tr, arm); ev = events(rec, s, arm); c = cell(rec, arm); i0 = at(s, t0)
        for name, ls in (('trigger', '-'), ('control_end', '--'), ('release_cmd', ':')):
            if ev.get(name) is None: continue
            i = at(s, ev[name]); _square(ax, s['A'][i], '#d62728', f'held cube @{name} t={ev[name]}', ls); _square(ax, s['B'][i], '#2a9d2a', f'target @{name}', ls)
        ax.plot(*s['tcp'][i0:, :2].T, color='#333', lw=0.8, label='TCP measured (after trigger)')
        rd = (c.get('control') or {}).get('readings') or []
        if rd and rd[0].get('target_xy'):
            tx = np.array(rd[0]['target_xy']); ax.plot(*tx, 'x', color='#2a9d2a', ms=10, mew=2, label=f"target ESTIMATE ({rd[0].get('target_src')})")
            if rd[0].get('held_xyz'): ax.plot(*np.array(rd[0]['held_xyz'])[:2], 'x', color='#d62728', ms=10, mew=2, label=f"held ESTIMATE ({rd[0].get('hold_src')})")
        e = [float(np.linalg.norm(offset_xy(s)[at(s, ev[n])])) * 1000 if ev.get(n) is not None else None for n in ('trigger', 'control_end', 'release_cmd')]
        ax.set_title(f"{arm} y={c['y']}\ntrue XY offset mm: trigger {e[0]:.1f}, end {e[1]:.1f}, open {e[2] if e[2] is None else round(e[2], 1)}", fontsize=8)
        ax.set_aspect('equal'); ax.set_xlabel('x [m] world'); ax.set_ylabel('y [m]'); ax.legend(fontsize=5); ax.grid(alpha=.3)
    fig.suptitle(_title(rec, policy) + ' | ' + POSTHOC + ' (squares), vision estimates (crosses)', fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_timeline(rec, tr, out, policy='train_seed_1', window=30):
    plt = _plt(); t0 = rec['trigger']['step']; fig, axs = plt.subplots(7, 1, figsize=(10, 15), sharex=True)
    for arm in rec['arms']:
        c = cell(rec, arm)
        if arm != 'C0' and not c.get('acted'): continue
        s = series(rec, tr, arm); m = s['t'] >= t0 - window; t = s['t'][m]; col, ls = ARM_STYLE.get(arm, ('k', '-')); kw = dict(color=col, ls=ls, lw=1.2, label=arm)
        axs[0].plot(t, np.linalg.norm(offset_xy(s)[m], axis=1) * 1000, **kw)
        axs[1].plot(t, s['tcp'][m, 2] * 1000, **kw); axs[1].plot(t, s['A'][m, 2] * 1000, color=col, ls=':', lw=.8)
        axs[2].plot(t, np.linalg.norm(s['B'][m, :2] - s['B'][at(s, t0), :2], axis=1) * 1000, **kw)
        axs[3].plot(t, np.linalg.norm((s['A'] - s['tcp'])[m] - (s['A'] - s['tcp'])[at(s, t0)], axis=1) * 1000, **kw)
        axs[4].plot(t, s['action'][m, -1], **kw); axs[4].plot(t, s['width'][m] * 10, color=col, ls=':', lw=.8)
        rd = (c.get('control') or {}).get('readings') or []
        if rd:
            tt = [x['t'] for x in rd if x.get('offset_norm_m') is not None]; axs[0].plot(tt, [x['offset_norm_m'] * 1000 for x in rd if x.get('offset_norm_m') is not None], 'o', color=col, ms=3, label=f'{arm} ESTIMATED offset')
            uu = [(x['t'], x['u']) for x in rd if x.get('u') is not None]
            if uu: axs[5].plot(*np.array(uu).T * [[1], [1000]], 's', color=col, ms=3, label=f'{arm} uncertainty u')
        g = c.get('gt_reading_error') or {}
        if g.get('offset_error_m') is not None: axs[5].axhline(g['offset_error_m'] * 1000, color=col, ls='--', lw=.8, label=f'{arm} offset error at trigger (post-hoc)')
        if c.get('acted') and (c.get('control') or {}).get('steps'):
            st = c['control']['steps']; tt = np.arange(t0 + 1, t0 + 1 + len(st)); axs[6].plot(tt, [np.linalg.norm(x['commanded_xy_m']) * 1000 for x in st], 'x--', color=col, label=f'{arm} commanded step')
            axs[6].plot(tt, [np.linalg.norm(s['tcp'][at(s, q), :2] - s['tcp'][at(s, q - 1), :2]) * 1000 for q in tt], 'o-', color=col, ms=3, label=f'{arm} actual TCP step')
    for ax, lab in zip(axs, ('true XY offset held-target [mm]', 'height [mm]: TCP solid, held cube dotted', 'target displacement since trigger [mm]', 'held cube shift rel. TCP since trigger [mm]',
                             'gripper command (solid, >0 open) / finger width x10 [m] (dotted)', 'memory uncertainty u / offset error [mm]', 'move step [mm]')):
        ax.set_ylabel(lab, fontsize=7); ax.grid(alpha=.3); ax.axvspan(t0 - window, t0, color='#eeeeee', zorder=-1)
        for arm in rec['arms']:
            c = cell(rec, arm)
            if c.get('acted') and c.get('control_steps'): ax.axvspan(t0, t0 + c['control_steps'], color=ARM_STYLE.get(arm, ('k',))[0], alpha=.08, zorder=-1)
    axs[0].legend(fontsize=6); axs[5].legend(fontsize=6); axs[6].legend(fontsize=6); axs[-1].set_xlabel('step (grey: policy before trigger; tinted: recovery move of that arm; white: policy resumed)')
    fig.suptitle(_title(rec, policy) + '\n' + POSTHOC + ' except estimated offset and u', fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


# ---------------------------------------------------------------- RGB evidence
def _font():
    from PIL import ImageFont
    try: return ImageFont.load_default(size=11)
    except TypeError: return ImageFont.load_default()


def rgb_panels(rec, tr, fr, policy='train_seed_1', window=10, overlay=False, scale=2):
    """Frames as recorded, side by side: the trunk up to the trigger, then each arm's control frames and, in C1 records,
    the frames kept after the trigger (`after_<ARM>`, steps in cells[].frames_after_trigger). Steps without a recorded frame
    are shown as an explicit grey panel saying so; nothing is interpolated or re-rendered.
    overlay=True draws the logged cube centres projected into the base image, labelled POST-HOC GROUND TRUTH."""
    from PIL import Image, ImageDraw
    from . import vision_center as V1
    t0 = rec['trigger']['step']; arms = [a for a in rec['arms'] if a != 'C0' and cell(rec, a).get('acted')]; cols = ['C0'] + arms
    after = {a: dict(zip(cell(rec, a)['frames_after_trigger']['steps'], fr[cell(rec, a)['frames_after_trigger']['key']])) for a in cols
             if fr is not None and (cell(rec, a).get('frames_after_trigger') or {}).get('key') in fr}
    last = max([t0] + [t0 + cell(rec, a)['control_steps'] for a in arms]) + 3
    if after: last = max([last] + [(cell(rec, a).get('first_open_command_after_move') or t0) + 5 for a in after])
    W, H = 2 * V1.IMG * scale, V1.IMG * scale; f = _font(); out = []
    rows = {r['t']: r for r in tr['trunk']}
    later = sorted({q for v in after.values() for q in v if q > last})                       # sparse frames kept after the window: shown, not filled in
    for t in [*range(max(0, t0 - window), last + 1), *later]:
        canvas = Image.new('RGB', (W * len(cols), H + 44), (255, 255, 255)); d = ImageDraw.Draw(canvas)
        for j, arm in enumerate(cols):
            img = None; note = None
            if t <= t0 and fr is not None and t < len(fr['trunk']): img = fr['trunk'][t]; note = 'policy' if t < t0 else 'trigger'
            elif arm in (fr or {}) and t - t0 < len(fr[arm]):
                a = fr[arm][t - t0]; img = a if a.shape[1] == 2 * V1.IMG else np.concatenate([np.full_like(a, 90), a], 1); note = 'recovery move' + ('' if a.shape[1] == 2 * V1.IMG else ' (hand camera only recorded)')
            elif t in after.get(arm, {}):
                img = after[arm][t]; c = cell(rec, arm); note = 'recovery move' if c.get('acted') and t <= t0 + c['control_steps'] else 'policy resumed'
            if img is None:
                d.rectangle([j * W, 44, (j + 1) * W - 1, 44 + H - 1], fill=(170, 170, 170)); d.text((j * W + 8, 44 + H // 2 - 12), 'NO RGB RECORDED\nfor this arm and step', fill=(0, 0, 0), font=f)
            else:
                pim = Image.fromarray(np.asarray(img, np.uint8)).resize((W, H), Image.NEAREST); canvas.paste(pim, (j * W, 44))
                if overlay:
                    s = series(rec, tr, arm) if arm in tr or arm == 'C0' else None; r = None
                    if t <= t0: r = rows.get(t)
                    elif s is not None and t in set(s['t']): i = at(s, t); r = {'cubeA': s['A'][i], 'cubeB': s['B'][i]}
                    if r is not None:
                        for X, colr in ((r['cubeA'], (255, 0, 255)), (r['cubeB'], (0, 255, 255))):
                            uv, z = V1.project(V1.cam_base(), np.asarray(X, float) + [0, 0, CUBE_HALF_M])
                            if z > 0:
                                x, y = j * W + uv[0] * scale, 44 + uv[1] * scale; d.line([x - 7, y, x + 7, y], fill=colr, width=3); d.line([x, y - 7, x, y + 7], fill=colr, width=3)
            c = cell(rec, arm); rd = ((c or {}).get('control') or {}).get('readings') or []
            src = f"{rd[0].get('target_src')}/{rd[0].get('hold_src')}" if rd else '-'
            d.text((j * W + 3, 2), f"{arm}\n{note or 'no frame'} | {(c.get('control') or {}).get('decision', '') if c else ''} | src {src}", fill=(0, 0, 0), font=f)
        d.text((3, 30), f"scene {rec['scene']} policy {policy} step {t} trigger {t0}" + ('   ' + POSTHOC + ': magenta = held cube, cyan = target' if overlay else ''), fill=(180, 0, 0) if overlay else (0, 0, 0), font=f)
        out.append(canvas)
    return out


def save_gif(panels, path, ms=200):
    panels[0].save(path, save_all=True, append_images=panels[1:], duration=ms, loop=0)


def outcome_strip(rec, tr, out, policy='train_seed_1'):
    """What happens after the move, from the trace only (no RGB exists there in Phase B): one row per arm, the true XY
    offset and the events to the end of the episode."""
    plt = _plt(); t0 = rec['trigger']['step']; fig, ax = plt.subplots(figsize=(10, 3))
    for arm in rec['arms']:
        s = series(rec, tr, arm); m = s['t'] >= t0; col, ls = ARM_STYLE.get(arm, ('k', '-')); ev = events(rec, s, arm)
        ax.plot(s['t'][m], np.linalg.norm(offset_xy(s)[m], axis=1) * 1000, color=col, ls=ls, label=f"{arm} y={cell(rec, arm)['y']} open@{ev['release_cmd']} success@{ev['success']}")
    ax.set_xlabel('step'); ax.set_ylabel('true XY offset [mm]'); ax.legend(fontsize=6); ax.grid(alpha=.3); ax.set_title(_title(rec, policy) + ' | after the trigger, ' + POSTHOC + ' (trace; no RGB recorded)', fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)
