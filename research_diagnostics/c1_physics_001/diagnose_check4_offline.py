"""B: decomposition of the trajectory-audit Check 4 failures (POST_HOC_EXPLORATORY; the original Check 4 stays as it is).

Check 4 (hci.trajectory_audit, unchanged): for every control step j of a move, the actual TCP xy step (row t0+j-1 -> t0+j)
must have cosine >= DIR_COS_MIN = 0.9 with its commanded xy step and length >= PROGRESS_MIN_M = 0.2 mm.

The command is the xy part of a pd_ee_delta_pos action: an end-effector delta TARGET, relative to the measured TCP when the
action is set (pinned ManiSkill a4a4f92: use_target False, interpolate False; the target joint positions come from IK and
are held by joint PD drives, stiffness 1e3, damping 1e2, force limit 100, over 5 PhysX sub-steps). It is not a torque and
not a displacement the controller promises.

Sources: trajectory_audit_001/per_scene_checks.csv decides which arm/scene FAILED (not any list from elsewhere); the raw
traces give every step. Quantities kept apart (never one 'tracking ratio'):
  (1) sum |cmd step| vs sum |TCP step|          (path lengths)
  (2) |sum cmd step| vs |sum TCP step|          (net displacements)
  (3) per-step cosine and progress (the Check 4 quantities)
  (4) lagged relations (TCP step j against command j-1, TCP step j-1, the motion before the trigger)
  (5) the true held-cube-to-target offset at the end of the move, rescue / harm against C0.
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from hci import trajectory_audit as T  # noqa: E402

DIR_COS_MIN = T.DIR_COS_MIN; PROGRESS_MIN_M = T.PROGRESS_MIN_M
ACTION_SCALE_M = 0.1            # pos_upper of arm_pd_ee_delta_pos (pinned ManiSkill panda.py); checked at run time by recovery_v1_oracle.arm_scale
ROBOT = ('panda_link', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger', 'camera_link')


def unit(v): n = np.linalg.norm(v); return v / n if n > 0 else v * 0
def cos(a, b): na, nb = np.linalg.norm(a), np.linalg.norm(b); return float(a @ b / (na * nb)) if na > 0 and nb > 0 else None


def robot_impulse(row, other):
    """Sum of |point impulse| of contacts between any robot link and `other` in the logged contact list of the last sub-step."""
    s = 0.
    for c in row.get('contacts', []):
        names = (c['a'], c['b'])
        if other in names and any(n.startswith(ROBOT) for n in names): s += sum(float(np.linalg.norm(p[6:9])) for p in c['points'])
    return s


def audit_rows(path):
    out = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if r['check'] == '4_motion_follows_command': out[(r['policy'], int(r['scene']), r['arm'])] = r
    return out


def analyse_move(policy, scene, arm, audit):
    rec, rec_sha = C.record(policy, scene); tr, tr_sha = C.traces(policy, scene); c = C.cell(rec, arm)
    if not (rec.get('triggered') and c and c.get('acted') and c['control_steps']): return None, []
    t0 = rec['trigger']['step']; k = int(c['control_steps']); s = T.series(rec, tr, arm)
    rows = {int(r['t']): r for r in [r for r in tr['trunk'] if r['t'] <= t0] + tr[arm]}
    official = T.check_arm(rec, tr, arm)['4_motion_follows_command']           # the unchanged audit function, re-run on the same trace
    cmd = np.array([x['commanded_xy_m'] for x in c['control']['steps']], float)
    tcp = np.array([rows[t0 + j]['tcp'] for j in range(-2, k + 1)], float)     # rows t0-2 .. t0+k
    act = np.diff(tcp[2:, :2], axis=0); v_prev = tcp[2, :2] - tcp[1, :2]; v_prev2 = tcp[1, :2] - tcp[0, :2]
    prev_action = np.asarray(rows[t0]['action'], float); prev_cmd = prev_action[:2] * ACTION_SCALE_M        # the policy action executed t0-1 -> t0
    qvel0 = np.asarray(rows[t0]['robot_qvel'][:7], float)
    c0 = C.cell(rec, 'C0'); y, yc0 = int(c['y']), int(c0['y']); oc = 'RESCUE' if (y, yc0) == (1, 0) else 'HARM' if (y, yc0) == (0, 1) else 'BOTH_SUCCESS' if y else 'BOTH_FAIL'
    off = lambda t: np.linalg.norm(np.asarray(rows[t]['cubeB'][:2]) - np.asarray(rows[t]['cubeA'][:2]))
    a_row = audit.get((policy, scene, arm))
    steps = []
    for j in range(k):
        cj, aj = cmd[j], act[j]; ch = unit(cj); prog = float(aj @ ch); lat = float(np.linalg.norm(aj - prog * ch)); cs = cos(aj, cj)
        rt = rows[t0 + j + 1]
        steps.append({'source_kind': C.SOURCE_KIND, 'label': C.LABEL, 'policy': policy, 'scene': scene, 'arm': arm, 'step': j + 1, 't': t0 + j + 1,
                      'record_sha256': rec_sha, 'trace_sha256': tr_sha,
                      'cmd_x_mm': cj[0] * 1e3, 'cmd_y_mm': cj[1] * 1e3, 'cmd_norm_mm': np.linalg.norm(cj) * 1e3,
                      'act_x_mm': aj[0] * 1e3, 'act_y_mm': aj[1] * 1e3, 'act_norm_mm': np.linalg.norm(aj) * 1e3, 'act_dz_mm': (tcp[j + 3, 2] - tcp[j + 2, 2]) * 1e3,
                      'cos_act_cmd': cs, 'progress_along_cmd_mm': prog * 1e3, 'lateral_mm': lat * 1e3, 'act_over_cmd_norm': np.linalg.norm(aj) / np.linalg.norm(cj),
                      'cos_fail': cs is None or cs < DIR_COS_MIN, 'progress_fail': np.linalg.norm(aj) < PROGRESS_MIN_M,
                      'cos_act_prev_cmd': cos(aj, cmd[j - 1]) if j else None, 'cos_act_prev_act': cos(aj, act[j - 1]) if j else cos(aj, v_prev),
                      'cos_act_motion_before_trigger': cos(aj, v_prev), 'robot_cubeB_impulse_Ns': robot_impulse(rt, 'cubeB'), 'robot_table_impulse_Ns': robot_impulse(rt, 'table'),
                      'arm_qvel_norm_rad_s': float(np.linalg.norm(rt['robot_qvel'][:7]))})
    fails = [x['step'] for x in steps if x['cos_fail'] or x['progress_fail']]
    m = {'source_kind': C.SOURCE_KIND, 'label': C.LABEL, 'policy': policy, 'scene': scene, 'arm': arm, 'record_sha256': rec_sha, 'trace_sha256': tr_sha,
         'trigger_step': t0, 'control_steps': k, 'end_reason': c['control'].get('end_reason'),
         'audit_csv_status': a_row['status'] if a_row else None, 'rerun_audit_status': official['status'], 'recomputed_status': 'FAIL' if fails else 'PASS',
         'failed_steps': ' '.join(map(str, fails)), 'n_cos_fail_steps': sum(x['cos_fail'] for x in steps), 'n_progress_fail_steps': sum(x['progress_fail'] for x in steps),
         'first_step_cos': steps[0]['cos_act_cmd'], 'first_step_act_mm': steps[0]['act_norm_mm'], 'first_step_cmd_mm': steps[0]['cmd_norm_mm'],
         'step2plus_min_cos': min((x['cos_act_cmd'] for x in steps[1:]), default=None), 'step2plus_min_act_mm': min((x['act_norm_mm'] for x in steps[1:]), default=None),
         'POST_HOC_STEP2PLUS_status': 'NOT_APPLICABLE_ONE_STEP' if k == 1 else ('FAIL' if any(s in fails for s in range(2, k + 1)) else 'PASS'),
         'motion_before_trigger_mm': np.linalg.norm(v_prev) * 1e3, 'motion_before_trigger_x_mm': v_prev[0] * 1e3, 'motion_before_trigger_y_mm': v_prev[1] * 1e3,
         'motion_two_before_trigger_mm': np.linalg.norm(v_prev2) * 1e3, 'cos_cmd1_motion_before_trigger': cos(cmd[0], v_prev),
         'cos_act1_motion_before_trigger': cos(act[0], v_prev), 'prev_policy_cmd_xy_mm': np.linalg.norm(prev_cmd) * 1e3, 'cos_cmd1_prev_policy_cmd': cos(cmd[0], prev_cmd),
         'prev_policy_dz_cmd_mm': prev_action[2] * ACTION_SCALE_M * 1e3, 'prev_policy_gripper': prev_action[-1], 'arm_qvel_norm_at_trigger_rad_s': float(np.linalg.norm(qvel0)),
         'path_cmd_mm': np.linalg.norm(cmd, axis=1).sum() * 1e3, 'path_act_mm': np.linalg.norm(act, axis=1).sum() * 1e3,
         'net_cmd_mm': np.linalg.norm(cmd.sum(0)) * 1e3, 'net_act_mm': np.linalg.norm(act.sum(0)) * 1e3,
         'true_offset_before_mm': off(t0) * 1e3, 'true_offset_after_mm': off(t0 + k) * 1e3, 'y_arm': y, 'y_C0': yc0, 'outcome_vs_C0': oc}
    return m, steps


def loso_fit(X, y, groups):
    """Least squares y = X b, leave-one-scene-out predictions (scenes shared by both policies and all arms are held out together)."""
    pred = np.zeros_like(y); coefs = []
    for g in np.unique(groups):
        tr = groups != g; b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None); pred[~tr] = X[~tr] @ b; coefs.append(b)
    b_all, *_ = np.linalg.lstsq(X, y, rcond=None); return pred, b_all, np.array(coefs)


def first_step_models(moves, steps):
    """Exploratory, associational: does the motion before the trigger account for the first-step direction?
    Model M0: act_1 = a * cmd_1.  M1: act_1 = a * cmd_1 + b * v_prev (v_prev = TCP step t0-1 -> t0). Fitted on the xy
    components of every moved branch of every arm (one vector per move), leave-one-scene-out. Later steps: M2 act_j =
    a * cmd_j + b * act_{j-1} for j >= 2."""
    S = {(m['policy'], m['scene'], m['arm']): [s for s in steps if (s['policy'], s['scene'], s['arm']) == (m['policy'], m['scene'], m['arm'])] for m in moves}
    rows = []
    for m in moves:
        st = S[(m['policy'], m['scene'], m['arm'])]; s1 = st[0]
        rows.append((m, np.array([s1['cmd_x_mm'], s1['cmd_y_mm']]), np.array([m['motion_before_trigger_x_mm'], m['motion_before_trigger_y_mm']]), np.array([s1['act_x_mm'], s1['act_y_mm']])))
    groups = np.repeat([r[0]['scene'] for r in rows], 2); y = np.concatenate([r[3] for r in rows])
    X0 = np.concatenate([r[1] for r in rows])[:, None]; X1 = np.stack([np.concatenate([r[1] for r in rows]), np.concatenate([r[2] for r in rows])], 1)
    p0, b0, _ = loso_fit(X0, y, groups); p1, b1, cf1 = loso_fit(X1, y, groups)
    out = {'n_moves': len(rows), 'M0_coef_a': float(b0[0]), 'M1_coef_a': float(b1[0]), 'M1_coef_b': float(b1[1]),
           'M1_coef_b_loso_range': [float(cf1[:, 1].min()), float(cf1[:, 1].max())],
           'M0_loso_rmse_mm': float(np.sqrt(np.mean((p0 - y) ** 2))), 'M1_loso_rmse_mm': float(np.sqrt(np.mean((p1 - y) ** 2)))}
    pred_cos = []
    for i, (m, cmd, vp, act) in enumerate(rows):
        pr = p1[2 * i:2 * i + 2]; pred_cos.append({'policy': m['policy'], 'scene': m['scene'], 'arm': m['arm'], 'actual_first_cos': cos(act, cmd), 'M1_predicted_first_cos': cos(pr, cmd),
                                                   'M0_predicted_first_cos': cos(p0[2 * i:2 * i + 2], cmd), 'audit_fail': m['audit_csv_status'] == 'FAIL'})
    a = np.array([r['actual_first_cos'] < DIR_COS_MIN for r in pred_cos]); p = np.array([r['M1_predicted_first_cos'] < DIR_COS_MIN for r in pred_cos])
    out['M1_loso_first_step_cos_fail_classification'] = {'actual_fail': int(a.sum()), 'predicted_fail': int(p.sum()), 'both': int((a & p).sum()), 'actual_only': int((a & ~p).sum()), 'predicted_only': int((~a & p).sum())}
    out['M1_loso_cos_correlation_actual_vs_predicted'] = float(np.corrcoef([r['actual_first_cos'] for r in pred_cos], [r['M1_predicted_first_cos'] for r in pred_cos])[0, 1])
    # later steps
    Xs, ys, gs = [], [], []
    for m in moves:
        st = S[(m['policy'], m['scene'], m['arm'])]
        for j in range(1, len(st)):
            for ax in ('x', 'y'): Xs.append([st[j][f'cmd_{ax}_mm'], st[j - 1][f'act_{ax}_mm']]); ys.append(st[j][f'act_{ax}_mm']); gs.append(m['scene'])
    if Xs:
        Xs = np.array(Xs); ys = np.array(ys); gs = np.array(gs); p2, b2, _ = loso_fit(Xs, ys, gs); p2c, b2c, _ = loso_fit(Xs[:, :1], ys, gs)
        out.update({'M2_steps': int(len(ys) // 2), 'M2_coef_a': float(b2[0]), 'M2_coef_b_prev_act': float(b2[1]), 'M2_loso_rmse_mm': float(np.sqrt(np.mean((p2 - ys) ** 2))),
                    'M2_cmd_only_loso_rmse_mm': float(np.sqrt(np.mean((p2c - ys) ** 2)))})
    out['reading'] = 'associational (least squares on stored trajectories); not evidence of PD dynamics or of a cause'
    return out, pred_cos


def summary(moves, steps, models):
    v2 = [m for m in moves if m['arm'] == 'VISION_CENTER_V2']; out = {'label': C.LABEL, 'source_kind': C.SOURCE_KIND, 'thresholds_unchanged': {'DIR_COS_MIN': DIR_COS_MIN, 'PROGRESS_MIN_M': PROGRESS_MIN_M}}
    out['original_check4_V2'] = {p: {'moves': sum(1 for m in v2 if m['policy'] == p), 'FAIL_audit_csv': sum(1 for m in v2 if m['policy'] == p and m['audit_csv_status'] == 'FAIL'),
                                     'FAIL_rerun_audit_function': sum(1 for m in v2 if m['policy'] == p and m['rerun_audit_status'] == 'FAIL'),
                                     'FAIL_independent_recompute': sum(1 for m in v2 if m['policy'] == p and m['recomputed_status'] == 'FAIL')} for p in C.POLICIES}
    out['original_check4_V2_expected'] = {'train_seed_1': '9/41', 'train_seed_2': '5/39'}
    out['agreement_audit_csv_vs_recompute_all_arms'] = {'moves': len(moves), 'disagreements': sum(1 for m in moves if m['audit_csv_status'] != m['recomputed_status'] or m['rerun_audit_status'] != m['recomputed_status'])}
    f = [m for m in v2 if m['audit_csv_status'] == 'FAIL']; ok = [m for m in v2 if m['audit_csv_status'] == 'PASS']
    out['V2_failures'] = {'n': len(f), 'failed_step_sets': sorted({m['failed_steps'] for m in f}), 'cos_only': sum(1 for m in f if m['n_cos_fail_steps'] and not m['n_progress_fail_steps']),
                          'progress_only': sum(1 for m in f if m['n_progress_fail_steps'] and not m['n_cos_fail_steps']), 'both_conditions': sum(1 for m in f if m['n_progress_fail_steps'] and m['n_cos_fail_steps']),
                          'POST_HOC_STEP2PLUS': {k: sum(1 for m in f if m['POST_HOC_STEP2PLUS_status'] == k) for k in ('PASS', 'FAIL', 'NOT_APPLICABLE_ONE_STEP')},
                          'outcomes': {k: sum(1 for m in f if m['outcome_vs_C0'] == k) for k in ('RESCUE', 'HARM', 'BOTH_SUCCESS', 'BOTH_FAIL')},
                          'intersection_with_harm': [(m['policy'], m['scene']) for m in f if m['outcome_vs_C0'] == 'HARM']}
    def dist(ms, k):
        x = np.array([m[k] for m in ms if m[k] is not None], float); return {'n': len(x), 'median': float(np.median(x)) if len(x) else None, 'min': float(x.min()) if len(x) else None, 'max': float(x.max()) if len(x) else None}
    keys = ('first_step_cos', 'first_step_act_mm', 'motion_before_trigger_mm', 'cos_cmd1_motion_before_trigger', 'cos_act1_motion_before_trigger', 'arm_qvel_norm_at_trigger_rad_s',
            'step2plus_min_cos', 'path_cmd_mm', 'path_act_mm', 'net_cmd_mm', 'net_act_mm', 'true_offset_after_mm', 'prev_policy_cmd_xy_mm', 'cos_cmd1_prev_policy_cmd')
    out['V2_fail_vs_pass'] = {k: {'FAIL': dist(f, k), 'PASS': dist(ok, k)} for k in keys}
    st_v2 = [s for s in steps if s['arm'] == 'VISION_CENTER_V2']
    out['V2_steps_first_vs_later'] = {'first': {'n': sum(1 for s in st_v2 if s['step'] == 1), 'cos_fail': sum(1 for s in st_v2 if s['step'] == 1 and s['cos_fail']), 'progress_fail': sum(1 for s in st_v2 if s['step'] == 1 and s['progress_fail']),
                                                'median_act_over_cmd': float(np.median([s['act_over_cmd_norm'] for s in st_v2 if s['step'] == 1]))},
                                      'later': {'n': sum(1 for s in st_v2 if s['step'] > 1), 'cos_fail': sum(1 for s in st_v2 if s['step'] > 1 and s['cos_fail']), 'progress_fail': sum(1 for s in st_v2 if s['step'] > 1 and s['progress_fail']),
                                                'median_act_over_cmd': float(np.median([s['act_over_cmd_norm'] for s in st_v2 if s['step'] > 1]))},
                                      'by_step_index_median_act_over_cmd': {j: float(np.median([s['act_over_cmd_norm'] for s in st_v2 if s['step'] == j])) for j in sorted({s['step'] for s in st_v2})}}
    allf = [s for s in steps if s['cos_fail'] or s['progress_fail']]
    out['all_arms_failed_steps_by_index'] = {j: sum(1 for s in allf if s['step'] == j) for j in sorted({s['step'] for s in allf})}
    out['all_arms_moves'] = {'moves': len(moves), 'FAIL': sum(1 for m in moves if m['recomputed_status'] == 'FAIL'), 'STEP2PLUS_FAIL': sum(1 for m in moves if m['POST_HOC_STEP2PLUS_status'] == 'FAIL'),
                             'by_arm': {a: {'moves': sum(1 for m in moves if m['arm'] == a), 'FAIL': sum(1 for m in moves if m['arm'] == a and m['recomputed_status'] == 'FAIL'),
                                            'STEP2PLUS_FAIL': sum(1 for m in moves if m['arm'] == a and m['POST_HOC_STEP2PLUS_status'] == 'FAIL')} for a in sorted({m['arm'] for m in moves})}}
    out['contact_in_failed_first_steps'] = {'V2_failed_first_steps_with_robot_cubeB_impulse': sum(1 for s in st_v2 if s['step'] == 1 and s['cos_fail'] and s['robot_cubeB_impulse_Ns'] > 0),
                                            'V2_failed_first_steps_with_robot_table_impulse': sum(1 for s in st_v2 if s['step'] == 1 and s['cos_fail'] and s['robot_table_impulse_Ns'] > 0)}
    out['first_step_models_EXPLORATORY'] = models
    return out


def figures(moves, steps, pred, fdir):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    v2 = [m for m in moves if m['arm'] == 'VISION_CENTER_V2']; fail = {(m['policy'], m['scene']) for m in v2 if m['audit_csv_status'] == 'FAIL'}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=110)
    for m in v2:
        st = [s for s in steps if (s['policy'], s['scene'], s['arm']) == (m['policy'], m['scene'], 'VISION_CENTER_V2')]
        f = (m['policy'], m['scene']) in fail
        ax[0].plot([s['step'] for s in st], [s['cos_act_cmd'] for s in st], color='#c0392b' if f else '#95a5a6', lw=1.2 if f else .6, alpha=.9 if f else .5, marker='o' if f else None, ms=3)
        ax[1].plot([s['step'] for s in st], [s['act_over_cmd_norm'] for s in st], color='#c0392b' if f else '#95a5a6', lw=1.2 if f else .6, alpha=.9 if f else .5)
    ax[0].axhline(DIR_COS_MIN, color='k', ls='--', lw=.8); ax[0].set_ylim(0.5, 1.01); ax[0].set_xlabel('control step of the move'); ax[0].set_ylabel('cos(actual TCP xy step, commanded step)')
    ax[0].set_title('B: V2 moves, both policies (red: the 14 Check 4 FAILs; dashed: 0.9)', fontsize=9, loc='left')
    ax[1].set_xlabel('control step of the move'); ax[1].set_ylabel('|actual TCP xy step| / |commanded step|'); ax[1].set_title('B: achieved fraction of the commanded delta target per step', fontsize=9, loc='left')
    fig.tight_layout(); fig.savefig(fdir / 'check4_v2_stepwise.png'); plt.close(fig)
    f14 = [m for m in v2 if (m['policy'], m['scene']) in fail]
    fig, axs = plt.subplots(2, 7, figsize=(17, 5.4), dpi=110)
    for ax, m in zip(axs.ravel(), f14):
        s1 = next(s for s in steps if (s['policy'], s['scene'], s['arm'], s['step']) == (m['policy'], m['scene'], 'VISION_CENTER_V2', 1))
        for v, col, lab in (((s1['cmd_x_mm'], s1['cmd_y_mm']), '#2a6fdb', 'command 1'), ((s1['act_x_mm'], s1['act_y_mm']), '#c0392b', 'actual step 1'),
                            ((m['motion_before_trigger_x_mm'], m['motion_before_trigger_y_mm']), '#7f8c8d', 'TCP step before trigger')):
            ax.annotate('', xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=col, lw=1.6)); ax.plot([], [], color=col, label=lab)
        lim = max(5, max(abs(x) for x in (s1['cmd_x_mm'], s1['cmd_y_mm'], s1['act_x_mm'], s1['act_y_mm'], m['motion_before_trigger_x_mm'], m['motion_before_trigger_y_mm']))) * 1.15
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal'); ax.axhline(0, color='#ddd', lw=.5); ax.axvline(0, color='#ddd', lw=.5)
        ax.set_title(f"{m['policy'][-6:]} {m['scene']}\ncos {m['first_step_cos']:.2f} | {m['outcome_vs_C0']}", fontsize=7); ax.tick_params(labelsize=6)
    axs[0, 0].legend(fontsize=5.5, loc='lower left'); fig.suptitle('B: first control step of the 14 V2 Check 4 FAILs, xy in mm (raw trace; POST_HOC_EXPLORATORY)', fontsize=9)
    fig.tight_layout(); fig.savefig(fdir / 'check4_first_step_vectors.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=110)
    a = np.array([r['actual_first_cos'] for r in pred]); p = np.array([r['M1_predicted_first_cos'] for r in pred]); f = np.array([r['audit_fail'] for r in pred])
    ax.scatter(p[~f], a[~f], s=8, color='#95a5a6', label='Check 4 PASS (all arms)'); ax.scatter(p[f], a[f], s=12, color='#c0392b', label='Check 4 FAIL (all arms)')
    ax.plot([-.2, 1], [-.2, 1], color='k', lw=.6); ax.axhline(.9, ls='--', color='k', lw=.6); ax.axvline(.9, ls='--', color='k', lw=.6)
    ax.set_xlabel('first-step cosine predicted by M1 (command + motion before trigger, leave-one-scene-out)', fontsize=7); ax.set_ylabel('actual first-step cosine'); ax.legend(fontsize=7)
    ax.set_title('B: exploratory association, not a mechanism', fontsize=9, loc='left'); fig.tight_layout(); fig.savefig(fdir / 'check4_first_step_model.png'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(C.OUT)); a = ap.parse_args(); out = Path(a.out); t0 = time.time()
    if (out / 'CHECK4_SUMMARY.json').exists(): raise SystemExit(f'refused: {out} already holds a Check 4 diagnosis')
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    src = C.TRAJ / 'per_scene_checks.csv'; audit = audit_rows(src)
    moves, steps = [], []
    for policy in C.POLICIES:
        for scene in sorted(int(p.stem) for p in (C.pass_dir(policy) / 'scenes').glob('[0-9]*.json')):
            rec, _ = C.record(policy, scene)
            for c in rec['cells']:
                m, st = analyse_move(policy, scene, c['arm'], audit)
                if m: moves.append(m); steps += st
    models, pred = first_step_models(moves, steps)
    S = summary(moves, steps, models); S['inputs'] = {'audit_csv': C.rel(src), 'audit_csv_sha256': C.sha256(src)}
    S['controller_semantics'] = {'source': 'pinned ManiSkill a4a4f92 (mani_skill/agents/robots/panda/panda.py, agents/controllers/pd_ee_pose.py, pd_joint_pos.py, envs/sapien_env.py)',
                                 'control_mode': 'pd_ee_delta_pos (configs/stackcube.yaml)', 'use_target': False, 'interpolate': False, 'pos_bounds_m': [-0.1, 0.1], 'stiffness': 1e3, 'damping': 1e2, 'force_limit': 100,
                                 'sim_steps_per_control': 5, 'target': 'measured EE pose at set_action + delta (frame root_translation), IK -> joint targets held for the control step',
                                 'runtime_check': 'recovery_v1_oracle.arm_scale raises unless frame root_translation, normalize_action True, use_target False; every C1 move passed it',
                                 'not_logged': 'IK target joint positions, drive targets, IK failures, per-sub-step states'}
    if S['original_check4_V2']['train_seed_1']['FAIL_audit_csv'] != 9 or S['original_check4_V2']['train_seed_2']['FAIL_audit_csv'] != 5 or \
       S['original_check4_V2']['train_seed_1']['moves'] != 41 or S['original_check4_V2']['train_seed_2']['moves'] != 39:
        S['status'] = 'NOT_REPRODUCED'
    else: S['status'] = 'REPRODUCED'
    C.write_csv(out / 'CHECK4_FAILURE_CASES.csv', [m for m in moves if m['arm'] == 'VISION_CENTER_V2' and m['audit_csv_status'] == 'FAIL'])
    C.write_csv(out / 'CHECK4_ALL_MOVES.csv', moves); C.write_csv(out / 'CHECK4_STEPWISE.csv', steps); C.write_csv(out / 'CHECK4_FIRST_STEP_MODEL_PREDICTIONS.csv', pred)
    figures(moves, steps, pred, out / 'figures'); S['seconds'] = round(time.time() - t0, 1); C.jdump(out / 'CHECK4_SUMMARY.json', S)
    print(json.dumps({k: S[k] for k in ('status', 'original_check4_V2', 'V2_failures', 'agreement_audit_csv_vs_recompute_all_arms')}, indent=1, default=str))


if __name__ == '__main__':
    main()
