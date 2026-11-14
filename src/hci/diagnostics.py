"""Diagnostic helpers for Stage 1C (StackCube-v1 only).

Everything here reads simulator ground truth or recorded demonstrations for LOGGING. Nothing in this module may be
used to build a policy or monitor input.
"""
from __future__ import annotations
import numpy as np

NEAR_M = 0.03      # TCP within 3 cm of the cubeA centre
ALIGN_M = 0.02     # cubeA centre within 2 cm (xy) of the cubeB centre (the cube half-width)
LIFT_M = 0.01      # cubeA centre raised 1 cm above its initial height
FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8)
ACT_HORIZON = 8


def even_pick(items, n):
    """n items at equal spacing over an ordered list (first and last included)."""
    items = list(items)
    if n >= len(items): return items
    if n <= 0: raise ValueError('n must be positive')
    idx = [int(round(i * (len(items) - 1) / (n - 1))) for i in range(n)] if n > 1 else [0]
    if len(set(idx)) != n: raise ValueError('Equal spacing produced duplicates')
    return [items[i] for i in idx]


def window_steps(length, act_horizon=ACT_HORIZON, fractions=FRACTIONS):
    """Observation steps t such that actions t..t+act_horizon-1 are all recorded (no padding)."""
    last = length - act_horizon
    if last < 0: return []
    out = []
    for f in fractions:
        t = int(round(f * last))
        if t not in out: out.append(t)
    return out


def gripper_phases(gripper):
    """Phase boundaries from the demonstration's OWN gripper commands: first close and the first re-open after it."""
    g = np.asarray(gripper, float); closed = np.flatnonzero(g < 0)
    if not len(closed): return None, None
    t_close = int(closed[0]); later = np.flatnonzero(g[t_close:] > 0)
    return t_close, (int(t_close + later[0]) if len(later) else None)


def phase_of(t, t_close, t_open):
    if t_close is None or t_open is None: return 'unknown'
    if t + ACT_HORIZON <= t_close: return 'approach'           # the whole executed segment is before the first close
    if t >= t_close and t + ACT_HORIZON <= t_open: return 'transport'
    if t >= t_open: return 'place'
    return 'boundary'                                          # the executed segment crosses a phase change


def first_true(flags):
    idx = np.flatnonzero(np.asarray(flags, bool))
    return int(idx[0]) + 1 if len(idx) else None               # step count after which the flag is first true


def step_state(u, t, action=None):
    """Ground-truth snapshot after control step t (t = 0: right after initialisation). Logging only."""
    from .trees import as_numpy
    e = u.evaluate(); A = as_numpy(u.cubeA.pose.p)[0]; B = as_numpy(u.cubeB.pose.p)[0]; T = as_numpy(u.agent.tcp.pose.p)[0]
    flag = lambda k: bool(as_numpy(e[k]).reshape(-1)[0])
    return {'t': int(t), 'tcp': T.tolist(), 'cubeA': A.tolist(), 'cubeB': B.tolist(),
            'tcp_to_cubeA': float(np.linalg.norm(T - A)), 'cubeA_to_cubeB_xy': float(np.linalg.norm(A[:2] - B[:2])),
            'is_cubeA_grasped': flag('is_cubeA_grasped'), 'is_cubeA_on_cubeB': flag('is_cubeA_on_cubeB'),
            'is_cubeA_static': flag('is_cubeA_static'), 'success': flag('success'),
            'action': None if action is None else [float(x) for x in action]}


def stage_events(initial, rows):
    """First step of each event and the furthest stage reached. Evidence is ranked, so a recorded grasp is never
    overruled by a distance threshold (the Stage 1B trace labelled a grasped-and-lifted episode 'approach')."""
    if not rows: return {'furthest_stage': 'unreadable'}
    z0 = initial['cubeA'][2]; lifted = lambda r: r['cubeA'][2] - z0 >= LIFT_M
    first = lambda cond: next((r['t'] for r in rows if cond(r)), None)
    grasp_rows = [r for r in rows if r['is_cubeA_grasped']]
    ev = {'first_near_step': first(lambda r: r['tcp_to_cubeA'] <= NEAR_M),
          'first_close_command_step': first(lambda r: r['action'] is not None and r['action'][-1] < 0),
          'first_grasp_step': first(lambda r: r['is_cubeA_grasped']),
          'first_lift_step': first(lifted),
          'first_grasped_lift_step': first(lambda r: r['is_cubeA_grasped'] and lifted(r)),
          'first_aligned_step': first(lambda r: lifted(r) and r['cubeA_to_cubeB_xy'] <= ALIGN_M),
          'first_on_cubeB_step': first(lambda r: r['is_cubeA_on_cubeB']),
          'first_success_step': first(lambda r: r['success']),
          'grasp_steps': len(grasp_rows), 'last_grasp_step': grasp_rows[-1]['t'] if grasp_rows else None,
          'close_command_steps': sum(1 for r in rows if r['action'] is not None and r['action'][-1] < 0),
          'min_tcp_to_cubeA_m': min(r['tcp_to_cubeA'] for r in rows),
          'min_cubeA_to_cubeB_xy_m': min(r['cubeA_to_cubeB_xy'] for r in rows),
          'max_cubeA_lift_m': max(r['cubeA'][2] - z0 for r in rows),
          'cubeA_xy_displacement_m': float(np.linalg.norm(np.subtract(rows[-1]['cubeA'][:2], initial['cubeA'][:2]))),
          'cubeB_xy_displacement_m': float(np.linalg.norm(np.subtract(rows[-1]['cubeB'][:2], initial['cubeB'][:2]))),
          'final_is_grasped': bool(rows[-1]['is_cubeA_grasped']), 'final_gripper_command': rows[-1]['action'][-1] if rows[-1]['action'] else None}
    if ev['first_success_step'] is not None: stage = 'success'
    elif ev['first_aligned_step'] is not None or ev['first_on_cubeB_step'] is not None: stage = 'aligned_no_success'
    elif ev['first_grasped_lift_step'] is not None: stage = 'lifted_not_aligned'
    elif ev['first_grasp_step'] is not None: stage = 'grasped_not_lifted'
    elif ev['first_near_step'] is not None: stage = 'near_no_grasp'
    else: stage = 'not_near'
    ev['furthest_stage'] = stage
    return ev


def rgb_diff_stats(a, b):
    """uint8 [0, 255] difference statistics of two HxWx3 images."""
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    return {'mae_u8': float(d.mean()), 'max_u8': int(d.max()), 'frac_ge1': float((d >= 1).mean()),
            'frac_ge8': float((d >= 8).mean()), 'frac_ge32': float((d >= 32).mean())}


def action_errors(pred, expert):
    """Normalised-action errors of an (8, 4) segment. Units are controller inputs in [-1, 1], not metres moved."""
    p = np.asarray(pred, float); e = np.asarray(expert, float); d = p - e; out = {}
    for j, name in enumerate('xyz'):
        out[f'mae_{name}'] = float(np.abs(d[:, j]).mean()); out[f'rmse_{name}'] = float(np.sqrt((d[:, j] ** 2).mean()))
        out[f'signed_{name}'] = float(d[:, j].mean()); out[f'first_err_{name}'] = float(d[0, j])
    out['mae_xyz'] = float(np.abs(d[:, :3]).mean()); out['rmse_xyz'] = float(np.sqrt((d[:, :3] ** 2).mean()))
    out['first_l2_xyz'] = float(np.linalg.norm(d[0, :3]))
    out['pred_saturation_xyz'] = float((np.abs(p[:, :3]) >= 0.999).mean()); out['expert_saturation_xyz'] = float((np.abs(e[:, :3]) >= 0.999).mean())
    out['gripper_mae'] = float(np.abs(d[:, 3]).mean()); out['gripper_first_err'] = float(d[0, 3])
    out['gripper_sign_agree'] = float(((p[:, 3] < 0) == (e[:, 3] < 0)).mean()); out['gripper_first_sign_agree'] = float((p[0, 3] < 0) == (e[0, 3] < 0))
    out['expert_norm_xyz_mean'] = float(np.linalg.norm(e[:, :3], axis=1).mean()); out['pred_norm_xyz_mean'] = float(np.linalg.norm(p[:, :3], axis=1).mean())
    return out
