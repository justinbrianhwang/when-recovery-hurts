"""Episode-content integrity of converted demonstrations (Stage 1D clean rebuild).

One training trajectory must come from ONE source episode. At the pinned ManiSkill commit the recorder keeps its
buffer when a replay fails, so the next saved trajectory can start with the failed episode, followed by a reset entry
(one sampled dummy action) and only then the adopted episode. Array shapes and the final success flag are normal in
such a file, so they are not evidence of integrity.

Signals are kept separate and reported one by one:
  hard (each one is direct evidence of a reset or of foreign content)
    internal_terminal        a terminated flag without success (the recorder marks a reset entry this way)
    truncated_restart        the time-limit flag goes from true back to false (the step counter was reset)
    first_obs_not_source     the first recorded joint positions are not the source episode's first state
    first_state_not_source   the stored first env state is not the source episode's state (see note below)
  soft (screening only: a single threshold is never a confirmed failure)
    kinematic_jump           cube B moves > 2 cm or a joint > 0.3 rad in one control step
    gripper_not_binary       a gripper command other than +-1 (used only when the source expert is +-1)
    length_not_source        length differs from the source episode (expected equal for this conversion)
    cube_path_not_source     cube A is more than 5 cm away from where the source episode has it at the same step
                             (measured on the 545 held trajectories: at most 3.4 cm without a reset, at least 15.7 cm with one)
Information only (never changes the verdict)
    success_flicker          success is true, then false, then true again inside one continuous episode (the task's
                             success test includes a stillness threshold); provenance is judged by the signals above
Verdict: FAIL with any hard signal, QUARANTINE with soft signals only (unresolved, excluded from training), else PASS.
PASS means "passed these declared checks", not a proof that the data are free of defects.

Note: the converted env_states[0] holds the SECOND source state by construction (a known metadata anomaly, not a
content defect). `first_state_not_source` therefore compares against the second source state.
Object poses and simulator states are read for data auditing only; nothing here is a policy or monitor input.
"""
from __future__ import annotations
import numpy as np

JUMP_CUBE_M = 0.02
JUMP_QPOS_RAD = 0.3
TOL = 1e-5
HARD = ('internal_terminal', 'truncated_restart', 'first_obs_not_source', 'first_state_not_source')
CUBE_PATH_M = 0.05
SOFT = ('kinematic_jump', 'gripper_not_binary', 'length_not_source', 'cube_path_not_source')


def content_integrity(traj, source, expert_gripper_binary=True, same_length_expected=True):
    """traj: actions (L,4), terminated/truncated/success (L,), qpos (L+1,9), cubeB (L+1,>=3), state0 {name: vector}.
    Optional: traj cubeA (L+1,>=3) and source cubeA_path (Ls+1,>=3) enable the step-by-step comparison with the source.
    source: first_qpos (9,), state1 {name: vector} (second source state), steps (int). Returns signals and a verdict."""
    a = np.asarray(traj['actions'], float); L = len(a); term = np.asarray(traj['terminated'], bool); trunc = np.asarray(traj['truncated'], bool)
    succ = np.asarray(traj['success'], bool); qpos = np.asarray(traj['qpos'], float); cube = np.asarray(traj['cubeB'], float)[:, :3]; sig = {}
    tail = int(np.flatnonzero(~succ)[-1]) + 1 if (~succ).any() else 0          # first index of the final all-success block
    bad_terminal = np.flatnonzero(term & ~succ).tolist(); flicker = np.flatnonzero(succ[:tail]).tolist()
    if bad_terminal: sig['internal_terminal'] = bad_terminal
    restarts = (np.flatnonzero(np.diff(trunc.astype(int)) < 0) + 1).tolist()
    if restarts: sig['truncated_restart'] = restarts
    q0 = float(np.abs(qpos[0] - np.asarray(source['first_qpos'], float)).max())
    if q0 > TOL: sig['first_obs_not_source'] = q0
    s0 = max(float(np.abs(np.asarray(traj['state0'][n], float) - np.asarray(source['state1'][n], float)).max()) for n in source['state1'])
    if s0 > TOL: sig['first_state_not_source'] = s0
    d_cube = np.abs(np.diff(cube, axis=0)).max(1); d_q = np.abs(np.diff(qpos, axis=0)).max(1)
    jumps = np.flatnonzero((d_cube > JUMP_CUBE_M) | (d_q > JUMP_QPOS_RAD)).tolist()
    if jumps: sig['kinematic_jump'] = jumps
    if expert_gripper_binary:
        odd = np.flatnonzero(np.abs(np.abs(a[:, -1]) - 1) > 1e-6).tolist()
        if odd: sig['gripper_not_binary'] = odd
    if same_length_expected and L != int(source['steps']): sig['length_not_source'] = L - int(source['steps'])
    path = None
    if 'cubeA' in traj and 'cubeA_path' in source:
        ca = np.asarray(traj['cubeA'], float)[:, :3]; sa = np.asarray(source['cubeA_path'], float)[:, :3]; n = min(len(ca), len(sa))
        path = float(np.abs(ca[1:n] - sa[1:n]).max()) if n > 1 else 0.       # index 0 of the converted file holds the second source state
        if path > CUBE_PATH_M: sig['cube_path_not_source'] = path
    hard = [k for k in HARD if k in sig]; soft = [k for k in SOFT if k in sig]
    # Boundary estimate: the step whose action is the reset dummy (the observation after it belongs to the next episode).
    votes = {}
    for k in ('internal_terminal', 'gripper_not_binary', 'kinematic_jump'):
        for v in sig.get(k, []): votes[v] = votes.get(v, 0) + 1
    best = max(votes.values(), default=0); boundary = sorted(v for v, n in votes.items() if n == best)
    return {'verdict': 'FAIL' if hard else 'QUARANTINE' if soft else 'PASS', 'hard_signals': hard, 'soft_signals': soft, 'signals': sig, 'steps': L,
            'source_steps': int(source['steps']), 'boundary_steps': boundary, 'max_cube_step_m': float(d_cube.max()) if L else 0., 'max_joint_step_rad': float(d_q.max()) if L else 0.,
            'first_success_step': int(np.flatnonzero(succ)[0]) + 1 if succ.any() else None,
            'success_flicker_steps': flicker, 'cube_path_vs_source_max_m': path}


def match_source(state0, table, tol=TOL):
    """Source episode ids whose stored second state equals `state0` (content provenance). table: {source_id: {name: vector}}."""
    return [i for i, st in table.items() if all(float(np.abs(np.asarray(state0[n], float) - np.asarray(st[n], float)).max()) <= tol for n in st)]


def window_counts(steps, boundary, obs_horizon=2, pred_horizon=16):
    """Training windows of one trajectory as the upstream dataset builds them (start in [-(obs_horizon-1), steps-obs_horizon)),
    split by how a reset at action index `boundary` touches them. Counts only; no claim about the effect on a policy."""
    starts = range(-(obs_horizon - 1), steps - obs_horizon); k = int(boundary)
    in_prefix = sum(1 for s in starts if s + obs_horizon - 1 <= k)                 # all conditioning observations from the failed prefix
    mixed_obs = sum(1 for s in starts if s <= k < s + obs_horizon - 1)             # observations on both sides of the reset
    with_dummy = sum(1 for s in starts if max(s, 0) <= k <= min(s + pred_horizon - 1, steps - 1))   # target sequence contains the dummy action
    touched = sum(1 for s in starts if s <= k)
    return {'windows': len(starts), 'obs_in_failed_prefix': in_prefix, 'obs_across_reset': mixed_obs, 'target_contains_dummy_action': with_dummy, 'touched_by_prefix_or_reset': touched}
