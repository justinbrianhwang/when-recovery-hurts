"""Stage 2D exploratory control with a matched return step (development only). See amendments/AMENDMENT_008.

Every (scene, checkpoint, block repeat r) block has four continuations from one snapshot:

  continue             C0   keeps the queue, next inference at the natural chunk boundary
  retreat_5cm          R5   the existing Retreat-Retry (Runtime.force(2)), unchanged
  retreat_2cm_matched  R2M  target 2 cm up; after arrival the arrival position is held until R5's return step
  hold_matched         H0M  the TCP position of the snapshot is held until R5's return step

T_ref of a snapshot is the number of control steps the existing R5 used there, read from one reference run of the
control part only (no policy inference). R5, R2M and H0M hand control back to the policy at checkpoint + T_ref with the
same noise key, unless the episode ended earlier. T_ref is a schedule taken from the simulator's response to R5; the two
matched arms are diagnostic variants, not deployable recovery controllers.

Runtime (simulation.py) is not modified: R5 goes through Runtime.force(2); the two matched arms are driven here through
Runtime.advance with the same command limit, arrival tolerance and held gripper command as Runtime.retreat.
Ground-truth object state enters only through `probe` (logging). It is never given to the policy or to a controller.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from .trees import clone, as_numpy
from .io import read_json
from .simulation import retreat_budget
from .pilot import (runtime_tree, compare_tree, obs_diff, live_obs_ok, jsonable, tree_hash, ineligibility, rows_array,
                    _same_run, _ref, _diff, _trace, _compare_traces, NUMERIC, TRACE_COLUMNS as BASE_COLUMNS,
                    collect_scenes, select_scenes, scale_decision)
from .pilot2b import Hooks, inputs_without_side_effect, pair_table, cls

ROLE = 'stage2d_exploratory_dev'
ARMS = ('continue', 'retreat_5cm', 'retreat_2cm_matched', 'hold_matched')
SHORT = {'continue': 'C0', 'retreat_5cm': 'R5', 'retreat_2cm_matched': 'R2M', 'hold_matched': 'H0M'}
CONTROL = ARMS[1:]
DISTANCE_M = {'retreat_2cm_matched': 0.02, 'hold_matched': 0.0}         # fixed; the R5 distance stays in the config
INTERVENTION = {'continue': None, 'retreat_5cm': 2, 'retreat_2cm_matched': 'matched_2cm', 'hold_matched': 'matched_hold'}
PRIMARY = (('R2M', 'R5'), ('H0M', 'R5'), ('R2M', 'H0M'))                   # (compared arm, reference arm)
SECONDARY = (('R5', 'C0'), ('R2M', 'C0'), ('H0M', 'C0'))
PAIRS = PRIMARY + SECONDARY
PHASES = ('continue', 'intervention_move', 'intervention_hold', 'policy_after_return')
TRACE_COLUMNS = BASE_COLUMNS + ('gripper_qpos_1', 'gripper_qpos_2', 'phase')
# thresholds of the existing diagnostic labels (diagnostics.py, Stage 2C); copied, not tuned
NEAR_M, ALIGN_M, LIFT_M, B_MOVED_M = 0.03, 0.02, 0.01, 0.002


# ---------------------------------------------------------------- the two matched control arms
def matched_control(r, arm, t_ref, on_phase=None):
    """Feedback control towards a fixed TCP target for at most t_ref control steps, then the queue is cleared.

    R2M: move towards start + 2 cm (z); at the first step where the measured TCP is within the arrival tolerance, that
    measured position becomes the hold target for the remaining steps. H0M: the start position is the hold target from
    the first step. Physics runs normally: no state is written, nothing is frozen. Every step goes through
    Runtime.advance (real observation history, horizon charged). No policy inference happens here.
    """
    if arm not in DISTANCE_M: raise ValueError(f'{arm} is not a matched control arm')
    if r.intervention is not None: raise ValueError('Invalid/repeated intervention')
    if not r.eligible()[2]: raise ValueError('Ineligible: the matched arms use the eligibility of the existing Retreat, unchanged')
    cfg = r.c['retreat']; t_ref = int(t_ref)
    if not 0 <= t_ref <= retreat_budget(cfg): raise ValueError('T_ref outside the Retreat step budget')
    tol = float(cfg['arrival_tolerance_m']); step = float(cfg['max_translation_per_step_m']); dist = float(DISTANCE_M[arm])
    k = r.env.agent.controller.controllers['arm'].config
    if not k.frame.startswith('root_translation') or not k.normalize_action or k.use_target:
        raise ValueError('Unsupported controller frame/normalization/target semantics')
    if not np.allclose(np.abs(as_numpy(r.env.agent.robot.pose.q).reshape(-1)), [1, 0, 0, 0], atol=1e-5):
        raise ValueError('This adapter requires an unrotated Panda base')
    scale = np.broadcast_to(np.asarray(k.pos_upper, float), (3,))
    if not np.allclose(np.broadcast_to(np.asarray(k.pos_lower, float), (3,)), -scale): raise ValueError('Symmetric normalized translation limits required')
    r.intervention = INTERVENTION[arm]; r.intervention_t = r.t; r.policy.clear()
    start = r.tcp_history[-1].copy(); target = start + np.array([0., 0., dist])
    grip = float(r.policy.last_action[-1]) if r.policy.last_action is not None else 0.
    anchor = start.copy() if dist == 0 else None; arrival = 0 if dist == 0 else None
    used = moved = held = saturated = 0; drift = 0.; at_arrival = 0. if dist == 0 else None
    while used < t_ref and not r.done:
        cur = r.tcp_history[-1]
        if anchor is None and float(np.linalg.norm(target - cur)) <= tol:
            anchor = cur.copy(); arrival = used; at_arrival = float(np.linalg.norm(cur - start))
        goal = target if anchor is None else anchor; hold = anchor is not None
        if on_phase: on_phase('intervention_hold' if hold else 'intervention_move')
        delta = goal - cur; norm = float(np.linalg.norm(delta))
        if norm > step: saturated += 1
        delta = delta * min(1., step / max(norm, 1e-12))
        a = np.zeros(r.adim, np.float32); a[:3] = np.clip(delta / scale, -1, 1); a[-1] = grip
        r.advance(a); used += 1
        if hold: held += 1; drift = max(drift, float(np.linalg.norm(r.tcp_history[-1] - anchor)))
        else: moved += 1
    final = r.tcp_history[-1].copy()
    if anchor is None and float(np.linalg.norm(target - final)) <= tol:      # arrived with the last allowed step: nothing left to hold
        anchor = final.copy(); arrival = used; at_arrival = float(np.linalg.norm(final - start))
    r.retreat_info = {'mode': arm, 'start_tcp': start.tolist(), 'target_tcp': target.tolist(), 'final_tcp': final.tolist(),
                      'anchor_tcp': None if anchor is None else anchor.tolist(), 'commanded_distance_m': dist,
                      'achieved_displacement_m': float(np.linalg.norm(final - start)), 'achieved_dz_m': float(final[2] - start[2]),
                      'displacement_at_arrival_m': at_arrival, 'target_error_m': float(np.linalg.norm(target - final)),
                      't_ref': t_ref, 'steps_used': used, 'move_steps': moved, 'hold_steps': held, 'first_arrival_step': arrival,
                      'hold_drift_max_m': drift if held else None, 'command_saturated_steps': saturated, 'gripper_command': grip,
                      'arrival_tolerance_m': tol, 'max_translation_per_step_m': step, 'reached': anchor is not None,
                      'end_reason': 'episode_ended' if r.done else 'matched_return'}
    r.policy.clear()


def apply_arm(r, arm, rep, t_ref, on_phase=None):
    """Called only after the restored runtime was compared with the snapshot. All arms use the block's own noise key."""
    r.repeat = int(rep)
    if arm == 'continue': return
    if arm == 'retreat_5cm':
        if on_phase: on_phase('intervention_move')
        r.force(2)                                   # the existing Retreat-Retry, exactly as in Stage 2A/2B
    else: matched_control(r, arm, t_ref, on_phase)
    if on_phase: on_phase('policy_after_return')


def control_record(arm, info):
    """One layout for the three control arms. R5 has no hold part by definition of the existing Retreat."""
    if arm == 'continue' or info is None: return None
    r5 = arm == 'retreat_5cm'; start = np.asarray(info['start_tcp'], float); final = np.asarray(info['final_tcp'], float)
    return {'commanded_distance_m': info['commanded_distance_m'], 'achieved_displacement_m': info['achieved_displacement_m'],
            'achieved_dz_m': float(final[2] - start[2]), 'target_error_m': info['target_error_m'], 'reached': bool(info['reached']),
            'end_reason': info['end_reason'], 'steps_used': info['steps_used'],
            'first_arrival_step': (info['steps_used'] if info['reached'] else None) if r5 else info['first_arrival_step'],
            'move_steps': info['steps_used'] if r5 else info['move_steps'], 'hold_steps': 0 if r5 else info['hold_steps'],
            'hold_anchor_tcp': None if r5 else info['anchor_tcp'], 'hold_drift_max_m': None if r5 else info['hold_drift_max_m'],
            'command_saturated_steps': info['command_saturated_steps'], 'gripper_command': info['gripper_command'],
            'start_tcp': info['start_tcp'], 'target_tcp': info['target_tcp'], 'final_tcp': info['final_tcp']}


def prefix_equal(a, b, tol):
    """Two recorded control parts: same number of steps, commands, simulator states, observations from the environment."""
    n = len(a) == len(b); act = max([_diff(x['action'], y['action']) for x, y in zip(a, b)] + [0.])
    state = max([_diff(x['state'], y['state']) for x, y in zip(a, b)] + [0.]); obs = [obs_diff(x['raw'], y['raw']) for x, y in zip(a, b)]
    o_s = max([o['state_max_abs'] for o in obs] + [0.]); o_r = max([o['rgb_max_abs'] for o in obs] + [0.])
    ok = bool(n and act <= tol['action_atol'] and state <= tol['state_atol'] and o_s <= tol['state_atol'] and o_r <= tol['observation_atol'])
    return {'pass': ok, 'steps': [len(a), len(b)], 'action_max_abs': act, 'state_max_abs': state, 'live_observation_state_max_abs': o_s,
            'live_observation_rgb_max_abs': o_r}


def handoff(cells, t, t_ref, ref_early):
    """R5, R2M and H0M: no inference during the control part; where the policy resumed, its first inference is at
    checkpoint + T_ref with one and the same seed. An arm whose episode ended during control has no handoff (N/A)."""
    expect = t + t_ref; rows = {}; ok = True; seeds = set()
    for arm in CONTROL:
        x = cells[arm]; first = x['inference_calls'][0] if x['inference_calls'] else None
        rows[SHORT[arm]] = {'policy_resumed': x['policy_resumed'], 'early_absorbing': x['early_absorbing'], 'control_steps': x['control_steps'],
                            'control_inference_calls': x['control_inference_calls'], 'first_policy_step': x['first_policy_step'],
                            'first_policy_seed': None if first is None else first['seed'], 'queue_len_at_return': x['queue_len_at_return'],
                            'history_len_at_return': x['history_len_at_return']}
        ok &= x['control_inference_calls'] == 0 and x['queue_len_at_return'] == 0 and x['policy_resumed'] != x['early_absorbing']
        if x['policy_resumed']:
            ok &= first is not None and first['t'] == expect == x['first_policy_step'] and x['control_steps'] == t_ref; seeds.add(None if first is None else first['seed'])
        else: ok &= first is None and x['first_policy_step'] is None and x['control_steps'] <= t_ref
    if ref_early: ok &= not cells['retreat_5cm']['policy_resumed']
    ok &= len(seeds) <= 1; n = sum(x['policy_resumed'] for x in rows.values())
    return {'pass': bool(ok), 'expected_first_policy_step': expect, 't_ref': t_ref, 'reference_early_absorbing': bool(ref_early), 'arms_resumed': n,
            'matched_handoff': 'NA' if ref_early else bool(ok and n == len(CONTROL)), 'same_first_seed': len(seeds) <= 1, **rows}


# ---------------------------------------------------------------- restore audit
def audit_scene(r, seed, c, checkpoints):
    """The Stage 2A audit (Continue trace, Retreat and Refresh branches, each followed by restore and the same trace),
    plus: the R5 control part alone as the reference, the three control arms run to the end, R5 against the reference,
    the handoff step/seed/queue/history of the three arms, and an exact check that saving the policy inputs changes nothing."""
    tol = c['audit']; n = tol['replay_steps']; most = retreat_budget(c['retreat']); r.reset(seed); out = []
    with Hooks(r) as h:
        for t in checkpoints:
            while not r.done and r.t < t: r.step_base()
            if r.done:
                out.append({'checkpoint': t, 'reached': False, 'reason': f'trunk ended at step {r.t} ({r.reason})'}); continue
            elig = [bool(x) for x in r.eligible()]; s = r.snapshot(); _, f0, side = inputs_without_side_effect(r); first = _trace(r, n); checks = {}

            def back(label):
                r.restore(s); live = obs_diff(r.replay_raw, s['raw'])
                checks[label + '_restored'] = {**compare_tree(f0, runtime_tree(r), tol), 'prefix_replay_state_error': r.replay_error,
                                               'live_observation': live, 'live_observation_pass': live_obs_ok(live, tol)}
                checks[label + '_trace'] = _compare_traces(first, _trace(r, n), tol)
            back('repeat'); branches = {}; ho = None
            r.restore(s)
            if bool(r.eligible()[1]) != elig[1]: branches['refresh'] = {'executed': False, 'reason': 'eligibility after restore differs from the snapshot', 'pass': False}; r.complete()
            elif elig[1]: r.force(1); res = r.complete(); branches['refresh'] = {'executed': True, 'success': res['success'], 'steps': res['steps']}
            else: branches['refresh'] = {'executed': False, 'reason': 'not eligible at this checkpoint'}; r.complete()
            back('after_refresh'); r.restore(s)
            if bool(r.eligible()[2]) != elig[2]:
                for arm in CONTROL: branches[arm] = {'executed': False, 'reason': 'eligibility after restore differs from the snapshot', 'pass': False}
            elif not elig[2]:
                for arm in CONTROL: branches[arm] = {'executed': False, 'reason': 'not eligible at this checkpoint'}
            else:
                h.calls = calls = []; h.prefix = ref = []; h.prefix_n = most; r.force(2); h.prefix = None
                T = r.retreat_info['steps_used']; ref_early = bool(r.done)
                branches['reference_r5_control'] = {'executed': True, 't_ref': T, 'end_reason': r.retreat_info['end_reason'], 'inference_calls': len(calls),
                                                    'reference_early_absorbing': ref_early, 'pass': not calls}
                back('after_reference_r5_control'); cells = {}
                for arm in CONTROL:
                    r.restore(s); h.calls = calls = []; h.prefix = pre = []; h.prefix_n = most; apply_arm(r, arm, s['repeat'], T); h.prefix = None
                    k = r.t - t; cc = len(calls); q = len(r.policy.queue); hl = len(r.policy.history); early = bool(r.done); res = r.complete()
                    cells[arm] = {'policy_resumed': not early, 'early_absorbing': early, 'control_steps': k, 'control_inference_calls': cc, 'queue_len_at_return': q,
                                  'history_len_at_return': hl, 'first_policy_step': calls[0]['t'] if calls else None, 'inference_calls': calls}
                    b = {'executed': True, 'success': res['success'], 'steps': res['steps'], 'control': control_record(arm, res['retreat']), 'pass': True}
                    if arm == 'retreat_5cm': b['same_as_reference'] = prefix_equal(ref, pre, tol); b['pass'] = b['same_as_reference']['pass']
                    branches[arm] = b; back('after_' + arm)
                ho = handoff(cells, t, T, ref_early)
            r.restore(s)
            ok = (all(v['pass'] for v in checks.values()) and all(v.get('live_observation_pass', True) for v in checks.values())
                  and all(v.get('pass', True) for v in branches.values()) and side['pass'] and (ho is None or ho['pass']))
            out.append({'checkpoint': t, 'reached': True, 'eligible': elig, 'trace_steps': len(first), 'branches': branches, 'handoff': ho, 'checks': checks,
                        'input_saving_changes_nothing': side, 'pass': ok})
    reached = [x for x in out if x['reached']]
    return jsonable({'seed': int(seed), 'checkpoints': out, 'reached': len(reached), 'pass': bool(reached) and all(x['pass'] for x in reached),
                     **({} if reached else {'reason': 'no checkpoint was reached'})})


def audit_report(scenes):
    names = ('refresh', 'reference_r5_control') + CONTROL
    n = {a: sum(1 for s in scenes for x in s['checkpoints'] if x['reached'] and x['branches'].get(a, {}).get('executed')) for a in names}
    return {'pass': bool(scenes) and all(s['pass'] for s in scenes) and min(n.values()) > 0, 'branches_executed': n, 'results': scenes,
            'compared': {'numeric': dict(NUMERIC), 'exact': ['t', 'done', 'y', 'reason', 'repeat', 'intervention', 'intervention_t', 'queue_len',
                                                              'history_len', 'n_actions', 'inference_calls', 'retreat_info', 'rng'],
                         'added_in_stage2d': ['restore and trace after the R5 control part alone (the T_ref reference)', 'restore and trace after R5, R2M and H0M',
                                              'R5 control part against the reference (commands, states, observations, steps)',
                                              'handoff of R5/R2M/H0M: no inference during control, empty queue, first inference at checkpoint + T_ref with one seed',
                                              'saving the policy inputs leaves the runtime unchanged (exact)'],
                         'not_compared': ['policy.inference_seconds (wall-clock time)']}}


# ---------------------------------------------------------------- four-arm blocks of one scene
def trace_array(rows):
    out = np.zeros((len(rows), len(TRACE_COLUMNS)), np.float32); out[:, :len(BASE_COLUMNS)] = rows_array(rows)
    for i, x in enumerate(rows): out[i, -3:] = [*(x.get('gripper_qpos') or [np.nan] * 2), PHASES.index(x['phase'])]
    return out


def collect_scene(r, seed, c, checkpoints, repeats, probe=None, frame=None, stage=None, video=False, recheck=False, order=ARMS):
    """Trunk (repeat 0, never restored); at every snapshot one reference run of the R5 control part (T_ref), then every
    (checkpoint, repeat) block with the four arms. A snapshot exists only where the three ORIGINAL choices are eligible.
    `order` exists for the order-independence test. Returns (record, traces, videos, inputs)."""
    if sorted(order) != sorted(ARMS): raise ValueError('order must be a permutation of the four arms')
    tol = c['audit']; most = retreat_budget(c['retreat']); started = time.perf_counter()
    checkpoints = [int(t) for t in checkpoints]; repeats = [int(x) for x in repeats]; ctx = {'phase': 'continue'}
    phase = lambda p: ctx.__setitem__('phase', p); tagged = (lambda rt, z: {**probe(rt, z), 'phase': ctx['phase']}) if probe else None
    r.reset(seed); initial = probe(r, None) if probe else None; snaps = {}; skips = []; traces = {}; videos = {}; inputs = {}
    stage_of = (lambda rows: stage(initial, rows)) if (stage and probe) else (lambda rows: None)
    with Hooks(r, tagged, frame) as h:
        h.rows = trunk_rows = []; h.frames = trunk_frames = [frame(r)] if (video and frame) else None; h.calls = trunk_calls = []
        while not r.done:
            if r.t in checkpoints:
                if r.eligible().all():
                    x, tree, side = inputs_without_side_effect(r)
                    if not side['pass']: raise RuntimeError(f'scene {seed} checkpoint {r.t}: saving the policy inputs changed the runtime ({side["failed"]})')
                    state_sha = tree_hash(r.env.get_state_dict()); inputs[r.t] = x; last = r.policy.last_action
                    snaps[r.t] = {'s': r.snapshot(), 'tree': tree,
                                  'info': {'checkpoint': r.t, 'snapshot_id': f'{int(seed)}:{r.t}:{state_sha[:16]}', 'queue_len': len(r.policy.queue),
                                           'history_len': len(r.policy.history), 'remaining_budget': r.horizon - r.t, 'eligible': [True, True, True],
                                           'policy_calls_so_far': r.policy.inference_calls, 'state_sha256': state_sha, 'policy_inputs_sha256': tree_hash(x),
                                           'tcp': np.asarray(r.tcp_history[-1], float).tolist(), 'qpos': as_numpy(r.raw['agent']['qpos']).reshape(-1).tolist(),
                                           'qvel': as_numpy(r.raw['agent']['qvel']).reshape(-1).tolist(),
                                           'last_gripper_command': None if last is None else float(last[-1]),
                                           'state_at_checkpoint': trunk_rows[-1] if trunk_rows else initial,
                                           'stage_so_far': (stage_of(list(trunk_rows)) or {}).get('furthest_stage')}}
                else:
                    skips.append({'checkpoint': r.t, 'reason': 'not_all_three_choices_eligible', 'detail': ineligibility(r),
                                  'eligible': [bool(x) for x in r.eligible()]})
            r.step_base()
        trunk = {'policy_noise_repeat_id': 0, 'success': r.y, 'steps': r.t, 'reason': r.reason, 'policy_calls': r.policy.inference_calls,
                 'policy_inference_seconds': r.policy.inference_seconds, 'stage': stage_of(trunk_rows), 'inference_calls': trunk_calls}
        trunk_ref = _ref(r)
        for t in checkpoints:
            if t not in snaps and not any(x['checkpoint'] == t for x in skips):
                skips.append({'checkpoint': t, 'reason': 'trunk_succeeded_before_checkpoint' if r.y else 'trunk_ended_before_checkpoint',
                              'detail': [f'trunk ended at step {r.t} ({r.reason})'], 'eligible': None})
        if probe: traces['trunk'] = trace_array(trunk_rows)
        blocks = []; duplicates = []; rechecks = []; first_snapshot = min(snaps) if snaps else None
        for t in sorted(snaps):
            s = snaps[t]['s']; info = snaps[t]['info']; refs = {}

            def start():
                r.restore(s); err = r.replay_error; cmp = compare_tree(snaps[t]['tree'], runtime_tree(r), tol); live = obs_diff(r.replay_raw, s['raw'])
                if not cmp['pass'] or not live_obs_ok(live, tol):
                    raise RuntimeError(f'scene {seed} checkpoint {t}: the restored runtime differs from the snapshot ({cmp["failed"]}, live observation {live})')
                return err, live
            # reference: the control part of the existing R5 alone, once per snapshot, no policy inference; it only fixes T_ref
            tick = time.perf_counter(); start(); phase('intervention_move')
            h.rows = ref_rows = []; h.frames = None; h.calls = ref_calls = []; h.prefix = ref_prefix = []; h.prefix_n = most
            r.force(2); h.prefix = None; phase('continue')
            if ref_calls: raise RuntimeError(f'scene {seed} checkpoint {t}: the reference control part called the policy')
            ri = clone(r.retreat_info); T = int(ri['steps_used']); ref_early = bool(r.done); ref_end = (r.y, r.t, r.reason)
            start()                                               # back at the snapshot: full state and observation verified, or the scene stops
            info['reference'] = {'t_ref': T, 'end_reason': ri['end_reason'], 'reached': bool(ri['reached']), 'reference_early_absorbing': ref_early,
                                 'ended_as': {'y': ref_end[0], 'step': ref_end[1], 'reason': ref_end[2]} if ref_early else None,
                                 'achieved_displacement_m': ri['achieved_displacement_m'], 'target_error_m': ri['target_error_m'],
                                 'commanded_distance_m': ri['commanded_distance_m'], 'command_saturated_steps': ri['command_saturated_steps'],
                                 'gripper_command': ri['gripper_command'], 'policy_inference_calls': 0, 'restored_and_verified_after_reference': True,
                                 'commands_sha256': tree_hash([x['action'] for x in ref_prefix]), 'expected_return_step': t + T,
                                 'wall_seconds': time.perf_counter() - tick,
                                 'note': 'preparation run, not a continuation; not counted in any outcome'}
            if probe: traces[f'ref_{t}'] = trace_array(ref_rows)

            def branch(rep, arm, keep_frames=False):
                tick = time.perf_counter(); err, live = start(); calls0 = r.policy.inference_calls; sec0 = r.policy.inference_seconds
                h.rows = rows = []; h.frames = frames = [] if keep_frames else None; h.calls = calls = []
                h.prefix = prefix = [] if arm != 'continue' else None; h.prefix_n = most
                phase('continue'); apply_arm(r, arm, rep, T, phase); h.prefix = None
                ctl = arm != 'continue'; k = r.t - t; cc = len(calls); q = len(r.policy.queue); early = ctl and bool(r.done)
                hist = tree_hash(list(r.policy.history)) if ctl else None; hl = len(r.policy.history) if ctl else None
                res = r.complete(); phase('continue'); same = None
                if arm == 'retreat_5cm':
                    same = prefix_equal(ref_prefix, prefix, tol); same['retreat_record_equal'] = _diff(jsonable(clone(res['retreat'])), jsonable(ri)) <= tol['state_atol']      # counters and labels exact, lengths within the audit tolerance
                    if not (same['pass'] and same['retreat_record_equal'] and k == T):
                        raise RuntimeError(f'scene {seed} checkpoint {t} repeat {rep}: the R5 control part differs from its reference ({jsonable(same)}); '
                                           'this is a technical fault, the result is not adjusted')
                if ctl and cc: raise RuntimeError(f'scene {seed} checkpoint {t} repeat {rep} {arm}: the policy was called during the control part')
                first = calls[0]['t'] if calls else None
                cell = {'experimental_arm': arm, 'arm': SHORT[arm], 'block_repeat_id': rep, 'policy_noise_repeat_id': rep,
                        'y': int(res['success']), 'steps': res['steps'], 'reason': res['reason'], 'success_step': res['steps'] if res['success'] else None,
                        'intervention': res['intervention'], 'intervention_step': res['intervention_step'], 'queue_len_after_choice': q,
                        't_ref': T if ctl else None, 'control': control_record(arm, res['retreat']), 'control_steps': k if ctl else 0,
                        'control_inference_calls': cc if ctl else 0, 'early_absorbing': bool(early), 'policy_resumed': bool(ctl and not early),
                        'first_policy_step': (first if not early else None) if ctl else first, 'queue_len_at_return': q if ctl else None,
                        'history_len_at_return': hl, 'policy_resume_history_sha256': hist,
                        'success_phase': None if not res['success'] else ('continue' if not ctl else ('control' if early else 'after_return')),
                        'steps_after_checkpoint': res['steps'] - t, 'post_return_steps': res['steps'] - t - k if ctl else None,
                        'policy_calls_after_snapshot': res['policy_calls'] - calls0, 'policy_seconds_after_snapshot': res['policy_inference_seconds'] - sec0,
                        'first_inference_step': first, 'inference_calls': calls, 'same_as_reference': same,
                        'prefix_replay_state_error': err, 'live_observation': live, 'stage': stage_of(list(trunk_rows[:t]) + rows),
                        'wall_seconds': time.perf_counter() - tick}
                return cell, rows, frames

            for rep in repeats:
                cells = {}
                for arm in order:
                    want = bool(video and frame) and t == first_snapshot and rep == 0
                    cell, rows, frames = branch(rep, arm, want); cells[arm] = cell
                    if probe: traces[f'b_{t}_{rep}_{SHORT[arm]}'] = trace_array(rows)
                    if want: videos[f'scene{seed}_t{t}_r{rep}_{arm}'] = list(trunk_frames[:t + 1]) + frames
                    if arm in DISTANCE_M and rep == 0: refs[arm] = (_ref(r), cell['inference_calls'], cell['control'])
                    if arm == 'continue' and rep == 0: duplicates.append({'checkpoint': t, 'repeat': 0, 'against': 'trunk', **_same_run(trunk_ref, r, tol)})
                ho = handoff(cells, t, T, ref_early)
                if not ho['pass']:
                    raise RuntimeError(f'scene {seed} checkpoint {t} repeat {rep}: the control arms did not hand over as declared ({jsonable(ho)})')
                blocks.append({'scene': int(seed), 'checkpoint': t, 'repeat': rep, 'block_repeat_id': rep, 'role': ROLE, 'snapshot_id': info['snapshot_id'],
                               't_ref': T, 'reference_early_absorbing': ref_early, **{'y_' + SHORT[a]: cells[a]['y'] for a in ARMS},
                               'cells': [cells[a] for a in ARMS], 'handoff': ho})
            if recheck:      # after every other branch of this snapshot: R2M and H0M of repeat 0 once more
                for arm in DISTANCE_M:
                    if arm not in refs: continue
                    cell, _, _ = branch(0, arm); same = _same_run(refs[arm][0], r, tol)
                    eq = cell['inference_calls'] == refs[arm][1] and _diff(jsonable(cell['control']), jsonable(refs[arm][2])) <= tol['state_atol']
                    rechecks.append({'checkpoint': t, 'repeat': 0, 'experimental_arm': arm, 'against': 'first run of the same arm in the same block',
                                     **same, 'inference_and_control_record_equal': bool(eq), 'match': bool(same['match'] and eq)})
    ok = all(x['match'] for x in duplicates + rechecks)
    record = {'scene': int(seed), 'role': ROLE, 'planned_checkpoints': checkpoints, 'repeats': repeats, 'arms': list(ARMS), 'initial_state': initial, 'trunk': trunk,
              'snapshots': [snaps[t]['info'] for t in sorted(snaps)], 'skips': sorted(skips, key=lambda x: x['checkpoint']), 'blocks': blocks,
              'no_intervention_duplicates': duplicates, 'rechecks': rechecks, 'reproducibility_ok': ok,
              'counts': {'snapshots': len(snaps), 'blocks': len(blocks), 'continuations': len(ARMS) * len(blocks), 'reference_control_runs': len(snaps),
                         'extra_reproducibility_continuations': len(rechecks)},
              'wall_seconds': time.perf_counter() - started, 'completed': True}
    record = jsonable(record); validate_scene(record, c)
    return record, traces, videos, inputs


def validate_scene(rec, c=None):
    """Accounting of one scene record: nothing missing, nothing twice, every control arm applied and handed over as declared."""
    cps = rec['planned_checkpoints']; reps = rec['repeats']; snap = {x['checkpoint']: x for x in rec['snapshots']}; skip = [x['checkpoint'] for x in rec['skips']]
    if len(skip) != len(set(skip)) or set(snap) & set(skip) or set(snap) | set(skip) != set(cps):
        raise ValueError(f'scene {rec["scene"]}: planned checkpoints are not each a snapshot or a listed skip')
    keys = [(b['checkpoint'], b['repeat']) for b in rec['blocks']]
    if len(keys) != len(set(keys)) or set(keys) != {(t, x) for t in snap for x in reps}:
        raise ValueError(f'scene {rec["scene"]}: blocks do not match snapshots x block repeats')
    for s in snap.values():
        ref = s['reference']
        if ref['policy_inference_calls'] != 0 or not ref['restored_and_verified_after_reference']: raise ValueError('the T_ref reference is not a verified control-only run')
        if c is not None and not 0 <= ref['t_ref'] <= c['retreat']['max_steps']: raise ValueError('T_ref outside the Retreat step budget')
    for b in rec['blocks']:
        cells = b['cells']; t = b['checkpoint']; rep = b['repeat']; q = snap[t]['queue_len']; T = snap[t]['reference']['t_ref']
        if b['scene'] != rec['scene'] or b['role'] != rec['role'] or b['block_repeat_id'] != rep: raise ValueError('block scene/role/repeat differs from its record')
        if b['t_ref'] != T or b['reference_early_absorbing'] != snap[t]['reference']['reference_early_absorbing']: raise ValueError('block T_ref differs from the reference of its snapshot')
        if [x['experimental_arm'] for x in cells] != list(ARMS): raise ValueError('a block needs exactly the four arms, once each')
        for x in cells:
            arm = x['experimental_arm']
            if x['block_repeat_id'] != rep or x['policy_noise_repeat_id'] != rep: raise ValueError('an arm left the noise key of its block')
            if any(k['policy_noise_repeat_id'] != rep for k in x['inference_calls']): raise ValueError('an inference call used another noise key than its block')
            if len(x['inference_calls']) != x['policy_calls_after_snapshot']: raise ValueError('inference log does not match the call counter')
            if b['y_' + SHORT[arm]] != x['y']: raise ValueError('joined outcome differs from the cell')
            if c is not None and x['steps'] > c['env']['horizon']: raise ValueError('a branch exceeded the episode horizon')
            if x['intervention'] != INTERVENTION[arm]: raise ValueError('intervention label differs from the arm')
        x = cells[0]
        if x['intervention_step'] is not None or x['control'] is not None or x['queue_len_after_choice'] != q or x['t_ref'] is not None:
            raise ValueError('Continue changed the queue or was recorded as a control arm')
        if x['first_inference_step'] not in (None, t + q): raise ValueError('Continue inferred before the natural chunk boundary')
        for x in cells[1:]:
            arm = x['experimental_arm']; ctl = x['control']
            if x['intervention_step'] != t or x['queue_len_after_choice'] != 0 or ctl is None or x['t_ref'] != T:
                raise ValueError('a control arm was not applied at the checkpoint with a cleared queue and the snapshot T_ref')
            if x['control_inference_calls'] != 0 or x['control_steps'] != ctl['steps_used'] or x['control_steps'] > T:
                raise ValueError('a control part called the policy or ran longer than T_ref')
            if ctl['gripper_command'] != snap[t]['last_gripper_command']: raise ValueError('the gripper command of the snapshot was not held')
            if arm in DISTANCE_M and ctl['commanded_distance_m'] != DISTANCE_M[arm]: raise ValueError('commanded distance differs from the declared one')
            if arm == 'retreat_5cm' and (x['control_steps'] != T or not (x['same_as_reference'] or {}).get('pass') or
                                         (c is not None and abs(ctl['commanded_distance_m'] - c['retreat']['distance_m']) > 1e-9)):
                raise ValueError('R5 differs from its reference or from the configured Retreat')
            if arm == 'hold_matched' and (ctl['move_steps'] != 0 or ctl['hold_steps'] != ctl['steps_used'] or ctl['hold_anchor_tcp'] != ctl['start_tcp']):
                raise ValueError('H0M did not hold the snapshot position for its whole control part')
            if ctl['move_steps'] + ctl['hold_steps'] != ctl['steps_used']: raise ValueError('move and hold steps do not add up')
            if x['policy_resumed'] == x['early_absorbing']: raise ValueError('an arm must either resume the policy or have ended during control')
            if x['policy_resumed']:
                if not (x['first_policy_step'] == x['first_inference_step'] == t + T and x['control_steps'] == T and x['post_return_steps'] >= 1):
                    raise ValueError('the policy did not resume at checkpoint + T_ref')
            elif x['first_policy_step'] is not None or x['inference_calls'] or x['steps'] != t + x['control_steps']:
                raise ValueError('an arm that ended during control has policy steps')
            if x['steps_after_checkpoint'] != x['control_steps'] + x['post_return_steps']: raise ValueError('control steps are not charged to the same horizon')
        if not b['handoff']['pass']: raise ValueError('handoff check failed')
    n = rec['counts']
    if (n['snapshots'], n['blocks'], n['continuations'], n['reference_control_runs']) != (len(snap), len(keys), len(ARMS) * len(keys), len(snap)):
        raise ValueError('counts do not match the record')
    return True


def repro_problems(rec, rerun):
    """Reproducibility is read from the recorded checks themselves (count, identity, match), not from a stored flag."""
    snaps = sorted(x['checkpoint'] for x in rec['snapshots']); out = []
    dup = sorted((d['checkpoint'], d['repeat'], d['against']) for d in rec['no_intervention_duplicates'])
    if dup != [(t, 0, 'trunk') for t in snaps]: out.append('trunk-duplicate checks are not one per snapshot')
    got = sorted((d['checkpoint'], d['repeat'], d['experimental_arm']) for d in rec['rechecks'])
    want = sorted((t, 0, a) for t in snaps for a in DISTANCE_M) if rerun else []
    if got != want: out.append('rerun checks are not the declared ones' if rerun else 'rerun checks recorded for a scene that has none planned')
    bad = sum(1 for d in rec['no_intervention_duplicates'] + rec['rechecks'] if d.get('match') is not True)
    if bad: out.append(f'{bad} reproducibility checks did not match')
    if bool(rec.get('reproducibility_ok')) != (not bad): out.append('stored reproducibility flag contradicts the recorded checks')
    return out


def finalize(out, seeds, c=None, rerun_scenes=()):
    """dataset.json. completed is true only when every scheduled scene has a valid record, no error file exists and
    every declared reproducibility check is present and matched."""
    out = Path(out); recs = []; problems = []; rerun_scenes = {int(s) for s in rerun_scenes}; repro = True
    for seed in seeds:
        p = out / 'scenes' / f'{int(seed)}.json'
        if (out / 'scenes' / f'{int(seed)}.error.json').exists(): problems.append(f'scene {seed}: error file present')
        if not p.exists(): problems.append(f'scene {seed}: no record'); continue
        rec = read_json(p)
        try: validate_scene(rec, c)
        except (ValueError, KeyError, TypeError) as e: problems.append(f'scene {seed}: {e!r}'); continue
        if not rec.get('completed'): problems.append(f'scene {seed}: not completed'); continue
        recs.append(rec); why = repro_problems(rec, int(seed) in rerun_scenes)
        if why: repro = False; problems.append(f'scene {seed}: {"; ".join(why)} (rows are kept as measured, interpretation is withheld)')
    extra = sorted(int(p.name.split('.')[0]) for p in (out / 'scenes').glob('*.json') if int(p.name.split('.')[0]) not in {int(s) for s in seeds})
    if extra: problems.append(f'records of unscheduled scenes: {extra}')
    meta = {'role': ROLE, 'arms': list(ARMS), 'scheduled_scenes': [int(s) for s in seeds], 'recorded_scenes': [x['scene'] for x in recs], 'completed': not problems,
            'problems': problems, 'snapshots': sum(x['counts']['snapshots'] for x in recs), 'blocks': sum(x['counts']['blocks'] for x in recs),
            'continuations': sum(x['counts']['continuations'] for x in recs),
            'reference_control_runs': sum(x['counts']['reference_control_runs'] for x in recs),
            'extra_reproducibility_continuations': sum(x['counts']['extra_reproducibility_continuations'] for x in recs),
            'reproducibility_ok': repro if recs else None, 'wall_seconds_sum_over_scenes': sum(x['wall_seconds'] for x in recs)}
    return meta, recs


# ---------------------------------------------------------------- descriptive tables (no interval, no p-value)
def block_rows(records):
    """One row per block: the four outcomes side by side, every pair classified against its reference arm, the control records."""
    rows = []
    for rec in records:
        snap = {x['checkpoint']: x for x in rec['snapshots']}
        for b in rec['blocks']:
            s = snap[b['checkpoint']]; at = s.get('state_at_checkpoint') or {}; cell = {x['arm']: x for x in b['cells']}
            row = {'scene': b['scene'], 'role': b['role'], 'checkpoint': b['checkpoint'], 'block_repeat_id': b['block_repeat_id'], 'snapshot_id': b['snapshot_id'],
                   't_ref': b['t_ref'], 'reference_early_absorbing': b['reference_early_absorbing'], 'matched_handoff': b['handoff']['matched_handoff'],
                   'arms_resumed': b['handoff']['arms_resumed'],
                   **{f'y_{k}': b[f'y_{k}'] for k in SHORT.values()}, **{f'{a}_vs_{ref}': cls(b[f'y_{ref}'], b[f'y_{a}']) for a, ref in PAIRS},
                   **{f'steps_{k}': cell[k]['steps'] for k in cell}, **{f'reason_{k}': cell[k]['reason'] for k in cell},
                   **{f'success_phase_{k}': cell[k]['success_phase'] for k in cell}, **{f'policy_calls_{k}': cell[k]['policy_calls_after_snapshot'] for k in cell},
                   **{f'first_policy_step_{k}': cell[k]['first_policy_step'] for k in cell},
                   **{f'stage_{k}': (cell[k]['stage'] or {}).get('furthest_stage') for k in cell},
                   'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'], 'queue_len_at_checkpoint': s['queue_len'],
                   'remaining_budget': s['remaining_budget'], 'stage_so_far_at_checkpoint': s.get('stage_so_far'), 'grasped_at_checkpoint': at.get('is_cubeA_grasped')}
            for k in ('R5', 'R2M', 'H0M'):
                ctl = cell[k]['control']
                row.update({f'{n}_{k}': cell[k][n] for n in ('control_steps', 'early_absorbing', 'policy_resumed', 'post_return_steps')})
                row.update({f'{n}_{k}': ctl[n] for n in ('commanded_distance_m', 'achieved_displacement_m', 'achieved_dz_m', 'target_error_m', 'reached', 'end_reason',
                                                         'first_arrival_step', 'move_steps', 'hold_steps', 'hold_drift_max_m')})
            rows.append(row)
    return rows


def pair_rows(rows):
    cps = sorted({x['checkpoint'] for x in rows}); out = []
    for a, ref in PAIRS:
        kind = 'primary' if (a, ref) in PRIMARY else 'secondary'
        out.append({'comparison': kind, 'checkpoint': 'all', **pair_table(rows, a, ref)})
        out += [{'comparison': kind, 'checkpoint': t, **pair_table([x for x in rows if x['checkpoint'] == t], a, ref)} for t in cps]
    return out


def scene_rows(records, rows=None):
    """Scene-equal weighting: within a scene the mean over its reached checkpoints and the block repeats."""
    rows = block_rows(records) if rows is None else rows; out = []
    for rec in records:
        mine = [x for x in rows if x['scene'] == rec['scene']]; n = len(mine)
        row = {'scene': rec['scene'], 'role': rec['role'], 'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'],
               'trunk_stage': (rec['trunk']['stage'] or {}).get('furthest_stage'), 'reached_checkpoints_m': rec['counts']['snapshots'],
               'checkpoints_used': ' '.join(str(x['checkpoint']) for x in rec['snapshots']),
               't_ref_by_checkpoint': ' '.join(str(x['reference']['t_ref']) for x in rec['snapshots']), 'blocks': n, 'reproducibility_ok': rec['reproducibility_ok']}
        for k in SHORT.values(): row[f'mean_y_{k}'] = sum(x[f'y_{k}'] for x in mine) / n if n else 'NA'
        for a, ref in PAIRS:
            row[f'd_{a}_{ref}'] = sum(x[f'y_{a}'] - x[f'y_{ref}'] for x in mine) / n if n else 'NA'
            row[f'n_0to1_{a}_{ref}'] = sum(x[f'{a}_vs_{ref}'] == '0to1' for x in mine); row[f'n_1to0_{a}_{ref}'] = sum(x[f'{a}_vs_{ref}'] == '1to0' for x in mine)
        row['in_conditional_summary'] = bool(n); out.append(row)
    return out


def scene_equal_summary(scenes):
    used = [s for s in scenes if s['in_conditional_summary']]; out = {'scenes_in_summary': len(used), 'scenes_without_a_snapshot': len(scenes) - len(used)}
    for a, ref in PAIRS:
        d = [s[f'd_{a}_{ref}'] for s in used]
        out[f'd_{a}_{ref}'] = ({'mean_over_scenes': float(np.mean(d)), 'median': float(np.median(d)), 'min': float(np.min(d)), 'max': float(np.max(d)),
                               'scenes_positive': sum(x > 0 for x in d), 'scenes_negative': sum(x < 0 for x in d), 'scenes_zero': sum(x == 0 for x in d)}
                              if d else 'NA')
    for k in SHORT.values():
        v = [s[f'mean_y_{k}'] for s in used]; out[f'mean_y_{k}'] = float(np.mean(v)) if v else 'NA'
    return out


def scene_equal_by_checkpoint(rows):
    out = {}
    for t in sorted({x['checkpoint'] for x in rows}):
        at = [x for x in rows if x['checkpoint'] == t]; scenes = sorted({x['scene'] for x in at}); row = {'scenes': len(scenes), 'blocks': len(at)}
        for a, ref in PAIRS:
            d = [np.mean([x[f'y_{a}'] - x[f'y_{ref}'] for x in at if x['scene'] == s]) for s in scenes]
            row[f'd_{a}_{ref}'] = {'mean_over_scenes': float(np.mean(d)), 'scenes_positive': sum(x > 0 for x in d), 'scenes_negative': sum(x < 0 for x in d),
                                   'scenes_zero': sum(x == 0 for x in d)}
        for k in SHORT.values(): row[f'success_blocks_{k}'] = sum(x[f'y_{k}'] for x in at)
        out[f't{t}'] = row
    return out


def reference_rows(records):
    out = []
    for rec in records:
        for s in rec['snapshots']:
            ref = s['reference']; mine = [b for b in rec['blocks'] if b['checkpoint'] == s['checkpoint']]
            same = [(b['cells'][1]['same_as_reference'] or {}) for b in mine]
            out.append({'scene': rec['scene'], 'checkpoint': s['checkpoint'], 'snapshot_id': s['snapshot_id'], 't_ref': ref['t_ref'], 'end_reason': ref['end_reason'],
                        'reached': ref['reached'], 'reference_early_absorbing': ref['reference_early_absorbing'], 'expected_return_step': ref['expected_return_step'],
                        'achieved_displacement_m': ref['achieved_displacement_m'], 'target_error_m': ref['target_error_m'],
                        'command_saturated_steps': ref['command_saturated_steps'], 'gripper_command_held': ref['gripper_command'],
                        'policy_inference_calls': ref['policy_inference_calls'], 'restored_and_verified_after_reference': ref['restored_and_verified_after_reference'],
                        'commands_sha256': ref['commands_sha256'], 'remaining_budget_at_checkpoint': s['remaining_budget'], 'queue_len_at_checkpoint': s['queue_len'],
                        'tcp_x': s['tcp'][0], 'tcp_y': s['tcp'][1], 'tcp_z': s['tcp'][2], 'r5_runs_compared': len(same),
                        'r5_runs_equal_to_reference': sum(1 for x in same if x.get('pass') and x.get('retreat_record_equal')),
                        'r5_max_action_abs': max([x.get('action_max_abs', 0.) for x in same] + [0.]), 'r5_max_state_abs': max([x.get('state_max_abs', 0.) for x in same] + [0.])})
    return out


def handoff_rows(records):
    out = []
    for rec in records:
        for b in rec['blocks']:
            ho = b['handoff']; row = {'scene': b['scene'], 'checkpoint': b['checkpoint'], 'block_repeat_id': b['block_repeat_id'], 't_ref': ho['t_ref'],
                                      'expected_first_policy_step': ho['expected_first_policy_step'], 'reference_early_absorbing': ho['reference_early_absorbing'],
                                      'arms_resumed': ho['arms_resumed'], 'matched_handoff': ho['matched_handoff'], 'same_first_seed': ho['same_first_seed'], 'pass': ho['pass']}
            for k in ('R5', 'R2M', 'H0M'):
                row.update({f'{n}_{k}': ho[k][n] for n in ('policy_resumed', 'early_absorbing', 'control_steps', 'control_inference_calls', 'first_policy_step',
                                                           'first_policy_seed', 'queue_len_at_return', 'history_len_at_return')})
                row[f'policy_resume_history_sha256_16_{k}'] = (b['cells'][ARMS.index({v: a for a, v in SHORT.items()}[k])]['policy_resume_history_sha256'] or '')[:16]
            out.append(row)
    return out


def call_rows(records):
    out = []
    for rec in records:
        out += [{'scene': rec['scene'], 'checkpoint': 'trunk', 'block_repeat_id': 0, 'experimental_arm': 'trunk', **k} for k in rec['trunk']['inference_calls']]
        for b in rec['blocks']:
            for x in b['cells']:
                out += [{'scene': b['scene'], 'checkpoint': b['checkpoint'], 'block_repeat_id': b['block_repeat_id'], 'experimental_arm': x['experimental_arm'], **k}
                        for k in x['inference_calls']]
    return out


def skip_rows(records):
    return [{'scene': rec['scene'], 'checkpoint': x['checkpoint'], 'reason': x['reason'], 'detail': '; '.join(x['detail'] or []),
             'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'], 'cells_not_run': len(ARMS) * len(rec['repeats'])}
            for rec in records for x in rec['skips']]


# ---------------------------------------------------------------- events around the control part (ground truth, logging only)
def derived(a, z0):
    """a: checkpoint row followed by the rows of one branch (2D trace columns)."""
    c = {n: i for i, n in enumerate(TRACE_COLUMNS)}; A = a[:, c['cubeA_x']:c['cubeA_z'] + 1]; B = a[:, c['cubeB_x']:c['cubeB_z'] + 1]
    return {'t': a[:, c['t']], 'grasp': a[:, c['is_cubeA_grasped']] > .5, 'on': a[:, c['is_cubeA_on_cubeB']] > .5, 'success': a[:, c['success']] > .5,
            'grip': a[:, c['a_gripper']], 'lift': A[:, 2] - z0, 'xy': np.linalg.norm(A[:, :2] - B[:, :2], axis=1),
            'b_moved': np.linalg.norm(B - B[0], axis=1), 'finger': a[:, c['gripper_qpos_1']] + a[:, c['gripper_qpos_2']], 'phase': a[:, c['phase']]}


def event_reading(d, k):
    """Row 0 is the checkpoint state, rows 1..k the control part (k = 0 for Continue), later rows the policy.
    Nothing is labelled as a cause. A missing event is recorded as not found, never dropped."""
    n = len(d['t']); g = d['grasp']; first = lambda idx: int(idx[0]) if len(idx) else None

    def at(i, before):
        if i is None: return {'found': False, 'steps_after_return': None, 'cubeA_to_cubeB_xy_m': None, 'cube_lift_m': None, 'within_align_xy': None,
                              'on_cubeB_flag': None, 'finger_qpos_sum': None}
        j = i - 1 if before else i
        return {'found': True, 'steps_after_return': int(i - k), 'cubeA_to_cubeB_xy_m': float(d['xy'][j]), 'cube_lift_m': float(d['lift'][j]),
                'within_align_xy': bool(d['xy'][j] <= ALIGN_M), 'on_cubeB_flag': bool(d['on'][i] or d['on'][i - 1]), 'finger_qpos_sum': float(d['finger'][i])}
    ctl = slice(1, k + 1); out = {'rows': n - 1, 'control_steps': k, 'grasped_at_checkpoint': bool(g[0]), 'lifted_at_checkpoint': bool(d['lift'][0] >= LIFT_M),
                                  'aligned_at_checkpoint': bool(d['xy'][0] <= ALIGN_M), 'on_cubeB_at_checkpoint': bool(d['on'][0])}
    lost = first(np.flatnonzero(~g[ctl])) if (k and g[0]) else None
    out['grasp_flag_kept_through_control'] = 'NA' if not (k and g[0]) else bool(lost is None)
    out['grasp_flag_off_in_control_step'] = None if lost is None else lost + 1
    out['lift_lost_in_control'] = 'NA' if not (k and out['lifted_at_checkpoint']) else bool((d['lift'][ctl] < LIFT_M).any())
    out['alignment_lost_in_control'] = 'NA' if not (k and out['aligned_at_checkpoint']) else bool((d['xy'][ctl] > ALIGN_M).any())
    out['cubeB_moved_in_control'] = 'NA' if not k else bool((d['b_moved'][ctl] > B_MOVED_M).any())
    out['cube_lift_change_in_control_m'] = None if not k or n <= k else float(d['lift'][min(k, n - 1)] - d['lift'][0])
    after = np.arange(k + 1, n); op = first(after[d['grip'][after] > 0]) if len(after) else None
    rel = first(after[g[after - 1] & ~g[after]]) if len(after) else None
    out.update({'open_command_' + a: v for a, v in at(op, True).items()}); out.update({'grasp_release_' + a: v for a, v in at(rel, True).items()})
    out['grasped_at_return'] = bool(g[min(k, n - 1)]); out['ended_during_control'] = bool(n - 1 <= k and k > 0)
    out['success'] = bool(d['success'].any()); out['grasp_flag_on_at_end'] = bool(g[-1])
    return out


def event_summary(rows):
    """Denominator first: ALL blocks whose cube was grasped at the checkpoint, per arm. Then the same for the other blocks."""
    out = []
    for grasped in (True, False):
        for arm in SHORT.values():
            sel = [x for x in rows if x['arm'] == arm and x['grasped_at_checkpoint'] == grasped]; n = len(sel)
            cnt = lambda f: sum(1 for x in sel if f(x))
            out.append({'grasped_at_checkpoint': grasped, 'arm': arm, 'blocks': n, 'scenes': len({x['scene'] for x in sel}), 'success': cnt(lambda x: x['success']),
                        'grasp_flag_kept_through_control': 'NA' if arm == 'C0' or not grasped else cnt(lambda x: x['grasp_flag_kept_through_control'] is True),
                        'cubeB_moved_in_control': 'NA' if arm == 'C0' else cnt(lambda x: x['cubeB_moved_in_control'] is True),
                        'ended_during_control': cnt(lambda x: x['ended_during_control']), 'grasped_at_return': cnt(lambda x: x['grasped_at_return']),
                        'open_command_found': cnt(lambda x: x['open_command_found']), 'open_command_none': cnt(lambda x: not x['open_command_found']),
                        'open_command_within_2cm_xy_of_cubeB': cnt(lambda x: x['open_command_within_align_xy'] is True),
                        'open_command_farther_than_2cm_xy': cnt(lambda x: x['open_command_within_align_xy'] is False),
                        'grasp_release_found': cnt(lambda x: x['grasp_release_found']), 'grasp_release_none': cnt(lambda x: not x['grasp_release_found']),
                        'grasp_release_within_2cm_xy_of_cubeB': cnt(lambda x: x['grasp_release_within_align_xy'] is True),
                        'grasp_release_farther_than_2cm_xy': cnt(lambda x: x['grasp_release_within_align_xy'] is False),
                        'grasp_release_with_on_cubeB_flag': cnt(lambda x: x['grasp_release_on_cubeB_flag'] is True),
                        'median_open_command_xy_m': float(np.median([x['open_command_cubeA_to_cubeB_xy_m'] for x in sel if x['open_command_found']])) if cnt(lambda x: x['open_command_found']) else 'NA',
                        'median_grasp_release_xy_m': float(np.median([x['grasp_release_cubeA_to_cubeB_xy_m'] for x in sel if x['grasp_release_found']])) if cnt(lambda x: x['grasp_release_found']) else 'NA'})
    return out


def summarize(records, planned_scenes=None):
    rows = block_rows(records); scenes = scene_rows(records, rows); skips = skip_rows(records); refs = reference_rows(records)
    stat = lambda v: {'n': len(v), 'mean': float(np.mean(v)), 'min': float(np.min(v)), 'median': float(np.median(v)), 'max': float(np.max(v))} if v else {'n': 0}
    num = lambda k: [x[k] for x in rows if x[k] is not None]; skip_counts = {}; ctl = {}
    for x in skips: skip_counts[x['reason']] = skip_counts.get(x['reason'], 0) + 1
    for k in ('R5', 'R2M', 'H0M'):
        reasons = {}
        for x in rows: reasons[str(x[f'end_reason_{k}'])] = reasons.get(str(x[f'end_reason_{k}']), 0) + 1
        ctl[k] = {'blocks': len(rows), 'target_reached': sum(1 for x in rows if x[f'reached_{k}']), 'target_not_reached': sum(1 for x in rows if not x[f'reached_{k}']),
                  'end_reasons': reasons, 'early_absorbing': sum(1 for x in rows if x[f'early_absorbing_{k}']), 'policy_resumed': sum(1 for x in rows if x[f'policy_resumed_{k}']),
                  'control_steps': stat(num(f'control_steps_{k}')), 'achieved_displacement_m': stat(num(f'achieved_displacement_m_{k}')),
                  'achieved_dz_m': stat(num(f'achieved_dz_m_{k}')), 'first_arrival_step': stat(num(f'first_arrival_step_{k}')),
                  'hold_steps': stat(num(f'hold_steps_{k}')), 'hold_drift_max_m': stat(num(f'hold_drift_max_m_{k}')),
                  'success_during_control': sum(1 for x in rows if x[f'success_phase_{k}'] == 'control'),
                  'success_after_return': sum(1 for x in rows if x[f'success_phase_{k}'] == 'after_return')}
    cost = {k: {'success_blocks': sum(x[f'y_{k}'] for x in rows), 'blocks': len(rows), 'steps_after_checkpoint': stat([x[f'steps_{k}'] - x['checkpoint'] for x in rows]),
                'steps_after_checkpoint_when_successful': stat([x[f'steps_{k}'] - x['checkpoint'] for x in rows if x[f'y_{k}']]),
                'policy_calls_after_checkpoint': stat([x[f'policy_calls_{k}'] for x in rows])} for k in SHORT.values()}
    dup = [d for rec in records for d in rec['no_intervention_duplicates']]; rc = [d for rec in records for d in rec['rechecks']]; ho = handoff_rows(records)
    return {'role': ROLE, 'arms': dict(SHORT), 'scenes_planned': planned_scenes, 'scenes_recorded': len(records),
            'trunk_success_scenes': sum(r['trunk']['success'] for r in records), 'trunk_failure_scenes': sum(1 - r['trunk']['success'] for r in records),
            'scenes_with_a_snapshot': sum(1 for r in records if r['snapshots']), 'snapshots': sum(r['counts']['snapshots'] for r in records),
            'blocks': len(rows), 'continuations': sum(r['counts']['continuations'] for r in records),
            'reference_control_runs': sum(r['counts']['reference_control_runs'] for r in records),
            'extra_reproducibility_continuations': sum(r['counts']['extra_reproducibility_continuations'] for r in records),
            'skipped_checkpoints': skip_counts, 'cells_not_run': sum(x['cells_not_run'] for x in skips),
            'success_blocks_by_arm': {k: sum(x[f'y_{k}'] for x in rows) for k in SHORT.values()},
            'pairs': pair_rows(rows), 'scene_equal_weight': scene_equal_summary(scenes), 'scene_equal_weight_by_checkpoint': scene_equal_by_checkpoint(rows),
            't_ref': {'snapshots': len(refs), 'steps': stat([x['t_ref'] for x in refs]), 'reference_reached_target': sum(1 for x in refs if x['reached']),
                      'reference_early_absorbing': sum(1 for x in refs if x['reference_early_absorbing']),
                      'reference_end_reasons': {k: sum(1 for x in refs if x['end_reason'] == k) for k in sorted({x['end_reason'] for x in refs})},
                      'r5_runs_compared_with_reference': sum(x['r5_runs_compared'] for x in refs), 'r5_runs_equal_to_reference': sum(x['r5_runs_equal_to_reference'] for x in refs)},
            'handoff': {'blocks': len(ho), 'pass': sum(1 for x in ho if x['pass']), 'matched_handoff_true': sum(1 for x in ho if x['matched_handoff'] is True),
                        'matched_handoff_NA_reference_ended_early': sum(1 for x in ho if x['matched_handoff'] == 'NA'),
                        'matched_handoff_false_an_arm_ended_during_control': sum(1 for x in ho if x['matched_handoff'] is False),
                        'control_inference_calls': sum(x[f'control_inference_calls_{k}'] for x in ho for k in ('R5', 'R2M', 'H0M'))},
            'control_part_by_arm': ctl, 'time_and_inference_by_arm': cost,
            'reproducibility': {'no_intervention_duplicates': len(dup), 'duplicate_mismatches': sum(1 for d in dup if not d['match']),
                                'rechecks': len(rc), 'recheck_mismatches': sum(1 for d in rc if not d['match']),
                                'rechecks_by_arm': {a: sum(1 for d in rc if d['experimental_arm'] == a) for a in DISTANCE_M},
                                'interpretation_withheld': any(not d['match'] for d in dup + rc)},
            'terms': {'0to1 / 1to0': 'reference arm outcome -> compared arm outcome within one block',
                      'primary pairs': 'R2M vs R5, H0M vs R5, R2M vs H0M: control paths with the same return step and the same first-inference noise key',
                      'secondary pairs': 'each control arm vs C0: these also contain the interruption, the discarded queue, another inference schedule, '
                                         'the spent budget and other noise draws',
                      'delta_mean_outcome': 'mean(y_arm - y_reference) over blocks', 'NA': 'the denominator is zero; this is not 0%'},
            'scope': ['Development experiment designed after the Stage 2B/2C results and cases were seen; not a confirmatory test, not an independent validation.',
                      'At most 12 independent root scenes. Blocks of one scene share the scene and the past. No interval, p-value or threshold search.',
                      'R5, R2M and H0M differ in the whole control path (speed, contact, history, final position), not in the final position alone.',
                      'H0M contains position holding, the interruption, the discarded chunk, the resumption and the spent budget; it is not "time only".',
                      'R2M returns at the R5 return step by construction; a recovery action that returns right after 2 cm was not tested.',
                      'T_ref comes from running the R5 control part in the simulator first; the matched arms are diagnostic variants, not deployable controllers.',
                      'A forced choice at a fixed checkpoint is not the value or the harm rate H(g) of an online intervention policy.',
                      'Stage 2A, 2B and 2D use different scenes and are reported separately; they are not pooled into a confirmatory sample.']}
