"""Handoff H1: one-shot latest-observation padding at the handoff after R5 (PREPARATION; nothing here has been run on
the simulator). See amendments/AMENDMENT_011 (draft).

Every (scene, checkpoint, block repeat r) block has three continuations from one snapshot:

  continue     C0          existing Continue; context only
  r_keep       R_KEEP      existing R5 (Runtime.force(2)); the policy resumes on the real last two observations
  r_pad_once   R_PAD_ONCE  the same R5; immediately before the first inference, and only once, the earlier entry of
                           the observation history is replaced by an independent copy of the latest entry

The primary comparison is R_PAD_ONCE against R_KEEP. Both pass through the same R5 control part; before the history is
touched, the complete runtime state of every R5 branch is compared with a reference run of the control part alone.
The padding changes policy.history and nothing else: no simulator state, no raw observation, no qvel, no time, no
queue, no last action, no RNG, no extra step. Duplicating an observation is not a physical stop.

Runtime (simulation.py) and the policy classes are not modified. Ground-truth object state enters only through `probe`
(logging); it never decides whether padding is applied.
"""
from __future__ import annotations
import hashlib, time
from collections import deque
from pathlib import Path
import numpy as np
from .trees import clone, as_numpy
from .io import read_json
from .simulation import retreat_budget
from .pilot import (runtime_tree, compare_tree, obs_diff, live_obs_ok, jsonable, tree_hash, ineligibility,
                    _same_run, _ref, _diff, _trace, _compare_traces, NUMERIC, collect_scenes, select_scenes, scale_decision)
from .pilot2b import inputs_without_side_effect, pair_table, cls
from .pilot2d import trace_array, TRACE_COLUMNS, PHASES, derived, event_reading

ROLE = 'handoff_h1_exploratory_dev'
ARMS = ('continue', 'r_keep', 'r_pad_once')
SHORT = {'continue': 'C0', 'r_keep': 'R_KEEP', 'r_pad_once': 'R_PAD_ONCE'}
R5_ARMS = ARMS[1:]
PRIMARY = (('R_PAD_ONCE', 'R_KEEP'),)                      # (compared arm, reference arm)
SECONDARY = (('R_KEEP', 'C0'), ('R_PAD_ONCE', 'C0'))
PAIRS = PRIMARY + SECONDARY
EXACT = {'state_atol': 0., 'action_atol': 0., 'observation_atol': 0.}


# ---------------------------------------------------------------- the one manipulation
def entry_hashes(history):
    return [tree_hash(h) for h in history]


def entry_schema(h):
    """dtype and shape of every array of one history entry (dict entries of the real policy, or any tree)."""
    def walk(x, pre=''):
        if isinstance(x, dict): return {k2: v2 for k in sorted(x) for k2, v2 in walk(x[k], f'{pre}{k}/').items()}
        a = as_numpy(x); return {pre.rstrip('/') or 'value': [str(a.dtype), list(a.shape)]}
    return walk(h)


def pad_history_once(policy):
    """history [o_(tau-1), o_tau] -> [copy(o_tau), o_tau]. The latest entry stays the same object, untouched; every
    earlier entry becomes an independent copy of it (no shared memory). Nothing but policy.history is assigned.
    policy.reset() is deliberately not used: it would also clear the queue, the last action and the counters."""
    hist = list(policy.history)
    if len(hist) != policy.obs_horizon or len(hist) < 2: raise ValueError('padding needs a full observation history of at least two entries')
    latest = hist[-1]; before = entry_hashes(hist); schema = entry_schema(latest)
    if any(entry_schema(h) != schema for h in hist): raise ValueError('history entries do not share dtype and shape')
    policy.history = deque([clone(latest) for _ in hist[:-1]] + [latest], maxlen=policy.obs_horizon)
    after = entry_hashes(policy.history)
    return {'entries': len(hist), 'history_sha256_before': before, 'history_sha256_after': after, 'latest_entry_unchanged': after[-1] == before[-1],
            'earlier_entries_equal_latest': all(h == after[-1] for h in after[:-1]), 'input_changed': before != after,
            'no_op': before == after, 'entry_schema': schema}


def apply_arm(r, arm, rep):
    """Called only after the restored runtime was compared with the snapshot. Returns the padding record (or None).
    The existing R5 runs first and unchanged; padding happens after it, once, and never if the episode already ended."""
    r.repeat = int(rep)
    if arm == 'continue': return None
    r.force(2)
    if arm != 'r_pad_once' or r.done: return None
    return pad_history_once(r.policy)


def return_state(r):
    """Everything that is compared at the return step, before the history may be touched."""
    return {'tree': runtime_tree(r), 'raw': clone(r.raw), 'controller': clone(r.env.agent.get_controller_state())}


def same_return_state(ref, r, rep, tol):
    """One R5 branch at its return step against the control-only reference of its snapshot. The noise key (repeat) is
    the only field allowed to differ; it does not enter the control part."""
    now = return_state(r); want = {**ref['tree'], 'discrete': {**ref['tree']['discrete'], 'repeat': int(rep)}}
    cmp = compare_tree(want, now['tree'], tol); raw = obs_diff(now['raw'], ref['raw']); ctl = _diff(now['controller'], ref['controller'])
    ok = bool(cmp['pass'] and live_obs_ok(raw, tol) and ctl <= tol['state_atol'])
    return {'pass': ok, 'failed': cmp['failed'], 'numeric_max_abs': cmp['numeric_max_abs'], 'raw_observation': raw, 'controller_state_max_abs': ctl}


def only_history_changed(before, r, pad):
    """After padding: the runtime is exactly what it was, except the earlier history entries."""
    after = runtime_tree(r); cmp = compare_tree(before, after, EXACT); allowed = {'history_state', 'history_rgb'}
    other = [k for k in cmp['failed'] if k not in allowed]
    ok = not other and pad['latest_entry_unchanged'] and pad['earlier_entries_equal_latest'] and (bool(set(cmp['failed']) & allowed) == pad['input_changed'])
    return {'pass': bool(ok), 'fields_changed': cmp['failed'], 'fields_changed_outside_history': other}


class Hooks:
    """Wraps Runtime.advance and policy.generate for logging (as pilot2b.Hooks) and keeps the generated chunks."""
    def __init__(self, r, probe=None, frame=None):
        self.r = r; self.probe = probe; self.frame = frame; self.rows = None; self.frames = None; self.calls = None; self.prefix = None; self.prefix_n = 0

    def __enter__(self):
        r = self.r; adv = type(r).advance; gen = type(r.policy).generate

        def advance(action):
            z = adv(r, action)
            if self.rows is not None and self.probe: self.rows.append(self.probe(r, z))
            if self.frames is not None and self.frame: self.frames.append(self.frame(r))
            if self.prefix is not None and len(self.prefix) < self.prefix_n:
                self.prefix.append({'action': np.asarray(z, np.float32).copy(), 'state': clone(r.env.get_state_dict()), 'raw': clone(r.raw)})
            return z

        def generate(episode, t, repeat=0):
            a = gen(r.policy, episode, t, repeat)
            if self.calls is not None:
                seed = getattr(r.policy, 'last_seed', None)
                self.calls.append({'t': int(t), 'policy_noise_repeat_id': int(repeat), 'seed': None if seed is None else int(seed),
                                   'chunk_sha256_16': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16], 'chunk': np.asarray(a, np.float32).copy()})
            return a
        r.advance = advance; r.policy.generate = generate; return self

    def __exit__(self, *exc):
        del self.r.advance; del self.r.policy.generate


def handoff_arrays(history, padded):
    """The real last two observations and the padded pair, under separate names (saved before any outcome exists)."""
    out = {}
    for name, hist in (('actual', history), ('padded', padded)):
        for i, h in enumerate(hist):
            tag = f'{name}_{"latest" if i == len(hist) - 1 else f"prev{len(hist) - 1 - i}"}'
            if isinstance(h, dict) and 'rgb' in h: out[f'{tag}_rgb'] = as_numpy(h['rgb']).copy(); out[f'{tag}_state'] = as_numpy(h['state']).copy()
            else: out[f'{tag}_tree_sha256'] = np.array(tree_hash(h))
    return out


def chunk_difference(a, b):
    """First resumed action chunk of R_PAD_ONCE against R_KEEP, in normalised controller units (not metres)."""
    if a is None or b is None: return None
    a = np.asarray(a, float); b = np.asarray(b, float); d = np.abs(a[:, :3] - b[:, :3])
    return {'arm_mean_abs': float(d.mean()), 'arm_max_abs': float(d.max()), 'first_action_arm_max_abs': float(d[0].max()), 'gripper_max_abs': float(np.abs(a[:, 3] - b[:, 3]).max()),
            'gripper_sign_differs_steps': int(((a[:, 3] > 0) != (b[:, 3] > 0)).sum()), 'identical': bool(np.array_equal(a, b)), 'units': 'normalised controller input'}


# ---------------------------------------------------------------- restore audit
def audit_scene(r, seed, c, checkpoints):
    """The Stage 2A audit (Continue trace, Retreat and Refresh branches, restore and the same trace after each) plus:
    the R5 control part alone as the reference, R_KEEP and R_PAD_ONCE run to the end, their return state against the
    reference before the history is touched, the padding touching only the earlier history entry, the same first
    inference step and seed, and a restore after the padded branch that shows no trace of the padding."""
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
            back('repeat'); branches = {}; ho = None; r.restore(s)
            if bool(r.eligible()[1]) != elig[1]: branches['refresh'] = {'executed': False, 'reason': 'eligibility after restore differs from the snapshot', 'pass': False}; r.complete()
            elif elig[1]: r.force(1); res = r.complete(); branches['refresh'] = {'executed': True, 'success': res['success'], 'steps': res['steps']}
            else: branches['refresh'] = {'executed': False, 'reason': 'not eligible at this checkpoint'}; r.complete()
            back('after_refresh'); r.restore(s)
            if bool(r.eligible()[2]) != elig[2] or not elig[2]:
                why = 'not eligible at this checkpoint' if not elig[2] else 'eligibility after restore differs from the snapshot'
                for arm in R5_ARMS: branches[arm] = {'executed': False, 'reason': why, **({} if not elig[2] else {'pass': False})}
            else:
                h.calls = calls = []; r.force(2); ref = return_state(r); T = r.retreat_info['steps_used']; early = bool(r.done)
                branches['reference_r5_control'] = {'executed': True, 'steps_used': T, 'end_reason': r.retreat_info['end_reason'], 'inference_calls': len(calls),
                                                    'ended_during_control': early, 'pass': not calls}
                back('after_reference_r5_control'); firsts = {}
                for arm in R5_ARMS:
                    r.restore(s); h.calls = calls = []; r.repeat = s['repeat']; r.force(2); same = same_return_state(ref, r, s['repeat'], tol); pad = None; only = None
                    if arm == 'r_pad_once' and not r.done:
                        before = runtime_tree(r); pad = pad_history_once(r.policy); only = only_history_changed(before, r, pad)
                    cc = len(calls); q = len(r.policy.queue); res = r.complete(); firsts[arm] = (calls[0]['t'], calls[0]['seed']) if calls else None
                    branches[arm] = {'executed': True, 'success': res['success'], 'steps': res['steps'], 'return_state_equals_reference': same, 'control_inference_calls': cc,
                                     'queue_len_at_return': q, 'padding': None if pad is None else {k: v for k, v in pad.items() if k != 'entry_schema'},
                                     'only_history_changed': only, 'first_inference': firsts[arm],
                                     'pass': bool(same['pass'] and cc == 0 and q == 0 and (only is None or only['pass']))}
                    back('after_' + arm)
                ho = {'expected_first_inference_step': None if early else t + T, 'first_inference': {SHORT[a]: firsts[a] for a in R5_ARMS},
                      'pass': bool(firsts['r_keep'] == firsts['r_pad_once'] and (early or (firsts['r_keep'] is not None and firsts['r_keep'][0] == t + T)))}
            r.restore(s)
            ok = (all(v['pass'] for v in checks.values()) and all(v.get('live_observation_pass', True) for v in checks.values())
                  and all(v.get('pass', True) for v in branches.values()) and side['pass'] and (ho is None or ho['pass']))
            out.append({'checkpoint': t, 'reached': True, 'eligible': elig, 'trace_steps': len(first), 'branches': branches, 'handoff': ho, 'checks': checks,
                        'input_saving_changes_nothing': side, 'pass': ok})
    reached = [x for x in out if x['reached']]
    return jsonable({'seed': int(seed), 'checkpoints': out, 'reached': len(reached), 'pass': bool(reached) and all(x['pass'] for x in reached),
                     **({} if reached else {'reason': 'no checkpoint was reached'})})


def audit_report(scenes):
    names = ('refresh', 'reference_r5_control') + R5_ARMS
    n = {a: sum(1 for s in scenes for x in s['checkpoints'] if x['reached'] and x['branches'].get(a, {}).get('executed')) for a in names}
    return {'pass': bool(scenes) and all(s['pass'] for s in scenes) and min(n.values()) > 0, 'branches_executed': n, 'results': scenes,
            'compared': {'numeric': dict(NUMERIC), 'exact': ['t', 'done', 'y', 'reason', 'repeat', 'intervention', 'intervention_t', 'queue_len', 'history_len', 'n_actions',
                                                              'inference_calls', 'retreat_info', 'rng'],
                         'added_for_h1': ['return state of every R5 branch against the control-only reference before the history is touched (runtime tree, raw observation, controller state)',
                                          'after padding: nothing but the earlier history entries differs (exact)', 'R_KEEP and R_PAD_ONCE: same first inference step and seed, no inference during control',
                                          'restore and the same Continue trace after the padded branch'],
                         'not_compared': ['policy.inference_seconds (wall-clock time)']}}


# ---------------------------------------------------------------- three-arm blocks of one scene
def collect_scene(r, seed, c, checkpoints, repeats, probe=None, frame=None, stage=None, video=False, recheck=False, order=ARMS):
    """Trunk (repeat 0, never restored); at every snapshot one reference run of the R5 control part (no inference), which
    fixes the return state and the handoff inputs before any outcome exists; then every (checkpoint, repeat) block with
    the three arms. A snapshot exists only where the three ORIGINAL choices are eligible. Returns
    (record, traces, videos, inputs, handoff_inputs)."""
    if sorted(order) != sorted(ARMS): raise ValueError('order must be a permutation of the three arms')
    tol = c['audit']; started = time.perf_counter(); checkpoints = [int(t) for t in checkpoints]; repeats = [int(x) for x in repeats]; ctx = {'phase': 'continue'}
    phase = lambda p: ctx.__setitem__('phase', p); tagged = (lambda rt, z: {**probe(rt, z), 'phase': ctx['phase']}) if probe else None
    r.reset(seed); initial = probe(r, None) if probe else None; snaps = {}; skips = []; traces = {}; videos = {}; inputs = {}; handoffs = {}
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
                                  'info': {'checkpoint': r.t, 'snapshot_id': f'{int(seed)}:{r.t}:{state_sha[:16]}', 'queue_len': len(r.policy.queue), 'history_len': len(r.policy.history),
                                           'remaining_budget': r.horizon - r.t, 'eligible': [True, True, True], 'policy_calls_so_far': r.policy.inference_calls, 'state_sha256': state_sha,
                                           'policy_inputs_sha256': tree_hash(x), 'last_gripper_command': None if last is None else float(last[-1]),
                                           'state_at_checkpoint': trunk_rows[-1] if trunk_rows else initial, 'stage_so_far': (stage_of(list(trunk_rows)) or {}).get('furthest_stage')}}
                else:
                    skips.append({'checkpoint': r.t, 'reason': 'not_all_three_choices_eligible', 'detail': ineligibility(r), 'eligible': [bool(x) for x in r.eligible()]})
            r.step_base()
        trunk = {'policy_noise_repeat_id': 0, 'success': r.y, 'steps': r.t, 'reason': r.reason, 'policy_calls': r.policy.inference_calls,
                 'policy_inference_seconds': r.policy.inference_seconds, 'stage': stage_of(trunk_rows),
                 'inference_calls': [{k: v for k, v in x.items() if k != 'chunk'} for x in trunk_calls]}
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
            # reference: the R5 control part alone. It fixes the return state every R5 branch must reproduce and the handoff
            # inputs (the real last two observations and their padded form), before any outcome of this snapshot exists.
            tick = time.perf_counter(); start(); phase('intervention_move'); h.rows = ref_rows = []; h.frames = None; h.calls = ref_calls = []
            r.force(2); phase('continue')
            if ref_calls: raise RuntimeError(f'scene {seed} checkpoint {t}: the reference control part called the policy')
            ref = return_state(r); ri = clone(r.retreat_info); T = int(ri['steps_used']); early = bool(r.done); hist = list(r.policy.history)
            actual = entry_hashes(hist); padded = [clone(hist[-1]) for _ in hist[:-1]] + [hist[-1]]
            handoffs[t] = handoff_arrays(hist, padded) if not early else {}
            start()
            info['reference'] = {'r5_steps': T, 'end_reason': ri['end_reason'], 'reached': bool(ri['reached']), 'ended_during_control': early, 'return_step': t + T,
                                 'remaining_budget_at_return': r.horizon - (t + T), 'achieved_displacement_m': ri['achieved_displacement_m'], 'target_error_m': ri['target_error_m'],
                                 'gripper_command': ri['gripper_command'], 'policy_inference_calls': 0, 'restored_and_verified_after_reference': True,
                                 'actual_history_sha256': actual, 'padded_history_sha256': [actual[-1]] * len(actual), 'padding_would_change_the_input': len(set(actual)) > 1,
                                 'history_entry_schema': entry_schema(hist[-1]), 'handoff_inputs_sha256': tree_hash(handoffs[t]) if handoffs[t] else None,
                                 'wall_seconds': time.perf_counter() - tick, 'note': 'preparation run, not a continuation; not counted in any outcome'}
            if probe: traces[f'ref_{t}'] = trace_array(ref_rows)

            def branch(rep, arm, keep_frames=False):
                tick = time.perf_counter(); err, live = start(); calls0 = r.policy.inference_calls; sec0 = r.policy.inference_seconds
                h.rows = rows = []; h.frames = frames = [] if keep_frames else None; h.calls = calls = []; ctl = arm != 'continue'; pad = None; only = None; same = None
                r.repeat = int(rep)
                if ctl:
                    phase('intervention_move'); r.force(2); phase('policy_after_return'); same = same_return_state(ref, r, rep, tol)
                    if not same['pass'] or r.t - t != T:
                        raise RuntimeError(f'scene {seed} checkpoint {t} repeat {rep} {arm}: the return state differs from the reference of this snapshot ({jsonable(same)}); '
                                           'a technical fault, the result is not adjusted')
                    if calls: raise RuntimeError(f'scene {seed} checkpoint {t} repeat {rep} {arm}: the policy was called during the control part')
                    if arm == 'r_pad_once' and not r.done:
                        before = runtime_tree(r); pad = pad_history_once(r.policy); only = only_history_changed(before, r, pad)
                        if not only['pass']: raise RuntimeError(f'scene {seed} checkpoint {t} repeat {rep}: padding changed more than the earlier history entry ({only})')
                k = r.t - t if ctl else 0; q = len(r.policy.queue); ended = ctl and bool(r.done); hist_now = entry_hashes(r.policy.history) if ctl else None
                res = r.complete(); phase('continue'); first = calls[0] if calls else None
                cell = {'experimental_arm': arm, 'arm': SHORT[arm], 'block_repeat_id': rep, 'policy_noise_repeat_id': rep, 'y': int(res['success']), 'steps': res['steps'],
                        'reason': res['reason'], 'success_step': res['steps'] if res['success'] else None, 'intervention': res['intervention'], 'intervention_step': res['intervention_step'],
                        'queue_len_after_choice': q, 'retreat': res['retreat'], 'r5_steps': k if ctl else None, 'return_state_equals_reference': same,
                        'ended_during_control': bool(ended), 'policy_resumed': bool(ctl and not ended), 'padding_applied': pad is not None,
                        'padding': None if pad is None else {n: v for n, v in pad.items() if n != 'entry_schema'}, 'only_history_changed': only,
                        'policy_input_history_sha256': hist_now, 'first_inference_step': None if first is None else first['t'], 'first_inference_seed': None if first is None else first['seed'],
                        'first_chunk': None if first is None else first['chunk'].tolist(), 'steps_after_checkpoint': res['steps'] - t,
                        'policy_calls_after_snapshot': res['policy_calls'] - calls0, 'policy_seconds_after_snapshot': res['policy_inference_seconds'] - sec0,
                        'inference_calls': [{n: v for n, v in x.items() if n != 'chunk'} for x in calls], 'prefix_replay_state_error': err, 'live_observation': live,
                        'stage': stage_of(list(trunk_rows[:t]) + rows), 'wall_seconds': time.perf_counter() - tick}
                return cell, rows, frames

            for rep in repeats:
                cells = {}
                for arm in order:
                    want = bool(video and frame) and t == first_snapshot and rep == 0
                    cell, rows, frames = branch(rep, arm, want); cells[arm] = cell
                    if probe: traces[f'b_{t}_{rep}_{SHORT[arm]}'] = trace_array(rows)
                    if want: videos[f'scene{seed}_t{t}_r{rep}_{arm}'] = list(trunk_frames[:t + 1]) + frames
                    if arm in R5_ARMS and rep == 0: refs[arm] = (_ref(r), cell['inference_calls'], cell['first_chunk'])
                    if arm == 'continue' and rep == 0: duplicates.append({'checkpoint': t, 'repeat': 0, 'against': 'trunk', **_same_run(trunk_ref, r, tol)})
                k, p = cells['r_keep'], cells['r_pad_once']
                ho = {'return_step': t + T, 'ended_during_control': early, 'first_inference_step_R_KEEP': k['first_inference_step'], 'first_inference_step_R_PAD_ONCE': p['first_inference_step'],
                      'first_inference_seed_R_KEEP': k['first_inference_seed'], 'first_inference_seed_R_PAD_ONCE': p['first_inference_seed'],
                      'padding_applied': p['padding_applied'], 'input_changed': bool(p['padding'] and p['padding']['input_changed']),
                      'first_chunk_difference': chunk_difference(p['first_chunk'], k['first_chunk'])}
                ho['pass'] = bool((k['first_inference_step'], k['first_inference_seed']) == (p['first_inference_step'], p['first_inference_seed'])
                                  and (early or k['first_inference_step'] == t + T) and p['padding_applied'] == (not early) and not k['padding_applied']
                                  and k['policy_input_history_sha256'] == info['reference']['actual_history_sha256']
                                  and (early or p['policy_input_history_sha256'] == info['reference']['padded_history_sha256']))
                if not ho['pass']: raise RuntimeError(f'scene {seed} checkpoint {t} repeat {rep}: R_KEEP and R_PAD_ONCE did not hand over as declared ({jsonable(ho)})')
                blocks.append({'scene': int(seed), 'checkpoint': t, 'repeat': rep, 'block_repeat_id': rep, 'role': ROLE, 'snapshot_id': info['snapshot_id'],
                               **{'y_' + SHORT[a]: cells[a]['y'] for a in ARMS}, 'cells': [cells[a] for a in ARMS], 'handoff': ho})
            if recheck:      # after every other branch of this snapshot: R_KEEP and R_PAD_ONCE of repeat 0 once more
                for arm in R5_ARMS:
                    if arm not in refs: continue
                    cell, _, _ = branch(0, arm); same = _same_run(refs[arm][0], r, tol); eq = cell['inference_calls'] == refs[arm][1] and cell['first_chunk'] == refs[arm][2]
                    rechecks.append({'checkpoint': t, 'repeat': 0, 'experimental_arm': arm, 'against': 'first run of the same arm in the same block', **same,
                                     'inference_and_first_chunk_equal': bool(eq), 'match': bool(same['match'] and eq)})
    ok = all(x['match'] for x in duplicates + rechecks)
    record = {'scene': int(seed), 'role': ROLE, 'planned_checkpoints': checkpoints, 'repeats': repeats, 'arms': list(ARMS), 'initial_state': initial, 'trunk': trunk,
              'snapshots': [snaps[t]['info'] for t in sorted(snaps)], 'skips': sorted(skips, key=lambda x: x['checkpoint']), 'blocks': blocks,
              'no_intervention_duplicates': duplicates, 'rechecks': rechecks, 'reproducibility_ok': ok,
              'counts': {'snapshots': len(snaps), 'blocks': len(blocks), 'continuations': len(ARMS) * len(blocks), 'reference_control_runs': len(snaps),
                         'extra_reproducibility_continuations': len(rechecks)},
              'wall_seconds': time.perf_counter() - started, 'completed': True}
    record = jsonable(record); validate_scene(record, c)
    return record, traces, videos, inputs, handoffs


def validate_scene(rec, c=None):
    """Accounting of one scene record: nothing missing, nothing twice, padding exactly where declared and nowhere else."""
    cps = rec['planned_checkpoints']; reps = rec['repeats']; snap = {x['checkpoint']: x for x in rec['snapshots']}; skip = [x['checkpoint'] for x in rec['skips']]
    if len(skip) != len(set(skip)) or set(snap) & set(skip) or set(snap) | set(skip) != set(cps):
        raise ValueError(f'scene {rec["scene"]}: planned checkpoints are not each a snapshot or a listed skip')
    keys = [(b['checkpoint'], b['repeat']) for b in rec['blocks']]
    if len(keys) != len(set(keys)) or set(keys) != {(t, x) for t in snap for x in reps}:
        raise ValueError(f'scene {rec["scene"]}: blocks do not match snapshots x block repeats')
    for b in rec['blocks']:
        cells = b['cells']; t = b['checkpoint']; rep = b['repeat']; ref = snap[t]['reference']; T = ref['r5_steps']; early = ref['ended_during_control']
        if b['scene'] != rec['scene'] or b['role'] != rec['role'] or b['block_repeat_id'] != rep: raise ValueError('block scene/role/repeat differs from its record')
        if [x['experimental_arm'] for x in cells] != list(ARMS): raise ValueError('a block needs exactly the three arms, once each')
        for x in cells:
            if x['block_repeat_id'] != rep or x['policy_noise_repeat_id'] != rep or any(k['policy_noise_repeat_id'] != rep for k in x['inference_calls']):
                raise ValueError('an arm left the noise key of its block')
            if len(x['inference_calls']) != x['policy_calls_after_snapshot']: raise ValueError('inference log does not match the call counter')
            if b['y_' + x['arm']] != x['y']: raise ValueError('joined outcome differs from the cell')
            if c is not None and x['steps'] > c['env']['horizon']: raise ValueError('a branch exceeded the episode horizon')
        c0, keep, pad = cells
        if c0['intervention'] is not None or c0['retreat'] is not None or c0['queue_len_after_choice'] != snap[t]['queue_len'] or c0['padding_applied']:
            raise ValueError('Continue changed the queue, was recorded as an intervention or was padded')
        if c0['first_inference_step'] not in (None, t + snap[t]['queue_len']): raise ValueError('Continue inferred before the natural chunk boundary')
        for x in (keep, pad):
            if x['intervention'] != 2 or x['intervention_step'] != t or x['queue_len_after_choice'] != 0 or x['retreat'] is None or x['r5_steps'] != T:
                raise ValueError('an R5 arm is not the existing Retreat at the checkpoint with the reference step count')
            if not x['return_state_equals_reference']['pass']: raise ValueError('an R5 arm did not reproduce the reference return state')
            if x['retreat']['gripper_command'] != snap[t]['last_gripper_command']: raise ValueError('the gripper command of the snapshot was not held during R5')
            if x['ended_during_control'] != early or x['policy_resumed'] == early: raise ValueError('ended-during-control flag differs from the reference')
            if early and (x['inference_calls'] or x['steps'] != t + T or x['padding_applied']): raise ValueError('an arm that ended during R5 was padded or resumed')
            if not early and x['first_inference_step'] != t + T: raise ValueError('the policy did not resume at the return step')
        if keep['padding_applied'] or keep['padding'] is not None: raise ValueError('R_KEEP was padded')
        if not early:
            p = pad['padding']
            if not pad['padding_applied'] or not p['latest_entry_unchanged'] or not p['earlier_entries_equal_latest'] or not pad['only_history_changed']['pass']:
                raise ValueError('R_PAD_ONCE was not padded exactly once on the earlier history entry only')
            if keep['policy_input_history_sha256'] != ref['actual_history_sha256'] or pad['policy_input_history_sha256'] != ref['padded_history_sha256']:
                raise ValueError('the policy inputs at the handoff are not the recorded real / padded histories')
            if (keep['first_inference_step'], keep['first_inference_seed']) != (pad['first_inference_step'], pad['first_inference_seed']): raise ValueError('R_KEEP and R_PAD_ONCE resumed with another step or seed')
            if p['input_changed'] != ref['padding_would_change_the_input']: raise ValueError('no-op flag differs from the recorded histories')
        if not b['handoff']['pass']: raise ValueError('handoff check failed')
    n = rec['counts']
    if (n['snapshots'], n['blocks'], n['continuations'], n['reference_control_runs']) != (len(snap), len(keys), len(ARMS) * len(keys), len(snap)):
        raise ValueError('counts do not match the record')
    return True


def repro_problems(rec, rerun):
    snaps = sorted(x['checkpoint'] for x in rec['snapshots']); out = []
    dup = sorted((d['checkpoint'], d['repeat'], d['against']) for d in rec['no_intervention_duplicates'])
    if dup != [(t, 0, 'trunk') for t in snaps]: out.append('trunk-duplicate checks are not one per snapshot')
    got = sorted((d['checkpoint'], d['repeat'], d['experimental_arm']) for d in rec['rechecks'])
    want = sorted((t, 0, a) for t in snaps for a in R5_ARMS) if rerun else []
    if got != want: out.append('rerun checks are not the declared ones' if rerun else 'rerun checks recorded for a scene that has none planned')
    bad = sum(1 for d in rec['no_intervention_duplicates'] + rec['rechecks'] if d.get('match') is not True)
    if bad: out.append(f'{bad} reproducibility checks did not match')
    if bool(rec.get('reproducibility_ok')) != (not bad): out.append('stored reproducibility flag contradicts the recorded checks')
    return out


def finalize(out, seeds, c=None, rerun_scenes=()):
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
    meta = {'role': ROLE, 'arms': list(ARMS), 'scheduled_scenes': [int(s) for s in seeds], 'recorded_scenes': [x['scene'] for x in recs], 'completed': not problems, 'problems': problems,
            'snapshots': sum(x['counts']['snapshots'] for x in recs), 'blocks': sum(x['counts']['blocks'] for x in recs), 'continuations': sum(x['counts']['continuations'] for x in recs),
            'reference_control_runs': sum(x['counts']['reference_control_runs'] for x in recs),
            'extra_reproducibility_continuations': sum(x['counts']['extra_reproducibility_continuations'] for x in recs),
            'reproducibility_ok': repro if recs else None, 'wall_seconds_sum_over_scenes': sum(x['wall_seconds'] for x in recs)}
    return meta, recs


# ---------------------------------------------------------------- descriptive tables (no interval, no p-value)
def block_rows(records):
    rows = []
    for rec in records:
        snap = {x['checkpoint']: x for x in rec['snapshots']}
        for b in rec['blocks']:
            s = snap[b['checkpoint']]; ref = s['reference']; at = s.get('state_at_checkpoint') or {}; cell = {x['arm']: x for x in b['cells']}; d = b['handoff']['first_chunk_difference'] or {}
            rows.append({'scene': b['scene'], 'role': b['role'], 'checkpoint': b['checkpoint'], 'block_repeat_id': b['block_repeat_id'], 'snapshot_id': b['snapshot_id'],
                         **{f'y_{k}': b[f'y_{k}'] for k in SHORT.values()}, **{f'{a}_vs_{r}': cls(b[f'y_{r}'], b[f'y_{a}']) for a, r in PAIRS},
                         'r5_steps': ref['r5_steps'], 'return_step': ref['return_step'], 'remaining_budget_at_return': ref['remaining_budget_at_return'],
                         'r5_reached': ref['reached'], 'r5_achieved_displacement_m': ref['achieved_displacement_m'], 'ended_during_control': ref['ended_during_control'],
                         'policy_resumed': cell['R_KEEP']['policy_resumed'], 'padding_applied': cell['R_PAD_ONCE']['padding_applied'], 'input_changed': b['handoff']['input_changed'],
                         'first_inference_step': cell['R_KEEP']['first_inference_step'], 'first_inference_seed': cell['R_KEEP']['first_inference_seed'],
                         **{f'first_chunk_{k}': d.get(k) for k in ('arm_mean_abs', 'arm_max_abs', 'first_action_arm_max_abs', 'gripper_max_abs', 'gripper_sign_differs_steps', 'identical')},
                         **{f'steps_{k}': cell[k]['steps'] for k in cell}, **{f'reason_{k}': cell[k]['reason'] for k in cell},
                         **{f'stage_{k}': (cell[k]['stage'] or {}).get('furthest_stage') for k in cell}, 'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'],
                         'stage_so_far_at_checkpoint': s.get('stage_so_far'), 'grasped_at_checkpoint': at.get('is_cubeA_grasped')})
    return rows


def pair_rows(rows):
    cps = sorted({x['checkpoint'] for x in rows}); out = []
    for a, ref in PAIRS:
        kind = 'primary' if (a, ref) in PRIMARY else 'secondary'
        out.append({'comparison': kind, 'checkpoint': 'all', **pair_table(rows, a, ref)})
        out += [{'comparison': kind, 'checkpoint': t, **pair_table([x for x in rows if x['checkpoint'] == t], a, ref)} for t in cps]
    return out


def scene_rows(records, rows=None):
    rows = block_rows(records) if rows is None else rows; out = []
    for rec in records:
        mine = [x for x in rows if x['scene'] == rec['scene']]; n = len(mine)
        row = {'scene': rec['scene'], 'role': rec['role'], 'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'],
               'reached_checkpoints_m': rec['counts']['snapshots'], 'checkpoints_used': ' '.join(str(x['checkpoint']) for x in rec['snapshots']), 'blocks': n,
               'blocks_with_padding': sum(bool(x['padding_applied']) for x in mine), 'blocks_with_changed_input': sum(bool(x['input_changed']) for x in mine),
               'reproducibility_ok': rec['reproducibility_ok']}
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
                               'scenes_positive': sum(x > 0 for x in d), 'scenes_negative': sum(x < 0 for x in d), 'scenes_zero': sum(x == 0 for x in d)} if d else 'NA')
    for k in SHORT.values():
        v = [s[f'mean_y_{k}'] for s in used]; out[f'mean_y_{k}'] = float(np.mean(v)) if v else 'NA'
    return out


def skip_rows(records):
    return [{'scene': rec['scene'], 'checkpoint': x['checkpoint'], 'reason': x['reason'], 'detail': '; '.join(x['detail'] or []), 'trunk_success': rec['trunk']['success'],
             'trunk_steps': rec['trunk']['steps'], 'cells_not_run': len(ARMS) * len(rec['repeats'])} for rec in records for x in rec['skips']]


def summarize(records, planned_scenes=None):
    rows = block_rows(records); scenes = scene_rows(records, rows); skips = skip_rows(records); skip_counts = {}
    for x in skips: skip_counts[x['reason']] = skip_counts.get(x['reason'], 0) + 1
    stat = lambda v: {'n': len(v), 'mean': float(np.mean(v)), 'min': float(np.min(v)), 'median': float(np.median(v)), 'max': float(np.max(v))} if v else {'n': 0}
    num = lambda k, sel=rows: [x[k] for x in sel if x[k] is not None]; res = [x for x in rows if x['policy_resumed']]; ch = [x for x in res if x['input_changed']]
    dup = [d for rec in records for d in rec['no_intervention_duplicates']]; rc = [d for rec in records for d in rec['rechecks']]
    return {'role': ROLE, 'arms': dict(SHORT), 'scenes_planned': planned_scenes, 'scenes_recorded': len(records), 'trunk_success_scenes': sum(r['trunk']['success'] for r in records),
            'scenes_with_a_snapshot': sum(1 for r in records if r['snapshots']), 'snapshots': sum(r['counts']['snapshots'] for r in records), 'blocks': len(rows),
            'continuations': sum(r['counts']['continuations'] for r in records), 'reference_control_runs': sum(r['counts']['reference_control_runs'] for r in records),
            'extra_reproducibility_continuations': sum(r['counts']['extra_reproducibility_continuations'] for r in records), 'skipped_checkpoints': skip_counts,
            'success_blocks_by_arm': {k: sum(x[f'y_{k}'] for x in rows) for k in SHORT.values()},
            'pairs': pair_rows(rows), 'scene_equal_weight': scene_equal_summary(scenes),
            'handoff': {'blocks': len(rows), 'policy_resumed_blocks': len(res), 'ended_during_r5_blocks': len(rows) - len(res), 'padding_applied_blocks': sum(bool(x['padding_applied']) for x in rows),
                        'input_changed_blocks': len(ch), 'no_op_blocks': len(res) - len(ch), 'first_chunk_identical_blocks': sum(bool(x['first_chunk_identical']) for x in res),
                        'first_chunk_arm_mean_abs': stat(num('first_chunk_arm_mean_abs', res)), 'first_chunk_arm_max_abs': stat(num('first_chunk_arm_max_abs', res)),
                        'first_chunk_blocks_with_a_gripper_sign_difference': sum(1 for x in res if x['first_chunk_gripper_sign_differs_steps']),
                        'outcome_differs_blocks': sum(x['y_R_PAD_ONCE'] != x['y_R_KEEP'] for x in rows),
                        'chunk_differs_but_outcome_equal_blocks': sum(1 for x in res if not x['first_chunk_identical'] and x['y_R_PAD_ONCE'] == x['y_R_KEEP']),
                        'r5_steps': stat(num('r5_steps'))},
            'resumed_blocks_only': {'note': 'subset selected by a condition of the R5 control part (the episode did not end during it); a separate denominator',
                                    'pairs': [{'checkpoint': 'all', **pair_table(res, a, ref)} for a, ref in PAIRS] if res else []},
            'reproducibility': {'no_intervention_duplicates': len(dup), 'duplicate_mismatches': sum(1 for d in dup if not d['match']), 'rechecks': len(rc),
                                'recheck_mismatches': sum(1 for d in rc if not d['match']), 'interpretation_withheld': any(not d['match'] for d in dup + rc)},
            'terms': {'primary pair': 'R_PAD_ONCE vs R_KEEP: same snapshot, same R5 control part, same return step, same first-inference seed; only the earlier observation-history entry differs',
                      'secondary pairs': 'each R5 arm vs C0: these also contain the retreat, the discarded queue, another inference schedule and the spent budget',
                      '0to1 / 1to0': 'reference arm outcome -> compared arm outcome within one block', 'NA': 'the denominator is zero; this is not 0%'},
            'scope': ['Exploratory development experiment; at most 12 independent scenes; blocks of one scene share the scene and the past. No p-value, interval or pass criterion.',
                      'A difference between R_PAD_ONCE and R_KEEP is a difference under a controlled manipulation of the resume input. It does not establish OOD, mode collapse or any single cause.',
                      'The padding acts directly on the first resumed chunk only; later differences come through the changed trajectory.',
                      'Duplicating the latest observation is not a physical stop and does not set any velocity to zero.',
                      'R_PAD_ONCE being better than R_KEEP would not make it better than C0; the two comparisons are reported separately.']}
