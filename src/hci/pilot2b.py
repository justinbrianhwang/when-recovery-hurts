"""Stage 2B exploratory four-arm control (development only). See amendments/AMENDMENT_006.

Every (scene, checkpoint, block repeat r) block has four continuations from one snapshot:

  continue            C0  keeps the queue, next inference at the natural chunk boundary, future policy noise key r
  continue_alt_noise  C1  the same Continue rule, future policy noise key r + 4 (nothing else changes)
  refresh             F   discards the queue and infers at the checkpoint, noise key r
  retreat_retry       R   retreat, then inference, noise key r

C1 is an experimental control, not a recovery action: it is never sent through Runtime.force and is never counted as
an intervention. The block repeat r is the pairing key; the policy noise repeat is a separate field.

Reuses the Stage 2A helpers of `pilot` (full-state comparison, restore checks, scene loop). Mock-testable; ground-truth
object state enters only through `probe` (logging) and is never given to the policy.
"""
from __future__ import annotations
import hashlib, time
from pathlib import Path
import numpy as np
from .trees import clone, as_numpy
from .schema import state_parts
from .io import read_json
from .pilot import (runtime_tree, compare_tree, obs_diff, live_obs_ok, jsonable, tree_hash, ineligibility, rows_array,
                    _same_run, _ref, _diff, _trace, _compare_traces, NUMERIC, collect_scenes, select_scenes, scale_decision)

ROLE = 'stage2b_exploratory_dev'
ARMS = ('continue', 'continue_alt_noise', 'refresh', 'retreat_retry')
SHORT = {'continue': 'C0', 'continue_alt_noise': 'C1', 'refresh': 'F', 'retreat_retry': 'R'}
FORCE = {'continue': None, 'continue_alt_noise': None, 'refresh': 1, 'retreat_retry': 2}
NOISE_OFFSET = 4
PAIRS = (('F', 'C0'), ('R', 'C0'), ('C1', 'C0'), ('F', 'C1'), ('R', 'C1'), ('R', 'F'))      # (compared arm, reference arm)
EXACT = {'state_atol': 0., 'action_atol': 0., 'observation_atol': 0.}


def noise_id(arm, r):
    """Key of the FUTURE policy noise. Only the alternative-noise Continue leaves the block's own repeat."""
    return int(r) + NOISE_OFFSET if arm == 'continue_alt_noise' else int(r)


def apply_arm(r, arm, rep):
    """Called only after the restored runtime was compared with the snapshot. The two Continue arms touch neither the
    queue nor the history and add no inference; the noise key is the single intended difference."""
    r.repeat = noise_id(arm, rep)
    if FORCE[arm] is not None: r.force(FORCE[arm])


# ---------------------------------------------------------------- logging hooks
class Hooks:
    """Wraps Runtime.advance and policy.generate for logging. The prefix replay inside restore passes through neither."""
    def __init__(self, r, probe=None, frame=None):
        self.r = r; self.probe = probe; self.frame = frame
        self.rows = None; self.frames = None; self.calls = None; self.prefix = None; self.prefix_n = 0

    def __enter__(self):
        r = self.r; adv = type(r).advance; gen = type(r.policy).generate

        def advance(action):
            z = adv(r, action)
            if self.rows is not None and self.probe: self.rows.append(self.probe(r, z))
            if self.frames is not None and self.frame: self.frames.append(self.frame(r))
            if self.prefix is not None and len(self.prefix) < self.prefix_n:
                self.prefix.append({'action': np.asarray(z, np.float32).copy(), 'state': clone(r.env.get_state_dict()), 'raw': clone(r.raw),
                                    'calls_so_far': len(self.calls) if self.calls is not None else 0})
            return z

        def generate(episode, t, repeat=0):
            a = gen(r.policy, episode, t, repeat)
            if self.calls is not None:
                seed = getattr(r.policy, 'last_seed', None)
                self.calls.append({'t': int(t), 'policy_noise_repeat_id': int(repeat), 'seed': None if seed is None else int(seed),
                                   'chunk_sha256_16': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]})
            return a
        r.advance = advance; r.policy.generate = generate; return self

    def __exit__(self, *exc):
        del self.r.advance; del self.r.policy.generate


def policy_inputs(r):
    """What an online model could read at the snapshot, saved once per snapshot BEFORE any outcome exists.
    Whitelisted observation history, last action, unexecuted queue, time. No object pose, no flags, no labels."""
    p = r.policy; hist = list(p.history); out = {}
    if hist and isinstance(hist[0], dict) and 'rgb' in hist[0]:
        rgb = np.stack([as_numpy(h['rgb'])[0] for h in hist])
        if rgb.dtype != np.uint8: raise ValueError(f'policy RGB history is {rgb.dtype}, not uint8')
        out['rgb_history'] = rgb; out['state_history'] = np.stack([as_numpy(h['state'])[0] for h in hist]).astype(np.float32)
    else:
        out['state_history'] = np.stack([np.concatenate([as_numpy(x).reshape(-1) for x in state_parts(h)]) for h in hist]).astype(np.float32)
    adim = r.adim
    out['last_action'] = np.zeros(adim, np.float32) if p.last_action is None else np.asarray(p.last_action, np.float32).copy()
    out['queue'] = np.asarray(p.queue, np.float32).reshape(-1, adim).copy()
    out['t'] = np.array(r.t, np.int64); out['remaining_budget'] = np.array(r.horizon - r.t, np.int64)
    if hasattr(p, 'features'): out['policy_feature'] = np.asarray(p.features(r.t, r.horizon), np.float32)
    return out


def inputs_without_side_effect(r):
    before = runtime_tree(r); x = policy_inputs(r); cmp = compare_tree(before, runtime_tree(r), EXACT)
    return x, before, cmp


# ---------------------------------------------------------------- restore audit (Stage 2A audit + the C1 branch)
def audit_scene(r, seed, c, checkpoints):
    """Stage 2A audit, plus: a Continue-alt-noise branch run to the end before another restore, and a check that
    saving the policy inputs changes nothing in the runtime (state, queue, history, RNG)."""
    tol = c['audit']; n = tol['replay_steps']; r.reset(seed); out = []
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
        back('repeat'); branches = {}
        for arm in ('retreat_retry', 'refresh', 'continue_alt_noise'):
            r.restore(s); a = FORCE[arm]
            if a is None:
                q = len(r.policy.queue); apply_arm(r, arm, s['repeat']); kept = len(r.policy.queue) == q and r.intervention is None
                res = r.complete()
                branches[arm] = {'executed': True, 'success': res['success'], 'steps': res['steps'], 'policy_noise_repeat_id': noise_id(arm, s['repeat']),
                                 'queue_and_intervention_untouched': bool(kept), 'pass': bool(kept)}
            else:
                now = bool(r.eligible()[a])
                if now != elig[a]:
                    branches[arm] = {'executed': False, 'reason': 'eligibility after restore differs from the snapshot', 'pass': False}; r.complete()
                elif now:
                    r.force(a); res = r.complete()
                    branches[arm] = {'executed': True, 'success': res['success'], 'steps': res['steps'], 'retreat': res['retreat']}
                else:
                    branches[arm] = {'executed': False, 'reason': 'not eligible at this checkpoint'}; r.complete()
            back('after_' + arm)
        r.restore(s)
        ok = (all(v['pass'] for v in checks.values()) and all(v.get('live_observation_pass', True) for v in checks.values())
              and all(v.get('pass', True) for v in branches.values()) and side['pass'])
        out.append({'checkpoint': t, 'reached': True, 'eligible': elig, 'trace_steps': len(first), 'branches': branches, 'checks': checks,
                    'input_saving_changes_nothing': side, 'pass': ok})
    reached = [x for x in out if x['reached']]
    return jsonable({'seed': int(seed), 'checkpoints': out, 'reached': len(reached), 'pass': bool(reached) and all(x['pass'] for x in reached),
                     **({} if reached else {'reason': 'no checkpoint was reached'})})


def audit_report(scenes):
    did = lambda name: sum(1 for s in scenes for x in s['checkpoints'] if x['reached'] and x['branches'][name]['executed'])
    n = {a: did(a) for a in ('retreat_retry', 'refresh', 'continue_alt_noise')}
    return {'pass': bool(scenes) and all(s['pass'] for s in scenes) and min(n.values()) > 0, 'branches_executed': n, 'results': scenes,
            'compared': {'numeric': dict(NUMERIC), 'exact': ['t', 'done', 'y', 'reason', 'repeat', 'intervention', 'intervention_t', 'queue_len',
                                                              'history_len', 'n_actions', 'inference_calls', 'retreat_info', 'rng'],
                         'added_in_stage2b': ['restore and trace after a Continue-alt-noise branch', 'saving the policy inputs leaves the runtime unchanged (exact)'],
                         'not_compared': ['policy.inference_seconds (wall-clock time)']}}


# ---------------------------------------------------------------- C0 against C1 while the old queue is executed
def prefix_parity(a, b, calls_a, calls_b, t, qlen, steps_a, steps_b, tol):
    """C1 must be the same rollout as C0 until the queue left at the snapshot is used up: same commands, same
    simulator state, same observation from the environment, no inference; the first new inference is at t + qlen."""
    want_a = min(qlen, steps_a - t); want_b = min(qlen, steps_b - t)
    length = len(a) == want_a and len(b) == want_b and len(a) == len(b)
    act = max([_diff(x['action'], y['action']) for x, y in zip(a, b)] + [0.])
    state = max([_diff(x['state'], y['state']) for x, y in zip(a, b)] + [0.])
    obs = [obs_diff(x['raw'], y['raw']) for x, y in zip(a, b)]
    o_state = max([o['state_max_abs'] for o in obs] + [0.]); o_rgb = max([o['rgb_max_abs'] for o in obs] + [0.])
    no_call = all(x['calls_so_far'] == 0 for x in a + b)
    fa = calls_a[0]['t'] if calls_a else None; fb = calls_b[0]['t'] if calls_b else None
    first_ok = fa == fb and fa in (None, t + qlen)
    seeds_differ = None if fa is None else calls_a[0]['seed'] != calls_b[0]['seed']
    ok = bool(length and no_call and first_ok and act <= tol['action_atol'] and state <= tol['state_atol']
              and o_state <= tol['state_atol'] and o_rgb <= tol['observation_atol'])
    return {'pass': ok, 'queue_len_at_snapshot': qlen, 'compared_steps': len(a), 'lengths_as_expected': bool(length),
            'action_max_abs': act, 'state_max_abs': state, 'live_observation_state_max_abs': o_state, 'live_observation_rgb_max_abs': o_rgb,
            'no_inference_while_queue_is_executed': bool(no_call), 'first_inference_step_C0': fa, 'first_inference_step_C1': fb,
            'first_inference_step_expected': t + qlen, 'first_inference_at_natural_boundary': bool(first_ok),
            'first_inference_seeds_differ': seeds_differ}


# ---------------------------------------------------------------- four-arm blocks of one scene
def collect_scene(r, seed, c, checkpoints, repeats, probe=None, frame=None, stage=None, video=False, recheck=False, order=ARMS):
    """Trunk (repeat 0, never restored), then every (checkpoint, repeat) block with the four arms.

    A snapshot exists only where the three ORIGINAL choices are eligible (the control arm does not loosen that).
    `order` exists for the order-independence test; results must not depend on it. Returns (record, traces, videos, inputs).
    """
    if sorted(order) != sorted(ARMS): raise ValueError('order must be a permutation of the four arms')
    tol = c['audit']; started = time.perf_counter(); checkpoints = [int(t) for t in checkpoints]; repeats = [int(x) for x in repeats]
    if max(repeats) >= NOISE_OFFSET or min(repeats) < 0: raise ValueError('block repeats must stay below the alternative-noise offset')
    r.reset(seed); initial = probe(r, None) if probe else None; snaps = {}; skips = []; traces = {}; videos = {}; inputs = {}
    stage_of = (lambda rows: stage(initial, rows)) if (stage and probe) else (lambda rows: None)
    with Hooks(r, probe, frame) as h:
        h.rows = trunk_rows = []; h.frames = trunk_frames = [frame(r)] if (video and frame) else None; h.calls = trunk_calls = []
        while not r.done:
            if r.t in checkpoints:
                if r.eligible().all():
                    x, tree, side = inputs_without_side_effect(r)
                    if not side['pass']: raise RuntimeError(f'scene {seed} checkpoint {r.t}: saving the policy inputs changed the runtime ({side["failed"]})')
                    state_sha = tree_hash(r.env.get_state_dict()); inputs[r.t] = x
                    snaps[r.t] = {'s': r.snapshot(), 'tree': tree,
                                  'info': {'checkpoint': r.t, 'snapshot_id': f'{int(seed)}:{r.t}:{state_sha[:16]}', 'queue_len': len(r.policy.queue),
                                           'history_len': len(r.policy.history), 'remaining_budget': r.horizon - r.t, 'eligible': [True, True, True],
                                           'policy_calls_so_far': r.policy.inference_calls, 'state_sha256': state_sha, 'policy_inputs_sha256': tree_hash(x),
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
        if probe: traces['trunk'] = rows_array(trunk_rows)
        blocks = []; duplicates = []; rechecks = []; first_snapshot = min(snaps) if snaps else None
        for t in sorted(snaps):
            s = snaps[t]['s']; info = snaps[t]['info']; qlen = info['queue_len']; refs = {}

            def branch(rep, arm, keep_frames=False):
                tick = time.perf_counter(); r.restore(s); err = r.replay_error
                start = compare_tree(snaps[t]['tree'], runtime_tree(r), tol); live = obs_diff(r.replay_raw, s['raw'])
                if not start['pass'] or not live_obs_ok(live, tol):
                    raise RuntimeError(f'scene {seed} checkpoint {t}: the restored runtime differs from the snapshot '
                                       f'({start["failed"]}, live observation {live})')
                # the restore is verified above; only now is the arm (and with it the intended noise key) applied
                calls0 = r.policy.inference_calls; sec0 = r.policy.inference_seconds
                h.rows = rows = []; h.frames = frames = [] if keep_frames else None; h.calls = calls = []
                h.prefix = prefix = [] if FORCE[arm] is None else None; h.prefix_n = qlen
                apply_arm(r, arm, rep); q = len(r.policy.queue); res = r.complete(); h.prefix = None
                cell = {'experimental_arm': arm, 'arm': SHORT[arm], 'block_repeat_id': rep, 'policy_noise_repeat_id': noise_id(arm, rep),
                        'noise_override': arm == 'continue_alt_noise', 'y': int(res['success']), 'steps': res['steps'], 'reason': res['reason'],
                        'success_step': res['steps'] if res['success'] else None, 'intervention': res['intervention'],
                        'intervention_step': res['intervention_step'], 'queue_len_after_choice': q,
                        'policy_calls_after_snapshot': res['policy_calls'] - calls0,
                        'policy_seconds_after_snapshot': res['policy_inference_seconds'] - sec0,
                        'first_inference_step': calls[0]['t'] if calls else None, 'inference_calls': calls, 'retreat': res['retreat'],
                        'prefix_replay_state_error': err, 'live_observation': live,
                        'stage': stage_of(list(trunk_rows[:t]) + rows), 'wall_seconds': time.perf_counter() - tick}
                return cell, rows, frames, prefix

            for rep in repeats:
                cells = {}; pre = {}
                for arm in order:
                    want = bool(video and frame) and t == first_snapshot and rep == 0
                    cell, rows, frames, prefix = branch(rep, arm, want); cells[arm] = cell; pre[arm] = prefix
                    if probe: traces[f'b_{t}_{rep}_{SHORT[arm]}'] = rows_array(rows)
                    if want: videos[f'scene{seed}_t{t}_r{rep}_{arm}'] = list(trunk_frames[:t + 1]) + frames
                    if FORCE[arm] is None: refs[(rep, arm)] = (_ref(r), cell['inference_calls'])
                    if arm == 'continue' and rep == 0: duplicates.append({'checkpoint': t, 'repeat': 0, 'against': 'trunk', **_same_run(trunk_ref, r, tol)})
                c0, c1 = cells['continue'], cells['continue_alt_noise']
                par = prefix_parity(pre['continue'], pre['continue_alt_noise'], c0['inference_calls'], c1['inference_calls'], t, qlen, c0['steps'], c1['steps'], tol)
                if not par['pass']:
                    raise RuntimeError(f'scene {seed} checkpoint {t} repeat {rep}: Continue-alt-noise is not the same rollout as Continue '
                                       f'while the old queue is executed ({jsonable(par)})')
                blocks.append({'scene': int(seed), 'checkpoint': t, 'repeat': rep, 'block_repeat_id': rep, 'role': ROLE, 'snapshot_id': info['snapshot_id'],
                               **{'y_' + SHORT[a]: cells[a]['y'] for a in ARMS}, 'cells': [cells[a] for a in ARMS], 'c0_c1_queue_prefix_parity': par})
            if recheck:      # after every other branch of this snapshot: C1 of repeat 0 and C0 of repeat 1 once more
                for rep, arm in ((0, 'continue_alt_noise'), (1, 'continue')):
                    if (rep, arm) not in refs: continue
                    cell, _, _, _ = branch(rep, arm); same = _same_run(refs[(rep, arm)][0], r, tol); calls_equal = cell['inference_calls'] == refs[(rep, arm)][1]
                    rechecks.append({'checkpoint': t, 'repeat': rep, 'experimental_arm': arm, 'against': 'first run of the same arm in the same block',
                                     **same, 'inference_steps_seeds_chunks_equal': bool(calls_equal), 'match': bool(same['match'] and calls_equal)})
    ok = all(x['match'] for x in duplicates + rechecks)
    record = {'scene': int(seed), 'role': ROLE, 'planned_checkpoints': checkpoints, 'repeats': repeats, 'arms': list(ARMS), 'initial_state': initial, 'trunk': trunk,
              'snapshots': [snaps[t]['info'] for t in sorted(snaps)], 'skips': sorted(skips, key=lambda x: x['checkpoint']), 'blocks': blocks,
              'no_intervention_duplicates': duplicates, 'rechecks': rechecks, 'reproducibility_ok': ok,
              'counts': {'snapshots': len(snaps), 'blocks': len(blocks), 'continuations': len(ARMS) * len(blocks),
                         'extra_reproducibility_continuations': len(rechecks)},
              'wall_seconds': time.perf_counter() - started, 'completed': True}
    record = jsonable(record); validate_scene(record, c)
    return record, traces, videos, inputs


def validate_scene(rec, c=None):
    """Accounting of one scene record: nothing missing, nothing twice, the control arm neither an intervention nor a new block."""
    cps = rec['planned_checkpoints']; reps = rec['repeats']; snap = {x['checkpoint']: x for x in rec['snapshots']}; skip = [x['checkpoint'] for x in rec['skips']]
    if len(skip) != len(set(skip)) or set(snap) & set(skip) or set(snap) | set(skip) != set(cps):
        raise ValueError(f'scene {rec["scene"]}: planned checkpoints are not each a snapshot or a listed skip')
    keys = [(b['checkpoint'], b['repeat']) for b in rec['blocks']]
    if len(keys) != len(set(keys)) or set(keys) != {(t, x) for t in snap for x in reps}:
        raise ValueError(f'scene {rec["scene"]}: blocks do not match snapshots x block repeats (the control arm must not create blocks)')
    for b in rec['blocks']:
        cells = b['cells']; t = b['checkpoint']; rep = b['repeat']; q = snap[t]['queue_len']
        if b['scene'] != rec['scene'] or b['role'] != rec['role'] or b['block_repeat_id'] != rep: raise ValueError('block scene/role/repeat differs from its record')
        if [x['experimental_arm'] for x in cells] != list(ARMS): raise ValueError('a block needs exactly the four arms, once each')
        for x in cells:
            arm = x['experimental_arm']; nid = noise_id(arm, rep)
            if x['block_repeat_id'] != rep or x['policy_noise_repeat_id'] != nid or x['noise_override'] != (arm == 'continue_alt_noise'):
                raise ValueError('block repeat and policy noise repeat are not recorded as planned')
            if any(k['policy_noise_repeat_id'] != nid for k in x['inference_calls']): raise ValueError('an inference call used another noise key than its arm')
            if len(x['inference_calls']) != x['policy_calls_after_snapshot']: raise ValueError('inference log does not match the call counter')
            if b['y_' + SHORT[arm]] != x['y']: raise ValueError('joined outcome differs from the cell')
            if c is not None and x['steps'] > c['env']['horizon']: raise ValueError('a branch exceeded the episode horizon')
        for x in cells[:2]:
            if x['intervention'] is not None or x['intervention_step'] is not None or x['retreat'] is not None or x['queue_len_after_choice'] != q:
                raise ValueError('a Continue arm changed the queue or was recorded as an intervention')
            if x['first_inference_step'] not in (None, t + q): raise ValueError('a Continue arm inferred before the natural chunk boundary')
        for x in cells[2:]:
            if x['intervention'] != FORCE[x['experimental_arm']] or x['intervention_step'] != t or x['queue_len_after_choice'] != 0:
                raise ValueError('an intervention was not applied at the checkpoint with a cleared queue')
        if cells[2]['retreat'] is not None or cells[3]['retreat'] is None: raise ValueError('retreat record attached to the wrong arm')
        if cells[2]['first_inference_step'] != t: raise ValueError('Refresh did not infer at the checkpoint')
        if not b['c0_c1_queue_prefix_parity']['pass']: raise ValueError('C0/C1 queue-prefix parity failed')
    n = rec['counts']
    if (n['snapshots'], n['blocks'], n['continuations']) != (len(snap), len(keys), len(ARMS) * len(keys)): raise ValueError('counts do not match the record')
    return True


def repro_problems(rec, rerun):
    """Reproducibility is read from the recorded checks themselves (count, identity, match), not from a stored flag."""
    snaps = sorted(x['checkpoint'] for x in rec['snapshots']); out = []
    dup = sorted((d['checkpoint'], d['repeat'], d['against']) for d in rec['no_intervention_duplicates'])
    if dup != [(t, 0, 'trunk') for t in snaps]: out.append('trunk-duplicate checks are not one per snapshot')
    got = sorted((d['checkpoint'], d['repeat'], d['experimental_arm']) for d in rec['rechecks'])
    want = sorted((t, r, a) for t in snaps for r, a in ((0, 'continue_alt_noise'), (1, 'continue'))) if rerun else []
    if got != want: out.append('rerun checks are not the declared ones' if rerun else 'rerun checks recorded for a scene that has none planned')
    bad = sum(1 for d in rec['no_intervention_duplicates'] + rec['rechecks'] if d.get('match') is not True)
    if bad: out.append(f'{bad} reproducibility checks did not match')
    if bool(rec.get('reproducibility_ok')) != (not bad): out.append('stored reproducibility flag contradicts the recorded checks')
    return out


def finalize(out, seeds, c=None, rerun_scenes=()):
    """dataset.json. completed is true only when every scheduled scene has a valid record, no error file exists and
    every declared reproducibility check is present and matched. `rerun_scenes`: scenes that must carry the reruns."""
    out = Path(out); recs = []; problems = []; rerun_scenes = {int(s) for s in rerun_scenes}; repro = True
    for seed in seeds:
        p = out / 'scenes' / f'{int(seed)}.json'
        if (out / 'scenes' / f'{int(seed)}.error.json').exists(): problems.append(f'scene {seed}: error file present')
        if not p.exists(): problems.append(f'scene {seed}: no record'); continue
        rec = read_json(p)
        try: validate_scene(rec, c)
        except (ValueError, KeyError) as e: problems.append(f'scene {seed}: {e!r}'); continue
        if not rec.get('completed'): problems.append(f'scene {seed}: not completed'); continue
        recs.append(rec); why = repro_problems(rec, int(seed) in rerun_scenes)
        if why: repro = False; problems.append(f'scene {seed}: {"; ".join(why)} (rows are kept as measured, interpretation is withheld)')
    extra = sorted(int(p.name.split('.')[0]) for p in (out / 'scenes').glob('*.json') if int(p.name.split('.')[0]) not in {int(s) for s in seeds})
    if extra: problems.append(f'records of unscheduled scenes: {extra}')
    meta = {'role': ROLE, 'arms': list(ARMS), 'scheduled_scenes': [int(s) for s in seeds], 'recorded_scenes': [x['scene'] for x in recs], 'completed': not problems,
            'problems': problems, 'snapshots': sum(x['counts']['snapshots'] for x in recs), 'blocks': sum(x['counts']['blocks'] for x in recs),
            'continuations': sum(x['counts']['continuations'] for x in recs),
            'extra_reproducibility_continuations': sum(x['counts']['extra_reproducibility_continuations'] for x in recs),
            'reproducibility_ok': repro if recs else None,
            'wall_seconds_sum_over_scenes': sum(x['wall_seconds'] for x in recs)}
    return meta, recs


# ---------------------------------------------------------------- descriptive tables (no interval, no p-value)
def cls(y_ref, y_arm):
    return f'{int(y_ref)}to{int(y_arm)}'


def block_rows(records):
    """One row per block: the four outcomes side by side and every pair classified against its reference arm."""
    rows = []
    for rec in records:
        snap = {x['checkpoint']: x for x in rec['snapshots']}
        for b in rec['blocks']:
            s = snap[b['checkpoint']]; at = s.get('state_at_checkpoint') or {}; cell = {x['arm']: x for x in b['cells']}; rt = cell['R']['retreat'] or {}
            row = {'scene': b['scene'], 'role': b['role'], 'checkpoint': b['checkpoint'], 'block_repeat_id': b['block_repeat_id'], 'snapshot_id': b['snapshot_id'],
                   **{f'y_{k}': b[f'y_{k}'] for k in SHORT.values()},
                   **{f'{a}_vs_{ref}': cls(b[f'y_{ref}'], b[f'y_{a}']) for a, ref in PAIRS},
                   'policy_noise_repeat_id_C1': cell['C1']['policy_noise_repeat_id'],
                   **{f'steps_{k}': cell[k]['steps'] for k in cell}, **{f'reason_{k}': cell[k]['reason'] for k in cell},
                   **{f'policy_calls_{k}': cell[k]['policy_calls_after_snapshot'] for k in cell},
                   **{f'first_inference_step_{k}': cell[k]['first_inference_step'] for k in cell},
                   **{f'stage_{k}': (cell[k]['stage'] or {}).get('furthest_stage') for k in cell},
                   'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'], 'queue_len_at_checkpoint': s['queue_len'],
                   'remaining_budget': s['remaining_budget'], 'stage_so_far_at_checkpoint': s.get('stage_so_far'), 'grasped_at_checkpoint': at.get('is_cubeA_grasped'),
                   'retreat_steps_used': rt.get('steps_used'), 'retreat_achieved_m': rt.get('achieved_displacement_m'), 'retreat_target_error_m': rt.get('target_error_m'),
                   'retreat_reached': rt.get('reached'), 'retreat_end_reason': rt.get('end_reason'), 'c0_c1_prefix_parity': b['c0_c1_queue_prefix_parity']['pass']}
            rows.append(row)
    return rows


def pair_table(rows, a, ref):
    n = {'0to1': 0, '1to0': 0, '1to1': 0, '0to0': 0}; k = f'{a}_vs_{ref}'
    for x in rows: n[x[k]] += 1
    blocks = len(rows); ok = n['1to0'] + n['1to1']; bad = n['0to1'] + n['0to0']
    return {'arm': a, 'reference': ref, 'blocks': blocks, **n, 'reference_success_blocks': ok, 'reference_failure_blocks': bad,
            'rate_0to1': n['0to1'] / bad if bad else 'NA', 'rate_1to0': n['1to0'] / ok if ok else 'NA',
            'delta_mean_outcome': (n['0to1'] - n['1to0']) / blocks if blocks else 'NA',
            'scenes': len({x['scene'] for x in rows}), 'scenes_with_0to1': len({x['scene'] for x in rows if x[k] == '0to1'}),
            'scenes_with_1to0': len({x['scene'] for x in rows if x[k] == '1to0'})}


def pair_rows(rows):
    cps = sorted({x['checkpoint'] for x in rows}); out = []
    for a, ref in PAIRS:
        out.append({'checkpoint': 'all', **pair_table(rows, a, ref)})
        out += [{'checkpoint': t, **pair_table([x for x in rows if x['checkpoint'] == t], a, ref)} for t in cps]
    return out


def scene_rows(records, rows=None):
    """Scene-equal weighting: within a scene the mean over its reached checkpoints and the block repeats."""
    rows = block_rows(records) if rows is None else rows; out = []
    for rec in records:
        mine = [x for x in rows if x['scene'] == rec['scene']]; m = rec['counts']['snapshots']; n = len(mine)
        row = {'scene': rec['scene'], 'role': rec['role'], 'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'],
               'trunk_stage': (rec['trunk']['stage'] or {}).get('furthest_stage'), 'reached_checkpoints_m': m,
               'checkpoints_used': ' '.join(str(x['checkpoint']) for x in rec['snapshots']), 'blocks': n, 'reproducibility_ok': rec['reproducibility_ok']}
        for k in SHORT.values(): row[f'mean_y_{k}'] = sum(x[f'y_{k}'] for x in mine) / n if n else 'NA'
        for a, ref in PAIRS: row[f'd_{a}_{ref}'] = sum(x[f'y_{a}'] - x[f'y_{ref}'] for x in mine) / n if n else 'NA'
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
        at = [x for x in rows if x['checkpoint'] == t]; scenes = sorted({x['scene'] for x in at}); row = {'scenes': len(scenes)}
        for a, ref in PAIRS:
            d = [np.mean([x[f'y_{a}'] - x[f'y_{ref}'] for x in at if x['scene'] == s]) for s in scenes]
            row[f'd_{a}_{ref}'] = {'mean_over_scenes': float(np.mean(d)), 'scenes_positive': sum(x > 0 for x in d), 'scenes_negative': sum(x < 0 for x in d),
                                   'scenes_zero': sum(x == 0 for x in d)}
        out[f't{t}'] = row
    return out


def complementarity_rows(rows):
    """What Retreat-Retry adds to or takes from Refresh, with either Continue as the reference. Counts, not a selector."""
    out = []; cps = ['all'] + sorted({x['checkpoint'] for x in rows})

    def add(scope, name, sel):
        hit = [x for x in sel]; per = {}
        for x in hit: per[x['snapshot_id']] = per.get(x['snapshot_id'], 0) + 1
        out.append({'checkpoint': scope, 'pattern': name, 'blocks': len(hit), 'scenes': len({x['scene'] for x in hit}), 'snapshots': len(per),
                    **{f'snapshots_with_{k}_repeats': sum(1 for v in per.values() if v == k) for k in (1, 2, 3, 4)}})
    for scope in cps:
        base = rows if scope == 'all' else [x for x in rows if x['checkpoint'] == scope]
        for f in (0, 1):
            for r in (0, 1):
                add(scope, f'F={f},R={r}', [x for x in base if (x['y_F'], x['y_R']) == (f, r)])
        for ref in ('C0', 'C1'):
            for y in (0, 1):
                for f in (0, 1):
                    for r in (0, 1):
                        add(scope, f'{ref}={y},F={f},R={r}', [x for x in base if (x[f'y_{ref}'], x['y_F'], x['y_R']) == (y, f, r)])
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


def parity_rows(records):
    return [{'scene': b['scene'], 'checkpoint': b['checkpoint'], 'block_repeat_id': b['block_repeat_id'], **b['c0_c1_queue_prefix_parity']}
            for rec in records for b in rec['blocks']]


def skip_rows(records):
    return [{'scene': rec['scene'], 'checkpoint': x['checkpoint'], 'reason': x['reason'], 'detail': '; '.join(x['detail'] or []),
             'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'], 'cells_not_run': len(ARMS) * len(rec['repeats'])}
            for rec in records for x in rec['skips']]


def summarize(records, planned_scenes=None):
    rows = block_rows(records); scenes = scene_rows(records, rows); skips = skip_rows(records); pairs = pair_rows(rows)
    stat = lambda v: {'n': len(v), 'mean': float(np.mean(v)), 'min': float(np.min(v)), 'median': float(np.median(v)), 'max': float(np.max(v))} if v else {'n': 0}
    cost = {}
    for k in SHORT.values():
        cost[k] = {'success_blocks': sum(x[f'y_{k}'] for x in rows), 'blocks': len(rows),
                   'steps_after_checkpoint': stat([x[f'steps_{k}'] - x['checkpoint'] for x in rows]),
                   'steps_after_checkpoint_when_successful': stat([x[f'steps_{k}'] - x['checkpoint'] for x in rows if x[f'y_{k}']]),
                   'policy_calls_after_checkpoint': stat([x[f'policy_calls_{k}'] for x in rows])}
    by_stage = {}
    for x in rows:
        d = by_stage.setdefault(str(x['stage_so_far_at_checkpoint']), {'blocks': 0, **{f'success_{k}': 0 for k in SHORT.values()}, 'scenes': set()})
        d['blocks'] += 1; d['scenes'].add(x['scene'])
        for k in SHORT.values(): d[f'success_{k}'] += x[f'y_{k}']
    for d in by_stage.values(): d['scenes'] = len(d['scenes'])
    num = lambda k: [x[k] for x in rows if x[k] is not None]; reasons = {}
    for x in rows: reasons[str(x['retreat_end_reason'])] = reasons.get(str(x['retreat_end_reason']), 0) + 1
    skip_counts = {}
    for x in skips: skip_counts[x['reason']] = skip_counts.get(x['reason'], 0) + 1
    dup = [d for rec in records for d in rec['no_intervention_duplicates']]; rc = [d for rec in records for d in rec['rechecks']]
    par = parity_rows(records); spread = {}
    for rec in records:
        for t in {b['checkpoint'] for b in rec['blocks']}:
            ys = {b['y_C0'] for b in rec['blocks'] if b['checkpoint'] == t}; spread[len(ys)] = spread.get(len(ys), 0) + 1
    return {'role': ROLE, 'arms': dict(SHORT), 'scenes_planned': planned_scenes, 'scenes_recorded': len(records),
            'trunk_success_scenes': sum(r['trunk']['success'] for r in records), 'trunk_failure_scenes': sum(1 - r['trunk']['success'] for r in records),
            'scenes_with_a_snapshot': sum(1 for r in records if r['snapshots']), 'snapshots': sum(r['counts']['snapshots'] for r in records),
            'blocks': len(rows), 'continuations': sum(r['counts']['continuations'] for r in records),
            'extra_reproducibility_continuations': sum(r['counts']['extra_reproducibility_continuations'] for r in records),
            'skipped_checkpoints': skip_counts, 'cells_not_run': sum(x['cells_not_run'] for x in skips),
            'success_blocks_by_arm': {k: sum(x[f'y_{k}'] for x in rows) for k in SHORT.values()},
            'pairs': pairs, 'scene_equal_weight': scene_equal_summary(scenes), 'scene_equal_weight_by_checkpoint': scene_equal_by_checkpoint(rows),
            'by_stage_at_checkpoint': by_stage, 'time_and_inference_by_arm': cost,
            'snapshots_by_number_of_distinct_C0_outcomes_over_repeats': {str(k): v for k, v in sorted(spread.items())},
            'retreat': {'executed': len(rows), 'reached_target': sum(1 for x in rows if x['retreat_reached']), 'end_reasons': reasons,
                        'steps_used': stat(num('retreat_steps_used')), 'achieved_displacement_m': stat(num('retreat_achieved_m')),
                        'target_error_m': stat(num('retreat_target_error_m'))},
            'c0_c1_queue_prefix_parity': {'blocks': len(par), 'pass': sum(1 for x in par if x['pass']),
                                          'blocks_with_a_later_inference': sum(1 for x in par if x['first_inference_step_C0'] is not None),
                                          'first_inference_seeds_differ': sum(1 for x in par if x['first_inference_seeds_differ']),
                                          'blocks_ended_while_queue_was_executed': sum(1 for x in par if x['first_inference_step_C0'] is None),
                                          'max_action_abs': max([x['action_max_abs'] for x in par] + [0.]), 'max_state_abs': max([x['state_max_abs'] for x in par] + [0.]),
                                          'max_live_observation_state_abs': max([x['live_observation_state_max_abs'] for x in par] + [0.]),
                                          'max_live_observation_rgb_abs': max([x['live_observation_rgb_max_abs'] for x in par] + [0.])},
            'reproducibility': {'no_intervention_duplicates': len(dup), 'duplicate_mismatches': sum(1 for d in dup if not d['match']),
                                'rechecks': len(rc), 'recheck_mismatches': sum(1 for d in rc if not d['match']),
                                'rechecks_by_arm': {a: sum(1 for d in rc if d['experimental_arm'] == a) for a in ('continue', 'continue_alt_noise')},
                                'interpretation_withheld': any(not d['match'] for d in dup + rc)},
            'terms': {'0to1 / 1to0': 'reference arm outcome -> compared arm outcome within one block',
                      'F or R against C0 or C1': '0to1 is a rescue and 1to0 a harm RELATIVE TO THAT REFERENCE under the declared noise coupling',
                      'C1 against C0': 'noise outcome switch: the same Continue rule under another future noise realisation; not a rescue, not a harm, '
                                       'not a safety violation of the baseline',
                      'delta_mean_outcome': 'mean(y_arm - y_reference) over blocks. delta_F_C0 - delta_C1_C0 equals delta_F_C1 algebraically; it is not a '
                                            'noise-corrected physical effect',
                      'NA': 'the denominator is zero; this is not 0%'},
            'scope': ['Development sample designed after the Stage 2A results were seen; not a pre-registered confirmatory test and not an independent validation.',
                      'At most 24 independent root scenes. Blocks of one scene share the scene and the past. No interval or p-value is computed.',
                      'The fourth arm controls for resampling variability at the natural replanning step. It does not separate the components of Refresh '
                      '(new observation, discarded chunk, early inference, other noise draws).',
                      'A forced choice at a fixed checkpoint is not the value or the harm rate H(g) of an online intervention policy.',
                      'A per-block maximum over realised outcomes is not an executable selector.',
                      'Zero observed harm would be zero events in this sample, not a safety statement.',
                      'Stage 2A and Stage 2B use different scenes and are reported separately; they are not pooled into a confirmatory sample.']}
