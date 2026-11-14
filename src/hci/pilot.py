"""Stage 2A exploratory recovery pilot (development only).

Three things live here, all independent of ManiSkill so that the control flow can be tested with mock physics:
  * a restore audit that compares the COMPLETE runtime state (simulator, counters, policy history, unexecuted action
    queue, RNG, completion flags, retreat record) and the observation re-read from the environment;
  * a per-scene collector of paired blocks (scene, checkpoint, repeat) x {continue, refresh, retreat_retry};
  * descriptive tables. Branch repeats are not independent episodes; nothing here computes an interval or a p-value.

Ground-truth object state enters only through `probe` (logging). It is never given to the policy.
"""
from __future__ import annotations
import hashlib, json, pickle, random, time, traceback
from pathlib import Path
import numpy as np
from .trees import clone, as_numpy, maxdiff
from .schema import STATE_PATHS
from .simulation import RNG_ATTRS, rng_state
from .io import write_json, read_json

ACTIONS = ('continue', 'refresh', 'retreat_retry')
ROLE = 'stage2a_exploratory_dev'
CLASSES = {(0, 0): 'both_fail', (0, 1): 'rescue', (1, 0): 'harm', (1, 1): 'both_succeed'}
TRACE_COLUMNS = ('t', 'tcp_x', 'tcp_y', 'tcp_z', 'cubeA_x', 'cubeA_y', 'cubeA_z', 'cubeB_x', 'cubeB_y', 'cubeB_z',
                 'is_cubeA_grasped', 'is_cubeA_on_cubeB', 'is_cubeA_static', 'success', 'a_x', 'a_y', 'a_z', 'a_gripper')


# ---------------------------------------------------------------- full runtime state and its comparison
def _sha(obj):
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


def rng_digests(env):
    import torch
    out = {'python': _sha(random.getstate()), 'numpy': _sha(np.random.get_state()),
           'torch': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
           'cuda': _sha([x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all()]) if torch.cuda.is_available() else None}
    for a in RNG_ATTRS:
        if hasattr(env, a): out['env' + a] = _sha(rng_state(getattr(env, a)))
    return out


def jsonable(x):
    """Non-finite differences (e.g. a queue of another length) are written as text; write_json refuses NaN/inf."""
    if isinstance(x, dict): return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)) and not np.isfinite(x): return str(float(x))
    return x


def tree_hash(x):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, dict):
            for k in sorted(v): h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for y in v: visit(y)
        else: h.update(np.ascontiguousarray(as_numpy(v)).tobytes())
    visit(x); return h.hexdigest()


def runtime_tree(r):
    """Everything that decides the future of a rollout, read from the live objects (not from a snapshot)."""
    p = r.policy; hist = list(p.history)
    split = bool(hist) and isinstance(hist[0], dict) and 'rgb' in hist[0]
    return {'env_state': clone(r.env.get_state_dict()), 'elapsed': clone(r.env._elapsed_steps),
            'history_state': [clone(h['state']) for h in hist] if split else clone(hist),
            'history_rgb': [clone(h['rgb']) for h in hist] if split else [],
            'queue': [np.asarray(q, np.float32).copy() for q in p.queue],
            'last_action': None if p.last_action is None else np.asarray(p.last_action, np.float32).copy(),
            'actions': [np.asarray(a, np.float32).copy() for a in r.actions],
            'tcp_history': [np.asarray(x, float).copy() for x in r.tcp_history],
            'discrete': {'t': int(r.t), 'done': bool(r.done), 'y': int(r.y), 'reason': r.reason, 'repeat': int(r.repeat),
                         'intervention': r.intervention, 'intervention_t': r.intervention_t,
                         'queue_len': len(p.queue), 'history_len': len(hist), 'n_actions': len(r.actions),
                         'inference_calls': int(p.inference_calls), 'retreat_info': clone(r.retreat_info), 'rng': rng_digests(r.env)}}


NUMERIC = {'env_state': 'state_atol', 'elapsed': None, 'history_state': 'state_atol', 'history_rgb': 'observation_atol',
           'queue': 'action_atol', 'last_action': 'action_atol', 'actions': 'action_atol', 'tcp_history': 'state_atol'}


def _diff(a, b):
    if a is None or b is None: return 0. if a is None and b is None else float('inf')
    try: return maxdiff(a, b)
    except ValueError: return float('inf')      # keys, lengths or shapes changed (e.g. a queue of another length)


def compare_tree(a, b, tol):
    """Existing audit tolerances only: state_atol, action_atol, observation_atol (RGB, uint8 units); counters are exact."""
    numeric = {k: _diff(a[k], b[k]) for k in NUMERIC}
    discrete = {k: a['discrete'][k] == b['discrete'][k] for k in a['discrete']}
    failed = [k for k, v in numeric.items() if v > (0. if NUMERIC[k] is None else tol[NUMERIC[k]])] + [k for k, v in discrete.items() if not v]
    return {'numeric_max_abs': numeric, 'discrete_equal': discrete, 'failed': failed, 'pass': not failed}


def obs_diff(live, cached):
    """Observation returned by the environment at the restored step against the snapshot's cached observation."""
    state = max([_diff(live[p][k], cached[p][k]) for p, k in STATE_PATHS if p in cached and k in cached[p]] + [0.])
    rgb = max([_diff(live['sensor_data'][n]['rgb'], cached['sensor_data'][n]['rgb']) for n in sorted(cached.get('sensor_data', {}))] + [0.])
    return {'state_max_abs': state, 'rgb_max_abs': rgb}


def live_obs_ok(d, tol):
    return d['state_max_abs'] <= tol['state_atol'] and d['rgb_max_abs'] <= tol['observation_atol']


# ---------------------------------------------------------------- restore audit
def _trace(r, steps):
    out = []
    for _ in range(steps):
        if r.done: break
        r.step_base(); out.append(runtime_tree(r))
    return out


def _compare_traces(a, b, tol):
    if len(a) != len(b): return {'pass': False, 'failed': ['trace_length'], 'steps': [len(a), len(b)]}
    worst = {k: 0. for k in NUMERIC}; failed = set()
    for x, y in zip(a, b):
        c = compare_tree(x, y, tol); failed.update(c['failed'])
        for k, v in c['numeric_max_abs'].items(): worst[k] = max(worst[k], v)
    return {'pass': not failed, 'failed': sorted(failed), 'steps': [len(a), len(b)], 'numeric_max_abs': worst}


def audit_scene(r, seed, c, checkpoints):
    """At every reached checkpoint: continue, restore and repeat; branch (retreat, refresh), restore and repeat.

    The restored runtime is compared with the state recorded when the snapshot was taken, the observation is the
    one the environment returned after the prefix replay, and the continuation is compared step by step.
    """
    tol = c['audit']; n = tol['replay_steps']; r.reset(seed); out = []
    for t in checkpoints:
        while not r.done and r.t < t: r.step_base()
        if r.done:
            out.append({'checkpoint': t, 'reached': False, 'reason': f'trunk ended at step {r.t} ({r.reason})'}); continue
        elig = [bool(x) for x in r.eligible()]; s = r.snapshot(); f0 = runtime_tree(r); first = _trace(r, n); checks = {}

        def back(label):
            r.restore(s); live = obs_diff(r.replay_raw, s['raw'])
            checks[label + '_restored'] = {**compare_tree(f0, runtime_tree(r), tol), 'prefix_replay_state_error': r.replay_error,
                                           'live_observation': live, 'live_observation_pass': live_obs_ok(live, tol)}
            checks[label + '_trace'] = _compare_traces(first, _trace(r, n), tol)
        back('repeat'); branches = {}
        for a in (2, 1):
            r.restore(s); now = bool(r.eligible()[a])
            if now != elig[a]:      # a wrong restore can change eligibility itself; that is a failure, not a skipped branch
                branches[ACTIONS[a]] = {'executed': False, 'reason': 'eligibility after restore differs from the snapshot', 'pass': False}; r.complete()
            elif now:
                r.force(a); res = r.complete()
                branches[ACTIONS[a]] = {'executed': True, 'success': res['success'], 'steps': res['steps'], 'retreat': res['retreat']}
            else:
                branches[ACTIONS[a]] = {'executed': False, 'reason': 'not eligible at this checkpoint'}; r.complete()
            back('after_' + ACTIONS[a])
        r.restore(s)
        ok = (all(v['pass'] for v in checks.values()) and all(v.get('live_observation_pass', True) for v in checks.values())
              and all(v.get('pass', True) for v in branches.values()))
        out.append({'checkpoint': t, 'reached': True, 'eligible': elig, 'trace_steps': len(first), 'branches': branches, 'checks': checks, 'pass': ok})
    reached = [x for x in out if x['reached']]
    return jsonable({'seed': int(seed), 'checkpoints': out, 'reached': len(reached), 'pass': bool(reached) and all(x['pass'] for x in reached),
                     **({} if reached else {'reason': 'no checkpoint was reached'})})


def audit_report(scenes):
    did = lambda name: sum(1 for s in scenes for x in s['checkpoints'] if x['reached'] and x['branches'][name]['executed'])
    n = {'retreat_retry': did('retreat_retry'), 'refresh': did('refresh')}
    return {'pass': bool(scenes) and all(s['pass'] for s in scenes) and min(n.values()) > 0, 'branches_executed': n, 'results': scenes,
            'compared': {'numeric': dict(NUMERIC), 'exact': ['t', 'done', 'y', 'reason', 'repeat', 'intervention', 'intervention_t', 'queue_len',
                                                              'history_len', 'n_actions', 'inference_calls', 'retreat_info', 'rng'],
                         'not_compared': ['policy.inference_seconds (wall-clock time)']}}


# ---------------------------------------------------------------- paired branches of one scene
class Recorder:
    """Wraps Runtime.advance for logging. The prefix replay inside restore does not pass through advance."""
    def __init__(self, r, probe=None, frame=None):
        self.r = r; self.probe = probe; self.frame = frame; self.rows = None; self.frames = None

    def __enter__(self):
        base = type(self.r).advance

        def advance(action):
            z = base(self.r, action)
            if self.rows is not None and self.probe: self.rows.append(self.probe(self.r, z))
            if self.frames is not None and self.frame: self.frames.append(self.frame(self.r))
            return z
        self.r.advance = advance; return self

    def __exit__(self, *exc):
        del self.r.advance


def ineligibility(r):
    p = r.c['protocol']; e = r.eligible(); rem = len(r.policy.queue); why = []
    if r.done: why.append('episode_already_ended')
    if r.intervention is not None: why.append('already_intervened')
    if r.t < p['min_intervention_step']: why.append('before_min_intervention_step')
    if r.t % p['decision_stride']: why.append('off_decision_stride')
    if not 0 < rem < r.policy.act_horizon: why.append('no_unexecuted_chunk_remaining' if rem == 0 else 'chunk_not_started')
    if e[1] and not e[2]: why.append('retreat_budget_exceeds_remaining_horizon')
    return why


def rows_array(rows):
    out = np.zeros((len(rows), len(TRACE_COLUMNS)), np.float32)
    for i, x in enumerate(rows):
        out[i] = [x['t'], *x['tcp'], *x['cubeA'], *x['cubeB'], x['is_cubeA_grasped'], x['is_cubeA_on_cubeB'], x['is_cubeA_static'],
                  x['success'], *(x['action'] or [np.nan] * 4)]
    return out


def _same_run(ref, r, tol):
    """A continuation that should be identical to `ref` = (actions, y, steps, reason, final state)."""
    acts, y, steps, reason, state = ref; mine = [np.asarray(a, np.float32) for a in r.actions]
    d = {'action_max_abs': _diff(acts, mine), 'final_state_max_abs': _diff(state, clone(r.env.get_state_dict())),
         'outcome_equal': (y, steps, reason) == (r.y, r.t, r.reason)}
    d['match'] = bool(d['outcome_equal'] and d['action_max_abs'] <= tol['action_atol'] and d['final_state_max_abs'] <= tol['state_atol'])
    return d


def _ref(r):
    return ([np.asarray(a, np.float32).copy() for a in r.actions], r.y, r.t, r.reason, clone(r.env.get_state_dict()))


def collect_scene(r, seed, c, checkpoints, repeats, probe=None, frame=None, stage=None, video=False, recheck_repeat=None):
    """Trunk (repeat 0, never restored) first, then every (checkpoint, repeat) block with the three choices.

    A snapshot is taken at every planned checkpoint where all three choices are eligible, whatever happens later;
    every other planned checkpoint is listed with its reason. Returns (record, traces, videos).
    """
    tol = c['audit']; started = time.perf_counter(); checkpoints = [int(t) for t in checkpoints]; repeats = [int(x) for x in repeats]
    r.reset(seed); initial = probe(r, None) if probe else None; snaps = {}; skips = []; traces = {}; videos = {}
    stage_of = (lambda rows: stage(initial, rows)) if (stage and probe) else (lambda rows: None)
    with Recorder(r, probe, frame) as rec:
        rec.rows = trunk_rows = []; rec.frames = trunk_frames = [frame(r)] if (video and frame) else None
        while not r.done:
            if r.t in checkpoints:
                if r.eligible().all():
                    snaps[r.t] = {'s': r.snapshot(), 'tree': runtime_tree(r),
                                  'info': {'checkpoint': r.t, 'queue_len': len(r.policy.queue), 'history_len': len(r.policy.history),
                                           'remaining_budget': r.horizon - r.t, 'eligible': [True, True, True],
                                           'policy_calls_so_far': r.policy.inference_calls, 'state_sha256': tree_hash(r.env.get_state_dict()),
                                           'state_at_checkpoint': trunk_rows[-1] if trunk_rows else initial,
                                           'stage_so_far': (stage_of(list(trunk_rows)) or {}).get('furthest_stage')}}
                else:
                    skips.append({'checkpoint': r.t, 'reason': 'not_all_three_choices_eligible', 'detail': ineligibility(r),
                                  'eligible': [bool(x) for x in r.eligible()]})
            r.step_base()
        trunk = {'policy_noise_key': '(scene, physical_step, repeat=0)', 'success': r.y, 'steps': r.t, 'reason': r.reason,
                 'policy_calls': r.policy.inference_calls, 'policy_inference_seconds': r.policy.inference_seconds, 'stage': stage_of(trunk_rows)}
        trunk_ref = _ref(r)
        for t in checkpoints:
            if t not in snaps and not any(x['checkpoint'] == t for x in skips):
                skips.append({'checkpoint': t, 'reason': 'trunk_succeeded_before_checkpoint' if r.y else 'trunk_ended_before_checkpoint',
                              'detail': [f'trunk ended at step {r.t} ({r.reason})'], 'eligible': None})
        if probe: traces['trunk'] = rows_array(trunk_rows)
        blocks = []; duplicates = []; rechecks = []; first_snapshot = min(snaps) if snaps else None
        for t in sorted(snaps):
            s = snaps[t]['s']; continue_refs = {}

            def branch(rep, a, keep_frames=False):
                tick = time.perf_counter(); r.restore(s); err = r.replay_error
                start = compare_tree(snaps[t]['tree'], runtime_tree(r), tol); live = obs_diff(r.replay_raw, s['raw'])
                if not start['pass'] or not live_obs_ok(live, tol):
                    raise RuntimeError(f'scene {seed} checkpoint {t}: the restored runtime differs from the snapshot '
                                       f'({start["failed"]}, live observation {live})')
                r.repeat = rep; calls0 = r.policy.inference_calls; sec0 = r.policy.inference_seconds
                rec.rows = rows = []; rec.frames = frames = [] if keep_frames else None
                r.force(a); qlen = len(r.policy.queue); res = r.complete()
                cell = {'action': ACTIONS[a], 'action_id': a, 'y': int(res['success']), 'steps': res['steps'], 'reason': res['reason'],
                        'success_step': res['steps'] if res['success'] else None, 'intervention': res['intervention'],
                        'intervention_step': res['intervention_step'], 'queue_len_after_choice': qlen,
                        'policy_calls_after_snapshot': res['policy_calls'] - calls0,
                        'policy_seconds_after_snapshot': res['policy_inference_seconds'] - sec0, 'retreat': res['retreat'],
                        'prefix_replay_state_error': err, 'live_observation': live,
                        'stage': stage_of(list(trunk_rows[:t]) + rows), 'wall_seconds': time.perf_counter() - tick}
                return cell, rows, frames

            for rep in repeats:
                cells = []
                for a in range(3):
                    want = bool(video and frame) and t == first_snapshot and rep == 0
                    cell, rows, frames = branch(rep, a, want); cells.append(cell)
                    if probe: traces[f'b_{t}_{rep}_{a}'] = rows_array(rows)
                    if want: videos[f'scene{seed}_t{t}_r{rep}_{ACTIONS[a]}'] = list(trunk_frames[:t + 1]) + frames
                    if a == 0:
                        continue_refs[rep] = _ref(r)
                        if rep == 0: duplicates.append({'checkpoint': t, 'repeat': 0, 'against': 'trunk', **_same_run(trunk_ref, r, tol)})
                blocks.append({'scene': int(seed), 'checkpoint': t, 'repeat': rep, 'role': ROLE, 'y_continue': cells[0]['y'], 'cells': cells})
            if recheck_repeat is not None and recheck_repeat in continue_refs:      # the same block's Continue once more, after every other branch
                cell, _, _ = branch(recheck_repeat, 0)
                rechecks.append({'checkpoint': t, 'repeat': recheck_repeat, 'against': 'first Continue of the same block',
                                 **_same_run(continue_refs[recheck_repeat], r, tol)})
    ok = all(x['match'] for x in duplicates + rechecks)
    record = {'scene': int(seed), 'role': ROLE, 'planned_checkpoints': checkpoints, 'repeats': repeats, 'initial_state': initial, 'trunk': trunk,
              'snapshots': [snaps[t]['info'] for t in sorted(snaps)], 'skips': sorted(skips, key=lambda x: x['checkpoint']), 'blocks': blocks,
              'no_intervention_duplicates': duplicates, 'continue_rechecks': rechecks, 'reproducibility_ok': ok,
              'counts': {'snapshots': len(snaps), 'blocks': len(blocks), 'continuations': 3 * len(blocks),
                         'extra_reproducibility_continuations': len(rechecks)},
              'wall_seconds': time.perf_counter() - started, 'completed': True}
    record = jsonable(record); validate_scene(record, c)
    return record, traces, videos


def validate_scene(rec, c=None):
    """Accounting of one scene record: nothing missing, nothing twice, every block tied to its own Continue."""
    cps = rec['planned_checkpoints']; reps = rec['repeats']; snap = {x['checkpoint']: x for x in rec['snapshots']}; skip = [x['checkpoint'] for x in rec['skips']]
    if len(skip) != len(set(skip)) or set(snap) & set(skip) or set(snap) | set(skip) != set(cps):
        raise ValueError(f'scene {rec["scene"]}: planned checkpoints are not each a snapshot or a listed skip')
    keys = [(b['checkpoint'], b['repeat']) for b in rec['blocks']]
    if len(keys) != len(set(keys)) or set(keys) != {(t, x) for t in snap for x in reps}:
        raise ValueError(f'scene {rec["scene"]}: blocks do not match snapshots x repeats')
    for b in rec['blocks']:
        cells = b['cells']; t = b['checkpoint']
        if b['scene'] != rec['scene'] or b['role'] != rec['role']: raise ValueError('block scene/role differs from its scene record')
        if [x['action_id'] for x in cells] != [0, 1, 2]: raise ValueError('a block needs exactly continue, refresh, retreat_retry')
        if b['y_continue'] != cells[0]['y']: raise ValueError('y_continue is not the Continue result of the same block')
        if cells[0]['intervention'] is not None or cells[0]['queue_len_after_choice'] != snap[t]['queue_len']:
            raise ValueError('Continue changed the remaining action queue or was recorded as an intervention')
        for x in cells[1:]:
            if x['intervention'] != x['action_id'] or x['intervention_step'] != t or x['queue_len_after_choice'] != 0:
                raise ValueError('an intervention was not applied at the checkpoint with a cleared queue')
        if (cells[2]['retreat'] is None) or cells[1]['retreat'] is not None or cells[0]['retreat'] is not None:
            raise ValueError('retreat record attached to the wrong choice')
        if c is not None and any(x['steps'] > c['env']['horizon'] for x in cells): raise ValueError('a branch exceeded the episode horizon')
    n = rec['counts']
    if (n['snapshots'], n['blocks'], n['continuations']) != (len(snap), len(keys), 3 * len(keys)): raise ValueError('counts do not match the record')
    return True


def collect_scenes(seeds, out, run_scene, save_extra=None):
    """Scene after scene; one JSON per finished scene (exclusive create). The first error stops the worker:
    the error is written next to the scenes and no completion flag is produced."""
    out = Path(out); (out / 'scenes').mkdir(parents=True, exist_ok=True); done = []
    for seed in seeds:
        path = out / 'scenes' / f'{int(seed)}.json'
        if path.exists(): raise FileExistsError(f'Refusing overwrite: {path}')
        try:
            record, traces, videos = run_scene(seed)
            if save_extra: save_extra(seed, record, traces, videos)
            write_json(path, record)
        except Exception as e:
            write_json(out / 'scenes' / f'{int(seed)}.error.json', {'scene': int(seed), 'completed': False, 'error': repr(e),
                                                                    'traceback': traceback.format_exc(), 'finished_before_error': done})
            raise
        done.append(int(seed))
    return done


def finalize(out, seeds, c=None):
    """dataset.json. completed is true only when every scheduled scene has a valid record and no error file exists."""
    out = Path(out); recs = []; problems = []
    for seed in seeds:
        p = out / 'scenes' / f'{int(seed)}.json'
        if (out / 'scenes' / f'{int(seed)}.error.json').exists(): problems.append(f'scene {seed}: error file present')
        if not p.exists(): problems.append(f'scene {seed}: no record'); continue
        rec = read_json(p)
        try: validate_scene(rec, c)
        except ValueError as e: problems.append(str(e)); continue
        if not rec.get('completed'): problems.append(f'scene {seed}: not completed'); continue
        recs.append(rec)
    bad = [x['scene'] for x in recs if not x['reproducibility_ok']]
    if bad: problems.append(f'reproducibility check failed in scenes {bad}: rows are kept as measured, interpretation is withheld')
    extra = sorted(int(p.name.split('.')[0]) for p in (out / 'scenes').glob('*.json') if int(p.name.split('.')[0]) not in {int(s) for s in seeds})
    if extra: problems.append(f'records of unscheduled scenes: {extra}')
    meta = {'role': ROLE, 'scheduled_scenes': [int(s) for s in seeds], 'recorded_scenes': [x['scene'] for x in recs], 'completed': not problems,
            'problems': problems, 'snapshots': sum(x['counts']['snapshots'] for x in recs), 'blocks': sum(x['counts']['blocks'] for x in recs),
            'continuations': sum(x['counts']['continuations'] for x in recs),
            'extra_reproducibility_continuations': sum(x['counts']['extra_reproducibility_continuations'] for x in recs),
            'reproducibility_ok': all(x['reproducibility_ok'] for x in recs) if recs else None,
            'wall_seconds_sum_over_scenes': sum(x['wall_seconds'] for x in recs)}
    return meta, recs


# ---------------------------------------------------------------- plan helpers
def select_scenes(train, n, declared, demo_seeds=()):
    """The first n train seeds in ascending order that are declared for no other use. Outcomes play no part."""
    other = {}
    for name, ids in declared.items():
        if name == 'splits/train': continue
        for i in ids: other.setdefault(int(i), []).append(name)
    for i in demo_seeds: other.setdefault(int(i), []).append('demonstration_scene')
    chosen = []; skipped = []
    for s in sorted(int(x) for x in train):
        if len(chosen) == n: break
        if s in other: skipped.append({'seed': s, 'reason': 'also declared as ' + ', '.join(other[s])})
        else: chosen.append(s)
    if len(chosen) < n: raise ValueError(f'train has only {len(chosen)} usable scenes; {n} wanted (nothing is borrowed from other splits)')
    return chosen, skipped


def scale_decision(wall_seconds, continuations, *, usd_per_hour, spent_usd, cap_usd, reserve_usd, workers, efficiency,
                   ladder=(24, 12, 2), done=2, per_scene_continuations=50, margin=1.25):
    """Largest pre-declared scene count whose projected cost fits the cap. Inputs are time and money only."""
    if not continuations: raise ValueError('no continuation was measured; the projection is undefined')
    per = sum(wall_seconds) / continuations; scene_s = per * per_scene_continuations * margin; options = []
    for n in ladder:
        hours = max(0, n - done) * scene_s / (workers * efficiency) / 3600
        options.append({'scenes': n, 'projected_extra_hours': hours, 'projected_total_usd': spent_usd + hours * usd_per_hour + reserve_usd})
    fit = [o for o in options if o['projected_total_usd'] <= cap_usd]
    chosen = fit[0]['scenes'] if fit else min(ladder)
    return {'chosen_scenes': chosen, 'options': options, 'measured_seconds_per_continuation': per, 'assumed_seconds_per_scene': scene_s,
            'inputs': {'usd_per_hour': usd_per_hour, 'spent_usd': spent_usd, 'cap_usd': cap_usd, 'reserve_usd': reserve_usd, 'workers': workers,
                       'parallel_efficiency': efficiency, 'per_scene_continuations': per_scene_continuations, 'margin': margin, 'ladder': list(ladder)},
            'rule': 'first entry of the ladder whose projected total is within the cap; branch outcomes are not an input',
            'fits_cap': bool(fit)}


# ---------------------------------------------------------------- descriptive tables
def pair_rows(records):
    """One row per (scene, checkpoint, repeat, non-continue action). The block's own Continue is the reference."""
    rows = []
    for rec in records:
        snap = {x['checkpoint']: x for x in rec['snapshots']}
        for b in rec['blocks']:
            c0 = b['cells'][0]; s = snap[b['checkpoint']]; at = s.get('state_at_checkpoint') or {}
            for x in b['cells'][1:]:
                rt = x['retreat'] or {}; st = lambda v: (v or {}).get('furthest_stage')
                rows.append({'scene': b['scene'], 'role': b['role'], 'checkpoint': b['checkpoint'], 'repeat': b['repeat'], 'action': x['action'],
                             'y_continue': b['y_continue'], 'y_action': x['y'], 'outcome': CLASSES[(b['y_continue'], x['y'])],
                             'steps_continue': c0['steps'], 'steps_action': x['steps'], 'reason_continue': c0['reason'], 'reason_action': x['reason'],
                             'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'],
                             'queue_len_at_checkpoint': s['queue_len'], 'history_len_at_checkpoint': s['history_len'], 'remaining_budget': s['remaining_budget'],
                             'stage_so_far_at_checkpoint': s.get('stage_so_far'), 'grasped_at_checkpoint': at.get('is_cubeA_grasped'),
                             'stage_continue': st(c0['stage']), 'stage_action': st(x['stage']),
                             'policy_calls_continue': c0['policy_calls_after_snapshot'], 'policy_calls_action': x['policy_calls_after_snapshot'],
                             'retreat_steps_used': rt.get('steps_used'), 'retreat_commanded_m': rt.get('commanded_distance_m'),
                             'retreat_achieved_m': rt.get('achieved_displacement_m'), 'retreat_target_error_m': rt.get('target_error_m'),
                             'retreat_reached': rt.get('reached'), 'retreat_end_reason': rt.get('end_reason'), 'gripper_command_held': rt.get('gripper_command')})
    return rows


def _four(rows):
    n = {k: 0 for k in CLASSES.values()}
    for x in rows: n[x['outcome']] += 1
    ok = n['harm'] + n['both_succeed']; bad = n['rescue'] + n['both_fail']
    return {**n, 'blocks': len(rows), 'continue_success_blocks': ok, 'continue_failure_blocks': bad,
            'H_branch': n['harm'] / ok if ok else 'NA', 'R_branch': n['rescue'] / bad if bad else 'NA',
            'scenes': len({x['scene'] for x in rows}), 'scenes_with_rescue': len({x['scene'] for x in rows if x['outcome'] == 'rescue'}),
            'scenes_with_harm': len({x['scene'] for x in rows if x['outcome'] == 'harm'})}


def scene_rows(records):
    out = []; pairs = pair_rows(records)
    for rec in records:
        mine = [x for x in pairs if x['scene'] == rec['scene']]; blocks = rec['blocks']
        row = {'scene': rec['scene'], 'role': rec['role'], 'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'],
               'trunk_stage': (rec['trunk']['stage'] or {}).get('furthest_stage'), 'snapshots': rec['counts']['snapshots'],
               'checkpoints_used': ' '.join(str(x['checkpoint']) for x in rec['snapshots']), 'blocks': len(blocks),
               'continue_success_blocks': sum(b['y_continue'] for b in blocks), 'reproducibility_ok': rec['reproducibility_ok']}
        for a in ACTIONS[1:]:
            f = _four([x for x in mine if x['action'] == a])
            for k in ('rescue', 'harm', 'both_succeed', 'both_fail'): row[f'{a}_{k}'] = f[k]
        diff = 0
        for b in blocks: diff += int(b['cells'][1]['y'] != b['cells'][2]['y'])
        row['blocks_where_the_two_actions_differ'] = diff; out.append(row)
    return out


def skip_rows(records):
    return [{'scene': rec['scene'], 'checkpoint': x['checkpoint'], 'reason': x['reason'], 'detail': '; '.join(x['detail'] or []),
             'trunk_success': rec['trunk']['success'], 'trunk_steps': rec['trunk']['steps'], 'cells_not_run': 3 * len(rec['repeats'])}
            for rec in records for x in rec['skips']]


def summarize(records, planned_scenes=None):
    pairs = pair_rows(records); scenes = scene_rows(records); skips = skip_rows(records); cps = sorted({x['checkpoint'] for x in pairs})
    table = {a: {'all_checkpoints': _four([x for x in pairs if x['action'] == a]),
                 **{f't{t}': _four([x for x in pairs if x['action'] == a and x['checkpoint'] == t]) for t in cps}} for a in ACTIONS[1:]}
    by_stage = {}
    for x in pairs:
        d = by_stage.setdefault(str(x['stage_so_far_at_checkpoint']), {}).setdefault(x['action'], {k: 0 for k in CLASSES.values()}); d[x['outcome']] += 1
    retreat = [x for x in pairs if x['action'] == 'retreat_retry']; reasons = {}
    for x in retreat: reasons[str(x['retreat_end_reason'])] = reasons.get(str(x['retreat_end_reason']), 0) + 1
    num = lambda k: [x[k] for x in retreat if x[k] is not None]
    stat = lambda v: {'n': len(v), 'min': float(np.min(v)), 'median': float(np.median(v)), 'max': float(np.max(v))} if v else {'n': 0}
    spread = {}
    for rec in records:
        for t in {b['checkpoint'] for b in rec['blocks']}:
            ys = {b['y_continue'] for b in rec['blocks'] if b['checkpoint'] == t}; spread[len(ys)] = spread.get(len(ys), 0) + 1
    skip_counts = {}
    for x in skips: skip_counts[x['reason']] = skip_counts.get(x['reason'], 0) + 1
    dup = [d for rec in records for d in rec['no_intervention_duplicates']]; rc = [d for rec in records for d in rec['continue_rechecks']]
    return {'role': ROLE, 'scenes_planned': planned_scenes, 'scenes_recorded': len(records),
            'trunk_success_scenes': sum(r['trunk']['success'] for r in records), 'trunk_failure_scenes': sum(1 - r['trunk']['success'] for r in records),
            'scenes_with_a_snapshot': sum(1 for r in records if r['snapshots']), 'snapshots': sum(r['counts']['snapshots'] for r in records),
            'paired_blocks': sum(r['counts']['blocks'] for r in records), 'continuations': sum(r['counts']['continuations'] for r in records),
            'extra_reproducibility_continuations': sum(r['counts']['extra_reproducibility_continuations'] for r in records),
            'continue_success_blocks': sum(b['y_continue'] for r in records for b in r['blocks']),
            'skipped_checkpoints': skip_counts, 'cells_not_run': sum(x['cells_not_run'] for x in skips),
            'by_action_and_checkpoint': table, 'by_stage_at_checkpoint': by_stage,
            'scenes_where_the_two_actions_differ': sum(1 for s in scenes if s['blocks_where_the_two_actions_differ']),
            'snapshots_by_number_of_distinct_continue_outcomes_over_repeats': {str(k): v for k, v in sorted(spread.items())},
            'retreat': {'executed': len(retreat), 'reached_target': sum(1 for x in retreat if x['retreat_reached']), 'end_reasons': reasons,
                        'steps_used': stat(num('retreat_steps_used')), 'achieved_displacement_m': stat(num('retreat_achieved_m')),
                        'target_error_m': stat(num('retreat_target_error_m'))},
            'reproducibility': {'no_intervention_duplicates': len(dup), 'duplicate_mismatches': sum(1 for d in dup if not d['match']),
                                'continue_rechecks': len(rc), 'recheck_mismatches': sum(1 for d in rc if not d['match']),
                                'interpretation_withheld': any(not d['match'] for d in dup + rc)},
            'denominators': {'blocks': 'one (scene, checkpoint, repeat); each has one Continue and is counted once per action',
                             'H_branch': 'harm / blocks whose own Continue succeeded', 'R_branch': 'rescue / blocks whose own Continue failed',
                             'NA': 'the denominator is zero; this is not 0%'},
            'scope': ['Development sample chosen after the qualification result; not a pre-registered confirmatory test.',
                      'Blocks of one scene share the scene and the past; they are not independent episodes. No interval or p-value is computed.',
                      'A forced choice at a fixed checkpoint is not the value or the harm rate H(g) of an online intervention policy.',
                      'Outcomes hold under this simulator and the declared noise coupling only.',
                      'Zero observed harm would be zero events in this sample, not a safety statement.']}
