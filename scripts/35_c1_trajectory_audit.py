"""Phase C1 trajectory and behaviour audit of a C1 run folder. POST-HOC ONLY: reads stored records, runs no simulator and
no policy, writes nothing into the run folder, and is not part of the plan (it is not hashed by plan_v2.json and not read
on the server).

    python scripts/35_c1_trajectory_audit.py --run <run folder> --plan <plan.json> --out <new folder>
                                             [--seeds 7,8 | --seeds 51100-51299] [--figure-scenes N] [--no-figures]

Layout read, as written by scripts/34_vision_center_c1.py and hci.vision_center_c1.collect_scene:
  <run>/pass_<policy>/scenes/<seed>.json          record: cells with branch_id, snapshot_digest, first_open_command_after_move,
                                                   success_step, frames_after_trigger; frame_index; plan_sha256
  <run>/pass_<policy>/traces/trace_<seed>.json.gz  per-step trace of the trunk and of every arm (trace_sha256 in the record)
  <run>/pass_<policy>/frames/frames_<seed>.npz     'trunk' (steps 0..trigger), '<VISION ARM>' (trigger + j), 'after_<ARM>'
  <run>/pass_<policy>/scenes/<seed>.error.json     a failed scene;  <run>/**/budget_stop.json  a stopped run
Policies: hci.vision_center_c1.POLICIES; arms: the plan's arm registry; seeds: the plan's seed registry unless --seeds.

Writes (the out folder must not exist):
  audit_summary.json        run status: COMPLETE (every planned seed of both policies, no stop or error file, every record
                            check passed), SUBSET_ONLY (--seeds narrower than the plan; never a run result) or INCOMPLETE, counts per policy, arm and check, failures,
                            record-integrity results, cases, thresholds, code hashes. No time stamp: equal inputs give an
                            identical file.
  per_scene_checks.csv      one row per policy, scene, arm and check (the ten behaviour checks and the record checks R_*)
  trajectory_manifest.json  every input file with sha256 (before and after the audit), every output with sha256, rules
  figures/<policy>/scene_<seed>/   3D, XY, timeline, outcome strip, recorded RGB and POST-HOC overlay GIFs, key frames
The ten checks are hci.trajectory_audit.check_arm, unchanged (C1_BEHAVIOR_EXPECTATION_SPEC.md). Ground truth from the
trace is used here only, after the fact. Steps without a stored frame are shown as NO RGB RECORDED; nothing is made up.
"""
import os
for _k in ('OMP', 'OPENBLAS', 'MKL', 'NUMEXPR', 'BLIS'): os.environ.setdefault(_k + '_NUM_THREADS', '1')
import argparse, csv, gzip, hashlib, io, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from hci import trajectory_audit as T
from hci import vision_center_c1 as C1

STATE_ATOL = 1e-6                                     # configs/stackcube.yaml audit.state_atol, as used by the collector
RECORD_CHECKS = ('R_files_and_hashes', 'R_record_identity', 'R_branch_and_snapshot', 'R_frame_index', 'R_abstain_pointer_to_C0', 'R_events_match_trace', 'R_handback', 'R_control_tcp_sync')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ROOT = Path(__file__).resolve().parents[1]


def _py(v): return T._py(v)
def ok(cond, **kw): return _py({'status': 'PASS' if cond else 'FAIL', **kw})
def no(why, **kw): return _py({'status': 'NOT_OBSERVABLE', 'why': why, **kw})


def file_sha(p):
    try: return sha(p)
    except Exception as e: return f'UNREADABLE {type(e).__name__}'


def exists(p):
    try: return p.exists()
    except Exception: return True                                  # cannot tell: try to read it, and a read error becomes a failed record


def inputs_digest(run):
    """sha256 of every file of the run folder; a file that cannot be read is recorded as UNREADABLE (and makes the run INCOMPLETE), never a crash."""
    out = {}
    try: files = sorted(Path(run).rglob('*'))
    except Exception as e: return {'<listing>': f'UNREADABLE {type(e).__name__}'}
    for p in files:
        try: is_file = p.is_file()
        except Exception: is_file = True
        if is_file: out[p.relative_to(run).as_posix()] = file_sha(p)
    return out


def parse_seeds(s, plan):
    if s is None:
        a, b = plan['seed_registry']['range']; return list(range(int(a), int(b) + 1))
    out = []
    for part in s.split(','):
        if '-' in part: a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out


# ---------------------------------------------------------------- run level
def run_status(run, seeds, plan_hash, plan_seeds=None):
    """INCOMPLETE unless every planned record exists and nothing stopped or failed. A partial run is never COMPLETE."""
    run = Path(run); reasons = []
    for p in sorted(run.rglob('budget_stop.json')): reasons.append(f'budget stop: {p.relative_to(run).as_posix()}')
    for p in sorted(run.rglob('*.error.json')): reasons.append(f'error file: {p.relative_to(run).as_posix()}')
    for pid in C1.POLICIES:
        d = run / f'pass_{pid}'
        if not d.is_dir(): reasons.append(f'pass folder missing: pass_{pid}'); continue
        miss = [s for s in seeds if not (d / 'scenes' / f'{s}.json').exists()]
        if miss: reasons.append(f'pass_{pid}: {len(miss)} planned scene records missing (first {miss[:5]})')
        extra = sorted(int(p.stem) for p in (d / 'scenes').glob('*.json') if not p.name.endswith('.error.json') and p.stem.isdigit() and int(p.stem) not in set(plan_seeds or seeds))
        if extra: reasons.append(f'pass_{pid}: records outside the planned seeds {extra[:5]}')
    for name in ('ledger.json',):
        if not (run / name).exists(): reasons.append(f'{name} missing')
        else:
            try: lg = json.loads((run / name).read_text(encoding='utf-8')); lg_plan = lg.get('plan_sha256') if isinstance(lg, dict) else None
            except Exception as e: reasons.append(f'{name} unreadable ({type(e).__name__})'); continue
            if lg_plan != plan_hash: reasons.append(f'{name} names another plan or none')
    return {'status': 'INCOMPLETE' if reasons else 'COMPLETE', 'reasons': reasons}


# ---------------------------------------------------------------- one scene
def load(d, seed):
    rec = json.loads((d / 'scenes' / f'{seed}.json').read_text(encoding='utf-8'))
    with gzip.open(d / rec['trace'], 'rt', encoding='utf-8') as f: tr = json.load(f)['traces']
    fr = None
    if rec.get('frames') and (d / rec['frames']).exists():
        with np.load(d / rec['frames']) as z: fr = {k: z[k] for k in z.files}
    return rec, tr, fr


def record_checks(d, seed, pid, rec, tr, fr, arms, plan_hash):
    """The R_* checks: files, identity, branch and snapshot, frame index, abstention pointer, events, hand-back, sync."""
    out = {}
    tsha = sha(d / rec['trace']); fsha = sha(d / rec['frames']) if rec.get('frames') and (d / rec['frames']).exists() else None
    # the collector stores trunk frames for every scene, with or without a trigger, so a record always names its frames file
    out['R_files_and_hashes'] = ok(tsha == rec.get('trace_sha256') and fsha is not None and fsha == rec.get('frames_sha256'), trace_matches=tsha == rec.get('trace_sha256'),
                                   frames_present=fsha is not None, frames_match=fsha is not None and fsha == rec.get('frames_sha256'))
    ident = {'completed': rec.get('completed') is True, 'scene': rec.get('scene') == int(seed), 'role': rec.get('role') == C1.ROLE, 'policy': rec.get('policy_id') == pid,
             'plan': rec.get('plan_sha256') == plan_hash, 'arms': list(rec.get('arms', [])) == list(arms), 'cells_in_order': [c['arm'] for c in rec['cells']] == list(arms)}
    try: C1.validate_scene(rec); ident['validate_scene'] = True
    except ValueError as e: ident['validate_scene'] = False; ident['validate_error'] = str(e)
    out['R_record_identity'] = ok(all(v is True for k, v in ident.items() if k != 'validate_error'), **ident)
    if not rec['triggered']:                                     # trunk frames: one per step of the whole episode (the loop observes until the end)
        n = len(fr['trunk']) if fr is not None and 'trunk' in fr else None; want = rec['trunk']['steps']; extra = sorted(k for k in (fr or {}) if k != 'trunk')
        out['R_frame_index'] = ok(n == want and not extra, trunk_frames=n, trunk_steps=want, other_keys=extra)
        for k in RECORD_CHECKS[2:]:
            if k != 'R_frame_index': out[k] = no('no trigger in this scene')
        return out
    t0 = rec['trigger']['step']; dig = rec.get('snapshot_digest')
    bad_branch = [c['arm'] for c in rec['cells'] if c.get('branch_id') != f'{seed}:{pid}:t{t0}:{c["arm"]}']
    bad_snap = [c['arm'] for c in rec['cells'] if not dig or c.get('snapshot_digest') != dig]
    out['R_branch_and_snapshot'] = ok(not bad_branch and not bad_snap, snapshot_digest=dig, wrong_branch_id=bad_branch, wrong_snapshot_digest=bad_snap)
    # frame index: trunk frames 0..t0; control frames trigger + j; after-trigger frames at the listed steps of FRAME_POLICY
    fi = {'frame_index_present': isinstance(rec.get('frame_index'), dict), 'trunk_frames': None, 'problems': [], 'duplicate_consecutive_frames': {}}
    if fr is None: fi['problems'].append('frames file missing')
    else:
        fi['trunk_frames'] = int(len(fr['trunk'])) if 'trunk' in fr else None
        if fi['trunk_frames'] != t0 + 1: fi['problems'].append(f'trunk has {fi["trunk_frames"]} frames, expected {t0 + 1} (steps 0..{t0})')
        for c in rec['cells']:
            a = c['arm']; fa = c.get('frames_after_trigger') or {}
            if a in C1.VISION_ARMS:
                n = len(fr[a]) if a in fr else 0; want = len(((c.get('control') or {}).get('readings')) or [])       # the arm stores one frame per reading (vision_center_v2.py, vision_center.py, center_move)
                if n != want: fi['problems'].append(f'{a}: {n} control frames for {want} readings')
                if not c.get('acted') and want != 1: fi['problems'].append(f'{a}: abstained after {want} readings, expected the trigger reading only')
            if a == 'C0' or c.get('acted'):
                key, steps = fa.get('key'), fa.get('steps') or []
                k = c['control_steps'] if c.get('acted') else 0
                expect = C1.post_frame_steps(t0, c['steps'], k, c.get('first_open_command_after_move'))
                if key != f'after_{a}' or key not in fr: fi['problems'].append(f'{a}: after-trigger frames missing ({key})'); continue
                if steps != expect: fi['problems'].append(f'{a}: stored steps differ from FRAME_POLICY')
                if len(set(steps)) != len(steps) or steps != sorted(steps): fi['problems'].append(f'{a}: repeated or unordered steps')
                if len(fr[key]) != len(steps): fi['problems'].append(f'{a}: {len(fr[key])} frames for {len(steps)} listed steps')
                if steps and steps[-1] != c['steps']: fi['problems'].append(f'{a}: last stored step {steps[-1]} is not the last step {c["steps"]}')
                fi['duplicate_consecutive_frames'][a] = int(sum(1 for i in range(1, len(fr[key])) if np.array_equal(fr[key][i], fr[key][i - 1])))
            elif f'after_{a}' in fr: fi['problems'].append(f'{a}: abstained but stores its own after-trigger frames')
    out['R_frame_index'] = ok(fi['frame_index_present'] and not fi['problems'], **fi)
    # abstaining arms: pointer to C0, and the trace itself equals C0 at every step
    ptr = {}
    for c in rec['cells']:
        if c['arm'] == 'C0' or c.get('acted'): continue
        fa = c.get('frames_after_trigger') or {}; s, r = T.series(rec, tr, c['arm']), T.series(rec, tr, 'C0')
        same_len = len(s['t']) == len(r['t']); diff = max(float(np.max(np.abs(s[q] - r[q]))) for q in ('tcp', 'A', 'B')) if same_len else None
        ptr[c['arm']] = {'same_as': fa.get('same_as'), 'recorded_max_difference_m': fa.get('max_logged_position_difference_to_C0_m'), 'recomputed_max_difference_m': diff,
                         'ok': bool(fa.get('same_as') == 'C0' and same_len and diff is not None and diff <= STATE_ATOL and (fa.get('max_logged_position_difference_to_C0_m') or 0.) <= STATE_ATOL)}
    out['R_abstain_pointer_to_C0'] = ok(all(v['ok'] for v in ptr.values()), arms=ptr) if ptr else no('no arm abstained')
    # events: the record's first open command and success step against the trace
    ev = {}
    for c in rec['cells']:
        a = c['arm']; s = T.series(rec, tr, a); e = T.events(rec, s, a); k = c['control_steps'] if c.get('acted') else 0
        opens = [int(x) for x, act in zip(s['t'], s['action']) if x > t0 + k and act[-1] > 0]; succ = [int(x) for x, y in zip(s['t'], s['success']) if y]
        ev[a] = {'first_open_record': c.get('first_open_command_after_move'), 'first_open_trace': opens[0] if opens else None, 'success_record': c.get('success_step'),
                 'success_trace': succ[0] if succ else None, 'outcome_matches_trace': bool(c['y']) == bool(succ), 'length_matches_trace': int(s['t'][-1]) == c['steps'],
                 'timeline_events': e}
        ev[a]['ok'] = bool(ev[a]['first_open_record'] == ev[a]['first_open_trace'] == e['release_cmd'] and ev[a]['success_record'] == ev[a]['success_trace'] == e['success'] and ev[a]['outcome_matches_trace'] and ev[a]['length_matches_trace'])
    out['R_events_match_trace'] = ok(all(v['ok'] for v in ev.values()), arms=ev)
    # hand-back: control steps logged, no policy inference before the hand-back, resume step
    hb = {}
    for c in rec['cells']:
        if not c.get('acted'): continue
        k = c['control_steps']; logged = len(((c.get('control') or {}).get('steps')) or []) if c['arm'] != 'CENTER_ORACLE' else (c.get('control') or {}).get('steps_used', k)
        fi_step = c.get('first_inference_step'); went_on = c['steps'] > t0 + k and not c.get('ended_during_control')
        # the move clears the action queue, so the policy must infer at the hand-back step t0 + k (its observation) unless the episode ended during control
        hb[c['arm']] = {'control_steps': k, 'logged_steps': logged, 'policy_resumes_at_step': t0 + k + 1 if went_on else None, 'first_inference_step': fi_step, 'expected_first_inference_step': t0 + k if went_on else None,
                        'ok': bool(logged == k and fi_step == (t0 + k if went_on else None))}
    out['R_handback'] = ok(all(v['ok'] for v in hb.values()), arms=hb) if hb else no('no arm moved')
    sync = T.control_tcp_sync(rec, tr)
    out['R_control_tcp_sync'] = ok(all(v['max_difference_m'] < 1e-9 for v in sync.values()), arms=sync) if sync else no('no logged control step with a TCP')
    return out


def failed(pid, seed, d, check, e):
    return _py({'policy': pid, 'seed': seed, 'metrics': {'scene': seed, 'triggered': False, 'arms': {}}, 'record_checks': {check: ok(False, error=f'{type(e).__name__}: {e}'.replace(str(d), '<pass>'))},
                'sync': None, 'move': {}, 'memory': {}})


def one(job):
    """One record. A record that cannot be read, or whose content breaks any step of the audit, becomes a failed record
    check (and so an INCOMPLETE run), never a crash and never a silent pass."""
    run, pid, seed, arms, plan_hash = job; d = Path(run) / f'pass_{pid}'
    try: rec, tr, fr = load(d, seed)
    except Exception as e: return failed(pid, seed, d, 'R_files_and_hashes', e)
    try: return analyse(d, pid, seed, arms, plan_hash, rec, tr, fr)
    except Exception as e: return failed(pid, seed, d, 'R_record_identity', e)


def analyse(d, pid, seed, arms, plan_hash, rec, tr, fr):
    m = T.scene_metrics(rec, tr); rc = record_checks(d, seed, pid, rec, tr, fr, arms, plan_hash)
    sync = T.sync_check(rec, tr, fr) if rec['triggered'] else None
    move = {}
    if rec['triggered']:
        t0 = rec['trigger']['step']
        for a in rec['arms']:
            c = T.cell(rec, a)
            if not c.get('acted'): continue
            s = T.series(rec, tr, a); i0, i1 = T.at(s, t0), T.at(s, t0 + c['control_steps']); off = np.linalg.norm(T.offset_xy(s), axis=1)
            move[a] = {'target_displacement_xy_m': float(np.linalg.norm(s['B'][i1, :2] - s['B'][i0, :2])), 'offset_before_m': float(off[i0]), 'offset_after_m': float(off[i1]),
                       'end_reason': (c.get('control') or {}).get('end_reason')}
    mem = {a: {'used_memory': bool((T.cell(rec, a).get('control') or {}).get('used_memory')), 'gt': T.cell(rec, a).get('gt_reading_error')} for a in rec['arms'] if rec['triggered'] and a in C1.VISION_ARMS}
    return _py({'policy': pid, 'seed': seed, 'metrics': m, 'record_checks': rc, 'sync': sync, 'move': move, 'memory': mem})


def chk(r, arm, name): return (((r['metrics']['arms'].get(arm) or {}).get('checks') or {}).get(name) or {})


CASE_RULES = {                                      # as the Phase B audit, over all nine arms
    'harm_any_arm': lambda r: any(v['harm'] for v in r['metrics']['arms'].values()),
    'target_pushed_during_move': lambda r: any((v.get('target_displacement_xy_m') or 0) > T.PUSH_TOL_M for v in r['move'].values()),
    'memory_error_move': lambda r: any(m['used_memory'] and (m['gt'] or {}).get('offset_error_m') is not None and ((m['gt']['offset_error_m'] > T.MEMORY_ERROR_M) or m['gt'].get('error_within_bound') is False)
                                       and r['metrics']['arms'][a]['acted'] for a, m in r['memory'].items()),
    'unnecessary_move': lambda r: any(chk(r, a, '2a_acted_only_when_needed').get('status') == 'FAIL' and chk(r, a, '2a_acted_only_when_needed').get('move_steps', 0) > 0 for a in r['metrics']['arms']),
    'v2_abstained_oracle_rescued': lambda r: 'CENTER_ORACLE' in r['metrics']['arms'] and r['metrics']['arms']['CENTER_ORACLE']['rescue'] and not r['metrics']['arms']['VISION_CENTER_V2']['acted'],
    'correction_undone_after_resume': lambda r: any(chk(r, a, '8_policy_keeps_correction').get('status') == 'FAIL' for a in r['metrics']['arms']),
    'step_budget_reached': lambda r: any(v.get('end_reason') == 'step_budget' for v in r['move'].values()),
    'record_check_failed': lambda r: any(x['status'] == 'FAIL' for x in r['record_checks'].values()),
    'any_check_failed': lambda r: any(x.get('status') == 'FAIL' and k != '2b_acted_when_needed' for v in r['metrics']['arms'].values() for k, x in v['checks'].items()),
}
REPRESENTATIVE = {
    'v2_rescue': lambda r: r['metrics']['arms']['VISION_CENTER_V2']['rescue'],
    'v2_moved_both_succeed': lambda r: r['metrics']['arms']['VISION_CENTER_V2']['acted'] and r['metrics']['arms']['C0']['y'] == 1 and r['metrics']['arms']['VISION_CENTER_V2']['y'] == 1,
    'v2_abstained_both_fail': lambda r: not r['metrics']['arms']['VISION_CENTER_V2']['acted'] and r['metrics']['arms']['C0']['y'] == 0,
}


def figures(run, pid, seed, out):
    d = Path(run) / f'pass_{pid}'; rec, tr, fr = load(d, seed); out.mkdir(parents=True); made = []
    for name, fn in (('3d_trajectory.png', T.fig_3d), ('xy_alignment.png', T.fig_xy), ('timeline_actions_gripper.png', T.fig_timeline), ('after_trigger_outcome.png', T.outcome_strip)):
        fn(rec, tr, out / name, policy=pid); made.append(out / name)
    if fr is not None:
        t0 = rec['trigger']['step']
        for name, ov in (('rgb_recorded', False), ('rgb_with_posthoc_ground_truth', True)):
            pan = T.rgb_panels(rec, tr, fr, policy=pid, overlay=ov); T.save_gif(pan, out / f'{name}.gif'); made.append(out / f'{name}.gif')
            first = max(0, t0 - 10); pan[t0 - first].save(out / f'{name}_step{t0}_trigger.png'); made.append(out / f'{name}_step{t0}_trigger.png')
    abst = [c['arm'] for c in rec['cells'] if c['arm'] != 'C0' and not c.get('acted')]
    return {'policy': pid, 'seed': seed, 'outputs': {p.name: sha(p) for p in made}, 'abstaining_arms_shown_as_C0': abst,
            'note': 'RGB panels show C0 and every arm that moved; an abstaining arm has no frames of its own (pointer to C0, checked in R_abstain_pointer_to_C0)'}


def audit(run, plan_path, out, seeds=None, figure_scenes=6, make_figures=True):
    run, out = Path(run), Path(out)
    if out.resolve() == run.resolve() or run.resolve() in out.resolve().parents or out.resolve() in run.resolve().parents: raise SystemExit(f'Refused: the output folder must be outside the run folder ({out})')
    if out.exists(): raise SystemExit(f'Refused overwrite: {out}')
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8')); plan_hash = sha(plan_path); arms = list(plan['arm_registry']['arms']); plan_seeds = parse_seeds(None, plan); seeds = list(seeds) if seeds else plan_seeds
    try: present = run.is_dir()
    except Exception: present = False
    before = inputs_digest(run) if present else {}
    try: status = run_status(run, seeds, plan_hash, plan_seeds) if present else {'status': 'INCOMPLETE', 'reasons': ['the run folder is missing or is not a folder']}
    except Exception as e: status = {'status': 'INCOMPLETE', 'reasons': [f'the run folder could not be inspected ({type(e).__name__})']}
    bad = sorted(k for k, v in before.items() if v.startswith('UNREADABLE'))
    if bad: status = {'status': 'INCOMPLETE', 'reasons': status['reasons'] + [f'{len(bad)} files of the run folder could not be read: {bad[:5]}']}
    rows = [one((run, pid, s, arms, plan_hash)) for pid in C1.POLICIES for s in seeds if exists(run / f'pass_{pid}' / 'scenes' / f'{s}.json')]
    trig = [r for r in rows if r['metrics']['triggered']]
    broken = sorted(f"{r['policy']}:{r['seed']}:{k}" for r in rows for k, x in r['record_checks'].items() if x.get('status') == 'FAIL')
    if broken: status['reasons'].append(f'{len(broken)} record checks failed (first {broken[:5]})'); status['status'] = 'INCOMPLETE'
    if status['status'] == 'COMPLETE' and seeds != plan_seeds: status = {'status': 'SUBSET_ONLY', 'reasons': [f'only {len(seeds)} of the {len(plan_seeds)} planned seeds were audited (--seeds); this is never a complete run']}
    out.mkdir(parents=True); counts = {}; rcounts = {}; fails = {}
    with open(out / 'per_scene_checks.csv', 'x', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['policy', 'scene', 'arm', 'y', 'acted', 'harm', 'rescue', 'check', 'status', 'evidence_json'])
        for r in rows:
            for k, x in r['record_checks'].items():
                rcounts.setdefault(r['policy'], {}).setdefault(k, {s: 0 for s in T.STATUS})[x['status']] += 1
                if x['status'] == 'FAIL': fails.setdefault(r['policy'], {}).setdefault('record', {}).setdefault(k, []).append(r['seed'])
                w.writerow([r['policy'], r['seed'], '*', '', '', '', '', k, x['status'], json.dumps({kk: vv for kk, vv in x.items() if kk != 'status'}, sort_keys=True)])
            for arm, v in r['metrics']['arms'].items():
                for k, x in v['checks'].items():
                    counts.setdefault(r['policy'], {}).setdefault(arm, {}).setdefault(k, {s: 0 for s in T.STATUS})[x['status']] += 1
                    if x['status'] == 'FAIL': fails.setdefault(r['policy'], {}).setdefault(arm, {}).setdefault(k, []).append(r['seed'])
                    w.writerow([r['policy'], r['seed'], arm, v['y'], int(v['acted']), int(v['harm']), int(v['rescue']), k, x['status'], json.dumps({kk: vv for kk, vv in x.items() if kk != 'status'}, sort_keys=True)])
    cases = {pid: {k: sorted(r['seed'] for r in trig if r['policy'] == pid and rule(r)) for k, rule in CASE_RULES.items()} for pid in C1.POLICIES}
    for pid in C1.POLICIES: cases[pid].update({f'representative_{k}': sorted(r['seed'] for r in trig if r['policy'] == pid and rule(r))[:3] for k, rule in REPRESENTATIVE.items()})
    pick = {pid: sorted({s for k, v in cases[pid].items() if k != 'any_check_failed' for s in v} | set(cases[pid]['any_check_failed'][:3]))[:figure_scenes] for pid in C1.POLICIES}
    figs = []
    for pid in C1.POLICIES:                                       # a figure that cannot be drawn (e.g. a record with missing frames) is reported, never fatal
        for s in (pick[pid] if make_figures else []):
            try: figs.append(figures(run, pid, s, out / 'figures' / pid / f'scene_{s}'))
            except Exception as e: figs.append({'policy': pid, 'seed': s, 'error': f'{type(e).__name__}: {e}'.replace(str(run), '<run>')})
    ferr = sorted(f"{f['policy']}:{f['seed']}" for f in figs if 'error' in f)
    if ferr: status = {'status': 'INCOMPLETE', 'reasons': status['reasons'] + [f'{len(ferr)} figure scenes could not be drawn: {ferr[:5]}']}
    after = inputs_digest(run) if present else {}; changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if changed: status = {'status': 'INCOMPLETE', 'reasons': status['reasons'] + [f'the run folder changed during the audit: {changed[:5]}']}
    sy = [r['sync'] for r in trig if r['sync']]; ctl = [v['max_difference_m'] for r in trig for v in ((r['record_checks'].get('R_control_tcp_sync') or {}).get('arms') or {}).values()]
    sync = {'scenes_with_frames': len(sy), 'image_best_lag_by_scene': {str(l): sum(1 for s in sy if s['best_lag'] == l) for l in (-2, -1, 0, 1, 2, None)},
            'coordinates_px_median': {k: (float(np.median([s['coordinates_px_median'][k] for s in sy if s['coordinates_px_median'][k] is not None])) if any(s['coordinates_px_median'][k] is not None for s in sy) else None) for k in ('cubeA', 'cubeB')},
            'control_steps_tcp_observation_vs_trace': {'moves': len(ctl), 'max_difference_m': max(ctl) if ctl else None}}
    code = {f: sha(ROOT / f) for f in ('scripts/35_c1_trajectory_audit.py', 'src/hci/trajectory_audit.py', 'src/hci/vision_center_c1.py')}
    summary = _py({'what': 'post-hoc trajectory and behaviour audit of a C1 run folder; records only, no rollout', 'run_status': status, 'plan_sha256': plan_hash, 'policies': list(C1.POLICIES), 'arms': arms,
                   'planned_seeds': [seeds[0], seeds[-1], len(seeds)], 'records_audited': {pid: sum(1 for r in rows if r['policy'] == pid) for pid in C1.POLICIES},
                   'triggered': {pid: sum(1 for r in trig if r['policy'] == pid) for pid in C1.POLICIES}, 'check_counts': counts, 'record_check_counts': rcounts, 'fail_scenes': fails,
                   'cases': cases, 'figure_scenes': pick, 'figure_errors': ferr, 'sync': sync,
                   'thresholds': {k: getattr(T, k) for k in ('RESIDUAL_TOL_M', 'STEP_CMD_M', 'TARGET_CAP_M', 'MAX_STEPS', 'DIR_COS_MIN', 'PROGRESS_MIN_M', 'DZ_TOL_M', 'WIDTH_TOL_M', 'SLIP_TOL_M', 'PUSH_TOL_M', 'REVERT_FRACTION_MAX', 'SAME_TOL', 'MEMORY_ERROR_M')} | {'STATE_ATOL': STATE_ATOL},
                   'code_sha256': code, 'interpretation': 'COMPLETE only if every planned record of both policies exists, no stop or error file is present and every record check R_* passed, every selected figure was drawn and the run folder did not change during the audit; SUBSET_ONLY when --seeds narrowed the plan; anything else is INCOMPLETE and is not a result'})
    (out / 'audit_summary.json').write_bytes(json.dumps(summary, indent=1, sort_keys=True).encode('utf-8'))
    man = {'run_folder_files_sha256': before, 'run_folder_unchanged_by_audit': not changed, 'run_folder_changed_files': changed, 'outputs': {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
           'figures': figs, 'figure_rule': f'per policy: every scene of every risk rule, the lowest three of any_check_failed and of each representative rule, at most {figure_scenes} scenes',
           'plan': {'path': str(plan_path), 'sha256': plan_hash}, 'code_sha256': code}
    (out / 'trajectory_manifest.json').write_bytes(json.dumps(_py(man), indent=1, sort_keys=True).encode('utf-8'))
    if changed: raise SystemExit(f'the run folder changed during the audit; INCOMPLETE summary written to {out}')       # after the summary, so the status is on record
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True); p.add_argument('--plan', required=True); p.add_argument('--out', required=True); p.add_argument('--seeds')
    p.add_argument('--figure-scenes', type=int, default=6); p.add_argument('--no-figures', action='store_true')
    a = p.parse_args(argv); plan = json.loads(Path(a.plan).read_text(encoding='utf-8'))
    s = audit(a.run, a.plan, a.out, parse_seeds(a.seeds, plan) if a.seeds else None, a.figure_scenes, not a.no_figures)
    print(json.dumps({'run_status': s['run_status'], 'records_audited': s['records_audited'], 'triggered': s['triggered'], 'fail_scenes': s['fail_scenes']}, indent=1))
    return {'COMPLETE': 0, 'SUBSET_ONLY': 4}.get(s['run_status']['status'], 3)


if __name__ == '__main__':
    sys.exit(main())
