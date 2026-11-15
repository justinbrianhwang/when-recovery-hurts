"""Handoff H1: one-shot latest-observation padding at the handoff after R5 (work id handoff_h1_prep_001).
See amendments/AMENDMENT_011 (draft). PREPARATION ONLY: at the time of writing nothing below `dry-run` has been run.

Same fixed policy as Stage 2 (BELOW the original 50/100 gate; not lifted). Arms: continue (C0), r_keep (existing R5),
r_pad_once (the same R5, then the earlier observation-history entry replaced once by a copy of the latest).

  plan            local: hashes, scene list from the usage ledger, arms, analysis, authorization state (false)
  inspect-inputs  local: what the saved records contain about the handoff; no inference, no rollout
  dry-run         local: execution volume, required files and permissions; creates no server, starts no environment
  scenes | audit | collect | techcheck | scale | finalize
                  execution commands. Every one of them refuses unless <run>/authorization.json exists, names this
                  plan by hash and carries a spending cap. No command in this file writes that authorization.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, importlib.util, json, subprocess, time
import numpy as np
from hci.io import load_config, read_json, write_json, sha256, digest, runtime_info, seed_for, UPSTREAM_COMMIT

_spec = importlib.util.spec_from_file_location('stage2b_runner', Path(__file__).with_name('20_stage2b_pilot.py'))
s2b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s2b)
bound, write_csv, IDENTICAL_M, USER_POLICY_SHA = s2b.bound, s2b.write_csv, s2b.IDENTICAL_M, s2b.USER_POLICY_SHA

CODE = ['src/hci/handoff_h1.py', 'src/hci/pilot2d.py', 'src/hci/pilot2b.py', 'src/hci/pilot.py', 'src/hci/simulation.py', 'src/hci/policies.py', 'src/hci/experiment.py', 'src/hci/io.py',
        'src/hci/diagnostics.py', 'src/hci/schema.py', 'src/hci/trees.py', 'scripts/26_handoff_h1.py', 'scripts/20_stage2b_pilot.py', 'scripts/18_stage2a_pilot.py',
        'scripts/handoff_h1/run.sh', 'scripts/stage1b/common.sh']
UNCHANGED_SINCE_2D = ['src/hci/pilot2d.py', 'src/hci/pilot2b.py', 'src/hci/pilot.py', 'src/hci/simulation.py', 'src/hci/policies.py', 'src/hci/experiment.py', 'src/hci/io.py',
                      'src/hci/diagnostics.py', 'src/hci/schema.py', 'src/hci/trees.py', 'scripts/20_stage2b_pilot.py', 'scripts/18_stage2a_pilot.py', 'scripts/stage1b/common.sh']
STATUS = {'legacy_stage3b': 'COMPLETED_DEV_NO_GATE_PASS', 'legacy_model_search': 'CLOSED_FOR_THIS_CYCLE', 'base_policy_qualification': 'BELOW_ORIGINAL_GATE (37/100 < 50/100)',
          'handoff_h1': 'PREPARATION_ONLY', 'new_training': 'NOT_RUN', 'new_policy_inference': 'NOT_RUN', 'new_simulator_rollouts': 'NOT_RUN', 'paid_compute': 'NOT_AUTHORIZED',
          'calibration': 'NOT_RUN', 'final_test': 'NOT_RUN'}
PRIOR = {'stage2a': ('results/stage2a/scenes.json', 'list'), 'stage2b': ('results/stage2b/scenes.json', 'list'), 'stage2d': ('results/stage2d/scenes.json', 'list')}
EXECUTION = ('scenes', 'audit', 'collect', 'techcheck', 'scale', 'finalize')


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()


def h1():
    from hci import handoff_h1
    return handoff_h1


def require_authorization(run, plan_path):
    """Execution needs a user-written authorization that names this plan and a cap. Nothing in this file creates it."""
    p = Path(run) / 'authorization.json'
    if not p.exists(): raise SystemExit(f'Refused: no authorization ({p}). paid_compute: NOT_AUTHORIZED. Execution needs a separate explicit approval by the user.')
    a = read_json(p); cap = a.get('cap_usd')
    if a.get('approved_by') != 'user' or a.get('paid_compute_authorized') is not True or not isinstance(cap, (int, float)) or isinstance(cap, bool) or not cap > 0:
        raise SystemExit('Refused: the authorization is not an explicit user approval with a positive spending cap')
    if a.get('plan_sha256') != sha256(plan_path): raise SystemExit('Refused: the authorization belongs to another plan')
    return a


def ledger(splits, pe):
    """Every recorded use of a scene seed: declared splits, policy-evaluation sets, and the scenes of earlier pilots."""
    used = {**{f'splits/{k}': list(v) for k, v in splits['splits'].items() if k != 'train'}, **{f'policy_eval/{k}': list(v) for k, v in pe['splits'].items()}}
    for name, (path, key) in PRIOR.items(): used[f'{name}/exploratory_dev'] = list(read_json(path)[key])
    return used


def cmd_plan(a):
    from hci import pilot
    H = h1(); c = load_config(a.config); splits = read_json(a.splits); pe = read_json(a.policy_eval); prev = read_json(a.stage2d_plan); problems = []
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'scope_plan.json').exists(): raise SystemExit(f'Refused overwrite: {out / "scope_plan.json"}')
    used = ledger(splits, pe); keys2d = read_json('results/stage2d/reference_scene_keys.json'); sc2d = read_json('results/stage2d/scenes.json')
    demo = [int(k.split('/')[1]) for k in keys2d['keys'] if k.startswith('demonstration/')]
    chosen, skipped = pilot.select_scenes(splits['splits']['train'], a.scenes, {'splits/train': splits['splits']['train'], **used}, demo)
    pol = Path(a.policy); meta = read_json(pol / 'policy_meta.json'); w = sha256(pol / 'weights.pt'); m = sha256(pol / 'policy_meta.json')
    if (w, m) != (USER_POLICY_SHA['weights.pt'], USER_POLICY_SHA['policy_meta.json']) or (w, m) != (prev['policy']['weights_sha256'], prev['policy']['policy_meta_sha256']):
        problems.append('policy files differ from the Stage 2 policy on record')
    backup = prev['policy']['verified_against']['backup_list']; listed = {}
    for line in Path(backup).read_text().splitlines():
        hh, name = line.split(None, 1); listed[name.strip().lstrip('*')] = hh
    for name, hh in ((f'{pol.as_posix()}/weights.pt', w), (f'{pol.as_posix()}/policy_meta.json', m)):
        if listed.get(name) != hh: problems.append(f'{name}: local SHA256 differs from the preserved backup list')
    if meta['weights_sha256'] != w or meta['upstream_commit'] != UPSTREAM_COMMIT: problems.append('policy metadata does not belong to these weights or to the pinned upstream commit')
    if [meta['args'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')] != [2, 8, 16] or any(meta['args'][k] != c['policy'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')):
        problems.append('policy horizons are not 2 / 8 / 16')
    if digest(c) != prev['config']['config_sha256'] or sha256(a.config) != prev['config']['file_sha256']: problems.append('config differs from the Stage 2D config')
    for k in ('splits', 'policy_eval_splits'):
        if sha256(prev[k]['path']) != prev[k]['file_sha256']: problems.append(f'{k} changed since Stage 2D')
    cp = list(c['collection']['checkpoint_steps']); reps = list(range(c['collection']['repeats'])); rt = c['retreat']
    if cp != [12, 52, 92, 132] or reps != [0, 1, 2, 3]: problems.append('checkpoints / repeats are not the fixed [12,52,92,132] / [0,1,2,3]')
    if (rt['mode'], rt['distance_m'], rt['max_translation_per_step_m'], rt['max_steps'], rt['arrival_tolerance_m']) != ('up', 0.05, 0.005, 40, 0.005) or c['env']['horizon'] != 200:
        problems.append('R5 or the horizon differ from 5 cm / 5 mm per step / 40 steps / 5 mm tolerance / 200 steps')
    changed = [f for f in UNCHANGED_SINCE_2D if sha256(f) != prev['code'].get(f)]
    if changed: problems.append(f'files that Stage 2D ran with have changed: {changed}')
    again = sorted(set(chosen) & {s for v in used.values() for s in v})
    if again: problems.append(f'chosen scenes have a recorded use: {again}')
    keys = {**keys2d['keys'], **{f'stage2d/{s}': v['key'] for s, v in sc2d['nearest'].items()}}
    if not (out / 'reference_scene_keys.json').exists():
        write_json(out / 'reference_scene_keys.json', {'meaning': 'initial cubeA xyz, cubeB xyz of placements already on record (no environment was reset to make this file)',
                                                       'sources': {f: sha256(f) for f in ('results/stage2d/reference_scene_keys.json', 'results/stage2d/scenes.json')}, 'keys': keys})
    n = a.scenes; per = len(cp) * len(reps) * len(H.ARMS)
    plan = {'work_id': a.out.replace('\\', '/').rstrip('/').split('/')[-1], 'stage': 'Handoff H1: one-shot latest-observation padding at handoff', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'pass': not problems, 'problems': problems, 'status': STATUS, 'role': H.ROLE,
            'authorization': {'paid_compute_authorized': False, 'policy_inference_authorized': False, 'simulator_rollouts_authorized': False, 'required_file': 'authorization.json in the run folder, written '
                              'by the user side only: {"approved_by": "user", "paid_compute_authorized": true, "cap_usd": <number>, "plan_sha256": "<sha256 of this plan>"}',
                              'proposed_cap_usd': 1.00, 'proposed_cap_is': 'a budget proposal, not an approval; earlier stage caps are not reused'},
            'nature': 'Preparation of an exploratory development experiment, designed after the Stage 2B-3B results and an external feedback review were seen. Not a confirmatory test.',
            'question': 'With the fixed Diffusion Policy, right after R5 ends and with physical state, return step, remaining budget and policy noise held equal, does changing the observation '
                        'history of the first re-inference from [previous, latest] to [latest, latest] change the generated action and the later task outcome?',
            'neutral_name': 'one-shot latest-observation padding at handoff', 'hypothesis_status': '"history contamination" is an untested hypothesis; no improvement and no OOD cause is assumed',
            'repository': {'head': git('rev-parse', 'HEAD'), 'remote': git('remote', 'get-url', 'origin'), 'working_tree_changes_at_plan_time': git('status', '--porcelain').splitlines(),
                           'files_identical_to_the_stage2d_run': {f: sha256(f) == prev['code'].get(f) for f in UNCHANGED_SINCE_2D}},
            'policy': {'path': pol.as_posix(), 'weights_sha256': w, 'policy_meta_sha256': m, 'verified_against': {'backup_list': backup, 'backup_list_sha256': sha256(backup)},
                       'settings': {'obs_horizon': 2, 'act_horizon': 8, 'pred_horizon': 16, 'ddpm_steps': 100}, 'qualification': '37/100, below the 50/100 gate (unchanged)'},
            'config': {'path': a.config, 'file_sha256': sha256(a.config), 'config_sha256': digest(c), 'env': c['env'], 'protocol': c['protocol'], 'retreat': c['retreat'], 'audit': c['audit']},
            'code': {f: sha256(f) for f in CODE}, 'upstream_commit': UPSTREAM_COMMIT,
            'splits': {'path': a.splits, 'file_sha256': sha256(a.splits), 'digest': digest(splits)}, 'policy_eval_splits': {'path': a.policy_eval, 'file_sha256': sha256(a.policy_eval)},
            'scenes': {'rule': f'first {n} seeds of splits/train in ascending order with no recorded use (Stage 2A, 2B, 2D pilots, declared splits, policy-evaluation sets, demonstration seeds); a seed whose '
                               'initial cube placement equals a placement on record is replaced by the next train seed at the server step `scenes`. Outcomes are never an input.',
                       'list': chosen, 'seed_level_skipped_count': len(skipped), 'ledger': {k: [min(v), max(v), len(v)] for k, v in used.items()},
                       'excluded_prior_pilots': {k: used[f'{k}/exploratory_dev'] for k in PRIOR}, 'seed_level_skipped': skipped, 'stage2a_scenes_excluded': used['stage2a/exploratory_dev'],
                       'stage2b_scenes_excluded': used['stage2b/exploratory_dev'] + used['stage2d/exploratory_dev'],
                       'shrink_ladder': [12, 6, 2], 'shrink_rule': 'prefix of the list, from measured time and cost of the first two scenes only; never from outcomes',
                       'future_use': 'development data; never moved to calibration or Test', 'not_reused': 'earlier harm or rescue cases are not picked as the new main sample',
                       'calibration_and_test': 'compared by seed number only; never reset, read or run'},
            'audit': {'seeds': list(splits['splits']['audit']), 'use': 'engineering check; not a performance sample',
                      'protocol': 'Stage 2A audit with the same tolerances + R5 control-only reference + R_KEEP and R_PAD_ONCE to the end + return state against the reference before the '
                                  'history is touched + only the earlier history entry changed + same first inference step and seed + restore and trace after each branch'},
            'checkpoints': cp, 'repeats': reps,
            'arms': {'continue': {'short': 'C0', 'definition': 'existing Continue; queue and history untouched; context only'},
                     'r_keep': {'short': 'R_KEEP', 'definition': 'existing R5 through Runtime.force(2); the policy resumes on the real last two observations [o_(tau-1), o_tau]'},
                     'r_pad_once': {'short': 'R_PAD_ONCE', 'definition': 'the same R5; immediately before the first inference and only once, policy.history becomes [copy(o_tau), o_tau]; afterwards '
                                                                         'the ordinary observe() FIFO continues'},
                     'padding_rules': ['the whole converted observation entry the policy receives is copied (RGB of both cameras and qpos/qvel/tcp_pose); no partial ablation',
                                       'the latest entry is left untouched; the earlier entry is an independent copy (no shared memory)', 'no simulator, controller, raw-observation or object state is written',
                                       'qvel is not set to zero; a duplicated observation is not a physical stop', 'no env.step, no waiting, no hold steps are added',
                                       'queue (already empty after R5), last action, time, intervention record, inference count and RNG are identical in the two R5 arms',
                                       'policy.reset() is not used; only policy.history is assigned', 'if the episode ends during R5 nothing is padded and the policy is not called',
                                       'if the two last observations are identical the padding is a no-op; such blocks are kept'],
                     'primary_comparison': 'R_PAD_ONCE vs R_KEEP', 'context_comparisons': ['R_KEEP vs C0', 'R_PAD_ONCE vs C0']},
            'noise': {'seed_rule': "seed_for(scene, physical_step_of_the_call, block_repeat_id, 'dp') for every arm (unchanged code path in policies.DiffusionPolicy.generate)",
                      'handoff': 'R_KEEP and R_PAD_ONCE make their first inference at tau = checkpoint + R5 steps with the same seed through the same generate() call; equal actions for different inputs are allowed'},
            'max': {'scenes': n, 'snapshots': n * len(cp), 'blocks': n * len(cp) * len(reps), 'continuations': n * per, 'first_two_scenes_continuations': 2 * per,
                    'reference_control_runs': n * len(cp), 'extra_reproducibility_continuations_first_two_scenes': 2 * len(cp) * 2, 'trunks': n, 'audit_scenes': 5},
            'records': ['R5 end step, steps used, reached, displacement, remaining budget', 'history hashes before padding and the actual policy input after it, dtype and shape',
                        'handoff inputs (real last two observations and the padded pair) as a small NPZ per scene, written before any outcome', 'first inference step and seed, the generated chunk',
                        'outcome, end reason and step; per-step trace with gripper command and gripper joint positions', 'object pose and task flags in diagnostic files only'],
            'video_rule': 'first two scenes, first usable checkpoint, block repeat 0, the three arms (at most 6 files); nothing is added after outcomes are seen',
            'analysis': {'primary': 'Y_PAD - Y_KEEP within a block; mean over checkpoints and repeats within a scene, then equal weight per scene',
                         'reported': ['KEEP fail -> PAD success, KEEP success -> PAD fail, both succeed, both fail: blocks and scenes', 'per-scene differences and +/-/0 scene counts; block totals auxiliary',
                                      'first resumed chunk: mean and max absolute difference of the arm command, gripper sign differences (normalised controller units)',
                                      'blocks with a real handoff, with a changed input, no-op blocks', 'rescue / harm of KEEP and of PAD against C0 with numerator and denominator (NA if the denominator is 0)',
                                      'for ALL blocks grasped at the checkpoint: first open command, first grasp-flag release and outcome after the return, in all three arms'],
                         'not_computed': ['p-values', 'pass criteria', 'risk bounds', 'online selector H(g)'], 'independent_units': f'at most {n} scenes'},
            'stopping': 'one run and one report; no other history length, padding frequency, wait steps, model or distance is added afterwards; no automatic follow-up',
            'not_done_in_preparation': ['GPU rental', 'policy inference', 'simulator rollout', 'training', 'calibration', 'final Test']}
    write_json(out / 'scope_plan.json', plan); print(json.dumps({k: plan[k] for k in ('pass', 'problems')}, indent=1)); print('scenes', chosen)
    if problems: raise SystemExit(2)


def cmd_inspect_inputs(a):
    """What the saved records hold about the handoff. Reads files only."""
    out = Path(a.out); res = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'stages': {}}
    for st, run, arm in (('2B', 'runs/stage2b_001', 'R'), ('2D', 'runs/stage2d_001', 'R5')):
        inp = sorted((Path(run) / 'branches' / 'inputs').glob('*.npz')); tr = sorted((Path(run) / 'branches' / 'traces').glob('*.npz')); fields = set(); post = 0; n = 0
        for p in inp:
            with np.load(p) as z:
                for k in z.files:
                    t, f = k.split('_', 1); fields.add(f)
                    if f == 't': n += 1; post += int(z[k]) != int(t[1:])
        cols = []; keys = []
        if tr:
            with np.load(tr[0]) as z: cols = [str(x) for x in z['columns']]; keys = sorted({k.split('_')[0] if not k.startswith('b_') else 'b_<t>_<r>_' + k.split('_')[-1] for k in z.files})
        rec = read_json(sorted(Path(f'results/stage{st.lower()}/scenes').glob('*.json'))[0]); cell = next(c for c in rec['blocks'][0]['cells'] if c['arm'] == arm)
        res['stages'][st] = {'input_archives': len(inp), 'input_fields': sorted(fields), 'snapshots_in_inputs': n, 'snapshots_saved_at_another_step_than_the_checkpoint': post,
                             'inputs_are': 'pre-intervention snapshots only (t equals the checkpoint in every saved set)' if not post else 'CHECK',
                             'trace_archives': len(tr), 'trace_columns': cols, 'trace_arrays': keys, 'traces_hold_rgb_or_policy_history': False,
                             'r5_branch_fields_about_the_return': sorted(k for k in cell if 'history' in k or 'retreat' in k or 'return' in k or 'control' in k)}
    pol = Path(a.policy); w = sha256(pol / 'weights.pt') if (pol / 'weights.pt').exists() else None
    res['policy'] = {'path': pol.as_posix(), 'weights_present': w is not None, 'weights_sha256': w, 'matches_record': w == USER_POLICY_SHA['weights.pt'],
                     'metadata_matches_record': (pol / 'policy_meta.json').exists() and sha256(pol / 'policy_meta.json') == USER_POLICY_SHA['policy_meta.json']}
    res['post_handoff_observations_saved'] = False
    res['what_exists'] = ['pre-intervention online inputs per snapshot (two RGB frames per camera, qpos/qvel/tcp history, queue, last action)',
                          'executed actions and ground-truth step traces of the trunk and of every branch, including the R5 control steps',
                          'Stage 2D only: SHA256 of the policy history at the return step (a hash, not the observations)', 'policy weights and metadata with matching hashes']
    res['what_is_missing'] = ['the two observations (RGB and proprioception) of the policy history at the return step tau', 'the simulator state at tau as a restorable object']
    res['reconstruction'] = ('the return state can be reproduced only by replaying the logged action prefix on the simulator (reset(seed) + actions), which this free preparation stage does not do; '
                             'the first-chunk comparison also needs policy inference')
    res['verdict'] = 'REPLAY_REQUIRED'
    write_json(out / 'HANDOFF_INPUT_AVAILABILITY.json', res); print(json.dumps({k: res[k] for k in ('verdict', 'post_handoff_observations_saved', 'policy')}, indent=1))


def cmd_dry_run(a):
    """Volume, files and permissions. Starts nothing."""
    plan = read_json(a.plan); run = Path(a.run); m = plan['max']; need = sorted(set(plan['code']) | {plan['config']['path'], plan['splits']['path'], plan['policy_eval_splits']['path'],
                                                                                              plan['policy']['path'] + '/weights.pt', plan['policy']['path'] + '/policy_meta.json'})
    missing = [f for f in need if not Path(f).exists()]; changed = [f for f, hh in plan['code'].items() if Path(f).exists() and sha256(f) != hh]; auth = (run / 'authorization.json').exists()
    out = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha256(a.plan), 'scenes': plan['scenes']['list'],
           'maximum_runs': {'trunks': m['trunks'], 'continuations': m['continuations'], 'reference_r5_control_runs': m['reference_control_runs'],
                            'reruns_first_two_scenes': m['extra_reproducibility_continuations_first_two_scenes'], 'audit_scenes': m['audit_scenes'],
                            'full_runs_total_upper_bound': m['trunks'] + m['continuations'] + m['extra_reproducibility_continuations_first_two_scenes']},
           'files_needed_on_the_server': need, 'files_missing': missing, 'code_files_changed_since_plan': changed,
           'authorization_present': auth, 'execution_commands_would_run': False, 'paid_compute': 'NOT_AUTHORIZED' if not auth else 'AUTHORIZATION_FILE_PRESENT_NOT_USED_BY_DRY_RUN',
           'permissions_needed': ['explicit user approval of the paid run with a cap (authorization.json)', 'Vast.ai instance rental by the operator after that approval'],
           'servers_created': 0, 'environments_started': 0, 'policy_inferences': 0}
    print(json.dumps(out, indent=1))
    if a.out: write_json(a.out, out)
    if missing or changed: raise SystemExit(2)


def cmd_scenes(a):
    plan = read_json(a.plan); require_authorization(Path(a.out).parent, a.plan); s2d = _load('23_stage2d_pilot.py'); s2d.p2 = h1(); s2d.cmd_scenes(a)


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), Path(__file__).with_name(name)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def cmd_audit(a):
    H = h1(); plan = read_json(a.plan); require_authorization(Path(a.out).parent, a.plan); c = bound(plan, a)
    from hci.experiment import runtime, fingerprint
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); r = runtime(c, plan['policy']['path'], a.upstream, a.device); results = []
    try:
        for seed in plan['audit']['seeds']:
            tick = time.perf_counter(); results.append(H.audit_scene(r, seed, c, plan['checkpoints'])); results[-1]['wall_seconds'] = time.perf_counter() - tick
            print(f"audit seed={seed} reached={results[-1]['reached']} pass={results[-1]['pass']}", flush=True)
        bad = [(s['seed'], x['checkpoint'], v) for s in results for x in s['checkpoints'] if x['reached'] and x['handoff'] for v in x['handoff']['first_inference'].values()
               if v is not None and v[1] != seed_for(s['seed'], v[0], 0, 'dp')]
        report = {**H.audit_report(results), **fingerprint(c, plan['policy']['path']), 'runtime': runtime_info(), 'plan_sha256': sha256(a.plan), 'tolerances': c['audit'],
                  'restore_method': 'prefix action replay from reset(seed); set_state_dict is not used', 'first_inference_seeds_not_following_the_declared_key': bad,
                  'scope': 'Same-scene restore of this policy and config at the tested states and on this host; not proof of global determinism, not a performance sample.'}
        report['pass'] = bool(report['pass'] and not bad); write_json(out / 'audit.json', report); print(json.dumps({'pass': report['pass'], 'branches_executed': report['branches_executed']}))
        if not report['pass']: raise SystemExit(2)
    finally:
        r.env.close()


def cmd_collect(a):
    import imageio.v2 as imageio
    H = h1(); plan = read_json(a.plan); run = Path(a.run); require_authorization(run, a.plan); c = bound(plan, a); scenes = s2b.scene_list(a); pos = list(range(a.start, a.stop, a.step))
    if not pos or min(pos) < 0 or max(pos) >= len(scenes): raise SystemExit('positions outside the scene list')
    from hci.experiment import runtime, require_audit
    from hci.diagnostics import step_state, stage_events
    from hci.trees import as_numpy
    require_audit(run / 'audit' / 'audit.json', c, plan['policy']['path'])
    if read_json(run / 'audit' / 'audit.json')['plan_sha256'] != sha256(a.plan): raise SystemExit('audit belongs to another plan')
    if max(pos) >= 2:
        if not read_json(run / 'techcheck.json')['pass']: raise SystemExit('technical check of the first two scenes did not pass')
        if max(pos) >= read_json(run / 'scale_decision.json')['chosen_scenes']: raise SystemExit('positions beyond the decided scale')
    out = run / 'branches'
    for d in ('workers', 'traces', 'videos', 'inputs', 'handoff_inputs'): (out / d).mkdir(parents=True, exist_ok=True)
    status = out / 'workers' / f'{a.label}.json'; seeds = [scenes[i] for i in pos]
    write_json(status, {'label': a.label, 'positions': pos, 'scenes': seeds, 'completed': False, 'started_utc': datetime.now(timezone.utc).isoformat()})
    r = runtime(c, plan['policy']['path'], a.upstream, a.device); started = time.perf_counter(); extra = {}
    probe = lambda rt, z: {**step_state(rt.env, rt.t, z), 'gripper_qpos': as_numpy(rt.raw['agent']['qpos']).reshape(-1)[-2:].tolist()}
    frame = lambda rt: np.concatenate([as_numpy(rt.raw['sensor_data'][n]['rgb'])[0] for n in sorted(rt.raw['sensor_data'])], axis=1).astype(np.uint8)

    def run_scene(seed):
        first = scenes.index(seed) < 2
        rec, traces, videos, inputs, handoffs = H.collect_scene(r, seed, c, plan['checkpoints'], plan['repeats'], probe=probe, frame=frame, stage=stage_events, video=first, recheck=first)
        rec.update(position=scenes.index(seed), worker=a.label, policy_sha256=plan['policy']['weights_sha256'], policy_meta_sha256=plan['policy']['policy_meta_sha256'],
                   config_sha256=plan['config']['config_sha256'], code_sha256=digest(plan['code']), cameras=list(r.policy.cameras))
        extra[seed] = (inputs, handoffs); return rec, traces, videos

    def save(seed, rec, traces, videos):
        inputs, handoffs = extra.pop(seed)
        np.savez_compressed(out / 'traces' / f'{seed}.npz', columns=np.array(H.TRACE_COLUMNS), phases=np.array(H.PHASES), **traces)
        np.savez_compressed(out / 'inputs' / f'{seed}.npz', **{f't{t}_{k}': v for t, x in inputs.items() for k, v in x.items()})
        np.savez_compressed(out / 'handoff_inputs' / f'{seed}.npz', **{f't{t}_{k}': v for t, x in handoffs.items() for k, v in x.items()})
        rec['videos'] = []
        for name, frames in videos.items():
            path = out / 'videos' / f'{name}.mp4'
            if path.exists(): raise FileExistsError(f'Refusing overwrite: {path}')
            with imageio.get_writer(str(path), fps=20) as w:
                for f in frames: w.append_data(f)
            rec['videos'].append(path.as_posix())
        print(f"scene={seed} trunk={rec['trunk']['success']}@{rec['trunk']['steps']} snapshots={rec['counts']['snapshots']} continuations={rec['counts']['continuations']} "
              f"repro_ok={rec['reproducibility_ok']} wall={rec['wall_seconds']:.1f}s", flush=True)
    try:
        done = H.collect_scenes(seeds, out, run_scene, save)
        write_json(status, {'label': a.label, 'positions': pos, 'scenes': seeds, 'completed': True, 'finished_scenes': done, 'wall_seconds': time.perf_counter() - started}, overwrite=True)
    finally:
        r.env.close()


def files_ok(run, rec, first):
    H = h1(); p = run / 'branches' / 'traces' / f"{rec['scene']}.npz"; hp = run / 'branches' / 'handoff_inputs' / f"{rec['scene']}.npz"
    if not p.exists() or not hp.exists(): return False
    want = ({'columns', 'phases', 'trunk'} | {f"ref_{s['checkpoint']}" for s in rec['snapshots']} | {f"b_{b['checkpoint']}_{b['repeat']}_{c['arm']}" for b in rec['blocks'] for c in b['cells']})
    with np.load(p) as z:
        ok = set(z.files) == want and z['trunk'].shape[0] == rec['trunk']['steps'] and all(
            z[f"b_{b['checkpoint']}_{b['repeat']}_{c['arm']}"].shape[0] == c['steps'] - b['checkpoint'] for b in rec['blocks'] for c in b['cells'])
    with np.load(hp) as z:
        for s in rec['snapshots']:
            x = {k[len(f"t{s['checkpoint']}_"):]: z[k] for k in z.files if k.startswith(f"t{s['checkpoint']}_")}; ref = s['reference']
            ok = ok and (not x if ref['ended_during_control'] else H.tree_hash(x) == ref['handoff_inputs_sha256'])
            if x and 'actual_latest_rgb' in x:      # the padded pair is the latest observation twice; the real pair is kept under its own name
                ok = ok and np.array_equal(x['padded_prev1_rgb'], x['actual_latest_rgb']) and np.array_equal(x['padded_latest_state'], x['actual_latest_state']) and x['actual_latest_rgb'].dtype == np.uint8
    videos = rec.get('videos', [])
    return bool(ok and len(videos) == (len(H.ARMS) if first and rec['snapshots'] else 0) and all(Path(v).exists() and Path(v).stat().st_size > 0 for v in videos))


def first_two(a):
    scenes = s2b.scene_list(a)[:2]; return scenes, [read_json(Path(a.run) / 'branches' / 'scenes' / f'{s}.json') for s in scenes]


def cmd_techcheck(a):
    H = h1(); plan = read_json(a.plan); run = Path(a.run); require_authorization(run, a.plan); c = load_config(plan['config']['path']); scenes, recs = first_two(a); k = {}
    blocks = [b for x in recs for b in x['blocks']]; cells = [(x['scene'], b, cell) for x in recs for b in x['blocks'] for cell in b['cells']]
    k['audit_pass'] = bool(read_json(run / 'audit' / 'audit.json')['pass']); k['records_valid'] = all(H.validate_scene(x, c) for x in recs)
    k['no_error_file'] = not list((run / 'branches' / 'scenes').glob('*.error.json'))
    k['continuations_within_limit'] = sum(x['counts']['continuations'] for x in recs) <= plan['max']['first_two_scenes_continuations']
    k['trunk_reproduced_by_continue_repeat0'] = all(d['match'] for x in recs for d in x['no_intervention_duplicates'])
    k['reproducibility_checks_present_and_matched'] = all(not H.repro_problems(x, True) for x in recs)
    k['every_r5_branch_reproduces_the_reference_return_state'] = all(cell['return_state_equals_reference']['pass'] for s, b, cell in cells if cell['arm'] != 'C0')
    k['padding_only_in_r_pad_once_and_only_on_history'] = all((cell['padding_applied'] and cell['only_history_changed']['pass']) if (cell['arm'] == 'R_PAD_ONCE' and cell['policy_resumed'])
                                                              else not cell['padding_applied'] for s, b, cell in cells)
    k['handoff_as_declared'] = all(b['handoff']['pass'] for b in blocks)
    k['every_inference_seed_follows_the_declared_key'] = all(q['seed'] == seed_for(s, q['t'], cell['policy_noise_repeat_id'], 'dp') for s, b, cell in cells for q in cell['inference_calls'])
    k['traces_handoff_inputs_and_videos_written'] = all(files_ok(run, x, True) for x in recs); k['online_inputs_saved_once_per_snapshot'] = all(s2b.inputs_ok(run, x) for x in recs)
    report = {'pass': all(k.values()), 'checks': k, 'scenes': scenes, 'snapshots': [x['counts']['snapshots'] for x in recs], 'continuations': [x['counts']['continuations'] for x in recs],
              'scope': 'Technical conditions only. Which arm succeeded is not read here and does not decide whether collection goes on.'}
    write_json(run / 'techcheck.json', report); print(json.dumps(report, indent=1))
    if not report['pass']: raise SystemExit(2)


def cmd_scale(a):
    H = h1(); run = Path(a.run); plan = read_json(a.plan); auth = require_authorization(run, a.plan); scenes, recs = first_two(a)
    if not read_json(run / 'techcheck.json')['pass']: raise SystemExit('technical check did not pass')
    n = sum(x['counts']['continuations'] + x['counts']['extra_reproducibility_continuations'] + x['counts']['reference_control_runs'] + 1 for x in recs)
    per = len(plan['checkpoints']) * (len(plan['repeats']) * len(H.ARMS) + 1) + 1
    d = H.scale_decision([x['wall_seconds'] for x in recs], n, usd_per_hour=a.usd_per_hour, spent_usd=a.spent_usd, cap_usd=float(auth['cap_usd']), reserve_usd=a.reserve_usd, workers=a.workers,
                         efficiency=a.efficiency, ladder=tuple(plan['scenes']['shrink_ladder']), per_scene_continuations=per)
    d.update(measured_scenes=scenes, cap_from='authorization.json'); write_json(run / 'scale_decision.json', d); print(json.dumps(d, indent=1))


def event_rows(run, recs):
    H = h1(); rows = []
    for rec in recs:
        z0 = float(rec['initial_state']['cubeA'][2])
        with np.load(run / 'branches' / 'traces' / f"{rec['scene']}.npz") as z:
            for b in rec['blocks']:
                t = b['checkpoint']; cp = z['trunk'][t - 1:t]
                for cell in b['cells']:
                    d = H.derived(np.concatenate([cp, z[f"b_{t}_{b['repeat']}_{cell['arm']}"]]), z0)
                    rows.append({'scene': rec['scene'], 'checkpoint': t, 'block_repeat_id': b['repeat'], 'arm': cell['arm'], 'y': cell['y'], **H.event_reading(d, cell['r5_steps'] or 0)})
    return rows


def cmd_finalize(a):
    H = h1(); plan = read_json(a.plan); run = Path(a.run); require_authorization(run, a.plan); c = bound(plan, a); scenes = s2b.scene_list(a)
    decided = (run / 'scale_decision.json').exists(); n = read_json(run / 'scale_decision.json')['chosen_scenes'] if decided else 2
    meta, recs = H.finalize(run / 'branches', scenes[:n], c, rerun_scenes=scenes[:2]); usable = []
    if not decided or n not in plan['scenes']['shrink_ladder']: meta['problems'].append('no valid scale decision'); meta['completed'] = False
    if not ((run / 'techcheck.json').exists() and read_json(run / 'techcheck.json')['pass']): meta['problems'].append('technical check missing or failed'); meta['completed'] = False
    for x in recs:
        if (x.get('policy_sha256'), x.get('config_sha256'), x.get('code_sha256')) != (plan['policy']['weights_sha256'], plan['config']['config_sha256'], digest(plan['code'])):
            meta['problems'].append(f"scene {x['scene']}: recorded with another policy, config or code"); meta['completed'] = False
        if not files_ok(run, x, first=x['scene'] in scenes[:2]) or not s2b.inputs_ok(run, x): meta['problems'].append(f"scene {x['scene']}: saved files missing or inconsistent"); meta['completed'] = False
        else: usable.append(x)
        bad = [q for b in x['blocks'] for cell in b['cells'] for q in cell['inference_calls'] if q['seed'] != seed_for(x['scene'], q['t'], cell['policy_noise_repeat_id'], 'dp')]
        if bad: meta['problems'].append(f"scene {x['scene']}: {len(bad)} inference seeds do not follow the declared key"); meta['completed'] = False
    meta.update(planned_scenes=len(scenes), decided_scenes=n, plan_sha256=sha256(a.plan), status={**plan['status'], 'handoff_h1': 'EXECUTED_ONCE'}, source_type='real_sim', runtime=runtime_info())
    meta['ready_for_interpretation'] = bool(meta['completed'] and meta['reproducibility_ok']); write_json(run / 'branches' / 'dataset.json', meta)
    res = Path(a.results); summary = H.summarize(recs, planned_scenes=len(scenes)); rows = H.block_rows(recs); ev = event_rows(run, usable)
    summary.update(decided_scenes=n, dataset_completed=meta['completed'], ready_for_interpretation=meta['ready_for_interpretation'], problems=meta['problems'], event_rows=len(ev))
    write_csv(res / 'pair_outcomes.csv', rows); write_csv(res / 'pair_summary.csv', H.pair_rows(rows)); write_csv(res / 'scene_summary.csv', H.scene_rows(recs, rows))
    write_csv(res / 'event_rows.csv', ev); write_csv(res / 'skip_reasons.csv', H.skip_rows(recs)); write_json(res / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('scenes_recorded', 'snapshots', 'blocks', 'continuations', 'dataset_completed', 'ready_for_interpretation', 'problems', 'handoff')}, indent=1))
    if not meta['ready_for_interpretation']: raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--out', required=True); q.add_argument('--config', default='configs/stackcube.yaml')
    q.add_argument('--policy', default='artifacts/policies/stack_1d_s1_d500_clean_v1_i080000'); q.add_argument('--splits', default='configs/splits.json')
    q.add_argument('--policy-eval', default='configs/policy_eval_splits.json'); q.add_argument('--scenes', type=int, default=12); q.add_argument('--stage2d-plan', default='results/stage2d/plan.json')
    q = s.add_parser('inspect-inputs'); q.add_argument('--out', required=True); q.add_argument('--policy', default='artifacts/policies/stack_1d_s1_d500_clean_v1_i080000')
    q = s.add_parser('dry-run'); q.add_argument('--plan', required=True); q.add_argument('--run', required=True); q.add_argument('--out')
    for name in EXECUTION:
        q = s.add_parser(name); q.add_argument('--plan', required=True)
        if name in ('audit', 'collect'): q.add_argument('--upstream', default='third_party/ManiSkill'); q.add_argument('--device', default='cuda')
        if name == 'scenes': q.add_argument('--keys', required=True); q.add_argument('--out', required=True)
        elif name == 'audit': q.add_argument('--out', required=True)
        else: q.add_argument('--run', required=True)
        if name == 'collect':
            q.add_argument('--start', type=int, required=True); q.add_argument('--stop', type=int, required=True); q.add_argument('--step', type=int, default=1); q.add_argument('--label', required=True)
        if name == 'scale':
            q.add_argument('--usd-per-hour', type=float, required=True); q.add_argument('--spent-usd', type=float, required=True)
            q.add_argument('--reserve-usd', type=float, default=0.20); q.add_argument('--workers', type=int, default=2); q.add_argument('--efficiency', type=float, default=0.7)
        if name == 'finalize': q.add_argument('--results', required=True)
    a = p.parse_args(); globals()['cmd_' + a.cmd.replace('-', '_')](a)


if __name__ == '__main__':
    main()
