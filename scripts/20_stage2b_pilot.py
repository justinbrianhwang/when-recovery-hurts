"""Stage 2B exploratory four-arm control (run id stage2b_001). Development only; see amendments/AMENDMENT_006.

Same fixed policy as Stage 2A (BELOW the original 50/100 qualification gate; the gate is not lifted anywhere).
New train scenes, the three Stage 2A choices plus `continue_alt_noise` (the same Continue rule with another future
policy noise key). No training, no qualification, no predictor, no calibration, no Test.

  plan       local, before a GPU is rented: hashes, new scene list, arms and noise keys, limits
  scenes     server: initial cube placement of the new scenes against every placement already on record
  audit      server: full-state restore audit with this policy on this host, including the alt-noise branch
  collect    server: four-arm blocks of the scenes at the given positions of the list
  techcheck  technical check of the first two scenes (never reads which arm succeeded)
  scale      24 -> 12 -> 2 from measured time and money only
  finalize   dataset.json and the descriptive tables
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, importlib.util, json, subprocess, time
import numpy as np
from hci.io import load_config, read_json, write_json, sha256, digest, runtime_info, seed_for, UPSTREAM_COMMIT
from hci import pilot, pilot2b as p2

_spec = importlib.util.spec_from_file_location('stage2a_runner', Path(__file__).with_name('18_stage2a_pilot.py'))
s2a = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s2a)
bound, write_csv, IDENTICAL_M = s2a.bound, s2a.write_csv, s2a.IDENTICAL_M

CODE = ['src/hci/pilot2b.py', 'src/hci/pilot.py', 'src/hci/simulation.py', 'src/hci/policies.py', 'src/hci/experiment.py', 'src/hci/io.py',
        'src/hci/diagnostics.py', 'src/hci/schema.py', 'src/hci/trees.py', 'scripts/20_stage2b_pilot.py', 'scripts/18_stage2a_pilot.py', 'scripts/stage2b/run.sh']
STATUS = {'base_policy_qualification': 'BELOW_ORIGINAL_GATE (37/100 < 50/100)', 'stage2a': 'COMPLETED_EXPLORATORY_ONLY', 'stage2b': 'EXPLORATORY_FOUR_ARM_CONTROL',
          'formal_study_ready': 'NOT_ESTABLISHED', 'calibration': 'NOT_RUN', 'final_test': 'NOT_RUN'}
NOT_RUN = ['base policy retraining', 'checkpoint re-selection', 'qualification A/B rerun', 'Retreat change', 'action chunk / inference step change',
           'Q / Rescue / Harm predictor training', 'online selector evaluation', 'LTT / calibration', 'final Test', '5,760-continuation collection']
USER_POLICY_SHA = {'weights.pt': 'faacd2d86464c725a4fb95563a5ed5808019a8b420795b9a2a931822ed89db0e',
                   'policy_meta.json': '9976053598632fd58c48b89c9dffabf8f97b3d18ccf5062127c9c0be3de8a468'}
FIELDS = {'online_inputs_file': 'branches/inputs/<scene>.npz, arrays named t<checkpoint>_<field>; written once per snapshot before any continuation is run',
          'online_inputs': {'rgb_history': 'uint8 [obs_horizon, H, W, 3 x cameras]; cameras in sorted name order, exactly what the policy receives',
                            'state_history': 'float32 [obs_horizon, D]; qpos, qvel, tcp_pose (the whitelist of src/hci/schema.py)',
                            'last_action': 'float32 [4]; last executed command', 'queue': 'float32 [n, 4]; unexecuted commands of the current chunk',
                            't': 'physical step of the snapshot', 'remaining_budget': 'horizon - t',
                            'policy_feature': 'float32; fixed encoder output of the frozen policy for the same history, + last action, t/horizon, queue fraction'},
          'diagnostic_privileged_only': {'scene JSON: initial_state, snapshots[].state_at_checkpoint, snapshots[].stage_so_far, cells[].stage':
                                             'cube poses, grasp/on-cube/static flags, stage labels read from simulator ground truth',
                                         'branches/traces/<scene>.npz': 'per-step TCP, cube positions, flags, success, executed action',
                                         'y_C0, y_C1, y_F, y_R and every pair class': 'future outcomes (labels)'},
          'rule': 'Nothing of the second group is written into the online input file, and none of it is given to the policy. No model was trained in this stage.'}


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()


def cmd_plan(a):
    c = load_config(a.config); splits = read_json(a.splits); pe = read_json(a.policy_eval); prev = read_json(a.stage2a_plan); problems = []
    s2a_run = Path(a.stage2a_plan).parent; s2a_scenes = read_json(s2a_run / 'scenes.json'); demo = read_json(s2a_run / 'demo_scene_keys.json')
    s2a_data = read_json(s2a_run / 'branches' / 'dataset.json'); dm = read_json(prev['data_provenance']['manifest'])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    if s2a_scenes['plan_sha256'] != sha256(a.stage2a_plan) or s2a_scenes['list'] != prev['scenes']['list']: problems.append('Stage 2A scenes.json does not belong to the Stage 2A plan')
    used_2a = list(s2a_scenes['list'])
    if s2a_data['recorded_scenes'] != used_2a: problems.append('Stage 2A dataset does not list the Stage 2A scenes')
    declared = {**{f'splits/{k}': v for k, v in splits['splits'].items()}, **{f'policy_eval/{k}': v for k, v in pe['splits'].items()},
                f'stage2a/{pilot.ROLE}': used_2a}
    chosen, skipped = pilot.select_scenes(splits['splits']['train'], a.scenes, declared, [int(k) for k in demo['keys']])
    pol = Path(a.policy); meta = read_json(pol / 'policy_meta.json'); w = sha256(pol / 'weights.pt'); m = sha256(pol / 'policy_meta.json')
    if (w, m) != (USER_POLICY_SHA['weights.pt'], USER_POLICY_SHA['policy_meta.json']): problems.append('policy files differ from the SHA256 given in the Stage 2B specification')
    if (w, m) != (prev['policy']['weights_sha256'], prev['policy']['policy_meta_sha256']): problems.append('policy differs from the Stage 2A policy')
    backup = prev['policy']['verified_against']['file']; listed = {}
    for line in Path(backup).read_text().splitlines():
        h, name = line.split(None, 1); listed[name.strip().lstrip('*')] = h
    for name, h in ((f'{pol.as_posix()}/weights.pt', w), (f'{pol.as_posix()}/policy_meta.json', m)):
        if listed.get(name) != h: problems.append(f'{name}: local SHA256 differs from the preserved backup list')
    if meta['weights_sha256'] != w: problems.append('policy_meta weights_sha256 differs from weights.pt')
    if meta['upstream_commit'] != UPSTREAM_COMMIT: problems.append('policy was trained on another upstream commit')
    for k in ('obs_horizon', 'act_horizon', 'pred_horizon'):
        if meta['args'][k] != c['policy'][k]: problems.append(f'policy {k} differs from the config')
    if digest(c) != prev['config']['config_sha256']: problems.append('config differs from the Stage 2A config')
    for k in ('splits', 'policy_eval_splits'):
        if sha256(prev[k]['path']) != prev[k]['file_sha256']: problems.append(f'{k} changed since Stage 2A')
    if sha256(prev['data_provenance']['manifest']) != prev['data_provenance']['manifest_sha256']: problems.append('data provenance manifest changed since Stage 2A')
    q = read_json(prev['policy']['qualification']['file'])
    if (q['policy_sha256'], q['success_once'], q['n']) != (w, 37, 100): problems.append('qualification record does not belong to this policy as 37/100')
    cp = list(c['collection']['checkpoint_steps']); reps = list(range(c['collection']['repeats']))
    if cp != [12, 52, 92, 132] or reps != [0, 1, 2, 3]: problems.append(f'config checkpoints/repeats are {cp}/{reps}, not the fixed [12,52,92,132]/[0,1,2,3]')
    prior = []
    for f in sorted(Path('runs').glob('*/dataset.json')) + sorted(Path('runs').glob('*/branches/dataset.json')):
        d = read_json(f); used = sorted((set(d.get('seeds', [])) | set(d.get('recorded_scenes', []))) & set(chosen))
        if used: prior.append({'file': f.as_posix(), 'scenes': used})
    if prior: problems.append(f'chosen scenes already appear in earlier runs: {prior}')
    keys = {**{f'demonstration/{s}': k for s, k in demo['keys'].items()}, **{f'stage2a/{s}': v['key'] for s, v in s2a_scenes['nearest'].items()}}
    write_json(out / 'reference_scene_keys.json', {'meaning': 'initial cubeA xyz, cubeB xyz of placements already on record (no environment was reset to make this file)',
                                                   'sources': {(s2a_run / 'demo_scene_keys.json').as_posix(): sha256(s2a_run / 'demo_scene_keys.json'),
                                                               (s2a_run / 'scenes.json').as_posix(): sha256(s2a_run / 'scenes.json')}, 'keys': keys})
    # the alternative noise key must give another seed than the block's own key at every step a call can happen
    H = c['env']['horizon']; same = 0; reused = 0; own = set()
    for s in chosen:
        for t in range(H):
            for r in reps: own.add(seed_for(s, t, r, 'dp'))
    for s in chosen:
        for t in range(H):
            for r in reps:
                alt = seed_for(s, t, p2.noise_id('continue_alt_noise', r), 'dp'); same += alt == seed_for(s, t, r, 'dp'); reused += alt in own
    if same or reused: problems.append(f'alternative noise seeds coincide with block seeds ({same} at the same step, {reused} anywhere)')
    n = a.scenes; arms = len(p2.ARMS)
    plan = {'run_id': a.out.replace('\\', '/').rstrip('/').split('/')[-1], 'stage': 'Stage 2B exploratory four-arm control', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'pass': not problems, 'problems': problems, 'status': STATUS, 'role': p2.ROLE,
            'nature': 'Development experiment designed after the Stage 2A results were seen. Not a pre-registered confirmatory test and not an independent validation.',
            'repository': {'head': git('rev-parse', 'HEAD'), 'working_tree_changes_at_plan_time': git('status', '--porcelain').splitlines(),
                           'note': 'the code that runs is fixed by the per-file SHA256 below, not by the commit'},
            'based_on': {'stage2a_plan': a.stage2a_plan, 'stage2a_plan_sha256': sha256(a.stage2a_plan), 'stage2a_dataset_sha256': sha256(s2a_run / 'branches' / 'dataset.json')},
            'policy': {'path': pol.as_posix(), 'weights_sha256': w, 'policy_meta_sha256': m, 'source_checkpoint_sha256': meta['source_checkpoint_sha256'],
                       'verified_against': {'specification': USER_POLICY_SHA, 'backup_list': backup, 'backup_list_sha256': sha256(backup), 'stage2a_plan': True},
                       'qualification': {'file': prev['policy']['qualification']['file'], 'set': q['set'], 'success_once': q['success_once'], 'n': q['n']},
                       'settings': {k: meta['args'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')}, 'unchanged': 'weights, schedule, DDPM steps, action clip'},
            'config': {'path': a.config, 'file_sha256': sha256(a.config), 'config_sha256': digest(c), 'env': c['env'], 'protocol': c['protocol'], 'retreat': c['retreat'], 'audit': c['audit']},
            'data_provenance': prev['data_provenance'], 'code': {f: sha256(f) for f in CODE}, 'upstream_commit': UPSTREAM_COMMIT,
            'splits': {'path': a.splits, 'file_sha256': sha256(a.splits), 'digest': digest(splits)}, 'policy_eval_splits': {'path': a.policy_eval, 'file_sha256': sha256(a.policy_eval)},
            'scenes': {'rule': f'first {n} seeds of splits/train in ascending order that are not Stage 2A scenes and are declared for no other use; a seed whose initial cube '
                               'placement equals a placement on record is replaced by the next train seed. Outcomes of this policy are never an input.',
                       'list': chosen, 'seed_level_skipped': skipped, 'stage2a_scenes_excluded': used_2a,
                       'scene_identity': f'server step `scenes` (cube placement within {IDENTICAL_M} m counts as identical); references: demonstrations and Stage 2A scenes '
                                         'from stored keys, audit/dev/policy-evaluation scenes by reset; calibration and Test by seed only, never reset',
                       'shrink_ladder': [24, 12, 2], 'shrink_rule': 'prefix of the list, chosen from measured time and cost of the first two scenes only',
                       'future_use': 'Stage 2A and Stage 2B scenes stay development data; they are never moved to calibration or Test'},
            'audit': {'seeds': list(splits['splits']['audit']), 'also_demonstration_scenes': sorted(set(splits['splits']['audit']) & set(dm['all_scene_seeds'])),
                      'use': 'engineering check of restore; not a performance sample and not a denominator',
                      'protocol': 'Stage 2A audit (same compared items, same tolerances) + a Continue-alt-noise branch run to the end before another restore and trace '
                                  '+ exact check that saving the policy inputs leaves the runtime unchanged'},
            'checkpoints': cp, 'repeats': reps,
            'arms': {arm: {'short': p2.SHORT[arm], 'runtime_force': p2.FORCE[arm], 'queue_at_snapshot': 'kept' if p2.FORCE[arm] is None else 'discarded',
                           'first_later_inference': {'continue': 'natural chunk boundary', 'continue_alt_noise': 'natural chunk boundary (same step as continue)',
                                                     'refresh': 'at the checkpoint', 'retreat_retry': 'after the retreat ends'}[arm],
                           'policy_noise_repeat_id': 'r + 4' if arm == 'continue_alt_noise' else 'r'} for arm in p2.ARMS},
            'noise': {'seed_rule': "seed_for(scene, physical_step_of_the_call, policy_noise_repeat_id, 'dp'), set immediately before each Agent.get_action inside "
                                   'torch.random.fork_rng (unchanged code path; covers the initial noise and the reverse-diffusion draws of that call)',
                      'block_repeat_id': 'pairing key r in [0,1,2,3]', 'alt_noise_ids': [p2.noise_id('continue_alt_noise', r) for r in reps],
                      'not_in_the_seed': ['execution order', 'worker id', 'wall-clock time', 'arm name'],
                      'seed_check': {'scenes': len(chosen), 'steps': H, 'alt_seed_equals_block_seed_at_same_step': same, 'alt_seed_equals_any_block_seed': reused},
                      'caveat': 'different integer seeds give another pseudo-random realisation; this is not a proof of statistical independence. '
                                'C0 and C1 may well have the same outcome; no check requires them to differ.'},
            'max': {'scenes': n, 'snapshots': n * len(cp), 'blocks': n * len(cp) * len(reps), 'continuations': n * len(cp) * len(reps) * arms,
                    'first_two_scenes_continuations': 2 * len(cp) * len(reps) * arms, 'extra_reproducibility_continuations_first_two_scenes': 2 * len(cp) * 2,
                    'trunks': n},
            'coupling': {'past': 'identical for the four arms: same scene seed, same executed action prefix (prefix replay), same policy history and queue',
                         'continue_alt_noise': 'restored and verified like every arm; then only Runtime.repeat is set to r + 4. No force, no clear, no extra inference, '
                                               'no extra action. Until the old queue is used up it must be the same rollout as continue (commands, simulator state, '
                                               'observation returned by the environment); a block that violates this is an error, not data',
                         'refresh_and_retreat': 'unchanged from Stage 2A: physical-step noise key r; calls at other steps than Continue draw other noise by design',
                         'reference': 'pairs are joined on (scene, checkpoint, block_repeat_id) only'},
            'reproducibility_checks': {'every_snapshot': 'continue with block repeat 0 must reproduce the never-restored trunk (actions, outcome, steps, final state)',
                                       'first_two_scenes': 'after all other branches of a snapshot: continue_alt_noise of repeat 0 once more, and continue of repeat 1 once more',
                                       'every_block': 'C0/C1 queue-prefix parity', 'on_mismatch': 'flagged, never subtracted; the data set is not marked complete'},
            'observation_fields': FIELDS,
            'video_rule': 'first two scenes, first usable checkpoint, block repeat 0, the four arms (at most 8 files); no outcome-selected example is added',
            'analysis': {'pairs': [f'{x} vs {y}' for x, y in p2.PAIRS], 'primary_summary': 'scene-equal weighting: mean over reached checkpoints and repeats within a scene, then over scenes',
                         'not_computed': ['p-values', 'confidence bounds', 'any threshold chosen after the results']},
            'budget': {'stage2b_cap_usd': 1.50, 'kept_separate_from': 'Stage 2A 0.38 USD and Stage 1D', 'cost_basis': 'credit difference, not hourly price x time'},
            'not_run_in_this_stage': NOT_RUN}
    write_json(out / 'plan.json', plan); print(json.dumps({k: plan[k] for k in ('pass', 'problems')}, indent=2)); print('scenes', chosen, 'skipped', skipped)
    if problems: raise SystemExit(2)


def cmd_scenes(a):
    plan = read_json(a.plan); c = bound(plan, a); splits = read_json(plan['splits']['path']); pe = read_json(plan['policy_eval_splits']['path'])
    from hci.simulation import make_env
    from hci.trees import as_numpy
    env = make_env(c)

    def key(seed):
        env.reset(seed=int(seed)); return np.array([*as_numpy(env.cubeA.pose.p)[0], *as_numpy(env.cubeB.pose.p)[0]], float)
    try:
        refs = {n: np.array(k) for n, k in read_json(a.keys)['keys'].items()}; stored = len(refs)
        reset_sets = {'splits/audit': splits['splits']['audit'], 'splits/dev': splits['splits']['dev'], **{f'policy_eval/{k}': v for k, v in pe['splits'].items()}}
        for name, ids in reset_sets.items():
            for s in ids: refs[f'{name}/{s}'] = key(s)
        names = list(refs); table = np.stack([refs[n] for n in names]); chosen = []; replaced = []; nearest = {}
        want = len(plan['scenes']['list']); excluded = {x['seed'] for x in plan['scenes']['seed_level_skipped']} | set(plan['scenes']['stage2a_scenes_excluded'])
        for s in sorted(splits['splits']['train']):
            if len(chosen) == want: break
            if s in excluded: continue
            k = key(s); d = np.abs(table - k).max(1); j = int(d.argmin())
            twin = [t for t in chosen if np.abs(np.array(nearest[str(t)]['key']) - k).max() <= IDENTICAL_M]
            if d[j] <= IDENTICAL_M or twin:
                replaced.append({'seed': s, 'reason': f'initial cube placement identical to {names[j] if d[j] <= IDENTICAL_M else twin[0]}'}); continue
            chosen.append(s); nearest[str(s)] = {'key': k.tolist(), 'nearest_reference': names[j], 'max_abs_difference_m': float(d[j])}
        if len(chosen) < want: raise SystemExit('train has too few usable scenes; nothing is borrowed from other splits')
        report = {'list': chosen, 'same_as_plan': chosen == plan['scenes']['list'], 'replaced': replaced, 'role': p2.ROLE, 'nearest': nearest, 'identical_threshold_m': IDENTICAL_M,
                  'references': {'stored_keys_demonstrations_and_stage2a': stored, **{k: len(v) for k, v in reset_sets.items()}},
                  'seed_level_only': ['splits/calibration', 'splits/test'], 'plan_sha256': sha256(a.plan),
                  'scope': 'Initial cube placement only. Decided before any rollout of the policy on these scenes.'}
        write_json(a.out, report); print(json.dumps({k: report[k] for k in ('list', 'same_as_plan', 'replaced')}))
    finally:
        env.close()


def cmd_audit(a):
    plan = read_json(a.plan); c = bound(plan, a)
    from hci.experiment import runtime, fingerprint
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); r = runtime(c, plan['policy']['path'], a.upstream, a.device); results = []
    try:
        for seed in plan['audit']['seeds']:
            tick = time.perf_counter(); results.append(p2.audit_scene(r, seed, c, plan['checkpoints'])); results[-1]['wall_seconds'] = time.perf_counter() - tick
            print(f"audit seed={seed} reached={results[-1]['reached']} pass={results[-1]['pass']}", flush=True)
        report = {**p2.audit_report(results), **fingerprint(c, plan['policy']['path']), 'runtime': runtime_info(), 'plan_sha256': sha256(a.plan),
                  'tolerances': c['audit'], 'restore_method': 'prefix action replay from reset(seed); set_state_dict is not used',
                  'audit_seeds_also_demonstration_scenes': plan['audit']['also_demonstration_scenes'],
                  'scope': 'Same-scene restore of this policy and config at the tested states; not proof of global simulator determinism, not a performance sample.'}
        write_json(out / 'audit.json', report); print(json.dumps({'pass': report['pass'], 'branches_executed': report['branches_executed']}))
        if not report['pass']: raise SystemExit(2)
    finally:
        r.env.close()


def scene_list(a):
    s = read_json(Path(a.run) / 'scenes.json')
    if s['plan_sha256'] != sha256(a.plan): raise SystemExit('scenes.json belongs to another plan')
    if not s['same_as_plan'] or s['list'] != read_json(a.plan)['scenes']['list']: raise SystemExit('scene list differs from the plan; a revised plan is required')
    return s['list']


def cmd_collect(a):
    import imageio.v2 as imageio
    plan = read_json(a.plan); c = bound(plan, a); run = Path(a.run); scenes = scene_list(a); pos = list(range(a.start, a.stop, a.step))
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
    for d in ('workers', 'traces', 'videos', 'inputs'): (out / d).mkdir(parents=True, exist_ok=True)
    status = out / 'workers' / f'{a.label}.json'; seeds = [scenes[i] for i in pos]
    write_json(status, {'label': a.label, 'positions': pos, 'scenes': seeds, 'completed': False, 'started_utc': datetime.now(timezone.utc).isoformat()})
    r = runtime(c, plan['policy']['path'], a.upstream, a.device); started = time.perf_counter()
    probe = lambda rt, z: step_state(rt.env, rt.t, z)
    frame = lambda rt: np.concatenate([as_numpy(rt.raw['sensor_data'][n]['rgb'])[0] for n in sorted(rt.raw['sensor_data'])], axis=1).astype(np.uint8)
    extra = {}

    def run_scene(seed):
        first = scenes.index(seed) < 2
        rec, traces, videos, inputs = p2.collect_scene(r, seed, c, plan['checkpoints'], plan['repeats'], probe=probe, frame=frame, stage=stage_events,
                                                       video=first, recheck=first)
        rec.update(position=scenes.index(seed), worker=a.label, policy_sha256=plan['policy']['weights_sha256'], policy_meta_sha256=plan['policy']['policy_meta_sha256'],
                   config_sha256=plan['config']['config_sha256'], code_sha256=digest(plan['code']), cameras=list(r.policy.cameras))
        extra[seed] = inputs; return rec, traces, videos

    def save(seed, rec, traces, videos):
        np.savez_compressed(out / 'traces' / f'{seed}.npz', columns=np.array(pilot.TRACE_COLUMNS), **traces)
        np.savez_compressed(out / 'inputs' / f'{seed}.npz', **{f't{t}_{k}': v for t, x in extra.pop(seed).items() for k, v in x.items()})
        rec['videos'] = []
        for name, frames in videos.items():
            path = out / 'videos' / f'{name}.mp4'
            if path.exists(): raise FileExistsError(f'Refusing overwrite: {path}')
            with imageio.get_writer(str(path), fps=20) as w:
                for f in frames: w.append_data(f)
            rec['videos'].append(path.as_posix())
        print(f"scene={seed} trunk={rec['trunk']['success']}@{rec['trunk']['steps']} snapshots={rec['counts']['snapshots']} "
              f"continuations={rec['counts']['continuations']} repro_ok={rec['reproducibility_ok']} wall={rec['wall_seconds']:.1f}s", flush=True)
    try:
        done = p2.collect_scenes(seeds, out, run_scene, save)
        write_json(status, {'label': a.label, 'positions': pos, 'scenes': seeds, 'completed': True, 'finished_scenes': done,
                            'wall_seconds': time.perf_counter() - started}, overwrite=True)
    finally:
        r.env.close()


def first_two(a):
    scenes = scene_list(a)[:2]; run = Path(a.run)
    return scenes, [read_json(run / 'branches' / 'scenes' / f'{s}.json') for s in scenes]


def inputs_ok(run, rec):
    """The saved online inputs of a scene: one set per snapshot, uint8 RGB, identical to what was hashed at the snapshot."""
    fields = set(FIELDS['online_inputs'])
    with np.load(run / 'branches' / 'inputs' / f"{rec['scene']}.npz") as z:
        names = set(z.files); ok = True
        for s in rec['snapshots']:
            pre = f"t{s['checkpoint']}_"; x = {k[len(pre):]: z[k] for k in names if k.startswith(pre)}
            ok &= set(x) == fields and x['rgb_history'].dtype == np.uint8 and int(x['t']) == s['checkpoint'] and len(x['queue']) == s['queue_len']
            ok &= pilot.tree_hash(x) == s['policy_inputs_sha256']
        ok &= len(names) == len(fields) * len(rec['snapshots'])
    return bool(ok)


def files_ok(run, rec, first):
    """Trace archive with the trunk and every branch of the record; the videos of the record exist and are the declared number."""
    p = run / 'branches' / 'traces' / f"{rec['scene']}.npz"
    if not p.exists(): return False
    want = {'columns', 'trunk'} | {f"b_{b['checkpoint']}_{b['repeat']}_{cell['arm']}" for b in rec['blocks'] for cell in b['cells']}
    with np.load(p) as z:
        ok = set(z.files) == want and all(z[f"b_{b['checkpoint']}_{b['repeat']}_{cell['arm']}"].shape[0] == cell['steps'] - b['checkpoint']
                                          for b in rec['blocks'] for cell in b['cells']) and z['trunk'].shape[0] == rec['trunk']['steps']
    videos = rec.get('videos', [])
    return bool(ok and len(videos) == (len(p2.ARMS) if first and rec['snapshots'] else 0) and all(Path(v).exists() and Path(v).stat().st_size > 0 for v in videos))


def cmd_techcheck(a):
    plan = read_json(a.plan); c = load_config(plan['config']['path']); run = Path(a.run); scenes, recs = first_two(a); checks = {}; rt = c['retreat']
    blocks = [b for x in recs for b in x['blocks']]; cells = [(x['scene'], b, cell) for x in recs for b in x['blocks'] for cell in b['cells']]
    checks['audit_pass'] = bool(read_json(run / 'audit' / 'audit.json')['pass'])
    checks['records_valid'] = all(p2.validate_scene(x, c) for x in recs)
    checks['no_error_file'] = not list((run / 'branches' / 'scenes').glob('*.error.json'))
    checks['continuations_within_limit'] = sum(x['counts']['continuations'] for x in recs) <= plan['max']['first_two_scenes_continuations']
    checks['trunk_reproduced_by_continue_repeat0'] = all(d['match'] for x in recs for d in x['no_intervention_duplicates'])
    checks['alt_noise_rerun_identical'] = all(d['match'] for x in recs for d in x['rechecks'] if d['experimental_arm'] == 'continue_alt_noise')
    checks['continue_rerun_identical'] = all(d['match'] for x in recs for d in x['rechecks'] if d['experimental_arm'] == 'continue')
    checks['reproducibility_checks_present_and_matched'] = all(not p2.repro_problems(x, True) for x in recs)
    checks['c0_c1_same_rollout_until_queue_is_used_up'] = all(b['c0_c1_queue_prefix_parity']['pass'] for b in blocks)
    # a block that ends while the old queue is executed has no later inference; it is identical in C0 and C1 and is not an error
    checks['alt_noise_seed_differs_at_first_inference'] = all(b['c0_c1_queue_prefix_parity']['first_inference_seeds_differ'] in (True, None) for b in blocks)
    checks['every_inference_seed_follows_the_declared_key'] = all(k['seed'] == seed_for(s, k['t'], cell['policy_noise_repeat_id'], 'dp')
                                                                  for s, b, cell in cells for k in cell['inference_calls'])
    checks['trunk_seeds_follow_repeat0'] = all(k['seed'] == seed_for(x['scene'], k['t'], 0, 'dp') and k['policy_noise_repeat_id'] == 0 for x in recs for k in x['trunk']['inference_calls'])
    checks['alt_noise_is_no_intervention'] = all(cell['intervention'] is None and cell['retreat'] is None for s, b, cell in cells if cell['experimental_arm'] == 'continue_alt_noise')
    checks['traces_and_videos_written'] = all(files_ok(run, x, True) for x in recs)
    checks['online_inputs_saved_once_per_snapshot'] = all(inputs_ok(run, x) for x in recs)
    real = True; grip = True; refresh = True; budget = True; same_start = True
    for x in recs:
        with np.load(run / 'branches' / 'traces' / f"{x['scene']}.npz") as z:
            for b in x['blocks']:
                t = b['checkpoint']; cell = {v['arm']: v for v in b['cells']}; ret = cell['R']; info = ret['retreat']; k = info['steps_used']; rows = z[f"b_{t}_{b['repeat']}_R"]
                # the retreat is made of executed control steps inside the same horizon, with the last gripper command held
                real &= rows.shape[0] == ret['steps'] - t and k <= rt['max_steps'] and (k > 0 or info['end_reason'] != 'budget_exhausted')
                real &= bool(np.all(np.abs(rows[:k, 14:17]).max(1) > 0)) if k else True
                grip &= bool(np.all(rows[:k, 17] == np.float32(info['gripper_command'])))
                refresh &= z[f"b_{t}_{b['repeat']}_F"].shape[0] == cell['F']['steps'] - t and cell['F']['policy_calls_after_snapshot'] >= 1
                budget &= all(v['steps'] <= c['env']['horizon'] for v in b['cells'])
                n = b['c0_c1_queue_prefix_parity']['compared_steps']; c0 = z[f"b_{t}_{b['repeat']}_C0"]; c1 = z[f"b_{t}_{b['repeat']}_C1"]
                same_start &= bool(np.array_equal(c0[:n], c1[:n], equal_nan=True))      # ground-truth log of the shared part, bit for bit
    checks.update(retreat_is_executed_control_steps=bool(real), gripper_command_held_during_retreat=bool(grip), refresh_infers_at_the_checkpoint=bool(refresh),
                  horizon_respected=bool(budget), c0_c1_ground_truth_trace_equal_until_queue_is_used_up=bool(same_start))
    report = {'pass': all(checks.values()), 'checks': checks, 'scenes': scenes, 'snapshots': [x['counts']['snapshots'] for x in recs],
              'continuations': [x['counts']['continuations'] for x in recs],
              'scope': 'Technical conditions only. Which arm succeeded is not read here and does not decide whether collection goes on.'}
    write_json(run / 'techcheck.json', report); print(json.dumps(report, indent=2))
    if not report['pass']: raise SystemExit(2)


def cmd_scale(a):
    run = Path(a.run); plan = read_json(a.plan); scenes, recs = first_two(a)
    if not read_json(run / 'techcheck.json')['pass']: raise SystemExit('technical check did not pass')
    n = sum(x['counts']['continuations'] + x['counts']['extra_reproducibility_continuations'] + 1 for x in recs)
    per_scene = len(plan['checkpoints']) * len(plan['repeats']) * len(p2.ARMS) + 2
    d = p2.scale_decision([x['wall_seconds'] for x in recs], n, usd_per_hour=a.usd_per_hour, spent_usd=a.spent_usd, cap_usd=plan['budget']['stage2b_cap_usd'],
                          reserve_usd=a.reserve_usd, workers=a.workers, efficiency=a.efficiency, ladder=tuple(plan['scenes']['shrink_ladder']),
                          per_scene_continuations=per_scene)
    d.update(measured_scenes=scenes, measured_wall_seconds=[x['wall_seconds'] for x in recs], measured_continuations_including_trunk=n)
    write_json(run / 'scale_decision.json', d); print(json.dumps(d, indent=2))


def cmd_finalize(a):
    plan = read_json(a.plan); c = bound(plan, a); run = Path(a.run); scenes = scene_list(a)
    decided = (run / 'scale_decision.json').exists(); n = read_json(run / 'scale_decision.json')['chosen_scenes'] if decided else 2
    meta, recs = p2.finalize(run / 'branches', scenes[:n], c, rerun_scenes=scenes[:2])
    if not decided or n not in plan['scenes']['shrink_ladder']:      # tables of what exists are still written; the data set is not complete
        meta['problems'].append('no valid scale decision: only the first two scenes were considered'); meta['completed'] = False
    if not ((run / 'techcheck.json').exists() and read_json(run / 'techcheck.json')['pass']):
        meta['problems'].append('technical check of the first two scenes is missing or did not pass'); meta['completed'] = False
    for x in recs:
        if (x.get('policy_sha256'), x.get('config_sha256'), x.get('code_sha256')) != (plan['policy']['weights_sha256'], plan['config']['config_sha256'], digest(plan['code'])):
            meta['problems'].append(f"scene {x['scene']}: recorded with another policy, config or code"); meta['completed'] = False
        if not (run / 'branches' / 'inputs' / f"{x['scene']}.npz").exists() or not inputs_ok(run, x):
            meta['problems'].append(f"scene {x['scene']}: saved online inputs are missing or differ from the snapshot"); meta['completed'] = False
        if not files_ok(run, x, first=x['scene'] in scenes[:2]):
            meta['problems'].append(f"scene {x['scene']}: trace archive or videos are missing or incomplete"); meta['completed'] = False
        bad = [k for b in x['blocks'] for cell in b['cells'] for k in cell['inference_calls'] if k['seed'] != seed_for(x['scene'], k['t'], cell['policy_noise_repeat_id'], 'dp')]
        if bad: meta['problems'].append(f"scene {x['scene']}: {len(bad)} inference seeds do not follow the declared key"); meta['completed'] = False
    meta.update(planned_scenes=len(scenes), decided_scenes=n, policy_sha256=plan['policy']['weights_sha256'], config_sha256=plan['config']['config_sha256'],
                plan_sha256=sha256(a.plan), status=plan['status'], coupling=plan['coupling'], noise=plan['noise'], source_type='real_sim',
                success_definition=c['protocol']['success_definition'], runtime=runtime_info(), not_run_in_this_stage=NOT_RUN)
    res = Path(a.results); summary = p2.summarize(recs, planned_scenes=len(scenes))
    meta['ready_for_interpretation'] = bool(meta['completed'] and meta['reproducibility_ok'])
    write_json(run / 'branches' / 'dataset.json', meta)
    summary.update(decided_scenes=n, dataset_completed=meta['completed'], ready_for_interpretation=meta['ready_for_interpretation'], problems=meta['problems'], status=plan['status'])
    rows = p2.block_rows(recs)
    write_csv(res / 'pair_outcomes_four_arm.csv', rows); write_csv(res / 'pair_summary.csv', p2.pair_rows(rows))
    write_csv(res / 'scene_equal_weight_summary.csv', p2.scene_rows(recs, rows)); write_csv(res / 'refresh_retreat_complementarity.csv', p2.complementarity_rows(rows))
    write_csv(res / 'inference_step_seed_trace.csv', p2.call_rows(recs)); write_csv(res / 'c0_c1_queue_prefix_parity.csv', p2.parity_rows(recs))
    write_csv(res / 'skip_reasons.csv', p2.skip_rows(recs)); write_json(res / 'observation_fields.json', {**FIELDS, 'cameras': sorted({n for x in recs for n in x.get('cameras', [])})})
    write_json(res / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('scenes_recorded', 'snapshots', 'blocks', 'continuations', 'dataset_completed', 'ready_for_interpretation', 'problems',
                                              'reproducibility', 'c0_c1_queue_prefix_parity')}, indent=2))
    if not meta['ready_for_interpretation']: raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--out', required=True); q.add_argument('--config', default='configs/stackcube.yaml')
    q.add_argument('--policy', default='artifacts/policies/stack_1d_s1_d500_clean_v1_i080000'); q.add_argument('--splits', default='configs/splits.json')
    q.add_argument('--policy-eval', default='configs/policy_eval_splits.json'); q.add_argument('--scenes', type=int, default=24)
    q.add_argument('--stage2a-plan', default='runs/stage2a_001/plan.json')
    for name in ('scenes', 'audit', 'collect', 'techcheck', 'scale', 'finalize'):
        q = s.add_parser(name); q.add_argument('--plan', required=True)
        if name in ('audit', 'collect'): q.add_argument('--upstream', default='third_party/ManiSkill'); q.add_argument('--device', default='cuda')
        if name == 'scenes': q.add_argument('--keys', required=True); q.add_argument('--out', required=True)
        elif name == 'audit': q.add_argument('--out', required=True)
        else: q.add_argument('--run', required=True)
        if name == 'collect':
            q.add_argument('--start', type=int, required=True); q.add_argument('--stop', type=int, required=True); q.add_argument('--step', type=int, default=1)
            q.add_argument('--label', required=True)
        if name == 'scale':
            q.add_argument('--usd-per-hour', type=float, required=True); q.add_argument('--spent-usd', type=float, required=True)
            q.add_argument('--reserve-usd', type=float, default=0.25); q.add_argument('--workers', type=int, default=2); q.add_argument('--efficiency', type=float, default=0.7)
        if name == 'finalize': q.add_argument('--results', required=True)
    a = p.parse_args(); globals()['cmd_' + a.cmd](a)


if __name__ == '__main__':
    main()
