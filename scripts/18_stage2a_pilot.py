"""Stage 2A exploratory recovery pilot (run id stage2a_001). Development only; see amendments/AMENDMENT_005.

The base policy stays BELOW the original 50/100 qualification gate. This runner does not lift that gate anywhere:
it only accepts the one policy, config, code and scene list whose hashes are written in the plan, and at most the
plan's number of scenes. No training, no qualification, no calibration, no Test.

  plan       local, before a GPU is rented: hashes, scene list (first 24 usable train seeds), limits
  scenes     server: initial cube placement of every pilot scene against demonstrations and other declared scenes
  audit      server: restore audit of the complete runtime state with THIS policy and config
  collect    server: paired blocks of the scenes at the given positions of the list
  techcheck  technical check of the first two scenes (never looks at which choice succeeded)
  scale      24 -> 12 -> 2 from measured time and money only
  finalize   dataset.json (completed only if every scheduled scene is valid) and the descriptive tables
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, json, sys, time
import numpy as np
from hci.io import load_config, read_json, write_json, sha256, digest, runtime_info, UPSTREAM_COMMIT
from hci import pilot

CODE = ['src/hci/pilot.py', 'src/hci/simulation.py', 'src/hci/policies.py', 'src/hci/experiment.py', 'src/hci/io.py', 'src/hci/diagnostics.py',
        'src/hci/schema.py', 'src/hci/trees.py', 'scripts/18_stage2a_pilot.py', 'scripts/stage2a/run.sh']
STATUS = {'base_policy_qualification': 'BELOW_ORIGINAL_GATE (37/100 < 50/100)', 'exploratory_recovery_pilot': 'AUTHORIZED_BY_STAGE2A_SCOPE',
          'formal_study_ready': 'NOT_ESTABLISHED', 'calibration': 'NOT_RUN', 'final_test': 'NOT_RUN'}
NOT_RUN = ['new training', 'data expansion', 'checkpoint re-selection', 'qualification', 'calibration', 'final Test',
           'intervention predictor / Q training', 'candidate (21) evaluation', '5,760-continuation pilot']
IDENTICAL_M = 1e-5


def bound(plan, a):
    """Hashes of what is about to run against the plan. Any difference refuses the command."""
    c = load_config(plan['config']['path']); bad = []
    now = {'config': digest(c), 'weights': sha256(Path(plan['policy']['path']) / 'weights.pt'),
           'meta': sha256(Path(plan['policy']['path']) / 'policy_meta.json'), 'splits': sha256(plan['splits']['path'])}
    want = {'config': plan['config']['config_sha256'], 'weights': plan['policy']['weights_sha256'],
            'meta': plan['policy']['policy_meta_sha256'], 'splits': plan['splits']['file_sha256']}
    bad += [k for k in now if now[k] != want[k]]
    bad += [f for f, h in plan['code'].items() if sha256(f) != h]
    if bad: raise SystemExit(f'Refused: these differ from the plan: {bad}')
    return c


def write_csv(path, rows):
    path = Path(path)
    if path.exists(): raise FileExistsError(f'Refusing overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def cmd_plan(a):
    import h5py
    c = load_config(a.config); splits = read_json(a.splits); pe = read_json(a.policy_eval); dm = read_json(a.data_manifest); src = read_json(a.source_json)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); problems = []
    declared = {**{f'splits/{k}': v for k, v in splits['splits'].items()}, **{f'policy_eval/{k}': v for k, v in pe['splits'].items()}}
    demo_seeds = [e['episode_seed'] for e in src['episodes']]
    chosen, skipped = pilot.select_scenes(splits['splits']['train'], a.scenes, declared, demo_seeds)
    pol = Path(a.policy); meta = read_json(pol / 'policy_meta.json'); w = sha256(pol / 'weights.pt'); m = sha256(pol / 'policy_meta.json')
    listed = {}
    for line in Path(a.backup_list).read_text().splitlines():
        h, name = line.split(None, 1); listed[name.strip().lstrip('*')] = h
    for name, h in ((f'{pol.as_posix()}/weights.pt', w), (f'{pol.as_posix()}/policy_meta.json', m)):
        if listed.get(name) != h: problems.append(f'{name}: local SHA256 differs from the preserved backup list')
    if meta['weights_sha256'] != w: problems.append('policy_meta weights_sha256 differs from weights.pt')
    if meta['upstream_commit'] != UPSTREAM_COMMIT: problems.append('policy was trained on another upstream commit')
    for k in ('obs_horizon', 'act_horizon', 'pred_horizon'):
        if meta['args'][k] != c['policy'][k]: problems.append(f'policy {k} differs from the config')
    if (dm['train_h5_sha256'], dm['name']) != (a.train_h5_sha256, 'train_d500_clean_v1') or not dm['pass']:
        problems.append('data provenance manifest is not the validated train_d500_clean_v1')
    q = read_json(a.qualification)
    if (q['policy_sha256'], q['success_once'], q['n']) != (w, 37, 100): problems.append('qualification record does not belong to this policy as 37/100')
    prior = []
    for f in sorted(Path('runs').glob('*/dataset.json')):
        d = read_json(f); used = sorted(set(d.get('seeds', [])) & set(chosen))
        if used: prior.append({'run': f.parent.name, 'scenes': used, 'policy_sha256': d.get('policy_sha256'), 'same_policy_as_this_pilot': d.get('policy_sha256') == w})
    keys = {}
    with h5py.File(a.source_h5, 'r') as f:
        for e in src['episodes']:
            g = f[f"traj_{e['episode_id']}/env_states/actors"]
            keys[str(e['episode_seed'])] = [*map(float, g['cubeA'][0][:3]), *map(float, g['cubeB'][0][:3])]
    write_json(out / 'demo_scene_keys.json', {'source': a.source_h5, 'source_sha256': sha256(a.source_h5), 'meaning': 'first env state of each source episode: cubeA xyz, cubeB xyz',
                                              'keys': keys})
    audit = list(splits['splits']['audit']); cp = list(c['collection']['checkpoint_steps']); reps = list(range(c['collection']['repeats']))
    if cp != [12, 52, 92, 132] or reps != [0, 1, 2, 3]: problems.append(f'config checkpoints/repeats are {cp}/{reps}, not the fixed [12,52,92,132]/[0,1,2,3]')
    plan = {'run_id': a.out.replace('\\', '/').rstrip('/').split('/')[-1], 'stage': 'Stage 2A exploratory recovery pilot', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'pass': not problems, 'problems': problems, 'status': STATUS, 'role': pilot.ROLE,
            'nature': 'Follow-up exploration decided after the qualification result. Not a pre-registered confirmatory experiment.',
            'policy': {'path': pol.as_posix(), 'weights_sha256': w, 'policy_meta_sha256': m, 'source_checkpoint_sha256': meta['source_checkpoint_sha256'],
                       'verified_against': {'file': a.backup_list, 'sha256': sha256(a.backup_list)},
                       'qualification': {'file': a.qualification, 'set': q['set'], 'success_once': q['success_once'], 'n': q['n'], 'success_at_end': q.get('success_at_end')},
                       'settings': {k: meta['args'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')}, 'unchanged': 'weights, schedule, DDPM steps, action clip'},
            'config': {'path': a.config, 'file_sha256': sha256(a.config), 'config_sha256': digest(c), 'env': c['env'], 'protocol': c['protocol'], 'retreat': c['retreat'], 'audit': c['audit']},
            'data_provenance': {'manifest': a.data_manifest, 'manifest_sha256': sha256(a.data_manifest), 'train_h5_sha256': dm['train_h5_sha256'], 'name': dm['name']},
            'code': {f: sha256(f) for f in CODE}, 'upstream_commit': UPSTREAM_COMMIT,
            'splits': {'path': a.splits, 'file_sha256': sha256(a.splits), 'digest': digest(splits)}, 'policy_eval_splits': {'path': a.policy_eval, 'file_sha256': sha256(a.policy_eval)},
            'scenes': {'rule': 'first 24 seeds of splits/train in ascending order; a seed declared for any other use, or whose initial cube placement equals a '
                               'demonstration or another declared scene, is replaced by the next train seed. Outcomes of this policy are never an input.',
                       'list': chosen, 'seed_level_replaced': skipped, 'prior_use_with_other_policies': prior,
                       'scene_identity': f'server step `scenes` (cube placement within {IDENTICAL_M} m of a reference counts as identical); '
                                         'calibration and Test seeds are compared by seed only and are never reset',
                       'shrink_ladder': [24, 12, 2], 'shrink_rule': 'prefix of the list, chosen from measured time and cost of the first two scenes only'},
            'audit': {'seeds': audit, 'also_demonstration_scenes': sorted(set(audit) & set(dm['all_scene_seeds'])),
                      'use': 'engineering check of restore; not a performance sample and not a denominator',
                      'protocol': 'at each reached checkpoint: Continue 20 steps; restore, repeat; Retreat branch to the end, restore, repeat; Refresh branch to the end, restore, repeat. '
                                  'Compared: simulator state, step counter, policy history, unexecuted queue, last action, action log, TCP history, flags, intervention and '
                                  'retreat records, RNG digests, and the observation returned by the environment after the prefix replay.'},
            'checkpoints': cp, 'repeats': reps,
            'max': {'scenes': a.scenes, 'snapshots': a.scenes * len(cp), 'paired_blocks': a.scenes * len(cp) * len(reps),
                    'continuations': a.scenes * len(cp) * len(reps) * 3, 'first_two_scenes_continuations': 2 * len(cp) * len(reps) * 3},
            'coupling': {'past': 'identical for the three choices: same scene seed, same executed action prefix (restored by prefix replay), same policy history and queue',
                         'future_noise': 'diffusion noise of an inference call is keyed by (scene, physical step of the call, repeat); it does not depend on the choice or on execution order',
                         'continue': 'keeps the remaining queue; its next inference is at the natural chunk boundary',
                         'refresh': 'discards the remaining queue and infers at the checkpoint step, so its calls happen at other steps than Continue: not the same noise draws, by definition of the choice',
                         'retreat_retry': 'physical control steps charged to the same 200-step budget, then inference; no extra time, no state reset',
                         'reference': 'each (scene, checkpoint, repeat) block is compared with the Continue of the same block only'},
            'reproducibility_checks': {'every_snapshot': 'Continue with repeat 0 must reproduce the never-restored trunk (actions, outcome, steps, final state)',
                                       'first_two_scenes': 'Continue with repeat 1 is run a second time after all other branches of the snapshot',
                                       'on_mismatch': 'flagged, never subtracted; interpretation is withheld'},
            'video_rule': 'first two scenes, first usable checkpoint, repeat 0, the three choices (at most 6 files)',
            'budget': {'stage2a_cap_usd': 1.50, 'kept_separate_from': 'Stage 1D cumulative about 1.13 USD', 'cost_basis': 'credit difference, not hourly price x time'},
            'not_run_in_this_stage': NOT_RUN}
    write_json(out / 'plan.json', plan); print(json.dumps({k: plan[k] for k in ('pass', 'problems')}, indent=2))
    print('scenes', chosen, 'prior use', prior)
    if problems: raise SystemExit(2)


def cmd_scenes(a):
    plan = read_json(a.plan); c = bound(plan, a); splits = read_json(plan['splits']['path']); pe = read_json(plan['policy_eval_splits']['path'])
    from hci.simulation import make_env
    from hci.trees import as_numpy
    env = make_env(c)

    def key(seed):
        env.reset(seed=int(seed)); return np.array([*as_numpy(env.cubeA.pose.p)[0], *as_numpy(env.cubeB.pose.p)[0]], float)
    try:
        refs = {f'demonstration/{s}': np.array(k) for s, k in read_json(a.demo_keys)['keys'].items()}
        reset_sets = {'splits/audit': splits['splits']['audit'], 'splits/dev': splits['splits']['dev'], **{f'policy_eval/{k}': v for k, v in pe['splits'].items()}}
        for name, ids in reset_sets.items():
            for s in ids: refs[f'{name}/{s}'] = key(s)
        names = list(refs); table = np.stack([refs[n] for n in names]); chosen = []; replaced = []; nearest = {}
        want = len(plan['scenes']['list']); declared_elsewhere = {x['seed'] for x in plan['scenes']['seed_level_replaced']}
        for s in sorted(splits['splits']['train']):
            if len(chosen) == want: break
            if s in declared_elsewhere: continue
            k = key(s); d = np.abs(table - k).max(1); j = int(d.argmin())
            twin = [t for t in chosen if np.abs(np.array(nearest[str(t)]['key']) - k).max() <= IDENTICAL_M]
            if d[j] <= IDENTICAL_M or twin:
                replaced.append({'seed': s, 'reason': f'initial cube placement identical to {names[j] if d[j] <= IDENTICAL_M else twin[0]}'}); continue
            chosen.append(s); nearest[str(s)] = {'key': k.tolist(), 'nearest_reference': names[j], 'max_abs_difference_m': float(d[j])}
        if len(chosen) < want: raise SystemExit('train has too few usable scenes; nothing is borrowed from other splits')
        report = {'list': chosen, 'same_as_plan': chosen == plan['scenes']['list'], 'replaced': replaced, 'role': pilot.ROLE,
                  'nearest': nearest, 'identical_threshold_m': IDENTICAL_M,
                  'references': {'demonstration_source_episodes': sum(n.startswith('demonstration/') for n in names), **{k: len(v) for k, v in reset_sets.items()}},
                  'seed_level_only': ['splits/calibration', 'splits/test'], 'plan_sha256': sha256(a.plan),
                  'scope': 'Initial cube placement only. Decided before any rollout of the pilot policy on these scenes.'}
        write_json(a.out, report); print(json.dumps({k: report[k] for k in ('list', 'same_as_plan', 'replaced')}))
    finally:
        env.close()


def cmd_audit(a):
    plan = read_json(a.plan); c = bound(plan, a)
    from hci.experiment import runtime, fingerprint
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); r = runtime(c, plan['policy']['path'], a.upstream, a.device); results = []
    try:
        for seed in plan['audit']['seeds']:
            tick = time.perf_counter(); results.append(pilot.audit_scene(r, seed, c, plan['checkpoints'])); results[-1]['wall_seconds'] = time.perf_counter() - tick
            print(f"audit seed={seed} reached={results[-1]['reached']} pass={results[-1]['pass']}", flush=True)
        report = {**pilot.audit_report(results), **fingerprint(c, plan['policy']['path']), 'runtime': runtime_info(), 'plan_sha256': sha256(a.plan),
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
    # a replaced scene changes the list that the plan froze: write a revised plan first, never collect on a silently changed list
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
    out = run / 'branches'; (out / 'workers').mkdir(parents=True, exist_ok=True); (out / 'traces').mkdir(exist_ok=True); (out / 'videos').mkdir(exist_ok=True)
    status = out / 'workers' / f'{a.label}.json'; seeds = [scenes[i] for i in pos]
    write_json(status, {'label': a.label, 'positions': pos, 'scenes': seeds, 'completed': False, 'started_utc': datetime.now(timezone.utc).isoformat()})
    r = runtime(c, plan['policy']['path'], a.upstream, a.device); started = time.perf_counter()
    probe = lambda rt, z: step_state(rt.env, rt.t, z)
    frame = lambda rt: np.concatenate([as_numpy(rt.raw['sensor_data'][n]['rgb'])[0] for n in sorted(rt.raw['sensor_data'])], axis=1).astype(np.uint8)

    def run_scene(seed):
        first = scenes.index(seed) < 2
        rec, traces, videos = pilot.collect_scene(r, seed, c, plan['checkpoints'], plan['repeats'], probe=probe, frame=frame, stage=stage_events,
                                                  video=first, recheck_repeat=1 if first else None)
        rec.update(position=scenes.index(seed), worker=a.label, policy_sha256=plan['policy']['weights_sha256'], config_sha256=plan['config']['config_sha256'])
        return rec, traces, videos

    def save(seed, rec, traces, videos):
        np.savez_compressed(out / 'traces' / f'{seed}.npz', columns=np.array(pilot.TRACE_COLUMNS), **traces); rec['videos'] = []
        for name, frames in videos.items():
            path = out / 'videos' / f'{name}.mp4'
            if path.exists(): raise FileExistsError(f'Refusing overwrite: {path}')
            with imageio.get_writer(str(path), fps=20) as w:
                for f in frames: w.append_data(f)
            rec['videos'].append(path.as_posix())
        print(f"scene={seed} trunk={rec['trunk']['success']}@{rec['trunk']['steps']} snapshots={rec['counts']['snapshots']} "
              f"continuations={rec['counts']['continuations']} repro_ok={rec['reproducibility_ok']} wall={rec['wall_seconds']:.1f}s", flush=True)
    try:
        done = pilot.collect_scenes(seeds, out, run_scene, save)
        write_json(status, {'label': a.label, 'positions': pos, 'scenes': seeds, 'completed': True, 'finished_scenes': done,
                            'wall_seconds': time.perf_counter() - started}, overwrite=True)
    finally:
        r.env.close()


def first_two(a):
    scenes = scene_list(a)[:2]; run = Path(a.run)
    return scenes, [read_json(run / 'branches' / 'scenes' / f'{s}.json') for s in scenes]


def cmd_techcheck(a):
    plan = read_json(a.plan); c = load_config(plan['config']['path']); run = Path(a.run); scenes, recs = first_two(a); checks = {}; r = c['retreat']
    checks['audit_pass'] = bool(read_json(run / 'audit' / 'audit.json')['pass'])
    checks['records_valid'] = all(pilot.validate_scene(x, c) for x in recs)
    checks['no_error_file'] = not list((run / 'branches' / 'scenes').glob('*.error.json'))
    checks['continuations_within_limit'] = sum(x['counts']['continuations'] for x in recs) <= plan['max']['first_two_scenes_continuations']
    checks['trunk_reproduced_by_continue_repeat0'] = all(d['match'] for x in recs for d in x['no_intervention_duplicates'])
    checks['continue_rerun_identical'] = all(d['match'] for x in recs for d in x['continue_rechecks'])
    checks['reproducibility_checks_present'] = all(len(x['no_intervention_duplicates']) == len(x['continue_rechecks']) == x['counts']['snapshots'] for x in recs)
    checks['videos_written'] = all(len(x['videos']) == (3 if x['snapshots'] else 0) and all(Path(v).stat().st_size > 0 for v in x['videos']) for x in recs)
    real = True; grip = True; refresh = True; budget = True
    for x in recs:
        with np.load(run / 'branches' / 'traces' / f"{x['scene']}.npz") as z:
            for b in x['blocks']:
                t = b['checkpoint']; ret = b['cells'][2]; info = ret['retreat']; k = info['steps_used']; rows = z[f"b_{t}_{b['repeat']}_2"]
                # the retreat is made of executed control steps inside the same horizon, with the last gripper command held
                real &= rows.shape[0] == ret['steps'] - t and k <= r['max_steps'] and (k > 0 or info['end_reason'] != 'budget_exhausted')
                real &= bool(np.all(np.abs(rows[:k, 14:17]).max(1) > 0)) if k else True
                grip &= bool(np.all(rows[:k, 17] == np.float32(info['gripper_command'])))
                ref = z[f"b_{t}_{b['repeat']}_1"]; refresh &= ref.shape[0] == b['cells'][1]['steps'] - t and b['cells'][1]['policy_calls_after_snapshot'] >= 1
                budget &= all(cell['steps'] <= c['env']['horizon'] for cell in b['cells'])
    checks.update(retreat_is_executed_control_steps=bool(real), gripper_command_held_during_retreat=bool(grip),
                  refresh_infers_at_the_checkpoint=bool(refresh), horizon_respected=bool(budget))
    report = {'pass': all(checks.values()), 'checks': checks, 'scenes': scenes, 'snapshots': [x['counts']['snapshots'] for x in recs],
              'continuations': [x['counts']['continuations'] for x in recs],
              'scope': 'Technical conditions only. Which choice succeeded is not read here and does not decide whether collection goes on.'}
    write_json(run / 'techcheck.json', report); print(json.dumps(report, indent=2))
    if not report['pass']: raise SystemExit(2)


def cmd_scale(a):
    run = Path(a.run); plan = read_json(a.plan); scenes, recs = first_two(a)
    if not read_json(run / 'techcheck.json')['pass']: raise SystemExit('technical check did not pass')
    n = sum(x['counts']['continuations'] + x['counts']['extra_reproducibility_continuations'] + 1 for x in recs)
    d = pilot.scale_decision([x['wall_seconds'] for x in recs], n, usd_per_hour=a.usd_per_hour, spent_usd=a.spent_usd, cap_usd=plan['budget']['stage2a_cap_usd'],
                             reserve_usd=a.reserve_usd, workers=a.workers, efficiency=a.efficiency, ladder=tuple(plan['scenes']['shrink_ladder']))
    d.update(measured_scenes=scenes, measured_wall_seconds=[x['wall_seconds'] for x in recs], measured_continuations_including_trunk=n)
    write_json(run / 'scale_decision.json', d); print(json.dumps(d, indent=2))


def cmd_finalize(a):
    plan = read_json(a.plan); c = bound(plan, a); run = Path(a.run); scenes = scene_list(a)
    n = read_json(run / 'scale_decision.json')['chosen_scenes'] if (run / 'scale_decision.json').exists() else 2
    meta, recs = pilot.finalize(run / 'branches', scenes[:n], c)
    for x in recs:
        if (x.get('policy_sha256'), x.get('config_sha256')) != (plan['policy']['weights_sha256'], plan['config']['config_sha256']):
            meta['problems'].append(f"scene {x['scene']}: recorded with another policy or config"); meta['completed'] = False
    meta.update(planned_scenes=len(scenes), decided_scenes=n, policy_sha256=plan['policy']['weights_sha256'], config_sha256=plan['config']['config_sha256'],
                plan_sha256=sha256(a.plan), status=plan['status'], coupling=plan['coupling'], source_type='real_sim',
                success_definition=c['protocol']['success_definition'], runtime=runtime_info(), not_run_in_this_stage=NOT_RUN)
    res = Path(a.results); summary = pilot.summarize(recs, planned_scenes=len(scenes))
    meta['ready_for_interpretation'] = bool(meta['completed'] and meta['reproducibility_ok'])
    write_json(run / 'branches' / 'dataset.json', meta)
    summary.update(decided_scenes=n, dataset_completed=meta['completed'], ready_for_interpretation=meta['ready_for_interpretation'],
                   problems=meta['problems'], status=plan['status'])
    write_csv(res / 'pair_outcomes.csv', pilot.pair_rows(recs)); write_csv(res / 'scene_summary.csv', pilot.scene_rows(recs))
    write_csv(res / 'skip_reasons.csv', pilot.skip_rows(recs)); write_json(res / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('scenes_recorded', 'snapshots', 'paired_blocks', 'continuations', 'dataset_completed', 'ready_for_interpretation', 'problems', 'reproducibility')}, indent=2))
    if not meta['ready_for_interpretation']: raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--out', required=True); q.add_argument('--config', default='configs/stackcube.yaml')
    q.add_argument('--policy', default='artifacts/policies/stack_1d_s1_d500_clean_v1_i080000'); q.add_argument('--splits', default='configs/splits.json')
    q.add_argument('--policy-eval', default='configs/policy_eval_splits.json'); q.add_argument('--scenes', type=int, default=24)
    q.add_argument('--data-manifest', required=True); q.add_argument('--train-h5-sha256', required=True); q.add_argument('--backup-list', required=True)
    q.add_argument('--qualification', required=True); q.add_argument('--source-json', required=True); q.add_argument('--source-h5', required=True)
    for name in ('scenes', 'audit', 'collect', 'techcheck', 'scale', 'finalize'):
        q = s.add_parser(name); q.add_argument('--plan', required=True)
        if name in ('audit', 'collect'): q.add_argument('--upstream', default='third_party/ManiSkill'); q.add_argument('--device', default='cuda')
        if name == 'scenes': q.add_argument('--demo-keys', required=True); q.add_argument('--out', required=True)
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
