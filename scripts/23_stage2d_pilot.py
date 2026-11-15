"""Stage 2D exploratory control with a matched return step (run id stage2d_001). Development only; see amendments/AMENDMENT_008.

Same fixed policy as Stage 2A/2B (BELOW the original 50/100 qualification gate; the gate is not lifted anywhere).
New train scenes; arms continue (C0), retreat_5cm (R5, the existing Retreat-Retry), retreat_2cm_matched (R2M) and
hold_matched (H0M). No training, no distance/time search, no predictor, no qualification, no calibration, no Test.

  plan       local, before a GPU is rented: hashes, new scene list, arm definitions, T_ref rule, limits
  scenes     server: initial cube placement of the new scenes against every placement already on record
  audit      server: full-state restore audit with this policy on this host, including the new arms and the handoff
  collect    server: four-arm blocks of the scenes at the given positions of the list
  techcheck  technical check of the first two scenes (never reads which arm succeeded)
  scale      12 -> 6 -> 2 from measured time and money only
  finalize   dataset.json, the descriptive tables and the event rows
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, importlib.util, json, subprocess, time
import numpy as np
from hci.io import load_config, read_json, write_json, sha256, digest, runtime_info, seed_for, UPSTREAM_COMMIT
from hci import pilot, pilot2b, pilot2d as p2

_spec = importlib.util.spec_from_file_location('stage2b_runner', Path(__file__).with_name('20_stage2b_pilot.py'))
s2b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s2b)
bound, write_csv, IDENTICAL_M, USER_POLICY_SHA = s2b.bound, s2b.write_csv, s2b.IDENTICAL_M, s2b.USER_POLICY_SHA

CODE = ['src/hci/pilot2d.py', 'src/hci/pilot2b.py', 'src/hci/pilot.py', 'src/hci/simulation.py', 'src/hci/policies.py', 'src/hci/experiment.py', 'src/hci/io.py',
        'src/hci/diagnostics.py', 'src/hci/schema.py', 'src/hci/trees.py', 'scripts/23_stage2d_pilot.py', 'scripts/20_stage2b_pilot.py', 'scripts/18_stage2a_pilot.py',
        'scripts/stage2d/run.sh', 'scripts/stage1b/common.sh']
UNCHANGED_SINCE_2B = ['src/hci/pilot2b.py', 'src/hci/pilot.py', 'src/hci/simulation.py', 'src/hci/policies.py', 'src/hci/experiment.py', 'src/hci/io.py',
                      'src/hci/diagnostics.py', 'src/hci/schema.py', 'src/hci/trees.py', 'scripts/20_stage2b_pilot.py', 'scripts/18_stage2a_pilot.py']
STATUS = {'base_policy_qualification': 'BELOW_ORIGINAL_GATE (37/100 < 50/100)', 'stage2d': 'EXPLORATORY_DURATION_MATCHED_CONTROL',
          'observation_based_selector': 'NOT_TRAINED', 'formal_study_ready': 'NOT_ESTABLISHED', 'calibration': 'NOT_RUN', 'final_test': 'NOT_RUN'}
NOT_RUN = ['base policy retraining', 'checkpoint re-selection', 'qualification rerun', 'distance / duration grid search', 'Refresh and alternative-noise Continue arms',
           'Q / Rescue / Harm predictor training', 'observation-based selector', 'LTT / calibration', 'final Test', 'collection beyond 12 scenes']
FIELDS = {**s2b.FIELDS, 'diagnostic_privileged_only': {
    'scene JSON: initial_state, snapshots[].state_at_checkpoint, snapshots[].stage_so_far, cells[].stage': 'cube poses, grasp/on-cube/static flags, stage labels read from simulator ground truth',
    'branches/traces/<scene>.npz': 'per-step TCP, cube positions, flags, success, executed action, gripper joint positions, phase; `ref_<checkpoint>` is the R5 control-only reference run',
    'y_C0, y_R5, y_R2M, y_H0M and every pair class': 'future outcomes (labels)'},
    'online_proprioception_in_traces': 'gripper_qpos_1/2 are the last two entries of the qpos observation the policy already receives; logged per step to tell the '
                                       'gripper command from the finger position. They are not a new policy input.',
    'controller_inputs_of_the_matched_arms': 'measured TCP position, the snapshot TCP position, the last gripper command, T_ref. No object pose, no task flag, no outcome.',
    'rule': 'Nothing of the diagnostic group is written into the online input file, given to the policy, or used for a control target. No model was trained in this stage.'}


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()


def cmd_plan(a):
    c = load_config(a.config); splits = read_json(a.splits); pe = read_json(a.policy_eval); prev = read_json(a.stage2b_plan); problems = []
    run2b = Path(a.stage2b_plan).parent; sc2b = read_json(run2b / 'scenes.json'); keys2b = read_json(run2b / 'reference_scene_keys.json')
    data2b = read_json(run2b / 'branches' / 'dataset.json'); dm = read_json(prev['data_provenance']['manifest'])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    if sc2b['plan_sha256'] != sha256(a.stage2b_plan) or sc2b['list'] != prev['scenes']['list'] or data2b['recorded_scenes'] != sc2b['list']:
        problems.append('Stage 2B scenes.json / dataset do not belong to the Stage 2B plan')
    used_2a = list(prev['scenes']['stage2a_scenes_excluded']); used_2b = list(sc2b['list'])
    declared = {**{f'splits/{k}': v for k, v in splits['splits'].items()}, **{f'policy_eval/{k}': v for k, v in pe['splits'].items()},
                f'stage2a/{pilot.ROLE}': used_2a, f'stage2b/{pilot2b.ROLE}': used_2b}
    demo = [int(k.split('/')[1]) for k in keys2b['keys'] if k.startswith('demonstration/')]
    chosen, skipped = pilot.select_scenes(splits['splits']['train'], a.scenes, declared, demo)
    pol = Path(a.policy); meta = read_json(pol / 'policy_meta.json'); w = sha256(pol / 'weights.pt'); m = sha256(pol / 'policy_meta.json')
    if (w, m) != (USER_POLICY_SHA['weights.pt'], USER_POLICY_SHA['policy_meta.json']): problems.append('policy files differ from the SHA256 on record')
    if (w, m) != (prev['policy']['weights_sha256'], prev['policy']['policy_meta_sha256']): problems.append('policy differs from the Stage 2A/2B policy')
    backup = prev['policy']['verified_against']['backup_list']; listed = {}
    for line in Path(backup).read_text().splitlines():
        h, name = line.split(None, 1); listed[name.strip().lstrip('*')] = h
    for name, h in ((f'{pol.as_posix()}/weights.pt', w), (f'{pol.as_posix()}/policy_meta.json', m)):
        if listed.get(name) != h: problems.append(f'{name}: local SHA256 differs from the preserved backup list')
    if meta['weights_sha256'] != w: problems.append('policy_meta weights_sha256 differs from weights.pt')
    if meta['upstream_commit'] != UPSTREAM_COMMIT: problems.append('policy was trained on another upstream commit')
    for k in ('obs_horizon', 'act_horizon', 'pred_horizon'):
        if meta['args'][k] != c['policy'][k]: problems.append(f'policy {k} differs from the config')
    if digest(c) != prev['config']['config_sha256'] or sha256(a.config) != prev['config']['file_sha256']: problems.append('config differs from the Stage 2B config')
    for k in ('splits', 'policy_eval_splits'):
        if sha256(prev[k]['path']) != prev[k]['file_sha256']: problems.append(f'{k} changed since Stage 2B')
    if sha256(prev['data_provenance']['manifest']) != prev['data_provenance']['manifest_sha256']: problems.append('data provenance manifest changed since Stage 2B')
    q = read_json(prev['policy']['qualification']['file'])
    if (q['policy_sha256'], q['success_once'], q['n']) != (w, 37, 100): problems.append('qualification record does not belong to this policy as 37/100')
    cp = list(c['collection']['checkpoint_steps']); reps = list(range(c['collection']['repeats'])); rt = c['retreat']
    if cp != [12, 52, 92, 132] or reps != [0, 1, 2, 3]: problems.append(f'config checkpoints/repeats are {cp}/{reps}, not the fixed [12,52,92,132]/[0,1,2,3]')
    if (rt['mode'], rt['distance_m'], rt['max_translation_per_step_m'], rt['max_steps'], rt['arrival_tolerance_m']) != ('up', 0.05, 0.005, 40, 0.005):
        problems.append('the existing Retreat settings are not 5 cm up / 5 mm per step / 40 steps / 5 mm tolerance')
    changed = [f for f in UNCHANGED_SINCE_2B if sha256(f) != prev['code'].get(f)]
    if changed: problems.append(f'files that Stage 2B ran with have changed: {changed}')
    prior = []
    for f in sorted(Path('runs').glob('*/dataset.json')) + sorted(Path('runs').glob('*/branches/dataset.json')):
        d = read_json(f); used = sorted((set(d.get('seeds', [])) | set(d.get('recorded_scenes', []))) & set(chosen))
        if used: prior.append({'file': f.as_posix(), 'scenes': used})
    if prior: problems.append(f'chosen scenes already appear in earlier runs: {prior}')
    keys = {**keys2b['keys'], **{f'stage2b/{s}': v['key'] for s, v in sc2b['nearest'].items()}}
    write_json(out / 'reference_scene_keys.json', {'meaning': 'initial cubeA xyz, cubeB xyz of placements already on record (no environment was reset to make this file)',
                                                   'sources': {(run2b / 'reference_scene_keys.json').as_posix(): sha256(run2b / 'reference_scene_keys.json'),
                                                               (run2b / 'scenes.json').as_posix(): sha256(run2b / 'scenes.json')}, 'keys': keys})
    n = a.scenes; arms = len(p2.ARMS); per = len(cp) * len(reps) * arms
    plan = {'run_id': a.out.replace('\\', '/').rstrip('/').split('/')[-1], 'stage': 'Stage 2D exploratory control with a matched return step',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'pass': not problems, 'problems': problems, 'status': STATUS, 'role': p2.ROLE,
            'nature': 'Development experiment designed after the Stage 2B results, the Stage 2C post hoc analysis and its supplementary case reading were seen. '
                      'Not a confirmatory test, not an independent validation, not a final Test.',
            'repository': {'head': git('rev-parse', 'HEAD'), 'working_tree_changes_at_plan_time': git('status', '--porcelain').splitlines(),
                           'note': 'the code that runs is fixed by the per-file SHA256 below, not by the commit',
                           'files_identical_to_the_stage2b_run': {f: sha256(f) == prev['code'].get(f) for f in UNCHANGED_SINCE_2B}},
            'based_on': {'stage2b_plan': a.stage2b_plan, 'stage2b_plan_sha256': sha256(a.stage2b_plan), 'stage2b_dataset_sha256': sha256(run2b / 'branches' / 'dataset.json'),
                         'reports_read': ['reports/STAGE_REPORT_stage2b_001.md', 'reports/STAGE_REPORT_stage2c_001.md', 'amendments/AMENDMENT_007_stage2c_offline_analysis.md']},
            'policy': {'path': pol.as_posix(), 'weights_sha256': w, 'policy_meta_sha256': m, 'source_checkpoint_sha256': meta['source_checkpoint_sha256'],
                       'verified_against': {'specification': USER_POLICY_SHA, 'backup_list': backup, 'backup_list_sha256': sha256(backup), 'stage2b_plan': True},
                       'qualification': {'file': prev['policy']['qualification']['file'], 'set': q['set'], 'success_once': q['success_once'], 'n': q['n']},
                       'settings': {k: meta['args'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')}, 'unchanged': 'weights, schedule, DDPM steps, action clip',
                       'absolute_episode_time_in_the_policy_input': 'no: the policy reads RGB history and qpos/qvel/tcp_pose history (schema rgb_qpos_qvel_tcp_v1); '
                                                                    't/horizon appears only in policy.features(), which the base policy does not use for acting'},
            'config': {'path': a.config, 'file_sha256': sha256(a.config), 'config_sha256': digest(c), 'env': c['env'], 'protocol': c['protocol'], 'retreat': c['retreat'], 'audit': c['audit']},
            'data_provenance': prev['data_provenance'], 'code': {f: sha256(f) for f in CODE}, 'upstream_commit': UPSTREAM_COMMIT,
            'splits': {'path': a.splits, 'file_sha256': sha256(a.splits), 'digest': digest(splits)}, 'policy_eval_splits': {'path': a.policy_eval, 'file_sha256': sha256(a.policy_eval)},
            'scenes': {'rule': f'first {n} seeds of splits/train in ascending order that are neither Stage 2A nor Stage 2B scenes and are declared for no other use; a seed whose '
                               'initial cube placement equals a placement on record is replaced by the next train seed. Outcomes of this policy are never an input.',
                       'list': chosen, 'seed_level_skipped': skipped, 'stage2a_scenes_excluded': used_2a, 'stage2b_scenes_excluded': used_2b,
                       'scene_identity': f'server step `scenes` (cube placement within {IDENTICAL_M} m counts as identical); references: demonstrations, Stage 2A and Stage 2B scenes '
                                         'from stored keys, audit/dev/policy-evaluation scenes by reset; calibration and Test by seed only, never reset',
                       'shrink_ladder': [12, 6, 2], 'shrink_rule': 'prefix of the list, chosen from measured time and cost of the first two scenes only',
                       'future_use': 'these scenes stay development data; they are never moved to calibration or Test',
                       'not_reused': 'Stage 2B scenes with favourable results (e.g. 10035, 10043) are not part of this sample'},
            'audit': {'seeds': list(splits['splits']['audit']), 'also_demonstration_scenes': sorted(set(splits['splits']['audit']) & set(dm['all_scene_seeds'])),
                      'use': 'engineering check of restore and of the new arms; not a performance sample and not a denominator',
                      'protocol': 'Stage 2A audit (same compared items, same tolerances) + restore and trace after the R5 control-only reference, after R5, R2M and H0M '
                                  '+ R5 control part against the reference + handoff step/seed/queue/history + exact check that saving the policy inputs changes nothing'},
            'checkpoints': cp, 'repeats': reps,
            'arms': {'continue': {'short': 'C0', 'definition': 'existing Continue: queue and history kept, next inference at the natural chunk boundary, Runtime.force not called'},
                     'retreat_5cm': {'short': 'R5', 'definition': 'existing Retreat-Retry through Runtime.force(2), unchanged: target snapshot TCP + 0.05 m (z), command increment '
                                                                  '<= 0.005 m/step, arrival tolerance 0.005 m, at most 40 steps, last gripper command held'},
                     'retreat_2cm_matched': {'short': 'R2M', 'commanded_distance_m': p2.DISTANCE_M['retreat_2cm_matched'],
                                             'definition': 'queue discarded once; feedback control towards snapshot TCP + 0.02 m (z) with the R5 controller, increment limit and '
                                                           'tolerance; at the first step with the measured TCP within tolerance that measured position (p_arrival) becomes a fixed hold '
                                                           'target; control ends after T_ref steps; if the target is not reached within T_ref steps the arm ends there with '
                                                           'target_reached=false (no extra time, not excluded); then queue cleared and the policy resumes on the real history'},
                     'hold_matched': {'short': 'H0M', 'commanded_distance_m': 0.0,
                                      'definition': 'queue discarded once; the snapshot TCP position is a fixed hold target followed by the same delta controller for T_ref steps; '
                                                    'physics runs normally (no set_state, nothing frozen); then queue cleared and the policy resumes on the real history'},
                     'common': 'gripper command of the snapshot held in every control step; every control step goes through Runtime.advance (real observation history, '
                               'horizon charged); success_once_absorbing and horizon 200 unchanged; an arm that ends during control stops at once',
                     'tolerance_note': 'the 5 mm arrival tolerance is not tightened for R2M; commanded distance and achieved displacement are recorded separately'},
            't_ref': {'rule': 'per snapshot: restore, run the control part of R5 alone (Runtime.force(2), no policy inference), read retreat_info.steps_used after it ended and '
                              'before the policy is called; restore and verify the full state and observation again',
                      'never': ['the Stage 2B mean (25.4)', 'a fixed 25 or 30', 'a value from another scene', 'a value chosen after any branch outcome'],
                      'reference_must': 'call the policy 0 times; every R5 of that snapshot must reproduce the reference commands, states, observations and step count, '
                                        'otherwise the scene stops as a technical fault',
                      'early_end': 'if the reference ends during control (success or termination) T_ref is its steps_used and reference_early_absorbing=true; the other arms '
                                   'still use that T_ref and stop at once if they end earlier; the matched-handoff check is N/A for such blocks; no block is removed',
                      'accounting': 'the reference is preparation; it is recorded per snapshot and never counted as a continuation or an outcome',
                      'caveat': 'T_ref is taken from the simulator response to R5. The matched arms could not be deployed as they are.'},
            'noise': {'seed_rule': "seed_for(scene, physical_step_of_the_call, block_repeat_id, 'dp') for all four arms (unchanged code path)",
                      'handoff': 'R5, R2M and H0M make their first inference at checkpoint + T_ref with the same seed unless the episode ended during control; '
                                 'their observations differ, so the action chunks are not required to be equal',
                      'continue': 'C0 infers at other steps with other seeds; differences from C0 are not called a pure movement or time effect',
                      'not_in_the_seed': ['execution order', 'worker id', 'wall-clock time', 'arm name'],
                      'caveat': 'equal seeds specify the same pseudo-random condition; this is not a proof of statistical independence'},
            'max': {'scenes': n, 'snapshots': n * len(cp), 'blocks': n * len(cp) * len(reps), 'continuations': n * per, 'first_two_scenes_continuations': 2 * per,
                    'reference_control_runs': n * len(cp), 'extra_reproducibility_continuations_first_two_scenes': 2 * len(cp) * 2, 'trunks': n},
            'reproducibility_checks': {'every_snapshot': 'continue with block repeat 0 must reproduce the never-restored trunk; the state is verified again after the T_ref reference',
                                       'every_block': 'R5 control part equals the reference; handoff of the three control arms',
                                       'first_two_scenes': 'after all other branches of a snapshot: R2M and H0M of repeat 0 once more',
                                       'on_mismatch': 'a wrong control part or handoff stops the scene; a rerun mismatch is flagged, never subtracted, and the data set is not marked complete'},
            'observation_fields': FIELDS,
            'events': {'thresholds': {'near_m': p2.NEAR_M, 'align_xy_m': p2.ALIGN_M, 'lift_m': p2.LIFT_M, 'cubeB_moved_m': p2.B_MOVED_M, 'source': 'unchanged from Stage 2C'},
                       'denominator': 'all blocks whose cube was grasped at the checkpoint, for all four arms; blocks not grasped are listed separately',
                       'read': ['grasp flag kept through the control part', 'first open command (gripper action > 0) after the return (C0: after the checkpoint)',
                                'first grasp-flag release after the return', 'cubeA-cubeB horizontal distance, cube height and on-cube flag at those steps'],
                       'missing': 'no open command / no release is a recorded value, never a dropped row', 'no_cause_labels': True,
                       'examples': 'chosen after the tables as the first scene/checkpoint/repeat in ascending order, and marked post hoc'},
            'video_rule': 'first two scenes, first usable checkpoint, block repeat 0, the four arms (at most 8 files); no outcome-selected example is added',
            'analysis': {'primary_pairs': ['R2M vs R5', 'H0M vs R5', 'R2M vs H0M'], 'secondary_pairs': ['R5 vs C0', 'R2M vs C0', 'H0M vs C0'],
                         'unit': 'block (scene, checkpoint, block_repeat_id): the four 2x2 cells with numerator and denominator',
                         'primary_summary': 'scene-equal weighting: mean over reached checkpoints and repeats within a scene, then over scenes; block totals shown separately',
                         'independent_units': f'at most {n} scenes', 'not_computed': ['p-values', 'intervals', 'LTT', 'any threshold search']},
            'budget': {'stage2d_cap_usd': 1.00, 'kept_separate_from': 'Stage 1D, 2A (0.38), 2B (0.30)', 'cost_basis': 'credit difference, not hourly price x time',
                       'includes': 'install, transfer, audit, trunks, reference runs, reruns, backup'},
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
        names = list(refs); table = np.stack([refs[n] for n in names]); chosen = []; replaced = []; nearest = {}; sc = plan['scenes']
        want = len(sc['list']); excluded = {x['seed'] for x in sc['seed_level_skipped']} | set(sc['stage2a_scenes_excluded']) | set(sc['stage2b_scenes_excluded'])
        for s in sorted(splits['splits']['train']):
            if len(chosen) == want: break
            if s in excluded: continue
            k = key(s); d = np.abs(table - k).max(1); j = int(d.argmin())
            twin = [t for t in chosen if np.abs(np.array(nearest[str(t)]['key']) - k).max() <= IDENTICAL_M]
            if d[j] <= IDENTICAL_M or twin:
                replaced.append({'seed': s, 'reason': f'initial cube placement identical to {names[j] if d[j] <= IDENTICAL_M else twin[0]}'}); continue
            chosen.append(s); nearest[str(s)] = {'key': k.tolist(), 'nearest_reference': names[j], 'max_abs_difference_m': float(d[j])}
        if len(chosen) < want: raise SystemExit('train has too few usable scenes; nothing is borrowed from other splits')
        report = {'list': chosen, 'same_as_plan': chosen == sc['list'], 'replaced': replaced, 'role': p2.ROLE, 'nearest': nearest, 'identical_threshold_m': IDENTICAL_M,
                  'references': {'stored_keys_demonstrations_stage2a_stage2b': stored, **{k: len(v) for k, v in reset_sets.items()}},
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
        bad = [(s['seed'], x['checkpoint'], k['t'], k['seed']) for s in results for x in s['checkpoints'] if x['reached'] and x['handoff']
               for arm in ('R5', 'R2M', 'H0M') for k in [{'t': x['handoff'][arm]['first_policy_step'], 'seed': x['handoff'][arm]['first_policy_seed']}]
               if k['t'] is not None and k['seed'] != seed_for(s['seed'], k['t'], 0, 'dp')]
        report = {**p2.audit_report(results), **s2b_fingerprint(fingerprint, c, plan), 'runtime': runtime_info(), 'plan_sha256': sha256(a.plan),
                  'tolerances': c['audit'], 'restore_method': 'prefix action replay from reset(seed); set_state_dict is not used',
                  'handoff_seeds_not_following_the_declared_key': bad,
                  'audit_seeds_also_demonstration_scenes': plan['audit']['also_demonstration_scenes'],
                  'scope': 'Same-scene restore of this policy and config at the tested states; not proof of global simulator determinism, not a performance sample.'}
        report['pass'] = bool(report['pass'] and not bad)
        write_json(out / 'audit.json', report); print(json.dumps({'pass': report['pass'], 'branches_executed': report['branches_executed']}))
        if not report['pass']: raise SystemExit(2)
    finally:
        r.env.close()


def s2b_fingerprint(fingerprint, c, plan):
    return fingerprint(c, plan['policy']['path'])


def scene_list(a):
    return s2b.scene_list(a)


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
    # gripper joint positions come from the observation the policy already receives (the last two qpos entries)
    probe = lambda rt, z: {**step_state(rt.env, rt.t, z), 'gripper_qpos': as_numpy(rt.raw['agent']['qpos']).reshape(-1)[-2:].tolist()}
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
        np.savez_compressed(out / 'traces' / f'{seed}.npz', columns=np.array(p2.TRACE_COLUMNS), phases=np.array(p2.PHASES), **traces)
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


def files_ok(run, rec, first):
    """Trace archive with the trunk, the reference of every snapshot and every branch of the record; the declared videos exist."""
    p = run / 'branches' / 'traces' / f"{rec['scene']}.npz"
    if not p.exists(): return False
    want = ({'columns', 'phases', 'trunk'} | {f"ref_{s['checkpoint']}" for s in rec['snapshots']}
            | {f"b_{b['checkpoint']}_{b['repeat']}_{cell['arm']}" for b in rec['blocks'] for cell in b['cells']})
    with np.load(p) as z:
        ok = set(z.files) == want and list(z['columns']) == list(p2.TRACE_COLUMNS) and z['trunk'].shape[0] == rec['trunk']['steps']
        ok = ok and all(z[f"b_{b['checkpoint']}_{b['repeat']}_{cell['arm']}"].shape[0] == cell['steps'] - b['checkpoint'] for b in rec['blocks'] for cell in b['cells'])
        ok = ok and all(z[f"ref_{s['checkpoint']}"].shape[0] == s['reference']['t_ref'] for s in rec['snapshots'])
    videos = rec.get('videos', [])
    return bool(ok and len(videos) == (len(p2.ARMS) if first and rec['snapshots'] else 0) and all(Path(v).exists() and Path(v).stat().st_size > 0 for v in videos))


def trace_checks(run, rec, c):
    """What the saved traces say about the control parts. Reads commands, positions and the phase column; never an outcome."""
    col = {n: i for i, n in enumerate(p2.TRACE_COLUMNS)}; ph = {n: i for i, n in enumerate(p2.PHASES)}; out = {k: True for k in (
        'gripper_command_held_in_every_control_step', 'control_steps_are_executed_steps_inside_the_horizon', 'phases_follow_the_declared_order',
        'hold_steps_match_the_record', 'reference_trace_equals_every_r5_control_part', 'no_time_added_beyond_t_ref', 'gripper_qpos_logged')}
    with np.load(run / 'branches' / 'traces' / f"{rec['scene']}.npz") as z:
        snap = {s['checkpoint']: s for s in rec['snapshots']}
        for b in rec['blocks']:
            t = b['checkpoint']; T = b['t_ref']; ref = z[f'ref_{t}']
            for cell in b['cells']:
                rows = z[f"b_{t}_{b['repeat']}_{cell['arm']}"]; k = cell['control_steps']; p = rows[:, col['phase']]
                out['gripper_qpos_logged'] &= bool(np.isfinite(rows[:, col['gripper_qpos_1']:col['gripper_qpos_2'] + 1]).all())
                out['control_steps_are_executed_steps_inside_the_horizon'] &= rows.shape[0] == cell['steps'] - t and cell['steps'] <= c['env']['horizon'] and k <= rows.shape[0]
                if cell['arm'] == 'C0': out['phases_follow_the_declared_order'] &= bool((p == ph['continue']).all()); continue
                out['no_time_added_beyond_t_ref'] &= k <= T and (k == T or cell['early_absorbing'])
                out['gripper_command_held_in_every_control_step'] &= bool(np.all(rows[:k, col['a_gripper']] == np.float32(snap[t]['last_gripper_command'])))
                ctl = p[:k]; moves = int((ctl == ph['intervention_move']).sum()); holds = int((ctl == ph['intervention_hold']).sum())
                order = moves + holds == k and bool((ctl[:moves] == ph['intervention_move']).all()) and bool((p[k:] == ph['policy_after_return']).all())
                out['phases_follow_the_declared_order'] &= order
                out['hold_steps_match_the_record'] &= (moves, holds) == (cell['control']['move_steps'], cell['control']['hold_steps'])
                if cell['arm'] == 'R5': out['reference_trace_equals_every_r5_control_part'] &= bool(np.array_equal(ref, rows[:k], equal_nan=True))
    return out


def cmd_techcheck(a):
    plan = read_json(a.plan); c = load_config(plan['config']['path']); run = Path(a.run); scenes, recs = first_two(a); checks = {}
    blocks = [b for x in recs for b in x['blocks']]; cells = [(x['scene'], b, cell) for x in recs for b in x['blocks'] for cell in b['cells']]
    checks['audit_pass'] = bool(read_json(run / 'audit' / 'audit.json')['pass'])
    checks['records_valid'] = all(p2.validate_scene(x, c) for x in recs)
    checks['no_error_file'] = not list((run / 'branches' / 'scenes').glob('*.error.json'))
    checks['continuations_within_limit'] = sum(x['counts']['continuations'] for x in recs) <= plan['max']['first_two_scenes_continuations']
    checks['extra_reruns_within_limit'] = sum(x['counts']['extra_reproducibility_continuations'] for x in recs) <= plan['max']['extra_reproducibility_continuations_first_two_scenes']
    checks['trunk_reproduced_by_continue_repeat0'] = all(d['match'] for x in recs for d in x['no_intervention_duplicates'])
    checks['r2m_rerun_identical'] = all(d['match'] for x in recs for d in x['rechecks'] if d['experimental_arm'] == 'retreat_2cm_matched')
    checks['h0m_rerun_identical'] = all(d['match'] for x in recs for d in x['rechecks'] if d['experimental_arm'] == 'hold_matched')
    checks['reproducibility_checks_present_and_matched'] = all(not p2.repro_problems(x, True) for x in recs)
    checks['reference_is_control_only_and_state_verified_after_it'] = all(s['reference']['policy_inference_calls'] == 0 and s['reference']['restored_and_verified_after_reference']
                                                                         for x in recs for s in x['snapshots'])
    checks['every_r5_control_part_equals_its_reference'] = all(b['cells'][1]['same_as_reference']['pass'] and b['cells'][1]['same_as_reference']['retreat_record_equal'] for b in blocks)
    checks['no_inference_during_any_control_part'] = all(cell['control_inference_calls'] == 0 for s, b, cell in cells)
    checks['handoff_as_declared'] = all(b['handoff']['pass'] for b in blocks)
    checks['first_inference_after_return_at_checkpoint_plus_t_ref_with_the_declared_seed'] = all(
        cell['inference_calls'][0]['t'] == b['checkpoint'] + b['t_ref'] and cell['inference_calls'][0]['seed'] == seed_for(s, b['checkpoint'] + b['t_ref'], b['repeat'], 'dp')
        for s, b, cell in cells if cell['policy_resumed'])
    checks['every_inference_seed_follows_the_declared_key'] = all(k['seed'] == seed_for(s, k['t'], cell['policy_noise_repeat_id'], 'dp') for s, b, cell in cells for k in cell['inference_calls'])
    checks['trunk_seeds_follow_repeat0'] = all(k['seed'] == seed_for(x['scene'], k['t'], 0, 'dp') and k['policy_noise_repeat_id'] == 0 for x in recs for k in x['trunk']['inference_calls'])
    checks['traces_and_videos_written'] = all(files_ok(run, x, True) for x in recs)
    checks['online_inputs_saved_once_per_snapshot'] = all(s2b.inputs_ok(run, x) for x in recs)
    tc = [trace_checks(run, x, c) for x in recs] if checks['traces_and_videos_written'] else []
    for k in (tc[0] if tc else []): checks[k] = all(t[k] for t in tc)
    report = {'pass': all(checks.values()) and bool(tc), 'checks': checks, 'scenes': scenes, 'snapshots': [x['counts']['snapshots'] for x in recs],
              'continuations': [x['counts']['continuations'] for x in recs],
              'scope': 'Technical conditions only. Which arm succeeded is not read here and does not decide whether collection goes on.'}
    write_json(run / 'techcheck.json', report); print(json.dumps(report, indent=2))
    if not report['pass']: raise SystemExit(2)


def cmd_scale(a):
    run = Path(a.run); plan = read_json(a.plan); scenes, recs = first_two(a)
    if not read_json(run / 'techcheck.json')['pass']: raise SystemExit('technical check did not pass')
    n = sum(x['counts']['continuations'] + x['counts']['extra_reproducibility_continuations'] + x['counts']['reference_control_runs'] + 1 for x in recs)
    per_scene = len(plan['checkpoints']) * (len(plan['repeats']) * len(p2.ARMS) + 1) + 1
    d = p2.scale_decision([x['wall_seconds'] for x in recs], n, usd_per_hour=a.usd_per_hour, spent_usd=a.spent_usd, cap_usd=plan['budget']['stage2d_cap_usd'],
                          reserve_usd=a.reserve_usd, workers=a.workers, efficiency=a.efficiency, ladder=tuple(plan['scenes']['shrink_ladder']),
                          per_scene_continuations=per_scene)
    d.update(measured_scenes=scenes, measured_wall_seconds=[x['wall_seconds'] for x in recs], measured_runs_including_trunk_reference_and_reruns=n)
    write_json(run / 'scale_decision.json', d); print(json.dumps(d, indent=2))


def event_rows(run, recs):
    rows = []
    for rec in recs:
        z0 = float(rec['initial_state']['cubeA'][2])
        with np.load(run / 'branches' / 'traces' / f"{rec['scene']}.npz") as z:
            for b in rec['blocks']:
                t = b['checkpoint']; cp = z['trunk'][t - 1:t]
                for cell in b['cells']:
                    d = p2.derived(np.concatenate([cp, z[f"b_{t}_{b['repeat']}_{cell['arm']}"]]), z0)
                    rows.append({'scene': rec['scene'], 'checkpoint': t, 'block_repeat_id': b['repeat'], 'arm': cell['arm'], 't_ref': b['t_ref'], 'y': cell['y'],
                                 'y_C0': b['y_C0'], 'y_R5': b['y_R5'], **p2.event_reading(d, cell['control_steps'])})
    return rows


def cmd_finalize(a):
    plan = read_json(a.plan); c = bound(plan, a); run = Path(a.run); scenes = scene_list(a)
    decided = (run / 'scale_decision.json').exists(); n = read_json(run / 'scale_decision.json')['chosen_scenes'] if decided else 2
    meta, recs = p2.finalize(run / 'branches', scenes[:n], c, rerun_scenes=scenes[:2])
    if not decided or n not in plan['scenes']['shrink_ladder']:      # tables of what exists are still written; the data set is not complete
        meta['problems'].append('no valid scale decision: only the first two scenes were considered'); meta['completed'] = False
    if not ((run / 'techcheck.json').exists() and read_json(run / 'techcheck.json')['pass']):
        meta['problems'].append('technical check of the first two scenes is missing or did not pass'); meta['completed'] = False
    usable = []
    for x in recs:
        if (x.get('policy_sha256'), x.get('config_sha256'), x.get('code_sha256')) != (plan['policy']['weights_sha256'], plan['config']['config_sha256'], digest(plan['code'])):
            meta['problems'].append(f"scene {x['scene']}: recorded with another policy, config or code"); meta['completed'] = False
        if not (run / 'branches' / 'inputs' / f"{x['scene']}.npz").exists() or not s2b.inputs_ok(run, x):
            meta['problems'].append(f"scene {x['scene']}: saved online inputs are missing or differ from the snapshot"); meta['completed'] = False
        if not files_ok(run, x, first=x['scene'] in scenes[:2]):
            meta['problems'].append(f"scene {x['scene']}: trace archive or videos are missing or incomplete"); meta['completed'] = False
        else:
            usable.append(x); bad = [k for k, v in trace_checks(run, x, c).items() if not v]
            if bad: meta['problems'].append(f"scene {x['scene']}: trace checks failed: {bad}"); meta['completed'] = False
        bad = [k for b in x['blocks'] for cell in b['cells'] for k in cell['inference_calls'] if k['seed'] != seed_for(x['scene'], k['t'], cell['policy_noise_repeat_id'], 'dp')]
        if bad: meta['problems'].append(f"scene {x['scene']}: {len(bad)} inference seeds do not follow the declared key"); meta['completed'] = False
    meta.update(planned_scenes=len(scenes), decided_scenes=n, policy_sha256=plan['policy']['weights_sha256'], config_sha256=plan['config']['config_sha256'],
                plan_sha256=sha256(a.plan), status=plan['status'], arms_definition=plan['arms'], t_ref=plan['t_ref'], noise=plan['noise'], source_type='real_sim',
                success_definition=c['protocol']['success_definition'], runtime=runtime_info(), not_run_in_this_stage=NOT_RUN)
    res = Path(a.results); summary = p2.summarize(recs, planned_scenes=len(scenes))
    meta['ready_for_interpretation'] = bool(meta['completed'] and meta['reproducibility_ok'])
    write_json(run / 'branches' / 'dataset.json', meta)
    ev = event_rows(run, usable); summary['events'] = {'rows': len(ev), 'scenes_with_usable_traces': len(usable), 'summary': p2.event_summary(ev),
                                                      'note': 'ground-truth positions and flags, logging only; no cause is attached to any row'}
    summary.update(decided_scenes=n, dataset_completed=meta['completed'], ready_for_interpretation=meta['ready_for_interpretation'], problems=meta['problems'], status=plan['status'])
    rows = p2.block_rows(recs)
    write_csv(res / 'pair_outcomes.csv', rows); write_csv(res / 'pair_summary.csv', p2.pair_rows(rows)); write_csv(res / 'scene_summary.csv', p2.scene_rows(recs, rows))
    write_csv(res / 'control_duration_reference.csv', p2.reference_rows(recs)); write_csv(res / 'handoff_parity.csv', p2.handoff_rows(recs))
    write_csv(res / 'event_rows.csv', ev); write_csv(res / 'event_summary.csv', p2.event_summary(ev))
    write_csv(res / 'inference_step_seed_trace.csv', p2.call_rows(recs)); write_csv(res / 'skip_reasons.csv', p2.skip_rows(recs))
    write_json(res / 'observation_fields.json', {**FIELDS, 'cameras': sorted({n for x in recs for n in x.get('cameras', [])}), 'trace_columns': list(p2.TRACE_COLUMNS),
                                                 'phases': list(p2.PHASES)})
    write_json(res / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('scenes_recorded', 'snapshots', 'blocks', 'continuations', 'dataset_completed', 'ready_for_interpretation', 'problems',
                                              'reproducibility', 'handoff', 't_ref')}, indent=2))
    if not meta['ready_for_interpretation']: raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--out', required=True); q.add_argument('--config', default='configs/stackcube.yaml')
    q.add_argument('--policy', default='artifacts/policies/stack_1d_s1_d500_clean_v1_i080000'); q.add_argument('--splits', default='configs/splits.json')
    q.add_argument('--policy-eval', default='configs/policy_eval_splits.json'); q.add_argument('--scenes', type=int, default=12)
    q.add_argument('--stage2b-plan', default='runs/stage2b_001/plan.json')
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
            q.add_argument('--reserve-usd', type=float, default=0.20); q.add_argument('--workers', type=int, default=2); q.add_argument('--efficiency', type=float, default=0.7)
        if name == 'finalize': q.add_argument('--results', required=True)
    a = p.parse_args(); globals()['cmd_' + a.cmd](a)


if __name__ == '__main__':
    main()
