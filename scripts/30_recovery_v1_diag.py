"""Recovery-V1 diagnostic (work id recovery_v1_diagnostic_001): residual failures of the frozen Vision-V1 base policy.
ONE pass of the fixed policy without intervention on 100 new scenes, with a per-step ground-truth trace, and one
mechanism label per failed episode by the fixed rules of hci.recovery_v1. No training, no recovery action, no
calibration, no Test. PREPARATION ONLY at the time of writing: nothing below `dry-run` has been run.

  plan        local: frozen policy and its backup against the Vision-V1 records, code hashes, the new seed manifest, rules
  dry-run     local: files and volume; starts nothing
  scenes      server: initial cube placement of the 100 diagnostic seeds against every placement on record
  probe       server: 3 already-used development scenes with the full recorder (speed, recorder check, reproduction of the record)
  budget      server: projected all-in cost against the approved cap (refuses if it does not fit)
  run         server: the diagnostic pass. The set is marked used in a ledger before any rollout. No resume.
  summarize   anywhere: labels recomputed from the stored traces, counts and cases per label
Server commands refuse to run unless <run>/authorization.json exists, names the plan by hash and carries a cap.
No command in this file writes that authorisation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, gzip, json, subprocess, time
import numpy as np
from hci.io import load_config, read_json, write_json, sha256, digest, runtime_info, UPSTREAM_COMMIT
from hci import recovery_v1 as RV

WORK = 'recovery_v1_diagnostic_001'
POLICY = {'path': 'artifacts/policies/stack_v1_s1_d500_clean_resnet18_i080000', 'weights.pt': 'a37be25603d6c7b7d5cfa1872e6c1134c436f78de02b745c481c88d5d0f69629',
          'policy_meta.json': '4ccfba34ee42b3bab344f7d320c57c6a97011f5adf8b93c04a36b886303b138e', 'train_seed': 1, 'iteration': 80000, 'weights': 'EMA',
          'source_checkpoint_sha256': '5cd8e95d589154981569cad2a3a5f4c8cd670da4444ac7ecb8bf6971484652ed'}
BACKUP = {'archive': 'runs/vision_v1_001/server_backup_run2/vision_v1_out.tgz', 'archive_sha256': '3a8d7cbf0d08aa61c18b51b2199cd2d97ba331f3f900bd0bb3ef595f8642c8cb',
          'file_list': 'results/vision_v1_001/run2/archive_file_sha256.txt', 'extracted': 'runs/vision_v1_001/server_backup_run2/extracted',
          'source_checkpoint': 'runs/policy/stack_v1_s1_d500_clean_resnet18/checkpoints/full_080000.pt'}
VISION = {'plan': 'runs/vision_v1_001/plan.json', 'plan_sha256': '5fdee368fa529c0096464332e1af5dc2ac0eca39ff102ad7bc0fa8605ed3678e', 'selection': 'results/vision_v1_001/run2/selection.json',
          'paired': 'results/vision_v1_001/run2/qualify/paired_result.json'}
CONFIG = 'configs/stackcube.yaml'
MANIFEST = 'configs/recovery_v1_splits.json'
SEED_SOURCES = {'splits': 'configs/splits.json', 'policy_eval': 'configs/policy_eval_splits.json', 'vision_v1': 'configs/vision_v1_splits.json'}
DEMO_MANIFEST = 'runs/stage1d_clean_002/data/train_d500_clean_manifest.json'
PRIOR_SCENES = ('results/stage2a/scenes.json', 'results/stage2b/scenes.json', 'results/stage2d/scenes.json', 'results/handoff_h1_001/scenes.json')
KEYS = 'runs/vision_v1_001/reference_scene_keys.json'
PROBE_REFERENCE = 'results/vision_v1_001/run2/select_seed1/episodes_select_seed1.jsonl'
PROBE_SEEDS = (50400, 50401, 50402)             # vision_v1_select, already used for selection; re-run only to time the recorder and to compare with the record
INFERENCE_CODE = ['src/hci/vision_v1.py', 'src/hci/policies.py', 'src/hci/simulation.py', 'src/hci/experiment.py', 'src/hci/observations.py', 'src/hci/io.py', 'src/hci/schema.py', 'src/hci/trees.py',
                  'src/hci/diagnostics.py', 'src/hci/pilot.py', 'src/hci/statistics.py']
CODE = INFERENCE_CODE + ['src/hci/recovery_v1.py', 'scripts/30_recovery_v1_diag.py', 'scripts/recovery_v1/run.sh', 'scripts/stage1b/common.sh', CONFIG, MANIFEST]
# every file a server command of this script opens for reading (the Vision-V1 run stopped once because such a list was incomplete; tests/test_recovery_v1.py checks this one against the source)
SERVER_READS = sorted({*CODE, *SEED_SOURCES.values(), DEMO_MANIFEST, *PRIOR_SCENES, KEYS, PROBE_REFERENCE, POLICY['path'] + '/weights.pt', POLICY['path'] + '/policy_meta.json'})
LEDGER = 'artifacts/ledger/recovery_v1_diag.json'
IDENTICAL_M = 1e-5
STATUS = {'vision_v1': 'EXECUTED_ONCE, TARGET_MET (79/100 on vision_v1_qualify), unchanged', 'base_policy': 'FROZEN: Vision-V1 train seed 1, 80k EMA', 'recovery_v1_diagnostic': 'PREPARATION_ONLY',
          'diagnostic_rollouts': 'NOT_RUN', 'paid_compute': 'NOT_AUTHORIZED', 'new_training': 'NOT_PLANNED', 'recovery_action': 'NOT_DESIGNED (follows the diagnosis)', 'recovery_pilot': 'NOT_RUN',
          'calibration': 'NOT_RUN', 'final_test': 'NOT_RUN'}


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()


def require_authorization(run, plan_path):
    from hci.vision_v1 import require_authorization as f
    return f(run, plan_path)


def used_ledger():
    used = {f'{n}/{k}': list(v) for n, p in SEED_SOURCES.items() for k, v in read_json(p)['splits'].items()}
    used['demonstration_scenes'] = list(read_json(DEMO_MANIFEST)['all_scene_seeds'])
    for p in PRIOR_SCENES: used[p] = list(read_json(p)['list'])
    return used


def check_policy(problems):
    """The frozen base policy on this machine is the one selected and qualified in Vision-V1."""
    d = Path(POLICY['path']); m = read_json(d / 'policy_meta.json'); sel = read_json(VISION['selection']); pr = read_json(VISION['paired'])
    listed = dict(reversed(l.split(None, 1)) for l in Path(BACKUP['file_list']).read_text().splitlines() if l.strip())
    facts = {'weights_sha256_now': sha256(d / 'weights.pt'), 'policy_meta_sha256_now': sha256(d / 'policy_meta.json'), 'meta_weights_sha256': m['weights_sha256'],
             'selection_weights_sha256': sel['selected_weights_sha256'], 'qualified_weights_sha256': pr['new']['weights_sha256'],
             'backup_list_weights_sha256': listed.get(POLICY['path'] + '/weights.pt'), 'backup_list_meta_sha256': listed.get(POLICY['path'] + '/policy_meta.json'),
             'backup_list_source_checkpoint_sha256': listed.get(BACKUP['source_checkpoint']), 'qualification': f"{pr['new']['success_once']}/{pr['new']['n']} ({pr['label']})"}
    if len({facts[k] for k in ('weights_sha256_now', 'meta_weights_sha256', 'selection_weights_sha256', 'qualified_weights_sha256', 'backup_list_weights_sha256')} | {POLICY['weights.pt']}) != 1:
        problems.append('the policy weights are not the selected and qualified Vision-V1 weights')
    if not (facts['policy_meta_sha256_now'] == facts['backup_list_meta_sha256'] == POLICY['policy_meta.json']): problems.append('policy metadata differs from the backup record')
    if (m.get('train_seed'), m.get('iteration'), m.get('weights'), sel['selected_train_seed'], m.get('upstream_commit')) != (1, 80000, 'EMA', 1, UPSTREAM_COMMIT): problems.append('not the train seed 1, 80k EMA candidate')
    if not (m['source_checkpoint_sha256'] == facts['backup_list_source_checkpoint_sha256'] == POLICY['source_checkpoint_sha256']): problems.append('source checkpoint hash differs from the backup record')
    arc = Path(BACKUP['archive']); facts['archive_present'] = arc.is_file()
    if arc.is_file():
        facts['archive_sha256_now'] = sha256(arc)
        if facts['archive_sha256_now'] != BACKUP['archive_sha256']: problems.append('the backup archive hash differs from the record')
        ck = Path(BACKUP['extracted']) / BACKUP['source_checkpoint']; facts['extracted_source_checkpoint_sha256'] = sha256(ck) if ck.is_file() else None
        if facts['extracted_source_checkpoint_sha256'] != POLICY['source_checkpoint_sha256']: problems.append('the extracted 80k checkpoint is missing or differs')
        cp = Path(BACKUP['extracted']) / POLICY['path'] / 'weights.pt'; facts['extracted_weights_sha256'] = sha256(cp) if cp.is_file() else None
        if facts['extracted_weights_sha256'] != POLICY['weights.pt']: problems.append('the extracted policy weights are missing or differ')
    else: facts['archive_note'] = 'INPUT_UNAVAILABLE on this machine: the archive is kept on the local PC only; the policy files themselves are checked'
    vp = read_json(VISION['plan']); facts['vision_plan_sha256_now'] = sha256(VISION['plan'])
    if facts['vision_plan_sha256_now'] != VISION['plan_sha256']: problems.append('the Vision-V1 plan file changed')
    same = {f: sha256(f) == vp['code'].get(f) for f in INFERENCE_CODE if f in vp['code']}; facts['inference_code_identical_to_the_vision_v1_plan'] = same
    if not all(same.values()): problems.append(f'inference-path files changed since the Vision-V1 run: {[f for f, v in same.items() if not v]}')
    return facts


def cmd_plan(a):
    c = load_config(CONFIG); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); problems = []
    if (out / 'plan.json').exists(): raise SystemExit(f'Refused overwrite: {out / "plan.json"}')
    try: seeds = RV.diag_seeds(read_json(MANIFEST), 'recovery_v1_diag', used_ledger())
    except ValueError as e: problems.append(str(e)); seeds = []
    policy = check_policy(problems)
    if [c['policy'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')] != [2, 8, 16] or c['env']['horizon'] != 200: problems.append('config is not the fixed 2/8/16, horizon 200')
    plan = {'work_id': WORK, 'stage': 'Recovery-V1: diagnosis of the residual failures of the Vision-V1 base policy', 'created_utc': datetime.now(timezone.utc).isoformat(), 'pass': not problems,
            'problems': problems, 'status': STATUS,
            'authorization': {'paid_compute_authorized': False, 'proposed_cap_usd': 2.00, 'proposed_cap_is': 'a proposal by the user, explicitly not an approval; earlier approvals are not inherited',
                              'required_file': 'authorization.json in the run folder: {"approved_by": "user", "paid_compute_authorized": true, "cap_usd": <number>, "plan_sha256": "<sha256 of this plan>"}'},
            'question': 'By which mechanism do the remaining episodes of the frozen Vision-V1 policy fail, and which single mechanism is the most frequent?',
            'repository': {'head': git('rev-parse', 'HEAD'), 'working_tree_changes_at_plan_time': git('status', '--porcelain').splitlines()},
            'base_policy': {**POLICY, 'checks': policy, 'rule': 'frozen; not retrained, not fine-tuned, no other checkpoint'},
            'config': {'path': CONFIG, 'file_sha256': sha256(CONFIG), 'config_sha256': digest(c), 'policy': c['policy'], 'env': c['env']},
            'existing_traces': {'vision_v1_per_episode_files': 'seed, success_once, steps, end reason, policy calls, timing, furthest stage only',
                                'per_step_trace_of_the_new_policy': 'none was stored in Vision-V1 (the stage label was computed in memory and the trace discarded)',
                                'sufficient_for': ['furthest stage reached'], 'missing': ['executed actions and gripper command per step', 'finger opening', 'cube poses, tilt and velocities per step',
                                'contact forces cube-cube, cube-finger, cube-table', 'time and pose of the release', 'whether the success pose was reached while still grasped', 'camera frames of failed episodes'],
                                'decision': 'a new diagnostic pass is needed; vision_v1_select / vision_v1_qualify are not re-run for it (3 select scenes are re-run by `probe` for timing only)'},
            'seed_manifest': {'path': MANIFEST, 'sha256': sha256(MANIFEST), 'sets': {k: [lo, hi, hi - lo + 1] for k, (lo, hi) in RV.SETS.items()}, 'runnable_here': list(RV.RUNNABLE),
                              'diag': seeds[:1] + seeds[-1:], 'reserved': 'recovery_v1_pilot_reserved (50700-50799) is declared so that no other work takes it; no command of this work resets or runs it',
                              'seed_level_check': 'no overlap with splits.json, policy_eval_splits.json, vision_v1_splits.json, the 550 demonstration scene seeds, or the scenes of Stage 2A / 2B / 2D / H1',
                              'placement_check': 'server step `scenes`: cube placements of the 100 diagnostic seeds against stored placements (demonstrations, pilots) and, by reset, against audit / dev / '
                                                 'policy-evaluation / Vision-V1 sets; calibration and Test by seed number only. An identical placement fails the step',
                              'untouched_sets': ['calibration', 'test', 'vision_v1_qualify', 'policy_qualify_A', 'policy_qualify_B', 'recovery_v1_pilot_reserved']},
            'recording': {'per_step': ['executed action (xyz delta, gripper command)', 'TCP position and orientation', 'finger joint positions (opening)', 'cubeA and cubeB position, orientation, linear and angular velocity',
                                       'task flags: grasped, cubeA on cubeB, static, success', 'contact force magnitude: cubeA-cubeB, cubeA-each finger, cubeA-table, cubeB-each finger'],
                          'frames': 'both policy cameras at every step, stored for failed episodes only (diagnostic evidence; the success flag decides storage, not any policy input)',
                          'privileged': 'all of it is simulator ground truth, written to files after each step; the policy receives exactly the Vision-V1 observation and nothing from the recorder'},
            'classification': {'labels': RV.LABELS, 'rule_order': RV.RULE_ORDER, 'thresholds': RV.THRESHOLDS, 'user_categories': {'not opened': 'NOT_RELEASED_AT_PLACE', 'opened too early': 'EARLY_RELEASE_ABOVE',
                               'position error': ['RELEASE_OFF_TARGET_XY', 'RELEASE_BELOW_REST_HEIGHT', 'HOLDING_NOT_PLACED'], 'unstable after contact': ['POST_RELEASE_UNSTABLE', 'IN_PLACE_NOT_STATIC']},
                               'frozen': 'rules and thresholds are fixed by the hash of src/hci/recovery_v1.py in this plan, before any diagnostic episode. A rule changed after seeing traces is reported as post hoc',
                               'unknown': 'an episode no rule explains is UNKNOWN; it is counted, not forced into a label'},
            'probe': {'seeds': list(PROBE_SEEDS), 'from': 'vision_v1_select (already used)', 'purpose': 'recorder works on this host; seconds per control step; does the run reproduce the recorded success and step count',
                      'reproduction_rule': 'a difference from the record is reported and does not block (another GPU may change floating-point results); a recorder error blocks'},
            'run': {'episodes': 100, 'intervention': 'none', 'ledger': LEDGER + ', created exclusively before any rollout; no resume, no second pass', 'on_error': 'the pass stops and stays INCOMPLETE; no episode is dropped or replaced'},
            'outputs_after_the_run': ['counts and cases per mechanism label with the log evidence of every verdict', 'ONE proposed local recovery action for the most frequent mechanism (not a repeat of the 5 cm retreat by default)',
                                      'a controlled pilot plan: C0 (no intervention), R5 (existing 5 cm retreat) and the new action from the same state with the same noise keys, on recovery_v1_pilot_reserved, '
                                      'triggered by a rule that does not use the outcome', 'projected GPU cost of that pilot'],
            'pilot_design_constraints': {'arms': ['C0 no intervention', 'R5 existing retreat', 'ONE new corrective action'], 'same_state': 'branches start from one snapshot restored by prefix action replay (hci.simulation.restore_env), same scene seed',
                                         'same_noise': "policy noise keyed by seed_for(scene, physical step, repeat, 'dp') as in every earlier pilot", 'selection_of_states': 'a trigger computed from quantities available at the '
                                         'decision step, fixed before the pilot; never chosen from the known outcome of the continuation', 'status': 'constraints only; the action and the trigger are written after the diagnosis and need their own approval'},
            'volume': {'placement_resets': 100 + 5 + 100 + 270 + 200, 'probe_episodes': len(PROBE_SEEDS), 'diagnostic_episodes': 100},
            'cost_estimate': {'basis': 'Vision-V1 run 2, RTX 4090 at US$0.4539/h: 19 s per episode on vision_v1_select with the same policy (1.1 s per policy call)',
                              'diagnostic_pass_minutes': 32, 'install_checks_transfer_minutes': 45, 'estimated_total_usd': [0.60, 1.00], 'proposed_cap_usd': 2.00,
                              'gate': 'the server `budget` step projects the total from the measured seconds per step at the full 200 steps for every episode and refuses above the approved cap'},
            'files_read_on_the_server': SERVER_READS, 'code': {f: sha256(f) for f in CODE}, 'upstream_commit': UPSTREAM_COMMIT,
            'not_in_this_work': ['policy training or fine-tuning', 'any recovery action', 'R5', 'selector training', 'calibration', 'final Test', 'the reserved pilot seeds']}
    write_json(out / 'plan.json', plan); print(json.dumps({'pass': plan['pass'], 'problems': problems, 'base_policy_checks': policy}, indent=1))
    if problems: raise SystemExit(2)


def cmd_dry_run(a):
    plan = read_json(a.plan); run = Path(a.run); need = plan['files_read_on_the_server']
    missing = [f for f in need if not Path(f).exists()]; changed = [f for f, h in plan['code'].items() if Path(f).exists() and sha256(f) != h]; auth = (run / 'authorization.json').exists()
    out = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha256(a.plan), 'volume': plan['volume'], 'files_needed_on_the_server': need,
           'upload_bytes': int(sum(Path(f).stat().st_size for f in need if Path(f).exists())), 'files_missing': missing, 'code_files_changed_since_plan': changed, 'authorization_present': auth,
           'ledger_exists': Path(LEDGER).exists(), 'paid_compute': 'AUTHORIZATION_FILE_PRESENT_NOT_USED_BY_DRY_RUN' if auth else 'NOT_AUTHORIZED', 'servers_created': 0, 'policy_inferences': 0}
    print(json.dumps({k: v for k, v in out.items() if k != 'files_needed_on_the_server'}, indent=1))
    if a.out: write_json(a.out, out)
    if missing or changed: raise SystemExit(2)


def bound(plan_path):
    """Hashes of what is about to run against the plan. Any difference refuses the command."""
    plan = read_json(plan_path); bad = [f for f, h in plan['code'].items() if sha256(f) != h]
    for f in ('weights.pt', 'policy_meta.json'):
        if sha256(Path(POLICY['path']) / f) != POLICY[f]: bad.append('policy ' + f)
    if bad: raise SystemExit(f'Refused: these differ from the plan: {bad}')
    return plan, load_config(CONFIG)


def cmd_scenes(a):
    run = Path(a.run); require_authorization(run, a.plan); plan, c = bound(a.plan); man = read_json(MANIFEST); seeds = RV.diag_seeds(man, 'recovery_v1_diag', used_ledger())
    from hci.simulation import make_env
    from hci.trees import as_numpy
    env = make_env(c)

    def key(seed):
        env.reset(seed=int(seed)); return np.array([*as_numpy(env.cubeA.pose.p)[0], *as_numpy(env.cubeB.pose.p)[0]], float)
    try:
        refs = {n: np.array(k) for n, k in read_json(KEYS)['keys'].items()}; stored = len(refs); sp = read_json(SEED_SOURCES['splits'])['splits']
        reset_sets = {'splits/audit': sp['audit'], 'splits/dev': sp['dev'], **{f'policy_eval/{k}': v for k, v in read_json(SEED_SOURCES['policy_eval'])['splits'].items()},
                      **{f'vision_v1/{k}': v for k, v in read_json(SEED_SOURCES['vision_v1'])['splits'].items()}}
        for name, ids in reset_sets.items():
            for s in ids: refs[f'{name}/{s}'] = key(s)
        names = list(refs); table = np.stack([refs[n] for n in names]); own = {}; clashes = []
        for s in seeds:
            k = key(s); d = np.abs(table - k).max(1); j = int(d.argmin()); twin = [t for t, v in own.items() if np.abs(v - k).max() <= IDENTICAL_M]
            if d[j] <= IDENTICAL_M or twin: clashes.append({'seed': s, 'identical_to': names[j] if d[j] <= IDENTICAL_M else twin[0]})
            own[str(s)] = k
        report = {'pass': not clashes, 'identical_placements': clashes, 'seeds_checked': len(own), 'identical_threshold_m': IDENTICAL_M,
                  'references': {'stored_keys_demonstrations_and_pilots': stored, **{k: len(v) for k, v in reset_sets.items()}}, 'seed_level_only': ['splits/calibration', 'splits/test'],
                  'not_reset': 'recovery_v1_pilot_reserved', 'plan_sha256': sha256(a.plan), 'scope': 'Initial cube placement only; no policy was run on these scenes.'}
        write_json(run / 'scenes.json', report); print(json.dumps({k: report[k] for k in ('pass', 'identical_placements', 'seeds_checked')}))
        if clashes: raise SystemExit(2)
    finally:
        env.close()


def frame_of(r):
    from hci.trees import as_numpy
    return np.concatenate([as_numpy(r.raw['sensor_data'][n]['rgb'])[0] for n in sorted(r.raw['sensor_data'])], axis=1).astype(np.uint8)


def check_rows(initial, rows):
    """Every recorded number is finite and every recorder field is present; a broken recorder must not produce labels."""
    need = ('tcp_q', 'finger_qpos', 'cubeA_q', 'cubeB_q', 'cubeA_v', 'cubeA_w', 'cubeB_v', 'cubeB_w', 'f_A_B', 'f_A_left', 'f_A_right', 'f_A_table', 'f_B_left', 'f_B_right')
    for r in [initial, *rows]:
        for k in need:
            if k not in r or not np.isfinite(np.asarray(r[k], float)).all(): raise RuntimeError(f"recorder field {k} missing or not finite at step {r.get('t')}")
    if [r['t'] for r in rows] != list(range(1, len(rows) + 1)): raise RuntimeError('trace steps are not 1..n')


def rollout(r, seed, out, frames_for_failures=True):
    """One episode of the fixed policy without intervention, fully recorded. Returns the episode row."""
    from hci.pilot import Recorder
    tick = time.perf_counter(); r.reset(seed); u = r.env; initial = RV.probe(u, 0)
    with Recorder(r, probe=lambda rt, z: RV.probe(rt.env, rt.t, z), frame=frame_of) as rec:
        rec.rows = rows = []; rec.frames = frames = [frame_of(r)]; res = r.complete()
    if res['intervention'] is not None: raise RuntimeError('an intervention happened in a no-intervention pass')
    check_rows(initial, rows); verdict = RV.classify(initial, rows)
    if (verdict['label'] == 'SUCCESS') != bool(res['success']): raise RuntimeError(f'seed {seed}: label {verdict["label"]} contradicts the runtime outcome')
    tp = out / 'traces' / f'trace_{seed}.json.gz'; tp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(tp, 'xt', encoding='utf-8') as f: json.dump({'seed': int(seed), 'initial': initial, 'rows': rows}, f)
    fp = None
    if frames_for_failures and not res['success']:
        fp = out / 'frames' / f'frames_{seed}.npz'; fp.parent.mkdir(parents=True, exist_ok=True)
        if fp.exists(): raise FileExistsError(fp)
        np.savez_compressed(fp, frames=np.stack(frames), cameras=np.array(sorted(r.raw['sensor_data'])))
    return {'seed': int(seed), 'success_once': int(res['success']), 'steps': res['steps'], 'reason': res['reason'], 'policy_calls': res['policy_calls'], 'policy_inference_seconds': res['policy_inference_seconds'],
            'furthest_stage': verdict['events']['furthest_stage'], 'label': verdict['label'], 'evidence': verdict['evidence'], 'events': verdict['events'], 'flags': verdict['flags'],
            'trace': tp.relative_to(out).as_posix(), 'trace_sha256': sha256(tp), 'frames': None if fp is None else fp.relative_to(out).as_posix(), 'wall_seconds': time.perf_counter() - tick}


def cmd_probe(a):
    run = Path(a.run); require_authorization(run, a.plan); plan, c = bound(a.plan); out = run / 'probe'; out.mkdir(parents=True, exist_ok=False)
    ref = {x['seed']: x for x in map(json.loads, Path(PROBE_REFERENCE).read_text().splitlines()) if x['seed'] in PROBE_SEEDS}
    from hci import vision_v1 as V
    r = V.runtime(c, POLICY['path'], a.upstream, a.device); rows = []
    try:
        for s in PROBE_SEEDS: rows.append(rollout(r, s, out, frames_for_failures=False)); print(json.dumps({k: rows[-1][k] for k in ('seed', 'success_once', 'steps', 'label', 'wall_seconds')}), flush=True)
    finally:
        r.env.close()
    same = {str(x['seed']): [x['success_once'], x['steps']] == [ref[x['seed']]['success_once'], ref[x['seed']]['steps']] for x in rows}
    write_json(out / 'probe.json', {'pass': True, 'episodes': rows, 'seconds_per_step': sum(x['wall_seconds'] for x in rows) / sum(x['steps'] for x in rows),
                                    'recorded': {str(s): [ref[s]['success_once'], ref[s]['steps']] for s in PROBE_SEEDS}, 'reproduces_recorded_run': same, 'all_reproduced': all(same.values()),
                                    'plan_sha256': sha256(a.plan), 'runtime': runtime_info(), 'scope': 'Timing and recorder check on already-used development scenes. Not diagnostic data.'})
    print(json.dumps({'seconds_per_step': read_json(out / 'probe.json')['seconds_per_step'], 'reproduces_recorded_run': same}))


def cmd_budget(a):
    run = Path(a.run); auth = require_authorization(run, a.plan); p = read_json(run / 'probe' / 'probe.json')
    if not p['pass'] or p['plan_sha256'] != sha256(a.plan): raise SystemExit('probe missing, failed or of another plan')
    d = RV.projected_cost(seconds_per_episode=p['seconds_per_step'] * 200, usd_per_hour=a.usd_per_hour, spent_usd=a.spent_usd, cap_usd=float(auth['cap_usd']))
    d.update(status='OK' if d['fits_cap'] else 'BUDGET_BLOCKED', plan_sha256=sha256(a.plan), seconds_per_step=p['seconds_per_step'], bound='every episode charged at the full 200 steps'); write_json(run / 'budget_gate.json', d); print(json.dumps(d, indent=1))
    if not d['fits_cap']: raise SystemExit(2)


def cmd_run(a):
    run = Path(a.run); require_authorization(run, a.plan); plan, c = bound(a.plan)
    for name in ('scenes.json', 'budget_gate.json'):
        if not (run / name).exists(): raise SystemExit(f'{name} missing: the placement check and the budget gate come first')
    sc = read_json(run / 'scenes.json'); bg = read_json(run / 'budget_gate.json')
    if not sc['pass'] or sc['plan_sha256'] != sha256(a.plan) or not bg['fits_cap'] or bg.get('plan_sha256') != sha256(a.plan): raise SystemExit('placement check or budget gate not passed for this plan')
    seeds = RV.diag_seeds(read_json(MANIFEST), 'recovery_v1_diag', used_ledger()); out = run / 'diagnostic'
    if out.exists(): raise SystemExit(f'{out} exists: the diagnostic pass is never resumed or repeated')
    # exclusive create BEFORE any rollout: from here on the set is used, whatever happens next
    write_json(LEDGER, {'set': 'recovery_v1_diag', 'used_utc': datetime.now(timezone.utc).isoformat(), 'session': out.as_posix(), 'plan_sha256': sha256(a.plan), 'seed_manifest_sha256': plan['seed_manifest']['sha256'],
                        'policy': POLICY['path'], 'weights_sha256': POLICY['weights.pt'], 'note': 'Single diagnostic pass without intervention. Written before any outcome existed. No resume, no second pass.'})
    out.mkdir(parents=True); started = time.perf_counter()
    from hci import vision_v1 as V
    r = V.runtime(c, POLICY['path'], a.upstream, a.device); rows = []
    try:
        with (out / 'episodes.jsonl').open('x', encoding='utf-8') as f:
            for s in seeds:
                row = rollout(r, s, out); rows.append(row); f.write(json.dumps(row) + '\n'); f.flush()
                print(f"diag seed={s} success_once={row['success_once']} steps={row['steps']} label={row['label']} ({sum(x['success_once'] for x in rows)}/{len(rows)})", flush=True)
    finally:
        r.env.close()
    write_json(out / 'result.json', {'set': 'recovery_v1_diag', 'seeds': seeds, 'policy': POLICY['path'], 'weights_sha256': POLICY['weights.pt'], **RV.summarize(rows), 'episodes_sha256': sha256(out / 'episodes.jsonl'),
                                     'wall_seconds': time.perf_counter() - started, 'plan_sha256': sha256(a.plan), 'runtime': runtime_info(),
                                     'scope': 'One development pass without intervention. Labels describe the logged mechanism of each failure by fixed rules; they are not causes proven by intervention.'})
    print(json.dumps({k: v for k, v in read_json(out / 'result.json').items() if k in ('n', 'success_once', 'failures', 'failure_labels', 'largest_label', 'unknown')}, indent=1))


def cmd_summarize(a):
    """Labels recomputed from the stored traces with the current rules; any difference from the stored label is listed, never overwritten."""
    d = Path(a.diagnostic); eps = [json.loads(l) for l in (d / 'episodes.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]; again = []; differ = []
    for e in eps:
        if sha256(d / e['trace']) != e['trace_sha256']: raise SystemExit(f"trace of seed {e['seed']} changed")
        with gzip.open(d / e['trace'], 'rt', encoding='utf-8') as f: t = json.load(f)
        v = RV.classify(t['initial'], t['rows']); again.append({**e, 'label': v['label'], 'evidence': v['evidence'], 'events': v['events'], 'flags': v['flags']})
        if v['label'] != e['label']: differ.append({'seed': e['seed'], 'stored': e['label'], 'recomputed': v['label']})
    s = RV.summarize(again)
    out = {**s, 'episodes_sha256': sha256(d / 'episodes.jsonl'), 'recomputed_labels_equal_stored': not differ, 'label_differences': differ, 'rules_sha256': sha256('src/hci/recovery_v1.py'),
           'cases': {k: [{'seed': e['seed'], 'steps': e['steps'], 'furthest_stage': e['furthest_stage'], 'evidence': e['evidence'], 'flags': e['flags']} for e in again if not e['success_once'] and e['label'] == k] for k in s['failure_labels']},
           'stage_by_label': {k: {st: sum(1 for e in again if not e['success_once'] and e['label'] == k and e['furthest_stage'] == st) for st in sorted({e['furthest_stage'] for e in again})} for k in s['failure_labels']}}
    write_json(a.out, out); print(json.dumps({k: out[k] for k in ('n', 'success_once', 'failures', 'largest_label', 'largest_is_tied', 'unknown', 'recomputed_labels_equal_stored')}, indent=1))
    print(json.dumps({k: v['count'] for k, v in out['failure_labels'].items()}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--out', required=True)
    q = s.add_parser('dry-run'); q.add_argument('--plan', required=True); q.add_argument('--run', required=True); q.add_argument('--out')
    for name in ('scenes', 'probe', 'budget', 'run'):
        q = s.add_parser(name); q.add_argument('--plan', required=True); q.add_argument('--run', required=True)
        if name in ('probe', 'run'): q.add_argument('--upstream', default='third_party/ManiSkill'); q.add_argument('--device', default='cuda')
        if name == 'budget': q.add_argument('--usd-per-hour', type=float, required=True); q.add_argument('--spent-usd', type=float, required=True)
    q = s.add_parser('summarize'); q.add_argument('--diagnostic', required=True); q.add_argument('--out', required=True)
    a = p.parse_args(); globals()['cmd_' + a.cmd.replace('-', '_')](a)


if __name__ == '__main__':
    main()
