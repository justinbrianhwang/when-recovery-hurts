"""Recovery-V1 oracle XY pilot (work id recovery_v1_oracle_xy_pilot_001): centre before release as a DIAGNOSTIC ORACLE arm.
Four arms from one state captured just before the frozen Vision-V1 policy's first pending release command:
C0, R5, HOLD_MATCHED, CENTER_ORACLE (hci.recovery_v1_oracle). PREPARATION ONLY at the time of writing: nothing below
`dry-run` has been run. No training, no calibration, no Test.

  plan        local: frozen policy, code hashes, seed registry, frozen semantics, analysis, cost estimate
  dry-run     local: files and volume; starts nothing
  scenes      server: cube placements of the 100 pilot seeds against every placement on record (no policy)
  audit       server: 5 already-used scenes: the full four-arm procedure with every technical check, controller scaling,
              and the tracking audit of the bounded horizontal move (TECHNICAL_BLOCKED if it fails)
  budget      server: projected all-in cost at the worst case against the approved cap (refuses if it does not fit)
  run         server: the one pilot pass. The set is marked used in a ledger before any rollout. No resume.
  summarize   anywhere: every table recomputed from the stored scene records
Server commands refuse to run unless <run>/authorization.json exists, names the plan by hash and carries a cap.
No command in this file writes that authorisation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, gzip, importlib.util, json, re, time
import numpy as np
from hci.io import load_config, read_json, write_json, sha256, digest, runtime_info, UPSTREAM_COMMIT
from hci import recovery_v1_oracle as O

_spec = importlib.util.spec_from_file_location('rv1_diag', Path(__file__).with_name('30_recovery_v1_diag.py')); D = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(D)

WORK = O.WORK
POLICY = D.POLICY; CONFIG = D.CONFIG; MANIFEST = D.MANIFEST; KEYS = D.KEYS
AUDIT_SEEDS = (50400, 50401, 50402, 50403, 50404)        # vision_v1_select, already used for selection; never an evaluation scene of this pilot
AUDIT_REFERENCE = D.PROBE_REFERENCE
DIAG_RECORD = {'ledger': 'results/recovery_v1_diagnostic/run1/ledger_recovery_v1_diag.json', 'result': 'results/recovery_v1_diagnostic/run1/result.json'}
OFFLINE = 'results/recovery_v1_oracle_xy_pilot_001/preparation/offline_trace_analysis.json'
UPSTREAM_SOURCES = 'results/recovery_v1_oracle_xy_pilot_001/preparation/upstream_controller_sources.json'
CODE = D.INFERENCE_CODE + ['src/hci/recovery_v1.py', 'src/hci/recovery_v1_oracle.py', 'src/hci/pilot2b.py', 'src/hci/pilot2d.py', 'scripts/30_recovery_v1_diag.py', 'scripts/31_recovery_v1_oracle_xy.py',
                           'scripts/recovery_v1_oracle/run.sh', 'scripts/stage1b/common.sh', CONFIG, MANIFEST]
SERVER_READS = sorted({*CODE, *D.SEED_SOURCES.values(), D.DEMO_MANIFEST, *D.PRIOR_SCENES, KEYS, AUDIT_REFERENCE, POLICY['path'] + '/weights.pt', POLICY['path'] + '/policy_meta.json'})
LEDGER = 'artifacts/ledger/recovery_v1_oracle_xy_pilot.json'
IDENTICAL_M = 1e-5
DRIVE = 'Robot Harm/results/recovery_v1_oracle_xy_pilot_001/'
STATUS = {'recovery_v1_oracle_xy_pilot': 'PHASE_A_PREPARATION_ONLY', 'paid_compute': 'NOT_AUTHORIZED', 'pilot_rollouts': 'NOT_RUN', 'base_policy': 'FROZEN: Vision-V1 train seed 1, 80k EMA, unchanged',
          'recovery_v1_diagnostic': 'EXECUTED_ONCE (79/100), unchanged', 'center_oracle': 'DIAGNOSTIC REFERENCE ARM, NOT DEPLOYABLE', 'image_estimator': 'NOT_IN_THIS_CYCLE',
          'new_training': 'NOT_PLANNED', 'calibration': 'NOT_RUN', 'final_test': 'NOT_RUN'}
SEMANTICS = {
    'trigger': {'rule': 'first control step at which ALL hold: (1) the action the policy will execute next has gripper command > 0 (open); (2) the 3 or more most recently executed gripper commands '
                        'are all < 0 (close); (3) the measured finger opening (sum of the two finger joints in agent/qpos) is between 0.02 m and 0.06 m',
                'inputs': 'the pending action, the executed actions, agent/qpos. No grasp flag, cube pose, task flag, outcome or future information',
                'pending_action': 'the head of the unexecuted queue; if the queue is empty the next chunk is inferred at this step exactly as the policy would infer it (same noise key and counters)',
                'capture': 'the full runtime (simulator by prefix replay, history, queue with the pending action, counters, RNG) BEFORE the pending action is executed', 'per_episode': 'at most one',
                'no_trigger': 'the scene stays in every denominator; every arm carries the outcome of the uninterrupted run; no continuation is run',
                'false_trigger': 'cubeA not grasped or not lifted 1 cm at the trigger, read from ground truth AFTER the fact; recorded, never excluded; all four arms are still run',
                'thresholds_from': 'gripper sign and limits from the pinned Panda source; the width window and the run length were chosen on the stored traces of the consumed diagnostic set (development data)',
                **O.TRIGGER},
    'gripper_in_control_arms': 'R5, HOLD_MATCHED and CENTER_ORACLE keep the last executed gripper command, which the trigger guarantees to be a close command (< 0); a state where it is not is a technical stop. '
                               'No arm ever commands the gripper open; the policy decides when to open after it gets control back',
    'C0': 'the pending action and the rest of the queue are executed; history untouched; must equal the uninterrupted run (actions, outcome, final state), otherwise the run stops',
    'R5': 'Runtime.retreat() unchanged (5 cm up, 5 mm per step, at most 40 steps, 5 mm arrival tolerance), queue cleared before and after, then normal re-inference. Runtime.force(2) is not used because its '
          'Stage-2 eligibility grid (step multiple of 4, inside a chunk, 48 steps left) does not contain the trigger step; the three statements of force() are executed without the grid. Secondary comparator only',
    'CENTER_ORACLE': {'reads': 'exact simulator positions of cubeA and cubeB (horizontal offset between their centres), re-read every control step', 'stops_when': 'offset <= 0.010 m, or 12 steps, or the target cap leaves no move, or the episode ends',
                      'command': 'horizontal only, towards removing the offset, at most 0.005 m per control step (0.05 in normalised action); vertical command exactly 0; gripper command held',
                      'target_cap': 'the commanded TCP target never leaves a horizontal distance of 0.030 m from the TCP position at the trigger',
                      'cap_reading': '"3 cm" bounds the commanded TARGET displacement (as `commanded_distance_m` of the existing retreat), not the sum of per-step commands; the sum is recorded and can reach 0.06 m in 12 steps. '
                                     'The alternative reading (sum of commands <= 0.03 m) would stop after 6 steps; see offline evidence', 'after': 'queue cleared, the unchanged policy re-infers at trigger + T',
                      'not': 'no forced opening, no teleport, no state write, no extra steps beyond the 200-step episode', **O.BOUNDS},
    'HOLD_MATCHED': 'queue cleared; for exactly T control steps (T = the number of steps CENTER_ORACLE used on this state, from a controller-only reference run without policy inference) the TCP horizontal position '
                    'of the trigger is held by feedback with the same 0.005 m bound, vertical command exactly 0, gripper command held; then the policy re-infers at trigger + T. Reads no object pose',
    'zero_step_case': 'if the offset is already <= 0.010 m, T = 0: in BOTH arms the pending queue is discarded and the policy re-infers at the trigger step with the trigger-step noise key; both arms are run and must be identical',
    'matched_handoff': 'HOLD_MATCHED and CENTER_ORACLE: no inference during control, empty queue at return, same number of control steps, first inference at trigger + T with one and the same seed; any difference stops the run',
    'episode_end_during_control': 'an arm whose episode ends inside its control part keeps that outcome; nothing is re-run',
    'noise': "repeat 0 only; every inference is keyed by seed_for(scene, physical step, 0, 'dp')", 'restore': 'reset(seed) + prefix action replay; set_state is never used; the restored runtime is compared with the snapshot before every branch'}
ANALYSIS = {'unit': 'scene (100)', 'primary': 'paired success_once_absorbing, CENTER_ORACLE against C0, on all 100 scenes: both succeed / rescue / harm / both fail, Net = rescue - harm, harm / C0 successes, Wilson intervals',
            'secondary': ['CENTER_ORACLE against HOLD_MATCHED', 'HOLD_MATCHED against C0', 'R5 against C0 (not duration-matched)', 'TCP and cube displacement, vertical drift, offset before and after, grasp kept, release location, extra steps'],
            'denominators': ['all', 'no trigger', 'triggered', 'false triggers', 'actually moved (T > 0)', 'already aligned (T = 0)', 'C0 success', 'C0 failure'],
            'followup_gate': {'all_of': ['CENTER_ORACLE rescues >= 4 scenes', 'Net >= +3', 'harm <= 3 C0-successful scenes', 'CENTER_ORACLE successes > HOLD_MATCHED successes', 'all technical audits pass'],
                              'otherwise': 'NO_ORACLE_GATE_PASS: the candidate is closed for this cycle; no extension, no tuning of bounds, no search for subsets', **O.GATE},
            'not_claimed': ['statistical significance', 'deployability of CENTER_ORACLE', 'anything about perception', 'a comparison with 79/100 of earlier sets']}


def used_ledger():
    used = D.used_ledger(); used['recovery_v1/recovery_v1_diag'] = list(read_json(MANIFEST)['splits']['recovery_v1_diag']); return used


def recorded_use_scan():
    """Any record in results/ or a ledger that names a pilot seed as a seed, scene or episode."""
    pat = re.compile(r'"(?:seed|scene|episode)":\s*(507\d\d)\b'); hits = {}; n = 0
    for p in sorted(Path('results').rglob('*')):
        if p.suffix not in ('.json', '.jsonl') or WORK in p.parts or not p.is_file(): continue
        n += 1; found = sorted(set(pat.findall(p.read_text(encoding='utf-8', errors='replace'))))
        if found: hits[p.as_posix()] = found[:5]
    ledgers = {p.name: read_json(p).get('set') for p in sorted(Path('artifacts/ledger').glob('*.json'))} if Path('artifacts/ledger').is_dir() else {}
    return {'files_scanned': n, 'files_naming_a_pilot_seed': hits, 'ledgers_on_this_machine': ledgers, 'pilot_ledger_exists': Path(LEDGER).exists()}


def cmd_plan(a):
    c = load_config(CONFIG); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); problems = []
    if (out / 'plan.json').exists(): raise SystemExit(f'Refused overwrite: {out / "plan.json"}')
    try: seeds = O.pilot_seeds(read_json(MANIFEST), used_ledger())
    except ValueError as e: problems.append(str(e)); seeds = []
    policy = D.check_policy(problems); scan = recorded_use_scan()
    if scan['files_naming_a_pilot_seed'] or scan['pilot_ledger_exists']: problems.append('a record or ledger already names a pilot seed')
    diag = read_json(DIAG_RECORD['ledger']); dres = read_json(DIAG_RECORD['result'])
    if diag.get('set') != 'recovery_v1_diag' or (dres['n'], dres['success_once']) != (100, 79): problems.append('the diagnostic record is not the executed 79/100 pass on recovery_v1_diag')
    if [c['policy'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')] != [2, 8, 16] or c['env']['horizon'] != 200 or c['env']['control_mode'] != 'pd_ee_delta_pos': problems.append('config is not the fixed 2/8/16, horizon 200, pd_ee_delta_pos')
    r = c['retreat']
    if (r['mode'], r['distance_m'], r['max_translation_per_step_m'], r['max_steps'], r['arrival_tolerance_m']) != ('up', 0.05, 0.005, 40, 0.005): problems.append('the R5 retreat configuration changed')
    aud = read_json(D.SEED_SOURCES['vision_v1'])['splits']['vision_v1_select']
    if not set(AUDIT_SEEDS) <= set(aud): problems.append('audit seeds are not in vision_v1_select')
    offline = read_json(OFFLINE) if Path(OFFLINE).exists() else None
    plan = {'work_id': WORK, 'stage': 'Recovery-V1: oracle XY centre-before-release pilot (Phase A: preparation)', 'created_utc': datetime.now(timezone.utc).isoformat(), 'pass': not problems, 'problems': problems,
            'status': STATUS,
            'authorization': {'paid_compute_authorized': False, 'proposed_cap_usd': 2.50, 'proposed_cap_is': 'a proposal, explicitly not an approval; earlier approvals are not inherited; a bare "proceed" is not an approval',
                              'needed': 'a separate explicit user instruction that states an exact USD cap AND names the SHA256 of this plan file',
                              'required_file': 'authorization.json in the run folder, written only after such an instruction: {"approved_by": "user", "paid_compute_authorized": true, "cap_usd": <number>, "plan_sha256": "<sha256 of this plan>"}'},
            'question': "At the first pending release command of the frozen Vision-V1 policy, does an oracle-guided bounded horizontal correction improve final task success without turning many C0-successful scenes into failures, beyond the effect of interruption, holding and re-planning?",
            'privileged_information': {'exception': 'exact simulator cube positions are read for control in ONE place (hci.recovery_v1_oracle.oracle_offset), only while the CENTER_ORACLE control part runs (OracleGate), every read counted per arm',
                                       'never': 'the policy observation whitelist (rgb, qpos, qvel, tcp pose), the trigger, C0, R5 and HOLD_MATCHED receive no object pose', 'logging': 'ground truth is also written to traces after each step, as in the diagnostic; logging only',
                                       'project_rule': 'AGENTS.md: "모니터/정책에 객체 정답 pose, task success, 미래 라벨을 입력하지 않습니다." The oracle value goes to a scripted reference controller, not to the monitor or the policy; AGENTS.md is not edited',
                                       'amendment': 'amendments/AMENDMENT_014_recovery_v1_oracle_xy_preparation_DRAFT.md', 'claim': 'CENTER_ORACLE is not deployable; a gain is feasibility evidence only'},
            'repository': {'head': D.git('rev-parse', 'HEAD'), 'working_tree_changes_at_plan_time': D.git('status', '--porcelain').splitlines()},
            'base_policy': {**POLICY, 'checks': policy, 'rule': 'frozen; not retrained, not fine-tuned, no other checkpoint'},
            'config': {'path': CONFIG, 'file_sha256': sha256(CONFIG), 'config_sha256': digest(c), 'policy': c['policy'], 'env': c['env'], 'retreat': c['retreat'], 'audit': c['audit']},
            'seed_registry': {'path': MANIFEST, 'sha256': sha256(MANIFEST), 'set': O.SET, 'range': list(O.RANGE), 'n': len(seeds), 'first_last': seeds[:1] + seeds[-1:],
                              'seed_level_check': 'no overlap with splits.json, policy_eval_splits.json, vision_v1_splits.json, recovery_v1_diag, the 550 demonstration scene seeds, or the scenes of Stage 2A / 2B / 2D / H1',
                              'recorded_use_scan': scan, 'consumed': {'vision_v1_qualify': '50500-50599 (ledger artifacts/ledger/vision_v1_qualify.json)', 'recovery_v1_diag': f"50600-50699 (ledger {DIAG_RECORD['ledger']}, used {diag.get('used_utc')})"},
                              'placement_check': 'server step `scenes`: placements of the 100 pilot seeds against stored placements (demonstrations, earlier pilots) and, by reset, against audit / dev / policy-evaluation / Vision-V1 / diagnostic sets; calibration and Test by seed number only. An identical placement fails the step',
                              'untouched_sets': ['calibration', 'test'], 'no_replacement': 'no scene is replaced or dropped for any reason'},
            'arms': list(O.ARMS), 'semantics': SEMANTICS, 'analysis': ANALYSIS,
            'audit': {'seeds': list(AUDIT_SEEDS), 'from': 'vision_v1_select (already used); outcomes on these scenes decide nothing',
                      'does': ['controller configuration: arm root-frame delta position, 0.1 m per unit, no virtual target; gripper normalised onto [-0.01, 0.04] m', 'the full four-arm procedure on each audit scene with every technical check of the pilot',
                               'tracking: from the trigger state, the bounded move towards four synthetic TCP targets (+-3 cm in x and y; no object pose): achieved displacement, per-step gain, direction, vertical drift, grasp kept',
                               'trunk outcome and step count against the Vision-V1 record (reported, does not block)', 'seconds per environment step for the budget gate'],
                      'pass_criteria': O.AUDIT_PASS, 'probe_targets_m': [list(x) for x in O.PROBE_TARGETS_M], 'on_failure': 'TECHNICAL_BLOCKED: the pilot set is not touched; no bound is changed'},
            'run': {'scenes': 100, 'repeat': O.REPEAT, 'ledger': LEDGER + ', created exclusively before any rollout; no resume, no second pass', 'on_error': 'the pass stops and stays INCOMPLETE; no scene is dropped or replaced; a partial run has no summary',
                    'after_two_scenes': 'a time-only re-projection of the total cost; above the cap the pass stops and is backed up (outcomes are not read for this)', 'stored': 'one record per scene, per-step ground-truth traces of the trunk, the reference and the four arms; no camera frames'},
            'offline_evidence': offline, 'upstream_sources': read_json(UPSTREAM_SOURCES) if Path(UPSTREAM_SOURCES).exists() else None,
            'volume': {'placement_resets': 100 + 5 + 100 + 270 + 200 + 100, 'audit_scenes': len(AUDIT_SEEDS), 'pilot_scenes': 100, 'steps_per_scene_bound': O.STEPS_PER_SCENE_BOUND},
            'cost_estimate': {'basis': 'diagnostic pass on an RTX 4090 at US$0.41/h: 0.077 s per control step with inference and recorder; diagnostic traces: mean episode 125 steps, mean trigger step 103',
                              'expected_env_steps_per_triggered_scene': 'about 1,000 (trunk 125, five prefix replays of about 103, four continuations, reference)', 'pilot_minutes': [100, 130], 'pilot_minutes_worst_case': 195,
                              'install_checks_audit_transfer_minutes': 45, 'estimated_total_usd': [1.10, 1.50], 'worst_case_gate_projection_usd_at_0_41_per_hour': 1.92, 'gate_refuses_above_usd_per_hour_about': 0.57, 'proposed_cap_usd': 2.50,
                              'gate': 'the server `budget` step projects the total from the measured seconds per environment step with every scene at the 1,212-step bound, margin 1.25, reserve US$0.20, and refuses above the approved cap'},
            'host': {'gpu': 'RTX 4090 class, 24 GB', 'create_options': "--env '-e NVIDIA_DRIVER_CAPABILITIES=all -e WANDB_MODE=disabled -e MS_SKIP_ASSET_DOWNLOAD_PROMPT=1'", 'first_check': 'NVIDIA graphics libraries present (Vulkan) before installing', 'transfer_price': 'offers filtered by per-GB transfer price'},
            'command_order': ['local: tests, `plan`, `dry-run`', 'user: explicit approval naming a USD cap and this plan hash -> authorization.json', 'server: install (00_system_packages.sh, 01_bootstrap_vast.sh)',
                              'server: run.sh env (doctor, versions, full test suite, dry-run against the plan)', 'server: run.sh scenes', 'server: run.sh audit', 'server: run.sh budget', 'server: run.sh pilot', 'server: run.sh pack (archive, per-file SHA256)',
                              'local: download, archive SHA256 and per-file SHA256 check, `summarize` from the raw records', 'local: delete only the instances created for this run, verify deletion', f'Drive: report and small results to {DRIVE}'],
            'outputs': {'local': 'results/recovery_v1_oracle_xy_pilot_001/', 'drive': DRIVE, 'report': 'RECOVERY_V1_ORACLE_XY_PILOT_REPORT.md', 'amendment': 'AMENDMENT_014 (execution)', 'report_md': 'not modified unless asked'},
            'files_read_on_the_server': SERVER_READS, 'code': {f: sha256(f) for f in CODE}, 'upstream_commit': UPSTREAM_COMMIT,
            'not_in_this_work': ['policy training or fine-tuning', 'an image-based estimator', 'any change of bounds, tolerance, steps or trigger after an outcome', 'a second pass', 'calibration', 'final Test']}
    write_json(out / 'plan.json', plan); print(json.dumps({'pass': plan['pass'], 'problems': problems, 'recorded_use_scan': scan, 'base_policy_checks': policy}, indent=1))
    if problems: raise SystemExit(2)


def cmd_dry_run(a):
    plan = read_json(a.plan); run = Path(a.run); need = plan['files_read_on_the_server']
    missing = [f for f in need if not Path(f).exists()]; changed = [f for f, h in plan['code'].items() if Path(f).exists() and sha256(f) != h]; auth = (run / 'authorization.json').exists()
    if plan.get('pass') is not True or plan.get('problems'): raise SystemExit('Refused: the plan did not pass its own preparation checks')
    out = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha256(a.plan), 'volume': plan['volume'], 'files_needed_on_the_server': need,
           'upload_bytes': int(sum(Path(f).stat().st_size for f in need if Path(f).exists())), 'files_missing': missing, 'code_files_changed_since_plan': changed, 'authorization_present': auth,
           'ledger_exists': Path(LEDGER).exists(), 'paid_compute': 'AUTHORIZATION_FILE_PRESENT_NOT_USED_BY_DRY_RUN' if auth else 'NOT_AUTHORIZED', 'servers_created': 0, 'simulator_steps': 0, 'policy_inferences': 0}
    print(json.dumps({k: v for k, v in out.items() if k != 'files_needed_on_the_server'}, indent=1))
    if a.out: write_json(a.out, out)
    if missing or changed: raise SystemExit(2)


def bound(plan_path):
    plan = read_json(plan_path)
    if plan.get('pass') is not True or plan.get('problems') or plan.get('work_id') != WORK: raise SystemExit('Refused: the plan did not pass its own preparation checks or belongs to another work')
    bad = [f for f, h in plan['code'].items() if sha256(f) != h]
    for f in ('weights.pt', 'policy_meta.json'):
        if sha256(Path(POLICY['path']) / f) != POLICY[f]: bad.append('policy ' + f)
    if bad: raise SystemExit(f'Refused: these differ from the plan: {bad}')
    return plan, load_config(CONFIG)


def cmd_scenes(a):
    run = Path(a.run); D.require_authorization(run, a.plan); plan, c = bound(a.plan); man = read_json(MANIFEST); seeds = O.pilot_seeds(man, used_ledger())
    from hci.simulation import make_env
    from hci.trees import as_numpy
    env = make_env(c)

    def key(seed):
        env.reset(seed=int(seed)); return np.array([*as_numpy(env.cubeA.pose.p)[0], *as_numpy(env.cubeB.pose.p)[0]], float)
    try:
        refs = {n: np.array(k) for n, k in read_json(KEYS)['keys'].items()}; stored = len(refs); sp = read_json(D.SEED_SOURCES['splits'])['splits']
        reset_sets = {'splits/audit': sp['audit'], 'splits/dev': sp['dev'], **{f'policy_eval/{k}': v for k, v in read_json(D.SEED_SOURCES['policy_eval'])['splits'].items()},
                      **{f'vision_v1/{k}': v for k, v in read_json(D.SEED_SOURCES['vision_v1'])['splits'].items()}, 'recovery_v1/recovery_v1_diag': man['splits']['recovery_v1_diag']}
        for name, ids in reset_sets.items():
            for s in ids: refs[f'{name}/{s}'] = key(s)
        names = list(refs); table = np.stack([refs[n] for n in names]); own = {}; clashes = []
        for s in seeds:
            k = key(s); d = np.abs(table - k).max(1); j = int(d.argmin()); twin = [t for t, v in own.items() if np.abs(v - k).max() <= IDENTICAL_M]
            if d[j] <= IDENTICAL_M or twin: clashes.append({'seed': s, 'identical_to': names[j] if d[j] <= IDENTICAL_M else twin[0]})
            own[str(s)] = k
        report = {'pass': not clashes, 'identical_placements': clashes, 'seeds_checked': len(own), 'identical_threshold_m': IDENTICAL_M,
                  'references': {'stored_keys_demonstrations_and_pilots': stored, **{k: len(v) for k, v in reset_sets.items()}}, 'seed_level_only': ['splits/calibration', 'splits/test'],
                  'plan_sha256': sha256(a.plan), 'scope': 'Initial cube placement only; no policy was run on these scenes.'}
        write_json(run / 'scenes.json', report); print(json.dumps({k: report[k] for k in ('pass', 'identical_placements', 'seeds_checked')}))
        if clashes: raise SystemExit(2)
    finally:
        env.close()


def gt_probe(rt, z):
    from hci import recovery_v1 as RV
    return RV.probe(rt.env, rt.t, z)


def save_traces(out, seed, traces):
    tp = out / 'traces' / f'trace_{seed}.json.gz'; tp.parent.mkdir(parents=True, exist_ok=True)
    if tp.exists(): raise FileExistsError(tp)
    with gzip.open(tp, 'xt', encoding='utf-8') as f: json.dump({'seed': int(seed), 'traces': traces}, f)
    return tp


def controller_facts(r):
    """Configuration of the live controllers, read on the server; the plan's units rest on these."""
    cs = r.env.agent.controller.controllers; arm = cs['arm'].config; g = cs['gripper'].config; scale = O.arm_scale(r)
    facts = {'control_mode': r.c['env']['control_mode'], 'arm_class': type(cs['arm']).__name__, 'arm_frame': arm.frame, 'arm_use_delta': bool(arm.use_delta), 'arm_use_target': bool(arm.use_target),
             'arm_normalize_action': bool(arm.normalize_action), 'arm_metres_per_unit': [float(x) for x in scale], 'gripper_class': type(cs['gripper']).__name__, 'gripper_lower': float(np.min(g.lower)),
             'gripper_upper': float(np.max(g.upper)), 'gripper_normalize_action': bool(g.normalize_action), 'gripper_use_delta': bool(g.use_delta), 'action_dim': int(r.adim)}
    ok = (facts['arm_metres_per_unit'] == [0.1, 0.1, 0.1] and facts['arm_use_delta'] and not facts['arm_use_target'] and facts['gripper_normalize_action'] and not facts['gripper_use_delta']
          and abs(facts['gripper_lower'] + 0.01) < 1e-9 and abs(facts['gripper_upper'] - 0.04) < 1e-9 and facts['action_dim'] == 4)
    return {**facts, 'pass': bool(ok), 'meaning': 'gripper command -1 = 2 x -0.01 m (closed), +1 = 2 x 0.04 m (open); 0.05 in the arm action = 5 mm commanded'}


def cmd_audit(a):
    run = Path(a.run); D.require_authorization(run, a.plan); plan, c = bound(a.plan); out = run / 'audit'; out.mkdir(parents=True, exist_ok=False)
    ref = {x['seed']: x for x in map(json.loads, Path(AUDIT_REFERENCE).read_text().splitlines()) if x['seed'] in AUDIT_SEEDS}
    from hci import vision_v1 as V
    r = V.runtime(c, POLICY['path'], a.upstream, a.device); scenes = []; probes = []; facts = None
    try:
        facts = controller_facts(r)
        for s in AUDIT_SEEDS:
            rec, traces = O.collect_scene(r, s, c, probe=gt_probe); tp = save_traces(out, s, traces); rec['trace'] = tp.relative_to(out).as_posix(); rec['trace_sha256'] = sha256(tp)
            write_json(out / 'scenes' / f'{s}.json', rec); scenes.append(rec)
            print(json.dumps({'audit_scene': s, 'triggered': rec['triggered'], 'trigger_step': (rec['trigger'] or {}).get('step'), 't_center': (rec['reference'] or {}).get('t_center'), 'wall_seconds': rec['wall_seconds']}), flush=True)
            if not rec['triggered']: continue
            r.reset(s); r.repeat = O.REPEAT
            while not r.done and not O.trigger_reading(r)['fires']: r.step_base()
            if r.done or r.t != rec['trigger']['step']: raise RuntimeError(f'audit scene {s}: the trigger state was not reached again')
            snap = r.snapshot()
            for target in O.PROBE_TARGETS_M:
                r.restore(snap); x = O.tracking_probe(r, target, gt_probe); probes.append({'scene': int(s), 'trigger_step': rec['trigger']['step'], **x})
                print(json.dumps({'probe': s, 'target': target, 'achieved_m': x['achieved_norm_m'], 'steps': x['steps_used'], 'gain': x['gain_median'], 'dz': x['tcp_dz_m']}), flush=True)
    finally:
        r.env.close()
    verdict = O.tracking_verdict(probes, sum(1 for x in scenes if x['triggered'])); steps = sum(x['counts']['env_steps'] for x in scenes); wall = sum(x['wall_seconds'] for x in scenes)
    same = {str(x['scene']): [x['trunk']['success'], x['trunk']['steps']] == [ref[x['scene']]['success_once'], ref[x['scene']]['steps']] for x in scenes}
    report = {'pass': bool(verdict['pass'] and facts['pass']), 'status': 'OK' if (verdict['pass'] and facts['pass']) else 'TECHNICAL_BLOCKED', 'controller': facts, 'tracking': verdict, 'probes': probes,
              'scenes': [{'scene': x['scene'], 'triggered': x['triggered'], 'trigger': x['trigger'], 'reference': x['reference'], 'handoff_pass': (x['handoff'] or {}).get('pass'), 'env_steps': x['counts']['env_steps'], 'wall_seconds': x['wall_seconds']} for x in scenes],
              'four_arm_procedure_completed_on': len(scenes), 'seconds_per_env_step': wall / steps, 'reproduces_vision_v1_record': same, 'all_reproduced': all(same.values()), 'plan_sha256': sha256(a.plan), 'runtime': runtime_info(),
              'scope': 'Technical audit on already-used development scenes. Outcomes here decide nothing and are not pilot data.'}
    write_json(out / 'audit.json', report); print(json.dumps({k: report[k] for k in ('pass', 'status', 'seconds_per_env_step', 'all_reproduced')} | {'tracking_problems': verdict['problems'], 'controller_pass': facts['pass']}, indent=1))
    if not report['pass']: raise SystemExit(2)


def cmd_budget(a):
    run = Path(a.run); auth = D.require_authorization(run, a.plan); p = read_json(run / 'audit' / 'audit.json')
    if not p['pass'] or p['plan_sha256'] != sha256(a.plan): raise SystemExit('audit missing, failed or of another plan')
    d = O.projected_cost(seconds_per_env_step=p['seconds_per_env_step'], usd_per_hour=a.usd_per_hour, spent_usd=a.spent_usd, cap_usd=float(auth['cap_usd']))
    d.update(status='OK' if d['fits_cap'] else 'BUDGET_BLOCKED', plan_sha256=sha256(a.plan)); write_json(run / 'budget_gate.json', d); print(json.dumps(d, indent=1))
    if not d['fits_cap']: raise SystemExit(2)


def cmd_run(a):
    run = Path(a.run); auth = D.require_authorization(run, a.plan); plan, c = bound(a.plan); h = sha256(a.plan)
    for name in ('scenes.json', 'audit/audit.json', 'budget_gate.json'):
        if not (run / name).exists(): raise SystemExit(f'{name} missing: placement check, audit and budget gate come first')
    sc = read_json(run / 'scenes.json'); au = read_json(run / 'audit' / 'audit.json'); bg = read_json(run / 'budget_gate.json')
    if not (sc['pass'] and sc['plan_sha256'] == h and au['pass'] and au['plan_sha256'] == h and bg['fits_cap'] and bg.get('plan_sha256') == h): raise SystemExit('placement check, audit or budget gate not passed for this plan')
    seeds = O.pilot_seeds(read_json(MANIFEST), used_ledger()); out = run / 'pilot'
    if out.exists(): raise SystemExit(f'{out} exists: the pilot pass is never resumed or repeated')
    # the gate is evaluated again with the price and the spend stated for THIS command; an earlier gate with other figures is not enough
    now = O.projected_cost(seconds_per_env_step=au['seconds_per_env_step'], usd_per_hour=a.usd_per_hour, spent_usd=a.spent_usd, cap_usd=float(auth['cap_usd']))
    now.update(status='OK' if now['fits_cap'] else 'BUDGET_BLOCKED', plan_sha256=h, earlier_gate_inputs=bg['inputs'])
    if not now['fits_cap'] or a.usd_per_hour < bg['inputs']['usd_per_hour'] or a.spent_usd < bg['inputs']['spent_usd']:      # refused before anything is written: the set stays unused and the command can be given again
        print(json.dumps(now, indent=1)); raise SystemExit('BUDGET_BLOCKED at the start of the pilot: the projection with the current figures exceeds the cap, or the figures are lower than those of the budget step')
    write_json(run / 'budget_gate_at_start.json', now)
    # exclusive create BEFORE any rollout: from here on the set is used, whatever happens next
    write_json(LEDGER, {'set': O.SET, 'used_utc': datetime.now(timezone.utc).isoformat(), 'session': out.as_posix(), 'plan_sha256': h, 'seed_manifest_sha256': plan['seed_registry']['sha256'], 'policy': POLICY['path'],
                        'weights_sha256': POLICY['weights.pt'], 'arms': list(O.ARMS), 'note': 'Single four-arm pilot pass. Written before any outcome existed. No resume, no second pass.'})
    (out / 'scenes').mkdir(parents=True); started = time.perf_counter()
    from hci import vision_v1 as V
    r = V.runtime(c, POLICY['path'], a.upstream, a.device); done = 0
    try:
        for s in seeds:
            try:
                rec, traces = O.collect_scene(r, s, c, probe=gt_probe); tp = save_traces(out, s, traces); rec['trace'] = tp.relative_to(out).as_posix(); rec['trace_sha256'] = sha256(tp); write_json(out / 'scenes' / f'{s}.json', rec)
            except Exception as e:
                import traceback
                write_json(out / 'scenes' / f'{s}.error.json', {'scene': int(s), 'completed': False, 'error': repr(e), 'traceback': traceback.format_exc(), 'scenes_finished_before': done}); raise
            done += 1; print(f"pilot scene={s} triggered={rec['triggered']} done={done}/100 wall={rec['wall_seconds']:.0f}s", flush=True)      # outcomes are not printed during the pass
            if done == 2:
                el = time.perf_counter() - started; total = a.spent_usd + (el + 98 * el / 2 * 1.25) / 3600 * a.usd_per_hour + 0.20
                gate = {'elapsed_seconds_two_scenes': el, 'projected_total_usd': total, 'cap_usd': float(auth['cap_usd']), 'fits_cap': bool(total <= float(auth['cap_usd'])), 'inputs': 'time and money only; no outcome is read', 'plan_sha256': h}
                write_json(out / 'budget_after_two_scenes.json', gate)
                if not gate['fits_cap']: raise SystemExit('BUDGET_BLOCKED after two scenes: the pass stops INCOMPLETE and is backed up; the experiment is not reduced')
    finally:
        r.env.close()
    cmd_summarize(argparse.Namespace(pilot=str(out), plan=a.plan, out=str(out / 'result.json'), audit=str(run / 'audit' / 'audit.json'), wall=time.perf_counter() - started))


def cmd_summarize(a):
    """Every table from the raw scene records; a missing, failed or invalid scene means no summary."""
    d = Path(a.pilot); plan = read_json(a.plan); c = load_config(CONFIG); seeds = O.pilot_seeds(read_json(MANIFEST), {}); recs = []
    errors = sorted(p.name for p in (d / 'scenes').glob('*.error.json'))
    if errors: raise SystemExit(f'INCOMPLETE: error files present {errors}; a partial run is not summarised')
    for s in seeds:
        p = d / 'scenes' / f'{s}.json'
        if not p.exists(): raise SystemExit(f'INCOMPLETE: no record for scene {s}; a partial run is not summarised')
        rec = read_json(p); O.validate_scene(rec, c)
        if sha256(d / rec['trace']) != rec['trace_sha256']: raise SystemExit(f'trace of scene {s} changed')
        recs.append(rec)
    audit = read_json(a.audit); res = O.summarize(recs, seeds, bool(audit['pass']))
    res.update(set=O.SET, seeds=[seeds[0], seeds[-1]], policy=POLICY['path'], weights_sha256=POLICY['weights.pt'], plan_sha256=sha256(a.plan), rules_sha256=sha256('src/hci/recovery_v1_oracle.py'),
               scene_records_sha256=digest({str(r['scene']): sha256(d / 'scenes' / f"{r['scene']}.json") for r in recs}), env_steps=sum(r['counts']['env_steps'] for r in recs),
               oracle_reads_outside_center=sum(x['oracle_reads'] for r in recs for x in r['cells'] if x['arm'] != 'CENTER_ORACLE' and not x['no_trigger']), wall_seconds=getattr(a, 'wall', None), runtime=runtime_info())
    write_json(a.out, res)
    print(json.dumps({'denominators': res['denominators'], 'success': {k: v['k'] for k, v in res['success'].items()}, 'primary': {k: res['primary_CENTER_ORACLE_vs_C0'][k] for k in ('both_succeed', 'rescue', 'harm', 'both_fail', 'net')},
                      'gate': res['followup_gate']['label']}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--out', required=True)
    q = s.add_parser('dry-run'); q.add_argument('--plan', required=True); q.add_argument('--run', required=True); q.add_argument('--out')
    for name in ('scenes', 'audit', 'budget', 'run'):
        q = s.add_parser(name); q.add_argument('--plan', required=True); q.add_argument('--run', required=True)
        if name in ('audit', 'run'): q.add_argument('--upstream', default='third_party/ManiSkill'); q.add_argument('--device', default='cuda')
        if name in ('budget', 'run'): q.add_argument('--usd-per-hour', type=float, required=True); q.add_argument('--spent-usd', type=float, required=True)
    q = s.add_parser('summarize'); q.add_argument('--pilot', required=True); q.add_argument('--plan', required=True); q.add_argument('--audit', required=True); q.add_argument('--out', required=True)
    a = p.parse_args(); globals()['cmd_' + a.cmd.replace('-', '_')](a)


if __name__ == '__main__':
    main()
