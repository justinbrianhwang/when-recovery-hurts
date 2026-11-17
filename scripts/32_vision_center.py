"""Vision-CENTER (work id vision_center_001): the bounded centring move of the Recovery-V1 oracle pilot driven by RGB and robot
state instead of simulator object positions. PREPARATION ONLY at the time of writing: nothing below `dry-run` has been run.
No training, no calibration, no Test.

  plan        local: frozen policy, code hashes, seed registry, frozen constants, criteria, cost estimate
  dry-run     local: files and volume; starts nothing
  scenes      server: cube placements of the 100 new pilot seeds against every placement on record (no policy)
  audit       server: 3 used selection scenes, all three arms: technical checks and timing for the budget gates
  budget      server: worst-case projection of the stage named by --stage (shadow | pilot, each including what follows it)
  shadow      server, stage B1: the 200 USED scenes 50600-50799, arms C0 and VISION_CENTER; hand-camera frames at the trigger
              and during control are stored; fixed pass criteria decide whether the new scenes are run
  run         server, stage B2: the one pilot pass on the 100 new scenes, arms C0, VISION_CENTER, CENTER_ORACLE. Ledger first. No resume.
  summarize   anywhere: every table recomputed from the stored scene records
Server commands refuse to run unless <run>/authorization.json exists, names the plan by hash and carries a cap.
No command in this file writes that authorisation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, gzip, importlib.util, json, time
import numpy as np
from hci.io import load_config, read_json, write_json, sha256, digest, runtime_info, UPSTREAM_COMMIT
from hci import vision_center as VC
from hci import recovery_v1_oracle as O

_spec = importlib.util.spec_from_file_location('oxy_driver', Path(__file__).with_name('31_recovery_v1_oracle_xy.py')); X = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(X)
D = X.D
WORK = VC.WORK
POLICY = D.POLICY; CONFIG = D.CONFIG; KEYS = D.KEYS
MANIFEST = 'configs/vision_center_splits.json'
USED_MANIFEST = D.MANIFEST                                  # recovery_v1_diag and recovery_v1_pilot_reserved: both used
AUDIT_SEEDS = (50400, 50401, 50402)                         # vision_v1_select, already used
ORACLE_RECORD = {'ledger': 'results/recovery_v1_oracle_xy_pilot_001/run1/ledger_recovery_v1_oracle_xy_pilot.json', 'result': 'results/recovery_v1_oracle_xy_pilot_001/run1/result.json'}
OFFLINE = 'results/vision_center_001/preparation/offline_evaluation.json'
CODE = D.INFERENCE_CODE + ['src/hci/recovery_v1.py', 'src/hci/recovery_v1_oracle.py', 'src/hci/vision_center.py', 'src/hci/pilot2b.py', 'src/hci/pilot2d.py', 'scripts/30_recovery_v1_diag.py',
                           'scripts/31_recovery_v1_oracle_xy.py', 'scripts/32_vision_center.py', 'scripts/vision_center/run.sh', 'scripts/stage1b/common.sh', CONFIG, MANIFEST, USED_MANIFEST]
SERVER_READS = sorted({*CODE, *D.SEED_SOURCES.values(), D.DEMO_MANIFEST, *D.PRIOR_SCENES, KEYS, POLICY['path'] + '/weights.pt', POLICY['path'] + '/policy_meta.json'})
LEDGER = 'artifacts/ledger/vision_center_pilot.json'
DRIVE = 'Robot Harm/results/vision_center_001/'
STATUS = {'vision_center': 'PHASE_A_PREPARATION_ONLY', 'paid_compute': 'NOT_AUTHORIZED', 'shadow_pass_on_used_scenes': 'NOT_RUN', 'pilot_on_new_scenes': 'NOT_RUN', 'base_policy': 'FROZEN: Vision-V1 train seed 1, 80k EMA, unchanged',
          'recovery_v1_oracle_xy_pilot': 'EXECUTED_ONCE (ORACLE_GATE_PASS), unchanged', 'new_training': 'NOT_PLANNED (no network is trained; perception is colour segmentation and projection)', 'calibration': 'NOT_RUN', 'final_test': 'NOT_RUN'}
SEMANTICS = {
    'inputs': 'hand_camera RGB (128 x 128) and extra/tcp_pose of the policy observation; fixed camera constants; for the trigger: pending and executed actions, agent/qpos. No cube pose, grasp flag, task flag, outcome or future information',
    'trigger': {'rule': 'the oracle pilot trigger, unchanged (hci.recovery_v1_oracle.trigger_rule)', **O.TRIGGER},
    'perception': {'target': 'lit green pixels of the hand image are explained by a 4 cm square on the plane z = 0.04 m (the top face of the target cube); red and grey pixels are occluders and are not scored; '
                             'accepted when the score is at least score_min and the best placement at least far_m away scores lower by margin_min', 'target_constants': {k: v for k, v in VC.PARTIAL_FIT.items()},
                   'held_cube': 'lit red pixels are explained by a 4 cm square at the depth of the held cube face (hold offset x, y, z and yaw in the gripper frame); the three best depths are re-scored with the outline of the whole cube against all red pixels; '
                                'accepted when both scores and the lead over a different solution pass', 'held_cube_constants': {k: v for k, v in VC.HOLD_FIT.items()},
                   'offset': '(TCP position + rotated hold offset) minus target, horizontal part', 'memory': 'none: only the current image is used at the trigger; during control the target estimate of the trigger is kept while the target is hidden',
                   'base_camera': 'not used for control (about 1 cm per pixel at the cubes; errors of several centimetres pass every single-image check)'},
    'abstain': {'meaning': 'nothing is touched: the pending queue, the history and the counters stay as they are, so the episode is the uninterrupted one (checked against the trunk in every scene)',
                'reasons': {'NO_HOLD_ESTIMATE': 'held-cube fit missing or not accepted', 'NO_TARGET_ESTIMATE': 'target not accepted in the current image (usually hidden by the held cube)',
                            'LOW_CARRY': f'estimated height of the held cube centre below {VC.CARRY_Z_MIN_M} m: the cube is beside the target, a horizontal move would push it', 'TOO_FEW_STEPS': f'fewer than {VC.STEPS_LEFT_MIN} steps left',
                            'ALIGNED': f'estimated offset below {VC.ACT_MIN_M} m'}, 'order': 'as listed; the first that applies'},
    'VISION_CENTER': {'acts_when': f'estimated offset >= {VC.ACT_MIN_M} m and no abstain reason', 'command': 'as CENTER_ORACLE: horizontal only, at most 0.005 m per step, commanded target within 0.030 m of the TCP at the trigger, at most 12 steps, vertical command 0, held gripper command',
                      'stops_when': f'estimated offset <= {VC.RESIDUAL_TOL_M} m, or 12 steps, or the target cap, or the episode ends, or the held-cube fit is lost, or a new accepted target estimate is more than {VC.TARGET_JUMP_MAX_M} m from the one at the trigger, or the held cube is estimated too low',
                      'after': 'queue cleared, the unchanged policy re-infers', **VC.DECISION},
    'CENTER_ORACLE': 'hci.recovery_v1_oracle.center_oracle, unchanged (reads simulator cube positions; diagnostic reference, not deployable; allowed by the user for this arm only)',
    'C0': 'the pending action and the rest of the queue are executed; must equal the uninterrupted run, otherwise the run stops',
    'noise': "repeat 0 only; every inference is keyed by seed_for(scene, physical step, 0, 'dp')", 'restore': 'reset(seed) + prefix action replay; the restored runtime is compared with the snapshot before every branch',
    'chosen_on': 'ACT threshold, carry limit, step limit and all perception constants were chosen on used data: the 200 used scenes (traces), the 21 failed diagnostic episodes with frames, and the training demonstrations'}
ANALYSIS = {'unit': 'scene', 'stage_B1_shadow': {'scenes': '200 used scenes (50600-50799), re-run on the server because no frames of them are stored', 'arms': list(VC.SHADOW_ARMS), 'pass_criteria': VC.SHADOW_PASS,
                                                'reports': ['estimation error at the trigger against logged ground truth', 'decisions by reason', 'unnecessary acts (acted, true offset <= 1.0 cm)', 'missed large offsets (abstained, true offset >= 2.0 cm)',
                                                            'paired outcome against C0', 'end reasons'], 'on_failure': 'SHADOW_BLOCKED: the new scenes are not touched; no constant is tuned and re-tried under this plan; the observation limits are reported'},
            'stage_B2_pilot': {'scenes': '100 new scenes (50800-50899)', 'arms': list(VC.ARMS), 'primary': 'paired success_once_absorbing, VISION_CENTER against C0, all 100 scenes: rescue / harm / net, Wilson intervals',
                               'reference': 'CENTER_ORACLE against C0 on the same scenes; oracle rescues also rescued by vision', 'gate': {'all_of': ['VISION_CENTER rescues >= 4', 'net >= +3', 'harm <= 3'], **VC.GATE, 'otherwise': 'NO_VISION_GATE_PASS: no extension, no tuning, no subset search'}},
            'not_claimed': ['statistical significance', 'transfer to real images (flat simulator colours, exact camera parameters, no noise)', 'anything beyond StackCube and this policy']}


def used_ledger():
    used = X.used_ledger(); used['recovery_v1/recovery_v1_pilot_reserved'] = list(read_json(USED_MANIFEST)['splits']['recovery_v1_pilot_reserved']); return used


def shadow_seeds():
    sp = read_json(USED_MANIFEST)['splits']; ids = []
    for name, (lo, hi) in VC.SHADOW_SETS:
        if list(sp[name]) != list(range(lo, hi + 1)): raise ValueError(f'{name} is not {lo}-{hi}')
        ids += list(sp[name])
    return ids


def cmd_plan(a):
    c = load_config(CONFIG); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); problems = []
    if (out / 'plan.json').exists(): raise SystemExit(f'Refused overwrite: {out / "plan.json"}')
    try: seeds = VC.pilot_seeds(read_json(MANIFEST), used_ledger())
    except ValueError as e: problems.append(str(e)); seeds = []
    policy = D.check_policy(problems); orc = read_json(ORACLE_RECORD['ledger']); ores = read_json(ORACLE_RECORD['result'])
    if orc.get('set') != 'recovery_v1_pilot_reserved' or ores['followup_gate']['label'] != 'ORACLE_GATE_PASS': problems.append('the oracle pilot record is not the executed ORACLE_GATE_PASS run')
    if Path(LEDGER).exists(): problems.append('a ledger for the new pilot set already exists')
    if [c['policy'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')] != [2, 8, 16] or c['env']['horizon'] != 200 or c['env']['control_mode'] != 'pd_ee_delta_pos' or c['env']['robot_uid'] != 'panda_wristcam': problems.append('config is not the fixed setting')
    if not set(AUDIT_SEEDS) <= set(read_json(D.SEED_SOURCES['vision_v1'])['splits']['vision_v1_select']): problems.append('audit seeds are not in vision_v1_select')
    offline = read_json(OFFLINE) if Path(OFFLINE).exists() else None
    if offline is None: problems.append('offline evaluation missing')
    short = None if offline is None else {k: ({kk: vv for kk, vv in v.items() if kk not in ('rows', 'hold_fit_other_frames')} if isinstance(v, dict) else v) for k, v in offline.items()}
    plan = {'work_id': WORK, 'stage': 'Vision-CENTER (Phase A: preparation)', 'created_utc': datetime.now(timezone.utc).isoformat(), 'pass': not problems, 'problems': problems, 'status': STATUS,
            'authorization': {'paid_compute_authorized': False, 'proposed_cap_usd': 3.00, 'proposed_cap_is': 'a proposal, explicitly not an approval; earlier approvals are not inherited',
                              'needed': 'a separate explicit user instruction that states an exact USD cap AND names the SHA256 of this plan file',
                              'required_file': 'authorization.json in the run folder, written only after such an instruction: {"approved_by": "user", "paid_compute_authorized": true, "cap_usd": <number>, "plan_sha256": "<sha256 of this plan>"}'},
            'question': 'Can the bounded centring move be driven by RGB and robot state alone, abstaining when the estimate is uncertain or no correction is needed, and how much of the oracle gain does it keep without harming scenes the policy already solves?',
            'privileged_information': {'vision_center': 'none', 'offline_labels': 'cube positions are used as labels in the offline evaluation and in after-the-fact error columns of the records', 'center_oracle': 'unchanged reference arm; the only control code that reads cube positions',
                                       'project_rule': 'AGENTS.md: "모니터/정책에 객체 정답 pose, task success, 미래 라벨을 입력하지 않습니다."'},
            'repository': {'head': D.git('rev-parse', 'HEAD'), 'working_tree_changes_at_plan_time': D.git('status', '--porcelain').splitlines()},
            'base_policy': {**POLICY, 'checks': policy, 'rule': 'frozen; not retrained, not fine-tuned, no other checkpoint'},
            'config': {'path': CONFIG, 'file_sha256': sha256(CONFIG), 'config_sha256': digest(c), 'env': c['env'], 'policy': c['policy'], 'audit': c['audit']},
            'seed_registry': {'path': MANIFEST, 'sha256': sha256(MANIFEST), 'set': VC.SET, 'range': list(VC.RANGE), 'n': len(seeds), 'first_last': seeds[:1] + seeds[-1:],
                              'seed_level_check': 'no overlap with splits.json (audit, train, dev, calibration, test), policy_eval_splits.json, vision_v1_splits.json, recovery_v1_splits.json (both sets), the 550 demonstration scene seeds, or the scenes of Stage 2A / 2B / 2D / H1',
                              'placement_check': 'server step `scenes`, as in the oracle pilot; an identical placement fails the step', 'shadow_scenes': {'sets': [list(x) for x in VC.SHADOW_SETS], 'n': len(shadow_seeds()), 'status': 'used; development data; no new scene is consumed by stage B1'},
                              'untouched_sets': ['calibration', 'test'], 'no_replacement': 'no scene is replaced or dropped for any reason'},
            'arms': list(VC.ARMS), 'semantics': SEMANTICS, 'analysis': ANALYSIS,
            'audit': {'seeds': list(AUDIT_SEEDS), 'does': ['all three arms with every technical check of the pilot', 'seconds per environment step and per vision reading for the budget gates'], 'outcomes': 'decide nothing'},
            'run': {'order': ['scenes', 'audit', 'budget --stage shadow', 'shadow', 'budget --stage pilot', 'run'], 'ledger': LEDGER + ', created exclusively before any rollout on a new scene; no resume, no second pass',
                    'on_error': 'the pass stops and stays INCOMPLETE; no scene is dropped or replaced; a partial run has no summary', 'stored': 'one record per scene, per-step ground-truth traces (logging), hand-camera frames at the trigger and during Vision-CENTER control'},
            'offline_evaluation': short, 'frame_inventory': None if offline is None else offline['inventory'],
            'volume': {'placement_resets': 100 + 5 + 100 + 270 + 200 + 200, 'audit_scenes': len(AUDIT_SEEDS), 'shadow_scenes': len(shadow_seeds()), 'pilot_scenes': 100, 'steps_per_scene_bound': {'shadow': 200 + 2 * 200, 'pilot': 200 + 3 * 200}, 'readings_per_scene_bound': VC.READINGS_PER_SCENE_BOUND},
            'cost_estimate': {'basis': 'oracle pilot on an RTX 4090 at US$0.44/h: 0.028 s per environment step; one vision reading about 1.8 s on the development machine (CPU, not measured on a server)',
                              'expected_minutes': {'install_checks_audit_transfer': 25, 'shadow_200_scenes': [45, 70], 'pilot_100_scenes': [30, 45]}, 'estimated_total_usd': [0.90, 1.40],
                              'worst_case_gate_projection_usd_at_0_44_per_hour': 2.51, 'worst_case_assumes': 'US$0.20 spent before the first gate, 0.028 s per step, 1.8 s per reading', 'proposed_cap_usd': 3.00,
                              'gate': 'each `budget` step projects the remaining stages from the measured seconds per step and per reading with every scene at its bound, margin 1.25, reserve US$0.20, and refuses above the approved cap'},
            'host': X.read_json(a.oracle_plan)['host'] if a.oracle_plan else None,
            'command_order': ['local: tests, `plan`, `dry-run`', 'user: explicit approval naming a USD cap and this plan hash -> authorization.json', 'server: install', 'server: run.sh env', 'server: run.sh scenes', 'server: run.sh audit',
                              'server: run.sh shadow (budget gate first; SHADOW_BLOCKED stops here)', 'server: run.sh pilot (budget gate first)', 'server: run.sh pack', 'local: download, hash checks, `summarize` from the raw records',
                              'local: delete only the instances created for this run', f'Drive: report and small results to {DRIVE}'],
            'outputs': {'local': 'results/vision_center_001/', 'drive': DRIVE, 'report': 'VISION_CENTER_EXECUTION_REPORT.md', 'amendment': 'AMENDMENT_015 (execution)', 'report_md': 'not modified unless asked'},
            'files_read_on_the_server': SERVER_READS, 'code': {f: sha256(f) for f in CODE}, 'upstream_commit': UPSTREAM_COMMIT,
            'not_in_this_work': ['policy training or fine-tuning', 'any neural perception model', 'any change of constants after an outcome', 'a second pass', 'calibration', 'final Test']}
    write_json(out / 'plan.json', plan); print(json.dumps({'pass': plan['pass'], 'problems': problems, 'base_policy_checks': policy}, indent=1))
    if problems: raise SystemExit(2)


def cmd_dry_run(a):
    plan = read_json(a.plan); run = Path(a.run); need = plan['files_read_on_the_server']
    if plan.get('pass') is not True or plan.get('problems'): raise SystemExit('Refused: the plan did not pass its own preparation checks')
    missing = [f for f in need if not Path(f).exists()]; changed = [f for f, h in plan['code'].items() if Path(f).exists() and sha256(f) != h]; auth = (run / 'authorization.json').exists()
    out = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha256(a.plan), 'volume': plan['volume'], 'files_needed_on_the_server': need, 'upload_bytes': int(sum(Path(f).stat().st_size for f in need if Path(f).exists())),
           'files_missing': missing, 'code_files_changed_since_plan': changed, 'authorization_present': auth, 'ledger_exists': Path(LEDGER).exists(),
           'paid_compute': 'AUTHORIZATION_FILE_PRESENT_NOT_USED_BY_DRY_RUN' if auth else 'NOT_AUTHORIZED', 'servers_created': 0, 'simulator_steps': 0, 'policy_inferences': 0}
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
    run = Path(a.run); D.require_authorization(run, a.plan); plan, c = bound(a.plan); seeds = VC.pilot_seeds(read_json(MANIFEST), used_ledger())
    from hci.simulation import make_env
    from hci.trees import as_numpy
    env = make_env(c)

    def key(seed):
        env.reset(seed=int(seed)); return np.array([*as_numpy(env.cubeA.pose.p)[0], *as_numpy(env.cubeB.pose.p)[0]], float)
    try:
        refs = {n: np.array(k) for n, k in read_json(KEYS)['keys'].items()}; stored = len(refs); sp = read_json(D.SEED_SOURCES['splits'])['splits']
        reset_sets = {'splits/audit': sp['audit'], 'splits/dev': sp['dev'], **{f'policy_eval/{k}': v for k, v in read_json(D.SEED_SOURCES['policy_eval'])['splits'].items()},
                      **{f'vision_v1/{k}': v for k, v in read_json(D.SEED_SOURCES['vision_v1'])['splits'].items()}, **{f'recovery_v1/{k}': v for k, v in read_json(USED_MANIFEST)['splits'].items()}}
        for name, ids in reset_sets.items():
            for s in ids: refs[f'{name}/{s}'] = key(s)
        names = list(refs); table = np.stack([refs[n] for n in names]); own = {}; clashes = []
        for s in seeds:
            k = key(s); d = np.abs(table - k).max(1); j = int(d.argmin()); twin = [t for t, v in own.items() if np.abs(v - k).max() <= X.IDENTICAL_M]
            if d[j] <= X.IDENTICAL_M or twin: clashes.append({'seed': s, 'identical_to': names[j] if d[j] <= X.IDENTICAL_M else twin[0]})
            own[str(s)] = k
        report = {'pass': not clashes, 'identical_placements': clashes, 'seeds_checked': len(own), 'identical_threshold_m': X.IDENTICAL_M, 'references': {'stored_keys_demonstrations_and_pilots': stored, **{k: len(v) for k, v in reset_sets.items()}},
                  'seed_level_only': ['splits/calibration', 'splits/test'], 'plan_sha256': sha256(a.plan), 'scope': 'Initial cube placement only; no policy was run on these scenes.'}
        write_json(run / 'scenes.json', report); print(json.dumps({k: report[k] for k in ('pass', 'identical_placements', 'seeds_checked')}))
        if clashes: raise SystemExit(2)
    finally:
        env.close()


def save_scene(out, seed, rec, traces, frames):
    tp = out / 'traces' / f'trace_{seed}.json.gz'; tp.parent.mkdir(parents=True, exist_ok=True)
    if tp.exists(): raise FileExistsError(tp)
    with gzip.open(tp, 'xt', encoding='utf-8') as f: json.dump({'seed': int(seed), 'traces': traces}, f)
    rec['trace'] = tp.relative_to(out).as_posix(); rec['trace_sha256'] = sha256(tp); fr = frames.get('VISION_CENTER')
    if fr:
        fp = out / 'frames' / f'hand_{seed}.npz'; fp.parent.mkdir(parents=True, exist_ok=True)
        if fp.exists(): raise FileExistsError(fp)
        np.savez_compressed(fp, frames=np.stack(fr)); rec['frames'] = fp.relative_to(out).as_posix(); rec['frames_sha256'] = sha256(fp)
    write_json(out / 'scenes' / f'{seed}.json', rec)


def timing(recs):
    steps = sum(x['counts']['env_steps'] for x in recs); rd = sum(x['counts']['readings'] for x in recs); rs = sum(x['counts'].get('reading_seconds', 0.) for x in recs); wall = sum(x['wall_seconds'] for x in recs)
    return {'seconds_per_env_step': (wall - rs) / steps, 'seconds_per_reading': (rs / rd) if rd else None, 'env_steps': steps, 'readings': rd}


def cmd_audit(a):
    run = Path(a.run); D.require_authorization(run, a.plan); plan, c = bound(a.plan); out = run / 'audit'; out.mkdir(parents=True, exist_ok=False)
    from hci import vision_v1 as V
    r = V.runtime(c, POLICY['path'], a.upstream, a.device); recs = []
    try:
        facts = X.controller_facts(r)
        for s in AUDIT_SEEDS:
            rec, traces, frames = VC.collect_scene(r, s, c, probe=X.gt_probe, arms=VC.ARMS, role='vision_center_audit'); save_scene(out, s, rec, traces, frames); recs.append(rec)
            v = next((x for x in rec['cells'] if x['arm'] == 'VISION_CENTER'), None)
            print(json.dumps({'audit_scene': s, 'triggered': rec['triggered'], 'decision': None if not rec['triggered'] else v['control']['decision'], 'wall_seconds': rec['wall_seconds']}), flush=True)
    finally:
        r.env.close()
    tm = timing(recs); trig = sum(1 for x in recs if x['triggered']); ok = bool(facts['pass'] and trig >= 2 and tm['seconds_per_reading'] is not None)
    report = {'pass': ok, 'status': 'OK' if ok else 'TECHNICAL_BLOCKED', 'controller': facts, 'triggered_scenes': trig, 'three_arm_procedure_completed_on': len(recs), **tm, 'plan_sha256': sha256(a.plan), 'runtime': runtime_info(),
              'scope': 'Technical audit on already-used scenes. Outcomes here decide nothing.'}
    write_json(out / 'audit.json', report); print(json.dumps({k: report[k] for k in ('pass', 'status', 'seconds_per_env_step', 'seconds_per_reading')}, indent=1))
    if not ok: raise SystemExit(2)


def projection(run, auth, plan_path, stage, usd_per_hour, spent_usd):
    au = read_json(run / 'audit' / 'audit.json')
    if not au['pass'] or au['plan_sha256'] != sha256(plan_path): raise SystemExit('audit missing, failed or of another plan')
    kw = dict(seconds_per_env_step=au['seconds_per_env_step'], seconds_per_reading=au['seconds_per_reading'], usd_per_hour=usd_per_hour, cap_usd=float(auth['cap_usd']))
    pilot = VC.projected_cost(spent_usd=spent_usd, scenes=100, arms=3, **kw)
    if stage == 'pilot': d = pilot
    else:                                                    # the shadow stage is only started if the pilot after it also fits
        sh = VC.projected_cost(spent_usd=spent_usd, scenes=len(shadow_seeds()), arms=2, reserve_usd=0., **kw)
        total = sh['projected_total_usd'] + pilot['projected_total_usd'] - spent_usd; d = {'shadow': sh, 'pilot_after_it': pilot, 'projected_total_usd': total, 'cap_usd': float(auth['cap_usd']), 'fits_cap': bool(total <= float(auth['cap_usd'])),
                                                                                   'inputs': {'usd_per_hour': usd_per_hour, 'spent_usd': spent_usd}}
    d.update(stage=stage, status='OK' if d['fits_cap'] else 'BUDGET_BLOCKED', plan_sha256=sha256(plan_path)); return d


def cmd_budget(a):
    run = Path(a.run); auth = D.require_authorization(run, a.plan); d = projection(run, auth, a.plan, a.stage, a.usd_per_hour, a.spent_usd)
    write_json(run / f'budget_gate_{a.stage}.json', d); print(json.dumps(d, indent=1))
    if not d['fits_cap']: raise SystemExit(2)


def passes(r, seeds, c, out, arms, role, usd_per_hour, spent_usd, cap, scenes_total, plan_hash, after_usd=0.):
    """One pass over `seeds`. `after_usd`: projected cost of the stage that follows this one (the pilot after the shadow pass)."""
    started = time.perf_counter(); done = 0
    for s in seeds:
        try:
            rec, traces, frames = VC.collect_scene(r, s, c, probe=X.gt_probe, arms=arms, role=role); save_scene(out, s, rec, traces, frames)
        except Exception as e:
            import traceback
            write_json(out / 'scenes' / f'{s}.error.json', {'scene': int(s), 'completed': False, 'error': repr(e), 'traceback': traceback.format_exc(), 'scenes_finished_before': done}); raise
        done += 1; print(f"{role} scene={s} triggered={rec['triggered']} done={done}/{scenes_total} wall={rec['wall_seconds']:.0f}s", flush=True)      # outcomes are not printed during a pass
        if done == 5:
            el = time.perf_counter() - started; total = spent_usd + (el + (scenes_total - 5) * el / 5 * 1.25) / 3600 * usd_per_hour + 0.20 + after_usd
            gate = {'elapsed_seconds_five_scenes': el, 'projected_total_usd_including_later_stage': total, 'later_stage_usd': after_usd, 'cap_usd': cap, 'fits_cap': bool(total <= cap), 'inputs': 'time and money only; no outcome is read', 'plan_sha256': plan_hash}
            write_json(out / 'budget_after_five_scenes.json', gate)
            if not gate['fits_cap']: raise SystemExit('BUDGET_BLOCKED after five scenes: the pass stops INCOMPLETE and is backed up; the experiment is not reduced')
    return time.perf_counter() - started


def load_records(d, seeds, c):
    d = Path(d); errors = sorted(p.name for p in (d / 'scenes').glob('*.error.json'))
    if errors: raise SystemExit(f'INCOMPLETE: error files present {errors}; a partial run is not summarised')
    recs = []
    for s in seeds:
        p = d / 'scenes' / f'{s}.json'
        if not p.exists(): raise SystemExit(f'INCOMPLETE: no record for scene {s}; a partial run is not summarised')
        rec = read_json(p); VC.validate_scene(rec, c)
        if sha256(d / rec['trace']) != rec['trace_sha256']: raise SystemExit(f'trace of scene {s} changed')
        recs.append(rec)
    return recs


def cmd_shadow(a):
    run = Path(a.run); auth = D.require_authorization(run, a.plan); plan, c = bound(a.plan); h = sha256(a.plan); seeds = shadow_seeds(); out = run / 'shadow'
    for name in ('scenes.json', 'audit/audit.json', 'budget_gate_shadow.json'):
        if not (run / name).exists(): raise SystemExit(f'{name} missing: placement check, audit and budget gate come first')
    sc = read_json(run / 'scenes.json'); au = read_json(run / 'audit' / 'audit.json'); bg = read_json(run / 'budget_gate_shadow.json')
    if not (sc['pass'] and sc['plan_sha256'] == h and au['pass'] and au['plan_sha256'] == h and bg['fits_cap'] and bg.get('plan_sha256') == h): raise SystemExit('placement check, audit or budget gate not passed for this plan')
    if out.exists(): raise SystemExit(f'{out} exists: the shadow pass is never resumed or repeated')
    now = projection(run, auth, a.plan, 'shadow', a.usd_per_hour, a.spent_usd)
    if not now['fits_cap']: print(json.dumps(now, indent=1)); raise SystemExit('BUDGET_BLOCKED at the start of the shadow pass')
    write_json(run / 'budget_gate_shadow_at_start.json', now); (out / 'scenes').mkdir(parents=True)
    from hci import vision_v1 as V
    r = V.runtime(c, POLICY['path'], a.upstream, a.device)
    later = now['pilot_after_it']['projected_total_usd'] - a.spent_usd - 0.20        # the pilot's own reserve is not counted twice
    try: wall = passes(r, seeds, c, out, VC.SHADOW_ARMS, VC.SHADOW_ROLE, a.usd_per_hour, a.spent_usd, float(auth['cap_usd']), len(seeds), h, after_usd=later)
    finally: r.env.close()
    res = VC.shadow_verdict(load_records(out, seeds, c)); res.update(plan_sha256=h, wall_seconds=wall, runtime=runtime_info()); write_json(out / 'shadow_result.json', res)
    print(json.dumps({'label': res['label'], 'criteria': res['criteria']}, indent=1))
    if not res['pass']: raise SystemExit(2)


def cmd_run(a):
    run = Path(a.run); auth = D.require_authorization(run, a.plan); plan, c = bound(a.plan); h = sha256(a.plan); out = run / 'pilot'
    for name in ('scenes.json', 'audit/audit.json', 'shadow/shadow_result.json', 'budget_gate_pilot.json'):
        if not (run / name).exists(): raise SystemExit(f'{name} missing: placement check, audit, shadow pass and budget gate come first')
    sc = read_json(run / 'scenes.json'); sh = read_json(run / 'shadow' / 'shadow_result.json'); au = read_json(run / 'audit' / 'audit.json'); bg = read_json(run / 'budget_gate_pilot.json')
    if not (sc['pass'] and sc['plan_sha256'] == h and sh['pass'] and sh['plan_sha256'] == h and au['pass'] and au['plan_sha256'] == h and bg['fits_cap'] and bg.get('plan_sha256') == h):
        raise SystemExit('placement check, audit, shadow pass or budget gate not passed for this plan')
    if Path(LEDGER).exists(): raise SystemExit(f'{LEDGER} exists: the pilot set already has a recorded use')
    Path(LEDGER).parent.mkdir(parents=True, exist_ok=True)
    seeds = VC.pilot_seeds(read_json(MANIFEST), used_ledger())
    if out.exists(): raise SystemExit(f'{out} exists: the pilot pass is never resumed or repeated')
    now = projection(run, auth, a.plan, 'pilot', a.usd_per_hour, a.spent_usd)
    if not now['fits_cap']: print(json.dumps(now, indent=1)); raise SystemExit('BUDGET_BLOCKED at the start of the pilot: the set stays unused')
    write_json(run / 'budget_gate_pilot_at_start.json', now)
    write_json(LEDGER, {'set': VC.SET, 'used_utc': datetime.now(timezone.utc).isoformat(), 'session': out.as_posix(), 'plan_sha256': h, 'seed_manifest_sha256': plan['seed_registry']['sha256'], 'policy': POLICY['path'],
                        'weights_sha256': POLICY['weights.pt'], 'arms': list(VC.ARMS), 'note': 'Single three-arm pilot pass. Written before any outcome existed. No resume, no second pass.'})
    (out / 'scenes').mkdir(parents=True)
    from hci import vision_v1 as V
    r = V.runtime(c, POLICY['path'], a.upstream, a.device)
    try: wall = passes(r, seeds, c, out, VC.ARMS, VC.ROLE, a.usd_per_hour, a.spent_usd, float(auth['cap_usd']), len(seeds), h)
    finally: r.env.close()
    cmd_summarize(argparse.Namespace(pilot=str(out), plan=a.plan, out=str(out / 'result.json'), wall=wall))


def cmd_summarize(a):
    c = load_config(CONFIG); seeds = VC.pilot_seeds(read_json(MANIFEST), {}); recs = load_records(a.pilot, seeds, c); res = VC.summarize(recs, seeds)
    res.update(set=VC.SET, seeds=[seeds[0], seeds[-1]], policy=POLICY['path'], weights_sha256=POLICY['weights.pt'], plan_sha256=sha256(a.plan), rules_sha256=sha256('src/hci/vision_center.py'),
               oracle_reads_outside_center=sum(x['oracle_reads'] for r in recs for x in r['cells'] if x['arm'] != 'CENTER_ORACLE' and not x['no_trigger']), wall_seconds=getattr(a, 'wall', None), runtime=runtime_info())
    write_json(a.out, res)
    print(json.dumps({'success': {k: v['k'] for k, v in res['success'].items()}, 'primary': {k: res['primary_VISION_CENTER_vs_C0'][k] for k in ('rescue', 'harm', 'net')}, 'gate': res['followup_gate']['label']}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--out', required=True); q.add_argument('--oracle-plan', default='results/recovery_v1_oracle_xy_pilot_001/preparation/plan.json')
    q = s.add_parser('dry-run'); q.add_argument('--plan', required=True); q.add_argument('--run', required=True); q.add_argument('--out')
    for name in ('scenes', 'audit', 'budget', 'shadow', 'run'):
        q = s.add_parser(name); q.add_argument('--plan', required=True); q.add_argument('--run', required=True)
        if name in ('audit', 'shadow', 'run'): q.add_argument('--upstream', default='third_party/ManiSkill'); q.add_argument('--device', default='cuda')
        if name in ('budget', 'shadow', 'run'): q.add_argument('--usd-per-hour', type=float, required=True); q.add_argument('--spent-usd', type=float, required=True)
        if name == 'budget': q.add_argument('--stage', choices=('shadow', 'pilot'), required=True)
    q = s.add_parser('summarize'); q.add_argument('--pilot', required=True); q.add_argument('--plan', required=True); q.add_argument('--out', required=True)
    a = p.parse_args(); globals()['cmd_' + a.cmd.replace('-', '_')](a)


if __name__ == '__main__':
    main()
