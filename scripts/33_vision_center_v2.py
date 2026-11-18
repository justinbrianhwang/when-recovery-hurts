"""Temporal Vision-CENTER v2 (work id vision_center_v2_001): Vision-CENTER with a memory of earlier RGB observations and robot
state. PREPARATION ONLY at the time of writing: nothing below `dry-run` has been run. No training, no calibration, no Test.

  plan        local: frozen policy, code hashes, seed registry, constants, PROPOSED criteria, cost estimate
  dry-run     local: files and volume; starts nothing
  scenes      server: cube placements of the 200 new pilot seeds against every placement on record (no policy)
  audit       server: 3 used selection scenes, all four arms: technical checks and timing for the budget gates
  budget      server: worst-case projection of the stage named by --stage (shadow | pilot, each including what follows it)
  shadow      server, stage B1: the 300 USED scenes 50600-50899, arms C0, VISION_CENTER_V1, VISION_CENTER_V2. Both cameras'
              frames of every step up to the trigger are stored (the video these scenes never had); fixed criteria decide
              whether the new scenes are run
  run         server, stage B2: the one pilot pass on the 200 new scenes, all four arms. Ledger first. No resume.
  summarize   anywhere: every table recomputed from the stored scene records
Server commands refuse to run unless <run>/authorization.json exists, names the plan by hash and carries a cap.
No command in this file writes that authorisation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, gzip, importlib.util, json, time
import numpy as np
from hci.io import load_config, read_json, write_json, sha256, digest, runtime_info, UPSTREAM_COMMIT
from hci import vision_center as V1
from hci import vision_center_v2 as V2
from hci import recovery_v1_oracle as O

_spec = importlib.util.spec_from_file_location('vc_driver', Path(__file__).with_name('32_vision_center.py')); Y = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(Y)
X = Y.X; D = Y.D
WORK = V2.WORK
POLICY = D.POLICY; CONFIG = D.CONFIG; KEYS = D.KEYS
MANIFEST = 'configs/vision_center_v2_splits.json'
USED = {'recovery_v1': Y.USED_MANIFEST, 'vision_center': Y.MANIFEST}             # recovery_v1_diag, recovery_v1_pilot_reserved, vision_center_pilot: all used
AUDIT_SEEDS = Y.AUDIT_SEEDS                                                       # vision_v1_select, already used
V1_RECORD = {'ledger': 'results/vision_center_001/run2/ledger_vision_center_pilot.json', 'result': 'results/vision_center_001/run2/pilot/result.json'}
OFFLINE = {'memory_replay': 'results/vision_center_v2_001/preparation/memory_replay.json', 'reconstruction_audit': 'results/vision_center_v2_001/preparation/reconstruction_audit.json',
           'ground_truth_audit': 'results/vision_center_v2_001/preparation/ground_truth_audit.json', 'base_camera_audit': 'results/vision_center_v2_001/preparation/base_camera_audit.json'}
CODE = D.INFERENCE_CODE + ['src/hci/recovery_v1.py', 'src/hci/recovery_v1_oracle.py', 'src/hci/vision_center.py', 'src/hci/vision_center_v2.py', 'src/hci/pilot2b.py', 'src/hci/pilot2d.py', 'scripts/30_recovery_v1_diag.py',
                           'scripts/31_recovery_v1_oracle_xy.py', 'scripts/32_vision_center.py', 'scripts/33_vision_center_v2.py', 'scripts/vision_center_v2/run.sh', 'scripts/stage1b/common.sh', CONFIG, MANIFEST, *USED.values()]
SERVER_READS = sorted({*CODE, *D.SEED_SOURCES.values(), D.DEMO_MANIFEST, *D.PRIOR_SCENES, KEYS, POLICY['path'] + '/weights.pt', POLICY['path'] + '/policy_meta.json'})
LEDGER = 'artifacts/ledger/vision_center_v2_pilot.json'
DRIVE = 'Robot Harm/results/vision_center_v2_001/'
STATUS = {'vision_center_v2': 'PHASE_A_PREPARATION_ONLY', 'paid_compute': 'NOT_AUTHORIZED', 'shadow_pass_on_used_scenes': 'NOT_RUN', 'pilot_on_new_scenes': 'NOT_RUN', 'criteria': 'PROPOSED; fixed by approving this plan by hash before any rollout',
          'vision_center_v1': 'EXECUTED_ONCE (run2: NO_VISION_GATE_PASS); records and code unchanged', 'base_policy': 'FROZEN: Vision-V1 train seed 1, 80k EMA, unchanged',
          'new_training': 'NOT_PLANNED (no network; colour segmentation, projection, and a memory of earlier estimates)', 'calibration': 'NOT_RUN', 'final_test': 'NOT_RUN'}
SEMANTICS = {
    'inputs': 'base_camera and hand_camera RGB (128 x 128), extra/tcp_pose and agent/qpos of the policy observation, the executed gripper commands, the step counter; fixed camera constants. No cube pose, grasp flag, task flag, outcome or future information. '
              'No added camera, no viewpoint motion, no trained model',
    'trigger': {'rule': 'the oracle pilot trigger, unchanged (hci.recovery_v1_oracle.trigger_rule)', **O.TRIGGER},
    'VISION_CENTER_V1': 'hci.vision_center.vision_center, unchanged: the single-frame baseline (hand camera at the trigger only)',
    'VISION_CENTER_V2': {
        'memory': {'target': 'at every step before the trigger: the hand-camera fit of v1 when accepted, otherwise the base-camera estimate (largest connected lit-green region, area of an unoccluded 4 cm face). Consistent observations confirm one track; '
                             'an inconsistent observation restarts it; the track is usable after confirm_min observations',
                   'invalidation': 'a restart on any conflicting observation; steps without an observation while the gripper is near the remembered target (contact_radius_m, contact_tcp_z_m) add u_push_m to the bound; '
                                   'at the trigger the remembered position is checked against the current hand image (lit-green pixels far outside it, or visible table inside it, refuse it)',
                   'hold': 'the last accepted held-cube fit of v1 (run every hold_every-th step while the gripper is closed on something); dropped when the finger opening changes by more than finger_tol_m or an open command was executed',
                   'never': 'a remembered position is never used without confirmation, never after a conflict, never against the current image', **V2.MEMORY},
        'estimate': 'current frame where v1 accepts it; memory otherwise; no estimate -> abstain. Bound u = target bound + hold bound',
        'abstain': {'meaning': 'nothing is touched: the episode is the uninterrupted one (checked against the trunk in every scene)', 'reasons': list(V2.ABSTAIN), 'order': 'as in hci.vision_center_v2.decide; the first that applies'},
        'safety': {'LOW_CARRY': f'estimated held-cube centre below {V2.CARRY_Z_MIN_M} m (v1 rule); with a remembered target also when its bottom is below the target top face',
                   'SIDE_CONTACT_RISK': f'estimated offset (plus u when the target is remembered) >= {V2.SIDE_OFFSET_M} m (the held cube is beside the target) and its bottom less than {V2.SIDE_GAP_MIN_M} m above the target top face: scene 50892',
                   'during_control': f'stop when a new accepted target estimate is more than {V2.TARGET_JUMP_MAX_M} m (plus the bound of a remembered target) from the one in use, when the held-cube fit moves more than {V2.HOLD_SHIFT_MAX_M} m in the gripper, '
                                     'when the estimate is lost, when the held cube is estimated too low'},
        'acts_when': f'estimated offset >= {V2.ACT_MIN_M} m and estimated offset - u >= {V2.RESIDUAL_TOL_M} m and no abstain reason',
        'command': 'as CENTER_ORACLE and v1: horizontal only, at most 0.005 m per step, commanded target within 0.030 m of the TCP at the trigger, at most 12 steps, vertical command 0, held gripper command',
        'stops_when': f'estimated offset <= {V2.RESIDUAL_TOL_M} m on a target seen in the current frame, <= {V2.MEMORY_STOP_TOL_M} m on a remembered target; or 12 steps; or the target cap; or the episode ends; or a safety stop',
        'after': 'queue cleared, the unchanged policy re-infers', 'decision': V2.DECISION},
    'CENTER_ORACLE': 'hci.recovery_v1_oracle.center_oracle, unchanged (reads simulator cube positions; diagnostic reference, not deployable)',
    'C0': 'the pending action and the rest of the queue are executed; must equal the uninterrupted run, otherwise the run stops',
    'noise': "repeat 0 only; every inference is keyed by seed_for(scene, physical step, 0, 'dp')", 'restore': 'reset(seed) + prefix action replay; the restored runtime is compared with the snapshot before every branch',
    'chosen_on': 'every v2 constant was chosen on used data: the 300 used scenes 50600-50899 (traces, trigger frames), the 21 failed diagnostic episodes with video, and the 500 training demonstrations. '
                 'The four scenes 50813, 50872, 50817, 50846 are part of that development data and are not evidence for v2'}
ANALYSIS = {'unit': 'scene',
            'stage_B1_shadow': {'scenes': '300 used scenes (50600-50899), re-run on the server; development data', 'arms': list(V2.SHADOW_ARMS), 'pass_criteria': V2.SHADOW_PASS,
                                'also_produces': 'both cameras\' frames of every step up to the trigger for all 300 scenes (the re-collection the observability audit could not do offline)',
                                'reports': ['memory: which source, age, exposure, error against logged ground truth', 'decisions by reason for v1 and v2', 'acts on a remembered position and their errors', 'unnecessary acts', 'target pushed during control',
                                            'paired outcomes of v2 and v1 against C0 and against each other'],
                                'on_failure': 'V2_SHADOW_BLOCKED: the new scenes are not touched; no constant is tuned and re-tried under this plan',
                                'caveat': 'rollouts are not bit-identical across hosts (run2: 9 of 200 uninterrupted outcomes differed from earlier runs), so a re-run scene need not be the episode recorded before'},
            'stage_B2_pilot': {'scenes': '200 new scenes (50900-51099)', 'arms': list(V2.ARMS), 'primary': 'paired success_once_absorbing, VISION_CENTER_V2 against C0, all 200 scenes: rescue / harm / net, Wilson intervals',
                               'secondary': ['VISION_CENTER_V1 against C0 on the same scenes', 'v2 against v1', 'CENTER_ORACLE against C0; share of oracle rescues also rescued by v2 and by v1', 'abstain rate of v1 and v2 among triggered scenes',
                                             f'memory errors: acts on a remembered position whose offset error exceeded {V2.MEMORY_ERROR_M} m or its own bound', 'unnecessary acts; target pushed more than 1 mm'],
                               'gate': {'thresholds': V2.GATE, 'all_of': ['v2 rescues >= rescue_min', 'net >= net_min', 'harm <= harm_max', 'memory errors <= memory_errors_max', 'share of oracle rescues recovered >= oracle_recovery_min',
                                                                          'v2 rescues - v1 rescues >= v2_rescue_minus_v1_rescue_min'], 'otherwise': 'NO_V2_GATE_PASS: no extension, no tuning, no subset search'}},
            'criteria_status': 'PROPOSED by the agent in Phase A. They are fixed, before any result exists, by the user approving this plan by its SHA256; a different choice means a new plan and a new hash',
            'not_claimed': ['statistical significance', 'transfer to real images (flat simulator colours, exact camera parameters, no noise)', 'anything beyond StackCube and this policy', 'the used scenes 50600-50899 as independent evidence']}


def used_ledger():
    used = X.used_ledger()
    for name, path in USED.items():
        for k, v in read_json(path)['splits'].items(): used[f'{name}/{k}'] = list(v)
    return used


def shadow_seeds():
    sets = {f'{name}/{k}': list(v) for name, path in USED.items() for k, v in read_json(path)['splits'].items()}; ids = []
    for name, (lo, hi) in V2.SHADOW_SETS:
        if sets[name] != list(range(lo, hi + 1)): raise ValueError(f'{name} is not {lo}-{hi}')
        ids += sets[name]
    return ids


def offline_short():
    out = {}
    for k, p in OFFLINE.items():
        if not Path(p).exists(): out[k] = None; continue
        out[k] = {'path': p, 'sha256': sha256(p), 'summary': read_json(p).get('summary')}
    return out


def cmd_plan(a):
    c = load_config(CONFIG); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); problems = []
    if (out / 'plan.json').exists(): raise SystemExit(f'Refused overwrite: {out / "plan.json"}')
    try: seeds = V2.pilot_seeds(read_json(MANIFEST), used_ledger())
    except ValueError as e: problems.append(str(e)); seeds = []
    policy = D.check_policy(problems); led = read_json(V1_RECORD['ledger']); res = read_json(V1_RECORD['result'])
    if led.get('set') != 'vision_center_pilot' or res['followup_gate']['label'] != 'NO_VISION_GATE_PASS': problems.append('the Vision-CENTER v1 record is not the executed NO_VISION_GATE_PASS run')
    if Path(LEDGER).exists(): problems.append('a ledger for the new pilot set already exists')
    if [c['policy'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')] != [2, 8, 16] or c['env']['horizon'] != 200 or c['env']['control_mode'] != 'pd_ee_delta_pos' or c['env']['robot_uid'] != 'panda_wristcam': problems.append('config is not the fixed setting')
    if not set(AUDIT_SEEDS) <= set(read_json(D.SEED_SOURCES['vision_v1'])['splits']['vision_v1_select']): problems.append('audit seeds are not in vision_v1_select')
    offline = offline_short()
    for k, v in offline.items():
        if v is None: problems.append(f'offline analysis missing: {k}')
    n_sh = len(shadow_seeds()); ce = a.cost
    plan = {'work_id': WORK, 'stage': 'Temporal Vision-CENTER v2 (Phase A: preparation)', 'created_utc': datetime.now(timezone.utc).isoformat(), 'pass': not problems, 'problems': problems, 'status': STATUS,
            'authorization': {'paid_compute_authorized': False, 'proposed_cap_usd': ce['proposed_cap_usd'], 'proposed_cap_is': 'a proposal, explicitly not an approval; earlier approvals are not inherited',
                              'needed': 'a separate explicit user instruction that states an exact USD cap AND names the SHA256 of this plan file (which also fixes the criteria it carries)',
                              'required_file': 'authorization.json in the run folder, written only after such an instruction: {"approved_by": "user", "paid_compute_authorized": true, "cap_usd": <number>, "plan_sha256": "<sha256 of this plan>"}'},
            'question': 'Can the positions of the target cube and of the held cube that a single frame does not give be recovered reliably from earlier RGB observations and the robot state history alone, and does that recover more of the oracle gain than '
                        'single-frame Vision-CENTER without harming scenes the policy already solves?',
            'privileged_information': {'vision_center_v2': 'none', 'vision_center_v1': 'none', 'offline_labels': 'cube positions are labels in the offline analyses and in after-the-fact error columns of the records',
                                       'center_oracle': 'unchanged reference arm; the only control code that reads cube positions', 'project_rule': 'AGENTS.md: "모니터/정책에 객체 정답 pose, task success, 미래 라벨을 입력하지 않습니다."'},
            'repository': {'head': D.git('rev-parse', 'HEAD'), 'working_tree_changes_at_plan_time': D.git('status', '--porcelain').splitlines()},
            'base_policy': {**POLICY, 'checks': policy, 'rule': 'frozen; not retrained, not fine-tuned, no other checkpoint'},
            'config': {'path': CONFIG, 'file_sha256': sha256(CONFIG), 'config_sha256': digest(c), 'env': c['env'], 'policy': c['policy'], 'audit': c['audit']},
            'seed_registry': {'path': MANIFEST, 'sha256': sha256(MANIFEST), 'set': V2.SET, 'range': list(V2.RANGE), 'n': len(seeds), 'first_last': seeds[:1] + seeds[-1:],
                              'seed_level_check': 'no overlap with splits.json (audit, train, dev, calibration, test), policy_eval_splits.json, vision_v1_splits.json, recovery_v1_splits.json (both sets), vision_center_splits.json, '
                                                  'the 550 demonstration scene seeds, or the scenes of Stage 2A / 2B / 2D / H1; checked by this command (a clash fails the plan)',
                              'placement_check': 'server step `scenes`: the initial cube placements of the 200 seeds against every placement on record and against each other; an identical placement fails the step. Not possible locally (needs the simulator)',
                              'shadow_scenes': {'sets': [[n, list(r)] for n, r in V2.SHADOW_SETS], 'n': n_sh, 'status': 'used; development data; no new scene is consumed by stage B1'},
                              'untouched_sets': ['calibration', 'test'], 'no_replacement': 'no scene is replaced or dropped for any reason'},
            'arms': list(V2.ARMS), 'semantics': SEMANTICS, 'analysis': ANALYSIS,
            'audit': {'seeds': list(AUDIT_SEEDS), 'does': ['all four arms with every technical check of the pilot', 'seconds per environment step, per vision reading and per memory step for the budget gates'], 'outcomes': 'decide nothing'},
            'run': {'order': ['scenes', 'audit', 'budget --stage shadow', 'shadow', 'budget --stage pilot', 'run'], 'ledger': LEDGER + ', created exclusively before any rollout on a new scene; no resume, no second pass',
                    'on_error': 'the pass stops and stays INCOMPLETE; no scene is dropped or replaced; a partial run has no summary',
                    'stored': 'one record per scene, per-step ground-truth traces (logging), both cameras\' frames of every step up to the trigger, and the frames of both vision arms at the trigger and during control',
                    'thread_limit': 'OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS, BLIS_NUM_THREADS = 1 before every Python process (run.sh exports them)'},
            'offline_analyses': offline,
            'volume': {'placement_resets': 'the 200 new seeds plus every reference set that has no stored placement key', 'audit_scenes': len(AUDIT_SEEDS), 'shadow_scenes': n_sh, 'pilot_scenes': len(seeds),
                       'steps_per_scene_bound': {'shadow': 200 + 3 * 200, 'pilot': 200 + 4 * 200}, 'memory_steps_per_scene_bound': 200, 'readings_per_scene_bound': 2 * 13},
            'cost_estimate': ce, 'host': X.read_json(a.oracle_plan)['host'] if a.oracle_plan else None,
            'command_order': ['local: tests, `plan`, `dry-run`', 'user: explicit approval naming a USD cap and this plan hash -> authorization.json', 'server: install', 'server: run.sh env', 'server: run.sh scenes', 'server: run.sh audit',
                              'server: run.sh shadow (budget gate first; V2_SHADOW_BLOCKED stops here)', 'server: run.sh pilot (budget gate first)', 'server: run.sh pack', 'local: download, hash checks, `summarize` from the raw records',
                              'local: delete only the instances created for this run', f'Drive: report and results to {DRIVE}'],
            'outputs': {'local': 'results/vision_center_v2_001/', 'drive': DRIVE, 'report_md': 'not modified unless asked'},
            'files_read_on_the_server': SERVER_READS, 'code': {f: sha256(f) for f in CODE}, 'upstream_commit': UPSTREAM_COMMIT,
            'not_in_this_work': ['policy training or fine-tuning', 'any neural perception model', 'an added camera or a viewpoint motion', 'any change of constants or criteria after an outcome', 'a second pass', 'calibration', 'final Test',
                                 'any change to the Vision-CENTER v1 code, records or result']}
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
    run = Path(a.run); D.require_authorization(run, a.plan); plan, c = bound(a.plan); seeds = V2.pilot_seeds(read_json(MANIFEST), used_ledger())
    from hci.simulation import make_env
    from hci.trees import as_numpy
    env = make_env(c)

    def key(seed):
        env.reset(seed=int(seed)); return np.array([*as_numpy(env.cubeA.pose.p)[0], *as_numpy(env.cubeB.pose.p)[0]], float)
    try:
        refs = {n: np.array(k) for n, k in read_json(KEYS)['keys'].items()}; stored = len(refs); sp = read_json(D.SEED_SOURCES['splits'])['splits']
        reset_sets = {'splits/audit': sp['audit'], 'splits/dev': sp['dev'], **{f'policy_eval/{k}': v for k, v in read_json(D.SEED_SOURCES['policy_eval'])['splits'].items()},
                      **{f'vision_v1/{k}': v for k, v in read_json(D.SEED_SOURCES['vision_v1'])['splits'].items()}, **{f'{n}/{k}': v for n, p in USED.items() for k, v in read_json(p)['splits'].items()}}
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
    rec['trace'] = tp.relative_to(out).as_posix(); rec['trace_sha256'] = sha256(tp); fr = {k: np.stack(v) for k, v in frames.items() if v}
    if fr:
        fp = out / 'frames' / f'frames_{seed}.npz'; fp.parent.mkdir(parents=True, exist_ok=True)
        if fp.exists(): raise FileExistsError(fp)
        np.savez_compressed(fp, **fr); rec['frames'] = fp.relative_to(out).as_posix(); rec['frames_sha256'] = sha256(fp); rec['frames_layout'] = 'each frame is 128 x 256: base_camera left, hand_camera right; VISION_CENTER_V1 frames are hand_camera only'
    write_json(out / 'scenes' / f'{seed}.json', rec)


def timing(recs):
    steps = sum(x['counts']['env_steps'] for x in recs); rd = sum(x['counts']['readings'] for x in recs); rs = sum(x['counts'].get('reading_seconds', 0.) for x in recs); wall = sum(x['wall_seconds'] for x in recs)
    ms = sum(x['counts']['memory_steps'] for x in recs); msec = sum(x['counts']['memory_seconds'] for x in recs)
    return {'seconds_per_env_step': (wall - rs - msec) / steps, 'seconds_per_reading': (rs / rd) if rd else None, 'seconds_per_memory_step': (msec / ms) if ms else None, 'env_steps': steps, 'readings': rd, 'memory_steps': ms}


def cmd_audit(a):
    run = Path(a.run); D.require_authorization(run, a.plan); plan, c = bound(a.plan); out = run / 'audit'; out.mkdir(parents=True, exist_ok=False)
    from hci import vision_v1 as V
    r = V.runtime(c, POLICY['path'], a.upstream, a.device); recs = []
    try:
        facts = X.controller_facts(r)
        for s in AUDIT_SEEDS:
            rec, traces, frames = V2.collect_scene(r, s, c, probe=X.gt_probe, arms=V2.ARMS, role='vision_center_v2_audit'); save_scene(out, s, rec, traces, frames); recs.append(rec)
            v = next((x for x in rec['cells'] if x['arm'] == 'VISION_CENTER_V2'), None)
            print(json.dumps({'audit_scene': s, 'triggered': rec['triggered'], 'v2_decision': None if not rec['triggered'] else v['control']['decision'], 'wall_seconds': rec['wall_seconds']}), flush=True)
    finally:
        r.env.close()
    tm = timing(recs); trig = sum(1 for x in recs if x['triggered']); ok = bool(facts['pass'] and trig >= 2 and tm['seconds_per_reading'] is not None and tm['seconds_per_memory_step'] is not None)
    report = {'pass': ok, 'status': 'OK' if ok else 'TECHNICAL_BLOCKED', 'controller': facts, 'triggered_scenes': trig, 'four_arm_procedure_completed_on': len(recs), **tm, 'plan_sha256': sha256(a.plan), 'runtime': runtime_info(),
              'scope': 'Technical audit on already-used scenes. Outcomes here decide nothing.'}
    write_json(out / 'audit.json', report); print(json.dumps({k: report[k] for k in ('pass', 'status', 'seconds_per_env_step', 'seconds_per_reading', 'seconds_per_memory_step')}, indent=1))
    if not ok: raise SystemExit(2)


def projection(run, auth, plan_path, stage, usd_per_hour, spent_usd):
    au = read_json(run / 'audit' / 'audit.json')
    if not au['pass'] or au['plan_sha256'] != sha256(plan_path): raise SystemExit('audit missing, failed or of another plan')
    kw = dict(seconds_per_env_step=au['seconds_per_env_step'], seconds_per_reading=au['seconds_per_reading'], seconds_per_memory_step=au['seconds_per_memory_step'], usd_per_hour=usd_per_hour, cap_usd=float(auth['cap_usd']))
    pilot = V2.projected_cost(spent_usd=spent_usd, scenes=V2.RANGE[1] - V2.RANGE[0] + 1, arms=4, **kw)
    if stage == 'pilot': d = pilot
    else:                                                    # the shadow stage is only started if the pilot after it also fits
        sh = V2.projected_cost(spent_usd=spent_usd, scenes=len(shadow_seeds()), arms=3, reserve_usd=0., **kw)
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
            rec, traces, frames = V2.collect_scene(r, s, c, probe=X.gt_probe, arms=arms, role=role); save_scene(out, s, rec, traces, frames)
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
        rec = read_json(p); V2.validate_scene(rec, c)
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
    try: wall = passes(r, seeds, c, out, V2.SHADOW_ARMS, V2.SHADOW_ROLE, a.usd_per_hour, a.spent_usd, float(auth['cap_usd']), len(seeds), h, after_usd=later)
    finally: r.env.close()
    res = V2.shadow_verdict(load_records(out, seeds, c)); res.update(plan_sha256=h, wall_seconds=wall, runtime=runtime_info()); write_json(out / 'shadow_result.json', res)
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
    seeds = V2.pilot_seeds(read_json(MANIFEST), used_ledger())
    if out.exists(): raise SystemExit(f'{out} exists: the pilot pass is never resumed or repeated')
    now = projection(run, auth, a.plan, 'pilot', a.usd_per_hour, a.spent_usd)
    if not now['fits_cap']: print(json.dumps(now, indent=1)); raise SystemExit('BUDGET_BLOCKED at the start of the pilot: the set stays unused')
    write_json(run / 'budget_gate_pilot_at_start.json', now)
    write_json(LEDGER, {'set': V2.SET, 'used_utc': datetime.now(timezone.utc).isoformat(), 'session': out.as_posix(), 'plan_sha256': h, 'seed_manifest_sha256': plan['seed_registry']['sha256'], 'policy': POLICY['path'],
                        'weights_sha256': POLICY['weights.pt'], 'arms': list(V2.ARMS), 'note': 'Single four-arm pilot pass. Written before any outcome existed. No resume, no second pass.'})
    (out / 'scenes').mkdir(parents=True)
    from hci import vision_v1 as V
    r = V.runtime(c, POLICY['path'], a.upstream, a.device)
    try: wall = passes(r, seeds, c, out, V2.ARMS, V2.ROLE, a.usd_per_hour, a.spent_usd, float(auth['cap_usd']), len(seeds), h)
    finally: r.env.close()
    cmd_summarize(argparse.Namespace(pilot=str(out), plan=a.plan, out=str(out / 'result.json'), wall=wall))


def cmd_summarize(a):
    c = load_config(CONFIG); seeds = V2.pilot_seeds(read_json(MANIFEST), {}); recs = load_records(a.pilot, seeds, c); res = V2.summarize(recs, seeds)
    res.update(set=V2.SET, seeds=[seeds[0], seeds[-1]], policy=POLICY['path'], weights_sha256=POLICY['weights.pt'], plan_sha256=sha256(a.plan), rules_sha256={f: sha256(f) for f in ('src/hci/vision_center.py', 'src/hci/vision_center_v2.py')},
               oracle_reads_outside_center=sum(x['oracle_reads'] for r in recs for x in r['cells'] if x['arm'] != 'CENTER_ORACLE' and not x['no_trigger']), wall_seconds=getattr(a, 'wall', None), runtime=runtime_info())
    write_json(a.out, res)
    print(json.dumps({'success': {k: v['k'] for k, v in res['success'].items()}, 'primary': {k: res['primary_V2_vs_C0'][k] for k in ('rescue', 'harm', 'net')}, 'gate': res['followup_gate']['label']}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--out', required=True); q.add_argument('--oracle-plan', default='results/recovery_v1_oracle_xy_pilot_001/preparation/plan.json'); q.add_argument('--cost', required=True, type=lambda x: read_json(x))
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
