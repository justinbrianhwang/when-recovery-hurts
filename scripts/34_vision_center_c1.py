"""Vision-CENTER v2, Phase C1 (work id vision_center_v2_c1_001): component ablation and a second base policy on 200 new
development scenes. PREPARATION ONLY at the time of writing: nothing below `dry-run` has been run. No training, no
calibration, no Test. v1, v2 and the oracle are used unchanged.

  plan        local: both frozen policies, code hashes, seed registry, arm registry, fixed analysis, cost projection
  dry-run     local: files and volume; starts nothing
  clock       server, first: writes the session clock once (start time, hourly price, spend before this instance)
  scenes      server: cube placements of the 200 new seeds against every placement on record (no policy)
  audit       server: 3 used selection scenes x both policies, every arm: technical checks and timing
  budget      server: worst-case projection of the whole run from the audit timing and the session clock
  run         server: ledger first, then the pass with train seed 1 and the pass with train seed 2. No resume.
  summarize   anywhere: every table recomputed from the stored scene records, per policy
Server commands refuse to run unless <run>/authorization.json exists, names the plan by hash and carries a cap.
No command in this file writes that authorisation. Spend is computed from the session clock, not taken from the caller.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, importlib.util, json, time
import numpy as np
from hci.io import load_config, read_json, write_json, sha256, digest, runtime_info, UPSTREAM_COMMIT
from hci import vision_center as V1
from hci import vision_center_v2 as V2
from hci import vision_center_c1 as C1
from hci import vision_center_v2s as S
from hci import recovery_v1_oracle as O

_spec = importlib.util.spec_from_file_location('vc2_driver', Path(__file__).with_name('33_vision_center_v2.py')); Z = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(Z)
X = Z.X; D = Z.D
WORK = C1.WORK; RUN_DEFAULT = 'runs/vision_center_v2_c1_001'
CONFIG = D.CONFIG; KEYS = D.KEYS
MANIFEST = 'configs/vision_center_c1_splits.json'; REGISTRY = 'configs/vision_center_c1_arms.json'
POLICIES = {'train_seed_1': dict(D.POLICY),
            'train_seed_2': {'path': 'artifacts/policies/stack_v1_s2_d500_clean_resnet18_i080000', 'weights.pt': '52eff6a94a075bb3947e68756f8185bc1bfcb1893b691ee3529bbbc0c521b97b',
                             'policy_meta.json': 'c3606292f65c91e7280ba08be983c7f7a65c76d9450abd5604934656973302c7', 'train_seed': 2, 'iteration': 80000, 'weights': 'EMA',
                             'source_checkpoint_sha256': 'd47ccf88adfcc1778974ab087303a0c23f3ecdc9e0e7a8e2833744b034a50f63'}}
S2_RECORD = {'backup_list': 'results/vision_v1_001/run2/archive_file_sha256.txt', 'select_result': 'results/vision_v1_001/run2/select_seed2/result.json', 'selection': 'results/vision_v1_001/run2/selection.json'}
V2_PLAN = 'results/vision_center_v2_001/preparation/plan.json'; V2_PLAN_SHA = '1b4ea3bb65bd4cfdffc13a610fee010d568a5837e632701d4b0d265e6ba66e86'
V2_RESULT = 'results/vision_center_v2_001/execution/pilot/result.json'
UNCHANGED = ('src/hci/vision_center.py', 'src/hci/vision_center_v2.py', 'src/hci/recovery_v1_oracle.py', 'src/hci/recovery_v1.py', 'src/hci/simulation.py', 'src/hci/policies.py', 'src/hci/vision_v1.py')     # must equal the hashes of the approved v2 plan
OFFLINE = 'results/vision_center_v2_c1_001/preparation/offline_replay_summary_r4.json'     # fourth replay: final code (recording change, Unix line endings); replays 1-3 are kept; all four have identical rows
AUDIT_SEEDS = Z.AUDIT_SEEDS
USED = {**Z.USED, 'vision_center_v2': Z.MANIFEST}
CODE = sorted({*Z.CODE, 'src/hci/vision_center_c1.py', 'src/hci/vision_center_v2s.py', 'src/hci/trajectory_audit.py', 'tests/test_trajectory_audit.py', 'scripts/34_vision_center_c1.py', 'scripts/vision_center_c1/run.sh', MANIFEST, REGISTRY, Z.MANIFEST})
SERVER_READS = sorted({*CODE, *Z.SERVER_READS, *(p['path'] + '/' + f for p in POLICIES.values() for f in ('weights.pt', 'policy_meta.json'))})
LEDGER = 'artifacts/ledger/vision_center_c1.json'
DRIVE = 'Robot Harm/results/vision_center_v2_c1_001/'
RESERVE_USD = 0.30; MARGIN = 1.25
STATUS = {'phase_c1': 'PREPARATION_ONLY', 'paid_compute': 'NOT_AUTHORIZED', 'new_scenes_51100_51299': 'NOT_USED', 'vision_center_v2': 'EXECUTED_ONCE (V2_GATE_PASS on development scenes); code, records and result unchanged',
          'base_policies': 'FROZEN: Vision-V1 train seed 1 and train seed 2, 80k EMA, unchanged; nothing is adapted to train seed 2', 'new_training': 'NOT_PLANNED', 'calibration': 'NOT_RUN, LOCKED', 'final_test': 'NOT_RUN, LOCKED'}


def registry():
    reg = read_json(REGISTRY); arms = [a['id'] for a in reg['arms'] if a['included']]
    if arms[0] != 'C0' or not set(arms) <= set(C1.ARMS) or [a for a in C1.BASE_ARMS if a not in arms] or len(set(arms)) != len(arms): raise ValueError('the arm registry is not C0 first, the nine base arms, optionally VISION_CENTER_V2S')
    if [a['id'] for a in reg['arms']] != list(C1.ARMS): raise ValueError('the arm registry does not list hci.vision_center_c1.ARMS in order')
    return reg, tuple(arms)


def used_ledger():
    used = Z.used_ledger()
    for k, v in read_json(Z.MANIFEST)['splits'].items(): used[f'vision_center_v2/{k}'] = list(v)
    return used


def check_policy_2(problems):
    """Train seed 2 is the file recorded in the Vision-V1 backup and evaluated once on the selection scenes; nothing else."""
    p = POLICIES['train_seed_2']; d = Path(p['path']); m = read_json(d / 'policy_meta.json'); listed = dict(reversed(l.split(None, 1)) for l in Path(S2_RECORD['backup_list']).read_text().splitlines() if l.strip())
    sel = read_json(S2_RECORD['select_result']); facts = {'weights_sha256_now': sha256(d / 'weights.pt'), 'policy_meta_sha256_now': sha256(d / 'policy_meta.json'), 'meta_weights_sha256': m['weights_sha256'], 'backup_list_weights_sha256': listed.get(p['path'] + '/weights.pt'),
                                                         'backup_list_meta_sha256': listed.get(p['path'] + '/policy_meta.json'), 'select_result_weights_sha256': sel['weights_sha256'], 'select_success_once': f"{sel['success_once']}/{sel['n']}",
                                                         'selected_in_vision_v1': read_json(S2_RECORD['selection'])['selected_train_seed'] == 2}
    if len({facts[k] for k in ('weights_sha256_now', 'meta_weights_sha256', 'backup_list_weights_sha256', 'select_result_weights_sha256')} | {p['weights.pt']}) != 1: problems.append('train seed 2: the weights are not the recorded Vision-V1 weights')
    if not (facts['policy_meta_sha256_now'] == facts['backup_list_meta_sha256'] == p['policy_meta.json']): problems.append('train seed 2: policy metadata differs from the backup record')
    m1 = read_json(Path(POLICIES['train_seed_1']['path']) / 'policy_meta.json')
    if (m.get('train_seed'), m.get('iteration'), m.get('weights'), m.get('upstream_commit')) != (2, 80000, 'EMA', UPSTREAM_COMMIT) or m['source_checkpoint_sha256'] != p['source_checkpoint_sha256']: problems.append('train seed 2: not the train seed 2, 80k EMA export')
    if any(m.get(k) != m1.get(k) for k in ('observation_schema', 'architecture_id', 'env_id', 'control_mode', 'horizon', 'robot_uid')) or m['args']['demo_path'] != m1['args']['demo_path'] or m['args']['num_demos'] != m1['args']['num_demos']:
        problems.append('train seed 2 does not share the architecture, observation schema, environment and demonstrations of train seed 1')
    return facts


def cost_estimate(arms):
    """From the timing measured in Phase B (one host). The budget gate on the server uses the times it measures itself."""
    au = read_json('results/vision_center_v2_001/execution/audit/audit.json'); res = read_json(V2_RESULT); price = 0.4604; vis = len([a for a in arms if a in C1.VISION_ARMS]); n = C1.RANGE[1] - C1.RANGE[0] + 1
    t = {'seconds_per_env_step': au['seconds_per_env_step'], 'seconds_per_reading': au['seconds_per_reading'], 'seconds_per_memory_step': 2.5 * au['seconds_per_memory_step']}
    worst = C1.projected_cost(**t, usd_per_hour=price, spent_usd=0.30, cap_usd=float('inf'), scene_runs=2 * n, arms=len(arms), vision_arms=vis, reserve_usd=RESERVE_USD, margin=MARGIN)
    per4 = res['wall_seconds'] / 200; exp = 2 * n * (per4 + (len(arms) - 4) * 2.5 + 3.0) / 3600
    return {'status': 'ESTIMATE; nothing was run. The budget gate on the server uses measured times, not these numbers', 'basis': {'host': 'Phase B: RTX 4090, AMD Ryzen 7 7800X3D, US$0.4604/h, thread limit 1', **{k + '_phase_b': au[k] for k in ('seconds_per_env_step', 'seconds_per_reading', 'seconds_per_memory_step')},
                      'memory_step_factor_assumed': 2.5, 'memory_step_note': 'each trunk step updates the original v2 memory with its own fits and all other memories from one shared set of fits; the factor is an assumption until the audit measures it',
                      'seconds_per_scene_4_arms_phase_b': per4, 'seconds_per_extra_arm_assumed': 2.5},
            'expected_minutes': {'install_checks_placement_audit': [25, 40], 'two_passes': [round(exp * 60 * 0.9), round(exp * 60 * 1.3)], 'pack_download': [10, 20]}, 'expected_total_usd': [round(0.35 + exp * 0.9 * price, 2), round(0.60 + exp * 1.3 * price, 2)],
            'worst_case_gate_projection_usd': worst['projected_total_usd'], 'worst_case_hours': worst['projected_remaining_hours'], 'worst_case_inputs': worst['inputs'],
            'proposed_cap_usd': 6.0, 'proposed_cap_is': 'a proposal, not an approval', 'storage_gb': [3.5, 6.0], 'storage_note': 'both cameras: every step up to the trigger, the control frames of the vision arms, and after the trigger the frames of C0 and of every arm that moved under hci.vision_center_c1.FRAME_POLICY (estimated from the Phase B records: about 2.7 GB before the trigger and 0.6 GB after it at 67 kB per compressed frame); the download adds the per-GB transfer price',
            'gate_rule': 'the budget step projects the whole run from the times measured in the audit with every scene at its bound, margin 1.25, reserve US$0.30, and refuses above the approved cap; spend so far is computed from the session clock; '
                         'before every scene the run stops INCOMPLETE if one more scene at its bound could pass the cap. Scenes, arms and bounds are never reduced'}


def analysis(arms):
    return {'unit': 'scene, within policy; the 200 scenes are shared by the two policies and are never counted as 400 independent observations', 'outcome': 'success_once_absorbing, all 200 scenes in every denominator',
            'primary': {'alpha_familywise': C1.ALPHA, 'multiplicity': 'Holm over H_gen and H_mem', 'H_gen': 'train seed 2: VISION_CENTER_V2 above C0; exact conditional paired test on the discordant scenes, one-sided',
                        'H_mem': 'VISION_CENTER_V2 above V2_SINGLE_FRAME with both policies; one exact conditional paired test per policy, combined by intersection-union (the larger of the two p-values)', 'implementation': 'hci.vision_center_c1.primary_tests'},
            'descriptive': ['per policy: success of every arm with its Wilson interval', 'per policy: every arm against C0 (rescue, harm, net)', 'per policy: v2 against v1, v2 against each leave-one-out arm, V2_RULES_OFF against v2, v2s against v2, oracle against v2',
                            'per policy: decisions by reason, moves by source, memory errors, unnecessary moves, target pushed, oracle rescues recovered', 'between policies: v2 net gain per policy, their difference on the same scenes with a scene bootstrap, as a range'],
            'safety_per_policy': {'arms': [a for a in C1.SAFETY_ARMS if a in arms], 'thresholds': C1.SAFETY, 'meaning': 'development criteria for going on; they do not replace a calibration and certify no harm rate', 'on_failure': 'reported as not met; no re-run, no tuning'},
            'not_claimed': ['a calibrated harm rate', 'the effect of any single v2 rule (V2_RULES_OFF removes a bundle)', 'a between-seed variance (two seeds)', 'transfer to real images', 'anything beyond StackCube']}


def cmd_plan(a):
    c = load_config(CONFIG); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); problems = []
    if (out / 'plan.json').exists(): raise SystemExit(f'Refused overwrite: {out / "plan.json"}')
    try: seeds = C1.pilot_seeds(read_json(MANIFEST), used_ledger())
    except ValueError as e: problems.append(str(e)); seeds = []
    try: reg, arms = registry()
    except ValueError as e: problems.append(str(e)); reg, arms = {}, ()
    p1 = D.check_policy(problems); p2 = check_policy_2(problems); vp = read_json(V2_PLAN)
    if sha256(V2_PLAN) != V2_PLAN_SHA or read_json(V2_RESULT)['followup_gate']['label'] != 'V2_GATE_PASS': problems.append('the v2 record is not the executed V2_GATE_PASS run of plan 1b4ea3bb')
    changed = [f for f in UNCHANGED if sha256(f) != vp['code'][f]]
    if changed: problems.append(f'files that must stay as in the approved v2 plan have changed: {changed}')
    if Path(LEDGER).exists(): problems.append('a ledger for the C1 scenes already exists')
    crlf = [f for f in SERVER_READS if Path(f).suffix in ('.py', '.sh', '.yaml') and b'\r' in Path(f).read_bytes()]
    if crlf: problems.append(f'Windows line endings in files executed on the Linux server: {crlf}')
    if [c['policy'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')] != [2, 8, 16] or c['env']['horizon'] != 200 or c['env']['control_mode'] != 'pd_ee_delta_pos' or c['env']['robot_uid'] != 'panda_wristcam': problems.append('config is not the fixed setting')
    off = read_json(OFFLINE) if Path(OFFLINE).exists() else None
    if off is None: problems.append('offline replay summary missing')
    elif not off['v2_reproduced_exactly'] or off['code_sha256'] != {k: sha256(k) for k in off['code_sha256']}: problems.append('the offline replay does not reproduce the recorded v2 decisions, or was made with other code')
    if off is not None and (C1.SAFETY_ARM in arms) != bool(off['v2s']['technically_valid']): problems.append('the arm registry and the offline verdict on v2s disagree')
    ce = cost_estimate(arms) if arms else None; n = len(seeds)
    plan = {'work_id': WORK, 'stage': 'Vision-CENTER v2, Phase C1: component ablation and second policy seed (preparation)', 'created_utc': datetime.now(timezone.utc).isoformat(), 'pass': not problems, 'problems': problems, 'status': STATUS,
            'authorization': {'paid_compute_authorized': False, 'proposed_cap_usd': None if ce is None else ce['proposed_cap_usd'], 'proposed_cap_is': 'a proposal, explicitly not an approval; earlier approvals are not inherited',
                              'needed': 'a separate explicit user instruction that states a total USD cap AND names the SHA256 of this plan file',
                              'required_file': 'authorization.json in the run folder, written only after such an instruction: {"approved_by": "user", "paid_compute_authorized": true, "cap_usd": <number>, "plan_sha256": "<sha256 of this plan>"}'},
            'questions': ['generalisation: does the unchanged v2 keep its paired gain and zero harm with a second base policy, and on new scenes with the first?',
                          'components: how much of the gain over v1 is temporal memory, as opposed to the base-camera estimator on a single frame and to the changed abstention rules?', 'safety: what does the separate candidate v2s cost and prevent next to v2?'],
            'data_role': 'development. Not calibration, not Test.',
            'privileged_information': {'vision_arms': 'none', 'offline_labels': 'cube positions are labels in after-the-fact columns of the records', 'center_oracle': 'unchanged reference arm; the only control code that reads cube positions',
                                       'project_rule': 'AGENTS.md: "모니터/정책에 객체 정답 pose, task success, 미래 라벨을 입력하지 않습니다."'},
            'repository': {'head': D.git('rev-parse', 'HEAD'), 'working_tree_changes_at_plan_time': D.git('status', '--porcelain').splitlines()},
            'builds_on': {'v2_plan': V2_PLAN, 'v2_plan_sha256': V2_PLAN_SHA, 'v2_execution_commit': 'fd23aa2', 'unchanged_files_equal_to_v2_plan': {f: sha256(f) for f in UNCHANGED}},
            'policies': {'train_seed_1': {**POLICIES['train_seed_1'], 'checks': p1}, 'train_seed_2': {**POLICIES['train_seed_2'], 'checks': p2}, 'order': list(C1.POLICIES),
                         'rule': 'frozen; not retrained, not fine-tuned, no other checkpoint; no constant or rule of any arm is adapted to either policy'},
            'config': {'path': CONFIG, 'file_sha256': sha256(CONFIG), 'config_sha256': digest(c), 'env': c['env'], 'policy': c['policy'], 'audit': c['audit']},
            'seed_registry': {'path': MANIFEST, 'sha256': sha256(MANIFEST), 'set': C1.SET, 'range': list(C1.RANGE), 'n': n, 'first_last': seeds[:1] + seeds[-1:], 'shared_by_both_policies': True,
                              'seed_level_check': 'no overlap with splits.json (audit, train, dev, calibration, test), policy_eval_splits.json, vision_v1_splits.json, recovery_v1_splits.json, vision_center_splits.json, vision_center_v2_splits.json, '
                                                  'the 550 demonstration scene seeds, or the scenes of Stage 2A / 2B / 2D / H1; checked by this command (a clash fails the plan)',
                              'placement_check': 'server step `scenes`: the initial cube placements of the 200 seeds against every placement on record and against each other; an identical placement fails the step',
                              'untouched_sets': ['calibration 30000-30399', 'test 90000-90399'], 'no_replacement': 'no scene is replaced or dropped for any reason'},
            'arm_registry': {'path': REGISTRY, 'sha256': sha256(REGISTRY), 'arms': list(arms), 'definitions': reg.get('arms'), 'v2s_included': C1.SAFETY_ARM in arms},
            'semantics': {'trigger': {'rule': 'the oracle pilot trigger, unchanged (hci.recovery_v1_oracle.trigger_rule)', **O.TRIGGER}, 'v2_memory': V2.MEMORY, 'v2_decision': V2.DECISION, 'single_frame': C1.SINGLE_FRAME,
                          'memory_variants': C1.MEMORY_VARIANTS, 'rules_off_removed_as_one_bundle': list(C1.RULES_OFF_REMOVED), 'v2s': S.SAFETY, 'move_bounds': dict(O.BOUNDS),
                          'abstain': 'nothing is touched: the episode is the uninterrupted one (checked against the trunk in every scene, for every arm)', 'noise': "repeat 0 only; every inference is keyed by seed_for(scene, physical step, 0, 'dp')",
                          'restore': 'reset(seed) + prefix action replay; the restored runtime is compared with the snapshot before every arm',
                          'memories': 'built in the trunk from the policy\'s own observation; they never touch the policy, its queue or its observation history; the mirrored memory must equal the original v2 memory at every trigger'},
            'analysis': analysis(arms), 'offline_replay': off,
            'audit': {'seeds': list(AUDIT_SEEDS), 'policies': list(C1.POLICIES), 'does': ['every arm with every technical check of the run, with both policies', 'seconds per environment step, per vision reading and per memory step for the budget gate'], 'outcomes': 'decide nothing'},
            'run': {'order': ['clock', 'scenes', 'audit', 'budget', 'run (train_seed_1, then train_seed_2)'], 'ledger': LEDGER + ', created exclusively before any rollout on a new scene; one ledger for both passes; no resume, no second pass',
                    'on_error': 'the run stops and stays INCOMPLETE; no scene is dropped or replaced; a partial run has no summary',
                    'stored': 'one record per scene and policy, per-step ground-truth traces (logging), both cameras\' frames of every step up to the trigger, the frames of every vision arm at the trigger and during control, and after the trigger the frames given by FRAME_POLICY; per arm a branch id, the snapshot digest, the first open command after the move and the success step', 'frame_policy': C1.FRAME_POLICY,
                    'thread_limit': 'OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS, BLIS_NUM_THREADS = 1 before every Python process (run.sh exports them)'},
            'volume': {'policies': 2, 'scenes': n, 'scene_runs': 2 * n, 'arms': len(arms), 'vision_arms': len([x for x in arms if x in C1.VISION_ARMS]), 'audit_scene_runs': 2 * len(AUDIT_SEEDS), 'steps_per_scene_bound': 200 + len(arms) * 200,
                       'env_steps_bound_total': 2 * n * (200 + len(arms) * 200), 'readings_per_scene_bound': len([x for x in arms if x in C1.VISION_ARMS]) * 13},
            'cost_estimate': ce,
            'technical_stop_conditions': ['no authorization, or one that names another plan', 'a code file, a policy file, the seed registry or the arm registry differs from this plan', 'tests fail on the server; CUDA or Vulkan check fails (the host is discarded before any scene)',
                                          'an identical initial placement for a new seed', 'the restored runtime differs from the snapshot; C0 or an abstaining arm does not reproduce the trunk', 'ground truth reaches a vision arm (oracle read counter)',
                                          'the mirrored memory differs from hci.vision_center_v2.Memory at a trigger', 'policy inference during a control part, or a queue left at the hand-back', 'the budget projection exceeds the cap before the run, after five scenes of a pass, or before any scene',
                                          'the ledger already exists', 'any error in a scene'],
            'command_order': ['local: tests, offline replay, `plan`, `dry-run`', 'user: explicit approval naming a total USD cap and this plan hash -> authorization.json', 'server: install', 'server: run.sh env', 'server: run.sh scenes', 'server: run.sh audit',
                              'server: run.sh run (budget gate first)', 'server: run.sh pack', 'local: download, hash checks, `summarize` from the raw records', 'local: delete only the instances created for this run', f'Drive: report and results to {DRIVE}'],
            'outputs': {'local': 'results/vision_center_v2_c1_001/', 'drive': DRIVE, 'report_md': 'not modified unless asked'},
            'files_read_on_the_server': SERVER_READS, 'code': {f: sha256(f) for f in CODE}, 'upstream_commit': UPSTREAM_COMMIT,
            'not_in_this_work': ['policy training or fine-tuning', 'any change to v1, v2 or the oracle', 'any change of constants, rules or criteria after an outcome', 'a second pass', 'calibration', 'final Test', 'any other task']}
    write_json(out / 'plan.json', plan); print(json.dumps({'pass': plan['pass'], 'problems': problems, 'arms': list(arms), 'cost': None if ce is None else {k: ce[k] for k in ('expected_total_usd', 'worst_case_gate_projection_usd', 'proposed_cap_usd')}}, indent=1))
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
    for pid, p in POLICIES.items():
        for f in ('weights.pt', 'policy_meta.json'):
            if sha256(Path(p['path']) / f) != p[f] or plan['policies'][pid][f] != p[f]: bad.append(f'{pid} {f}')
    reg, arms = registry()
    if list(arms) != plan['arm_registry']['arms'] or sha256(MANIFEST) != plan['seed_registry']['sha256'] or sha256(REGISTRY) != plan['arm_registry']['sha256']: bad.append('arm or seed registry')
    if bad: raise SystemExit(f'Refused: these differ from the plan: {bad}')
    return plan, load_config(CONFIG), arms


# ---------------------------------------------------------------- session clock: spend is computed, not supplied
def clock_path(run): return Path(run) / 'session_clock.json'


def cmd_clock(a):
    run = Path(a.run); D.require_authorization(run, a.plan); p = clock_path(run)
    if p.exists(): raise SystemExit(f'{p} exists: the session clock is written once')
    if not (a.usd_per_hour > 0 and a.spent_before_usd >= 0): raise SystemExit('Refused: the hourly price must be positive and the earlier spend not negative')
    with open(p, 'x', encoding='utf-8') as f: json.dump({'started_utc': datetime.now(timezone.utc).isoformat(), 'started_epoch': time.time(), 'usd_per_hour': float(a.usd_per_hour), 'spent_before_usd': float(a.spent_before_usd), 'plan_sha256': sha256(a.plan),
                                                         'note': 'usd_per_hour is the all-in hourly price of this instance as shown by the provider; spent_before_usd is what this work spent before this instance existed'}, f, indent=1)
    print(p.read_text())


def spent_now(run, plan_path, now=None):
    p = clock_path(run)
    if not p.exists(): raise SystemExit('no session clock: `clock` comes first')
    k = read_json(p)
    if k['plan_sha256'] != sha256(plan_path): raise SystemExit('the session clock belongs to another plan')
    el = (time.time() if now is None else now) - k['started_epoch']
    if el < 0: raise SystemExit('the session clock is in the future')
    return k['spent_before_usd'] + el / 3600 * k['usd_per_hour'], k


def projection(run, auth, plan_path, arms, scene_runs, now=None):
    au = read_json(Path(run) / 'audit' / 'audit.json')
    if not au['pass'] or au['plan_sha256'] != sha256(plan_path): raise SystemExit('audit missing, failed or of another plan')
    spent, k = spent_now(run, plan_path, now)
    d = C1.projected_cost(seconds_per_env_step=au['seconds_per_env_step'], seconds_per_reading=au['seconds_per_reading'], seconds_per_memory_step=au['seconds_per_memory_step'], usd_per_hour=k['usd_per_hour'], spent_usd=spent, cap_usd=float(auth['cap_usd']),
                          scene_runs=scene_runs, arms=len(arms), vision_arms=len([x for x in arms if x in C1.VISION_ARMS]), reserve_usd=RESERVE_USD, margin=MARGIN)
    d.update(status='OK' if d['fits_cap'] else 'BUDGET_BLOCKED', plan_sha256=sha256(plan_path), spent_from_session_clock_usd=spent); return d


def cmd_budget(a):
    run = Path(a.run); auth = D.require_authorization(run, a.plan); plan, c, arms = bound(a.plan); d = projection(run, auth, a.plan, arms, plan['volume']['scene_runs'])
    write_json(run / 'budget_gate.json', d); print(json.dumps(d, indent=1))
    if not d['fits_cap']: raise SystemExit(2)


def cmd_scenes(a):
    run = Path(a.run); D.require_authorization(run, a.plan); plan, c, arms = bound(a.plan); seeds = C1.pilot_seeds(read_json(MANIFEST), used_ledger())
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


def runtime_for(c, pid, a):
    from hci import vision_v1 as V
    return V.runtime(c, POLICIES[pid]['path'], a.upstream, a.device)


def cmd_audit(a):
    run = Path(a.run); D.require_authorization(run, a.plan); plan, c, arms = bound(a.plan); out = run / 'audit'; out.mkdir(parents=True, exist_ok=False); recs = []; facts = {}
    for pid in C1.POLICIES:
        r = runtime_for(c, pid, a)
        try:
            facts[pid] = X.controller_facts(r)
            for s in AUDIT_SEEDS:
                rec, traces, frames = C1.collect_scene(r, s, c, probe=X.gt_probe, arms=arms, role='vision_center_v2_c1_audit', policy_id=pid); (out / pid).mkdir(exist_ok=True); Z.save_scene(out / pid, s, rec, traces, frames); recs.append(rec)
                print(json.dumps({'audit_scene': s, 'policy': pid, 'triggered': rec['triggered'], 'wall_seconds': rec['wall_seconds']}), flush=True)
        finally:
            r.env.close()
    tm = Z.timing(recs); trig = sum(1 for x in recs if x['triggered']); ok = bool(all(f['pass'] for f in facts.values()) and trig >= 3 and tm['seconds_per_reading'] is not None and tm['seconds_per_memory_step'] is not None)
    report = {'pass': ok, 'status': 'OK' if ok else 'TECHNICAL_BLOCKED', 'controller': facts, 'triggered_scene_runs': trig, 'all_arm_procedure_completed_on': len(recs), 'arms': list(arms), **tm, 'plan_sha256': sha256(a.plan), 'runtime': runtime_info(),
              'scope': 'Technical audit on already-used scenes with both policies. Outcomes here decide nothing.'}
    write_json(out / 'audit.json', report); print(json.dumps({k: report[k] for k in ('pass', 'status', 'seconds_per_env_step', 'seconds_per_reading', 'seconds_per_memory_step')}, indent=1))
    if not ok: raise SystemExit(2)


def one_scene_bound_usd(run, arms, k):
    au = read_json(Path(run) / 'audit' / 'audit.json'); vis = len([x for x in arms if x in C1.VISION_ARMS])
    sec = (200 + len(arms) * 200) * au['seconds_per_env_step'] + 200 * au['seconds_per_memory_step'] + vis * 13 * au['seconds_per_reading']
    return sec * MARGIN / 3600 * k['usd_per_hour']


class BudgetDeadline(BaseException):
    """The session clock reached the cap minus the reserve. BaseException, so no `except Exception` in a scene swallows it."""


CURRENT = {}            # where the run is, for the stop record: policy, scene, scenes finished


def deadline_seconds_left(k, cap, now=None):
    """Seconds until the session clock reaches cap - reserve (spend before this instance included)."""
    end = k['started_epoch'] + (cap - RESERVE_USD - k['spent_before_usd']) / k['usd_per_hour'] * 3600
    return end - (time.time() if now is None else now)


def arm_deadline(k, cap):
    """A hard stop inside a scene, not a projection: SIGALRM at the moment the clock reaches cap - reserve."""
    import signal
    if not hasattr(signal, 'setitimer'): raise SystemExit('Refused: the budget deadline needs signal.setitimer (the Linux server)')
    left = deadline_seconds_left(k, cap)
    if left <= 0: raise SystemExit('BUDGET_STOP: the session clock is already at the cap minus the reserve')

    def fire(*_): raise BudgetDeadline()
    signal.signal(signal.SIGALRM, fire); signal.setitimer(signal.ITIMER_REAL, left); return left


def disarm_deadline():
    import signal
    if hasattr(signal, 'setitimer'): signal.setitimer(signal.ITIMER_REAL, 0)


def passes(r, pid, seeds, c, out, arms, run, plan_path, cap, runs_left_after, plan_hash):
    """One pass (one policy) over `seeds`. Before every scene: spend from the session clock plus one scene at its bound plus the reserve must fit the cap."""
    started = time.perf_counter(); done = 0
    for s in seeds:
        CURRENT.update(policy=pid, scene=int(s), scenes_finished_in_pass=done)
        spent, k = spent_now(run, plan_path)
        if spent + one_scene_bound_usd(run, arms, k) + RESERVE_USD > cap:
            write_json(out / 'budget_stop.json', {'before_scene': int(s), 'scenes_finished': done, 'spent_from_session_clock_usd': spent, 'cap_usd': cap, 'plan_sha256': plan_hash}); raise SystemExit('BUDGET_STOP: one more scene could pass the cap; the run stops INCOMPLETE and is backed up')
        try:
            rec, traces, frames = C1.collect_scene(r, s, c, probe=X.gt_probe, arms=arms, role=C1.ROLE, policy_id=pid); rec['plan_sha256'] = plan_hash; Z.save_scene(out, s, rec, traces, frames)
        except Exception as e:
            import traceback
            write_json(out / 'scenes' / f'{s}.error.json', {'scene': int(s), 'policy': pid, 'completed': False, 'error': repr(e), 'traceback': traceback.format_exc(), 'scenes_finished_before': done}); raise
        done += 1; print(f"{C1.ROLE} policy={pid} scene={s} triggered={rec['triggered']} done={done}/{len(seeds)} wall={rec['wall_seconds']:.0f}s", flush=True)      # outcomes are not printed during a pass
        if done == 5:
            el = time.perf_counter() - started; spent, k = spent_now(run, plan_path); total = spent + ((len(seeds) - 5 + runs_left_after) * el / 5 * MARGIN) / 3600 * k['usd_per_hour'] + RESERVE_USD
            gate = {'elapsed_seconds_five_scenes': el, 'projected_total_usd': total, 'scene_runs_still_to_do': len(seeds) - 5 + runs_left_after, 'cap_usd': cap, 'fits_cap': bool(total <= cap), 'inputs': 'time and money only; no outcome is read', 'plan_sha256': plan_hash}
            write_json(out / 'budget_after_five_scenes.json', gate)
            if not gate['fits_cap']: raise SystemExit('BUDGET_BLOCKED after five scenes: the run stops INCOMPLETE and is backed up; the experiment is not reduced')
    return time.perf_counter() - started


def load_records(d, seeds, c, arms, pid=None, plan_hash=None):
    d = Path(d); errors = sorted(p.name for p in (d / 'scenes').glob('*.error.json'))
    if errors: raise SystemExit(f'INCOMPLETE: error files present {errors}; a partial run is not summarised')
    recs = []
    for s in seeds:
        p = d / 'scenes' / f'{s}.json'
        if not p.exists(): raise SystemExit(f'INCOMPLETE: no record for scene {s}; a partial run is not summarised')
        rec = read_json(p); C1.validate_scene(rec, c)
        if list(rec['arms']) != list(arms): raise SystemExit(f'scene {s} does not carry the planned arms')
        if rec.get('completed') is not True or rec.get('scene') != int(s) or rec.get('role') != C1.ROLE or (pid is not None and rec.get('policy_id') != pid) or (plan_hash is not None and rec.get('plan_sha256') != plan_hash): raise SystemExit(f'scene {s}: the record is not a completed {C1.ROLE} record of policy {pid}')
        if sha256(d / rec['trace']) != rec['trace_sha256']: raise SystemExit(f'trace of scene {s} changed')
        recs.append(rec)
    return recs


def cmd_run(a):
    run = Path(a.run); auth = D.require_authorization(run, a.plan); plan, c, arms = bound(a.plan); h = sha256(a.plan); cap = float(auth['cap_usd'])
    for name in ('scenes.json', 'audit/audit.json', 'budget_gate.json'):
        if not (run / name).exists(): raise SystemExit(f'{name} missing: placement check, audit and budget gate come first')
    sc = read_json(run / 'scenes.json'); au = read_json(run / 'audit' / 'audit.json'); bg = read_json(run / 'budget_gate.json')
    if not (sc['pass'] and sc['plan_sha256'] == h and au['pass'] and au['plan_sha256'] == h and list(au['arms']) == list(arms) and bg['fits_cap'] and bg.get('plan_sha256') == h): raise SystemExit('placement check, audit or budget gate not passed for this plan')
    if Path(LEDGER).exists(): raise SystemExit(f'{LEDGER} exists: the C1 scenes already have a recorded use')
    outs = {pid: run / f'pass_{pid}' for pid in C1.POLICIES}
    if any(o.exists() for o in outs.values()): raise SystemExit('a pass folder exists: the run is never resumed or repeated')
    seeds = C1.pilot_seeds(read_json(MANIFEST), used_ledger()); now = projection(run, auth, a.plan, arms, 2 * len(seeds))
    if not now['fits_cap']: print(json.dumps(now, indent=1)); raise SystemExit('BUDGET_BLOCKED at the start of the run: the set stays unused')
    write_json(run / 'budget_gate_at_start.json', now); Path(LEDGER).parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, 'x', encoding='utf-8') as f:                         # exclusive: a second run cannot write it
        json.dump({'set': C1.SET, 'used_utc': datetime.now(timezone.utc).isoformat(), 'session': run.as_posix(), 'plan_sha256': h, 'seed_manifest_sha256': plan['seed_registry']['sha256'], 'arm_registry_sha256': plan['arm_registry']['sha256'],
                   'policies': {pid: POLICIES[pid]['weights.pt'] for pid in C1.POLICIES}, 'arms': list(arms), 'note': 'Single run: one pass per policy on the same 200 scenes. Written before any outcome existed. No resume, no second pass.'}, f, indent=1)
    with open(run / 'ledger.json', 'x', encoding='utf-8') as f: f.write(Path(LEDGER).read_text(encoding='utf-8'))      # the copy that travels with the run folder
    walls = run_passes(run, a, c, seeds, arms, cap, h, outs)
    cmd_summarize(argparse.Namespace(run=str(run), plan=a.plan, out=str(run / 'result.json'), walls=walls))


def run_passes(run, a, c, seeds, arms, cap, h, outs):
    """Both passes under one hard deadline. A deadline stop writes budget_stop.json and leaves the run INCOMPLETE."""
    walls = {}; _, k = spent_now(run, a.plan); left = arm_deadline(k, cap); write_json(run / 'budget_deadline.json', {'seconds_left_at_start': left, 'cap_usd': cap, 'reserve_usd': RESERVE_USD, 'plan_sha256': h})
    try:
        for i, pid in enumerate(C1.POLICIES):
            out = outs[pid]; (out / 'scenes').mkdir(parents=True); r = runtime_for(c, pid, a)
            try: walls[pid] = passes(r, pid, seeds, c, out, arms, run, a.plan, cap, (len(C1.POLICIES) - 1 - i) * len(seeds), h)
            finally: r.env.close()
    except BudgetDeadline:
        spent, _ = spent_now(run, a.plan)
        write_json(run / 'budget_stop.json', {'kind': 'deadline', **CURRENT, 'spent_from_session_clock_usd': spent, 'cap_usd': cap, 'reserve_usd': RESERVE_USD, 'plan_sha256': h})
        raise SystemExit('BUDGET_STOP: the session clock reached the cap minus the reserve during a scene; the run stops INCOMPLETE and is backed up')
    finally:
        disarm_deadline()
    return walls


def check_run_gates(run, plan, h, arms):
    """Every step of the approved procedure passed for this plan, and nothing stopped the run."""
    run = Path(run)
    stops = sorted(str(p.relative_to(run)) for p in run.rglob('budget_stop.json'))
    if stops: raise SystemExit(f'INCOMPLETE: budget stop {stops}; a partial run is not summarised')
    for name in ('scenes.json', 'audit/audit.json', 'budget_gate.json', 'budget_gate_at_start.json', 'ledger.json'):
        if not (run / name).exists() or read_json(run / name).get('plan_sha256') != h: raise SystemExit(f'{name} missing or of another plan: the run did not go through the approved steps')
    sc, au, bg, b0, lg = (read_json(run / n) for n in ('scenes.json', 'audit/audit.json', 'budget_gate.json', 'budget_gate_at_start.json', 'ledger.json'))
    if not (sc.get('pass') is True and au.get('pass') is True and list(au.get('arms', [])) == list(arms) and bg.get('fits_cap') is True and b0.get('fits_cap') is True):
        raise SystemExit('a gate of this run did not pass (placement, audit or budget): it is not summarised')
    if lg.get('set') != C1.SET or list(lg.get('arms', [])) != list(arms) or lg.get('policies') != {pid: POLICIES[pid]['weights.pt'] for pid in C1.POLICIES} or lg.get('arm_registry_sha256') != plan['arm_registry']['sha256'] or lg.get('seed_manifest_sha256') != plan['seed_registry']['sha256']:
        raise SystemExit('the ledger of this run does not match the plan')


def cmd_summarize(a):
    run = Path(a.run); D.require_authorization(run, a.plan); plan, c, arms = bound(a.plan); h = sha256(a.plan); seeds = C1.pilot_seeds(read_json(MANIFEST), {})
    check_run_gates(run, plan, h, arms)
    recs = {pid: load_records(run / f'pass_{pid}', seeds, c, arms, pid, h) for pid in C1.POLICIES}; res = C1.summarize(recs, seeds, arms)
    reads = sum(x['oracle_reads'] for rs in recs.values() for r in rs for x in r['cells'] if x['arm'] != 'CENTER_ORACLE' and not x['no_trigger'])
    res.update(set=C1.SET, seeds=[seeds[0], seeds[-1]], policies={pid: {'path': POLICIES[pid]['path'], 'weights_sha256': POLICIES[pid]['weights.pt']} for pid in C1.POLICIES}, plan_sha256=sha256(a.plan),
               rules_sha256={f: sha256(f) for f in ('src/hci/vision_center.py', 'src/hci/vision_center_v2.py', 'src/hci/vision_center_c1.py', 'src/hci/vision_center_v2s.py')}, oracle_reads_outside_center=reads, wall_seconds=getattr(a, 'walls', None), runtime=runtime_info())
    write_json(a.out, res); p = res['primary']
    print(json.dumps({'success': {pid: {k: v['k'] for k, v in res['per_policy'][pid]['success'].items()} for pid in C1.POLICIES}, 'H_gen': {k: p['H_gen'][k] for k in ('p_one_sided', 'rejected_null')},
                      'H_mem': {k: p['H_mem'][k] for k in ('p_intersection_union', 'rejected_null')}, 'safety_all_met': res['safety_all_met']}, indent=1))


def cmd_check_auth(a):
    D.require_authorization(Path(a.run), a.plan); print('authorization names this plan')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--out', required=True)
    q = s.add_parser('dry-run'); q.add_argument('--plan', required=True); q.add_argument('--run', default=RUN_DEFAULT); q.add_argument('--out')
    for name in ('clock', 'scenes', 'audit', 'budget', 'run'):
        q = s.add_parser(name); q.add_argument('--plan', required=True); q.add_argument('--run', default=RUN_DEFAULT)
        if name in ('audit', 'run'): q.add_argument('--upstream', default='third_party/ManiSkill'); q.add_argument('--device', default='cuda')
        if name == 'clock': q.add_argument('--usd-per-hour', type=float, required=True); q.add_argument('--spent-before-usd', type=float, required=True)
    q = s.add_parser('check-auth'); q.add_argument('--plan', required=True); q.add_argument('--run', default=RUN_DEFAULT)
    q = s.add_parser('summarize'); q.add_argument('--run', required=True); q.add_argument('--plan', required=True); q.add_argument('--out', required=True)
    a = p.parse_args(); globals()['cmd_' + a.cmd.replace('-', '_')](a)


if __name__ == '__main__':
    main()
