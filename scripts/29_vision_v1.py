"""Vision-V1 base-policy upgrade (work id vision_v1_001): plan, seed gates, evaluation drivers.
See amendments/AMENDMENT_012 (draft). PREPARATION ONLY at the time of writing: nothing below `dry-run` has been run.

  plan            local: hashes of data, legacy policy, pretrained file, code; the new seed manifest; rules; authorisation = false
  dry-run         local: volume, files, permissions; starts nothing
  scenes          server: initial cube placement of the 200 new evaluation seeds against every placement on record
  budget          server: after the throughput probe, projected all-in cost against the approved cap (refuses if it does not fit)
  eval-select     server: one fixed 80k EMA candidate on vision_v1_select (100 scenes), no intervention
  choose          the training seed with more successes (tie -> seed 1); below 50/100 for both -> stop
  qualify-paired  server: ONE paired session on vision_v1_qualify: the selected new policy and the frozen legacy policy on the
                  same 100 scenes. The set is marked consumed in a ledger before any rollout. No resume.
Execution commands refuse to run unless <run>/authorization.json exists, names the plan by hash and carries a cap.
No command in this file writes that authorisation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, re, subprocess, time
import numpy as np
from hci.io import load_config, read_json, write_json, sha256, digest, runtime_info, UPSTREAM_COMMIT
from hci.statistics import wilson_interval
from hci import vision_v1 as V

SETS = {'vision_v1_select': (50400, 50499), 'vision_v1_qualify': (50500, 50599)}
LEGACY = {'path': 'artifacts/policies/stack_1d_s1_d500_clean_v1_i080000', 'weights.pt': 'faacd2d86464c725a4fb95563a5ed5808019a8b420795b9a2a931822ed89db0e',
          'policy_meta.json': '9976053598632fd58c48b89c9dffabf8f97b3d18ccf5062127c9c0be3de8a468'}
DATA = {'h5': 'runs/stage1d_clean_002/data/train_d500_clean_v1.h5', 'manifest': 'runs/stage1d_clean_002/data/train_d500_clean_manifest.json',
        'gate': 'runs/stage1d_clean_002/gate_train_start.json', 'gate_build': 'runs/stage1d_clean_002/data/data_gate.json', 'h5_sha256': '93416a4a8b7e2203d4cca1fe3c5d1e577830fa720ea4f3855021a168ea65f342'}
PRETRAINED = 'artifacts/pretrained/' + V.PRETRAINED['file']
TRAIN = {'seeds': [1, 2], 'num_demos': 500, 'total_iters': 100000, 'batch_size': 256, 'eval_freq': 10000, 'num_eval_episodes': 5, 'candidate_iteration': 80000, 'candidate_weights': 'EMA'}
RUN_NAME = 'stack_v1_s{seed}_d500_clean_resnet18'
POLICY_DIR = 'artifacts/policies/' + RUN_NAME + '_i080000'
CODE = ['src/hci/vision_v1.py', 'src/hci/policies.py', 'src/hci/simulation.py', 'src/hci/experiment.py', 'src/hci/observations.py', 'src/hci/training.py', 'src/hci/io.py', 'src/hci/schema.py',
        'src/hci/trees.py', 'src/hci/diagnostics.py', 'src/hci/pilot.py', 'src/hci/statistics.py', 'scripts/27_vision_v1_train.py', 'scripts/28_vision_v1_export.py', 'scripts/29_vision_v1.py',
        'scripts/vision_v1/run.sh', 'scripts/stage1b/common.sh', 'configs/vision_v1_splits.json']
LEGACY_PATH_FILES = ['src/hci/policies.py', 'src/hci/simulation.py', 'src/hci/experiment.py', 'src/hci/observations.py', 'src/hci/training.py', 'scripts/03_train_dp.py', 'scripts/04_export_policy.py',
                     'scripts/09_eval_policy.py']
PRIOR_SCENES = ('results/stage2a/scenes.json', 'results/stage2b/scenes.json', 'results/stage2d/scenes.json', 'results/handoff_h1_001/scenes.json')
STATUS = {'legacy_base_policy_qualification': 'BELOW_ORIGINAL_GATE (37/100 < 50/100), historical, unchanged', 'legacy_stage3b': 'COMPLETED_DEV_NO_GATE_PASS', 'handoff_h1': 'EXECUTED_ONCE',
          'vision_v1': 'PREPARATION_ONLY', 'new_training': 'NOT_RUN', 'new_policy_inference': 'NOT_RUN', 'paid_compute': 'NOT_AUTHORIZED', 'calibration': 'NOT_RUN', 'final_test': 'NOT_RUN'}
EXECUTION = ('scenes', 'budget', 'eval_select', 'choose', 'qualify_paired')
IDENTICAL_M = 1e-5


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()


require_authorization = V.require_authorization


def bound(plan):
    """Hashes of what is about to run against the plan. Any difference refuses the command."""
    bad = [f for f, h in plan['code'].items() if sha256(f) != h]
    for name, (path, h) in {'config': (plan['config']['path'], plan['config']['file_sha256']), 'splits manifest': (plan['seed_manifest']['path'], plan['seed_manifest']['sha256']),
                            'legacy weights': (LEGACY['path'] + '/weights.pt', LEGACY['weights.pt']), 'legacy metadata': (LEGACY['path'] + '/policy_meta.json', LEGACY['policy_meta.json']),
                            'pretrained file': (PRETRAINED, V.PRETRAINED['sha256'])}.items():
        if sha256(path) != h: bad.append(name)
    if bad: raise SystemExit(f'Refused: these differ from the plan: {bad}')
    return load_config(plan['config']['path'])


def ledger_of_use(splits, pe, demo_seeds):
    used = {**{f'splits/{k}': list(v) for k, v in splits['splits'].items()}, **{f'policy_eval/{k}': list(v) for k, v in pe['splits'].items()}, 'demonstration_scenes': list(demo_seeds)}
    for p in PRIOR_SCENES: used[p] = list(read_json(p)['list'])
    return used


def new_seeds(manifest, name, used):
    """Seeds of one of the two new sets, after checking the frozen ranges and that no seed has any recorded use."""
    if name not in SETS: raise KeyError(f'{name} is not a Vision-V1 evaluation set; other sets are not touched by this driver')
    table = manifest['splits']
    if set(table) != set(SETS): raise ValueError('the Vision-V1 manifest must hold exactly the two declared sets')
    for k, (lo, hi) in SETS.items():
        if list(table[k]) != list(range(lo, hi + 1)): raise ValueError(f'{k} is not the frozen range {lo}-{hi}')
    if set(table['vision_v1_select']) & set(table['vision_v1_qualify']): raise ValueError('the two new sets overlap')
    clash = {src: sorted(set(ids) & {s for v in table.values() for s in v})[:10] for src, ids in used.items()}
    clash = {k: v for k, v in clash.items() if v}
    if clash: raise ValueError(f'new evaluation seeds have a recorded use: {clash}. The manifest is not patched silently; a revised manifest needs approval')
    return list(table[name])


def cmd_plan(a):
    c = load_config(a.config); splits = read_json(a.splits); pe = read_json(a.policy_eval); dm = read_json(DATA['manifest']); gate = read_json(DATA['gate']); man = read_json(a.seed_manifest)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); problems = []
    if (out / 'plan.json').exists(): raise SystemExit(f'Refused overwrite: {out / "plan.json"}')
    used = ledger_of_use(splits, pe, dm['all_scene_seeds'])
    try: sel = new_seeds(man, 'vision_v1_select', used); qual = new_seeds(man, 'vision_v1_qualify', used)
    except ValueError as e: problems.append(str(e)); sel = qual = []
    h5 = sha256(DATA['h5']) if Path(DATA['h5']).is_file() else None
    if h5 is None: problems.append('INPUT_UNAVAILABLE: the training file is not on this machine')
    elif (h5, dm['train_h5_sha256'], dm['name'], dm['num_train']) != (DATA['h5_sha256'], DATA['h5_sha256'], 'train_d500_clean_v1', 500) or not dm['pass'] or not gate.get('ok') or gate['train_h5_sha256'] != DATA['h5_sha256'] or read_json(DATA['gate_build'])['train_h5_sha256'] != DATA['h5_sha256']:
        problems.append('the training file, its manifest or its data gate is not the validated train_d500_clean_v1')
    for f in ('weights.pt', 'policy_meta.json'):
        if sha256(Path(LEGACY['path']) / f) != LEGACY[f]: problems.append(f'legacy policy {f} differs from the record')
    if sha256(PRETRAINED) != V.PRETRAINED['sha256']: problems.append('pretrained file hash differs')
    if [c['policy'][k] for k in ('obs_horizon', 'act_horizon', 'pred_horizon')] != [2, 8, 16] or c['policy']['unet_dims'] != [64, 128, 256] or c['env']['horizon'] != 200:
        problems.append('config is not the fixed 2/8/16, U-Net 64/128/256, horizon 200')
    ref = read_json('results/handoff_h1_001/execution_plan.json'); same = {f: sha256(f) == ref['code'].get(f) for f in LEGACY_PATH_FILES if f in ref['code']}
    if not all(same.values()): problems.append(f'legacy-path files changed since the H1 run: {[f for f, v in same.items() if not v]}')
    keys = {**read_json('runs/handoff_h1_001/reference_scene_keys.json')['keys'],
            **{f'handoff_h1/{s}': v['key'] for s, v in read_json('results/handoff_h1_001/scenes.json')['nearest'].items()}}
    if not (out / 'reference_scene_keys.json').exists():
        write_json(out / 'reference_scene_keys.json', {'meaning': 'initial cubeA xyz, cubeB xyz of placements already on record (no environment was reset to make this file)', 'keys': keys})
    plan = {'work_id': a.out.replace('\\', '/').rstrip('/').split('/')[-1], 'stage': 'Vision-V1: pretrained ResNet-18 base-policy upgrade', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'pass': not problems, 'problems': problems, 'status': STATUS,
            'authorization': {'paid_compute_authorized': False, 'proposed_cap_usd': 3.00, 'proposed_cap_is': 'a proposal, not an approval; the H1 approval is not inherited',
                              'required_file': 'authorization.json in the run folder: {"approved_by": "user", "paid_compute_authorized": true, "cap_usd": <number>, "plan_sha256": "<sha256 of this plan>"}'},
            'question': 'Does replacing the weak visual encoder with an ImageNet-pretrained ResNet-18-based encoder improve task success for a Diffusion Policy trained on exactly the same '
                        'validated 500 StackCube demonstrations?',
            'caveat': 'Against the legacy 4-layer encoder this changes architecture, capacity and pretraining together. A gain could not be attributed to pretraining alone.',
            'repository': {'head': git('rev-parse', 'HEAD'), 'remote': git('remote', 'get-url', 'origin'), 'working_tree_changes_at_plan_time': git('status', '--porcelain').splitlines(),
                           'legacy_path_files_identical_to_the_h1_run': same},
            'data': {**DATA, 'h5_sha256_now': h5, 'manifest_sha256': sha256(DATA['manifest']), 'gate_sha256': sha256(DATA['gate']), 'name': dm['name'], 'num_train': dm['num_train'],
                     'train_transitions': dm.get('train_transitions'), 'rule': 'the same file bytes, the same 500 trajectories in the same order; no re-conversion, addition, removal or augmentation'},
            'legacy_comparator': {**LEGACY, 'architecture': V.LEGACY_ARCH, 'historic_qualification': '37/100 on policy_qualify_B (not reused, not a contemporaneous baseline)'},
            'config': {'path': a.config, 'file_sha256': sha256(a.config), 'config_sha256': digest(c), 'policy': c['policy'], 'env': c['env'], 'protocol': c['protocol']},
            'encoder': V.SPEC, 'pretrained': {**V.PRETRAINED, 'local_path': PRETRAINED, 'sha256_now': sha256(PRETRAINED)},
            'unchanged': ['task and environment', '500 demonstrations', 'observation whitelist and camera order', 'proprioceptive state', 'targets', 'action U-Net (64/128/256, group norm 8)',
                          'DDPM scheduler, 100 steps', 'horizons 2/8/16', 'loss', 'AdamW and LR schedule of the upstream trainer', 'EMA', 'batch 256', '100,000 updates', 'success definition'],
            'deviations': ['BatchNorm layers of the backbone use fixed ImageNet running statistics in training and deployment (the upstream EMA covers parameters only); affine parameters are trained'],
            'training': {**TRAIN, 'run_names': [RUN_NAME.format(seed=s) for s in TRAIN['seeds']], 'candidate_policy_dirs': [POLICY_DIR.format(seed=s) for s in TRAIN['seeds']],
                         'checkpoint_rule': 'both seeds: EMA weights of full_080000.pt (upstream tag 80000 = 80,001 optimizer updates, the same tag as the legacy 80k policy). '
                                            'The 100k schedule is completed; the 100k checkpoint is archived and cannot replace 80k.',
                         'upstream_monitoring': 'eval-freq 10000, 5 unseeded episodes: descriptive log only, never a selection criterion',
                         'on_failure': 'partial output kept as INCOMPLETE; no retry with another seed, learning rate, budget or fewer updates'},
            'seed_manifest': {'path': a.seed_manifest, 'sha256': sha256(a.seed_manifest), 'sets': {k: [lo, hi, hi - lo + 1] for k, (lo, hi) in SETS.items()}, 'select': sel[:1] + sel[-1:], 'qualify': qual[:1] + qual[-1:],
                              'seed_level_check': 'no overlap with splits.json, policy_eval_splits.json, the 550 demonstration scene seeds, or the scenes of Stage 2A / 2B / 2D / H1',
                              'placement_check': 'server step `scenes` before any training: cube placements of the 200 seeds against stored placements (demonstrations, pilots) and against audit / dev / '
                                                 'policy-evaluation sets by reset; calibration and Test are compared by seed number only. An identical placement fails the step; nothing is replaced silently',
                              'untouched_sets': ['policy_monitor', 'policy_select', 'policy_qualify_A', 'policy_qualify_B', 'calibration', 'test']},
            'selection': {'set': 'vision_v1_select', 'rule': 'more success_once wins; tie -> train seed 1', 'gate': 'both candidates below 50/100 -> NO_VIABLE_DEV_CANDIDATE, qualification not consumed'},
            'qualification': {'set': 'vision_v1_qualify', 'use': 'once: the selected new policy and the frozen legacy policy on the same 100 initial scenes, no intervention',
                              'ledger': 'artifacts/ledger/vision_v1_qualify.json, created exclusively before any rollout; a second session is refused; no resume',
                              'noise': "both policies use seed_for(scene, physical step of the call, 0, 'dp'); the same scene seed gives the same initial state. The policies differ, so their "
                                       'rollouts are not coupled beyond the initial state and the noise keys'},
            'labels': {'TARGET_MET': 'selected new policy >= 60/100 AND at least 10 more successful scenes than the legacy policy on the same 100 scenes', 'PROMISING_BELOW_TARGET': '>= 50/100 but TARGET_MET not reached',
                       'BELOW_GATE': '< 50/100', 'other': ['NO_VIABLE_DEV_CANDIDATE', 'INCOMPLETE', 'BUDGET_BLOCKED', 'TECHNICAL_BLOCKED'],
                       'meaning': 'engineering go / no-go labels fixed before any outcome; not a significance test'},
            'volume': {'throughput_probe': 'full data, batch 256, 300 updates, non-candidate run vision_v1_probe (never exported)', 'training_runs': 2, 'updates_per_run': 100000,
                       'select_episodes': 200, 'qualify_episodes': 200, 'placement_resets': 200 + 5 + 100 + 270},
            'code': {f: sha256(f) for f in CODE}, 'upstream_commit': UPSTREAM_COMMIT,
            'not_in_this_work': ['recovery selector', 'R5', 'oracle correction', 'new control primitive', 'LLM', 'calibration', 'final Test', 'pretrained-versus-scratch control', 'more seeds or demonstrations'],
            'stopping': 'one execution. Even with TARGET_MET no recovery experiment follows automatically.'}
    write_json(out / 'plan.json', plan); print(json.dumps({k: plan[k] for k in ('pass', 'problems')}, indent=1))
    if problems: raise SystemExit(2)


def cmd_dry_run(a):
    plan = read_json(a.plan); run = Path(a.run); need = sorted(set(plan['code']) | {plan['config']['path'], DATA['h5'], DATA['manifest'], DATA['gate'], PRETRAINED, LEGACY['path'] + '/weights.pt', LEGACY['path'] + '/policy_meta.json'})
    missing = [f for f in need if not Path(f).exists()]; changed = [f for f, h in plan['code'].items() if Path(f).exists() and sha256(f) != h]; auth = (run / 'authorization.json').exists()
    out = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha256(a.plan), 'volume': plan['volume'], 'training': plan['training'], 'files_needed_on_the_server': need,
           'upload_bytes': int(sum(Path(f).stat().st_size for f in need if Path(f).exists())), 'files_missing': missing, 'code_files_changed_since_plan': changed, 'authorization_present': auth,
           'execution_commands_would_run': False, 'paid_compute': 'NOT_AUTHORIZED' if not auth else 'AUTHORIZATION_FILE_PRESENT_NOT_USED_BY_DRY_RUN', 'servers_created': 0, 'trainings_started': 0, 'policy_inferences': 0}
    print(json.dumps({k: v for k, v in out.items() if k != 'files_needed_on_the_server'}, indent=1))
    if a.out: write_json(a.out, out)
    if missing or changed: raise SystemExit(2)


def cmd_scenes(a):
    plan = read_json(a.plan); require_authorization(Path(a.out).parent, a.plan); c = bound(plan); man = read_json(plan['seed_manifest']['path'])
    splits = read_json('configs/splits.json'); pe = read_json('configs/policy_eval_splits.json')
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
        names = list(refs); table = np.stack([refs[n] for n in names]); own = {}; clashes = []
        for name in SETS:
            for s in man['splits'][name]:
                k = key(s); d = np.abs(table - k).max(1); j = int(d.argmin())
                twin = [t for t, v in own.items() if np.abs(v - k).max() <= IDENTICAL_M]
                if d[j] <= IDENTICAL_M or twin: clashes.append({'seed': s, 'set': name, 'identical_to': names[j] if d[j] <= IDENTICAL_M else twin[0]})
                own[f'{name}/{s}'] = k
        report = {'pass': not clashes, 'identical_placements': clashes, 'seeds_checked': len(own), 'identical_threshold_m': IDENTICAL_M,
                  'references': {'stored_keys_demonstrations_and_pilots': stored, **{k: len(v) for k, v in reset_sets.items()}}, 'seed_level_only': ['splits/calibration', 'splits/test'],
                  'plan_sha256': sha256(a.plan), 'on_clash': 'this step fails; no seed is replaced; a revised manifest needs approval',
                  'scope': 'Initial cube placement only; no policy was run on these scenes.'}
        write_json(a.out, report); print(json.dumps({k: report[k] for k in ('pass', 'identical_placements', 'seeds_checked')}))
        if clashes: raise SystemExit(2)
    finally:
        env.close()


def projected_cost(*, updates_per_second, usd_per_hour, spent_usd, cap_usd, reserve_usd, eval_seconds_per_episode, monitor_seconds_per_episode=36., fixed_seconds=900, margin=1.15):
    """All-in projection for the two fixed trainings and the four 100-scene passes. Inputs are time and money only."""
    # the upstream trainer evaluates at tag 0, every eval_freq updates and at the end: 11 evaluations of 5 episodes per run. Their time is not measured by the
    # probe, so a deliberately high bound per episode is charged (default 36 s; a full 200-step project episode took about 5 s with the legacy policy)
    monitor_s = 2 * (TRAIN['total_iters'] // TRAIN['eval_freq'] + 1) * TRAIN['num_eval_episodes'] * monitor_seconds_per_episode
    train_s = 2 * TRAIN['total_iters'] / updates_per_second + monitor_s
    eval_s = 400 * eval_seconds_per_episode; hours = (train_s + eval_s + fixed_seconds) * margin / 3600
    total = spent_usd + hours * usd_per_hour + reserve_usd
    return {'train_hours_both_runs': train_s / 3600, 'evaluation_hours': eval_s / 3600, 'projected_remaining_hours': hours, 'projected_total_usd': total, 'cap_usd': cap_usd, 'fits_cap': bool(total <= cap_usd),
            'monitoring_hours_bound': monitor_s / 3600, 'inputs': {'monitor_seconds_per_episode': monitor_seconds_per_episode, 'updates_per_second': updates_per_second, 'usd_per_hour': usd_per_hour, 'spent_usd': spent_usd, 'reserve_usd': reserve_usd, 'eval_seconds_per_episode': eval_seconds_per_episode,
                       'fixed_seconds': fixed_seconds, 'margin': margin}, 'rule': 'if the projection exceeds the cap, training does not start; seeds, scenes and architecture are not reduced'}


def cmd_budget(a):
    plan = read_json(a.plan); run = Path(a.run); auth = require_authorization(run, a.plan)
    log = Path(a.probe_log).read_text(errors='ignore'); rates = [float(x) for x in re.findall(r'([0-9.]+)it/s', log)]
    if not rates: raise SystemExit('no it/s found in the probe log; the projection is undefined')
    d = projected_cost(updates_per_second=float(np.median(rates[-20:])), usd_per_hour=a.usd_per_hour, spent_usd=a.spent_usd, cap_usd=float(auth['cap_usd']), reserve_usd=a.reserve_usd,
                       eval_seconds_per_episode=a.eval_seconds_per_episode, monitor_seconds_per_episode=a.monitor_seconds_per_episode)
    d.update(peak_vram_mib=a.peak_vram_mib, probe_log=a.probe_log, status='OK' if d['fits_cap'] else 'BUDGET_BLOCKED'); write_json(run / 'budget_gate.json', d); print(json.dumps(d, indent=1))
    if not d['fits_cap']: raise SystemExit(2)


def rollout_table(c, policy, seeds, upstream, device, out, label):
    """One fixed policy without intervention on the given scene seeds, through the project Runtime. Stage labels are simulator ground truth, logged only."""
    from hci.diagnostics import step_state, stage_events
    from hci.pilot import Recorder
    r = V.runtime(c, policy, upstream, device); rows = []; started = time.perf_counter()
    try:
        with (out / f'episodes_{label}.jsonl').open('x', encoding='utf-8') as f:
            for seed in seeds:
                tick = time.perf_counter(); r.reset(seed); initial = step_state(r.env, 0)
                with Recorder(r, probe=lambda rt, z: step_state(rt.env, rt.t, z)) as rec:
                    rec.rows = trace = []; res = r.complete()
                row = {'seed': int(seed), 'success_once': int(res['success']), 'steps': res['steps'], 'reason': res['reason'], 'policy_calls': res['policy_calls'],
                       'policy_inference_seconds': res['policy_inference_seconds'], 'furthest_stage': stage_events(initial, trace)['furthest_stage'], 'wall_seconds': time.perf_counter() - tick}
                rows.append(row); f.write(json.dumps(row) + '\n'); f.flush()
                print(f"{label} seed={seed} success_once={row['success_once']} steps={row['steps']} ({sum(x['success_once'] for x in rows)}/{len(rows)})", flush=True)
    finally:
        r.env.close()
    return rows, time.perf_counter() - started


def table_summary(rows):
    n = len(rows); k = int(sum(r['success_once'] for r in rows)); lo, hi = wilson_interval(k, n); stages = {}
    for r in rows: stages[r['furthest_stage']] = stages.get(r['furthest_stage'], 0) + 1
    return {'n': n, 'success_once': k, 'wilson95_descriptive': [lo, hi], 'furthest_stage': dict(sorted(stages.items())), 'mean_steps_successful': float(np.mean([r['steps'] for r in rows if r['success_once']])) if k else None,
            'seconds_per_policy_call': sum(r['policy_inference_seconds'] for r in rows) / max(1, sum(r['policy_calls'] for r in rows)), 'wall_seconds_per_episode': float(np.mean([r['wall_seconds'] for r in rows]))}


def used_ledger():
    return ledger_of_use(read_json('configs/splits.json'), read_json('configs/policy_eval_splits.json'), read_json(DATA['manifest'])['all_scene_seeds'])


def check_candidate(plan, policy, train_seed):
    m = read_json(Path(policy) / 'policy_meta.json'); want = POLICY_DIR.format(seed=train_seed)
    if Path(policy).as_posix().rstrip('/') != want: raise SystemExit(f'the candidate of train seed {train_seed} is {want}')
    if (m.get('architecture_id'), m.get('iteration'), m.get('weights'), m.get('train_seed')) != (V.ARCH_ID, TRAIN['candidate_iteration'], 'EMA', train_seed) or m['source_training_metadata']['demo_sha256'] != DATA['h5_sha256']:
        raise SystemExit('not the predeclared 80k EMA Vision-V1 candidate of this training seed and data file')
    a = m['args']
    if (a['total_iters'], a['batch_size'], a['num_demos'], a['seed']) != (TRAIN['total_iters'], TRAIN['batch_size'], TRAIN['num_demos'], train_seed): raise SystemExit('candidate was not trained with the frozen settings')
    return m


def cmd_eval_select(a):
    plan = read_json(a.plan); run = Path(a.run); require_authorization(run, a.plan); c = bound(plan)
    if not read_json(run / 'scenes.json')['pass'] or read_json(run / 'scenes.json')['plan_sha256'] != sha256(a.plan): raise SystemExit('placement check missing, failed or of another plan')
    m = check_candidate(plan, a.policy, a.train_seed); seeds = new_seeds(read_json(plan['seed_manifest']['path']), 'vision_v1_select', used_ledger())
    out = run / 'select' / f'seed{a.train_seed}'; out.mkdir(parents=True, exist_ok=False)
    rows, wall = rollout_table(c, a.policy, seeds, a.upstream, a.device, out, f'select_seed{a.train_seed}')
    write_json(out / 'result.json', {'set': 'vision_v1_select', 'train_seed': a.train_seed, 'policy': a.policy, 'weights_sha256': m['weights_sha256'], 'architecture_id': m['architecture_id'], 'seeds': seeds,
                                     **table_summary(rows), 'episodes_sha256': sha256(out / f'episodes_select_seed{a.train_seed}.jsonl'), 'wall_seconds': wall, 'plan_sha256': sha256(a.plan),
                                     'runtime': runtime_info(), 'scope': 'Development selection of a fixed checkpoint without intervention. The interval is descriptive.'})


def cmd_choose(a):
    plan = read_json(a.plan); run = Path(a.run); require_authorization(run, a.plan); res = {s: read_json(run / 'select' / f'seed{s}' / 'result.json') for s in TRAIN['seeds']}
    for s, r in res.items():
        if (r['n'], r['set'], r['plan_sha256']) != (100, 'vision_v1_select', sha256(a.plan)) or sha256(run / 'select' / f'seed{s}' / f'episodes_select_seed{s}.jsonl') != r['episodes_sha256']:
            raise SystemExit(f'selection result of seed {s} is incomplete or was changed')
    d = V.select_seed({s: r['success_once'] for s, r in res.items()}); k = d['selected_train_seed']
    d.update(selected_policy=None if k is None else POLICY_DIR.format(seed=k), selected_weights_sha256=None if k is None else res[k]['weights_sha256'], plan_sha256=sha256(a.plan),
             decided_before='any rollout on vision_v1_qualify', qualification_ledger_exists=(Path(a.ledger) / 'vision_v1_qualify.json').exists())
    if d['qualification_ledger_exists']: raise SystemExit('the qualification set was already consumed; a selection cannot be made afterwards')
    write_json(run / 'selection.json', d); print(json.dumps(d, indent=1))
    if k is None: raise SystemExit(3)


def cmd_qualify_paired(a):
    plan = read_json(a.plan); run = Path(a.run); require_authorization(run, a.plan); c = bound(plan); sel = read_json(run / 'selection.json')
    if sel['status'] != 'SELECTED' or sel['plan_sha256'] != sha256(a.plan): raise SystemExit('no selected candidate for this plan; the qualification set is not consumed')
    new = sel['selected_policy']; m = check_candidate(plan, new, sel['selected_train_seed'])
    if m['weights_sha256'] != sel['selected_weights_sha256']: raise SystemExit('the selected policy changed after selection')
    seeds = new_seeds(read_json(plan['seed_manifest']['path']), 'vision_v1_qualify', used_ledger()); out = run / 'qualify'
    if out.exists(): raise SystemExit(f'{out} exists: a qualification session is never resumed or repeated')
    # exclusive create BEFORE any rollout: from here on the set is consumed, whatever happens next
    write_json(Path(a.ledger) / 'vision_v1_qualify.json', {'set': 'vision_v1_qualify', 'consumed_utc': datetime.now(timezone.utc).isoformat(), 'session': out.as_posix(), 'plan_sha256': sha256(a.plan),
                                                          'seed_manifest_sha256': plan['seed_manifest']['sha256'], 'new_policy': new, 'new_weights_sha256': m['weights_sha256'], 'legacy_policy': LEGACY['path'],
                                                          'legacy_weights_sha256': LEGACY['weights.pt'], 'note': 'Single use. Written before any policy outcome existed. No resume, no second session.'})
    out.mkdir(parents=True); rows_new, wall_new = rollout_table(c, new, seeds, a.upstream, a.device, out, 'new'); rows_old, wall_old = rollout_table(c, LEGACY['path'], seeds, a.upstream, a.device, out, 'legacy')
    yn = {r['seed']: r['success_once'] for r in rows_new}; yo = {r['seed']: r['success_once'] for r in rows_old}; sn = table_summary(rows_new); so = table_summary(rows_old)
    write_json(out / 'paired_result.json', {'set': 'vision_v1_qualify', 'seeds': seeds, 'new': {'policy': new, 'weights_sha256': m['weights_sha256'], 'train_seed': sel['selected_train_seed'], **sn},
                                            'legacy': {'policy': LEGACY['path'], 'weights_sha256': LEGACY['weights.pt'], **so}, 'paired': V.paired_counts(yn, yo),
                                            'difference_in_successful_scenes': sn['success_once'] - so['success_once'], 'label': V.outcome_label(sn['success_once'], so['success_once'], len(seeds)),
                                            'label_definition': plan['labels'], 'episodes_sha256': {k: sha256(out / f'episodes_{k}.jsonl') for k in ('new', 'legacy')}, 'plan_sha256': sha256(a.plan),
                                            'wall_seconds': {'new': wall_new, 'legacy': wall_old}, 'runtime': runtime_info(),
                                            'scope': 'One paired development qualification without intervention. Labels are engineering go / no-go labels, not a significance test. '
                                                     'The legacy 37/100 on policy_qualify_B is a different scene set and is not compared here.'})
    print(json.dumps({k: v for k, v in read_json(out / 'paired_result.json').items() if k in ('paired', 'difference_in_successful_scenes', 'label')}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('plan'); q.add_argument('--out', required=True); q.add_argument('--config', default='configs/stackcube.yaml'); q.add_argument('--splits', default='configs/splits.json')
    q.add_argument('--policy-eval', default='configs/policy_eval_splits.json'); q.add_argument('--seed-manifest', default='configs/vision_v1_splits.json')
    q = s.add_parser('dry-run'); q.add_argument('--plan', required=True); q.add_argument('--run', required=True); q.add_argument('--out')
    q = s.add_parser('scenes'); q.add_argument('--plan', required=True); q.add_argument('--keys', required=True); q.add_argument('--out', required=True)
    q = s.add_parser('budget'); q.add_argument('--plan', required=True); q.add_argument('--run', required=True); q.add_argument('--probe-log', required=True); q.add_argument('--usd-per-hour', type=float, required=True)
    q.add_argument('--spent-usd', type=float, required=True); q.add_argument('--reserve-usd', type=float, default=0.30); q.add_argument('--eval-seconds-per-episode', type=float, default=8.0); q.add_argument('--monitor-seconds-per-episode', type=float, default=36.0); q.add_argument('--peak-vram-mib', type=float)
    for name in ('eval-select', 'choose', 'qualify-paired'):
        q = s.add_parser(name); q.add_argument('--plan', required=True); q.add_argument('--run', required=True)
        if name != 'choose': q.add_argument('--upstream', default='third_party/ManiSkill'); q.add_argument('--device', default='cuda')
        if name == 'eval-select': q.add_argument('--policy', required=True); q.add_argument('--train-seed', type=int, required=True, choices=TRAIN['seeds'])
        if name != 'eval-select': q.add_argument('--ledger', default='artifacts/ledger')
    a = p.parse_args(); globals()['cmd_' + a.cmd.replace('-', '_')](a)


if __name__ == '__main__':
    main()
