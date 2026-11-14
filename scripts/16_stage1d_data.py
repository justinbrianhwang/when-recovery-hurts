"""Stage 1D data: an explicit D100 ⊂ D500 training set with the Stage 1B held-out scenes kept out of training.

Sub-commands
  plan      CPU, before any GPU is rented. Verifies the preserved artifacts, fixes the roles of every existing
            demonstration (old_train / protected_dev / protected_remainder), declares which SOURCE episodes are
            converted for the 400 new demonstrations, the candidate and qualification rules, and the budget.
  subset    Writes a source folder (trajectory.h5 copy + trajectory.json restricted to the declared source
            episodes) so that the pinned replay_trajectory converts exactly those episodes with the Stage 1B flags.
  build     Copies the 100 old training trajectories from the preserved converted file (no re-rendering) and the
            first `num_new` successfully converted new trajectories into a train-only HDF5 (traj_0..traj_499, the
            only ids the upstream loader can read with --num-demos 500), then audits payload identity and overlap.
  validate  Needs the simulator: structure check and expert action replay of old and new demonstrations that were
            picked by position only.

Initial-scene fingerprints use the FIRST state of the SOURCE trajectory. At the pinned commit the converted file's
env_states[0] holds the second source state while its first observation is rendered from the first one, so the
converted env_states[0] is never used as the initial scene. Object poses are read for logging only.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, shutil, subprocess
import numpy as np
from hci.io import load_config, read_json, write_json, new_run, finish_run, sha256, digest, UPSTREAM_COMMIT
from hci.diagnostics import even_pick, first_true

H5NAME = 'trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5'
CONVERT_FLAGS = ['--use-first-env-state', '-c', 'pd_ee_delta_pos', '-o', 'rgb', '-b', 'physx_cpu', '--num-envs', '1', '--save-traj']
INITIAL_STATE_SOURCE = 'source trajectory.h5, env_states[0] of the source episode (not the converted file env_states[0])'
STOP_CONDITIONS = ['D100 payload identity or protected-set overlap check fails', 'observation/action meaning differs or the new-weights I/O parity fails',
                   'NaN/Inf, missing data, or an expert replay failure that cannot be explained', 'VRAM, disk or budget would be exceeded',
                   'qualification ledger conflict', 'no safe backup destination']


def expected_hashes(path):
    out = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip(): h, name = line.split(None, 1); out[name.strip().lstrip('*')] = h
    return out


def check_list(path, root='.'):
    """Compare a preserved SHA256 list with the files on disk. Nothing is skipped silently: missing files are listed."""
    ok = 0; missing = []; mismatch = []
    for name, want in expected_hashes(path).items():
        f = Path(root) / name
        if not f.is_file(): missing.append(name)
        elif sha256(f) != want: mismatch.append(name)
        else: ok += 1
    return {'list': str(path), 'list_sha256': sha256(path), 'n': ok + len(missing) + len(mismatch), 'ok': ok, 'missing': missing, 'mismatch': mismatch}


def dataset_hashes(group):
    """SHA256 of dtype, shape and bytes of every dataset below an HDF5 group (independent of chunking/compression)."""
    import h5py
    out = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            h = hashlib.sha256(); h.update(str(obj.dtype).encode()); h.update(str(obj.shape).encode()); h.update(np.ascontiguousarray(obj[()]).tobytes()); out[name] = h.hexdigest()
    group.visititems(visit); return out


def combine(hashes, keep=lambda name: True):
    h = hashlib.sha256()
    for name in sorted(hashes):
        if keep(name): h.update(name.encode()); h.update(b'\0'); h.update(hashes[name].encode())
    return h.hexdigest()


def is_policy_payload(name): return name == 'actions' or name.startswith('obs/')


def source_scene(sg):
    """Initial scene of a SOURCE trajectory group: fingerprint of all first states and the cube placement key."""
    h = hashlib.sha256(); first = {}
    def visit(name, obj):
        if hasattr(obj, 'shape') and name.startswith('env_states/'): first[name] = np.asarray(obj[0], np.float32)
    sg.visititems(visit)
    for name in sorted(first): h.update(name.encode()); h.update(first[name].tobytes())
    key = [round(float(x), 5) for n in ('env_states/actors/cubeA', 'env_states/actors/cubeB') for x in first[n][:7]]
    return h.hexdigest(), key


def roles_from_manifest(dm, old_info, source_info):
    """Roles of every trajectory of the preserved converted file, keyed by its own ids and mapped to source episodes by seed."""
    by_seed = {e['episode_seed']: e['episode_id'] for e in source_info['episodes']}
    if len(by_seed) != len(source_info['episodes']): raise ValueError('Source scene seeds are not unique')
    used = {x['traj_id'] for x in dm['train'] + dm['heldout']}; rest = []
    for e in sorted(old_info['episodes'], key=lambda e: e['episode_id']):
        if e['episode_id'] not in used: rest.append({'traj_id': e['episode_id'], 'source_episode_id': by_seed[e['episode_seed']], 'scene_seed': e['episode_seed'], 'steps': e['elapsed_steps']})
    for x in dm['train'] + dm['heldout']:
        if by_seed.get(x['scene_seed']) != x['source_episode_id']: raise ValueError(f"traj {x['traj_id']}: manifest source id does not match the source file")
    return {'old_train': dm['train'], 'protected_dev': dm['heldout'], 'protected_remainder': rest, 'old_dropped': dm['dropped_source_episodes']}


def declared_seeds(splits, policy_eval):
    out = {}
    for label, table in (('splits', splits['splits']), ('policy_eval', policy_eval['splits'])):
        for name, ids in table.items(): out[f'{label}/{name}'] = list(ids)
    return out


def candidate_window(source_info, start, window):
    eps = source_info['episodes'][start:start + window]
    if len(eps) < window: raise ValueError(f'Source has only {len(eps)} episodes from position {start}; wanted {window}')
    return [{'source_episode_id': e['episode_id'], 'scene_seed': e['episode_seed'], 'source_steps': e['elapsed_steps']} for e in eps]


def overlap_report(roles, candidates, declared):
    """Seed-level overlaps. Returns (report, problems)."""
    problems = []; groups = {k: [x['scene_seed'] for x in roles[k]] for k in ('old_train', 'protected_dev', 'protected_remainder')}
    groups['old_dropped'] = [x['scene_seed'] for x in roles['old_dropped']]; groups['new_candidates'] = [x['scene_seed'] for x in candidates]
    names = list(groups); between = {}
    for i, a in enumerate(names):
        if len(set(groups[a])) != len(groups[a]): problems.append(f'duplicate seeds inside {a}')
        for b in names[i + 1:]:
            both = sorted(set(groups[a]) & set(groups[b])); between[f'{a}&{b}'] = both
            if both: problems.append(f'{a} and {b} share seeds {both[:10]}')
    demo = {s for v in groups.values() for s in v}; with_eval = {}
    for name, ids in declared.items():
        both = sorted(demo & set(ids))
        if both: with_eval[name] = both
    # Known and kept on record since Stage 1B: the five restore-audit seeds are scenes of the OLD conversion (training or
    # protected). That split measures determinism, not performance. Any other overlap with a declared split, and any
    # overlap that involves a NEW demonstration, stops the stage.
    for name, both in with_eval.items():
        if name != 'splits/audit' or set(both) & set(groups['new_candidates']): problems.append(f'demonstration seeds overlap {name}: {both[:10]}')
    return {'between_groups': between, 'with_declared_splits': with_eval, 'sizes': {k: len(v) for k, v in groups.items()}}, problems


def cmd_plan(a):
    import h5py
    c = load_config(a.config); demo = Path(a.demo); source = Path(a.source); dm = read_json(a.demo_manifest); problems = []
    for key, f in (('demo_sha256', demo), ('demo_json_sha256', demo.with_suffix('.json')), ('source_sha256', source), ('source_json_sha256', source.with_suffix('.json'))):
        if sha256(f) != dm[key]: problems.append(f'{f.name} differs from the Stage 1B demo manifest')
    preserved = {'stage1b': check_list(a.backup_sha), 'stage1c': check_list(a.stage1c_sha)}
    for k, v in preserved.items():
        if v['mismatch'] or v['missing']: problems.append(f'{k}: {len(v["mismatch"])} preserved files changed, {len(v["missing"])} missing')
    old_info = read_json(demo.with_suffix('.json')); source_info = read_json(source.with_suffix('.json'))
    roles = roles_from_manifest(dm, old_info, source_info)
    with h5py.File(demo, 'r') as f:
        ids = sorted(int(k.split('_')[-1]) for k in f if k.startswith('traj_'))
    known = [x['traj_id'] for k in ('old_train', 'protected_dev', 'protected_remainder') for x in roles[k]]
    if ids != known: problems.append('roles do not cover the preserved converted file exactly')
    # The old conversion read the leading source episodes; new candidates start right after the last one it touched.
    touched = max(x['source_episode_id'] for k in roles for x in roles[k]); order = [e['episode_id'] for e in source_info['episodes']]
    start = order.index(touched) + 1
    if sorted(order[:start]) != sorted(x['source_episode_id'] for k in roles for x in roles[k]): problems.append('old roles are not exactly the leading source episodes')
    candidates = candidate_window(source_info, start, a.window)
    pe = read_json(a.policy_eval_manifest); declared = declared_seeds(read_json(a.splits), pe)
    overlap, p2 = overlap_report(roles, candidates, declared); problems += p2
    diag = read_json(a.diag_manifest); train_ids = {x['traj_id'] for x in roles['old_train']}; dev_ids = {x['traj_id'] for x in roles['protected_dev']}
    d_train = [s['traj_id'] for s in diag['scenes'] if s['set'] == 'train']; d_dev = [s['traj_id'] for s in diag['scenes'] if s['set'] == 'heldout']
    if not set(d_train) <= train_ids or not set(d_dev) <= dev_ids: problems.append('Stage 1C diagnostic scenes do not match the roles')
    ledger = Path(a.ledger) / f'{a.qualify_set}.json'; prior = []
    for f in sorted(Path('runs').glob('*/policy_eval.json')) + sorted(Path('results').glob('**/*.json')):
        try: v = read_json(f)
        except (ValueError, OSError): continue
        if isinstance(v, dict) and v.get('set') == a.qualify_set: prior.append(str(f))
    if ledger.exists() or prior: problems.append(f'{a.qualify_set} is already consumed: ledger={ledger.exists()} runs={prior}')
    used_a = Path(a.ledger) / 'policy_qualify_A.json'
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip().splitlines()
    pol = c['policy']
    plan = {'stage': '1D', 'purpose': 'One retraining on D500 = D100 (preserved) + 400 new independent demonstrations. Development only: not calibration, not the final Test.',
            'run_ids': {'stage': a.stage_run, 'policy': a.policy_run}, 'pass': not problems, 'problems': problems,
            'code': {'git_head': head, 'uncommitted_files': dirty, 'upstream_commit': UPSTREAM_COMMIT}, 'config_sha256': digest(c),
            'inputs': {'old_demo': str(demo), 'old_demo_sha256': dm['demo_sha256'], 'old_demo_json_sha256': dm['demo_json_sha256'], 'source': str(source),
                       'source_sha256': dm['source_sha256'], 'source_json_sha256': dm['source_json_sha256'], 'demo_manifest_sha256': sha256(a.demo_manifest),
                       'policy_eval_manifest_sha256': sha256(a.policy_eval_manifest), 'splits_sha256': sha256(a.splits), 'diagnostic_manifest_sha256': sha256(a.diag_manifest)},
            'preserved_artifacts': preserved, 'roles': roles,
            'new_candidates': {'rule': f'source episodes in source-file order starting right after the last episode read by the Stage 1B conversion (position {start}); '
                                       f'a window of {a.window} is converted with the Stage 1B command and flags; the first {a.num_new} that pass the same replay success filter are adopted, '
                                       'the later ones are recorded as unused. Fewer than the required number stops the stage; the window is not extended silently.',
                               'source_position_start': start, 'window': a.window, 'num_new': a.num_new, 'convert_flags': CONVERT_FLAGS, 'episodes': candidates},
            'd500': {'definition': 'D500 = old_train (100, copied from the preserved converted file) + first num_new adopted new demonstrations; ids traj_0..traj_499 in a train-only file',
                     'never_trained': 'protected_dev (the 20 Stage 1B held-out demonstrations) and protected_remainder (the other converted demonstrations that were not in D100)',
                     'initial_state_source': INITIAL_STATE_SOURCE},
            'overlap': overlap, 'declared_split_sizes': {k: len(v) for k, v in declared.items()},
            'training': {'env': c['env']['id'], 'num_demos': 100 + a.num_new, 'init': 'from scratch (no warm start; the probe model is discarded)', 'seed': 1, 'total_iters': 100000,
                         'batch_size': 256, 'lr': pol.get('lr'), 'obs_horizon': pol['obs_horizon'], 'act_horizon': pol['act_horizon'], 'pred_horizon': pol['pred_horizon'],
                         'policy_config': pol, 'horizon': c['env']['horizon'], 'success_definition': c['protocol']['success_definition'],
                         'changed_variable': 'the training demonstrations only; at a fixed update budget each demonstration is revisited about five times less often'},
            'selection': {'candidates_iteration_tags': [60000, 80000, 100000], 'weights': 'EMA', 'set': 'policy_select', 'primary': 'success_once', 'tie': 'later iteration',
                          'secondary': 'success_at_end (reported, never used for the choice)', 'execution': 'sequential on one GPU',
                          'tag_note': 'tag i means i+1 optimizer updates for i < total; updates_completed is stored in each checkpoint'},
            'qualification': {'set': a.qualify_set, 'seeds': pe['splits'][a.qualify_set], 'ledger_exists_before': ledger.exists(), 'prior_runs': prior,
                              'qualify_A_ledger_present': used_a.is_file(), 'rule': 'the selected candidate only, once, after tensor parity and new-weights I/O parity pass',
                              'gate': '>= 50/100 operational gate; 10-49 and 0-9 end the stage without further training'},
            'diagnostics': {'manifest': str(a.diag_manifest), 'old_train_scenes': d_train, 'protected_dev_scenes': d_dev, 'cells': ['F_policy', 'R_policy'],
                            'offline': 'same 100 histories x 4 noise seeds; RGB sensitivity (D4) not repeated'},
            'erratum_stage1c': 'The Stage 1C report says F-policy and R-policy agree in 39 of 40 pairs. The unit is scene pairs: 19 of 20 (traj_55 differs). The CSV is unchanged.',
            'budget': {'cap_usd': a.budget_cap, 'note': 'recomputed from the actual rental price before training; never exceeded silently'}, 'stop_conditions': STOP_CONDITIONS}
    print(json.dumps({'pass': plan['pass'], 'problems': problems, 'sizes': overlap['sizes'], 'with_declared_splits': overlap['with_declared_splits'],
                      'new_source_ids': [candidates[0]['source_episode_id'], candidates[-1]['source_episode_id']], 'new_seeds': [candidates[0]['scene_seed'], candidates[-1]['scene_seed']],
                      'preserved': {k: {x: (v[x] if isinstance(v[x], int) else len(v[x])) for x in ('n', 'ok', 'missing', 'mismatch')} for k, v in preserved.items()},
                      'qualify_set': a.qualify_set, 'qualify_seeds': [pe['splits'][a.qualify_set][0], pe['splits'][a.qualify_set][-1], len(pe['splits'][a.qualify_set])],
                      'ledger_exists': ledger.exists()}, indent=1))
    if a.dry_run: print('dry run: nothing written'); return
    out = new_run(a.out, 'stage1d-plan', c); write_json(out / 'stage1d_plan.json', plan); finish_run(out, plan_pass=plan['pass'])
    if problems: raise SystemExit(2)


def cmd_subset(a):
    plan = read_json(a.plan); source = Path(a.source); info = read_json(source.with_suffix('.json'))
    if sha256(source) != plan['inputs']['source_sha256'] or sha256(source.with_suffix('.json')) != plan['inputs']['source_json_sha256']: raise SystemExit('Source files differ from the plan')
    want = [x['source_episode_id'] for x in plan['new_candidates']['episodes']]; by_id = {e['episode_id']: e for e in info['episodes']}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, out / 'trajectory.h5')
    write_json(out / 'trajectory.json', {**info, 'episodes': [by_id[i] for i in want]})
    cmd = ['python', '-m', 'mani_skill.trajectory.replay_trajectory', '--traj-path', str(out / 'trajectory.h5'), *CONVERT_FLAGS, '--count', str(len(want))]
    write_json(out / 'subset.json', {'source_episode_ids': want, 'n': len(want), 'source_copy_sha256': sha256(out / 'trajectory.h5'), 'convert_command': cmd,
                                     'note': 'Episode entries are unchanged; only the list is restricted, so replay_trajectory reads traj_<source id> of the same source file.'})
    print(' '.join(cmd))


def build(plan, old_demo, new_demo, source, out, horizon):
    """Write the train-only file and all manifests below `out`. Returns the audit (audit['pass'] False on any problem)."""
    import h5py
    old_demo = Path(old_demo); new_demo = Path(new_demo); source = Path(source); problems = []; num_new = plan['new_candidates']['num_new']
    old_info = read_json(old_demo.with_suffix('.json')); new_info = read_json(new_demo.with_suffix('.json')); source_info = read_json(source.with_suffix('.json'))
    if sha256(old_demo) != plan['inputs']['old_demo_sha256']: problems.append('old converted file differs from the plan')
    if sha256(source) != plan['inputs']['source_sha256']: problems.append('source file differs from the plan')
    for key in ('env_info', 'commit_info'):
        if old_info.get(key) != new_info.get(key): problems.append(f'{key} differs between the old and the new conversion')
    if new_info.get('commit_info', {}).get('commit_id') != UPSTREAM_COMMIT: problems.append('new conversion was not made at the pinned commit')
    cand = {x['scene_seed']: x for x in plan['new_candidates']['episodes']}; cand_order = [x['scene_seed'] for x in plan['new_candidates']['episodes']]
    new_eps = sorted(new_info['episodes'], key=lambda e: e['episode_id']); new_seeds = [e['episode_seed'] for e in new_eps]
    unknown = [s for s in new_seeds if s not in cand]
    if unknown: problems.append(f'converted episodes outside the declared window: {unknown[:10]}')
    if len(set(new_seeds)) != len(new_seeds): problems.append('duplicate seeds in the new conversion')
    if [s for s in cand_order if s in set(new_seeds)] != new_seeds: problems.append('new conversion is not in declared source order')
    dropped = [cand[s] for s in cand_order if s not in set(new_seeds)]
    if len(new_eps) < num_new: problems.append(f'only {len(new_eps)} new demonstrations passed the replay filter; {num_new} are required')
    adopted = new_eps[:num_new]; unused = new_eps[num_new:]
    old_eps = {e['episode_id']: e for e in old_info['episodes']}; protected = plan['roles']['protected_dev'] + plan['roles']['protected_remainder']
    protected_seeds = {x['scene_seed'] for x in protected}; old_train_seeds = {x['scene_seed'] for x in plan['roles']['old_train']}
    declared = {s for ids in plan.get('declared_seeds', {}).values() for s in ids}
    for e in adopted:
        if e['episode_seed'] in protected_seeds | old_train_seeds | declared: problems.append(f"new demonstration with a protected/used seed: {e['episode_seed']}")
    out = Path(out); train_dir = out / 'train_d500'; train_dir.mkdir(parents=True, exist_ok=False); sample_dir = out / 'io_parity_sample'; sample_dir.mkdir()
    rows = []; episodes = []; over = []; firsts = []; keys = {}; quirk = {'converted0_vs_source0_max': 0., 'converted0_vs_source1_max': 0.}
    items = [('old_train', x['traj_id'], old_eps[x['traj_id']], x['source_episode_id']) for x in plan['roles']['old_train']] + \
            [('new_train', e['episode_id'], e, cand[e['episode_seed']]['source_episode_id']) for e in adopted if e['episode_seed'] in cand]
    with h5py.File(old_demo, 'r') as fo, h5py.File(new_demo, 'r') as fn, h5py.File(source, 'r') as fs, h5py.File(train_dir / H5NAME, 'x') as ft:
        for new_id, (role, src_traj, ep, source_id) in enumerate(items):
            origin = fo if role == 'old_train' else fn; g = origin[f'traj_{src_traj}']
            origin.copy(g, ft, name=f'traj_{new_id}')                      # stored bytes are copied; nothing is re-rendered
            want = dataset_hashes(g); got = dataset_hashes(ft[f'traj_{new_id}'])
            if want != got: problems.append(f'{role} traj {src_traj}: payload changed by the copy')
            sg = fs[f'traj_{source_id}']; fp, key = source_scene(sg); L = int(g['actions'].shape[0]); first = first_true(g['success'][()]) if 'success' in g else None
            for name in ('env_states/actors/cubeA', 'env_states/actors/cubeB'):
                if name in g and name in sg:
                    quirk['converted0_vs_source0_max'] = max(quirk['converted0_vs_source0_max'], float(np.abs(g[name][0] - sg[name][0]).max()))
                    if sg[name].shape[0] > 1: quirk['converted0_vs_source1_max'] = max(quirk['converted0_vs_source1_max'], float(np.abs(g[name][0] - sg[name][1]).max()))
            if ep['elapsed_steps'] != L: problems.append(f'{role} traj {src_traj}: elapsed_steps differs from the action count')
            if not ep.get('success', False): problems.append(f'{role} traj {src_traj}: conversion success flag is not true')
            keys.setdefault(tuple(key), []).append(new_id)
            if first is not None: firsts.append(first)
            if first is None or first > horizon: over.append({'train_traj_id': new_id, 'role': role, 'origin_traj_id': src_traj, 'source_episode_id': source_id, 'first_success_step': first})
            rows.append({'train_traj_id': new_id, 'role': role, 'origin_file': 'old_converted' if role == 'old_train' else 'new_converted', 'origin_traj_id': src_traj,
                         'source_episode_id': source_id, 'scene_seed': ep['episode_seed'], 'reset_kwargs': ep['reset_kwargs'], 'steps': L, 'first_success_step': first,
                         'initial_scene_fingerprint': fp, 'initial_cube_poses': key, 'payload_sha256': combine(want), 'obs_action_sha256': combine(want, is_policy_payload),
                         'payload_equal_to_origin': want == got})
            episodes.append({**ep, 'episode_id': new_id, 'hci_origin': {'role': role, 'origin_traj_id': src_traj, 'source_episode_id': source_id}})
        got_ids = sorted(int(k.split('_')[-1]) for k in ft)
        if got_ids != list(range(len(items))): problems.append('train-only file ids are not 0..N-1')
        # Protected scenes: fingerprints from the source file, to prove that no training scene repeats one of them.
        prot_rows = []
        for x in protected:
            fp, key = source_scene(fs[f"traj_{x['source_episode_id']}"]); keys.setdefault(tuple(key), []).append(f"protected:{x['traj_id']}")
            prot_rows.append({**x, 'role': 'protected_dev' if x in plan['roles']['protected_dev'] else 'protected_remainder', 'initial_scene_fingerprint': fp, 'initial_cube_poses': key})
        # A small file for the I/O audit of the new dataset path: old and new ids picked by position only.
        n_old = len(plan['roles']['old_train']); picks = even_pick(range(n_old), 3) + (even_pick(range(n_old, len(items)), 3) if len(items) - n_old >= 3 else [])
        with h5py.File(sample_dir / H5NAME, 'x') as fp_:
            for k, i in enumerate(picks): ft.copy(ft[f'traj_{i}'], fp_, name=f'traj_{k}')
    write_json(sample_dir / H5NAME.replace('.h5', '.json'), {**{k: v for k, v in old_info.items() if k != 'episodes'},
                                                             'episodes': [{**episodes[i], 'episode_id': k, 'hci_train_traj_id': i} for k, i in enumerate(picks)]})
    write_json(train_dir / H5NAME.replace('.h5', '.json'), {**{k: v for k, v in old_info.items() if k != 'episodes'}, 'episodes': episodes})
    dup = {str(k): v for k, v in keys.items() if len(v) > 1}
    if dup: problems.append(f'identical initial cube placements: {list(dup.values())[:5]}')
    n_old_rows = sum(r['role'] == 'old_train' for r in rows); n_new_rows = sum(r['role'] == 'new_train' for r in rows)
    if n_old_rows != len(plan['roles']['old_train']) or n_new_rows != num_new: problems.append(f'counts: old {n_old_rows}, new {n_new_rows}')
    train_seeds = [r['scene_seed'] for r in rows]
    if set(train_seeds) & protected_seeds: problems.append('a protected scene is in the training file')
    if len(set(train_seeds)) != len(train_seeds): problems.append('duplicate seeds in the training file')
    h5 = train_dir / H5NAME; hashes = {'train_h5_sha256': sha256(h5), 'train_json_sha256': sha256(h5.with_suffix('.json')), 'new_converted_sha256': sha256(new_demo),
                                       'new_converted_json_sha256': sha256(new_demo.with_suffix('.json')), 'io_parity_sample_sha256': sha256(sample_dir / H5NAME)}
    all_seeds = sorted(set(train_seeds) | protected_seeds | {e['episode_seed'] for e in unused} | {x['scene_seed'] for x in plan['roles']['old_dropped']} | {x['scene_seed'] for x in dropped})
    f_arr = np.array(firsts) if firsts else np.zeros(1)
    manifest = {'pass': not problems, 'problems': problems, 'demo_path': str(h5), **hashes, 'plan_sha256': digest(plan), 'env_info': old_info['env_info'], 'commit_info': old_info.get('commit_info'),
                'selection_rule': plan['new_candidates']['rule'], 'loader_rule': 'upstream load_traj_hdf5 sorts traj_<id> numerically and takes the first num_demos; this file holds exactly ids 0..N-1',
                'num_train': len(rows), 'num_old_train': n_old_rows, 'num_new_train': n_new_rows, 'initial_state_source': INITIAL_STATE_SOURCE,
                'train_transitions': int(sum(r['steps'] for r in rows)), 'train_steps_min_mean_max': [int(min(r['steps'] for r in rows)), float(np.mean([r['steps'] for r in rows])), int(max(r['steps'] for r in rows))],
                'first_success': {'horizon': horizon, 'n_after_horizon': len(over), 'fraction_after_horizon': len(over) / len(rows), 'after_horizon': over,
                                  'min_median_mean_max': [int(f_arr.min()), float(np.median(f_arr)), float(f_arr.mean()), int(f_arr.max())],
                                  'note': 'Recorded before training. Such demonstrations are kept: the success filter and the horizon are unchanged.'},
                'train': rows, 'new_conversion_dropped_source_episodes': dropped,
                'new_conversion_unused': [{'origin_traj_id': e['episode_id'], 'scene_seed': e['episode_seed'], 'source_episode_id': cand[e['episode_seed']]['source_episode_id']} for e in unused if e['episode_seed'] in cand],
                # Every demonstration scene known to the project (trained, protected, unused, dropped): scripts/09_eval_policy.py refuses evaluation seeds that appear here.
                'all_scene_seeds': all_seeds,
                'first_env_state_note': {**quirk, 'meaning': 'cube state of the converted env_states[0] against the first and the second source state; the converted value is not used as the initial scene'}}
    audit = {'pass': not problems, 'problems': problems, 'counts': {'D100': n_old_rows, 'Dnew400': n_new_rows, 'D500': len(rows), 'V20': len(plan['roles']['protected_dev']),
                                                                 'protected_remainder': len(plan['roles']['protected_remainder']), 'new_dropped': len(dropped), 'new_unused': len(unused)},
             'old_train_payload_equal': all(r['payload_equal_to_origin'] for r in rows if r['role'] == 'old_train'),
             'new_train_payload_equal': all(r['payload_equal_to_origin'] for r in rows if r['role'] == 'new_train'),
             'train_and_protected_seed_overlap': sorted(set(train_seeds) & protected_seeds), 'duplicate_initial_placements': dup,
             'distinct_initial_placements': len(keys), 'io_parity_sample_train_ids': picks, **hashes}
    write_json(out / 'train_d500_manifest.json', manifest)
    write_json(out / 'protected_dev_manifest.json', {'note': 'Never used for training. protected_dev = the 20 Stage 1B held-out demonstrations; protected_remainder = converted but unused in Stage 1B.',
                                                    'initial_state_source': INITIAL_STATE_SOURCE, 'scenes': prot_rows})
    write_json(out / 'train_d500_ids.json', [{k: r[k] for k in ('train_traj_id', 'role', 'origin_traj_id', 'source_episode_id', 'scene_seed')} for r in rows])
    write_json(out / 'subset_and_overlap_audit.json', audit)
    return audit


def cmd_build(a):
    c = load_config(a.config); plan = read_json(a.plan)
    if not plan['pass']: raise SystemExit('The plan did not pass')
    for key, f in (('splits_sha256', a.splits), ('policy_eval_manifest_sha256', a.policy_eval_manifest), ('source_json_sha256', Path(a.source).with_suffix('.json')),
                   ('old_demo_json_sha256', Path(a.old_demo).with_suffix('.json'))):
        if sha256(f) != plan['inputs'][key]: raise SystemExit(f'{f} differs from the file fixed in the plan')
    plan['declared_seeds'] = declared_seeds(read_json(a.splits), read_json(a.policy_eval_manifest))
    del plan['declared_seeds']['splits/audit']       # known, recorded overlap with five OLD training scenes only
    out = new_run(a.out, 'stage1d-build-d500', c); audit = build(plan, a.old_demo, a.new_demo, a.source, out, c['env']['horizon']); finish_run(out, data_pass=audit['pass'])
    print(json.dumps({k: audit[k] for k in ('pass', 'problems', 'counts', 'old_train_payload_equal', 'new_train_payload_equal', 'train_and_protected_seed_overlap', 'distinct_initial_placements')}, indent=1))
    if not audit['pass']: raise SystemExit(2)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), Path(__file__).resolve().parent / name)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def cmd_validate(a):
    c = load_config(a.config); data = Path(a.data); m = read_json(data / 'train_d500_manifest.json'); h5 = data / 'train_d500' / H5NAME; v07 = load_script('07_validate_demos.py')
    if sha256(h5) != m['train_h5_sha256']: raise SystemExit('Training file changed after the build')
    rows, problems = v07.check_structure(h5, read_json(h5.with_suffix('.json')), c)
    old = [r for r in m['train'] if r['role'] == 'old_train']; new = [r for r in m['train'] if r['role'] == 'new_train']
    picks = [r['train_traj_id'] for r in even_pick(old, a.replay_old) + even_pick(new, a.replay_new)]      # by position only, before any policy exists
    by_id = {r['traj_id']: r for r in rows}; out = new_run(a.out, 'stage1d-validate-d500', c)
    replayed = v07.replay(c, h5, [by_id[i] for i in picks], out, a.record)
    role = {r['train_traj_id']: r['role'] for r in m['train']}; first = {r['train_traj_id']: r['first_success_step'] for r in m['train']}
    for x in replayed:
        x['role'] = role[x['traj_id']]; x['file_first_success_step'] = first[x['traj_id']]
        if not x['success_once']: problems.append(f"expert replay failed: train traj {x['traj_id']} ({x['role']})")
    arm = max(r['arm_action_abs_max'] for r in rows)
    report = {'pass': not problems, 'problems': problems, 'n_trajectories': len(rows), 'train_h5_sha256': m['train_h5_sha256'], 'arm_action_abs_max': arm,
              'gripper_values': sorted({v for r in rows for v in r['gripper_values']}), 'replay_picks': picks, 'replay': replayed,
              'replay_rule': f'{a.replay_old} old and {a.replay_new} new training demonstrations at equal spacing over their id lists; runtime reset(seed) and recorded actions, as in Stage 1B',
              'note': 'Success of the expert actions is not exact trajectory parity; differences from the file are reported per trajectory.'}
    write_json(data / 'demo_validation.json' if a.report is None else a.report, report); finish_run(out, demo_pass=report['pass'])
    print(json.dumps({k: report[k] for k in ('pass', 'problems', 'n_trajectories', 'arm_action_abs_max', 'gripper_values', 'replay_picks')}, indent=1))
    if not report['pass']: raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('plan'); q.add_argument('--config', required=True); q.add_argument('--demo', required=True); q.add_argument('--source', required=True)
    q.add_argument('--demo-manifest', required=True); q.add_argument('--backup-sha', required=True); q.add_argument('--stage1c-sha', required=True)
    q.add_argument('--diag-manifest', required=True); q.add_argument('--out', required=True); q.add_argument('--window', type=int, default=420); q.add_argument('--num-new', type=int, default=400)
    q.add_argument('--splits', default='configs/splits.json'); q.add_argument('--policy-eval-manifest', default='configs/policy_eval_splits.json')
    q.add_argument('--ledger', default='artifacts/ledger'); q.add_argument('--qualify-set', default='policy_qualify_B'); q.add_argument('--stage-run', default='stage1d_001')
    q.add_argument('--policy-run', default='stack_1d_s1_d500'); q.add_argument('--budget-cap', type=float, default=2.0); q.add_argument('--dry-run', action='store_true'); q.set_defaults(fn=cmd_plan)
    q = sub.add_parser('subset'); q.add_argument('--plan', required=True); q.add_argument('--source', required=True); q.add_argument('--out-dir', required=True); q.set_defaults(fn=cmd_subset)
    q = sub.add_parser('build'); q.add_argument('--config', required=True); q.add_argument('--plan', required=True); q.add_argument('--old-demo', required=True); q.add_argument('--new-demo', required=True)
    q.add_argument('--source', required=True); q.add_argument('--out', required=True); q.add_argument('--splits', default='configs/splits.json')
    q.add_argument('--policy-eval-manifest', default='configs/policy_eval_splits.json'); q.set_defaults(fn=cmd_build)
    q = sub.add_parser('validate'); q.add_argument('--config', required=True); q.add_argument('--data', required=True); q.add_argument('--out', required=True); q.add_argument('--report')
    q.add_argument('--replay-old', type=int, default=5); q.add_argument('--replay-new', type=int, default=5); q.add_argument('--record', type=int, default=0); q.set_defaults(fn=cmd_validate)
    a = p.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
