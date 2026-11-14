"""Stage 1D clean rebuild: audit every converted demonstration for episode-content integrity, exclude every
trajectory that fails, fill the gaps from the unused reserve in source order, and gate the training start.

Sub-commands (all CPU except `replay`)
  audit   every converted trajectory the project holds (old conversion 129 + new conversion 416): content integrity,
          content provenance, quarantine list, protected-set status, replacement choice, and the contamination
          statistics of the aborted stage1d_001 training file.
  build   train_d500_clean_v1 (+ JSON): retained payloads copied unchanged from the preserved converted files,
          replacements placed at the ids of the excluded trajectories; then the built file itself is validated and
          data_gate.json is written.
  replay  (simulator) expert action replay of all replacements and of retained old/new demonstrations picked by
          source-id rule; actions only, no state is written back during the replay.
  gate    re-reads the files and reports and refuses (exit 2) unless everything a training start needs is in place.

The four known ids are never used as a blacklist: exclusion follows from the validator only.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, csv, importlib.util, json
import numpy as np
from hci.io import load_config, read_json, write_json, new_run, finish_run, sha256, digest, UPSTREAM_COMMIT
from hci.diagnostics import even_pick
from hci.integrity import content_integrity, match_source, window_counts, HARD, SOFT, JUMP_CUBE_M, JUMP_QPOS_RAD, TOL, CUBE_PATH_M

HERE = Path(__file__).resolve().parent
CUBES = ('cubeA', 'cubeB')
LAYERS = ('structural_validation', 'source_and_split_validation', 'episode_content_integrity', 'sampled_action_replay', 'policy_io_parity')


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), HERE / name)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def validator_hash():
    return digest({f: sha256(p) for f, p in (('integrity.py', HERE.parent / 'src/hci/integrity.py'), ('17_stage1d_clean.py', HERE / '17_stage1d_clean.py'))})


def source_tables(fs, source_info):
    """Per source episode: first joint positions, first and second cube states, length; and whether the expert gripper is +-1."""
    table = {}; binary = True
    for e in source_info['episodes']:
        g = fs[f"traj_{e['episode_id']}"]; art = g['env_states/articulations/panda_wristcam']
        table[e['episode_id']] = {'first_qpos': art[0, 13:22], 'state0': {n: g[f'env_states/actors/{n}'][0] for n in CUBES},
                                  'state1': {n: g[f'env_states/actors/{n}'][1] for n in CUBES}, 'cubeA_path': g['env_states/actors/cubeA'][()], 'steps': int(g['actions'].shape[0]), 'seed': e['episode_seed']}
        binary = binary and bool(np.all(np.abs(np.abs(g['actions'][:, -1]) - 1) < 1e-6))
    return table, binary


def check_group(g, src, binary):
    traj = {'actions': g['actions'][()], 'terminated': g['terminated'][()], 'truncated': g['truncated'][()], 'success': g['success'][()],
            'qpos': g['obs/agent/qpos'][()], 'cubeB': g['env_states/actors/cubeB'][()], 'cubeA': g['env_states/actors/cubeA'][()], 'state0': {n: g[f'env_states/actors/{n}'][0] for n in CUBES}}
    return content_integrity(traj, src, expert_gripper_binary=binary), traj


def audit_file(f, info, table, binary, by_seed, label):
    """Integrity rows of one converted file. Declared origin comes from the JSON seed; content origin from the stored first state."""
    rows = []; second = {i: t['state1'] for i, t in table.items()}
    for e in sorted(info['episodes'], key=lambda e: e['episode_id']):
        sid = by_seed[e['episode_seed']]; g = f[f"traj_{e['episode_id']}"]; res, traj = check_group(g, table[sid], binary)
        origin = match_source(traj['state0'], second); tail_origin = None
        if res['boundary_steps'] and res['boundary_steps'][-1] + 1 < len(traj['qpos']):
            k = res['boundary_steps'][-1] + 1                                   # first observation after the reset
            cube_after = {n: g[f'env_states/actors/{n}'][k] for n in CUBES}
            tail_origin = [i for i, t in table.items() if all(float(np.abs(cube_after[n][:7] - t['state0'][n][:7]).max()) <= 1e-4 for n in CUBES)]
        rows.append({'file': label, 'traj_id': e['episode_id'], 'scene_seed': e['episode_seed'], 'declared_source_episode_id': sid, 'steps': res['steps'], 'source_steps': res['source_steps'],
                     'verdict': res['verdict'], 'hard_signals': res['hard_signals'], 'soft_signals': res['soft_signals'], 'boundary_steps': res['boundary_steps'],
                     'content_origin_of_first_state': origin, 'declared_matches_content_origin': origin == [sid], 'origin_after_boundary': tail_origin,
                     'first_obs_vs_source_first_qpos_max_abs': res['signals'].get('first_obs_not_source', 0.), 'first_state_vs_source_second_state_max_abs': res['signals'].get('first_state_not_source', 0.),
                     'cube_path_vs_source_max_m': res['cube_path_vs_source_max_m'], 'max_cube_step_m': res['max_cube_step_m'], 'max_joint_step_rad': res['max_joint_step_rad'], 'first_success_step': res['first_success_step'],
                     'gripper_not_binary_steps': res['signals'].get('gripper_not_binary', []), 'internal_terminal_steps': res['signals'].get('internal_terminal', []),
                     'truncated_restart_steps': res['signals'].get('truncated_restart', []), 'success_flicker_steps': res['success_flicker_steps'], 'length_minus_source': res['steps'] - res['source_steps']})
    return rows


def cmd_audit(a):
    import h5py
    c = load_config(a.config); plan = read_json(a.plan1d); old_manifest = read_json(a.d500_manifest); problems = []
    old = Path(a.old_demo); new = Path(a.new_demo); source = Path(a.source)
    for key, f in (('old_demo_sha256', old), ('source_sha256', source), ('source_json_sha256', source.with_suffix('.json'))):
        if sha256(f) != plan['inputs'][key]: raise SystemExit(f'{f} is not the file fixed in the Stage 1D plan')
    if sha256(new) != old_manifest['new_converted_sha256']: raise SystemExit('new conversion differs from the stage1d_001 manifest')
    source_info = read_json(source.with_suffix('.json')); by_seed = {e['episode_seed']: e['episode_id'] for e in source_info['episodes']}
    with h5py.File(source, 'r') as fs, h5py.File(old, 'r') as fo, h5py.File(new, 'r') as fn:
        table, binary = source_tables(fs, source_info)
        rows = audit_file(fo, read_json(old.with_suffix('.json')), table, binary, by_seed, 'old_converted') + audit_file(fn, read_json(new.with_suffix('.json')), table, binary, by_seed, 'new_converted')
    role = {}
    for k in ('old_train', 'protected_dev', 'protected_remainder'):
        for x in plan['roles'][k]: role[('old_converted', x['traj_id'])] = k
    train_id = {('old_converted' if r['role'] == 'old_train' else 'new_converted', r['origin_traj_id']): r['train_traj_id'] for r in old_manifest['train']}
    for r in old_manifest['train']:
        if r['role'] == 'new_train': role[('new_converted', r['origin_traj_id'])] = 'new_train'
    for x in old_manifest['new_conversion_unused']: role[('new_converted', x['origin_traj_id'])] = 'unused_reserve'
    for r in rows:
        r['role_stage1d_001'] = role.get((r['file'], r['traj_id'])); r['stage1d_001_train_id'] = train_id.get((r['file'], r['traj_id']))
        if r['role_stage1d_001'] is None: problems.append(f"{r['file']} traj {r['traj_id']} has no recorded role")
    if len({r['declared_source_episode_id'] for r in rows}) != len(rows): problems.append('a source episode appears in more than one converted trajectory')
    counts = {}
    for r in rows: counts.setdefault(r['role_stage1d_001'], {}).setdefault(r['verdict'], 0); counts[r['role_stage1d_001']][r['verdict']] += 1
    bad = [r for r in rows if r['verdict'] != 'PASS']
    # Every failed source episode of both conversions and the converted success right after it.
    failed = [x['source_episode_id'] for x in plan['roles']['old_dropped']] + [x['source_episode_id'] for x in old_manifest['new_conversion_dropped_source_episodes']]
    order = [e['episode_id'] for e in source_info['episodes']]; by_source = {r['declared_source_episode_id']: r for r in rows}; after_failed = []
    for sid in sorted(failed):
        nxt = order[order.index(sid) + 1] if order.index(sid) + 1 < len(order) else None; r = by_source.get(nxt)
        after_failed.append({'failed_source_episode_id': sid, 'next_source_episode_id': nxt, 'next_converted': None if r is None else {k: r[k] for k in ('file', 'traj_id', 'verdict', 'hard_signals', 'boundary_steps', 'role_stage1d_001', 'stage1d_001_train_id')},
                             'prefix_is_the_failed_episode': None if r is None else r['content_origin_of_first_state'] == [sid]})
    # Hidden mixing: does any excluded trajectory carry content of a protected or evaluation scene?
    protected_sources = {x['source_episode_id'] for k in ('protected_dev', 'protected_remainder') for x in plan['roles'][k]}
    foreign = sorted({i for r in bad for i in (r['content_origin_of_first_state'] or [])})
    # Replacement choice: reserve trajectories that pass, in canonical source-id order.
    reserve = sorted([r for r in rows if r['role_stage1d_001'] == 'unused_reserve'], key=lambda r: r['declared_source_episode_id'])
    excluded_train = sorted([r for r in bad if r['role_stage1d_001'] in ('old_train', 'new_train')], key=lambda r: (r['role_stage1d_001'] != 'old_train', r['stage1d_001_train_id']))
    eligible = [r for r in reserve if r['verdict'] == 'PASS' and r['declared_source_episode_id'] not in protected_sources]
    if len(eligible) < len(excluded_train): problems.append(f'reserve too small: {len(eligible)} eligible for {len(excluded_train)} excluded training trajectories')
    pairs = [{'slot_train_id': x['stage1d_001_train_id'], 'replaces': {k: x[k] for k in ('file', 'traj_id', 'declared_source_episode_id', 'scene_seed', 'role_stage1d_001')},
              'replacement': {k: y[k] for k in ('file', 'traj_id', 'declared_source_episode_id', 'scene_seed')}, 'label': f'r{n + 1}',
              'belongs_to': 'D100_clean_v1' if x['role_stage1d_001'] == 'old_train' else 'Dnew400_clean_v1'} for n, (x, y) in enumerate(zip(excluded_train, eligible))]
    # Contamination of the aborted stage1d_001 training file (counts only).
    contam = []
    for r in excluded_train:
        k = r['boundary_steps'][-1] if r['boundary_steps'] else None
        contam.append({'stage1d_001_train_id': r['stage1d_001_train_id'], 'steps': r['steps'], 'boundary_action_index': k, 'failed_prefix_steps': k,
                       **(window_counts(r['steps'], k, c['policy']['obs_horizon'], c['policy']['pred_horizon']) if k is not None else {})})
    total_windows = old_manifest['train_transitions'] - old_manifest['num_train'] * (c['policy']['obs_horizon'] - 1)
    summary = {'pass': not problems, 'problems': problems, 'n_trajectories': len(rows), 'by_role': counts, 'expert_gripper_is_binary_in_source': binary,
               'not_pass': [{k: r[k] for k in ('file', 'traj_id', 'declared_source_episode_id', 'role_stage1d_001', 'stage1d_001_train_id', 'verdict', 'hard_signals', 'soft_signals', 'boundary_steps', 'content_origin_of_first_state', 'origin_after_boundary')} for r in bad],
               'success_flicker': [{k: r[k] for k in ('file', 'traj_id', 'declared_source_episode_id', 'role_stage1d_001', 'verdict', 'success_flicker_steps')} for r in rows if r['success_flicker_steps']],
               'after_failed_source_episodes': after_failed, 'foreign_content_source_ids': foreign, 'foreign_content_in_protected_or_eval': sorted(set(foreign) & protected_sources),
               'stage1d_001_contamination': {'trajectories': contam, 'failed_prefix_steps_total': sum(x['failed_prefix_steps'] or 0 for x in contam), 'dummy_transitions': sum(1 for x in contam if x['boundary_action_index'] is not None),
                                             'train_transitions': old_manifest['train_transitions'], 'windows_total': total_windows,
                                             'windows_touched': sum(x.get('touched_by_prefix_or_reset', 0) for x in contam), 'windows_target_contains_dummy_action': sum(x.get('target_contains_dummy_action', 0) for x in contam),
                                             'note': 'Counts of affected steps and training windows. They say nothing about the size of the effect on a policy.'},
               'thresholds': {'cube_path_m': CUBE_PATH_M, 'jump_cube_m': JUMP_CUBE_M, 'jump_qpos_rad': JUMP_QPOS_RAD, 'state_tol': TOL, 'hard': HARD, 'soft': SOFT},
               'validator_sha256': validator_hash(), 'inputs': {'old_converted_sha256': sha256(old), 'new_converted_sha256': sha256(new), 'source_sha256': sha256(source)}}
    if not getattr(a, 'quiet', False): print(json.dumps({k: summary[k] for k in ('pass', 'problems', 'n_trajectories', 'by_role', 'expert_gripper_is_binary_in_source', 'not_pass', 'after_failed_source_episodes', 'foreign_content_in_protected_or_eval')}, indent=1))
    print('replacements:', [(p['label'], p['slot_train_id'], p['replacement']['declared_source_episode_id']) for p in pairs]); print(json.dumps(summary['stage1d_001_contamination'], indent=1))
    if a.dry_run: return
    out = new_run(a.out, 'stage1d-clean-audit', c); keys = list(rows[0])
    with (out / 'content_integrity_all.csv').open('x', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    write_json(out / 'integrity_summary.json', summary)
    write_json(out / 'quarantine_manifest.json', {'rule': 'every trajectory whose verdict is not PASS is excluded from training; nothing is trimmed or repaired', 'excluded': [r for r in bad],
                                                 'kept_on_disk': 'the original converted files and the stage1d_001 training file are preserved unchanged'})
    prot = [r for r in rows if r['role_stage1d_001'] in ('protected_dev', 'protected_remainder')]
    write_json(out / 'protected_integrity_manifest.json', {'note': 'Protected scenes stay excluded from training whatever their verdict.', 'n': len(prot), 'n_pass': sum(r['verdict'] == 'PASS' for r in prot),
                                                          'not_pass': [r for r in prot if r['verdict'] != 'PASS'], 'scenes': [{k: r[k] for k in ('traj_id', 'declared_source_episode_id', 'scene_seed', 'role_stage1d_001', 'verdict')} for r in prot]})
    write_json(out / 'replacement_manifest.json', {'rule': 'reserve = successful conversions of the stage1d_001 window that were not adopted; candidates must pass the content validator and must not be protected; '
                                                          'they are taken in ascending source-episode id; the first replaces the excluded old (D100) trajectory, the next ones the excluded new trajectories in ascending old training id',
                                                  'reserve': [{k: r[k] for k in ('traj_id', 'declared_source_episode_id', 'scene_seed', 'verdict', 'hard_signals')} for r in reserve], 'pairs': pairs,
                                                  'reserve_left_unused': [r['declared_source_episode_id'] for r in eligible[len(excluded_train):]]})
    finish_run(out, audit_pass=summary['pass'])
    if problems: raise SystemExit(2)


def replay_picks(m, old=5, new=5):
    """Training rows that the expert replay must cover: every replacement, then retained old/new at equal spacing over ascending source id."""
    by_source = lambda role: sorted([r for r in m['train'] if r['role'] == role], key=lambda r: r['source_episode_id'])
    o = by_source('retained_old'); n = by_source('retained_new')
    return by_source('replacement') + even_pick(o, min(old, len(o))) + even_pick(n, min(new, len(n)))


def build(audit_dir, plan, old_manifest, old_demo, new_demo, source, out, c, recheck):
    """recheck: the audit arguments (config, plan1d, d500_manifest, old_demo, new_demo, source). The audit is recomputed from the inputs and
    must reproduce the saved audit files byte for byte, so an edited verdict table or replacement list cannot steer the build."""
    import h5py
    out = Path(out); audit_dir = Path(audit_dir)
    cmd_audit(SimpleNamespace(**recheck, out=str(out / 'audit_recheck'), dry_run=False, quiet=True))
    for f in ('integrity_summary.json', 'replacement_manifest.json', 'content_integrity_all.csv', 'quarantine_manifest.json'):
        if sha256(out / 'audit_recheck' / f) != sha256(audit_dir / f): raise SystemExit(f'audit artifact {f} is not what the validator produces from these inputs')
    m16 = load_script('16_stage1d_data.py'); v07 = load_script('07_validate_demos.py'); audit_dir = Path(audit_dir); out = Path(out); problems = []
    summary = read_json(audit_dir / 'integrity_summary.json'); repl = read_json(audit_dir / 'replacement_manifest.json')
    if not summary['pass']: raise SystemExit('The audit did not pass')
    if summary['validator_sha256'] != validator_hash(): raise SystemExit('The validator changed after the audit')
    verdict = {}
    with (audit_dir / 'content_integrity_all.csv').open(encoding='utf-8') as fh:
        for r in csv.DictReader(fh): verdict[(r['file'], int(r['traj_id']))] = r['verdict']
    slot = {p['slot_train_id']: p for p in repl['pairs']}; old_demo = Path(old_demo); new_demo = Path(new_demo); source = Path(source)
    old_info = read_json(old_demo.with_suffix('.json')); new_info = read_json(new_demo.with_suffix('.json')); source_info = read_json(source.with_suffix('.json'))
    eps = {'old_converted': {e['episode_id']: e for e in old_info['episodes']}, 'new_converted': {e['episode_id']: e for e in new_info['episodes']}}
    by_seed = {e['episode_seed']: e['episode_id'] for e in source_info['episodes']}
    protected = plan['roles']['protected_dev'] + plan['roles']['protected_remainder']; protected_seeds = {x['scene_seed'] for x in protected}; protected_sources = {x['source_episode_id'] for x in protected}
    items = []
    for r in old_manifest['train']:
        key = ('old_converted' if r['role'] == 'old_train' else 'new_converted', r['origin_traj_id'])
        if verdict[key] == 'PASS':
            if r['train_traj_id'] in slot: problems.append(f"train id {r['train_traj_id']} passes but has a replacement")
            items.append({'role': 'retained_old' if r['role'] == 'old_train' else 'retained_new', 'file': key[0], 'origin_traj_id': key[1], 'expected_payload_sha256': r['payload_sha256'], 'replaces': None, 'label': None})
        else:
            p = slot.get(r['train_traj_id'])
            if p is None: problems.append(f"excluded train id {r['train_traj_id']} has no replacement"); continue
            y = p['replacement']; items.append({'role': 'replacement', 'file': y['file'], 'origin_traj_id': y['traj_id'], 'expected_payload_sha256': None, 'label': p['label'], 'belongs_to': p['belongs_to'],
                                                'replaces': {'stage1d_001_train_id': r['train_traj_id'], **p['replaces']}})
    if problems: raise SystemExit('; '.join(problems))
    h5 = out / 'train_d500_clean_v1.h5'; sample = out / 'io_parity_sample_clean_v1.h5'; rows = []; episodes = []; keys = {}
    with h5py.File(old_demo, 'r') as fo, h5py.File(new_demo, 'r') as fn, h5py.File(source, 'r') as fs, h5py.File(h5, 'x') as ft:
        table, binary = load_tables(fs, source_info)
        for new_id, it in enumerate(items):
            origin = fo if it['file'] == 'old_converted' else fn; g = origin[f"traj_{it['origin_traj_id']}"]; ep = eps[it['file']][it['origin_traj_id']]; sid = by_seed[ep['episode_seed']]
            if verdict[(it['file'], it['origin_traj_id'])] != 'PASS': problems.append(f'train id {new_id}: origin did not pass the audit')
            if ep['episode_seed'] in protected_seeds or sid in protected_sources: problems.append(f'train id {new_id}: protected scene')
            origin.copy(g, ft, name=f'traj_{new_id}'); want = m16.dataset_hashes(g); got = m16.dataset_hashes(ft[f'traj_{new_id}']); payload = m16.combine(want)
            if want != got: problems.append(f'train id {new_id}: payload changed by the copy')
            if it['expected_payload_sha256'] and payload != it['expected_payload_sha256']: problems.append(f'train id {new_id}: payload differs from the stage1d_001 record')
            res, _ = check_group(ft[f'traj_{new_id}'], table[sid], binary)       # the validator runs again on the BUILT file
            if res['verdict'] != 'PASS': problems.append(f"train id {new_id}: built trajectory is {res['verdict']} {res['hard_signals'] + res['soft_signals']}")
            fp, key = m16.source_scene(fs[f'traj_{sid}']); keys.setdefault(tuple(key), []).append(new_id)
            rows.append({'train_traj_id': new_id, 'role': it['role'], 'label': it['label'], 'origin_file': it['file'], 'origin_traj_id': it['origin_traj_id'], 'source_episode_id': sid, 'scene_seed': ep['episode_seed'],
                         'reset_kwargs': ep['reset_kwargs'], 'steps': res['steps'], 'first_success_step': res['first_success_step'], 'content_verdict': res['verdict'], 'initial_scene_fingerprint': fp,
                         'payload_sha256': payload, 'obs_action_sha256': m16.combine(want, m16.is_policy_payload), 'payload_equal_to_origin': want == got,
                         'payload_equal_to_stage1d_001': None if it['expected_payload_sha256'] is None else payload == it['expected_payload_sha256'], 'replaces': it['replaces']})
            episodes.append({**ep, 'episode_id': new_id, 'hci_origin': {'role': it['role'], 'origin_file': it['file'], 'origin_traj_id': it['origin_traj_id'], 'source_episode_id': sid}})
        for x in protected:
            _, key = m16.source_scene(fs[f"traj_{x['source_episode_id']}"]); keys.setdefault(tuple(key), []).append(f"protected:{x['traj_id']}")
        # what the upstream loader reads: numeric sort of traj_<id>, first num_demos
        loader_ids = sorted(ft.keys(), key=lambda x: int(x.split('_')[-1]))[:len(items)]
        if loader_ids != [f'traj_{i}' for i in range(len(items))] or len(ft.keys()) != len(items): problems.append('loader order does not enumerate ids 0..N-1 exactly')
        # I/O audit sample: every replacement, then retained old and retained new picked by position
        r_ids = [r['train_traj_id'] for r in rows if r['role'] == 'replacement']; o_ids = [r['train_traj_id'] for r in rows if r['role'] == 'retained_old']; n_ids = [r['train_traj_id'] for r in rows if r['role'] == 'retained_new']
        picks = r_ids + even_pick(o_ids, min(3, len(o_ids))) + even_pick(n_ids, min(3, len(n_ids)))
        with h5py.File(sample, 'x') as fp_:
            for k, i in enumerate(picks): ft.copy(ft[f'traj_{i}'], fp_, name=f'traj_{k}')
    head = {k: v for k, v in old_info.items() if k != 'episodes'}
    if old_info.get('env_info') != new_info.get('env_info') or old_info.get('commit_info') != new_info.get('commit_info'): problems.append('env_info/commit_info differ between the conversions')
    write_json(h5.with_suffix('.json'), {**head, 'episodes': episodes}); write_json(sample.with_suffix('.json'), {**head, 'episodes': [{**episodes[i], 'episode_id': k, 'hci_train_traj_id': i} for k, i in enumerate(picks)]})
    structure_rows, structure_problems = v07.check_structure(h5, read_json(h5.with_suffix('.json')), c)
    dup = {str(k): v for k, v in keys.items() if len(v) > 1}; seeds = [r['scene_seed'] for r in rows]; sources = [r['source_episode_id'] for r in rows]
    if dup: problems.append(f'identical initial cube placements: {list(dup.values())[:5]}')
    if len(set(sources)) != len(sources) or len(set(seeds)) != len(seeds): problems.append('a source episode or seed is used twice')
    if set(seeds) & protected_seeds: problems.append('a protected scene is in the training file')
    declared = {s for name, ids in m16.declared_seeds(read_json('configs/splits.json'), read_json('configs/policy_eval_splits.json')).items() for s in ids}
    audit_seeds = set(read_json('configs/splits.json')['splits']['audit'])
    clash = sorted((set(seeds) & declared) - audit_seeds)
    if clash: problems.append(f'training seeds in a declared evaluation split: {clash[:10]}')
    if any(r['role'] == 'replacement' and r['scene_seed'] in declared for r in rows): problems.append('a replacement carries a declared split seed')
    n = {k: sum(r['role'] == k for r in rows) for k in ('retained_old', 'retained_new', 'replacement')}; d100 = [r for r in rows if r['train_traj_id'] < len(plan['roles']['old_train'])]
    excluded_all = [x['declared_source_episode_id'] for x in summary['not_pass']]
    all_seeds = sorted(set(old_manifest['all_scene_seeds']) | set(seeds))
    manifest = {'pass': not problems and not structure_problems, 'problems': problems + structure_problems, 'name': 'train_d500_clean_v1',
                'meaning_of_clean': 'passed the declared content-integrity checks; not a guarantee that no defect remains', 'demo_path': str(h5), 'train_h5_sha256': sha256(h5),
                'train_json_sha256': sha256(h5.with_suffix('.json')), 'io_parity_sample_sha256': sha256(sample), 'io_parity_sample_train_ids': picks, 'validator_sha256': validator_hash(),
                'env_info': old_info['env_info'], 'commit_info': old_info.get('commit_info'), 'num_train': len(rows), 'counts': n, 'train_transitions': int(sum(r['steps'] for r in rows)),
                'train_steps_min_mean_max': [int(min(r['steps'] for r in rows)), float(np.mean([r['steps'] for r in rows])), int(max(r['steps'] for r in rows))],
                'first_success_after_horizon': [{k: r[k] for k in ('train_traj_id', 'source_episode_id', 'first_success_step', 'steps')} for r in rows if r['first_success_step'] is None or r['first_success_step'] > c['env']['horizon']],
                'gripper_values': sorted({v for r in structure_rows for v in r['gripper_values']}), 'excluded_source_episode_ids': excluded_all, 'initial_state_source': m16.INITIAL_STATE_SOURCE,
                'loader_rule': 'upstream load_traj_hdf5 sorts traj_<id> numerically and takes the first num_demos; this file holds exactly ids 0..N-1 (enumerated at build time)',
                'retained_payload_equal_to_stage1d_001': all(r['payload_equal_to_stage1d_001'] for r in rows if r['role'] != 'replacement'),
                'all_scene_seeds': all_seeds, 'train': rows,
                'comparison_note': 'Not a "demonstration count only" change relative to the Stage 1B D100: one D100 trajectory is excluded and replaced.'}
    write_json(out / 'train_d500_clean_manifest.json', manifest)
    write_json(out / 'train_d100_clean_ids.json', {'note': 'Manifest only. No D100-clean policy is trained in this stage and no performance is claimed for it.', 'definition': 'ids 0..99 of train_d500_clean_v1 = 99 retained D100 trajectories + r1',
                                                  'ids': [{k: r[k] for k in ('train_traj_id', 'role', 'label', 'source_episode_id', 'scene_seed')} for r in d100]})
    layers = {'structural_validation': {'status': 'PASS' if not structure_problems else 'FAIL', 'n': len(structure_rows), 'problems': structure_problems},
              'source_and_split_validation': {'status': 'PASS' if not problems else 'FAIL', 'problems': problems, 'protected_overlap': sorted(set(seeds) & protected_seeds), 'distinct_initial_placements': len(keys)},
              'episode_content_integrity': {'status': 'PASS' if all(r['content_verdict'] == 'PASS' for r in rows) and len(rows) == len(old_manifest['train']) else 'FAIL', 'n_pass': sum(r['content_verdict'] == 'PASS' for r in rows), 'n': len(rows)},
              'sampled_action_replay': {'status': 'NOT RUN'}, 'policy_io_parity': {'status': 'NOT RUN'}}
    write_json(out / 'data_gate.json', {'dataset': 'train_d500_clean_v1', 'n_train_expected': len(old_manifest['train']), 'train_h5_sha256': manifest['train_h5_sha256'], 'train_json_sha256': manifest['train_json_sha256'],
                                        'manifest_sha256': sha256(out / 'train_d500_clean_manifest.json'), 'validator_sha256': validator_hash(), 'layers': layers,
                                        'unresolved_candidates': [r['train_traj_id'] for r in rows if r['content_verdict'] != 'PASS'],
                                        'note': 'Written at build time. The two simulator layers are added as separate reports and checked by the gate command; an exit code or a final success flag replaces none of the layers.'})
    return manifest


def load_tables(fs, source_info): return source_tables(fs, source_info)


def cmd_build(a):
    c = load_config(a.config); plan = read_json(a.plan1d); old_manifest = read_json(a.d500_manifest)
    recheck = dict(config=a.config, plan1d=a.plan1d, d500_manifest=a.d500_manifest, old_demo=a.old_demo, new_demo=a.new_demo, source=a.source)
    out = new_run(a.out, 'stage1d-clean-build', c); m = build(a.audit, plan, old_manifest, a.old_demo, a.new_demo, a.source, out, c, recheck); finish_run(out, data_pass=m['pass'])
    print(json.dumps({k: m[k] for k in ('pass', 'problems', 'counts', 'num_train', 'train_transitions', 'train_steps_min_mean_max', 'first_success_after_horizon', 'gripper_values', 'retained_payload_equal_to_stage1d_001', 'train_h5_sha256')}, indent=1))
    if not m['pass']: raise SystemExit(2)


def gate(data, ledger='artifacts/ledger', qualify_set='policy_qualify_B', require=LAYERS, budget=None):
    """Re-read everything a training start depends on. Returns (ok, report)."""
    import h5py
    data = Path(data); problems = []; h5 = data / 'train_d500_clean_v1.h5'
    for f in ('data_gate.json', 'train_d500_clean_manifest.json'):
        if not (data / f).is_file(): return False, {'ok': False, 'problems': [f'missing validation report: {f}']}
    g = read_json(data / 'data_gate.json'); m = read_json(data / 'train_d500_clean_manifest.json'); layers = dict(g['layers'])
    if not h5.is_file(): problems.append('training file missing')
    else:
        if sha256(h5) != g['train_h5_sha256'] or sha256(h5) != m['train_h5_sha256']: problems.append('the validation report belongs to a different HDF5')
        if sha256(h5.with_suffix('.json')) != g['train_json_sha256']: problems.append('the validation report belongs to a different JSON')
    if sha256(data / 'train_d500_clean_manifest.json') != g['manifest_sha256']: problems.append('manifest changed after validation')
    if g['validator_sha256'] != validator_hash(): problems.append('validator source changed after validation')
    if g['unresolved_candidates']: problems.append(f"unresolved defect candidates: {g['unresolved_candidates'][:10]}")
    n = g['n_train_expected']
    if not m['pass'] or m['num_train'] != n: problems.append(f'manifest does not describe {n} passing trajectories')
    if h5.is_file() and not problems:
        with h5py.File(h5, 'r') as f:
            if sorted(f.keys(), key=lambda x: int(x.split('_')[-1])) != [f'traj_{i}' for i in range(n)]: problems.append('training file ids are not 0..N-1')
    for name, fname, key in (('sampled_action_replay', 'sampled_action_replay.json', 'train_h5_sha256'), ('policy_io_parity', 'policy_io_parity.json', 'demo_sha256')):
        f = data / fname
        if f.is_file():
            v = read_json(f); want = m['train_h5_sha256'] if name == 'sampled_action_replay' else m['io_parity_sample_sha256']
            if name == 'sampled_action_replay':      # coverage: exactly the rows the rule demands, each replayed successfully
                need = [r['train_traj_id'] for r in replay_picks(m)]; got = [x.get('traj_id') for x in v.get('replay', [])]
                covered = got == need and all(x.get('success_once') for x in v.get('replay', []))
            else:
                covered = v.get('same_raw_frames', {}).get('n_trajectories') == len(m['io_parity_sample_train_ids']) and bool(v.get('actions', {}).get('all_pass'))
            layers[name] = {'status': 'PASS' if v.get('pass') and v.get(key) == want and covered else 'FAIL', 'report': str(f), 'same_file': v.get(key) == want, 'coverage_ok': bool(covered)}
    for name in require:
        if layers.get(name, {}).get('status') != 'PASS': problems.append(f"{name}: {layers.get(name, {}).get('status', 'NOT RUN')}")
    if (Path(ledger) / f'{qualify_set}.json').exists(): problems.append(f'{qualify_set} is already consumed')
    if budget is not None:
        for k in ('cap_usd', 'spent_usd', 'usd_per_hour', 'train_limit_s', 'reserve_usd'):
            if budget.get(k) is None: problems.append(f'budget field missing: {k}')
        if not problems and budget['spent_usd'] + budget['usd_per_hour'] * budget['train_limit_s'] / 3600 + budget['reserve_usd'] > budget['cap_usd']: problems.append('time limit x price + reserve exceeds the remaining budget')
    return not problems, {'ok': not problems, 'problems': problems, 'layers': layers, 'required': list(require), 'budget': budget, 'train_h5_sha256': m.get('train_h5_sha256')}


def cmd_gate(a):
    budget = None
    if a.train_start:
        budget = {'cap_usd': a.cap_usd, 'spent_usd': a.spent_usd, 'usd_per_hour': a.usd_per_hour, 'train_limit_s': a.train_limit_s, 'reserve_usd': a.reserve_usd}
    ok, report = gate(a.data, a.ledger, a.qualify_set, LAYERS if a.train_start else LAYERS[:3], budget)
    print(json.dumps(report, indent=1))
    if a.out: write_json(a.out, report)
    if not ok: raise SystemExit(2)


def cmd_replay(a):
    c = load_config(a.config); data = Path(a.data); m = read_json(data / 'train_d500_clean_manifest.json'); h5 = data / 'train_d500_clean_v1.h5'; v07 = load_script('07_validate_demos.py')
    if sha256(h5) != m['train_h5_sha256']: raise SystemExit('Training file is not the validated one')
    picks = replay_picks(m, a.old, a.new)        # by source-id rule, before any policy exists
    rows = [{'traj_id': r['train_traj_id'], 'scene_seed': r['scene_seed']} for r in picks]; out = new_run(a.out, 'stage1d-clean-replay', c)
    done = v07.replay(c, h5, rows, out, 0); info = {r['train_traj_id']: r for r in picks}; problems = []
    for x in done:
        r = info[x['traj_id']]; x.update(role=r['role'], source_episode_id=r['source_episode_id'], file_first_success_step=r['first_success_step'], recorded_success=r['first_success_step'] is not None)
        if not x['success_once']: problems.append(f"expert replay failed: train id {x['traj_id']} ({r['role']}, source {r['source_episode_id']})")
    report = {'pass': not problems and len(done) == len(picks), 'problems': problems, 'train_h5_sha256': m['train_h5_sha256'], 'n': len(done), 'replay': done,
              'method': 'project environment, reset(seed) once, then the recorded actions only up to the config horizon; no simulator state is written during the replay (same meaning as Stage 1B validate_demos)',
              'pick_rule': f'all replacements + {a.old} retained old + {a.new} retained new at equal spacing over ascending source-episode id',
              'note': 'A failed replay stops the data gate; it is never swapped for another demonstration. Success of the actions is not bitwise trajectory identity.'}
    write_json(data / 'sampled_action_replay.json', report); finish_run(out, replay_pass=report['pass'])
    print(json.dumps({k: report[k] for k in ('pass', 'problems', 'n')}, indent=1))
    for x in done: print(x['traj_id'], x['role'], x['source_episode_id'], x['success_once'], x['first_success_step'], x['file_first_success_step'], x['initial_rgb_max_abs_vs_file'], '%.1e' % x['final_tcp_pose_max_abs_vs_file'])
    if not report['pass']: raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = p.add_subparsers(dest='cmd', required=True)
    def common(q):
        q.add_argument('--config', required=True); q.add_argument('--plan1d', default='runs/stage1d_001/plan/stage1d_plan.json'); q.add_argument('--d500-manifest', default='runs/stage1d_001/data/train_d500_manifest.json')
        q.add_argument('--old-demo', required=True); q.add_argument('--new-demo', required=True); q.add_argument('--source', required=True); q.add_argument('--out', required=True)
    q = sub.add_parser('audit'); common(q); q.add_argument('--dry-run', action='store_true'); q.set_defaults(fn=cmd_audit)
    q = sub.add_parser('build'); common(q); q.add_argument('--audit', required=True); q.set_defaults(fn=cmd_build)
    q = sub.add_parser('replay'); q.add_argument('--config', required=True); q.add_argument('--data', required=True); q.add_argument('--out', required=True)
    q.add_argument('--old', type=int, default=5); q.add_argument('--new', type=int, default=5); q.set_defaults(fn=cmd_replay)
    q = sub.add_parser('gate'); q.add_argument('--data', required=True); q.add_argument('--ledger', default='artifacts/ledger'); q.add_argument('--qualify-set', default='policy_qualify_B'); q.add_argument('--out')
    q.add_argument('--train-start', action='store_true', help='require all five layers and the budget fields'); q.add_argument('--cap-usd', type=float); q.add_argument('--spent-usd', type=float)
    q.add_argument('--usd-per-hour', type=float); q.add_argument('--train-limit-s', type=int); q.add_argument('--reserve-usd', type=float); q.set_defaults(fn=cmd_gate)
    a = p.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
