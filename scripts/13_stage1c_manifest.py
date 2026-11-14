"""Stage 1C: verify the preserved artifacts and fix the diagnostic sample BEFORE any diagnostic result exists.

CPU only (h5py). Writes diagnostic_manifest.json: hashes, trajectory/window selection, noise seeds, video rule and
planned execution scope. The selection uses only trajectory ids, lengths and the demonstrations' own recorded
gripper commands; no policy output is read.
"""
from pathlib import Path
import argparse, subprocess
import h5py
import numpy as np
from hci.io import load_config, read_json, write_json, new_run, finish_run, sha256, digest
from hci.diagnostics import even_pick, window_steps, gripper_phases, phase_of, first_true, FRACTIONS, ACT_HORIZON

NOISE_SEEDS = (910001, 910002, 910003, 910004)
LOSS_TIMESTEPS = (0, 10, 25, 50, 75, 99)


def expected_hashes(path):
    out = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        h, name = line.split(None, 1); out[name.strip().lstrip('*')] = h
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--demo', required=True, help='converted RGB demonstration HDF5'); p.add_argument('--source', required=True, help='source HDF5 before conversion')
    p.add_argument('--demo-manifest', required=True); p.add_argument('--policy', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--backup-sha', required=True, help='SHA256 list written at the end of Stage 1B')
    p.add_argument('--io-audit', default='results/stage1b/policy_io_audit_004.json')
    p.add_argument('--trajectories-per-set', type=int, default=10); p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(); c = load_config(a.config)
    if c['env']['id'] != 'StackCube-v1': p.error('StackCube-v1 only')
    want = expected_hashes(a.backup_sha); dm = read_json(a.demo_manifest); demo = Path(a.demo); source = Path(a.source)
    files = {'demo_h5': (demo, 'data_backup/demos_stack_002/' + demo.name), 'demo_json': (demo.with_suffix('.json'), 'data_backup/demos_stack_002/' + demo.with_suffix('.json').name),
             'source_h5': (source, 'data_backup/demos_stack_002/' + source.name), 'source_json': (source.with_suffix('.json'), 'data_backup/demos_stack_002/' + source.with_suffix('.json').name),
             'checkpoint_60k': (Path(a.checkpoint), 'runs/policy/stack_1b_s1_d100/checkpoints/full_060000.pt'),
             'policy_weights': (Path(a.policy) / 'weights.pt', 'artifacts/policies/stack_1b_s1_d100_i060000/weights.pt'),
             'policy_meta': (Path(a.policy) / 'policy_meta.json', 'artifacts/policies/stack_1b_s1_d100_i060000/policy_meta.json')}
    hashes = {}; problems = []
    for key, (path, name) in files.items():
        if not path.is_file(): problems.append(f'missing {key}: {path}'); continue
        got = sha256(path); hashes[key] = {'path': str(path), 'sha256': got, 'expected': want.get(name), 'match': got == want.get(name)}
        if got != want.get(name): problems.append(f'hash mismatch {key}')
    for key, field in (('demo_h5', 'demo_sha256'), ('demo_json', 'demo_json_sha256'), ('source_h5', 'source_sha256'), ('source_json', 'source_json_sha256')):
        if key in hashes and hashes[key]['sha256'] != dm[field]: problems.append(f'{key} differs from the demo manifest')
    meta = read_json(Path(a.policy) / 'policy_meta.json') if (Path(a.policy) / 'policy_meta.json').is_file() else {}
    if meta and 'policy_weights' in hashes and meta['weights_sha256'] != hashes['policy_weights']['sha256']: problems.append('policy_meta weights hash mismatch')
    if meta and 'checkpoint_60k' in hashes and meta.get('source_checkpoint_sha256') != hashes['checkpoint_60k']['sha256']: problems.append('policy_meta source checkpoint is not full_060000.pt')
    old = read_json(a.io_audit); qualify = read_json('results/stage1b/policy_eval_002/qualify_stack_1b_s1_d100.json')
    if problems: raise SystemExit('Preserved artifacts do not verify: ' + '; '.join(problems))

    info = read_json(demo.with_suffix('.json')); episodes = {e['episode_id']: e for e in info['episodes']}
    structure = {}; trajs = {}
    with h5py.File(demo, 'r') as f, h5py.File(source, 'r') as s:
        ids = sorted(int(k.split('_')[-1]) for k in f if k.startswith('traj_'))
        train_ids = [x['traj_id'] for x in dm['train']]; held_ids = [x['traj_id'] for x in dm['heldout']]
        if ids[:len(train_ids)] != train_ids or ids[len(train_ids):len(train_ids) + len(held_ids)] != held_ids: raise SystemExit('Manifest ids are not the leading ids of the file')
        src_of = {x['traj_id']: x for x in dm['train'] + dm['heldout']}
        g = f[f'traj_{ids[0]}']
        structure = {'state_keys': {k: list(g[f'obs/{pa}/{k}'].shape[1:]) for pa, k in (('agent', 'qpos'), ('agent', 'qvel'), ('extra', 'tcp_pose'))},
                     'cameras_sorted': sorted(g['obs/sensor_data']), 'rgb_shape': list(g['obs/sensor_data/base_camera/rgb'].shape[1:]),
                     'rgb_dtype': str(g['obs/sensor_data/base_camera/rgb'].dtype), 'action_dim': int(g['actions'].shape[1]), 'action_dtype': str(g['actions'].dtype),
                     'env_state_keys': sorted(f'{a_}/{k}' for a_ in g['env_states'] for k in g[f'env_states/{a_}']), 'env_kwargs': info['env_info']['env_kwargs'],
                     'source_action_dim': int(s[f'traj_{src_of[ids[0]]["source_episode_id"]}/actions'].shape[1]),
                     'source_control_mode': read_json(source.with_suffix('.json'))['episodes'][0]['control_mode']}
        lo = np.full(structure['action_dim'], np.inf); hi = -lo; rgb_min, rgb_max = 255, 0; over = []
        for i in train_ids + held_ids:
            g = f[f'traj_{i}']; act = g['actions'][()]; L = len(act); succ = g['success'][()]; first = first_true(succ)
            lo = np.minimum(lo, act.min(0)); hi = np.maximum(hi, act.max(0))
            t_close, t_open = gripper_phases(act[:, -1]); sg = s[f'traj_{src_of[i]["source_episode_id"]}']
            st0 = {f'{a_}/{k}': float(np.abs(g[f'env_states/{a_}/{k}'][0] - sg[f'env_states/{a_}/{k}'][0]).max()) for a_ in g['env_states'] for k in g[f'env_states/{a_}']}
            trajs[i] = {'traj_id': i, 'set': 'train' if i in train_ids else 'heldout', 'source_episode_id': src_of[i]['source_episode_id'], 'scene_seed': src_of[i]['scene_seed'],
                        'reset_kwargs': episodes[i]['reset_kwargs'], 'steps': L, 'first_success_step': first, 'success_steps': int(succ.sum()),
                        'steps_after_first_success': None if first is None else L - first, 'first_success_after_horizon': bool(first is None or first > c['env']['horizon']),
                        'source_steps': int(len(sg['actions'])), 'gripper_first_close_step': t_close, 'gripper_reopen_step': t_open,
                        'converted_vs_source_first_env_state_max_abs': max(st0.values())}
            if episodes[i]['episode_seed'] != src_of[i]['scene_seed']: raise SystemExit(f'traj {i}: episode seed differs from the manifest')
            if trajs[i]['first_success_after_horizon']: over.append(i)
        for i in even_pick(train_ids, 3) + even_pick(held_ids, 2):
            for cam in structure['cameras_sorted']:
                x = f[f'traj_{i}/obs/sensor_data/{cam}/rgb'][()]; rgb_min = min(rgb_min, int(x.min())); rgb_max = max(rgb_max, int(x.max()))
        structure.update(action_min=lo.tolist(), action_max=hi.tolist(), rgb_value_range_sampled=[rgb_min, rgb_max],
                         gripper_values=sorted({float(v) for i in train_ids[:10] for v in np.unique(f[f'traj_{i}/actions'][:, -1])}))
    tails = [trajs[i]['steps_after_first_success'] for i in trajs if trajs[i]['steps_after_first_success'] is not None]
    lengths = {'train_steps_min_mean_max': [min(trajs[i]['steps'] for i in train_ids), float(np.mean([trajs[i]['steps'] for i in train_ids])), max(trajs[i]['steps'] for i in train_ids)],
               'first_success_min_mean_max': [min(trajs[i]['first_success_step'] for i in trajs), float(np.mean([trajs[i]['first_success_step'] for i in trajs])), max(trajs[i]['first_success_step'] for i in trajs)],
               'steps_after_first_success_min_max': [min(tails), max(tails)], 'trajectories_with_first_success_after_horizon': over, 'horizon': c['env']['horizon'],
               'note': 'A demonstration whose first success comes after the horizon cannot succeed under the evaluation horizon; a short tail after the first success is not such a case.'}
    scenes = []
    for name, pool in (('train', train_ids), ('heldout', held_ids)):
        for i in even_pick(pool, a.trajectories_per_set):
            t = trajs[i]; steps = window_steps(t['steps'])
            scenes.append({**t, 'windows': [{'t': w, 'fraction': FRACTIONS[k], 'phase': phase_of(w, t['gripper_first_close_step'], t['gripper_reopen_step'])} for k, w in enumerate(steps)]})
    order = [s_['traj_id'] for s_ in scenes]; tr = [s_['traj_id'] for s_ in scenes if s_['set'] == 'train']; he = [s_['traj_id'] for s_ in scenes if s_['set'] == 'heldout']
    videos = [{'cell': 'F_policy', 'traj_id': i} for i in tr[:3] + he[:3]] + [{'cell': 'R_policy', 'traj_id': i} for i in tr[:1] + he[:1]]
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip().splitlines()
    manifest = {'stage': '1C', 'purpose': 'Diagnosis of the preserved 60k EMA policy. No training, no qualification, not calibration, not the final Test.',
                'code': {'git_head': head, 'uncommitted_files': dirty, 'stage1b_report_code': '700bb1e', 'upstream_commit': read_json('upstream.lock.json') if Path('upstream.lock.json').is_file() else None},
                'config_sha256': digest(c), 'hashes': hashes, 'policy': str(a.policy), 'checkpoint': str(a.checkpoint), 'policy_iteration_tag': meta.get('source_iteration', 60000),
                'kept_results': {'qualify_A': f"{qualify['success_once']}/{qualify['n']}", 'previous_io_audit_pass': old['pass'], 'previous_io_audit_policy': old['policy']},
                'demo_structure': structure, 'demo_lengths': lengths,
                'selection_rule': f'{a.trajectories_per_set} of the training ids and {a.trajectories_per_set} of the held-out ids at equal spacing over the ascending id list; '
                                  f'windows at {list(FRACTIONS)} of the range [0, steps-{ACT_HORIZON}] so that all {ACT_HORIZON} executed actions are recorded (no padding in the executed segment)',
                'phase_rule': 'From the demonstration gripper commands only: approach = executed segment ends before the first close; transport = between first close and re-open; '
                              'place = at/after re-open; boundary = segment crosses a change; unknown = no close/re-open found',
                'scenes': scenes, 'scene_order': order, 'n_histories': sum(len(s_['windows']) for s_ in scenes),
                'noise_seeds': list(NOISE_SEEDS), 'loss_timesteps': list(LOSS_TIMESTEPS),
                'rollout_cells': {'F_expert': 'converted-data initialisation, recorded actions', 'F_policy': 'converted-data initialisation, 60k EMA closed loop',
                                  'R_expert': 'runtime reset(seed), recorded actions', 'R_policy': 'runtime reset(seed), 60k EMA closed loop',
                                  'Fdemo_expert': 'environment built from the demo JSON kwargs, converted-data initialisation, recorded actions (observation reproducibility only)'},
                'F_definition': 'env.reset(**reset_kwargs) then set_state_dict(first env state of the SOURCE trajectory), as replay_trajectory --use-first-env-state does at the pinned commit; '
                                'the stored state is used at the initial step only',
                'video_rule': 'fixed before results: F_policy for the first three selected training and first three selected held-out scenes, R_policy for the first of each', 'videos': videos,
                'usage_note': 'The 20 held-out demonstrations are used for development diagnosis from this stage on. Training-scene rollouts measure fit, not generalisation.',
                'limits': {'max_rollouts': 100, 'max_histories': 100, 'noise_draws': len(NOISE_SEEDS), 'budget_usd_cap': 1.0}}
    print(f"scenes train={tr} heldout={he} histories={manifest['n_histories']} first-success-after-horizon={over}")
    print('phases:', {ph: sum(1 for s_ in scenes for w in s_['windows'] if w['phase'] == ph) for ph in ('approach', 'transport', 'place', 'boundary', 'unknown')})
    if a.dry_run: print('dry run: nothing written'); return
    out = new_run(a.out, 'stage1c-manifest', c); write_json(out / 'diagnostic_manifest.json', manifest); finish_run(out, n_scenes=len(scenes), n_histories=manifest['n_histories'])


if __name__ == '__main__':
    main()
