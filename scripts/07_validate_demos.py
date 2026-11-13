"""Validate a converted RGB demonstration file and write the manifest that fixes which demonstrations are used.

The upstream loader takes the first N trajectories in ascending `traj_<id>` order. This script records exactly that
choice (train = first --num-train, held-out = the following --num-heldout), maps every adopted trajectory back to
its source episode through the scene seed, lists source episodes that were dropped by the success filter, checks
shapes/ranges, and replays some demonstrations by action in the project environment at the project horizon.
Object poses read here are used for diagnostics only; they are never policy or monitor inputs.
"""
from pathlib import Path
import argparse, json
import h5py
import numpy as np
from hci.io import load_config, read_json, write_json, new_run, finish_run, sha256, UPSTREAM_COMMIT

CAMERAS = ['base_camera', 'hand_camera']


def traj_ids(f): return sorted((int(k.split('_')[-1]) for k in f if k.startswith('traj_')))


def check_structure(demo, info, c):
    """Static checks that need no simulator. Returns (per-trajectory rows, list of problems)."""
    problems = []; rows = []; kw = info['env_info']['env_kwargs']; e = c['env']
    for key, want, got in [('env_id', e['id'], info['env_info']['env_id']), ('obs_mode', e['obs_mode'], kw.get('obs_mode')),
                           ('control_mode', e['control_mode'], kw.get('control_mode')), ('sim_backend', e['sim_backend'], kw.get('sim_backend')),
                           ('num_envs', 1, kw.get('num_envs', 1)), ('upstream_commit', UPSTREAM_COMMIT, info.get('commit_info', {}).get('commit_id'))]:
        if want != got: problems.append(f'{key}: expected {want!r}, file has {got!r}')
    if 'robot_uids' in kw and kw['robot_uids'] != e['robot_uid']: problems.append(f"robot_uids {kw['robot_uids']!r} != {e['robot_uid']!r}")
    episodes = {x['episode_id']: x for x in info['episodes']}
    with h5py.File(demo, 'r') as f:
        ids = traj_ids(f)
        if sorted(episodes) != ids: problems.append('trajectory ids in HDF5 and JSON differ')
        for i in ids:
            g = f[f'traj_{i}']; ep = episodes.get(i, {}); a = g['actions'][()]; L = len(a); bad = []
            cams = sorted(g['obs/sensor_data'])
            if cams != CAMERAS: bad.append(f'cameras {cams}')
            for cam in cams:
                d = g[f'obs/sensor_data/{cam}/rgb']
                if d.shape != (L + 1, 128, 128, 3) or d.dtype != np.uint8: bad.append(f'{cam} rgb {d.shape} {d.dtype}')
            for name, dim in (('agent/qpos', 9), ('agent/qvel', 9), ('extra/tcp_pose', 7)):
                d = g[f'obs/{name}']
                if d.shape != (L + 1, dim): bad.append(f'{name} {d.shape}')
                elif not np.isfinite(d[()]).all(): bad.append(f'{name} non-finite')
            if a.ndim != 2 or a.shape[1] != 4: bad.append(f'action shape {a.shape}')
            if not np.isfinite(a).all(): bad.append('non-finite action')
            if np.abs(a).max() > 1 + 1e-6: bad.append(f'action outside [-1,1]: {np.abs(a).max()}')
            if not ep.get('success', False): bad.append('JSON success flag is not true')
            if ep.get('control_mode') != e['control_mode']: bad.append(f"episode control_mode {ep.get('control_mode')}")
            if ep.get('elapsed_steps') != L: bad.append(f"elapsed_steps {ep.get('elapsed_steps')} != {L}")
            actors = {n: g[f'env_states/actors/{n}'][0, :3].astype(float).tolist() for n in g['env_states/actors']} if 'env_states/actors' in g else {}
            rows.append({'traj_id': i, 'scene_seed': ep.get('episode_seed'), 'steps': L, 'problems': bad,
                         'arm_action_abs_max': float(np.abs(a[:, :3]).max()), 'gripper_values': sorted({round(float(x), 4) for x in a[:, 3]}),
                         'initial_actor_positions': actors})
            problems += [f'traj_{i}: {b}' for b in bad]
    return rows, problems


def source_mapping(rows, source_info):
    """Source episodes (in source order, up to the last adopted one) that are absent from the converted file."""
    order = [(x['episode_id'], x['episode_seed']) for x in source_info['episodes']]
    seeds = [s for _, s in order]
    if len(seeds) != len(set(seeds)): raise ValueError('Source scene seeds are not unique; cannot map trajectories back')
    by_seed = {s: i for i, s in order}; adopted = [r['scene_seed'] for r in rows]
    unknown = [s for s in adopted if s not in by_seed]
    if unknown: raise ValueError(f'Converted trajectories with seeds absent from the source: {unknown[:5]}')
    positions = [seeds.index(s) for s in adopted]
    if positions != sorted(positions): raise ValueError('Converted trajectories are not in source order')
    last = max(positions) if positions else -1; kept = set(adopted)
    dropped = [{'source_episode_id': i, 'scene_seed': s} for i, s in order[:last + 1] if s not in kept]
    return {r['traj_id']: by_seed[r['scene_seed']] for r in rows}, dropped


def replay(c, demo, rows, out, record):
    """Action replay in the project environment: reset(scene seed), execute the stored actions, horizon from config."""
    import imageio.v2 as imageio, torch
    from hci.simulation import make_env, success
    from hci.trees import as_numpy
    env = make_env(c); results = []
    try:
        with h5py.File(demo, 'r') as f:
            for n, r in enumerate(rows):
                g = f[f"traj_{r['traj_id']}"]; actions = g['actions'][()]; raw, info = env.reset(seed=int(r['scene_seed']))
                state0 = {k: as_numpy(v)[0, :3] for k, v in env.get_state_dict()['actors'].items()}
                init_err = max([float(np.abs(state0[k] - g[f'env_states/actors/{k}'][0, :3]).max()) for k in state0 if f'env_states/actors/{k}' in g] + [0.])
                rgb0 = max(int(np.abs(as_numpy(raw['sensor_data'][cam]['rgb'])[0].astype(int) - g[f'obs/sensor_data/{cam}/rgb'][0].astype(int)).max()) for cam in CAMERAS)
                frames = []; done_at = None
                def frame(): frames.append(np.concatenate([as_numpy(raw['sensor_data'][cam]['rgb'])[0] for cam in CAMERAS], 1).astype(np.uint8))
                if n < record: frame()
                for t, a in enumerate(actions[:c['env']['horizon']]):
                    raw, _, _, _, info = env.step(torch.as_tensor(a[None], device=env.device))
                    if n < record: frame()
                    if success(info) and done_at is None: done_at = t + 1
                tcp_err = float(np.abs(as_numpy(raw['extra']['tcp_pose'])[0] - g['obs/extra/tcp_pose'][min(len(actions), c['env']['horizon'])]).max())
                results.append({'traj_id': r['traj_id'], 'scene_seed': r['scene_seed'], 'steps': len(actions), 'success_once': done_at is not None,
                                'first_success_step': done_at, 'success_at_last_step': bool(success(info)),
                                'initial_actor_position_max_abs_vs_file': init_err, 'initial_rgb_max_abs_vs_file': rgb0,
                                'final_tcp_pose_max_abs_vs_file': tcp_err})
                if frames:
                    with imageio.get_writer(str(out / f"replay_traj_{r['traj_id']}.mp4"), fps=20) as w:
                        for x in frames: w.append_data(x)
                print('replayed', results[-1], flush=True)
    finally:
        env.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--demo', required=True); p.add_argument('--source', required=True)
    p.add_argument('--out', required=True); p.add_argument('--num-train', type=int, default=100); p.add_argument('--num-heldout', type=int, default=20)
    p.add_argument('--replay', type=int, default=10); p.add_argument('--record', type=int, default=3)
    p.add_argument('--convert-command', default='', help='the exact conversion command, stored verbatim')
    p.add_argument('--splits', default='configs/splits.json'); p.add_argument('--policy-eval-manifest', default='configs/policy_eval_splits.json')
    a = p.parse_args(); c = load_config(a.config); demo = Path(a.demo).resolve(); source = Path(a.source).resolve()
    if c['env']['id'] != 'StackCube-v1' or c['env']['control_mode'] != 'pd_ee_delta_pos': p.error('This script is validated for StackCube-v1 with pd_ee_delta_pos (4-D actions) only')
    demo_json = demo.with_suffix('.json'); source_json = source.with_suffix('.json')
    for f in (demo, demo_json, source, source_json):
        if not f.is_file(): p.error(f'Missing file: {f}')
    info = read_json(demo_json); rows, problems = check_structure(demo, info, c)
    mapping, dropped = source_mapping(rows, read_json(source_json))
    if len(rows) < a.num_train + a.num_heldout: problems.append(f'only {len(rows)} successful trajectories; need {a.num_train}+{a.num_heldout}')
    seeds = [r['scene_seed'] for r in rows]
    if len(set(seeds)) != len(seeds): problems.append('duplicate scene seeds: demonstrations are not independent')
    train = rows[:a.num_train]; heldout = rows[a.num_train:a.num_train + a.num_heldout]
    pos = [tuple(np.round(np.concatenate([np.asarray(v) for _, v in sorted(r['initial_actor_positions'].items())]), 4)) for r in train]
    if len(set(pos)) != len(pos): problems.append('identical initial object placements inside the training set')
    arr = np.array(pos) if pos else np.zeros((0, 1))
    overlaps = {}
    for label, path in (('splits', a.splits), ('policy_eval', a.policy_eval_manifest)):
        for name, ids in read_json(path)['splits'].items():
            both = sorted(set(ids) & set(seeds))
            if both: overlaps[f'{label}/{name}'] = both
    # Evaluation seeds must never be demonstration scenes. The restore-audit split measures determinism, not
    # performance, so an overlap there is reported but does not fail the check.
    for name, both in overlaps.items():
        if name != 'splits/audit': problems.append(f'demonstration scenes overlap {name}: {both[:10]}')
    out = new_run(a.out, 'validate-demos', c)
    replayed = replay(c, demo, train[:a.replay], out, a.record) if a.replay else []
    failed = [x['traj_id'] for x in replayed if not x['success_once']]
    if failed: problems.append(f'action replay did not succeed for trajectories {failed}')
    manifest = {'pass': not problems, 'problems': problems, 'demo_path': str(demo), 'demo_sha256': sha256(demo), 'demo_json_sha256': sha256(demo_json),
                'source_path': str(source), 'source_sha256': sha256(source), 'source_json_sha256': sha256(source_json),
                'convert_command': a.convert_command, 'env_info': info['env_info'], 'commit_info': info.get('commit_info'),
                'selection_rule': 'ascending traj id (the upstream loader order); train = first num_train, heldout = next num_heldout; success filter applied by replay_trajectory',
                'n_successful_trajectories': len(rows), 'num_train': len(train), 'num_heldout': len(heldout),
                'train': [{'traj_id': r['traj_id'], 'source_episode_id': mapping[r['traj_id']], 'scene_seed': r['scene_seed'], 'steps': r['steps']} for r in train],
                'heldout': [{'traj_id': r['traj_id'], 'source_episode_id': mapping[r['traj_id']], 'scene_seed': r['scene_seed'], 'steps': r['steps']} for r in heldout],
                'dropped_source_episodes': dropped, 'all_scene_seeds': seeds, 'seed_overlap_with_declared_splits': overlaps,
                'train_transitions': int(sum(r['steps'] for r in train)),
                'train_steps_min_mean_max': [int(min(r['steps'] for r in train)), float(np.mean([r['steps'] for r in train])), int(max(r['steps'] for r in train))] if train else None,
                'train_arm_action_abs_max': max((r['arm_action_abs_max'] for r in train), default=None),
                'train_gripper_values': sorted({v for r in train for v in r['gripper_values']}),
                'train_initial_placement_distinct': len(set(pos)), 'train_initial_placement_std': arr.std(0).round(5).tolist(),
                'action_replay': replayed,
                'note': 'Initial object positions are privileged diagnostics for diversity only. Replay compares against file values that were '
                        'recorded with --use-first-env-state; small differences are reported, not hidden.'}
    write_json(out / 'demo_manifest.json', manifest); finish_run(out, demo_pass=manifest['pass'])
    print(json.dumps({k: manifest[k] for k in ('pass', 'problems', 'n_successful_trajectories', 'num_train', 'num_heldout', 'dropped_source_episodes',
                                               'seed_overlap_with_declared_splits', 'train_gripper_values', 'train_arm_action_abs_max')}, indent=2))
    if not manifest['pass']: raise SystemExit(2)


if __name__ == '__main__':
    main()
