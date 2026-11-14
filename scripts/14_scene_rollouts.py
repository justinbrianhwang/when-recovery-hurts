"""Stage 1C / D2: can the 60k EMA policy solve scenes it was trained on, and do the recorded expert actions still work?

Cells (per selected demonstration scene, see diagnostic_manifest.json):
  F_expert      project runtime env, converted-data initialisation, recorded actions replayed
  F_policy      project runtime env, converted-data initialisation, policy closed loop (every action from the policy)
  R_expert      project runtime env, Runtime.reset(seed), recorded actions replayed
  R_policy      project runtime env, Runtime.reset(seed), policy closed loop
  Fdemo_expert  env built from the demo JSON kwargs (as replay_trajectory did), converted-data initialisation,
                recorded actions. Observation reproducibility only; the policy is never run in this env.

F = env.reset(**reset_kwargs) then set_state_dict(first env state of the SOURCE trajectory): the sequence
replay_trajectory --use-first-env-state executes at the pinned commit. The stored state is written once, before the
first observation; nothing is written during a rollout and the policy never sees a demonstration observation.

Ground truth (cube poses, grasp flag, env states) is logged and compared with the file. It is not a policy input.
A failing expert replay is kept as a row (`replay_failure`); no scene is dropped.

Note on the file: at the pinned commit replay_trajectory overwrites the FIRST stored env state of the converted file with
the source trajectory's SECOND state, while the first stored observation is rendered from the source's first state.
So `env_states[0]` of the converted file is not the state the first observation shows (measured before any GPU run:
up to 8.4e-3 apart on the selected scenes). The initial state is therefore also compared with the source's first state.
"""
from pathlib import Path
import argparse, csv, json, time
import h5py
import numpy as np
import torch
from hci.io import load_config, read_json, write_json, new_run, finish_run, sha256, runtime_info
from hci.schema import STATE_PATHS
from hci.trees import as_numpy, clone
from hci.diagnostics import step_state, stage_events, rgb_diff_stats

CELLS = ('F_expert', 'F_policy', 'R_expert', 'R_policy', 'Fdemo_expert')
CAMERAS = ('base_camera', 'hand_camera')


def live_obs(raw):
    """(state 25, rgb HxWx6 uint8) of one raw observation, cameras in sorted order."""
    state = np.concatenate([as_numpy(raw[pa][k])[0] for pa, k in STATE_PATHS], -1).astype(np.float32)
    rgb = np.concatenate([as_numpy(raw['sensor_data'][n]['rgb'])[0] for n in CAMERAS], -1).astype(np.uint8)
    return state, rgb


def file_obs(g, t):
    state = np.concatenate([g[f'obs/{pa}/{k}'][t] for pa, k in STATE_PATHS], -1).astype(np.float32)
    rgb = np.concatenate([g[f'obs/sensor_data/{n}/rgb'][t] for n in CAMERAS], -1).astype(np.uint8)
    return state, rgb


def env_state_diff(u, g, t):
    """Largest absolute difference between the live simulator state and the recorded env state at index t."""
    live = u.get_state_dict(); out = {}
    for kind in g['env_states']:
        for name in g[f'env_states/{kind}']:
            out[f'{kind}/{name}'] = float(np.abs(as_numpy(live[kind][name])[0] - g[f'env_states/{kind}/{name}'][t]).max())
    return out


def source_state_diff(u, state0):
    """Live simulator state vs the SOURCE trajectory's first env state (what the F initialisation writes)."""
    live = u.get_state_dict()
    return max(float(np.abs(as_numpy(live[kind][name])[0] - np.asarray(v)).max()) for kind in state0 for name, v in state0[kind].items())


def compare(u, raw, g, t):
    s, rgb = live_obs(raw); fs, frgb = file_obs(g, t); es = env_state_diff(u, g, t)
    row = {'t': t, 'proprio_max_abs': float(np.abs(s - fs).max()), 'env_state_max_abs': max(es.values()),
           'cubes_state_max_abs': max(v for k, v in es.items() if 'cube' in k), 'robot_state_max_abs': max(v for k, v in es.items() if k.startswith('articulations'))}
    for j, cam in enumerate(CAMERAS):
        for k, v in rgb_diff_stats(rgb[..., 3 * j:3 * j + 3], frgb[..., 3 * j:3 * j + 3]).items(): row[f'{cam}_{k}'] = v
    return row, s, rgb


def frame_of(raw):
    return np.concatenate([as_numpy(raw['sensor_data'][n]['rgb'])[0] for n in CAMERAS], axis=1).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True); p.add_argument('--manifest', required=True, help='diagnostic_manifest.json'); p.add_argument('--out', required=True)
    p.add_argument('--demo', required=True); p.add_argument('--source', required=True); p.add_argument('--policy', required=True)
    p.add_argument('--upstream', default='third_party/ManiSkill'); p.add_argument('--cells', nargs='+', default=list(CELLS), choices=CELLS)
    p.add_argument('--limit', type=int, help='first N scenes of each set only (throughput probe; not a result)')
    p.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    p.add_argument('--other-policy', action='store_true', help='Stage 1D: run a policy other than the one fixed in the manifest on the same fixed scenes, histories and noise (demo files are still checked)')
    a = p.parse_args(); c = load_config(a.config); m = read_json(a.manifest); horizon = c['env']['horizon']
    if c['env']['id'] != 'StackCube-v1' or c['env']['control_mode'] != 'pd_ee_delta_pos': p.error('StackCube-v1 with pd_ee_delta_pos only')
    scenes = m['scenes']
    if a.limit: scenes = [s for name in ('train', 'heldout') for s in [x for x in m['scenes'] if x['set'] == name][:a.limit]]
    video_keys = {(v['cell'], v['traj_id']) for v in m['videos']}
    print(f"cells={a.cells} scenes={[s['traj_id'] for s in scenes]} rollouts={len(a.cells) * len(scenes)}", flush=True)
    if a.dry_run: return
    for key, path in (('demo_h5', a.demo), ('source_h5', a.source), ('policy_weights', Path(a.policy) / 'weights.pt')):
        if key == 'policy_weights' and a.other_policy: continue
        if sha256(path) != m['hashes'][key]['sha256']: raise SystemExit(f'{key} is not the file fixed in the manifest')
    import gymnasium as gym
    import imageio.v2 as imageio
    from mani_skill.trajectory import utils as trajectory_utils
    from hci.experiment import runtime
    from hci.simulation import success
    out = new_run(a.out, 'stage1c-scene-rollouts', c); (out / 'traces').mkdir(); (out / 'replay_obs').mkdir(); (out / 'videos').mkdir()
    r = runtime(c, a.policy, a.upstream); u = r.env; demo_env = None; rows = []; steps_rows = []; started = time.perf_counter()
    info = read_json(Path(a.demo).with_suffix('.json'))
    if 'Fdemo_expert' in a.cells:
        demo_env = gym.make(info['env_info']['env_id'], **info['env_info']['env_kwargs']).unwrapped

    def init_F(seed, reset_kwargs, state0):
        """Runtime bookkeeping of Runtime.reset, with the converted-data initialisation instead of reset(seed) alone."""
        r.episode = int(seed); r.repeat = 0; r.t = 0; r.intervention = None; r.intervention_t = None; r.actions = []; r.retreat_info = None
        u.reset(**reset_kwargs); u.set_state_dict(state0); r.raw = u.get_obs()
        r.done = success(u.evaluate()); r.y = int(r.done); r.reason = 'success' if r.done else None
        r.policy.reset(r.policy.convert(r.raw)); r.tcp_history = [as_numpy(r.raw['extra']['tcp_pose'][0, :3]).copy()]
        r.initial_state = clone(u.get_state_dict())

    try:
        with h5py.File(a.demo, 'r') as f, h5py.File(a.source, 'r') as src:
            for cell in a.cells:
                init, actor = cell.split('_')
                for sc in scenes:
                    i = sc['traj_id']; seed = sc['scene_seed']; g = f[f'traj_{i}']; actions = g['actions'][()]; L = len(actions)
                    if sc['reset_kwargs'].get('seed') != seed: raise RuntimeError(f'traj {i}: reset_kwargs seed differs from the scene seed')
                    state0 = trajectory_utils.dict_to_list_of_dicts(src[f"traj_{sc['source_episode_id']}"]['env_states'])[0]
                    windows = sorted({t for w in sc['windows'] for t in (max(0, w['t'] - 1), w['t'])}); keep = {}; tick = time.perf_counter()
                    frames = [] if (cell, i) in video_keys else None; trace = []; diffs = []
                    if cell == 'Fdemo_expert':
                        env = demo_env; env.reset(**sc['reset_kwargs']); env.set_state_dict(state0); raw = env.get_obs()
                        initial = step_state(env, 0); d0, s, rgb = compare(env, raw, g, 0); diffs.append(d0); src0 = source_state_diff(env, state0)
                        if 0 in windows: keep[0] = (s, rgb)
                        t = 0; done = initial['success']; reason = 'success' if done else None
                        while not done:
                            if t >= L: reason = 'actions_exhausted'; break
                            if t >= horizon: reason = 'horizon'; break
                            raw, _, _, _, _ = env.step(torch.as_tensor(actions[t][None], device=env.device)); t += 1
                            st = step_state(env, t, actions[t - 1]); trace.append(st); d, s, rgb = compare(env, raw, g, t); diffs.append(d)
                            if t in windows: keep[t] = (s, rgb)
                            if st['success']: done = True; reason = 'success'
                        result = {'success': int(done), 'steps': t, 'reason': reason, 'policy_calls': 0}
                    else:
                        if init == 'F': init_F(seed, sc['reset_kwargs'], state0)
                        else: r.reset(seed)
                        initial = step_state(u, 0); d0, s, rgb = compare(u, r.raw, g, 0); diffs.append(d0); src0 = source_state_diff(u, state0)
                        if 0 in windows: keep[0] = (s, rgb)
                        if frames is not None: frames.append(frame_of(r.raw))
                        old = type(r).advance

                        def advance(action, old=old, trace=trace, frames=frames):
                            z = old(r, action); trace.append(step_state(u, r.t, z))
                            if frames is not None: frames.append(frame_of(r.raw))
                            return z
                        r.advance = advance
                        try:
                            if actor == 'policy':
                                res = r.complete(); result = {'success': int(res['success']), 'steps': res['steps'], 'reason': res['reason'], 'policy_calls': res['policy_calls']}
                            else:
                                reason = None
                                while not r.done:
                                    if r.t >= L: reason = 'actions_exhausted'; break
                                    r.advance(actions[r.t]); d, s, rgb = compare(u, r.raw, g, r.t); diffs.append(d)
                                    if r.t in windows: keep[r.t] = (s, rgb)
                                result = {'success': int(r.y), 'steps': r.t, 'reason': reason or r.reason, 'policy_calls': 0}
                        finally:
                            del r.advance
                    ev = stage_events(initial, trace)
                    row = {'set': sc['set'], 'traj_id': i, 'scene_seed': seed, 'cell': cell, 'init': init, 'actor': actor, 'env': 'demo_json_kwargs' if cell == 'Fdemo_expert' else 'project_runtime',
                           'executed': True, **result, 'replay_failure': bool(actor == 'expert' and not result['success']),
                           'demo_steps': L, 'demo_first_success_step': sc['first_success_step'], **ev,
                           'init_proprio_max_abs_vs_file': d0['proprio_max_abs'], 'init_cubes_state_max_abs_vs_file': d0['cubes_state_max_abs'],
                           'init_robot_state_max_abs_vs_file': d0['robot_state_max_abs'],
                           'init_base_rgb_mae_u8': d0['base_camera_mae_u8'], 'init_base_rgb_max_u8': d0['base_camera_max_u8'],
                           'init_hand_rgb_mae_u8': d0['hand_camera_mae_u8'], 'init_hand_rgb_max_u8': d0['hand_camera_max_u8'],
                           'init_env_state_max_abs_vs_source_first_state': src0,
                           'init_tcp': initial['tcp'], 'init_cubeA': initial['cubeA'], 'init_cubeB': initial['cubeB']}
                    if actor == 'expert':
                        for k in ('proprio_max_abs', 'cubes_state_max_abs', 'robot_state_max_abs', 'base_camera_mae_u8', 'hand_camera_mae_u8', 'base_camera_frac_ge8', 'hand_camera_frac_ge8'):
                            row[f'replay_{k}_max_over_steps'] = max(d[k] for d in diffs); row[f'replay_{k}_mean_over_steps'] = float(np.mean([d[k] for d in diffs]))
                        row['replay_first_step_proprio_over_1e-4'] = next((d['t'] for d in diffs if d['proprio_max_abs'] > 1e-4), None)
                        for d in diffs: steps_rows.append({'cell': cell, 'set': sc['set'], 'traj_id': i, **d})
                        missing = [t for t in windows if t not in keep]
                        row['window_frames_missing'] = missing
                        np.savez_compressed(out / 'replay_obs' / f'{cell}_traj{i}.npz', steps=np.array(sorted(keep), np.int32),
                                            state=np.stack([keep[t][0] for t in sorted(keep)]), rgb=np.stack([keep[t][1] for t in sorted(keep)]))
                    row['wall_seconds'] = time.perf_counter() - tick; rows.append(row)
                    with (out / 'traces' / f'{cell}_traj{i}.jsonl').open('x', encoding='utf-8') as fh:
                        fh.write(json.dumps({'initial': initial}) + '\n')
                        for st in trace: fh.write(json.dumps(st) + '\n')
                    if frames is not None:
                        with imageio.get_writer(str(out / 'videos' / f'{cell}_{sc["set"]}_traj{i}.mp4'), fps=20) as w:
                            for fr in frames: w.append_data(fr)
                    print(f"{cell} {sc['set']} traj={i} success={row['success']} steps={row['steps']} reason={row['reason']} stage={ev['furthest_stage']} "
                          f"init_cubes_diff={d0['cubes_state_max_abs']:.2e} ({row['wall_seconds']:.1f}s)", flush=True)

        def table(path, data):
            keys = []
            for x in data:
                for k in x:
                    if k not in keys: keys.append(k)
            with open(path, 'x', newline='', encoding='utf-8') as fh:
                w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
                for x in data: w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in x.items()})
        table(out / 'scene_rollouts.csv', rows)
        if steps_rows: table(out / 'replay_obs_diff.csv', steps_rows)
        cells = {}
        for cell in a.cells:
            for name in ('train', 'heldout'):
                sub = [x for x in rows if x['cell'] == cell and x['set'] == name]
                if not sub: continue
                stages = {}
                for x in sub: stages[x['furthest_stage']] = stages.get(x['furthest_stage'], 0) + 1
                cells[f'{cell}/{name}'] = {'n': len(sub), 'success': sum(x['success'] for x in sub), 'furthest_stage': stages,
                                          'success_traj_ids': [x['traj_id'] for x in sub if x['success']], 'reasons': sorted({x['reason'] for x in sub}),
                                          'init_cubes_state_max_abs_vs_file_max': max(x['init_cubes_state_max_abs_vs_file'] for x in sub),
                                          'init_robot_state_max_abs_vs_file_max': max(x['init_robot_state_max_abs_vs_file'] for x in sub)}
                if cell.endswith('expert'):
                    cells[f'{cell}/{name}'].update(replay_proprio_max_abs=max(x['replay_proprio_max_abs_max_over_steps'] for x in sub),
                                                  replay_cubes_state_max_abs=max(x['replay_cubes_state_max_abs_max_over_steps'] for x in sub),
                                                  replay_base_rgb_mae_u8_mean=float(np.mean([x['replay_base_camera_mae_u8_mean_over_steps'] for x in sub])),
                                                  replay_hand_rgb_mae_u8_mean=float(np.mean([x['replay_hand_camera_mae_u8_mean_over_steps'] for x in sub])))
        report = {'cells': cells, 'n_rollouts': len(rows), 'demo_env_kwargs_used_for_Fdemo': info['env_info']['env_kwargs'] if demo_env is not None else None, 'limit': a.limit, 'horizon': horizon, 'policy': a.policy, 'manifest_sha256': sha256(a.manifest),
                  'wall_seconds': time.perf_counter() - started, 'runtime': runtime_info(),
                  'scope': 'Development diagnosis on demonstration scenes. Training-scene success measures fit, not generalisation. Small n: counts, not rates to compare. '
                           'Not a qualification, not calibration, not the final Test.' + (' THROUGHPUT PROBE (limited scenes), not a result.' if a.limit else '')}
        write_json(out / 'scene_rollouts_summary.json', report); finish_run(out, n_rollouts=len(rows))
        print(json.dumps(cells, indent=1))
    finally:
        r.env.close()
        if demo_env is not None: demo_env.close()


if __name__ == '__main__':
    main()
