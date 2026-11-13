"""Policy input/output parity audit (run before any long training).

Three observation paths must mean the same thing:
  A  training data      : demo HDF5 -> upstream SmallDemoDataset with the in-memory patches (convert_obs path)
  B  in-training eval   : upstream make_eval_envs + StrictRGBObservationWrapper + FrameStack + CPUGymWrapper
  C  deployment adapter : hci make_env + DiffusionPolicy.convert / history / stacked
"Upstream" here means the pinned trainer WITH this project's observation whitelist; the stock privileged-state
agent has a different input dimension and is not comparable with these checkpoints.

Part 1  artifact tensors : source checkpoint `ema_agent` vs exported `weights.pt`
Part 2  same raw frames  : identical recorded raw observations pushed through A, B and C
Part 3  live environments: B and C environments reset to the same scene seed and driven by the same actions
Part 4  actions          : upstream Agent.get_action on B's observation vs the adapter on C's, same weights, same RNG
A passing state-restore audit is not evidence for any of this. Tolerances come from the config and are not relaxed.
"""
from pathlib import Path
from functools import partial
from types import SimpleNamespace
import argparse, copy, csv, json
import h5py
import numpy as np
import torch
from hci.io import load_config, read_json, write_json, new_run, finish_run, sha256, seed_for, runtime_info
from hci.schema import STATE_PATHS

CHECK_STEPS = (0, 1, 8, 12, 16, 20)   # start, first step, chunk boundaries (8, 16), inside a chunk (12, 20)


def maxabs(a, b):
    a = torch.as_tensor(np.asarray(a.cpu() if torch.is_tensor(a) else a)).double(); b = torch.as_tensor(np.asarray(b.cpu() if torch.is_tensor(b) else b)).double()
    if a.shape != b.shape: return float('inf')
    return float((a - b).abs().max()) if a.numel() else 0.


def tensor_parity(checkpoint, artifact):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'ema_agent' not in ck: raise KeyError('Source checkpoint has no ema_agent')
    src = ck['ema_agent']; dst = torch.load(Path(artifact) / 'weights.pt', map_location='cpu', weights_only=True)
    rows = [{'tensor': k, 'shape': list(src[k].shape), 'dtype': str(src[k].dtype), 'same_shape_dtype': src[k].shape == dst[k].shape and src[k].dtype == dst[k].dtype,
             'max_abs': maxabs(src[k], dst[k]), 'bit_equal': bool(torch.equal(src[k], dst[k]))} for k in src if k in dst]
    meta = read_json(Path(artifact) / 'policy_meta.json')
    out = {'missing_in_artifact': sorted(set(src) - set(dst)), 'extra_in_artifact': sorted(set(dst) - set(src)), 'n_tensors': len(rows),
           'max_abs_over_tensors': max([r['max_abs'] for r in rows] + [0.]), 'all_bit_equal': all(r['bit_equal'] for r in rows),
           'source_checkpoint_sha256': sha256(checkpoint), 'meta_source_checkpoint_sha256': meta['source_checkpoint_sha256'],
           'weights_sha256': sha256(Path(artifact) / 'weights.pt'), 'meta_weights_sha256': meta['weights_sha256'],
           'raw_agent_differs_from_ema': any(not torch.equal(ck['agent'][k], src[k]) for k in src) if 'agent' in ck else None,
           'buffers_note': 'EMAModel.copy_to copies parameters only; any tensor listed in state_dict but not in named_parameters is a buffer.',
           'tensors': rows}
    out['pass'] = bool(not out['missing_in_artifact'] and not out['extra_in_artifact'] and out['all_bit_equal'] and rows
                       and all(r['same_shape_dtype'] for r in rows) and out['source_checkpoint_sha256'] == out['meta_source_checkpoint_sha256']
                       and out['weights_sha256'] == out['meta_weights_sha256'])
    return out, src


def raw_frame(g, t):
    """One recorded raw observation as a batched torch dict, in the layout the simulator returns."""
    obs = {'agent': {}, 'extra': {}, 'sensor_data': {}}
    for parent in ('agent', 'extra'):
        for k in g[f'obs/{parent}']: obs[parent][k] = torch.from_numpy(g[f'obs/{parent}/{k}'][t][None])
    for cam in g['obs/sensor_data']: obs['sensor_data'][cam] = {'rgb': torch.from_numpy(g[f'obs/sensor_data/{cam}/rgb'][t][None])}
    return obs


def find_wrapper(env, cls):
    while env is not None:
        if isinstance(env, cls): return env
        env = getattr(env, 'env', None)
    raise LookupError(f'{cls.__name__} is not in the evaluation wrapper stack')


def env_facts(env):
    """Configuration that must be identical between the training-time evaluation env and the deployment env."""
    u = env.unwrapped; arm = u.agent.controller.controllers['arm'].config; grip = u.agent.controller.controllers['gripper'].config
    def val(x): return x.tolist() if hasattr(x, 'tolist') else x
    cams = {name: {k: np.asarray(v.cpu() if torch.is_tensor(v) else v, dtype=float).round(7).tolist() for k, v in params.items()}
            for name, params in sorted(u.get_sensor_params().items())}
    return {'robot_uid': u.agent.uid, 'control_mode': u.control_mode, 'obs_mode': u.obs_mode, 'sim_freq': u.sim_freq, 'control_freq': u.control_freq,
            'backend': u.backend.sim_backend if hasattr(u, 'backend') else None,
            'arm': {'class': type(arm).__name__, 'frame': arm.frame, 'pos_lower': val(arm.pos_lower), 'pos_upper': val(arm.pos_upper),
                    'normalize_action': arm.normalize_action, 'use_target': arm.use_target, 'use_delta': getattr(arm, 'use_delta', None)},
            'gripper': {'class': type(grip).__name__, 'lower': val(grip.lower), 'upper': val(grip.upper), 'normalize_action': getattr(grip, 'normalize_action', None)},
            'action_low': u.single_action_space.low.tolist(), 'action_high': u.single_action_space.high.tolist(),
            'cameras': sorted(u._sensors), 'camera_params': cams,
            'camera_shader': {n: getattr(getattr(s, 'config', None), 'shader_pack', None) for n, s in sorted(u._sensors.items())},
            'raw_obs_keys': {k: sorted(v) for k, v in u.get_obs().items() if isinstance(v, dict)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--policy', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--demo', required=True); p.add_argument('--out', required=True); p.add_argument('--upstream', default='third_party/ManiSkill')
    p.add_argument('--trajectories', type=int, default=6); p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    a = p.parse_args(); c = load_config(a.config); atol_s = c['audit']['state_atol']; atol_a = c['audit']['action_atol']
    if c['env']['id'] != 'StackCube-v1' or c['env']['control_mode'] != 'pd_ee_delta_pos': p.error('This script is validated for StackCube-v1 with pd_ee_delta_pos (4-D actions) only')
    if a.trajectories < 5: p.error('Use at least 5 independent source episodes')
    device = torch.device(a.device); out = new_run(a.out, 'policy-io-audit', c)
    tensors, ema_weights = tensor_parity(a.checkpoint, a.policy); write_json(out / 'artifact_tensor_parity.json', tensors)

    # Same order as scripts/03_train_dp.py: patches first, then the trainer module binds the patched names.
    from hci.observations import install_training_patches, StrictRGBObservationWrapper
    from hci.policies import load_upstream, DiffusionPolicy
    from hci.simulation import make_env
    # load_upstream puts the baseline folder on sys.path (needed by the patches); the trainer module is then
    # re-executed so that, like the runpy namespace of a real training run, it binds the patched names.
    import sys
    load_upstream(a.upstream); install_training_patches(); sys.modules.pop('_hci_upstream_rgb_dp'); mod = load_upstream(a.upstream)
    if mod.FlattenRGBDObservationWrapper is not StrictRGBObservationWrapper: raise RuntimeError('Trainer module did not bind the strict wrapper')
    import gymnasium as gym
    from mani_skill.utils import common
    e = c['env']; meta = read_json(Path(a.policy) / 'policy_meta.json'); targs = meta['args']
    env_kwargs = dict(control_mode=targs['control_mode'], reward_mode='sparse', obs_mode=targs['obs_mode'], render_mode='rgb_array',
                      human_render_camera_configs=dict(shader_pack='default'), max_episode_steps=targs['max_episode_steps'])
    envs = mod.make_eval_envs(targs['env_id'], 1, targs['sim_backend'], env_kwargs, dict(obs_horizon=targs['obs_horizon']),
                              video_dir=None, wrappers=[mod.FlattenRGBDObservationWrapper])
    strict = find_wrapper(envs.envs[0], StrictRGBObservationWrapper)
    env = make_env(c); pol = DiffusionPolicy(env, a.policy, a.upstream, device=a.device)
    envs.reset(seed=0); env.reset(seed=0)   # same scene, so pose-dependent camera extrinsics (wrist camera) are comparable
    facts_b = env_facts(envs.envs[0]); facts_c = env_facts(env)
    config_equal = {k: facts_b[k] == facts_c[k] for k in facts_b}

    # Path A: the real upstream dataset class, fed by the patched convert_obs exactly as in the trainer's __main__.
    mod.args = SimpleNamespace(control_mode=targs['control_mode'], obs_horizon=targs['obs_horizon'], pred_horizon=targs['pred_horizon'])
    mod.device = device
    tmp = gym.make(targs['env_id'], **env_kwargs); original_space = tmp.observation_space; tmp.close()
    process = partial(mod.convert_obs, concat_fn=partial(np.concatenate, axis=-1), transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
                      state_obs_extractor=mod.build_state_obs_extractor(targs['env_id']), depth='rgbd' in str(a.demo))
    dataset = mod.SmallDemoDataset_DiffusionPolicy(data_path=str(a.demo), obs_process_fn=process, obs_space=original_space,
                                                   include_rgb=True, include_depth=False, device=device, num_traj=a.trajectories)
    index = {(i, s): n for n, (i, s, _) in enumerate(dataset.slices)}
    info = read_json(Path(a.demo).with_suffix('.json')); seeds = {x['episode_id']: x['episode_seed'] for x in info['episodes']}

    # Upstream-side agent: built from the evaluation envs like the trainer does, loaded with the checkpoint EMA.
    agent_b = mod.Agent(envs, SimpleNamespace(**targs)).to(device); agent_b.load_state_dict(ema_weights, strict=True); agent_b.eval()
    sched = {'upstream': dict(agent_b.noise_scheduler.config), 'adapter': dict(pol.agent.noise_scheduler.config),
             'num_inference_steps_upstream': len(agent_b.noise_scheduler.timesteps), 'num_inference_steps_adapter': len(pol.agent.noise_scheduler.timesteps),
             'adapter_training_flag': pol.agent.training, 'upstream_training_flag': agent_b.training,
             'horizons': {k: (getattr(agent_b, k), getattr(pol.agent, k), c['policy'][k]) for k in ('obs_horizon', 'act_horizon', 'pred_horizon')}}
    sched['equal'] = bool(sched['upstream'] == sched['adapter'] and sched['num_inference_steps_upstream'] == sched['num_inference_steps_adapter'] == 100
                          and not agent_b.training and not pol.agent.training and all(len(set(v)) == 1 for v in sched['horizons'].values()))

    pre = []; act = []; example = None
    with h5py.File(a.demo, 'r') as f:
        ids = sorted(int(k.split('_')[-1]) for k in f if k.startswith('traj_'))[:a.trajectories]
        for traj_index, i in enumerate(ids):
            g = f[f'traj_{i}']; actions = g['actions'][()]; L = len(actions); seed = int(seeds[i])
            steps = [t for t in (*CHECK_STEPS, L - 2) if 0 <= t <= L - 2]
            # ---- Part 2: the same recorded raw frames through A, B and C ----
            frame0 = raw_frame(g, 0); pol.reset(pol.convert(frame0)); frames_b = [strict.observation(frame0)] * 2
            for t in range(max(steps) + 1):
                if t > 0:
                    fr = raw_frame(g, t); pol.observe(pol.convert(fr)); frames_b = [frames_b[-1], strict.observation(fr)]
                if t not in steps: continue
                fr = raw_frame(g, t); C = pol.stacked(); A = dataset[index[(traj_index, t - 1)]]
                B = {k: torch.stack([x[k] for x in frames_b], 0).transpose(0, 1) for k in ('state', 'rgb')}   # FrameStack.observation
                a_obs = A['observations']; a_rgb_hwc = a_obs['rgb'].permute(0, 2, 3, 1)
                want_state = torch.cat([fr[pa][k] for pa, k in STATE_PATHS], -1)[0]
                want_rgb = torch.cat([fr['sensor_data'][n]['rgb'] for n in ('base_camera', 'hand_camera')], -1)[0]
                with torch.no_grad():
                    feat_a = pol.agent.encode_obs({'state': a_obs['state'][None], 'rgb': a_obs['rgb'][None]}, eval_mode=True)
                    # Values are compared in the SAME memory layout. The dataset keeps trajectories as a transposed (NHWC-strided)
                    # tensor and get_action permutes a view, while padded samples are contiguous; cuDNN uses a different convolution
                    # path per layout (measured: ~5e-5 feature difference for bit-identical inputs, exactly 0 with cuDNN disabled).
                    perm = C['rgb'].permute(0, 1, 4, 2, 3); same_layout = a_obs['rgb'].is_contiguous()
                    feat_c = pol.agent.encode_obs({'state': C['state'], 'rgb': perm.contiguous() if same_layout else perm}, eval_mode=True)
                    feat_view = pol.agent.encode_obs({'state': C['state'], 'rgb': perm if same_layout else perm.contiguous()}, eval_mode=True)
                pad = max(0, t + 8 - L); want_act = actions[t:t + 8]
                if pad: want_act = np.concatenate([want_act, np.tile(np.r_[np.zeros(3, np.float32), actions[-1, 3]], (pad, 1))])
                row = {'part': 'same_raw_frames', 'traj_id': i, 'scene_seed': seed, 't': t, 'steps_in_traj': L,
                       'shape_state_A': list(a_obs['state'].shape), 'shape_rgb_A': list(a_obs['rgb'].shape),
                       'shape_state_C': list(C['state'].shape), 'shape_rgb_C': list(C['rgb'].shape),
                       'rgb_A_contiguous': bool(a_obs['rgb'].is_contiguous()), 'dtype_rgb_A': str(a_obs['rgb'].dtype), 'dtype_rgb_C': str(C['rgb'].dtype), 'dtype_state_A': str(a_obs['state'].dtype), 'dtype_state_C': str(C['state'].dtype),
                       'rgb_max_value_C': int(C['rgb'].max()),
                       'state_A_vs_C': maxabs(a_obs['state'], C['state'][0]), 'rgb_A_vs_C_exact': bool(torch.equal(a_rgb_hwc.cpu(), C['rgb'][0].cpu())),
                       'state_B_vs_C': maxabs(B['state'], C['state']), 'rgb_B_vs_C_exact': bool(torch.equal(B['rgb'].cpu(), C['rgb'].cpu())),
                       'state_C_last_vs_qpos_qvel_tcp': maxabs(C['state'][0, -1], want_state), 'rgb_C_last_is_base_then_hand': bool(torch.equal(C['rgb'][0, -1].cpu(), want_rgb)),
                       'history_first_is_previous_frame': bool(torch.equal(C['rgb'][0, 0].cpu(), torch.cat([raw_frame(g, max(0, t - 1))['sensor_data'][n]['rgb'] for n in ('base_camera', 'hand_camera')], -1)[0])),
                       'encoder_feature_A_vs_C': maxabs(feat_a, feat_c), 'info_encoder_feature_layout_effect': maxabs(feat_a, feat_view),
                       'train_target_actions_executed_slice_vs_file_actions_t': maxabs(A['actions'][targs['obs_horizon'] - 1:targs['obs_horizon'] - 1 + 8], want_act)}
                row['pass'] = bool(row['shape_state_C'] == [1, 2, 25] and row['shape_rgb_C'] == [1, 2, 128, 128, 6] and row['shape_rgb_A'] == [2, 6, 128, 128]
                                   and row['dtype_rgb_A'] == row['dtype_rgb_C'] == 'torch.uint8' and row['dtype_state_A'] == row['dtype_state_C'] == 'torch.float32'
                                   and row['rgb_max_value_C'] > 1 and row['state_A_vs_C'] <= atol_s and row['state_B_vs_C'] <= atol_s and row['rgb_A_vs_C_exact']
                                   and row['rgb_B_vs_C_exact'] and row['state_C_last_vs_qpos_qvel_tcp'] <= atol_s and row['rgb_C_last_is_base_then_hand']
                                   and row['history_first_is_previous_frame'] and row['encoder_feature_A_vs_C'] <= atol_s
                                   and row['train_target_actions_executed_slice_vs_file_actions_t'] <= atol_a)
                pre.append(row)
                if example is None and t == 12: example = C['rgb'][0].cpu().numpy()
            # ---- Parts 3 and 4: two live environments, same scene seed, same actions ----
            obs_b, _ = envs.reset(seed=seed); raw, _ = env.reset(seed=seed); pol.reset(pol.convert(raw))
            for t in range(max(steps) + 1):
                if t > 0:
                    obs_b, _, _, _, _ = envs.step(actions[t - 1][None]); raw, _, _, _, _ = env.step(torch.as_tensor(actions[t - 1][None], device=env.device))
                    pol.observe(pol.convert(raw))
                if t not in steps: continue
                C = pol.stacked(); B = common.to_tensor(obs_b, device); fr = raw_frame(g, t)
                file_state = torch.cat([fr[pa][k] for pa, k in STATE_PATHS], -1)[0]
                file_rgb = torch.cat([fr['sensor_data'][n]['rgb'] for n in ('base_camera', 'hand_camera')], -1)[0]
                row = {'part': 'live_environments', 'traj_id': i, 'scene_seed': seed, 't': t, 'steps_in_traj': L,
                       'shape_state_B': list(B['state'].shape), 'shape_rgb_B': list(B['rgb'].shape), 'shape_state_C': list(C['state'].shape), 'shape_rgb_C': list(C['rgb'].shape),
                       'dtype_rgb_B': str(B['rgb'].dtype), 'dtype_state_B': str(B['state'].dtype),
                       'state_B_vs_C': maxabs(B['state'], C['state']), 'rgb_B_vs_C_exact': bool(torch.equal(B['rgb'].cpu(), C['rgb'].cpu())),
                       'rgb_B_vs_C_max_abs': maxabs(B['rgb'], C['rgb']),
                       'info_state_live_vs_file': maxabs(C['state'][0, -1], file_state), 'info_rgb_live_vs_file_max_abs': maxabs(C['rgb'][0, -1], file_rgb),
                       'info_rgb_live_vs_file_mean_abs': float((C['rgb'][0, -1].cpu().double() - file_rgb.double()).abs().mean()),
                       'info_rgb_live_vs_file_fraction_differing': float((C['rgb'][0, -1].cpu() != file_rgb).float().mean())}
                row['pass'] = bool(row['shape_state_B'] == row['shape_state_C'] == [1, 2, 25] and row['shape_rgb_B'] == row['shape_rgb_C'] == [1, 2, 128, 128, 6]
                                   and row['dtype_rgb_B'] == 'torch.uint8' and row['dtype_state_B'] == 'torch.float32'
                                   and row['state_B_vs_C'] <= atol_s and row['rgb_B_vs_C_exact'])
                pre.append(row)
                # Part 4. get_action permutes its input dict in place, so each side gets its own deep copy.
                k = seed_for(seed, t, 0, 'dp'); devices = list(range(torch.cuda.device_count())) if device.type == 'cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(k); up = agent_b.get_action(copy.deepcopy(B))
                    torch.manual_seed(k); up_on_c = agent_b.get_action(copy.deepcopy(C))
                    torch.manual_seed(k); pure = torch.randn((1, 16, 4), device=device)     # proves the two sides start from the same RNG state
                up = up.cpu().numpy(); up_on_c = up_on_c.cpu().numpy(); ad = pol.generate(seed, t, 0); ad2 = pol.generate(seed, t, 0)
                r2 = {'traj_id': i, 'scene_seed': seed, 't': t, 'chunk_shape_upstream': list(up.shape), 'chunk_shape_adapter': list(ad.shape),
                      'upstream_abs_max': float(np.abs(up).max()), 'clip_changes_upstream_output': float(np.abs(np.clip(up, -1, 1) - up).max()),
                      'upstream_on_B_vs_adapter': float(np.abs(up[0] - ad).max()), 'upstream_on_C_vs_adapter': float(np.abs(up_on_c[0] - ad).max()),
                      'adapter_repeat_same_seed': float(np.abs(ad - ad2).max()), 'first_noise_abs_sum': float(pure.abs().sum()),
                      'gripper_upstream': np.round(up[0, :, 3], 5).tolist(), 'gripper_adapter': np.round(ad[:, 3], 5).tolist(),
                      'first_action_upstream': up[0, 0].tolist(), 'first_action_adapter': ad[0].tolist()}
                r2['pass'] = bool(r2['chunk_shape_upstream'] == [1, 8, 4] and r2['chunk_shape_adapter'] == [8, 4] and r2['upstream_on_B_vs_adapter'] <= atol_a
                                  and r2['upstream_on_C_vs_adapter'] <= atol_a and r2['adapter_repeat_same_seed'] <= atol_a and r2['clip_changes_upstream_output'] == 0.)
                act.append(r2)
            print(f'traj {i} seed {seed}: checked steps {steps}', flush=True)
    envs.close(); env.close()

    def table(path, rows):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        with open(path, 'x', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    table(out / 'preprocessing_parity.csv', pre); table(out / 'action_parity.csv', act)
    if example is not None:
        import imageio.v2 as imageio
        imageio.imwrite(out / 'example_policy_input_t12.png', np.concatenate([np.concatenate([fr[..., :3], fr[..., 3:]], 1) for fr in example], 0))
    same = [r for r in pre if r['part'] == 'same_raw_frames']; live = [r for r in pre if r['part'] == 'live_environments']
    def worst(rows, key): return max([r[key] for r in rows] + [0.])
    report = {'artifact_tensors': {k: v for k, v in tensors.items() if k != 'tensors'},
              'same_raw_frames': {'n_histories': len(same), 'n_trajectories': len({r['traj_id'] for r in same}), 'all_pass': all(r['pass'] for r in same),
                                  'state_A_vs_C_max': worst(same, 'state_A_vs_C'), 'state_B_vs_C_max': worst(same, 'state_B_vs_C'),
                                  'rgb_A_vs_C_all_exact': all(r['rgb_A_vs_C_exact'] for r in same), 'rgb_B_vs_C_all_exact': all(r['rgb_B_vs_C_exact'] for r in same),
                                  'encoder_feature_A_vs_C_max': worst(same, 'encoder_feature_A_vs_C'),
                                  'info_encoder_feature_layout_effect_max': worst(same, 'info_encoder_feature_layout_effect'),
                                  'info_layout_note': 'Gate: same input values in the same memory layout must give identical encoder features. Info: the other layout (contiguous vs NHWC-strided view) changes the cuDNN path and the low-order float bits.',
                                  'train_action_alignment_max': worst(same, 'train_target_actions_executed_slice_vs_file_actions_t')},
              'live_environments': {'n_histories': len(live), 'all_pass': all(r['pass'] for r in live), 'state_B_vs_C_max': worst(live, 'state_B_vs_C'),
                                    'rgb_B_vs_C_all_exact': all(r['rgb_B_vs_C_exact'] for r in live), 'rgb_B_vs_C_max_abs': worst(live, 'rgb_B_vs_C_max_abs'),
                                    'info_state_live_vs_file_max': worst(live, 'info_state_live_vs_file'), 'info_rgb_live_vs_file_max_abs': worst(live, 'info_rgb_live_vs_file_max_abs'),
                                    'info_rgb_live_vs_file_mean_abs_max': worst(live, 'info_rgb_live_vs_file_mean_abs'),
                                    'info_rgb_live_vs_file_fraction_differing_max': worst(live, 'info_rgb_live_vs_file_fraction_differing'),
                                    'info_note': 'live-vs-file values are informational: the file was recorded with --use-first-env-state, which is not the same restore path as reset(seed).'},
              'actions': {'n_chunks': len(act), 'all_pass': all(r['pass'] for r in act), 'upstream_on_B_vs_adapter_max': worst(act, 'upstream_on_B_vs_adapter'),
                          'upstream_on_C_vs_adapter_max': worst(act, 'upstream_on_C_vs_adapter'), 'upstream_abs_max': worst(act, 'upstream_abs_max'),
                          'clip_changes_upstream_output_max': worst(act, 'clip_changes_upstream_output'),
                          'rng_note': 'Both sides call torch.manual_seed(seed_for(scene, step, 0, "dp")) immediately before get_action; that seeds the CPU and all CUDA generators, '
                                      'covering the initial noise and the noise drawn inside DDPMScheduler.step.'},
              'scheduler': sched, 'environment_config_equal': config_equal, 'environment_B_training_eval': facts_b, 'environment_C_deployment': facts_c,
              'observation': {'state_layout': [f'{pa}/{k}' for pa, k in STATE_PATHS], 'state_dim': 25, 'camera_channel_order': pol.cameras,
                              'normalization': 'uint8 RGB enters both paths; Agent.encode_obs divides by 255 once (checked through equal encoder features and rgb_max_value > 1).',
                              'privileged_inputs': 'none: only agent/qpos, agent/qvel, extra/tcp_pose and camera RGB are read'},
              'tolerances': {'state_atol': atol_s, 'action_atol': atol_a, 'rgb': 'exact equality'},
              'policy': str(a.policy), 'checkpoint': str(a.checkpoint), 'demo_sha256': sha256(a.demo), 'runtime': runtime_info(),
              'scope': 'Parity of data, in-training evaluation and deployment paths at the checked histories. It says nothing about whether the policy solves the task.'}
    report['pass'] = bool(tensors['pass'] and same and live and act and report['same_raw_frames']['all_pass'] and report['live_environments']['all_pass']
                          and report['actions']['all_pass'] and sched['equal'] and all(config_equal.values())
                          and len(same) >= 20 and report['same_raw_frames']['n_trajectories'] >= 5)
    write_json(out / 'policy_io_audit.json', report); finish_run(out, io_audit_pass=report['pass'])
    print(json.dumps({k: report[k] for k in ('pass', 'same_raw_frames', 'live_environments', 'actions', 'environment_config_equal')}, indent=2))
    print('artifact tensors pass:', tensors['pass'], '| scheduler equal:', sched['equal'])
    if not report['pass']: raise SystemExit(2)


if __name__ == '__main__':
    main()
