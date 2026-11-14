"""Stage 1C / D3 + D4 (+ optional D5-B): what the preserved policy generates on recorded demonstration histories.

D3  On the manifest histories (training-data path: upstream dataset class with the project patches) call the real
    Agent.get_action with fixed noise seeds and compare with the recorded expert actions. Also the denoising loss
    (noise-prediction MSE, the quantity the trainer printed) at fixed timesteps and averaged over all 100 timesteps.
D4  Input sensitivity to the RGB difference between the file and a replay. For the same history:
      S        file proprioception + file RGB
      V_<cell> file proprioception + RGB observed while replaying the recorded actions in <cell> (14_scene_rollouts.py)
      L_<cell> replayed proprioception + replayed RGB
    S vs V is a synthetic input contrast (same proprioception, same noise); it is not a causal effect on task success.
D5-B (only with --raw-checkpoint) the non-EMA weights of the same checkpoint on the same histories and noise.

Action errors are in normalised controller units [-1, 1] (1.0 = the controller's 0.1 m position-delta limit as a
COMMAND; it is not the distance the TCP moved). RGB statistics are uint8 [0, 255]. Diffusion Policy may produce a
different valid action than the single recorded expert segment, so a distance is not by itself a failure.
"""
from pathlib import Path
from functools import partial
from types import SimpleNamespace
import argparse, copy, csv, json, sys, time
import h5py
import numpy as np
import torch
from hci.io import load_config, read_json, write_json, new_run, finish_run, sha256, runtime_info
from hci.schema import STATE_PATHS
from hci.diagnostics import action_errors, rgb_diff_stats

CAMERAS = ('base_camera', 'hand_camera'); VIS = 256


def table(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, 'x', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def mean_std(values):
    v = np.asarray([x for x in values if x is not None], float)
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else None, 'n': int(len(v))} if len(v) else {'mean': None, 'std': None, 'n': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True); p.add_argument('--manifest', required=True); p.add_argument('--out', required=True)
    p.add_argument('--demo', required=True); p.add_argument('--policy', required=True); p.add_argument('--upstream', default='third_party/ManiSkill')
    p.add_argument('--replay-obs', help='replay_obs folder written by 14_scene_rollouts.py (enables D4)')
    p.add_argument('--raw-checkpoint', help='full checkpoint; adds the non-EMA model (D5-B)')
    p.add_argument('--limit', type=int, help='first N scenes of each set only (throughput probe; not a result)')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu']); p.add_argument('--dry-run', action='store_true')
    p.add_argument('--other-policy', action='store_true', help='Stage 1D: run a policy other than the one fixed in the manifest on the same fixed scenes, histories and noise (demo files are still checked)')
    a = p.parse_args(); c = load_config(a.config); m = read_json(a.manifest)
    if c['env']['id'] != 'StackCube-v1' or c['env']['control_mode'] != 'pd_ee_delta_pos': p.error('StackCube-v1 with pd_ee_delta_pos only')
    scenes = m['scenes']
    if a.limit: scenes = [s for name in ('train', 'heldout') for s in [x for x in m['scenes'] if x['set'] == name][:a.limit]]
    seeds = m['noise_seeds']; n_hist = sum(len(s['windows']) for s in scenes)
    print(f'histories={n_hist} noise_seeds={seeds} get_action_calls(S)={n_hist * len(seeds)} loss_timesteps={m["loss_timesteps"]}', flush=True)
    if a.dry_run: return
    if sha256(a.demo) != m['hashes']['demo_h5']['sha256'] or (not a.other_policy and sha256(Path(a.policy) / 'weights.pt') != m['hashes']['policy_weights']['sha256']):
        raise SystemExit('Demo or policy is not the file fixed in the manifest')
    device = torch.device(a.device); out = new_run(a.out, 'stage1c-offline-actions', c); started = time.perf_counter()
    from hci.observations import install_training_patches, StrictRGBObservationWrapper
    from hci.policies import load_upstream, DiffusionPolicy
    from hci.simulation import make_env
    load_upstream(a.upstream); install_training_patches(); sys.modules.pop('_hci_upstream_rgb_dp'); mod = load_upstream(a.upstream)
    if mod.FlattenRGBDObservationWrapper is not StrictRGBObservationWrapper: raise RuntimeError('Trainer module did not bind the strict wrapper')
    import gymnasium as gym
    meta = read_json(Path(a.policy) / 'policy_meta.json'); targs = meta['args']; OH, AH, PH = targs['obs_horizon'], targs['act_horizon'], targs['pred_horizon']
    env = make_env(c); pol = DiffusionPolicy(env, a.policy, a.upstream, device=a.device); env.close()
    models = {'ema': pol.agent}
    if a.raw_checkpoint:
        if sha256(a.raw_checkpoint) != m['hashes']['checkpoint_60k']['sha256']: raise SystemExit('Raw checkpoint is not the file fixed in the manifest')
        ck = torch.load(a.raw_checkpoint, map_location=device, weights_only=True); raw_agent = copy.deepcopy(pol.agent)
        raw_agent.load_state_dict(ck['agent'], strict=True); raw_agent.eval(); models['raw'] = raw_agent
        if not all(torch.equal(ck['ema_agent'][k].to(device), v) for k, v in pol.agent.state_dict().items()): raise SystemExit('Checkpoint EMA differs from the exported policy')
    # Path A: the upstream dataset class fed by the patched convert_obs, as in the trainer (same construction as 08_policy_io_audit.py).
    mod.args = SimpleNamespace(control_mode=targs['control_mode'], obs_horizon=OH, pred_horizon=PH); mod.device = device
    env_kwargs = dict(control_mode=targs['control_mode'], reward_mode='sparse', obs_mode=targs['obs_mode'], render_mode='rgb_array',
                      human_render_camera_configs=dict(shader_pack='default'), max_episode_steps=targs['max_episode_steps'])
    tmp = gym.make(targs['env_id'], **env_kwargs); space = tmp.observation_space; tmp.close()
    process = partial(mod.convert_obs, concat_fn=partial(np.concatenate, axis=-1), transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
                      state_obs_extractor=mod.build_state_obs_extractor(targs['env_id']), depth=False)
    n_traj = max(s['traj_id'] for s in scenes) + 1
    dataset = mod.SmallDemoDataset_DiffusionPolicy(data_path=str(a.demo), obs_process_fn=process, obs_space=space, include_rgb=True, include_depth=False, device=device, num_traj=n_traj)
    index = {(i, s): n for n, (i, s, _) in enumerate(dataset.slices)}
    sched = pol.agent.noise_scheduler

    def generate(agent, state, rgb_chw, seed):
        """state (OH, 25), rgb (OH, 6, H, W) uint8. A fresh input dict per call: get_action replaces entries of its argument."""
        obs = {'state': state[None].clone(), 'rgb': rgb_chw.permute(0, 2, 3, 1)[None].clone()}
        torch.manual_seed(seed)                      # CPU and all CUDA generators: initial noise and the noise inside scheduler.step
        with torch.no_grad(): x = agent.get_action(obs)[0]
        raw = x.cpu().numpy().astype(np.float64); return np.clip(raw, -1, 1), float(np.abs(np.clip(raw, -1, 1) - raw).max())

    def features(agent, state, rgb_chw):
        with torch.no_grad(): z = agent.encode_obs({'state': state[None], 'rgb': rgb_chw[None].contiguous()}, eval_mode=True)
        return z.reshape(OH, -1)[:, :VIS].flatten().double().cpu().numpy()

    def losses(agent, state, rgb_chw, act_seq, valid, seed):
        """Noise-prediction MSE for every diffusion timestep with one fixed noise draw (eval mode; the trainer has no image augmentation)."""
        T = sched.config.num_train_timesteps; g = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn((1, PH, act_seq.shape[-1]), generator=g).to(device).repeat(T, 1, 1); ts = torch.arange(T, device=device).long()
        with torch.no_grad():
            cond = agent.encode_obs({'state': state[None], 'rgb': rgb_chw[None].contiguous()}, eval_mode=True).repeat(T, 1)
            noisy = sched.add_noise(act_seq[None].repeat(T, 1, 1), noise, ts)
            err = (agent.noise_pred_net(noisy, ts, global_cond=cond) - noise) ** 2          # (T, PH, act_dim)
        v = torch.as_tensor(valid, device=device)
        return err.mean((1, 2)).cpu().numpy(), err[:, v].mean((1, 2)).cpu().numpy()

    replay = {}
    if a.replay_obs:
        for path in sorted(Path(a.replay_obs).glob('*_traj*.npz')):
            cell, tid = path.stem.rsplit('_traj', 1); z = np.load(path)
            replay[(cell, int(tid))] = {int(t): (z['state'][k], z['rgb'][k]) for k, t in enumerate(z['steps'])}
    cells = sorted({k[0] for k in replay})
    action_rows = []; loss_rows = []; sens_rows = []; hist_rows = []; diffmaps = {}; examples = {}
    with h5py.File(a.demo, 'r') as f:
        for sc in scenes:
            i = sc['traj_id']; g = f[f'traj_{i}']; actions = g['actions'][()]; L = len(actions)
            for w in sc['windows']:
                t = w['t']; A = dataset[index[(i, t - 1)]]; state = A['observations']['state']; rgb = A['observations']['rgb']; act_seq = A['actions']
                expert = actions[t:t + AH].astype(np.float64)
                if len(expert) != AH: raise RuntimeError(f'traj {i} t {t}: executed segment is padded')
                if float(np.abs(act_seq[OH - 1:OH - 1 + AH].cpu().numpy() - expert).max()) > c['audit']['action_atol']: raise RuntimeError(f'traj {i} t {t}: dataset target is not aligned with the file actions')
                fstate = np.stack([np.concatenate([g[f'obs/{pa}/{k}'][x] for pa, k in STATE_PATHS], -1) for x in (max(0, t - 1), t)])
                frgb = np.stack([np.concatenate([g[f'obs/sensor_data/{n}/rgb'][x] for n in CAMERAS], -1) for x in (max(0, t - 1), t)])
                if float(np.abs(state.cpu().numpy() - fstate).max()) > c['audit']['state_atol'] or not np.array_equal(rgb.permute(0, 2, 3, 1).cpu().numpy(), frgb):
                    raise RuntimeError(f'traj {i} t {t}: dataset history differs from the file frames')
                valid = [0 <= t - (OH - 1) + j < L for j in range(PH)]; base = {'set': sc['set'], 'traj_id': i, 't': t, 'fraction': w['fraction'], 'phase': w['phase'], 'steps': L}
                S = {}
                for name, agent in models.items():
                    per_seed = []
                    for k, seed in enumerate(seeds):
                        pred, clip = generate(agent, state, rgb, seed); S[(name, seed)] = pred; err = action_errors(pred, expert); per_seed.append(pred)
                        row = {'model': name, **base, 'noise_seed': seed, 'condition': 'S', 'executed_segment_padded': False, 'pred_segment_padded_steps': PH - sum(valid),
                               **err, 'clip_changed_output_max': clip, 'pred_first': pred[0].tolist(), 'expert_first': expert[0].tolist()}
                        if k == 0:
                            again, _ = generate(agent, state, rgb, seed); row['repeat_same_seed_max_abs'] = float(np.abs(again - pred).max())
                        action_rows.append(row)
                    stack = np.stack(per_seed); pair = [np.abs(stack[x] - stack[y])[:, :3].mean() for x in range(len(seeds)) for y in range(x + 1, len(seeds))]
                    all_t, valid_t = losses(agent, state, rgb, act_seq, valid, seeds[0])
                    hist_rows.append({'model': name, **base, 'seed_pair_mean_abs_xyz': float(np.mean(pair)), 'seed_std_first_xyz': float(stack[:, 0, :3].std(0).mean()),
                                      'seed_std_gripper_first': float(stack[:, 0, 3].std()), 'mean_action_mae_xyz_vs_expert': float(np.abs(stack.mean(0) - expert)[:, :3].mean()),
                                      'loss_all_timesteps_mean': float(all_t.mean()), 'loss_all_timesteps_mean_valid_only': float(valid_t.mean())})
                    for seed in seeds:
                        all_t, valid_t = losses(agent, state, rgb, act_seq, valid, seed)
                        for ts in m['loss_timesteps']:
                            loss_rows.append({'model': name, **base, 'noise_seed': seed, 'timestep': ts, 'mse_all': float(all_t[ts]), 'mse_valid_only': float(valid_t[ts]), 'n_valid_steps': sum(valid)})
                # ---- D4: same proprioception, RGB from a replay (EMA model only) ----
                agent = models['ema']; feat_s = features(agent, state, rgb)
                for cell in cells:
                    frames = replay.get((cell, i), {})
                    if t not in frames or max(0, t - 1) not in frames: continue
                    rstate = torch.as_tensor(np.stack([frames[max(0, t - 1)][0], frames[t][0]]), device=device).float()
                    rrgb_np = np.stack([frames[max(0, t - 1)][1], frames[t][1]]); rrgb = torch.as_tensor(rrgb_np, device=device).permute(0, 3, 1, 2)
                    stats = {}
                    for j, cam in enumerate(CAMERAS):
                        per = [rgb_diff_stats(rrgb_np[x][..., 3 * j:3 * j + 3], frgb[x][..., 3 * j:3 * j + 3]) for x in range(OH)]
                        for key in per[0]: stats[f'{cam}_{key}'] = float(np.mean([q[key] for q in per])) if key != 'max_u8' else int(max(q[key] for q in per))
                        d = np.abs(rrgb_np[-1][..., 3 * j:3 * j + 3].astype(np.int16) - frgb[-1][..., 3 * j:3 * j + 3]).mean(-1)
                        acc = diffmaps.setdefault((cell, cam), [np.zeros_like(d, dtype=np.float64), 0]); acc[0] += d; acc[1] += 1
                    if w['fraction'] == 0.4 and len(examples.setdefault(cell, [])) < 4:
                        examples[cell].append(np.concatenate([np.concatenate([frgb[-1][..., 3 * j:3 * j + 3], rrgb_np[-1][..., 3 * j:3 * j + 3],
                                              np.clip(np.abs(rrgb_np[-1][..., 3 * j:3 * j + 3].astype(np.int16) - frgb[-1][..., 3 * j:3 * j + 3]) * 8, 0, 255).astype(np.uint8)], 1) for j in range(2)], 1))
                    proprio = float(np.abs(rstate.cpu().numpy() - fstate).max())
                    for kind, st in (('V', state), ('L', rstate)):
                        feat = features(agent, st, rrgb); fd = feat - feat_s
                        for seed in seeds:
                            pred, _ = generate(agent, st, rrgb, seed); ref = S[('ema', seed)]; d = pred - ref
                            others = [np.abs(S[('ema', o)] - ref)[:, :3].mean() for o in seeds if o != seed]; e = action_errors(pred, expert); es = action_errors(ref, expert)
                            sens_rows.append({**base, 'replay_cell': cell, 'condition': f'{kind}_{cell}', 'kind': kind, 'noise_seed': seed, **stats,
                                              'proprio_replay_vs_file_max_abs': proprio, 'proprio_used': 'file' if kind == 'V' else 'replay',
                                              'visual_feature_l2_diff': float(np.linalg.norm(fd)), 'visual_feature_l2_S': float(np.linalg.norm(feat_s)),
                                              'visual_feature_cosine': float(feat @ feat_s / (np.linalg.norm(feat) * np.linalg.norm(feat_s) + 1e-12)),
                                              'action_first_xyz_l2_vs_S': float(np.linalg.norm(d[0, :3])), 'action_first_xyz_linf_vs_S': float(np.abs(d[0, :3]).max()),
                                              'action_seg_xyz_mean_abs_vs_S': float(np.abs(d[:, :3]).mean()), 'action_seg_xyz_max_abs_vs_S': float(np.abs(d[:, :3]).max()),
                                              'gripper_mean_abs_vs_S': float(np.abs(d[:, 3]).mean()), 'gripper_sign_flip_fraction_vs_S': float(((pred[:, 3] < 0) != (ref[:, 3] < 0)).mean()),
                                              'S_other_seed_xyz_mean_abs': float(np.mean(others)), 'mae_xyz_vs_expert': e['mae_xyz'], 'mae_xyz_vs_expert_S': es['mae_xyz'],
                                              'first_l2_xyz_vs_expert': e['first_l2_xyz'], 'first_l2_xyz_vs_expert_S': es['first_l2_xyz']})
            print(f"{sc['set']} traj={i}: {len(sc['windows'])} histories done", flush=True)
    table(out / 'offline_action_metrics.csv', action_rows); table(out / 'offline_history_metrics.csv', hist_rows); table(out / 'denoise_loss.csv', loss_rows)
    if sens_rows: table(out / 'rgb_action_sensitivity.csv', sens_rows)
    if diffmaps:
        import imageio.v2 as imageio
        for (cell, cam), (acc, n) in diffmaps.items():
            mean = acc / n; np.save(out / f'diffmap_{cell}_{cam}.npy', mean.astype(np.float32))
            imageio.imwrite(out / f'diffmap_{cell}_{cam}_x16.png', np.clip(mean * 16, 0, 255).astype(np.uint8))
        for cell, imgs in examples.items(): imageio.imwrite(out / f'examples_{cell}_file_replay_absdiff_x8.png', np.concatenate(imgs, 0))

    def group(rows, keys, fields):
        out_ = {}
        for r in rows:
            out_.setdefault('/'.join(str(r[k]) for k in keys), []).append(r)
        return {k: {'n_rows': len(v), **{fl: mean_std([x.get(fl) for x in v]) for fl in fields}} for k, v in sorted(out_.items())}
    af = ['mae_xyz', 'rmse_xyz', 'mae_x', 'mae_y', 'mae_z', 'signed_x', 'signed_y', 'signed_z', 'first_l2_xyz', 'pred_saturation_xyz', 'expert_saturation_xyz',
          'gripper_mae', 'gripper_sign_agree', 'gripper_first_sign_agree', 'expert_norm_xyz_mean', 'pred_norm_xyz_mean', 'clip_changed_output_max']
    hf = ['seed_pair_mean_abs_xyz', 'seed_std_first_xyz', 'seed_std_gripper_first', 'mean_action_mae_xyz_vs_expert', 'loss_all_timesteps_mean', 'loss_all_timesteps_mean_valid_only']
    sf = ['base_camera_mae_u8', 'hand_camera_mae_u8', 'base_camera_frac_ge1', 'base_camera_frac_ge8', 'base_camera_frac_ge32', 'hand_camera_frac_ge1', 'hand_camera_frac_ge8', 'hand_camera_frac_ge32',
          'proprio_replay_vs_file_max_abs', 'visual_feature_l2_diff', 'visual_feature_l2_S', 'visual_feature_cosine', 'action_first_xyz_l2_vs_S', 'action_seg_xyz_mean_abs_vs_S',
          'action_seg_xyz_max_abs_vs_S', 'gripper_mean_abs_vs_S', 'gripper_sign_flip_fraction_vs_S', 'S_other_seed_xyz_mean_abs', 'mae_xyz_vs_expert', 'mae_xyz_vs_expert_S']
    repeats = [r['repeat_same_seed_max_abs'] for r in action_rows if 'repeat_same_seed_max_abs' in r]
    report = {'n_histories': n_hist, 'noise_seeds': seeds, 'models': list(models), 'replay_cells': cells, 'limit': a.limit,
              'same_input_same_seed_repeat_max_abs': max(repeats), 'units': {'action': 'normalised controller input in [-1, 1]; arm 1.0 = 0.1 m commanded position delta, not measured motion',
                                                                             'rgb': 'uint8 [0, 255]; the encoder divides by 255', 'loss': 'noise-prediction MSE, eval mode'},
              'actions_by_model_set': group(action_rows, ('model', 'set'), af), 'actions_by_model_set_phase': group(action_rows, ('model', 'set', 'phase'), af),
              'histories_by_model_set': group(hist_rows, ('model', 'set'), hf), 'histories_by_model_set_phase': group(hist_rows, ('model', 'set', 'phase'), hf),
              'loss_by_model_set_timestep': group(loss_rows, ('model', 'set', 'timestep'), ['mse_all', 'mse_valid_only']),
              'sensitivity_by_condition_set': group(sens_rows, ('condition', 'set'), sf), 'sensitivity_by_condition': group(sens_rows, ('condition',), sf),
              'policy': a.policy, 'manifest_sha256': sha256(a.manifest), 'wall_seconds': time.perf_counter() - started, 'runtime': runtime_info(),
              'scope': 'Offline diagnosis on recorded histories. Means are over histories and noise seeds; std is across rows, not a confidence interval. Best-of-seeds is never reported. '
                       'Rows of one trajectory are not independent.' + (' THROUGHPUT PROBE (limited scenes), not a result.' if a.limit else '')}
    write_json(out / 'offline_summary.json', report); finish_run(out, n_histories=n_hist)
    print(json.dumps({k: report[k] for k in ('same_input_same_seed_repeat_max_abs', 'actions_by_model_set', 'histories_by_model_set', 'sensitivity_by_condition')}, indent=1))


if __name__ == '__main__':
    main()
