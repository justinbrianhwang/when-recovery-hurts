"""Retreat tracking diagnostic: commanded target vs measured TCP displacement, in free space and with an object held.

States are prepared by replaying demonstration action prefixes (scene seeds of the demonstrations), so the result
does not depend on the quality of the learned policy. No calibration or Test seed is touched. Object poses and
the grasp flag are privileged values used only to label and describe the diagnostic states; they are never fed
to a policy or monitor. A retreat is not required to reach 5 cm or to be harmless when the robot is in contact:
tracking of the control specification and the effect on the task are reported separately.
"""
from pathlib import Path
import argparse, csv, json
import h5py
import numpy as np
from hci.io import load_config, read_json, write_json, new_run, finish_run, runtime_info
from hci.policies import QueuePolicy
from hci.trees import as_numpy


class PrefixPolicy(QueuePolicy):
    """No learned policy: the Runtime is driven with recorded actions; only its buffers are needed."""
    def convert(self, raw): return {'tcp': raw['extra']['tcp_pose'].clone()}
    def generate(self, episode, t, repeat=0): raise RuntimeError('Diagnostic states are prepared from recorded actions only')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--demo', required=True); p.add_argument('--out', required=True)
    p.add_argument('--demos', type=int, default=10, help='number of demonstrations; each yields one free-space and two object states')
    a = p.parse_args(); c = load_config(a.config)
    if c['env']['id'] != 'StackCube-v1' or c['env']['control_mode'] != 'pd_ee_delta_pos': p.error('This script is validated for StackCube-v1 with pd_ee_delta_pos (4-D actions) only')
    import torch
    from hci.simulation import make_env, Runtime
    info = read_json(Path(a.demo).with_suffix('.json')); seeds = {x['episode_id']: x['episode_seed'] for x in info['episodes']}
    out = new_run(a.out, 'retreat-diagnostic', c); env = make_env(c); pol = PrefixPolicy(c['policy']['act_horizon'], c['policy']['obs_horizon'])
    r = Runtime(env, pol, c); rows = []; steps_csv = []

    def privileged():
        actors = {k: as_numpy(v)[0, :3].copy() for k, v in env.get_state_dict()['actors'].items() if k.startswith('cube')}
        return actors, bool(as_numpy(env.agent.is_grasping(env.cubeA)).reshape(-1)[0])

    def prepare(seed, actions, upto):
        r.reset(seed)
        for x in actions[:upto]:
            if r.done: break
            r.advance(x)

    try:
        with h5py.File(a.demo, 'r') as f:
            ids = sorted(int(k.split('_')[-1]) for k in f if k.startswith('traj_'))[:a.demos]
            for i in ids:
                actions = f[f'traj_{i}/actions'][()]; seed = int(seeds[i]); L = len(actions)
                # Label states with the privileged grasp flag along the recorded trajectory.
                r.reset(seed); grasp = []
                for x in actions:
                    if r.done: break
                    r.advance(x); grasp.append(privileged()[1])
                first = next((t + 1 for t, g in enumerate(grasp) if g), None); last = max((t + 1 for t, g in enumerate(grasp) if g), default=None)
                plan = [('free_space', 5)]
                if first is not None:
                    plan += [('object_held_after_grasp', min(first + 5, last)), ('object_held_near_placement', max(first, last - 3))]
                for label, t0 in plan:
                    prepare(seed, actions, t0)
                    if r.done: rows.append({'traj_id': i, 'scene_seed': seed, 'label': label, 'prefix_steps': t0, 'skipped': 'episode already complete'}); continue
                    cubes0, grasp0 = privileged(); qpos0 = as_numpy(r.raw['agent']['qpos'])[0].copy(); quat0 = as_numpy(r.raw['extra']['tcp_pose'])[0, 3:].copy()
                    tcp0 = r.tcp_history[-1].copy(); dist = float(min(np.linalg.norm(tcp0 - v) for v in cubes0.values())); n0 = len(r.actions)
                    r.intervention = 2; r.intervention_t = r.t; r.retreat()
                    cubes1, grasp1 = privileged(); qpos1 = as_numpy(r.raw['agent']['qpos'])[0]; quat1 = as_numpy(r.raw['extra']['tcp_pose'])[0, 3:]
                    v = dict(r.retreat_info); path = np.array(r.tcp_history[-(v['steps_used'] + 1):]); cmds = np.array(r.actions[n0:]).reshape(-1, r.adim)
                    row = {'traj_id': i, 'scene_seed': seed, 'label': label, 'prefix_steps': t0, 'demo_steps': L,
                           'grasping_before': grasp0, 'grasping_after': grasp1, 'tcp_to_nearest_cube_before_m': dist,
                           'reached': v['reached'], 'end_reason': v['end_reason'], 'steps_used': v['steps_used'], 'max_steps': v['max_steps'],
                           'commanded_distance_m': v['commanded_distance_m'], 'achieved_displacement_m': v['achieved_displacement_m'],
                           'achieved_dz_m': float(path[-1, 2] - path[0, 2]), 'achieved_dxy_m': float(np.linalg.norm(path[-1, :2] - path[0, :2])),
                           'target_error_m': v['target_error_m'], 'command_saturated_steps': v['command_saturated_steps'],
                           'mean_dz_per_step_m': float(np.diff(path[:, 2]).mean()) if len(path) > 1 else 0.,
                           'max_normalized_arm_command': float(np.abs(cmds[:, :3]).max()) if len(cmds) else 0.,
                           'gripper_command': v['gripper_command'], 'gripper_command_constant': bool(len(cmds) == 0 or np.all(cmds[:, -1] == cmds[0, -1])),
                           'finger_qpos_before': qpos0[-2:].tolist(), 'finger_qpos_after': qpos1[-2:].tolist(),
                           'tcp_quat_max_abs_change': float(np.abs(quat1 - quat0).max()),
                           'cube_displacement_m': {k: float(np.linalg.norm(cubes1[k] - cubes0[k])) for k in cubes0},
                           'success_after_retreat': int(r.y), 'episode_done_during_retreat': bool(r.done),
                           'start_tcp': v['start_tcp'], 'target_tcp': v['target_tcp'], 'final_tcp': v['final_tcp']}
                    rows.append(row)
                    for k in range(len(path)):
                        steps_csv.append({'traj_id': i, 'label': label, 'retreat_step': k, 'tcp_dz_m': float(path[k, 2] - path[0, 2]),
                                          'target_error_m': float(np.linalg.norm(np.array(v['target_tcp']) - path[k]))})
                    print(json.dumps({k: row[k] for k in ('traj_id', 'label', 'prefix_steps', 'grasping_before', 'reached', 'steps_used', 'achieved_dz_m', 'target_error_m')}), flush=True)
        done = [x for x in rows if 'skipped' not in x]
        def group(label):
            g = [x for x in done if x['label'] == label]
            if not g: return {'n': 0}
            return {'n': len(g), 'reached': int(sum(x['reached'] for x in g)), 'steps_used_min_mean_max': [int(min(x['steps_used'] for x in g)), float(np.mean([x['steps_used'] for x in g])), int(max(x['steps_used'] for x in g))],
                    'achieved_dz_m_min_mean_max': [float(min(x['achieved_dz_m'] for x in g)), float(np.mean([x['achieved_dz_m'] for x in g])), float(max(x['achieved_dz_m'] for x in g))],
                    'target_error_m_max': float(max(x['target_error_m'] for x in g)), 'mean_dz_per_step_m': float(np.mean([x['mean_dz_per_step_m'] for x in g])),
                    'still_grasping_after': int(sum(x['grasping_after'] for x in g)), 'grasping_before': int(sum(x['grasping_before'] for x in g)),
                    'max_cube_displacement_m': float(max(max(x['cube_displacement_m'].values()) for x in g))}
        report = {'retreat_config': c['retreat'], 'n_states': len(done), 'skipped': [x for x in rows if 'skipped' in x],
                  'free_space': group('free_space'), 'object_held_after_grasp': group('object_held_after_grasp'), 'object_held_near_placement': group('object_held_near_placement'),
                  'states': done, 'runtime': runtime_info(),
                  'scope': 'Development diagnostic on demonstration scenes. max_translation_per_step_m bounds the commanded increment; the measured displacement is reported separately. '
                           'Reaching the target is not a safety or harmlessness claim.'}
        write_json(out / 'retreat_diagnostic.json', report)
        with open(out / 'retreat_steps.csv', 'x', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=['traj_id', 'label', 'retreat_step', 'tcp_dz_m', 'target_error_m']); w.writeheader(); w.writerows(steps_csv)
        finish_run(out, n_states=len(done))
        print(json.dumps({k: report[k] for k in ('n_states', 'free_space', 'object_held_after_grasp', 'object_held_near_placement')}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
