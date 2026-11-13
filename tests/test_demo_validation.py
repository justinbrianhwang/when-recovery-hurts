"""Simulator-free demonstration checks (needs h5py; skipped where it is not installed)."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), ROOT / 'scripts' / name)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


h5py = pytest.importorskip('h5py')
CFG = {'env': {'id': 'StackCube-v1', 'obs_mode': 'rgb', 'control_mode': 'pd_ee_delta_pos', 'sim_backend': 'physx_cpu', 'robot_uid': 'panda_wristcam'}}


def write_demo(path, lengths, seeds, break_traj=None):
    from hci.io import UPSTREAM_COMMIT
    with h5py.File(path, 'w') as f:
        for i, L in enumerate(lengths):
            g = f.create_group(f'traj_{i}'); a = np.zeros((L, 4), np.float32); a[:, 3] = 1
            if break_traj == i: a[0, 0] = 1.5
            g['actions'] = a
            for cam in ('base_camera', 'hand_camera'): g.create_dataset(f'obs/sensor_data/{cam}/rgb', data=np.zeros((L + 1, 128, 128, 3), np.uint8), compression='gzip')
            g['obs/agent/qpos'] = np.zeros((L + 1, 9), np.float32); g['obs/agent/qvel'] = np.zeros((L + 1, 9), np.float32)
            g['obs/extra/tcp_pose'] = np.zeros((L + 1, 7), np.float32)
            g['env_states/actors/cubeA'] = np.full((L + 1, 13), i, np.float32)
    return {'env_info': {'env_id': 'StackCube-v1', 'env_kwargs': {'obs_mode': 'rgb', 'control_mode': 'pd_ee_delta_pos', 'sim_backend': 'physx_cpu', 'num_envs': 1}},
            'commit_info': {'commit_id': UPSTREAM_COMMIT},
            'episodes': [{'episode_id': i, 'episode_seed': s, 'control_mode': 'pd_ee_delta_pos', 'elapsed_steps': L, 'success': True}
                         for i, (L, s) in enumerate(zip(lengths, seeds))]}


def test_demo_structure_and_source_mapping(tmp_path):
    vd = load('07_validate_demos.py')
    info = write_demo(tmp_path / 'ok.h5', [5, 6, 7], [0, 1, 3])            # source episode 2 failed the success filter
    rows, problems = vd.check_structure(tmp_path / 'ok.h5', info, CFG)
    assert problems == [] and [r['scene_seed'] for r in rows] == [0, 1, 3] and rows[0]['gripper_values'] == [1.0]
    source = {'episodes': [{'episode_id': i, 'episode_seed': i} for i in range(6)]}
    mapping, dropped = vd.source_mapping(rows, source)
    assert mapping == {0: 0, 1: 1, 2: 3} and dropped == [{'source_episode_id': 2, 'scene_seed': 2}]


def test_demo_structure_reports_bad_actions_and_wrong_control_mode(tmp_path):
    vd = load('07_validate_demos.py')
    info = write_demo(tmp_path / 'bad.h5', [5, 6], [0, 1], break_traj=1)
    info['env_info']['env_kwargs']['control_mode'] = 'pd_joint_delta_pos'
    _, problems = vd.check_structure(tmp_path / 'bad.h5', info, CFG)
    assert any('control_mode' in x for x in problems) and any('traj_1' in x and 'outside' in x for x in problems)
