"""Stage 1D data construction (simulator-free; needs h5py): D100 kept byte-identical, protected scenes kept out,
ids as the upstream loader reads them, initial scenes taken from the SOURCE first state."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
h5py = pytest.importorskip('h5py')
spec = importlib.util.spec_from_file_location('stage1d_data', ROOT / 'scripts' / '16_stage1d_data.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from hci.io import UPSTREAM_COMMIT, read_json, write_json, sha256

N_SRC, N_OLD_CONV, N_TRAIN, N_DEV, WINDOW, N_NEW = 30, 12, 5, 3, 12, 8


def seed_of(i): return i + 1 + (i > 3)        # unique, not equal to the episode id


def cube(i, t=0, name='A'): v = np.zeros(13, np.float32); v[:3] = (0.01 * i + 0.001 * t, 0.02 * i if name == 'A' else -0.02 * i - 0.1, 0.02); v[3] = 1; return v


def write_source(path, same_scene=None):
    eps = []
    with h5py.File(path, 'w') as f:
        for i in range(N_SRC):
            L = 6 + i % 3; g = f.create_group(f'traj_{i}'); j = same_scene.get(i, i) if same_scene else i
            g['actions'] = np.zeros((L, 8), np.float32)
            for n in 'AB': g[f'env_states/actors/cube{n}'] = np.stack([cube(j, t, n) for t in range(L + 1)])
            g['env_states/articulations/panda_wristcam'] = np.zeros((L + 1, 31), np.float32)
            eps.append({'episode_id': i, 'episode_seed': seed_of(i), 'control_mode': 'pd_joint_pos', 'elapsed_steps': L, 'reset_kwargs': {'options': {}, 'seed': seed_of(i)}, 'success': True})
    write_json(path.with_suffix('.json'), {'env_info': {'env_id': 'StackCube-v1'}, 'episodes': eps})


def write_converted(path, source_ids, drop=()):
    """Like replay_trajectory: ids renumbered in flush order, env_states[0] = the SECOND source state."""
    eps = []; rng = np.random.default_rng(sum(source_ids))
    with h5py.File(path, 'w') as f:
        for i in [s for s in source_ids if s not in drop]:
            k = len(eps); L = 5 + i % 4; g = f.create_group(f'traj_{k}'); a = rng.uniform(-.5, .5, (L, 4)).astype(np.float32); a[:, 3] = 1; g['actions'] = a
            for cam in ('base_camera', 'hand_camera'): g.create_dataset(f'obs/sensor_data/{cam}/rgb', data=rng.integers(0, 255, (L + 1, 8, 8, 3), dtype=np.uint8), compression='gzip')
            g['obs/agent/qpos'] = rng.normal(size=(L + 1, 9)).astype(np.float32); g['obs/extra/tcp_pose'] = rng.normal(size=(L + 1, 7)).astype(np.float32)
            for n in 'AB': g[f'env_states/actors/cube{n}'] = np.stack([cube(i, t + 1, n) for t in range(L + 1)])
            s = np.zeros(L, bool); s[-2:] = True; g['success'] = s
            eps.append({'episode_id': k, 'episode_seed': seed_of(i), 'control_mode': 'pd_ee_delta_pos', 'elapsed_steps': L, 'reset_kwargs': {'options': {}, 'seed': seed_of(i)}, 'success': True})
    write_json(path.with_suffix('.json'), {'env_info': {'env_id': 'StackCube-v1', 'env_kwargs': {'obs_mode': 'rgb'}}, 'commit_info': {'commit_id': UPSTREAM_COMMIT}, 'episodes': eps})


@pytest.fixture
def files(tmp_path):
    src = tmp_path / 'trajectory.h5'; write_source(src); (tmp_path / 'old').mkdir(); (tmp_path / 'new').mkdir()
    old = tmp_path / 'old' / m.H5NAME; write_converted(old, range(N_OLD_CONV), drop=(2,))
    new = tmp_path / 'new' / m.H5NAME; write_converted(new, range(N_OLD_CONV, N_OLD_CONV + WINDOW), drop=(15,))
    return src, old, new


def make_plan(src, old, num_new=N_NEW):
    oi = read_json(old.with_suffix('.json')); si = read_json(src.with_suffix('.json')); by_seed = {e['episode_seed']: e['episode_id'] for e in si['episodes']}
    row = lambda e: {'traj_id': e['episode_id'], 'source_episode_id': by_seed[e['episode_seed']], 'scene_seed': e['episode_seed'], 'steps': e['elapsed_steps']}
    dm = {'train': [row(e) for e in oi['episodes'][:N_TRAIN]], 'heldout': [row(e) for e in oi['episodes'][N_TRAIN:N_TRAIN + N_DEV]],
          'dropped_source_episodes': [{'source_episode_id': 2, 'scene_seed': seed_of(2)}]}
    roles = m.roles_from_manifest(dm, oi, si)
    return {'pass': True, 'roles': roles, 'inputs': {'old_demo_sha256': sha256(old), 'source_sha256': sha256(src)},
            'new_candidates': {'rule': 'test', 'num_new': num_new, 'episodes': m.candidate_window(si, N_OLD_CONV, WINDOW)}}


def test_build_keeps_d100_and_excludes_protected(files, tmp_path):
    src, old, new = files; plan = make_plan(src, old); out = tmp_path / 'data'; out.mkdir(); audit = m.build(plan, old, new, src, out, horizon=200)
    assert audit['pass'], audit['problems']
    assert audit['counts'] == {'D100': N_TRAIN, 'Dnew400': N_NEW, 'D500': N_TRAIN + N_NEW, 'V20': N_DEV, 'protected_remainder': N_OLD_CONV - 1 - N_TRAIN - N_DEV, 'new_dropped': 1, 'new_unused': WINDOW - 1 - N_NEW}
    assert audit['old_train_payload_equal'] and audit['new_train_payload_equal'] and audit['train_and_protected_seed_overlap'] == []
    man = read_json(out / 'train_d500_manifest.json'); prot = read_json(out / 'protected_dev_manifest.json'); train = out / 'train_d500' / m.H5NAME
    with h5py.File(train, 'r') as ft, h5py.File(old, 'r') as fo, h5py.File(new, 'r') as fn:
        # what the upstream loader reads with num_traj = N: numeric sort, first N -> every trajectory of this file, in manifest order
        keys = sorted(ft.keys(), key=lambda x: int(x.split('_')[-1]))[:N_TRAIN + N_NEW]
        assert keys == [f'traj_{i}' for i in range(N_TRAIN + N_NEW)] and len(ft.keys()) == N_TRAIN + N_NEW
        for r in man['train']:
            origin = fo if r['role'] == 'old_train' else fn
            for name in ('actions', 'obs/sensor_data/base_camera/rgb', 'obs/sensor_data/hand_camera/rgb', 'obs/agent/qpos'):
                assert np.array_equal(ft[f"traj_{r['train_traj_id']}/{name}"][()], origin[f"traj_{r['origin_traj_id']}/{name}"][()])
    assert [r['role'] for r in man['train']] == ['old_train'] * N_TRAIN + ['new_train'] * N_NEW
    assert [r['source_episode_id'] for r in man['train']] == [0, 1, 3, 4, 5] + [12, 13, 14, 16, 17, 18, 19, 20]      # dropped 2 and 15 are skipped, not renumbered away
    protected = {s['scene_seed'] for s in prot['scenes']}; assert len(protected) == N_OLD_CONV - 1 - N_TRAIN and not protected & {r['scene_seed'] for r in man['train']}
    assert protected <= set(man['all_scene_seeds']) and seed_of(15) in man['all_scene_seeds']
    eps = read_json(train.with_suffix('.json'))['episodes']
    assert [e['episode_id'] for e in eps] == list(range(N_TRAIN + N_NEW)) and [e['episode_seed'] for e in eps] == [r['scene_seed'] for r in man['train']]
    sample = read_json(out / 'io_parity_sample' / m.H5NAME.replace('.h5', '.json'))['episodes']
    assert len(sample) == 6 and sum(e['hci_origin']['role'] == 'new_train' for e in sample) == 3


def test_initial_scene_comes_from_source_first_state(files, tmp_path):
    src, old, new = files; plan = make_plan(src, old); out = tmp_path / 'data'; out.mkdir(); m.build(plan, old, new, src, out, horizon=200)
    man = read_json(out / 'train_d500_manifest.json'); r = man['train'][0]
    with h5py.File(src, 'r') as fs, h5py.File(old, 'r') as fo:
        fp, key = m.source_scene(fs[f"traj_{r['source_episode_id']}"])
        assert r['initial_scene_fingerprint'] == fp and key[0] == pytest.approx(float(fs['traj_0/env_states/actors/cubeA'][0, 0]))
        assert key[0] != pytest.approx(float(fo['traj_0/env_states/actors/cubeA'][0, 0]))       # the converted first state is the second source state
    assert man['first_env_state_note']['converted0_vs_source1_max'] == 0 and man['first_env_state_note']['converted0_vs_source0_max'] > 0
    assert 'source' in man['initial_state_source']


def test_too_few_new_demonstrations_fail(files, tmp_path):
    src, old, new = files; out = tmp_path / 'data'; out.mkdir(); audit = m.build(make_plan(src, old, num_new=WINDOW), old, new, src, out, horizon=200)
    assert not audit['pass'] and any('required' in p for p in audit['problems'])


def test_repeated_scene_and_protected_seed_are_detected(tmp_path):
    src = tmp_path / 'trajectory.h5'; write_source(src, same_scene={13: 6}); (tmp_path / 'old').mkdir(); (tmp_path / 'new').mkdir()   # new episode 13 repeats protected scene 6
    old = tmp_path / 'old' / m.H5NAME; write_converted(old, range(N_OLD_CONV), drop=(2,)); new = tmp_path / 'new' / m.H5NAME; write_converted(new, range(N_OLD_CONV, N_OLD_CONV + WINDOW))
    out = tmp_path / 'data'; out.mkdir(); audit = m.build(make_plan(src, old), old, new, src, out, horizon=200)
    assert not audit['pass'] and audit['duplicate_initial_placements']
    new2 = tmp_path / 'new2'; new2.mkdir(); write_converted(new2 / m.H5NAME, [6, *range(N_OLD_CONV, N_OLD_CONV + WINDOW)])                # a protected episode inside the new conversion
    out2 = tmp_path / 'data2'; out2.mkdir(); audit = m.build(make_plan(src, old), old, new2 / m.H5NAME, src, out2, horizon=200)
    assert not audit['pass'] and any('outside the declared window' in p for p in audit['problems'])


def test_first_success_after_horizon_is_recorded_not_removed(files, tmp_path):
    src, old, new = files; out = tmp_path / 'data'; out.mkdir(); audit = m.build(make_plan(src, old), old, new, src, out, horizon=4)
    man = read_json(out / 'train_d500_manifest.json')
    assert audit['pass'] and man['first_success']['n_after_horizon'] > 0 and man['num_train'] == N_TRAIN + N_NEW


def test_overlap_rules():
    roles = {'old_train': [{'scene_seed': 100}, {'scene_seed': 1}], 'protected_dev': [{'scene_seed': 102}], 'protected_remainder': [], 'old_dropped': []}
    rep, problems = m.overlap_report(roles, [{'scene_seed': 500}], {'splits/audit': [100, 102], 'policy_eval/policy_qualify_B': [50300]})
    assert problems == [] and rep['with_declared_splits'] == {'splits/audit': [100, 102]}
    assert m.overlap_report(roles, [{'scene_seed': 50300}], {'policy_eval/policy_qualify_B': [50300]})[1]
    assert m.overlap_report(roles, [{'scene_seed': 100}], {})[1]                                  # a new candidate repeating an old training seed
    assert m.overlap_report(roles, [{'scene_seed': 7}], {'splits/audit': [7]})[1]                 # the audit exception does not cover new demonstrations


def test_payload_hash_ignores_storage_layout(tmp_path):
    x = np.arange(600, dtype=np.uint8).reshape(2, 10, 10, 3)
    with h5py.File(tmp_path / 'a.h5', 'w') as a, h5py.File(tmp_path / 'b.h5', 'w') as b:
        a.create_dataset('g/x', data=x, compression='gzip'); b.create_dataset('g/x', data=x, chunks=(1, 10, 10, 3)); b2 = b.create_group('h'); b2['x'] = x.astype(np.int16)
        assert m.dataset_hashes(a['g']) == m.dataset_hashes(b['g']) != m.dataset_hashes(b['h'])
