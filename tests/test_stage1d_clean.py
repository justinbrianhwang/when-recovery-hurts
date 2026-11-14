"""Stage 1D clean rebuild: episode-content validator, exclusion/replacement, and the training-start gate.
Simulator-free (needs h5py for the file tests)."""
import importlib.util, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from hci.integrity import content_integrity, window_counts, match_source

ROOT = Path(__file__).resolve().parents[1]
CFG = str(ROOT / 'configs' / 'stackcube.yaml')


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), ROOT / 'scripts' / name)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


# ---------- synthetic episodes ----------
def qpos0(i): return np.linspace(0.1, 0.9, 9) + 0.01 * i


def cube(i, t, name): v = np.zeros(13); v[:3] = (0.05 * i + 0.0005 * t, (0.03 * i if name == 'A' else -0.03 * i - 0.1), 0.02); v[3] = 1; return v


def source_ep(i, L):
    return {'first_qpos': qpos0(i), 'state0': {n: cube(i, 0, 'A' if n == 'cubeA' else 'B') for n in ('cubeA', 'cubeB')},
            'state1': {n: cube(i, 1, 'A' if n == 'cubeA' else 'B') for n in ('cubeA', 'cubeB')}, 'steps': L}


def clean(i, L, trunc_at=4):
    """One converted episode of source i: smooth joints, +-1 gripper, success on the last two steps."""
    a = np.zeros((L, 4)); a[:, :3] = 0.1; a[:, 3] = np.where(np.arange(L) < L // 2, 1, -1); succ = np.zeros(L, bool); succ[-2:] = True
    q = qpos0(i)[None] + 0.01 * np.arange(L + 1)[:, None]
    return {'actions': a, 'terminated': succ.copy(), 'truncated': np.arange(L) >= trunc_at, 'success': succ, 'qpos': q,
            'cubeB': np.stack([cube(i, t + 1, 'B') for t in range(L + 1)]), 'cubeA': np.stack([cube(i, t + 1, 'A') for t in range(L + 1)]),
            'state0': {'cubeA': cube(i, 1, 'A'), 'cubeB': cube(i, 1, 'B')}}


def concatenated(j, Lj, i, Li, dummy=(0.3, -0.2, 0.1, 0.5036), same_scene=False):
    """Failed replay of source j, a reset entry with a sampled dummy action, then the adopted episode i."""
    p = clean(j, Lj); t = clean(j if same_scene else i, Li); p['success'][:] = False; p['terminated'][:] = False
    out = {'actions': np.concatenate([p['actions'][:-1], np.asarray(dummy, float)[None], t['actions']]),
           'terminated': np.concatenate([p['terminated'][:-1], [True], t['terminated']]), 'truncated': np.concatenate([p['truncated'][:-1], [False], t['truncated']]),
           'success': np.concatenate([p['success'][:-1], [False], t['success']]), 'qpos': np.concatenate([p['qpos'][:Lj], t['qpos']]),
           'cubeB': np.concatenate([p['cubeB'][:Lj], t['cubeB']]), 'cubeA': np.concatenate([p['cubeA'][:Lj], t['cubeA']]), 'state0': p['state0']}
    return out


# ---------- validator (regression list of the Stage 1D v0.2 spec, items 1-6) ----------
def test_single_episode_passes():
    r = content_integrity(clean(3, 30), source_ep(3, 30)); assert r['verdict'] == 'PASS' and not r['signals'] and r['boundary_steps'] == []


def test_failed_prefix_reset_then_success_fails_and_locates_the_boundary():
    r = content_integrity(concatenated(2, 20, 3, 30), source_ep(3, 30))
    assert r['verdict'] == 'FAIL' and set(r['hard_signals']) == {'internal_terminal', 'truncated_restart', 'first_obs_not_source', 'first_state_not_source'}
    assert r['boundary_steps'] == [19] and 'gripper_not_binary' in r['soft_signals'] and 'kinematic_jump' in r['soft_signals']


def test_normal_length_and_final_success_do_not_rescue_a_concatenated_trajectory():
    t = concatenated(2, 20, 3, 30); r = content_integrity(t, source_ep(3, 50))        # declared source happens to have the same total length
    assert len(t['actions']) == 50 and t['success'][-1] and 'length_not_source' not in r['soft_signals'] and r['verdict'] == 'FAIL'


def test_reset_is_found_without_a_physical_jump_or_an_odd_gripper_value():
    # same scene before and after, joints continue smoothly, dummy action looks like a normal command
    t = concatenated(3, 20, 3, 30, dummy=(0.1, 0.1, 0.1, 1.0), same_scene=True); t['qpos'] = qpos0(3)[None] + 0.001 * np.arange(len(t['qpos']))[:, None]
    t['cubeB'][:] = t['cubeB'][0]; r = content_integrity(t, source_ep(3, 30))
    assert 'kinematic_jump' not in r['signals'] and 'gripper_not_binary' not in r['signals'] and 'first_obs_not_source' not in r['signals']
    assert r['verdict'] == 'FAIL' and {'internal_terminal', 'truncated_restart'} <= set(r['hard_signals'])


def test_large_motion_alone_is_not_a_confirmed_failure():
    t = clean(3, 30); t['cubeB'][15:, 0] += 0.05                                     # e.g. the cube is knocked during contact
    r = content_integrity(t, source_ep(3, 30)); assert r['verdict'] == 'QUARANTINE' and r['hard_signals'] == [] and r['soft_signals'] == ['kinematic_jump']


def test_long_continuous_demonstration_and_success_flicker_pass():
    t = clean(3, 330, trunc_at=49); r = content_integrity(t, source_ep(3, 330)); assert r['verdict'] == 'PASS' and r['first_success_step'] == 329
    t = clean(3, 30); t['success'][-5] = True; t['terminated'][-5] = True              # true, false, true: stillness threshold, same episode
    r = content_integrity(t, source_ep(3, 30)); assert r['verdict'] == 'PASS' and r['success_flicker_steps'] == [25]


def test_continuous_gripper_values_are_only_a_signal_when_the_expert_is_binary():
    t = clean(3, 30); t['actions'][4, 3] = 0.2
    assert content_integrity(t, source_ep(3, 30))['verdict'] == 'QUARANTINE' and content_integrity(t, source_ep(3, 30), expert_gripper_binary=False)['verdict'] == 'PASS'


def test_departure_from_the_source_path_is_flagged_even_when_flags_look_normal():
    t = clean(3, 30); src = {**source_ep(3, 30), 'cubeA_path': np.stack([cube(3, k, 'A') for k in range(31)])}
    assert content_integrity(t, src)['verdict'] == 'PASS'
    t['cubeA'][10:, 1] += 0.2; r = content_integrity(t, src); assert r['verdict'] == 'QUARANTINE' and 'cube_path_not_source' in r['soft_signals']


def test_content_provenance_and_window_counts():
    table = {i: source_ep(i, 10)['state1'] for i in range(6)}; t = concatenated(2, 20, 3, 30)
    assert match_source(t['state0'], table) == [2] and match_source(clean(3, 30)['state0'], table) == [3]
    w = window_counts(205, 108); assert w == {'windows': 204, 'obs_in_failed_prefix': 109, 'obs_across_reset': 1, 'target_contains_dummy_action': 16, 'touched_by_prefix_or_reset': 110}


# ---------- files: audit -> build -> gate (items 7-10) ----------
h5py = pytest.importorskip('h5py')
m16 = load('16_stage1d_data.py'); m17 = load('17_stage1d_clean.py')
from hci.io import UPSTREAM_COMMIT, read_json, write_json, sha256, load_config

N_SRC = 24; OLD = range(0, 10); OLD_FAIL = (3,); NEW = range(10, 22); NEW_FAIL = (12, 19); N_TRAIN, N_DEV, N_NEW = 5, 2, 6


def steps(i): return 6 + i % 3


def seed(i): return i + 1


def write_source(path):
    eps = []
    with h5py.File(path, 'w') as f:
        for i in range(N_SRC):
            L = steps(i); g = f.create_group(f'traj_{i}'); a = np.zeros((L, 8), np.float32); a[:, -1] = 1; g['actions'] = a
            for n in ('cubeA', 'cubeB'): g[f'env_states/actors/{n}'] = np.stack([cube(i, t, n[-1]) for t in range(L + 1)]).astype(np.float32)
            art = np.zeros((L + 1, 31), np.float32); art[:, 13:22] = qpos0(i); g['env_states/articulations/panda_wristcam'] = art
            eps.append({'episode_id': i, 'episode_seed': seed(i), 'control_mode': 'pd_joint_pos', 'elapsed_steps': L, 'reset_kwargs': {'options': {}, 'seed': seed(i)}, 'success': True})
    write_json(path.with_suffix('.json'), {'env_info': {'env_id': 'StackCube-v1'}, 'episodes': eps})


def write_converted(path, ids, fail):
    """Mimics the pinned converter: a failed replay is not saved and its buffer is prepended to the next saved episode."""
    eps = []; pending = None; rng = np.random.default_rng(len(ids))
    with h5py.File(path, 'w') as f:
        for i in ids:
            if i in fail: pending = i; continue
            t = clean(i, steps(i), trunc_at=3) if pending is None else concatenated(pending, steps(pending), i, steps(i)); pending = None
            t['qpos'] = t['qpos'].astype(np.float32); L = len(t['actions']); k = len(eps); g = f.create_group(f'traj_{k}')
            for name in ('actions', 'terminated', 'truncated', 'success'): g[name] = t[name].astype(np.float32) if name == 'actions' else t[name]
            g['obs/agent/qpos'] = t['qpos']; g['obs/agent/qvel'] = np.zeros((L + 1, 9), np.float32); g['obs/extra/tcp_pose'] = np.zeros((L + 1, 7), np.float32)
            for cam in ('base_camera', 'hand_camera'): g.create_dataset(f'obs/sensor_data/{cam}/rgb', data=rng.integers(0, 255, (L + 1, 128, 128, 3), dtype=np.uint8), compression='gzip')
            g['env_states/actors/cubeA'] = t['cubeA'].astype(np.float32); g['env_states/actors/cubeB'] = t['cubeB'].astype(np.float32)
            g['env_states/actors/cubeA'][0] = t['state0']['cubeA']; g['env_states/actors/cubeB'][0] = t['state0']['cubeB']
            eps.append({'episode_id': k, 'episode_seed': seed(i), 'control_mode': 'pd_ee_delta_pos', 'elapsed_steps': L, 'reset_kwargs': {'options': {}, 'seed': seed(i)}, 'success': True})
    write_json(path.with_suffix('.json'), {'env_info': {'env_id': 'StackCube-v1', 'env_kwargs': {'obs_mode': 'rgb', 'control_mode': 'pd_ee_delta_pos', 'sim_backend': 'physx_cpu', 'num_envs': 1}},
                                          'commit_info': {'commit_id': UPSTREAM_COMMIT}, 'episodes': eps})


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """Source, old and new conversions, and a stage1d_001-style (contaminated) training manifest built by script 16."""
    tmp = tmp_path_factory.mktemp('s1d'); src = tmp / 'trajectory.h5'; write_source(src); (tmp / 'old').mkdir(); (tmp / 'new').mkdir()
    old = tmp / 'old' / m16.H5NAME; write_converted(old, list(OLD), OLD_FAIL); new = tmp / 'new' / m16.H5NAME; write_converted(new, list(NEW), NEW_FAIL)
    oi = read_json(old.with_suffix('.json')); si = read_json(src.with_suffix('.json')); by_seed = {e['episode_seed']: e['episode_id'] for e in si['episodes']}
    row = lambda e: {'traj_id': e['episode_id'], 'source_episode_id': by_seed[e['episode_seed']], 'scene_seed': e['episode_seed'], 'steps': e['elapsed_steps']}
    dm = {'train': [row(e) for e in oi['episodes'][:N_TRAIN]], 'heldout': [row(e) for e in oi['episodes'][N_TRAIN:N_TRAIN + N_DEV]], 'dropped_source_episodes': [{'source_episode_id': i, 'scene_seed': seed(i)} for i in OLD_FAIL]}
    plan = {'pass': True, 'roles': m16.roles_from_manifest(dm, oi, si), 'inputs': {'old_demo_sha256': sha256(old), 'source_sha256': sha256(src), 'source_json_sha256': sha256(src.with_suffix('.json'))},
            'new_candidates': {'rule': 'test', 'num_new': N_NEW, 'episodes': m16.candidate_window(si, len(OLD), len(NEW))}}
    d1 = tmp / 'data001'; d1.mkdir(); m16.build(plan, old, new, src, d1, horizon=200)       # passes: the Stage 1D v0.1 checks do not see the defect
    write_json(tmp / 'plan.json', plan)
    return SimpleNamespace(tmp=tmp, src=src, old=old, new=new, plan=tmp / 'plan.json', manifest=d1 / 'train_d500_manifest.json')


def run_audit(w, out):
    m17.cmd_audit(SimpleNamespace(config=CFG, plan1d=str(w.plan), d500_manifest=str(w.manifest), old_demo=str(w.old), new_demo=str(w.new), source=str(w.src), out=str(out), dry_run=False))


def run_build(w, audit, out):
    out.mkdir(); recheck = dict(config=CFG, plan1d=str(w.plan), d500_manifest=str(w.manifest), old_demo=str(w.old), new_demo=str(w.new), source=str(w.src))
    return m17.build(audit, read_json(w.plan), read_json(w.manifest), w.old, w.new, w.src, out, load_config(CFG), recheck)


@pytest.fixture(scope='module')
def built(world):
    audit = world.tmp / 'audit'; run_audit(world, audit); data = world.tmp / 'clean'; m = run_build(world, audit, data); return audit, data, m


def test_audit_finds_every_concatenated_trajectory_without_a_blacklist(world, built):
    audit, _, _ = built; s = read_json(audit / 'integrity_summary.json'); bad = {(x['file'], x['declared_source_episode_id']) for x in s['not_pass']}
    assert read_json(world.manifest)['pass']                                          # the earlier checks passed the contaminated file
    assert bad == {('old_converted', 4), ('new_converted', 13), ('new_converted', 20)} and all(x['verdict'] == 'FAIL' for x in s['not_pass'])
    assert [x['prefix_is_the_failed_episode'] for x in s['after_failed_source_episodes']] == [True, True, True] and s['foreign_content_in_protected_or_eval'] == []
    pairs = read_json(audit / 'replacement_manifest.json')['pairs']
    assert [(p['label'], p['replaces']['declared_source_episode_id'], p['replacement']['declared_source_episode_id'], p['belongs_to']) for p in pairs] == \
        [('r1', 4, 17, 'D100_clean_v1'), ('r2', 13, 18, 'Dnew400_clean_v1')]         # source 20 is in the reserve but fails, so 17 and 18 are taken in source order
    assert s['stage1d_001_contamination']['dummy_transitions'] == 2


def test_build_keeps_retained_payloads_and_the_built_file_passes(world, built):
    _, data, m = built; assert m['pass'], m['problems']
    assert m['counts'] == {'retained_old': N_TRAIN - 1, 'retained_new': N_NEW - 1, 'replacement': 2} and m['retained_payload_equal_to_stage1d_001'] and m['gripper_values'] == [-1.0, 1.0]
    rows = m['train']; assert [r['train_traj_id'] for r in rows] == list(range(N_TRAIN + N_NEW)) and all(r['content_verdict'] == 'PASS' for r in rows)
    assert not {4, 13, 20} & {r['source_episode_id'] for r in rows} and rows[3]['role'] == 'replacement' and rows[3]['source_episode_id'] == 17 and rows[3]['replaces']['stage1d_001_train_id'] == 3
    d100 = read_json(data / 'train_d100_clean_ids.json')['ids']; assert [x['source_episode_id'] for x in d100[:N_TRAIN]] == [0, 1, 2, 17, 5]
    g = read_json(data / 'data_gate.json'); assert [g['layers'][k]['status'] for k in m17.LAYERS] == ['PASS', 'PASS', 'PASS', 'NOT RUN', 'NOT RUN']
    with h5py.File(data / 'train_d500_clean_v1.h5', 'r') as f:
        assert all(np.all(np.abs(np.abs(f[k]['actions'][:, 3]) - 1) < 1e-6) for k in f)       # no reset dummy action is left in any training window


def test_replacement_from_the_protected_set_is_rejected(world, built):
    audit, _, _ = built; a2 = world.tmp / 'audit_protected'; a2.mkdir()
    for f in audit.iterdir(): (a2 / f.name).write_bytes(f.read_bytes())
    r = read_json(a2 / 'replacement_manifest.json'); dev = read_json(world.plan)['roles']['protected_dev'][0]
    r['pairs'][0]['replacement'] = {'file': 'old_converted', 'traj_id': dev['traj_id'], 'declared_source_episode_id': dev['source_episode_id'], 'scene_seed': dev['scene_seed']}
    write_json(a2 / 'replacement_manifest.json', r, overwrite=True)
    with pytest.raises(SystemExit, match='replacement_manifest.json is not what the validator produces'): run_build(world, a2, world.tmp / 'clean_protected')
    reserve = read_json(audit / 'replacement_manifest.json'); protected = {x['source_episode_id'] for k in ('protected_dev', 'protected_remainder') for x in read_json(world.plan)['roles'][k]}
    assert not protected & {p['replacement']['declared_source_episode_id'] for p in reserve['pairs']}


def test_changed_retained_payload_is_rejected(world, built):
    _, _, _ = built; man = read_json(world.manifest); man['train'][0]['payload_sha256'] = '0' * 64; w2 = SimpleNamespace(**{**vars(world), 'manifest': world.tmp / 'manifest_changed.json'})
    write_json(w2.manifest, man); audit = world.tmp / 'audit_changed'; run_audit(w2, audit); m = run_build(w2, audit, world.tmp / 'clean_payload')
    assert not m['pass'] and any('payload differs' in p for p in m['problems'])


def copy_data(data, dst):
    dst.mkdir()
    for f in data.iterdir():
        if f.is_file(): (dst / f.name).write_bytes(f.read_bytes())
    return dst


def add_reports(d):
    m = read_json(d / 'train_d500_clean_manifest.json')
    write_json(d / 'sampled_action_replay.json', {'pass': True, 'train_h5_sha256': m['train_h5_sha256'], 'replay': [{'traj_id': r['train_traj_id'], 'success_once': True} for r in m17.replay_picks(m)]})
    write_json(d / 'policy_io_parity.json', {'pass': True, 'demo_sha256': m['io_parity_sample_sha256'], 'same_raw_frames': {'n_trajectories': len(m['io_parity_sample_train_ids'])}, 'actions': {'all_pass': True}})


BUDGET = {'cap_usd': 2.0, 'spent_usd': 0.3, 'usd_per_hour': 0.3, 'train_limit_s': 7200, 'reserve_usd': 0.3}


def test_gate_requires_every_layer_and_the_same_files(world, built):
    _, data, _ = built; led = world.tmp / 'ledger'; led.mkdir(exist_ok=True)
    ok, rep = m17.gate(data, led, require=m17.LAYERS[:3]); assert ok, rep
    ok, rep = m17.gate(data, led, budget=BUDGET); assert not ok and {'sampled_action_replay: NOT RUN', 'policy_io_parity: NOT RUN'} <= set(rep['problems'])
    d = copy_data(data, world.tmp / 'g_ok'); add_reports(d); ok, rep = m17.gate(d, led, budget=BUDGET); assert ok, rep
    ok, rep = m17.gate(world.tmp / 'nowhere', led); assert not ok and 'missing validation report' in rep['problems'][0]
    d = copy_data(data, world.tmp / 'g_other'); add_reports(d)                                  # a report that validated another HDF5
    with h5py.File(d / 'train_d500_clean_v1.h5', 'a') as f: f['traj_0/actions'][0, 0] = 0.123
    ok, rep = m17.gate(d, led, budget=BUDGET); assert not ok and any('different HDF5' in p for p in rep['problems'])
    d = copy_data(data, world.tmp / 'g_replay'); add_reports(d); v = read_json(d / 'sampled_action_replay.json'); v['replay'] = v['replay'][1:]      # right file, one replacement not replayed
    write_json(d / 'sampled_action_replay.json', v, overwrite=True); ok, rep = m17.gate(d, led, budget=BUDGET); assert not ok and 'sampled_action_replay: FAIL' in rep['problems']
    d = copy_data(data, world.tmp / 'g_io'); add_reports(d); v = read_json(d / 'policy_io_parity.json'); v['same_raw_frames']['n_trajectories'] -= 1
    write_json(d / 'policy_io_parity.json', v, overwrite=True); ok, rep = m17.gate(d, led, budget=BUDGET); assert not ok and 'policy_io_parity: FAIL' in rep['problems']


def test_gate_refuses_unresolved_candidates_used_ledger_and_missing_budget(world, built):
    _, data, _ = built; led = world.tmp / 'ledger2'; led.mkdir()
    d = copy_data(data, world.tmp / 'g_unres'); add_reports(d); g = read_json(d / 'data_gate.json'); g['unresolved_candidates'] = [7]; write_json(d / 'data_gate.json', g, overwrite=True)
    ok, rep = m17.gate(d, led, budget=BUDGET); assert not ok and any('unresolved' in p for p in rep['problems'])
    d = copy_data(data, world.tmp / 'g_budget'); add_reports(d)
    ok, rep = m17.gate(d, led, budget={**BUDGET, 'train_limit_s': None}); assert not ok and any('budget field missing' in p for p in rep['problems'])
    ok, rep = m17.gate(d, led, budget={**BUDGET, 'spent_usd': 1.5}); assert not ok and any('exceeds' in p for p in rep['problems'])
    write_json(led / 'policy_qualify_B.json', {'set': 'policy_qualify_B'}); ok, rep = m17.gate(d, led, budget=BUDGET); assert not ok and any('already consumed' in p for p in rep['problems'])
