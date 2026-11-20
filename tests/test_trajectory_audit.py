"""Trajectory audit (hci.trajectory_audit) and the C1 record fields it needs (frames after the trigger, branch id, events).

Mock physics and a synthetic renderer, as in test_vision_center_c1.py. Nothing here says anything about ManiSkill."""
import copy, gzip, importlib.util, json
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from test_recovery_v1_oracle import CFG, probe
from test_vision_center import BiasPolicy
from test_vision_center_v2 import VisionEnv2
from hci.simulation import Runtime
from hci import vision_center_c1 as C1
from hci import recovery_v1_oracle as O
from hci import trajectory_audit as T


def probe2(rt, z):
    w = O.finger_width(rt.raw)
    return {**probe(rt, z), 'tcp_q': [1., 0., 0., 0.], 'finger_qpos': [w / 2, w / 2], 'f_A_B': 0., 'f_B_left': 0., 'f_B_right': 0.}


def cscene(bias, arms=C1.BASE_ARMS, seed=7):
    env = VisionEnv2(); r = Runtime(env, BiasPolicy(bias), CFG); return C1.collect_scene(r, seed, CFG, probe=probe2, arms=arms, policy_id='train_seed_1')


@pytest.fixture(scope='module')
def moved():
    return cscene((.028, .012))


@pytest.fixture(scope='module')
def stored(moved, tmp_path_factory):
    spec = importlib.util.spec_from_file_location('c1_driver_t', Path('scripts/34_vision_center_c1.py')); X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X)
    out = tmp_path_factory.mktemp('pass'); rec, traces, frames = copy.deepcopy(moved); X.Z.save_scene(out, 7, rec, traces, frames); return out


def cell(rec, arm): return next(x for x in rec['cells'] if x['arm'] == arm)


# ---------------------------------------------------------------- C1 record fields
def test_post_frame_steps_follow_the_fixed_policy():
    s = C1.post_frame_steps(100, 200, 6, 110)
    assert s[0] == 101 and all(q in s for q in range(101, 131)) and 131 not in s and 132 in s and 200 in s and all((q - 100) % 4 == 0 for q in s if 130 < q < 200)
    assert C1.post_frame_steps(100, 104, 0, None) == [101, 102, 103, 104] and C1.post_frame_steps(100, 100, 0, None) == []
    assert C1.first_open_after([{'t': 101, 'action': [0, 0, 0, -1.]}, {'t': 103, 'action': [0, 0, 0, 1.]}], 100, 2) == 103 and C1.first_open_after([{'t': 101, 'action': [0, 0, 0, 1.]}], 100, 2) is None


def test_every_cell_carries_branch_snapshot_events_and_its_frames_or_a_pointer_to_C0(moved):
    rec, traces, frames = moved; t = rec['trigger']['step']; assert rec['snapshot_digest'] and 'frame_index' in rec
    for c in rec['cells']:
        assert c['branch_id'] == f"7:train_seed_1:t{t}:{c['arm']}" and c['snapshot_digest'] == rec['snapshot_digest']
        fa = c['frames_after_trigger']; rows = traces[c['arm']]
        if c['arm'] == 'C0' or c['acted']:
            k = fa['key']; assert k == f"after_{c['arm']}" and len(frames[k]) == len(fa['steps']) and fa['steps'] == C1.post_frame_steps(t, c['steps'], c['control_steps'], c['first_open_command_after_move'])
            assert fa['steps'][-1] == c['steps'] and frames[k][0].shape == frames['trunk'][0].shape
        else: assert fa['same_as'] == 'C0' and fa['max_logged_position_difference_to_C0_m'] == 0. and f"after_{c['arm']}" not in frames
        assert c['success_step'] == next((x['t'] for x in rows if x['success']), None)
    acted = [c['arm'] for c in rec['cells'] if c['acted']]; assert acted and 'VISION_CENTER_V2' in acted


def test_frames_after_the_trigger_are_the_observations_of_those_steps(moved):
    """The C0 frame at the last step equals a fresh render of the same state: re-run C0 alone and compare."""
    rec, _, frames = moved; again, _, f2 = cscene((.028, .012), arms=('C0',))
    assert np.array_equal(frames['after_C0'], f2['after_C0']) and again['trunk'] == rec['trunk']
    t = rec['trigger']['step']; steps = cell(rec, 'C0')['frames_after_trigger']['steps']
    assert len(frames['trunk']) == t + 1 and steps[0] == t + 1


def test_recording_does_not_change_any_outcome(moved):
    """Frames are cut from the observation after each step; outcomes and traces equal a run of each arm alone."""
    rec, traces, _ = moved
    for arm in ('VISION_CENTER_V2', 'CENTER_ORACLE', 'V2_RULES_OFF'):
        one, tr1, _ = cscene((.028, .012), arms=('C0', arm)); assert (cell(one, arm)['y'], cell(one, arm)['steps']) == (cell(rec, arm)['y'], cell(rec, arm)['steps']) and tr1[arm] == traces[arm]


# ---------------------------------------------------------------- the audit on a stored mock scene
def test_audit_reads_a_stored_scene_and_classifies_every_check(stored):
    rec, tr, fr = T.load_scene(stored, 7); m = T.scene_metrics(rec, tr); assert m['triggered'] and set(m['arms']) == set(C1.BASE_ARMS)
    for arm, v in m['arms'].items():
        assert all(x['status'] in T.STATUS for x in v['checks'].values()), arm
        json.dumps(v)                                                                     # plain JSON types only
        if arm != 'C0': assert set(v['checks']) == set(T.CHECKS), arm
    v2 = m['arms']['VISION_CENTER_V2']['checks']; assert m['arms']['VISION_CENTER_V2']['acted'] and v2['5_move_bounds']['status'] == 'PASS' and v2['9_abstain_keeps_policy']['status'] == 'NOT_OBSERVABLE'
    v1 = m['arms']['VISION_CENTER_V1']; assert not v1['acted'] and v1['checks']['9_abstain_keeps_policy']['status'] == 'PASS' and v1['checks']['3_command_reduces_true_error']['status'] == 'NOT_OBSERVABLE'


def test_checks_detect_a_push_a_wrong_direction_and_a_changed_abstention(stored):
    rec, tr, _ = T.load_scene(stored, 7, frames=False); t0 = rec['trigger']['step']; k = cell(rec, 'VISION_CENTER_V2')['control_steps']
    bad = copy.deepcopy(tr)
    for r in bad['VISION_CENTER_V2']:
        if t0 < r['t'] <= t0 + k: r['cubeB'] = [r['cubeB'][0] + .002 * (r['t'] - t0), r['cubeB'][1], r['cubeB'][2]]
    assert T.check_arm(rec, bad, 'VISION_CENTER_V2')['7_no_slip_no_push']['status'] == 'FAIL'
    rec2 = copy.deepcopy(rec)
    for x in cell(rec2, 'VISION_CENTER_V2')['control']['steps']: x['commanded_xy_m'] = [-v for v in x['commanded_xy_m']]
    c = T.check_arm(rec2, tr, 'VISION_CENTER_V2'); assert c['3_command_reduces_true_error']['status'] == 'FAIL' and c['4_motion_follows_command']['status'] == 'FAIL'
    bad = copy.deepcopy(tr); bad['VISION_CENTER_V1'][-1]['tcp'] = [9., 9., 9.]
    assert T.check_arm(rec, bad, 'VISION_CENTER_V1')['9_abstain_keeps_policy']['status'] == 'FAIL'


def test_series_refuses_gaps_and_the_time_convention_holds(stored):
    rec, tr, _ = T.load_scene(stored, 7, frames=False); t0 = rec['trigger']['step']; s = T.series(rec, tr, 'VISION_CENTER_V2')
    assert s['t'][0] == 1 and np.all(np.diff(s['t']) == 1) and T.at(s, t0) == t0 - 1
    bad = copy.deepcopy(tr); del bad['VISION_CENTER_V2'][1]
    with pytest.raises(ValueError, match='consecutive'): T.series(rec, bad, 'VISION_CENTER_V2')
    sync = T.control_tcp_sync(rec, tr); assert sync and all(v['max_difference_m'] < 1e-9 for v in sync.values())


def test_figures_and_rgb_panels_are_made_and_label_ground_truth(stored, tmp_path):
    pytest.importorskip('matplotlib'); pytest.importorskip('PIL')                     # analysis libraries; the run server may not have them
    rec, tr, fr = T.load_scene(stored, 7)
    for name, fn in (('a.png', T.fig_3d), ('b.png', T.fig_xy), ('c.png', T.fig_timeline), ('d.png', T.outcome_strip)): fn(rec, tr, tmp_path / name); assert (tmp_path / name).stat().st_size > 1000
    raw = T.rgb_panels(rec, tr, fr, overlay=False); gt = T.rgb_panels(rec, tr, fr, overlay=True); assert len(raw) == len(gt) > 0
    T.save_gif(raw, tmp_path / 'r.gif'); assert (tmp_path / 'r.gif').stat().st_size > 1000
    src = Path('src/hci/trajectory_audit.py').read_text(encoding='utf-8'); assert src.count('POSTHOC') >= 5 and 'NO RGB RECORDED' in src
    for bad in ('interpolat', 'render('): assert bad not in src.replace('nothing is interpolated or re-rendered', '')


def test_rgb_panels_use_the_frames_kept_after_the_trigger(stored):
    pytest.importorskip('PIL')
    rec, tr, fr = T.load_scene(stored, 7); t0 = rec['trigger']['step']; c0 = cell(rec, 'C0')['frames_after_trigger']
    pan = T.rgb_panels(rec, tr, fr, window=0, scale=1); first = 0
    q = c0['steps'][0]; img = np.asarray(pan[q - t0])[44:, :256]; assert np.array_equal(img, fr['after_C0'][0])          # C0 column at step q is the stored frame of step q
    shown = [*range(t0, max(t0 + cell(rec, a)['control_steps'] for a in rec['arms']) + 4)]
    assert np.array_equal(np.asarray(pan[-1])[44:, :256], fr['after_C0'][-1]) and len(pan) >= len(set(shown) | set(c0['steps']))     # the last kept step is shown
