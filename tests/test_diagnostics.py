import numpy as np, pytest
from hci.diagnostics import even_pick, window_steps, gripper_phases, phase_of, first_true, stage_events, rgb_diff_stats, action_errors

def test_even_pick_is_fixed_and_spans_the_list():
    assert even_pick(range(100), 10) == [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
    assert even_pick(range(100, 120), 10) == [100, 102, 104, 106, 108, 111, 113, 115, 117, 119]
    assert even_pick([3, 4], 10) == [3, 4]

def test_windows_leave_eight_recorded_actions():
    w = window_steps(107); assert w == [0, 20, 40, 59, 79] and all(t + 8 <= 107 for t in w)
    assert window_steps(8) == [0] and window_steps(7) == []

def test_phases_from_gripper_commands():
    g = [1] * 40 + [-1] * 50 + [1] * 10; c, o = gripper_phases(g); assert (c, o) == (40, 90)
    assert phase_of(0, c, o) == 'approach' and phase_of(32, c, o) == 'approach' and phase_of(33, c, o) == 'boundary'
    assert phase_of(40, c, o) == 'transport' and phase_of(82, c, o) == 'transport' and phase_of(85, c, o) == 'boundary' and phase_of(90, c, o) == 'place'
    assert gripper_phases([1, 1]) == (None, None) and phase_of(0, None, None) == 'unknown'
    assert first_true([False, False, True, True]) == 3 and first_true([False]) is None

def row(t, d=0.2, grasp=False, z=0.02, xy=0.2, on=False, ok=False, g=1.0):
    return {'t': t, 'tcp_to_cubeA': d, 'is_cubeA_grasped': grasp, 'cubeA': [0, 0, z], 'cubeB': [xy, 0, 0.02], 'cubeA_to_cubeB_xy': xy,
            'is_cubeA_on_cubeB': on, 'success': ok, 'action': [0, 0, 0, g]}

def test_stage_ranking_keeps_recorded_grasp():
    init = row(0); assert stage_events(init, [])['furthest_stage'] == 'unreadable'
    assert stage_events(init, [row(1)])['furthest_stage'] == 'not_near'
    assert stage_events(init, [row(1, d=0.01)])['furthest_stage'] == 'near_no_grasp'
    # Stage 1B seed 50107: grasp flag and lift recorded while the TCP-centre distance stayed just above the threshold
    ev = stage_events(init, [row(1, d=0.031, grasp=True, g=-1.0), row(2, d=0.031, grasp=True, z=0.06, g=-1.0)])
    assert ev['furthest_stage'] == 'lifted_not_aligned' and ev['first_grasp_step'] == 1 and ev['first_close_command_step'] == 1 and ev['first_near_step'] is None
    assert stage_events(init, [row(1, grasp=True)])['furthest_stage'] == 'grasped_not_lifted'
    assert stage_events(init, [row(1, grasp=True, z=0.06, xy=0.01)])['furthest_stage'] == 'aligned_no_success'
    assert stage_events(init, [row(1, on=True, ok=True)])['first_success_step'] == 1

def test_rgb_and_action_statistics_units():
    a = np.zeros((4, 4, 3), np.uint8); b = a.copy(); b[0, 0] = 40
    s = rgb_diff_stats(a, b); assert s['max_u8'] == 40 and s['frac_ge32'] == pytest.approx(3 / 48) and s['frac_ge1'] == s['frac_ge8']
    e = np.zeros((8, 4)); e[:, 3] = -1; p = e.copy(); p[:, 0] = 0.1; p[4:, 3] = 1
    r = action_errors(p, e); assert r['mae_x'] == pytest.approx(0.1) and r['signed_x'] == pytest.approx(0.1) and r['mae_y'] == 0
    assert r['gripper_sign_agree'] == 0.5 and r['gripper_first_sign_agree'] == 1.0 and r['first_l2_xyz'] == pytest.approx(0.1)
