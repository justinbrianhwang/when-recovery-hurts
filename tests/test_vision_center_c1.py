"""Vision-CENTER v2 Phase C1: ablation arms, the safety candidate v2s, the fixed statistics and the guarded driver.

Mock physics and a synthetic renderer (flat colours, the module's own camera constants). These tests say nothing about
ManiSkill, the GPU or the research policy."""
import argparse, importlib.util, inspect, json, time
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from test_recovery_v1_oracle import CFG, probe
from test_vision_center import BiasPolicy, render, pose, held, target
from test_vision_center_v2 import VisionEnv2, render_base, far, blank, remembered, HID_T, HID_P
from hci.simulation import Runtime
from hci import vision_center as V1
from hci import vision_center_v2 as V2
from hci import vision_center_c1 as C1
from hci import vision_center_v2s as S
from hci import recovery_v1_oracle as O


def so(t, hand, base, p, width=.04, grip=-1.0): return C1.StepObs(t, hand, base, p, width, grip)


def cell(rec, arm): return next(x for x in rec['cells'] if x['arm'] == arm)


# ---------------------------------------------------------------- shared fits and the mirrored memory
def test_reading_from_a_stepobs_is_the_v1_reading():
    p = pose(.03, -.02)
    for cubes in ([target(.05, .0, .3), held(p)], [held(p, (.010, -.002, -.006), -.3)], []):
        img = render(p, cubes); a = V1.reading(img, p); b = C1.reading_from(so(5, img, None, p))
        assert V1.summary(a) == V1.summary(b)


def sequence():
    """Steps that exercise every branch of Memory.update: base views, a hand view, hidden steps near and far, a held cube, finger motion."""
    b = render_base([target(.0, .10)]); near = pose(.02, .10, .065); p = pose(.03, -.02); hold_img = render(p, [held(p, (.010, -.002, -.006), -.3)], finger=True); seq = []
    for t in range(3): seq.append((t, blank(), b, far(), .08, 1.0))
    seq.append((3, render(pose(.0, .10, .12), [target(.0, .10)]), b, pose(.0, .10, .12), .08, 1.0)); seq.append((4, hold_img, None, p, .04, -1.0)); seq.append((5, blank(), None, near, .04, -1.0)); seq.append((6, blank(), None, near, .04, -1.0))
    seq.append((7, blank(), render_base([target(.05, .14)]), far(), .04, -1.0)); seq.append((8, hold_img, None, p, .04, -1.0)); seq.append((9, blank(), None, near, .043, -1.0)); seq.append((10, blank(), b, far(), .08, 1.0)); return seq


def test_mirrored_memory_equals_the_original_step_by_step():
    a = V2.Memory(); b = C1.Mem()
    for t, hand, base, p, w, g in sequence():
        oa = a.update(t, hand, base, p, w, g); ob = b.update_obs(so(t, hand, base, p, w, g)); assert oa == ob and C1.same_memory(a, b), t
    assert a.target_estimate()[1] == b.target_estimate()[1] and a.log['conflicts'] >= 1 and a.log['hold_obs'] >= 1 and a.log['hold_dropped'] >= 1
    c = b.copy(); c.update_obs(so(11, blank(), None, pose(.05, .14, .065), .04, -1.0)); assert C1.same_memory(a, b) and not C1.same_memory(a, c)      # a copy is independent


def test_switched_off_components_never_enter_the_memory():
    nh, nb, no = C1.Mem(use_hand=False), C1.Mem(use_base=False), C1.Mem(use_hold=False)
    for t, hand, base, p, w, g in sequence():
        for m in (nh, nb, no): m.update_obs(so(t, hand, base, p, w, g))
    assert nh.log['hand_obs'] == 0 and nh.log['base_obs'] > 0 and nb.log['base_obs'] == 0 and nb.log['hand_obs'] == 1 and no.log['hold_obs'] == 0 and no.hold is None
    p = pose(.03, -.02); img = render(p, [held(p)]); rd = V1.reading(img, p); no.note_hold(9, rd['hold_fit'], .04); assert no.hold is None and no.hold_estimate(.04, -1.0) is None
    assert set(C1.MEMORY_VARIANTS) == {'V2_NO_HAND_MEMORY', 'V2_NO_BASE_MEMORY', 'V2_NO_HOLD_MEMORY'} and [list(v) for v in C1.MEMORY_VARIANTS.values()] == [['use_hand'], ['use_base'], ['use_hold']]


def test_single_frame_memory_holds_the_trigger_step_only_and_accepts_one_base_view():
    b = render_base([target(*HID_T)]); p = pose(*HID_P); m = C1.single_frame_memory(so(60, blank(), b, p)); e, why = m.target_estimate()
    assert why is None and e['src'] == 'base' and e['confirmations'] == 1 and e['t_last'] == 60 and m.steps == 1 and C1.SINGLE_FRAME == {'confirm_min': 1}
    assert C1.single_frame_memory(so(60, blank(), None, p)).target_estimate() == (None, 'never_seen')
    full = remembered(HID_T, n=1); assert full.target_estimate() == (None, 'not_confirmed')                           # the same single view is not enough for v2 itself


def test_rules_off_keeps_the_v1_rules_and_drops_the_v2_bundle():
    est = lambda n, u, gap, src='frame', z=None: {'why': None, 'offset_norm_m': n, 'u': u, 'gap_m': gap, 'target_src': src, 'held_xyz': np.array([0., 0., .06 + gap if z is None else z])}
    d = lambda *a, **k: C1.decide_rules_off(est(*a, **k), 100)['reason']; v = lambda *a, **k: V2.decide(est(*a, **k), 100)['reason']
    assert v(.016, .007, .005, 'memory_base') == 'ALIGNED_WITHIN_UNCERTAINTY' and d(.016, .007, .005, 'memory_base') == 'ACT'
    assert v(.0409, .004, -.0002) == 'SIDE_CONTACT_RISK' and d(.0409, .004, -.0002) == 'ACT'
    assert v(.030, .007, -.001, 'memory_base') == 'LOW_CARRY' and d(.030, .007, -.001, 'memory_base') == 'ACT'
    assert d(.030, .004, -.004) == 'LOW_CARRY' and d(.014, .004, .005) == 'ALIGNED' and C1.decide_rules_off(est(.03, .004, .005), 10)['reason'] == 'TOO_FEW_STEPS' and C1.decide_rules_off({'why': 'NO_HOLD_ESTIMATE'}, 100)['reason'] == 'NO_HOLD_ESTIMATE'
    assert len(C1.RULES_OFF_REMOVED) == 3


# ---------------------------------------------------------------- the copied control loop against the original
def to_trigger(bias, seed=7):
    env = VisionEnv2(); r = Runtime(env, BiasPolicy(bias), CFG); r.reset(seed); m = V2.Memory(); n = C1.Mem()
    while True:
        hand, base, p, w, g = V2.observe(r); m.update(r.t, hand, base, p, w, g); n.update_obs(so(r.t, hand, base, p, w, g))
        if O.trigger_reading(r)['fires']: return env, r, m, n
        r.step_base()


@pytest.mark.parametrize('bias', [(.028, .012), (-.028, .0), (-.004, .0)])
def test_center_move_with_defaults_is_the_original_v2_controller(bias):
    e1, r1, m1, n1 = to_trigger(bias); e2, r2, m2, n2 = to_trigger(bias); assert C1.same_memory(m1, n2)
    a = V2.vision_center_v2(r1, m1, fa := []); b = C1.center_move(r2, n2, fb := [], mode='VISION_CENTER_V2')
    drop = ('mode', 'memory_updated_during_control', 'memory_log_after'); assert {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}
    assert r1.t == r2.t and r1.intervention == r2.intervention and np.allclose(r1.tcp_history[-1], r2.tcp_history[-1], atol=0) and len(fa) == len(fb) and all((x == y).all() for x, y in zip(fa, fb))
    assert torch.equal(e1.p, e2.p) and torch.equal(e1.A, e2.A) and torch.equal(e1.B, e2.B)


# ---------------------------------------------------------------- one scene, every arm
def cscene(bias, arms=C1.ARMS, seed=7):
    env = VisionEnv2(); r = Runtime(env, BiasPolicy(bias), CFG); return (*C1.collect_scene(r, seed, CFG, probe=probe, arms=arms, policy_id='train_seed_1'), env)


def test_scene_with_every_arm_and_the_v2_arm_equals_the_v2_harness():
    rec, traces, frames, env = cscene((.028, .012)); assert rec['triggered'] and rec['arms'] == list(C1.ARMS) and C1.validate_scene(rec, CFG) and set(traces) == {'trunk', *C1.ARMS}; json.dumps(rec)
    env2 = VisionEnv2(); r2 = Runtime(env2, BiasPolicy((.028, .012)), CFG); ref, *_ = V2.collect_scene(r2, 7, CFG, probe=probe)
    for arm in ('C0', 'VISION_CENTER_V1', 'VISION_CENTER_V2', 'CENTER_ORACLE'):
        a, b = cell(rec, arm), cell(ref, arm); assert (a['y'], a['steps'], a['acted'], a['control_steps'], a['oracle_reads']) == (b['y'], b['steps'], b['acted'], b['control_steps'], b['oracle_reads']), arm
    a, b = cell(rec, 'VISION_CENTER_V2')['control'], cell(ref, 'VISION_CENTER_V2')['control']; assert a['readings'] == b['readings'] and a['steps'] == b['steps'] and a['end_reason'] == b['end_reason'] and rec['memory']['log'] == {**ref['memory']['log']} | rec['memory']['log']
    assert rec['trunk']['steps'] == ref['trunk']['steps'] and rec['trigger']['step'] == ref['trigger']['step']
    v1, v2, sf, nb, nh, no, ro, vs = (cell(rec, x) for x in ('VISION_CENTER_V1', 'VISION_CENTER_V2', 'V2_SINGLE_FRAME', 'V2_NO_BASE_MEMORY', 'V2_NO_HAND_MEMORY', 'V2_NO_HOLD_MEMORY', 'V2_RULES_OFF', C1.SAFETY_ARM))
    src = lambda x: x['control']['readings'][0]['target_src']
    assert not v1['acted'] and v2['acted'] and src(v2).startswith('memory_')                                                     # the target is hidden at the trigger: only a memory gives it
    assert src(nb) != 'memory_base' and src(nh) != 'memory_hand' and (src(nb) is None) == (nb['control']['decision'] == 'NO_TARGET_ESTIMATE') and no['acted'] and ro['acted'] and nh['control']['mode'] == 'V2_NO_HAND_MEMORY'
    assert rec['memory']['variant_logs']['V2_NO_BASE_MEMORY']['base_obs'] == 0 and rec['memory']['variant_logs']['V2_NO_HAND_MEMORY']['hand_obs'] == 0 and rec['memory']['variant_logs']['V2_NO_HOLD_MEMORY']['hold_obs'] == 0
    assert sf['control']['mode'] == 'V2_SINGLE_FRAME' and (sf['control']['readings'][0].get('memory_target') or {}).get('confirmations', 1) == 1
    assert all(cell(rec, x)['oracle_reads'] == 0 for x in C1.VISION_ARMS) and cell(rec, 'CENTER_ORACLE')['oracle_reads'] > 0 and vs['control']['mode'] == 'VISION_CENTER_V2S' and vs['control']['memory_updated_during_control']
    for x in C1.VISION_ARMS:
        c = cell(rec, x)
        if c['acted']: assert c['intervention'] == C1.intervention_of(x) and c['control']['steps_used'] <= O.MAX_STEPS and x in frames
        else: assert c['reproduces_trunk']['match'] and (c['y'], c['steps']) == (cell(rec, 'C0')['y'], cell(rec, 'C0')['steps']) and c['intervention'] is None
    assert len(frames['trunk']) == rec['trigger']['step'] + 1 and set(rec['memory']['variant_logs']) == {'FULL', *C1.MEMORY_VARIANTS, C1.SAFETY_ARM}


def test_the_trunk_does_not_depend_on_which_arms_or_memories_exist():
    a, ta, *_ = cscene((.028, .012)); b, tb, *_ = cscene((.028, .012), arms=('C0',)); c, tc, *_ = cscene((.028, .012), arms=C1.BASE_ARMS)
    assert a['trunk'] == b['trunk'] == c['trunk'] and ta['trunk'] == tb['trunk'] == tc['trunk'] and a['trigger'] == b['trigger'] and cell(a, 'C0')['y'] == cell(b, 'C0')['y']
    assert c['arms'] == list(C1.BASE_ARMS) and set(c['memory']['variant_logs']) == {'FULL', *C1.MEMORY_VARIANTS}
    for arm in C1.BASE_ARMS[1:]: assert (cell(a, arm)['y'], cell(a, arm)['steps'], cell(a, arm)['acted']) == (cell(c, arm)['y'], cell(c, arm)['steps'], cell(c, arm)['acted']), arm       # an arm does not depend on the arms run before it


def test_abstaining_arms_leave_the_policy_untouched():
    rec, *_ = cscene((-.004, .0)); c0 = cell(rec, 'C0')
    for x in C1.VISION_ARMS:
        c = cell(rec, x); assert not c['acted'] and c['control']['decision'] != 'ACT' and c['reproduces_trunk']['match'] and (c['y'], c['steps']) == (c0['y'], c0['steps']) and c['first_inference_step'] == c0['first_inference_step'], x
    env, r, m, n = to_trigger((-.004, .0)); ms = S.MemS()
    q = [x.copy() for x in r.policy.queue]; t = r.t; calls = r.policy.inference_calls; hist = [np.asarray(x).copy() for x in r.tcp_history]; reads = env.reads
    for info in (C1.center_move(r, n.copy(), mode='x', decide_fn=C1.decide_rules_off), S.vision_center_v2s(r, ms)):
        assert not info['acted'] and r.t == t and r.intervention is None and r.policy.inference_calls == calls and env.reads == reads and len(r.policy.queue) == len(q) and all((x == y).all() for x, y in zip(q, r.policy.queue)) and len(r.tcp_history) == len(hist)


def test_validate_scene_rejects_inconsistent_records():
    rec, *_ = cscene((.028, .012)); bad = json.loads(json.dumps(rec)); cell(bad, 'V2_NO_HAND_MEMORY')['oracle_reads'] = 1
    with pytest.raises(ValueError, match='ground truth'): C1.validate_scene(bad, CFG)
    bad = json.loads(json.dumps(rec)); cell(bad, 'V2_RULES_OFF')['control']['steps'][0]['commanded_xy_m'] = [.02, 0]
    with pytest.raises(ValueError, match='bound'): C1.validate_scene(bad, CFG)
    bad = json.loads(json.dumps(rec)); assert not cell(bad, 'VISION_CENTER_V1')['acted']; cell(bad, 'VISION_CENTER_V1')['y'] = 1 - cell(bad, 'C0')['y']
    with pytest.raises(ValueError, match='abstaining'): C1.validate_scene(bad, CFG)
    bad = json.loads(json.dumps(rec)); cell(bad, C1.SAFETY_ARM)['intervention'] = V2.INTERVENTION
    if cell(rec, C1.SAFETY_ARM)['acted']:
        with pytest.raises(ValueError, match='inconsistent'): C1.validate_scene(bad, CFG)
    with pytest.raises(ValueError, match='arms'): C1.collect_scene(None, 1, CFG, arms=('VISION_CENTER_V2', 'C0'))


def test_no_new_code_touches_object_state_and_the_originals_are_untouched():
    for f in (C1.StepObs, C1.reading_from, C1.Mem, C1.single_frame_memory, C1.decide_rules_off, C1.center_move, C1.run_vision_arm, S.MemS, S.decide, S.at_trigger, S.vision_center_v2s):
        src = inspect.getsource(f)
        for word in ('cubeA', 'cubeB', 'get_state', 'is_grasp', 'success', 'oracle_offset', 'probe', 'gate', '.env'): assert word not in src, (f.__name__, word)
    assert set(inspect.signature(C1.Mem.update_obs).parameters) == {'self', 'so'} and set(inspect.signature(C1.StepObs.__init__).parameters) == {'self', 't', 'hand', 'base', 'pose', 'width', 'grip'}
    plan = json.loads(Path('results/vision_center_v2_001/preparation/plan.json').read_text(encoding='utf-8'))['code']; import hashlib
    for f in ('src/hci/vision_center.py', 'src/hci/vision_center_v2.py', 'src/hci/recovery_v1_oracle.py'): assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == plan[f], f
    assert V2.MEMORY['confirm_min'] == 3 and V2.GATE == {'rescue_min': 6, 'net_min': 5, 'harm_max': 2, 'memory_errors_max': 1, 'oracle_recovery_min': 0.5, 'v2_rescue_minus_v1_rescue_min': 2}


# ---------------------------------------------------------------- v2s
def seen(xy, n=3, cls=S.MemS):
    m = cls(); b = render_base([target(*xy)])
    for t in range(n): m.update_obs(so(t, blank(), b, far(), .08, 1.0))
    return m


def holding(m, t, p, hold=(0., 0., 0.)):
    """One step with the gripper closed on a cube seen by the hand camera (t must be a hold-fit step), the target unseen."""
    return m.update_obs(so(t, render(p, [held(p, hold)], finger=True), None, p, .04, -1.0))


def test_v2s_drops_the_target_when_the_held_cube_was_low_beside_it_and_keeps_it_when_clearly_above():
    xy = (.0, .10); m = seen(xy); assert m.target_estimate()[0]['u'] == S.U_BASE_S_M
    high = pose(.03, .10, .085); o = holding(m, 4, high); assert not o['invalidated'] and m.target is not None and m.target['exposed'] == 1      # bottom 25 mm above the top face
    low = pose(.03, .10, .058); o = holding(m.copy(), 8, low); k = m.copy(); o = holding(k, 8, low)
    assert o['invalidated'] and k.target is None and k.log['invalidations'][-1]['reason'] == 'held_cube_low_near_target' and k.target_estimate() == (None, 'invalidated_by_contact')
    away = pose(.15, .10, .058); k = m.copy(); assert not holding(k, 8, away)['invalidated']                                                       # low, but too far to touch
    v = remembered(xy); v.update(4, render(low, [held(low)], finger=True), None, low, .04, -1.0); assert v.target is not None                       # v2 keeps the position and only widens the bound
    b = render_base([target(*xy)]); k = m.copy(); holding(k, 8, low); assert k.target is None
    for t in (9, 10, 11): k.update_obs(so(t, blank(), b, far(), .04, -1.0))
    assert k.target_estimate()[1] is None and k.target['n'] == 3                                                                                   # seen again three times: usable again


def test_v2s_drops_the_target_when_contact_cannot_be_excluded():
    xy = (.0, .10); m = seen(xy); near = pose(.02, .10, .065); k = m.copy(); o = k.update_obs(so(5, blank(), None, near, .04, -1.0))           # closed on something, no held-cube estimate, near and low
    assert o['invalidated'] and k.log['invalidations'][-1]['reason'] == 'held_cube_unknown_near_target'
    k = m.copy(); assert not k.update_obs(so(5, blank(), None, pose(.02, .10, .20), .04, -1.0))['invalidated']                                    # high above
    k = m.copy(); o = k.update_obs(so(5, blank(), None, pose(.03, .10, .045), .08, 1.0)); assert o['invalidated'] and k.log['invalidations'][-1]['reason'] == 'gripper_low_near_target'     # open gripper, fingers at the height of the target
    k = m.copy(); assert not k.update_obs(so(5, blank(), None, pose(.03, .10, .12), .08, 1.0))['invalidated']
    k = m.copy(); b = render_base([target(*xy)]); assert not k.update_obs(so(5, blank(), b, near, .04, -1.0))['invalidated'] and k.target['n'] == 4   # a step that sees the target is not a contact step


def test_v2s_history_needs_three_views_before_the_trigger_step():
    xy = (.0, .10); b = render_base([target(*xy)])
    m = seen(xy, n=2); m.update_obs(so(2, blank(), b, far(), .08, 1.0)); m.freeze_history(2); assert m.target['n'] == 3 and m.target_estimate() == (None, 'not_confirmed_before_trigger')     # the third view is the trigger step itself
    v = remembered(xy, n=3); assert v.target_estimate()[1] is None                                                                               # v2 counts it
    m = seen(xy, n=3); m.update_obs(so(3, blank(), b, far(), .08, 1.0)); m.freeze_history(3); e, why = m.target_estimate(); assert why is None and e['confirmations_before_trigger'] == 3 and e['t_last'] == 3
    m = seen(xy, n=3); m.update_obs(so(3, blank(), None, far(), .08, 1.0)); m.freeze_history(3); assert m.target_estimate()[0]['confirmations_before_trigger'] == 3
    m.update_obs(so(4, blank(), render_base([target(.05, .15)]), far(), .08, 1.0)); assert m.target_estimate() == (None, 'not_confirmed_before_trigger') and m.log['conflicts'] == 1          # a conflict after the trigger: nothing usable
    for t in (5, 6, 7): m.update_obs(so(t, blank(), render_base([target(.05, .15)]), far(), .08, 1.0))
    assert m.target['n'] == 4 and m.target_estimate()[0] is None                                                                                  # a track started after the trigger never becomes history
    with pytest.raises(ValueError, match='once'): m.freeze_history(8)
    c = m.copy(); assert c.t_trigger == 3 and c.log['invalidations'] == m.log['invalidations'] and c.log['invalidations'] is not m.log['invalidations']


def test_v2s_decision_margin_and_label():
    est = lambda n, u, gap, src='frame': {'why': None, 'offset_norm_m': n, 'u': u, 'gap_m': gap, 'target_src': src, 'held_xyz': np.array([0., 0., .06 + gap])}
    d = lambda *a: S.decide(est(*a), 100)['reason']; v = lambda *a: V2.decide(est(*a), 100)['reason']
    for a in ((.016, .004, .005), (.014, .004, .005), (.0409, .004, -.0002), (.030, .004, -.004), (.030, .004, -.001), (.038, .004, .0006)): assert d(*a) == v(*a), a      # a target seen in the current frame: the v2 ladder
    assert v(.030, .007, .001, 'memory_base') == 'ACT' and d(.030, .007, .001, 'memory_base') == 'LOW_CARRY' and d(.030, .007, .002, 'memory_base') == 'ACT'
    assert v(.012, .007, .005, 'memory_base') == 'ALIGNED' and d(.012, .007, .005, 'memory_base') == 'ALIGNED_ON_MEMORY' and d(.016, .0085, .005, 'memory_base') == 'ALIGNED_WITHIN_UNCERTAINTY'
    assert S.Z_MARGIN_M == V2.U_HOLD_M and S.U_BASE_S_M > V2.U_BASE_M and abs(S.CUBE_REACH_M - 0.0566) < 1e-4 and 'ALIGNED_ON_MEMORY' in S.ABSTAIN


def test_v2s_stops_a_move_when_the_memory_is_dropped_during_control():
    class R:                                                    # the smallest runtime center_move needs: a TCP that follows the command, a scripted observation per step
        horizon = 200; adim = 4; done = False; intervention = None; intervention_t = None; retreat_info = None
        def __init__(s, frames): s.t = 60; s.frames = frames; s.i = 0; s.tcp_history = [frames[0][2]]; s.policy = type('P', (), {'last_action': np.array([0, 0, 0, -1.0]), 'clear': lambda self: None, 'queue': []})()
        def advance(s, a): s.i += 1; s.t += 1; s.tcp_history.append(s.frames[min(s.i, len(s.frames) - 1)][2])
    xy = HID_T; b = render_base([target(*xy)]); m = seen(xy, n=4); steps = []
    for i in range(6):
        p = pose(HID_P[0] - .004 * i, HID_P[1], .075 if i < 3 else .058); steps.append((render(p, [target(*xy), held(p)]), None, p))               # from the fourth frame on the held cube is at the height of the target
    r = R(steps); orig = (V2.observe, O.held_grip, O.arm_scale)
    V2.observe = lambda rr: (rr.frames[min(rr.i, len(rr.frames) - 1)][0], rr.frames[min(rr.i, len(rr.frames) - 1)][1], rr.frames[min(rr.i, len(rr.frames) - 1)][2], .04, -1.0); O.held_grip = lambda rr: -1.0; O.arm_scale = lambda rr: np.ones(3) * .1
    try: info = S.vision_center_v2s(r, m)
    finally: V2.observe, O.held_grip, O.arm_scale = orig
    assert info['acted'] and info['readings'][0]['target_src'] == 'memory_base' and info['end_reason'] == 'estimate_lost' and info['steps_used'] == 3 and info['memory_log_after']['invalidations'][-1]['reason'] == 'held_cube_low_near_target'
    assert info['readings'][-1]['why'] == 'NO_TARGET_ESTIMATE' and info['readings'][-1]['memory_reason'] == 'invalidated_by_contact' and r.intervention == S.INTERVENTION


# ---------------------------------------------------------------- statistics
def test_exact_one_sided_paired_test():
    assert C1.exact_one_sided(0, 0) == 1.0 and C1.exact_one_sided(5, 0) == 1 / 32 and C1.exact_one_sided(11, 0) == 1 / 2048 and abs(C1.exact_one_sided(8, 2) - 56 / 1024) < 1e-12 and C1.exact_one_sided(0, 5) == 1.0 and C1.exact_one_sided(3, 3) == 42 / 64
    assert C1.holm([.03, .01]) == [True, True] and C1.holm([.03, .04]) == [False, False] and C1.holm([.02, .06]) == [True, False] and C1.holm([.026, .03]) == [False, False] and C1.ALPHA == .05
    assert C1.discordant({1: 0, 2: 1, 3: 0}, {1: 1, 2: 0, 3: 0}) == (1, 1)
    with pytest.raises(ValueError, match='same scenes'): C1.discordant({1: 0}, {2: 0})


def ys(n, c0, v2, sf):
    """Outcome maps over scenes 0..n-1: c0 / v2 / single-frame succeed on the given index sets."""
    f = lambda s: {i: int(i in s) for i in range(n)}; return {'C0': f(c0), 'VISION_CENTER_V2': f(v2), 'V2_SINGLE_FRAME': f(sf)}


def test_primary_tests_use_the_larger_p_for_both_policies_and_holm():
    base = set(range(20)); good = ys(40, base, base | set(range(20, 32)), base | {20})                       # v2 only: 12 against C0, 11 against single frame
    p = C1.primary_tests({'train_seed_1': good, 'train_seed_2': good}); assert p['H_gen']['v2_only'] == 12 and p['H_gen']['p_one_sided'] == 1 / 4096 and p['H_mem']['p_intersection_union'] == 1 / 2048 and p['H_gen']['rejected_null'] and p['H_mem']['rejected_null']
    weak = ys(40, base, base | set(range(20, 32)), base | set(range(20, 30)))                               # with one policy memory adds only 2 scenes: p = 0.25
    p = C1.primary_tests({'train_seed_1': good, 'train_seed_2': weak}); assert p['H_mem']['p_intersection_union'] == .25 and not p['H_mem']['rejected_null'] and p['H_gen']['rejected_null']
    p = C1.primary_tests({'train_seed_1': weak, 'train_seed_2': good}); assert p['H_mem']['p_intersection_union'] == .25 and not p['H_mem']['rejected_null']
    none = ys(40, base, base, base); p = C1.primary_tests({'train_seed_1': good, 'train_seed_2': none}); assert p['H_gen']['p_one_sided'] == 1.0 and not p['H_gen']['rejected_null'] and not p['H_mem']['rejected_null']
    assert 'intersection-union' in p['method'] and p['alpha_familywise'] == .05


def fake(scene, y, arms=C1.ARMS, acted=(), err=.002, pushed=0., mem=True):
    cells = []
    for a in arms:
        x = {'arm': a, 'y': y.get(a, y['C0']), 'no_trigger': False}
        if a in C1.V2_LIKE:
            act = a in acted; x.update(acted=act, control={'decision': 'ACT' if act else 'ALIGNED', 'end_reason': 'within_tolerance', 'used_memory': mem, 'readings': [{'target_src': 'memory_base' if mem else 'frame', 'hold_src': 'frame'}]},
                                       gt_reading_error={'offset_error_m': err, 'error_within_bound': err <= .007, 'unnecessary_act': False}, gt_effect_of_control={'cubeB_xy_displacement_m': pushed})
        cells.append(x)
    return {'scene': scene, 'triggered': True, 'arms': list(arms), 'cells': cells}


def test_summarize_reports_each_policy_separately_and_needs_every_scene():
    v = ('VISION_CENTER_V2', 'V2_NO_HOLD_MEMORY', 'V2_RULES_OFF', C1.SAFETY_ARM, 'CENTER_ORACLE', 'V2_NO_HAND_MEMORY')
    def recs(k): return [fake(i, {'C0': 0, **{a: 1 for a in v}}, acted=v) for i in range(k)] + [fake(100 + i, {'C0': 1}) for i in range(10)] + [fake(200 + i, {'C0': 0}) for i in range(4)]
    a, b = recs(12), recs(7); a[0]['cells'][3]['gt_effect_of_control']['cubeB_xy_displacement_m'] = .004; planned_a = [r['scene'] for r in a]
    with pytest.raises(ValueError, match='planned'): C1.summarize({'train_seed_1': a, 'train_seed_2': b}, planned_a, C1.ARMS)
    b = recs(12); b[1]['cells'][3]['y'] = 0; s = C1.summarize({'train_seed_1': a, 'train_seed_2': b}, planned_a, C1.ARMS)
    p1, p2 = s['per_policy']['train_seed_1'], s['per_policy']['train_seed_2']; assert p1['success']['C0'] == {'k': 10, 'n': 26, 'wilson95': p1['success']['C0']['wilson95']} and p1['success']['VISION_CENTER_V2']['k'] == 22 and p2['success']['VISION_CENTER_V2']['k'] == 21
    assert p1['against_C0']['VISION_CENTER_V2']['rescue'] == 12 and p1['paired']['VISION_CENTER_V2_against_V2_SINGLE_FRAME']['rescue'] == 12 and p1['oracle_rescue_recovery']['VISION_CENTER_V2'] == {'recovered': 12, 'oracle': 12}
    assert p1['safety']['VISION_CENTER_V2']['met'] and p1['safety']['VISION_CENTER_V2']['target_pushed_over_1mm'] == 1 and set(p1['safety']) == {'VISION_CENTER_V2', C1.SAFETY_ARM} and s['safety_all_met']
    assert s['primary']['H_gen']['v2_only'] == 11 and s['primary']['H_mem']['per_policy']['train_seed_2']['v2_only'] == 11 and s['primary']['H_mem']['rejected_null'] and s['between_policies']['v2_net_gain'] == {'train_seed_1': 12, 'train_seed_2': 11}
    assert s['scenes'] == 26 and 'never' not in s['unit'] and '400' not in json.dumps(s['unit'])
    with pytest.raises(ValueError, match='both policies'): C1.summarize({'train_seed_1': a}, planned_a, C1.ARMS)
    bad = recs(12)
    for i in range(3): bad[i]['cells'][3]['gt_reading_error'] = {'offset_error_m': .009, 'error_within_bound': False, 'unnecessary_act': False}
    s = C1.summarize({'train_seed_1': a, 'train_seed_2': bad}, planned_a, C1.ARMS); assert not s['per_policy']['train_seed_2']['safety']['VISION_CENTER_V2']['criteria']['memory_errors_at_most'] and not s['safety_all_met'] and C1.SAFETY == {'harm_max': 2, 'memory_errors_max': 2, 'target_pushed_over_1mm_max': 1}
    nine = [fake(i, {'C0': 1}, arms=C1.BASE_ARMS) for i in range(5)]; s9 = C1.summarize({'train_seed_1': nine, 'train_seed_2': nine}, range(5), C1.BASE_ARMS); assert set(s9['per_policy']['train_seed_1']['safety']) == {'VISION_CENTER_V2'}


def test_projected_cost_and_seed_set():
    kw = dict(seconds_per_env_step=.013, seconds_per_reading=.22, seconds_per_memory_step=.07, usd_per_hour=.46, spent_usd=.3, scene_runs=400, arms=10, vision_arms=8)
    a = C1.projected_cost(cap_usd=6., **kw); assert a['fits_cap'] and not C1.projected_cost(cap_usd=2., **kw)['fits_cap'] and a['inputs']['steps_per_scene_bound'] == 2200
    assert C1.projected_cost(cap_usd=6., **{**kw, 'arms': 9, 'vision_arms': 7})['projected_total_usd'] < a['projected_total_usd']
    man = json.loads(Path('configs/vision_center_c1_splits.json').read_text(encoding='utf-8')); assert C1.pilot_seeds(man, {'x': [51099, 51300]}) == list(range(51100, 51300))
    with pytest.raises(ValueError, match='recorded use'): C1.pilot_seeds(man, {'x': [51150]})
    locked = json.loads(Path('configs/splits.json').read_text(encoding='utf-8'))['splits']; assert not set(man['splits'][C1.SET]) & (set(locked['calibration']) | set(locked['test']))
    assert C1.ARMS == C1.BASE_ARMS + ('VISION_CENTER_V2S',) and len(C1.BASE_ARMS) == 9 and C1.POLICIES == ('train_seed_1', 'train_seed_2')


# ---------------------------------------------------------------- driver
@pytest.fixture(scope='module')
def X34():
    spec = importlib.util.spec_from_file_location('c1_driver', Path('scripts/34_vision_center_c1.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def ns(plan, run, **k): return argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu', **k)


def test_every_server_command_refuses_without_an_explicit_authorization(X34, tmp_path):
    plan = tmp_path / 'plan.json'; plan.write_text('{"code": {}}'); run = tmp_path / 'run'; run.mkdir(); had_ledger = Path(X34.LEDGER).exists(); cmds = (X34.cmd_clock, X34.cmd_scenes, X34.cmd_audit, X34.cmd_budget, X34.cmd_run)
    for cmd in cmds:
        with pytest.raises(SystemExit, match='no authorization'): cmd(ns(plan, run, usd_per_hour=.4, spent_before_usd=0.))
    (run / 'authorization.json').write_text(json.dumps({'approved_by': 'agent', 'paid_compute_authorized': True, 'cap_usd': 2.5, 'plan_sha256': X34.sha256(plan)}))
    for cmd in cmds:
        with pytest.raises(SystemExit, match='Refused'): cmd(ns(plan, run, usd_per_hour=.4, spent_before_usd=0.))
    (run / 'authorization.json').write_text(json.dumps({'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 2.5, 'plan_sha256': X34.sha256(plan)}))
    for cmd in cmds[1:]:
        with pytest.raises(SystemExit, match='Refused'): cmd(ns(plan, run))                                # a plan that did not pass its own checks
    assert not [p for p in run.iterdir() if p.is_dir()] and Path(X34.LEDGER).exists() == had_ledger and not (run / 'session_clock.json').exists()


def test_session_clock_is_written_once_and_the_spend_is_computed_from_it(X34, tmp_path):
    plan = tmp_path / 'plan.json'; plan.write_text('{"code": {}}'); run = tmp_path / 'run'; run.mkdir()
    (run / 'authorization.json').write_text(json.dumps({'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 8.0, 'plan_sha256': X34.sha256(plan)}))
    with pytest.raises(SystemExit, match='positive'): X34.cmd_clock(ns(plan, run, usd_per_hour=0., spent_before_usd=0.))
    with pytest.raises(SystemExit, match='clock'): X34.spent_now(run, plan)
    X34.cmd_clock(ns(plan, run, usd_per_hour=.5, spent_before_usd=.25)); k = json.loads((run / 'session_clock.json').read_text())
    with pytest.raises(SystemExit, match='once'): X34.cmd_clock(ns(plan, run, usd_per_hour=.01, spent_before_usd=0.))
    spent, _ = X34.spent_now(run, plan, now=k['started_epoch'] + 7200); assert abs(spent - 1.25) < 1e-9
    with pytest.raises(SystemExit, match='future'): X34.spent_now(run, plan, now=k['started_epoch'] - 10)
    (run / 'audit').mkdir(); (run / 'audit' / 'audit.json').write_text(json.dumps({'pass': True, 'plan_sha256': X34.sha256(plan), 'seconds_per_env_step': .013, 'seconds_per_reading': .22, 'seconds_per_memory_step': .07}))
    auth = json.loads((run / 'authorization.json').read_text()); arms = C1.ARMS
    early = X34.projection(run, auth, plan, arms, 400, now=k['started_epoch'] + 600); late = X34.projection(run, auth, plan, arms, 400, now=k['started_epoch'] + 10 * 3600)
    assert early['fits_cap'] and not late['fits_cap'] and late['spent_from_session_clock_usd'] == pytest.approx(5.25) and late['status'] == 'BUDGET_BLOCKED'
    other = tmp_path / 'other.json'; other.write_text('{"code": {}, "x": 1}')
    with pytest.raises(SystemExit, match='another plan'): X34.spent_now(run, other)
    assert X34.one_scene_bound_usd(run, arms, k) == pytest.approx(((200 + 10 * 200) * .013 + 200 * .07 + 8 * 13 * .22) * 1.25 / 3600 * .5)


def test_driver_constants_registry_and_file_list(X34):
    reg, arms = X34.registry(); assert arms[:9] == C1.BASE_ARMS and set(arms) <= set(C1.ARMS) and [a['id'] for a in reg['arms']] == list(C1.ARMS)
    assert all(Path(f).exists() for f in X34.CODE) and {'src/hci/vision_center_c1.py', 'src/hci/vision_center_v2s.py', 'scripts/34_vision_center_c1.py', X34.MANIFEST, X34.REGISTRY} <= set(X34.CODE)
    assert all(p['path'] + '/weights.pt' in X34.SERVER_READS for p in X34.POLICIES.values()) and list(X34.POLICIES) == list(C1.POLICIES) and X34.POLICIES['train_seed_1']['weights.pt'] != X34.POLICIES['train_seed_2']['weights.pt']
    src = Path('scripts/34_vision_center_c1.py').read_text(encoding='utf-8'); assert 'authorization.json' in src and "write_json(run / 'authorization" not in src and "open(LEDGER, 'x'" in src and 'RUN_FROZEN_TEST' not in src
    used = X34.used_ledger(); assert set(range(50900, 51100)) <= {int(s) for v in used.values() for s in v} and not {int(s) for v in used.values() for s in v} & set(range(51100, 51300))
    assert not Path(X34.LEDGER).exists()                                                                   # nothing has been run
    problems = []; f = X34.check_policy_2(problems); assert not problems and f['select_success_once'] == '83/100' and not f['selected_in_vision_v1']
    a = X34.analysis(arms); assert a['primary']['alpha_familywise'] == .05 and a['safety_per_policy']['thresholds'] == C1.SAFETY and 'intersection-union' in a['primary']['H_mem']
    sh = Path('scripts/vision_center_c1/run.sh').read_text(encoding='utf-8'); assert 'clock' in sh and '|| true' not in sh and 'authorization.json' in sh


# ---------------------------------------------------------------- review fixes (Codex review 1)
def test_arm_view_exposes_only_the_declared_inputs_and_changes_nothing():
    env, r, m, n = to_trigger((.028, .012)); view = C1.ArmView(r)
    for bad in (lambda: view.y, lambda: view.initial_state, lambda: view.reason, lambda: view.snapshot, lambda: view.env.get_state_dict, lambda: view.env.cubeA,
                lambda: view.raw['extra'].keys, lambda: view.raw['sensor_data']['hand_camera']['depth'], lambda: view.policy.queue, lambda: view.policy.model, lambda: view.env.agent.robot.qpos):
        with pytest.raises(C1.PrivilegedAccess): bad()
    for bad in ('done', 'y', 't', 'raw'):
        with pytest.raises(C1.PrivilegedAccess): setattr(view, bad, 1)
    with pytest.raises(C1.PrivilegedAccess): view.policy.queue = []
    assert {'y', 'env.cubeA', 'policy.queue'} <= view.reads                                                   # refused attempts are recorded too
    view = C1.ArmView(r); hand, base, p, w, g = V2.observe(view); h2, b2, p2, w2, g2 = V2.observe(r); assert (hand == h2).all() and (base == b2).all() and (p == p2).all() and (w, g) == (w2, g2) and O.arm_scale(view).tolist() == O.arm_scale(r).tolist()
    assert view.reads <= C1.ARM_READS | {'raw.' + '.'.join(x) for x in C1.RAW_PATHS} | {'.'.join(x[:i]) for x in C1.RAW_PATHS | C1.ENV_PATHS for i in range(1, len(x) + 1)} | {'raw.' + '.'.join(x[:i]) for x in C1.RAW_PATHS for i in range(1, len(x) + 1)} | {'policy.' + k for k in C1.POLICY_READS}


@pytest.mark.parametrize('bias', [(.028, .012), (-.004, .0)])
def test_v2_through_the_arm_view_equals_v2_on_the_runtime(bias):
    _, r1, m1, _ = to_trigger(bias); _, r2, m2, _ = to_trigger(bias)
    a = V2.vision_center_v2(C1.ArmView(r1), m1.copy()); b = V2.vision_center_v2(r2, m2.copy())
    assert json.dumps(O.jsonable(a), sort_keys=True) == json.dumps(O.jsonable(b), sort_keys=True) and r1.t == r2.t and np.allclose(np.asarray(r1.tcp_history), np.asarray(r2.tcp_history)) and r1.intervention == r2.intervention


def test_every_vision_arm_runs_through_the_view_and_records_what_it_read():
    rec, *_ = cscene((.028, .012)); allowed = C1.ARM_READS | {'policy.' + k for k in C1.POLICY_READS}
    for x in C1.VISION_ARMS:
        reads = set(cell(rec, x)['control']['runtime_reads']); top = {s for s in reads if '.' not in s}
        assert reads and top <= C1.ARM_READS and 'y' not in reads and 'initial_state' not in reads, x
    assert 'runtime_reads' not in (cell(rec, 'CENTER_ORACLE')['control'] or {})


def test_budget_deadline_is_a_hard_stop_that_no_scene_handler_swallows(X34, tmp_path):
    assert not issubclass(X34.BudgetDeadline, Exception)
    k = {'started_epoch': 1000., 'usd_per_hour': .5, 'spent_before_usd': .25}
    assert X34.deadline_seconds_left(k, 2.0, now=1000.) == pytest.approx((2.0 - X34.RESERVE_USD - .25) / .5 * 3600)
    plan = tmp_path / 'plan.json'; plan.write_text('{"code": {}}'); run = tmp_path / 'run'; run.mkdir(); out = run / 'pass'; (out / 'scenes').mkdir(parents=True)
    (run / 'session_clock.json').write_text(json.dumps({'started_epoch': time.time(), 'usd_per_hour': .5, 'spent_before_usd': 0., 'plan_sha256': X34.sha256(plan)}))
    (run / 'audit').mkdir(); (run / 'audit' / 'audit.json').write_text(json.dumps({'pass': True, 'plan_sha256': X34.sha256(plan), 'seconds_per_env_step': .013, 'seconds_per_reading': .22, 'seconds_per_memory_step': .07}))
    def boom(*a, **k): raise X34.BudgetDeadline()
    old = X34.C1.collect_scene; X34.C1.collect_scene = boom
    try:
        with pytest.raises(X34.BudgetDeadline): X34.passes(None, 'train_seed_1', [51100, 51101], CFG, out, C1.BASE_ARMS, run, plan, 6.0, 0, 'h')
    finally: X34.C1.collect_scene = old
    assert not list((out / 'scenes').glob('*')) and X34.CURRENT['scene'] == 51100
    import signal
    if hasattr(signal, 'setitimer'):
        with pytest.raises(X34.BudgetDeadline):
            X34.arm_deadline({'started_epoch': time.time(), 'usd_per_hour': 3600 * 10., 'spent_before_usd': 0.}, X34.RESERVE_USD + .5); time.sleep(1.)
        X34.disarm_deadline()
    else:
        with pytest.raises(SystemExit, match='setitimer'): X34.arm_deadline(k, 6.0)
    src = Path('scripts/34_vision_center_c1.py').read_text(encoding='utf-8'); assert 'except BudgetDeadline' in src and 'arm_deadline(k, cap)' in src


def test_summarize_and_pack_verify_the_plan_and_the_provenance_of_every_record(X34, tmp_path):
    plan = tmp_path / 'plan.json'; plan.write_text('{"code": {}}'); run = tmp_path / 'run'; run.mkdir()
    for cmd in (X34.cmd_summarize, X34.cmd_check_auth):
        with pytest.raises(SystemExit, match='no authorization'): cmd(ns(plan, run, out=str(tmp_path / 'r.json')))
    (run / 'authorization.json').write_text(json.dumps({'approved_by': 'user', 'paid_compute_authorized': True, 'cap_usd': 6.0, 'plan_sha256': X34.sha256(plan)}))
    X34.cmd_check_auth(ns(plan, run))
    with pytest.raises(SystemExit, match='Refused'): X34.cmd_summarize(ns(plan, run, out=str(tmp_path / 'r.json')))           # a plan that did not pass its own checks
    assert not (tmp_path / 'r.json').exists()
    rec, traces, frames, _ = cscene((.028, .012)); out = tmp_path / 'pass_train_seed_1'; X34.Z.save_scene(out, 7, rec, traces, frames)
    assert len(X34.load_records(out, [7], CFG, C1.ARMS, 'train_seed_1')) == 1
    with pytest.raises(SystemExit, match='not a completed'): X34.load_records(out, [7], CFG, C1.ARMS, 'train_seed_2')
    p = out / 'scenes' / '7.json'; d = json.loads(p.read_text()); d['completed'] = False; p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match='not a completed'): X34.load_records(out, [7], CFG, C1.ARMS, 'train_seed_1')
    sh = Path('scripts/vision_center_c1/run.sh').read_text(encoding='utf-8'); assert 'check-auth --plan $PLAN --run $R' in sh and sh.index('check-auth') < sh.index('case "$1"')


# ---------------------------------------------------------------- review fixes (Codex review 2)
def test_no_vision_arm_code_can_step_around_the_view():
    """The view guards against accidental reads; Python cannot stop deliberate introspection. This checks that the arm code has none."""
    import ast
    lines = Path('src/hci/vision_center_c1.py').read_text(encoding='utf-8').splitlines(); a = next(i for i, l in enumerate(lines) if l.startswith('class PrivilegedAccess')); b = next(i for i, l in enumerate(lines) if l.startswith('def run_vision_arm'))
    sources = {'c1 outside the view': '\n'.join(lines[:a] + lines[b:]), **{f: Path(f).read_text(encoding='utf-8') for f in ('src/hci/vision_center.py', 'src/hci/vision_center_v2.py', 'src/hci/vision_center_v2s.py')}}
    for name, src in sources.items():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute): assert not (node.attr.startswith('_') and node.attr not in ('__init__',)) or not isinstance(node.value, ast.Name) or node.value.id not in ('r', 'view'), (name, node.attr)
            if isinstance(node, ast.Attribute): assert node.attr not in ('__self__', '__closure__', '__dict__', '__globals__', '_r', '_o'), (name, node.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name): assert node.func.id not in ('getattr', 'vars', 'globals', 'eval', 'exec', '__import__'), (name, node.func.id)


@pytest.mark.parametrize('bias', [(.028, .012), (-.028, .0), (-.004, .0)])
def test_every_arm_function_through_the_view_equals_the_runtime(bias):
    runs = {'v1': lambda r, m, n: V1.vision_center(r), 'rules_off': lambda r, m, n: C1.center_move(r, n.copy(), mode='x', decide_fn=C1.decide_rules_off),
            'v2s': lambda r, m, n: S.vision_center_v2s(r, S.MemS())}
    for name, f in runs.items():
        _, r1, m1, n1 = to_trigger(bias); _, r2, m2, n2 = to_trigger(bias)
        a = f(C1.ArmView(r1), m1, n1); b = f(r2, m2, n2)
        assert json.dumps(O.jsonable(a), sort_keys=True) == json.dumps(O.jsonable(b), sort_keys=True) and r1.t == r2.t and r1.intervention == r2.intervention and len(r1.policy.queue) == len(r2.policy.queue), name


def test_a_deadline_stop_writes_the_marker_and_the_run_is_not_summarised(X34, tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'; plan.write_text('{"code": {}}'); run = tmp_path / 'run'; run.mkdir(); h = X34.sha256(plan)
    (run / 'session_clock.json').write_text(json.dumps({'started_epoch': time.time(), 'usd_per_hour': .5, 'spent_before_usd': 0., 'plan_sha256': h}))
    class R:
        class env:
            closed = 0
            @classmethod
            def close(cls): cls.closed += 1
    def stop(*a, **k): X34.CURRENT.update(policy='train_seed_1', scene=51103, scenes_finished_in_pass=3); raise X34.BudgetDeadline()
    monkeypatch.setattr(X34, 'arm_deadline', lambda k, cap: 100.); monkeypatch.setattr(X34, 'runtime_for', lambda c, pid, a: R()); monkeypatch.setattr(X34, 'passes', stop)
    outs = {pid: run / f'pass_{pid}' for pid in C1.POLICIES}
    with pytest.raises(SystemExit, match='BUDGET_STOP'): X34.run_passes(run, argparse.Namespace(plan=str(plan)), CFG, [51100], C1.BASE_ARMS, 6.0, h, outs)
    m = json.loads((run / 'budget_stop.json').read_text()); assert m['kind'] == 'deadline' and m['scene'] == 51103 and m['plan_sha256'] == h and R.env.closed == 1 and not outs['train_seed_2'].exists()
    with pytest.raises(SystemExit, match='budget stop'): X34.check_run_gates(run, {}, h, C1.BASE_ARMS)


def test_summary_gates_refuse_failed_steps_and_a_foreign_ledger(X34, tmp_path):
    h = 'p' * 64; arms = list(C1.BASE_ARMS); run = tmp_path / 'run'; (run / 'audit').mkdir(parents=True)
    plan = {'arm_registry': {'sha256': 'a'}, 'seed_registry': {'sha256': 's'}}
    good = {'scenes.json': {'pass': True}, 'audit/audit.json': {'pass': True, 'arms': arms}, 'budget_gate.json': {'fits_cap': True}, 'budget_gate_at_start.json': {'fits_cap': True},
            'ledger.json': {'set': C1.SET, 'arms': arms, 'policies': {pid: X34.POLICIES[pid]['weights.pt'] for pid in C1.POLICIES}, 'arm_registry_sha256': 'a', 'seed_manifest_sha256': 's'}}
    def write(d):
        for n, v in d.items(): (run / n).write_text(json.dumps({**v, 'plan_sha256': h}))
    write(good); X34.check_run_gates(run, plan, h, arms)
    for n, k, v in (('scenes.json', 'pass', False), ('audit/audit.json', 'pass', False), ('audit/audit.json', 'arms', arms[:-1]), ('budget_gate.json', 'fits_cap', False), ('budget_gate_at_start.json', 'fits_cap', False)):
        write({**good, n: {**good[n], k: v}})
        with pytest.raises(SystemExit, match='did not pass'): X34.check_run_gates(run, plan, h, arms)
    write({**good, 'ledger.json': {**good['ledger.json'], 'arms': arms + [C1.SAFETY_ARM]}})
    with pytest.raises(SystemExit, match='ledger'): X34.check_run_gates(run, plan, h, arms)
    write(good); (run / 'ledger.json').unlink()
    with pytest.raises(SystemExit, match='ledger.json missing'): X34.check_run_gates(run, plan, h, arms)
    write(good); (run / 'scenes.json').write_text(json.dumps({'pass': True, 'plan_sha256': 'q' * 64}))
    with pytest.raises(SystemExit, match='another plan'): X34.check_run_gates(run, plan, h, arms)


def test_records_are_bound_to_the_plan(X34, tmp_path):
    rec, traces, frames, _ = cscene((.028, .012)); rec['plan_sha256'] = 'p' * 64; out = tmp_path / 'pass_train_seed_1'; X34.Z.save_scene(out, 7, rec, traces, frames)
    assert len(X34.load_records(out, [7], CFG, C1.ARMS, 'train_seed_1', 'p' * 64)) == 1
    with pytest.raises(SystemExit, match='not a completed'): X34.load_records(out, [7], CFG, C1.ARMS, 'train_seed_1', 'q' * 64)
    src = Path('scripts/34_vision_center_c1.py').read_text(encoding='utf-8'); assert "rec['plan_sha256'] = plan_hash; Z.save_scene" in src and "load_records(run / f'pass_{pid}', seeds, c, arms, pid, h)" in src
