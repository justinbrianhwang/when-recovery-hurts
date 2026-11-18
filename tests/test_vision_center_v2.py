"""Temporal Vision-CENTER v2: base-camera target estimate, memory, uncertainty-aware decision, safety rules, controller, accounting.

Mock physics and a synthetic renderer (flat colours, the module's own camera constants). These tests say nothing about
ManiSkill, the GPU or the research policy."""
import argparse, importlib.util, inspect, json
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from test_recovery_v1_oracle import CFG, probe
from test_vision_center import VisionEnv, BiasPolicy, render, pose, held, target, DOWN, TABLE
from hci.simulation import Runtime
from hci import vision_center as V1
from hci import vision_center_v2 as V2
from hci import recovery_v1_oracle as O


def render_base(cubes, stray=()):
    """Base-camera image of cubes [(centre xyz, yaw, colour index)]; the top face is lit. `stray`: extra lit-green pixels (row, col)."""
    cam = V1.cam_base(); img = np.empty((V1.IMG, V1.IMG, 3), np.uint8); img[:] = TABLE; h = V1.CUBE_HALF_M
    for c, yaw, col in sorted(cubes, key=lambda x: -np.linalg.norm(np.asarray(x[0]) - cam[1])):
        cs, sn = np.cos(yaw), np.sin(yaw); P = np.array([V1.project(cam, [c[0] + h * (a * cs - b * sn), c[1] + h * (a * sn + b * cs), c[2] + dz])[0] for dz in (h, -h) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        sil = np.zeros((V1.IMG, V1.IMG), bool)
        for f in V1._FACES: sil |= V1.raster(P[list(f)])
        dark = np.zeros(3, np.uint8); dark[col] = 147; lit = np.zeros(3, np.uint8); lit[col] = 230; img[sil] = dark; img[V1.raster(P[:4])] = lit
    for v, u in stray: img[v, u] = (0, 230, 0)
    return img


# ---------------------------------------------------------------- base camera
@pytest.mark.parametrize('x,y,yaw', [(.0, .10, 0.), (-.05, .15, .4), (.08, -.12, -.3), (-.02, .0, .7)])
def test_base_estimate_of_an_unoccluded_target(x, y, yaw):
    r = V2.target_base(render_base([target(x, y, yaw)])); assert r['valid'] and np.linalg.norm(r['xy'] - [x, y]) < V2.U_BASE_M


def test_stray_green_pixels_do_not_move_the_base_estimate_but_move_the_v1_centroid():
    clean = V2.target_base(render_base([target(.0, .10)])); img = render_base([target(.0, .10)], stray=[(20, 20), (20, 21), (21, 20)]); r = V2.target_base(img)
    assert r['valid'] and np.allclose(r['xy'], clean['xy']); old = V1.target_full_view(img, V1.cam_base(), 'base_camera'); assert np.linalg.norm(old['xy'] - clean['xy']) > 0.02


def test_base_estimate_is_refused_when_part_of_the_face_is_covered_or_nothing_is_seen():
    img = render_base([target(.0, .10)]); v, u = np.nonzero(V1.classes(img)['green_lit']); img[v[:len(v) // 3], u[:len(v) // 3]] = 128
    assert not V2.target_base(img)['valid'] and V2.target_base(render_base([])) is None


# ---------------------------------------------------------------- memory
def far(): return pose(.3, -.3, .2)                    # a TCP pose from which the hand camera sees no cube


def blank(): return render(far(), [])


def test_memory_needs_three_consistent_observations_and_restarts_on_a_conflict():
    m = V2.Memory(); b = render_base([target(.0, .10)])
    for t in range(2): m.update(t, blank(), b, far(), .08, 1.0)
    assert m.target_estimate() == (None, 'not_confirmed')
    m.update(2, blank(), b, far(), .08, 1.0); e, why = m.target_estimate(); assert why is None and e['src'] == 'base' and e['u'] == V2.U_BASE_M and np.linalg.norm(e['xy'] - [.0, .10]) < V2.U_BASE_M
    m.update(3, blank(), render_base([target(.03, .12)]), far(), .08, 1.0)                                  # the target is somewhere else now
    assert m.log['conflicts'] == 1 and m.target_estimate() == (None, 'not_confirmed') and V2.Memory().target_estimate() == (None, 'never_seen')


def test_hand_view_takes_over_the_position_and_base_views_keep_confirming_it():
    m = V2.Memory(); b = render_base([target(.04, .0)]); p = pose()
    for t in range(3): m.update(t, blank(), b, far(), .08, 1.0)
    m.update(3, render(p, [target(.04, .0)]), b, p, .08, 1.0); e, _ = m.target_estimate()
    assert e['src'] == 'hand' and e['u'] == V2.U_HAND_M and np.linalg.norm(e['xy'] - [.04, .0]) < 0.001
    m.update(4, blank(), b, far(), .08, 1.0); e, _ = m.target_estimate(); assert e['src'] == 'hand' and e['t_last'] == 4 and e['confirmations'] == 5


def test_unobserved_steps_near_the_target_widen_the_bound_and_a_new_view_resets_it():
    m = V2.Memory(); b = render_base([target(.0, .10)])
    for t in range(3): m.update(t, blank(), b, far(), .08, 1.0)
    near = pose(.02, .10, .065); m.update(3, blank(), None, near, .04, -1.0); e, _ = m.target_estimate(); assert e['exposed_steps'] == 1 and e['u'] == V2.U_BASE_M + V2.U_PUSH_M
    m2 = m.copy(); m2.update(4, blank(), None, pose(.02, .10, .20), .04, -1.0); assert m2.target_estimate()[0]['exposed_steps'] == 1          # high above: no contact possible
    m.update(4, blank(), b, near, .04, -1.0); assert m.target_estimate()[0]['u'] == V2.U_BASE_M and m.target_estimate()[0]['exposed_steps'] == 0


def test_hold_memory_is_dropped_when_the_fingers_move_or_an_open_command_ran():
    p = pose(.03, -.02); img = render(p, [held(p, (.010, -.002, -.006), -.3)], finger=True); m = V2.Memory(); m.update(4, img, None, p, .04, -1.0)
    h = m.hold_estimate(.04, -1.0); assert h is not None and np.linalg.norm((h['h'] - [.010, -.002, -.006])[:2]) < 0.003
    assert m.hold_estimate(.0415, -1.0) is None and m.hold_estimate(.04, 1.0) is None and m.hold_estimate(.04, None) is None
    m.update(5, img, None, p, .043, -1.0); assert m.hold is None and m.log['hold_dropped'] == 1                # not a fit step (5 % 4 != 0): dropped, not refreshed
    c = V2.Memory(); c.update(4, img, None, p, .08, 1.0); assert c.hold is None and c.log['hold_fits_run'] == 0   # gripper open: no fit is run


# ---------------------------------------------------------------- estimate and decision
HID_T = (.05, .04); HID_P = (.078, .04)          # the target sits behind the held cube as seen from the hand camera; the base camera sees its whole top face


def remembered(xy, n=3):
    m = V2.Memory(); b = render_base([target(*xy)])
    for t in range(n): m.update(t, blank(), b, far(), .08, 1.0)
    return m


def at_trigger(p, cubes, m, steps_left=100):
    img = render(p, cubes); rd = V1.reading(img, p); m.note_hold(50, rd['hold_fit'], .04); est = V2.estimate(rd, m, img, p, .04, -1.0); return est, V2.decide(est, steps_left), V1.decide(rd, steps_left)


def test_hidden_target_is_taken_from_memory_and_v1_abstains():
    p = pose(*HID_P); est, d2, d1 = at_trigger(p, [target(*HID_T), held(p)], remembered(HID_T))
    assert d1['reason'] == 'NO_TARGET_ESTIMATE' and d2['reason'] == 'ACT' and est['target_src'] == 'memory_base' and est['hold_src'] == 'frame' and est['u'] == V2.U_BASE_M + V2.U_HOLD_M
    assert np.linalg.norm(est['offset_xy'] - [-.028, .0]) < est['u'] and V2.stop_tol(est) == V2.MEMORY_STOP_TOL_M and not est['memory_check']['contradicted']


def test_no_memory_or_unconfirmed_memory_abstains():
    p = pose(*HID_P); cubes = [target(*HID_T), held(p)]
    assert at_trigger(p, cubes, V2.Memory())[1]['reason'] == 'NO_TARGET_ESTIMATE' and at_trigger(p, cubes, remembered(HID_T, n=2))[1]['reason'] == 'NO_TARGET_ESTIMATE'


def test_visible_target_is_used_as_in_v1():
    p = pose(); est, d2, d1 = at_trigger(p, [target(.03, .012, .3), held(p)], remembered((.0, .20)))             # a wrong memory is ignored when the frame gives the target
    assert d1['reason'] == d2['reason'] == 'ACT' and est['target_src'] == 'frame' and V2.stop_tol(est) == V2.RESIDUAL_TOL_M


def test_a_memory_contradicted_by_the_current_image_is_refused():
    p = pose(); img = render(p, [target(.09, .0), held(p)]); m = remembered((.05, .04)); rd = V1.reading(render(p, [held(p)]), p); m.note_hold(50, rd['hold_fit'], .04)
    rd = {**rd, 'target_fresh': False}; est = V2.estimate(rd, m, img, p, .04, -1.0)                               # the image shows the target elsewhere, table where it is remembered
    assert est['why'] == 'MEMORY_CONTRADICTED' and V2.decide(est, 100)['reason'] == 'MEMORY_CONTRADICTED'
    ok = V2.memory_check(render(p, [target(.05, .04), held(p)]), p, np.array([.05, .04]), V2.U_BASE_M); assert not ok['contradicted'] and ok['lit_fraction_inside'] > 0.9


def fake_est(n, u, gap, src='frame', z=None):
    return {'why': None, 'offset_norm_m': n, 'u': u, 'gap_m': gap, 'target_src': src, 'held_xyz': np.array([0., 0., .06 + gap if z is None else z])}


def test_uncertainty_and_safety_rules():
    d = lambda *a, **k: V2.decide(fake_est(*a, **k), 100)['reason']
    assert d(.016, .004, .005) == 'ACT' and d(.016, .007, .005, 'memory_base') == 'ALIGNED_WITHIN_UNCERTAINTY' and d(.014, .004, .005) == 'ALIGNED' and d(.018, .007, .005, 'memory_base') == 'ACT'
    assert d(.0409, .004, -.0002) == 'SIDE_CONTACT_RISK'                                 # scene 50892: beside the target, bottom not above its top face
    assert d(.038, .004, .0006) == 'ACT' and d(.0409, .004, .004) == 'ACT'              # over the target (50618: estimate 38.0 mm), or beside it but clearly above
    assert d(.036, .007, .001, 'memory_base') == 'SIDE_CONTACT_RISK'                    # a remembered target counts its bound
    assert d(.030, .007, -.001, 'memory_base') == 'LOW_CARRY' and d(.030, .004, -.001) == 'ACT' and d(.030, .004, -.004) == 'LOW_CARRY'
    assert V2.decide(fake_est(.03, .004, .005), 10)['reason'] == 'TOO_FEW_STEPS' and V2.decide({'why': 'NO_HOLD_ESTIMATE'}, 100)['reason'] == 'NO_HOLD_ESTIMATE'
    assert V2.ACT_MIN_M == V1.ACT_MIN_M and V2.RESIDUAL_TOL_M == V1.RESIDUAL_TOL_M and V2.TARGET_JUMP_MAX_M < V1.TARGET_JUMP_MAX_M


def test_v2_code_does_not_touch_object_state_and_v1_is_unchanged():
    for f in (V2.target_base, V2.Memory, V2.memory_check, V2.estimate, V2.decide, V2.observe, V2.vision_center_v2):
        src = inspect.getsource(f)
        for word in ('cubeA', 'cubeB', 'get_state', 'is_grasp', 'success', 'oracle_offset', 'probe'): assert word not in src, (f.__name__, word)
    assert set(inspect.signature(V2.Memory.update).parameters) == {'self', 't', 'img_hand', 'img_base', 'tcp_pose', 'finger_width', 'last_grip'}
    assert V1.DECISION == {'act_min_m': .015, 'residual_tol_m': .010, 'carry_z_min_m': .058, 'steps_left_min': 20, 'target_jump_max_m': .005} and V1.GATE == {'rescue_min': 4, 'net_min': 3, 'harm_max': 3}


# ---------------------------------------------------------------- simulator side (mock physics, synthetic images)
class VisionEnv2(VisionEnv):
    """VisionEnv plus a base-camera image of both cubes (the arm is not drawn)."""
    def obs(self):
        o = super().obs(); p = o['extra']['tcp_pose'][0].numpy().astype(float)
        A = (self.A[0].numpy().astype(float) if not self.held else np.array([p[0], p[1], .065]), 0., 0); B = (self.B[0].numpy().astype(float), 0., 1)
        o['sensor_data']['base_camera'] = {'rgb': torch.from_numpy(render_base([B, A]))[None]}; return o


def vscene(bias, arms=V2.ARMS, seed=7):
    env = VisionEnv2(); r = Runtime(env, BiasPolicy(bias), CFG); return (*V2.collect_scene(r, seed, CFG, probe=probe, arms=arms), env)


def cell(rec, arm): return next(x for x in rec['cells'] if x['arm'] == arm)


def test_scene_where_only_the_memory_lets_v2_act():
    rec, traces, frames, env = vscene((.028, .012)); a, b, c0 = cell(rec, 'VISION_CENTER_V1'), cell(rec, 'VISION_CENTER_V2'), cell(rec, 'C0'); c = b['control']
    assert rec['triggered'] and a['control']['decision'] == 'NO_TARGET_ESTIMATE' and not a['acted'] and (a['y'], a['steps']) == (c0['y'], c0['steps'])
    assert b['acted'] and c['used_memory'] and c['readings'][0]['target_src'].startswith('memory_') and 1 <= c['steps_used'] <= O.MAX_STEPS and b['oracle_reads'] == 0 and cell(rec, 'CENTER_ORACLE')['oracle_reads'] > 0
    g = b['gt_reading_error']; assert g['error_within_bound'] and not g['unnecessary_act'] and b['gt_effect_of_control']['offset_xy_after_m'] < b['gt_effect_of_control']['offset_xy_before_m']
    assert c0['y'] == 0 and c['end_reason'] in ('within_tolerance', 'step_budget', 'target_cap') and b['gt_effect_of_control']['offset_xy_after_m'] < .015 and rec['memory']['log']['base_obs'] + rec['memory']['log']['hand_obs'] >= V2.CONFIRM_MIN and len(frames['trunk']) == rec['trigger']['step'] + 1 and frames['trunk'][0].shape == (V1.IMG, 2 * V1.IMG, 3)
    assert V2.validate_scene(rec, CFG) and set(traces) == {'trunk', *V2.ARMS}; json.dumps(rec)
    s = V2.v2_stats([rec]); assert s['acts_using_memory'] == 1 and s['memory_errors'] == 0 and sum(v for k, v in s['acts_by_source'].items() if k.startswith('target_memory')) == 1


def test_scene_where_both_see_the_target():
    rec, *_ = vscene((-.028, .0)); a, b = cell(rec, 'VISION_CENTER_V1'), cell(rec, 'VISION_CENTER_V2')
    assert a['acted'] and b['acted'] and not b['control']['used_memory'] and b['control']['readings'][0]['target_src'] == 'frame' and a['y'] == b['y'] and V2.stop_tol(b['control']['readings'][0]) == V2.RESIDUAL_TOL_M


def test_abstaining_v2_is_the_uninterrupted_run_and_touches_nothing():
    rec, traces, frames, env = vscene((-.004, .0), arms=V2.SHADOW_ARMS); b, c0 = cell(rec, 'VISION_CENTER_V2'), cell(rec, 'C0')
    assert not b['acted'] and b['control']['decision'] in V2.ABSTAIN and (b['y'], b['steps']) == (c0['y'], c0['steps']) and b['reproduces_trunk']['match'] and b['intervention'] is None and rec['oracle_reads'] == 0
    env = VisionEnv2(); r = Runtime(env, BiasPolicy((-.004, .0)), CFG); r.reset(3); m = V2.Memory()
    while not O.trigger_reading(r)['fires']: m.update(r.t, *V2.observe(r)); r.step_base()
    q = [x.copy() for x in r.policy.queue]; t = r.t; before = env.reads; info = V2.vision_center_v2(r, m)
    assert not info['acted'] and r.t == t and r.intervention is None and env.reads == before and len(r.policy.queue) == len(q) and all((x == y).all() for x, y in zip(q, r.policy.queue))


def test_validate_scene_rejects_inconsistent_records():
    rec, *_ = vscene((.028, .012)); bad = json.loads(json.dumps(rec)); cell(bad, 'VISION_CENTER_V2')['oracle_reads'] = 1
    with pytest.raises(ValueError, match='ground truth'): V2.validate_scene(bad, CFG)
    bad = json.loads(json.dumps(rec)); cell(bad, 'VISION_CENTER_V2')['control']['readings'][0]['u'] = .03
    with pytest.raises(ValueError, match='uncertainty'): V2.validate_scene(bad, CFG)
    bad = json.loads(json.dumps(rec)); cell(bad, 'VISION_CENTER_V2')['control']['steps'][0]['commanded_xy_m'] = [.02, 0]
    with pytest.raises(ValueError, match='bound'): V2.validate_scene(bad, CFG)
    bad = json.loads(json.dumps(rec)); cell(bad, 'VISION_CENTER_V1')['y'] = 1 - cell(bad, 'C0')['y']
    with pytest.raises(ValueError, match='abstaining'): V2.validate_scene(bad, CFG)


# ---------------------------------------------------------------- accounting
def fake(scene, y0, y1, y2, yo=None, acted=True, mem=True, err=.002, u=.007, true=.03, pushed=0.):
    ge = {'offset_error_m': err, 'true_offset_m': true, 'error_within_bound': err <= u, 'unnecessary_act': acted and true <= .01, 'missed_large_offset': (not acted) and true >= .02}
    rd = {'target_src': 'memory_base' if mem else 'frame', 'hold_src': 'frame'}
    v2 = {'arm': 'VISION_CENTER_V2', 'y': y2, 'acted': acted, 'control': {'decision': 'ACT' if acted else 'ALIGNED', 'end_reason': 'within_tolerance', 'used_memory': mem, 'readings': [rd]}, 'gt_reading_error': ge, 'gt_effect_of_control': {'cubeB_xy_displacement_m': pushed}}
    cells = [{'arm': 'C0', 'y': y0}, {'arm': 'VISION_CENTER_V1', 'y': y1, 'acted': y1 != y0, 'control': {'decision': 'ACT' if y1 != y0 else 'NO_TARGET_ESTIMATE'}}, v2]
    if yo is not None: cells.append({'arm': 'CENTER_ORACLE', 'y': yo})
    return {'scene': scene, 'triggered': True, 'cells': cells}


def test_shadow_verdict_criteria():
    ok = [fake(i, 0, 0, 1) for i in range(6)] + [fake(10 + i, 0, 1, 1, mem=False) for i in range(4)] + [fake(30 + i, 1, 1, 1, acted=False, true=.005) for i in range(20)]
    v = V2.shadow_verdict(ok); assert v['pass'] and v['label'] == 'V2_SHADOW_PASS' and v['acts_using_memory'] == 6 and v['paired_V2_vs_C0']['net'] == 10 and v['paired_V1_vs_C0']['net'] == 4
    assert not V2.shadow_verdict(ok[2:])['criteria']['memory_acts_at_least']
    assert not V2.shadow_verdict(ok + [fake(90 + i, 1, 1, 0) for i in range(3)])['criteria']['harm_at_most']
    assert not V2.shadow_verdict(ok + [fake(95 + i, 1, 1, 1, true=.008) for i in range(3)])['criteria']['unnecessary_acts_at_most']
    bad = V2.shadow_verdict(ok + [fake(70 + i, 1, 1, 1, err=.009) for i in range(3)]); assert not bad['criteria']['memory_errors_at_most'] and bad['memory_error_scenes'] == [70, 71, 72]
    assert not V2.shadow_verdict(ok + [fake(80 + i, 1, 1, 1, pushed=.004) for i in range(2)])['criteria']['target_pushed_at_most']
    assert not V2.shadow_verdict([fake(i, 0, 1, 1) for i in range(8)])['criteria']['net_above_v1_by_at_least']


def test_summarize_needs_every_planned_scene_and_applies_the_gate():
    recs = [fake(i, 0, 0, 1, 1) for i in range(5)] + [fake(5 + i, 0, 1, 1, 1, mem=False) for i in range(2)] + [fake(10 + i, 1, 1, 1, 1, acted=False, true=.004) for i in range(5)]; s = V2.summarize(recs, [r['scene'] for r in recs])
    assert s['success']['C0']['k'] == 5 and s['success']['VISION_CENTER_V2']['k'] == 12 and s['followup_gate']['label'] == 'V2_GATE_PASS' and s['oracle_rescue_recovery'] == {'v2': 7, 'v1': 2, 'oracle': 7, 'v2_share': 1.0}
    assert s['v2']['abstain_rate_of_triggered'] == 5 / 12 and s['V2_vs_V1']['rescue'] == 5
    with pytest.raises(ValueError, match='planned'): V2.summarize(recs[:-1], [r['scene'] for r in recs])
    assert V2.summarize([fake(i, 0, 0, 0, 1) for i in range(8)], list(range(8)))['followup_gate']['label'] == 'NO_V2_GATE_PASS'
    assert not V2.summarize(recs + [fake(50 + i, 1, 1, 1, 1, err=.009) for i in range(2)], [r['scene'] for r in recs] + [50, 51])['followup_gate']['criteria']['memory_errors_at_most']


def test_projected_cost_and_seed_set():
    kw = dict(seconds_per_env_step=.012, seconds_per_reading=.3, seconds_per_memory_step=.15, usd_per_hour=.47, spent_usd=.2, scenes=200, arms=4)
    a = V2.projected_cost(cap_usd=3., **kw); assert a['fits_cap'] and not V2.projected_cost(cap_usd=.5, **kw)['fits_cap']
    assert V2.projected_cost(cap_usd=3., **{**kw, 'seconds_per_memory_step': .6})['projected_total_usd'] > a['projected_total_usd']
    man = json.loads(Path('configs/vision_center_v2_splits.json').read_text(encoding='utf-8')); assert V2.pilot_seeds(man, {'x': [50899, 51100]}) == list(range(50900, 51100))
    with pytest.raises(ValueError, match='recorded use'): V2.pilot_seeds(man, {'x': [50950]})
    old = {s for f in ('configs/recovery_v1_splits.json', 'configs/vision_center_splits.json') for v in json.loads(Path(f).read_text(encoding='utf-8'))['splits'].values() for s in v}
    assert not set(man['splits'][V2.SET]) & old


@pytest.fixture(scope='module')
def X33():
    spec = importlib.util.spec_from_file_location('vc2_driver', Path('scripts/33_vision_center_v2.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_every_server_command_refuses_without_an_explicit_authorization(X33, tmp_path):
    plan = tmp_path / 'plan.json'; plan.write_text('{"code": {}}'); run = tmp_path / 'run'; run.mkdir()
    ns = lambda: argparse.Namespace(plan=str(plan), run=str(run), upstream='x', device='cpu', usd_per_hour=.4, spent_usd=0., stage='shadow')
    had_ledger = Path(X33.LEDGER).exists()
    for cmd in (X33.cmd_scenes, X33.cmd_audit, X33.cmd_budget, X33.cmd_shadow, X33.cmd_run):
        with pytest.raises(SystemExit, match='no authorization'): cmd(ns())
    (run / 'authorization.json').write_text(json.dumps({'approved_by': 'agent', 'paid_compute_authorized': True, 'cap_usd': 2.5, 'plan_sha256': X33.sha256(plan)}))
    for cmd in (X33.cmd_scenes, X33.cmd_audit, X33.cmd_budget, X33.cmd_shadow, X33.cmd_run):
        with pytest.raises(SystemExit, match='Refused'): cmd(ns())
    assert not [p for p in run.iterdir() if p.is_dir()] and Path(X33.LEDGER).exists() == had_ledger


def test_driver_constants_and_file_list(X33):
    assert X33.shadow_seeds() == list(range(50600, 50900)) and not set(X33.shadow_seeds()) & set(range(50900, 51100))
    if Path(X33.LEDGER).exists(): assert json.loads(Path(X33.LEDGER).read_text(encoding='utf-8'))['plan_sha256'] == '1b4ea3bb65bd4cfdffc13a610fee010d568a5837e632701d4b0d265e6ba66e86'      # the one executed pass; before it, no ledger existed
    assert all(Path(f).exists() for f in X33.CODE) and {'src/hci/vision_center.py', 'src/hci/vision_center_v2.py'} <= set(X33.CODE) and X33.MANIFEST in X33.SERVER_READS
    src = Path('scripts/33_vision_center_v2.py').read_text(encoding='utf-8'); assert 'authorization.json' in src and "write_json(run / 'authorization" not in src
    assert X33.ANALYSIS['stage_B1_shadow']['pass_criteria'] == V2.SHADOW_PASS and X33.ANALYSIS['stage_B2_pilot']['gate']['thresholds'] == V2.GATE and X33.SEMANTICS['VISION_CENTER_V2']['decision'] == V2.DECISION
