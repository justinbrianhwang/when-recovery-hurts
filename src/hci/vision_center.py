"""Vision-CENTER (work id vision_center_001): the bounded horizontal centring move of the Recovery-V1 oracle pilot, driven by
RGB images and robot state instead of simulator object positions. PREPARATION: perception is evaluated offline on stored
frames; nothing in this module has been run in a simulator.

What the controller may read: the two RGB images of the policy observation (base_camera, hand_camera), the TCP pose
(extra/tcp_pose), the finger joints (agent/qpos), the pending and executed actions, the step counter. It never reads a cube
pose, a grasp flag, a task flag or an outcome. Ground truth is used only by offline labelling and by after-the-fact logging.

Perception is colour segmentation plus projective geometry with fixed, known camera parameters:
  target cube (green): centroid of its lit top face, back-projected onto the plane of that face (z = 0.04 m), accepted only
      when the whole face is visible (pixel count against the area the face would have at the estimated place);
  held cube (red): the lit face seen by the hand camera is fitted with a projected 4 cm square in the gripper frame
      (hold offset and yaw), which gives the cube centre relative to the TCP;
  offset = (TCP + hold offset) - target, horizontal part.
Abstaining keeps the policy's action queue untouched, so an abstaining episode is the uninterrupted episode.
"""
import numpy as np

WORK = 'vision_center_001'
CAMERAS = ('base_camera', 'hand_camera')
# ---------------------------------------------------------------- fixed geometry (pinned ManiSkill commit; checked against stored sensor_param by the offline script)
IMG = 128
FOCAL_PX = 64.0; CENTRE_PX = 64.0                 # both cameras: 128 x 128, 90 degree field of view
PIXEL_CENTRE = 0.5                                # a pixel index u covers [u, u+1); its centre is u + 0.5
BASE_EXTRINSIC = np.array([[0., 1., 0., 0.], [0.78086877, 0., -0.62469506, 0.14055645], [-0.62469506, 0., -0.78086877, 0.6559298]])   # world -> camera (OpenCV), look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
HAND_R_TCP = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])      # camera axes in the TCP frame (camera looks along the approach axis)
HAND_P_TCP = np.array([0.0465, -0.02, -0.0674])                         # camera position in the TCP frame, metres
CUBE_HALF_M = 0.02
TARGET_TOP_Z_M = 0.04                             # top face of the target cube resting on the table (table surface z = 0)

# ---------------------------------------------------------------- colour classes (simulator renders flat colours; thresholds are wide)
LIT_MIN = 215; DARK_MIN = 120; OTHER_MAX = 60; GREY_SPREAD_MAX = 30


def quat_R(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cam_base():
    R = BASE_EXTRINSIC[:, :3]; return R.T, -R.T @ BASE_EXTRINSIC[:, 3]            # camera-to-world rotation, camera position


def cam_hand(tcp_pose):
    """Hand camera pose from the TCP pose (position, quaternion wxyz) and the fixed mount transform. Robot state only."""
    p = np.asarray(tcp_pose, float).reshape(-1); R = quat_R(p[3:7]); return R @ HAND_R_TCP, p[:3] + R @ HAND_P_TCP


def classes(img):
    a = np.asarray(img).astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return {'green_lit': (g >= LIT_MIN) & (r < OTHER_MAX) & (b < OTHER_MAX), 'green': (g >= DARK_MIN) & (r < OTHER_MAX) & (b < OTHER_MAX),
            'red_lit': (r >= LIT_MIN) & (g < OTHER_MAX) & (b < OTHER_MAX), 'red': (r >= DARK_MIN) & (g < OTHER_MAX) & (b < OTHER_MAX),
            'grey': (np.abs(r - g) <= GREY_SPREAD_MAX) & (np.abs(g - b) <= GREY_SPREAD_MAX) & (np.abs(r - b) <= GREY_SPREAD_MAX)}


_V, _U = np.mgrid[0:IMG, 0:IMG]; _PU = _U + PIXEL_CENTRE; _PV = _V + PIXEL_CENTRE


def project(cam, X):
    R, p = cam; x = R.T @ (np.asarray(X, float) - p)
    return np.array([FOCAL_PX * x[0] / x[2] + CENTRE_PX, FOCAL_PX * x[1] / x[2] + CENTRE_PX]), float(x[2])


def plane_points(cam, z):
    """World x, y of every pixel's ray on the horizontal plane at height z; `ahead` is False where the ray never reaches it."""
    R, p = cam; d = np.stack([(_PU - CENTRE_PX) / FOCAL_PX, (_PV - CENTRE_PX) / FOCAL_PX, np.ones_like(_PU)], -1) @ R.T
    with np.errstate(divide='ignore', invalid='ignore'): s = (z - p[2]) / d[..., 2]
    ahead = np.isfinite(s) & (s > 0); s = np.where(ahead, s, 0.)
    return p[0] + s * d[..., 0], p[1] + s * d[..., 1], ahead


def ray_plane(cam, u, v, z):
    R, p = cam; d = R @ np.array([(u - CENTRE_PX) / FOCAL_PX, (v - CENTRE_PX) / FOCAL_PX, 1.0]); s = (z - p[2]) / d[2]
    return p + s * d, float(s)


def quad_area(P):
    x, y = P[:, 0], P[:, 1]; return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def raster(P):
    """Pixels whose centre lies inside the convex quadrilateral P (4 x 2, image coordinates)."""
    P = np.asarray(P, float); x, y = P[:, 0], P[:, 1]; sg = 1.0 if (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0 else -1.0; m = np.ones((IMG, IMG), bool)
    for i in range(4):
        x0, y0 = P[i]; x1, y1 = P[(i + 1) % 4]; m &= sg * ((x1 - x0) * (_PV - y0) - (y1 - y0) * (_PU - x0)) >= 0
    return m


# ---------------------------------------------------------------- target cube from one full view
FULL_VIEW = {'hand_camera': {'ratio_min': 0.95, 'ratio_max': 1.05, 'pixels_min': 60}, 'base_camera': {'ratio_min': 0.80, 'ratio_max': 1.25, 'pixels_min': 9}}


def target_full_view(img, cam, name):
    """Centre of the target cube's top face from one image, or None. `valid` only when the whole face is seen: the lit green
    pixels do not touch the image border and their count matches the area a 4 cm square would have at the estimated place."""
    m = classes(img)['green_lit']; n = int(m.sum())
    if n < 4: return None
    v, u = np.nonzero(m); X, s = ray_plane(cam, u.mean() + PIXEL_CENTRE, v.mean() + PIXEL_CENTRE, TARGET_TOP_Z_M)
    if not s > 0: return None
    h = CUBE_HALF_M; quad = np.array([project(cam, [X[0] + a, X[1] + b, TARGET_TOP_Z_M])[0] for a, b in ((-h, -h), (h, -h), (h, h), (-h, h))])
    ratio = n / max(quad_area(quad), 1e-6); border = bool(u.min() == 0 or v.min() == 0 or u.max() == IMG - 1 or v.max() == IMG - 1); k = FULL_VIEW[name]
    return {'xy': X[:2].copy(), 'pixels': n, 'area_ratio': float(ratio), 'touches_border': border, 'camera': name,
            'valid': bool(not border and n >= k['pixels_min'] and k['ratio_min'] <= ratio <= k['ratio_max'])}


# ---------------------------------------------------------------- a 4 cm square on a known plane, scored against a pixel mask
def _square_scores(PX, PY, lit, known, gx, gy, yaws):
    """For every placement (gx[i], gy[i]) the best score over `yaws` of a 4 cm square on the plane whose pixel coordinates
    are PX, PY: |square and lit| / (|lit| + |square, not lit, not an occluder|). Returns scores and the best yaw."""
    n = int(lit.sum()); best = np.full(len(gx), -1.0); arg = np.zeros(len(gx)); h = CUBE_HALF_M; extra = known & ~lit
    dx = PX[None, :] - gx[:, None]; dy = PY[None, :] - gy[:, None]
    for yw in yaws:
        c, s = np.cos(yw), np.sin(yw); inside = (np.abs(c * dx + s * dy) <= h) & (np.abs(-s * dx + c * dy) <= h)
        sc = inside[:, lit].sum(1) / np.maximum(n + inside[:, extra].sum(1), 1); up = sc > best; best[up] = sc[up]; arg[up] = yw
    return best, arg


def _grid(c0, span, st):
    a = np.arange(-span, span + 1e-9, st); gx, gy = np.meshgrid(c0[0] + a, c0[1] + a, indexing='ij'); return gx.ravel(), gy.ravel()


# ---------------------------------------------------------------- held cube in the hand camera
HOLD_FIT = {'pixels_min': 200, 'iou_min': 0.90, 'outline_min': 0.85, 'margin_min': 0.02, 'far_m': 0.004, 'far_z_m': 0.006, 'z_span_m': 0.024, 'z_step_m': (0.004, 0.001), 'stages': ((0.020, 0.004, 5), (0.004, 0.001, 5), (0.001, 0.0005, 3))}


def _hold_plane(hz):
    """TCP-frame x, y of every pixel on the plane of the held cube's camera-facing face, for vertical hold offset hz."""
    zc = hz - CUBE_HALF_M - HAND_P_TCP[2]
    if zc <= 1e-3: return None
    xc = (_PU - CENTRE_PX) / FOCAL_PX * zc; yc = (_PV - CENTRE_PX) / FOCAL_PX * zc
    return HAND_R_TCP[0, 0] * xc + HAND_R_TCP[0, 1] * yc + HAND_R_TCP[0, 2] * zc + HAND_P_TCP[0], HAND_R_TCP[1, 0] * xc + HAND_R_TCP[1, 1] * yc + HAND_R_TCP[1, 2] * zc + HAND_P_TCP[1]


def _hold_at(lit, hz, c0, yaw0, yaw_span, stages, stride=1):
    P = _hold_plane(hz)
    if P is None: return (-1.0, np.array([c0[0], c0[1], hz]), yaw0)
    X, Y = P; reach = CUBE_HALF_M * 1.5 + stages[0][0]; sel = ((np.abs(X - c0[0]) < reach) & (np.abs(Y - c0[1]) < reach)) | lit
    if stride > 1: sel &= (_U % stride == 0) & (_V % stride == 0)          # every stride-th pixel in both directions: the score is a ratio of counts
    PX, PY, L = X[sel], Y[sel], lit[sel]; K = np.ones(L.shape, bool); best = (-1.0, np.array([c0[0], c0[1]], float), yaw0)
    for span, st, ny in stages:
        yaws = best[2] + np.linspace(-yaw_span, yaw_span, ny, endpoint=False) + yaw_span / ny; yaw_span = yaw_span / ny * 1.5
        gx, gy = _grid(best[1], span, st); sc, arg = _square_scores(PX, PY, L, K, gx, gy, yaws); i = int(sc.argmax())
        if sc[i] > best[0]: best = (float(sc[i]), np.array([gx[i], gy[i]]), float(arg[i]))
    return (best[0], np.array([best[1][0], best[1][1], hz]), best[2])


_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def hold_silhouette(h, yaw):
    """Pixels covered by the whole held cube (all six faces) for hold offset h and yaw, or None if a corner is behind the camera."""
    c, s = np.cos(yaw), np.sin(yaw); P = []
    for dz in (-CUBE_HALF_M, CUBE_HALF_M):
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x = HAND_R_TCP.T @ (np.array([h[0] + CUBE_HALF_M * (a * c - b * s), h[1] + CUBE_HALF_M * (a * s + b * c), h[2] + dz]) - HAND_P_TCP)
            if x[2] <= 1e-3: return None
            P.append((FOCAL_PX * x[0] / x[2] + CENTRE_PX, FOCAL_PX * x[1] / x[2] + CENTRE_PX))
    P = np.array(P); m = np.zeros((IMG, IMG), bool)
    for f in _FACES: m |= raster(P[list(f)])
    return m


def hold_fit(img_hand):
    """Hold offset of the grasped cube (cube centre minus TCP, in the TCP frame) from the hand image.
    Step 1: the 4 cm square, at the depth given by the vertical hold offset, that best overlaps the lit red pixels (the
    camera-facing face), searched coarse to fine; the three best depths are kept, because depth and position trade off
    when that face is cut by the image border.
    Step 2: each of the three is scored again with the outline of the whole cube against all red pixels (grey robot pixels
    are not counted), which separates the depths. `margin` is the lead of the chosen one over the best different one.
    Returns None when too little of the face is visible."""
    k = classes(img_hand); lit = k['red_lit']; n = int(lit.sum())
    if n < HOLD_FIT['pixels_min']: return None
    red = k['red']; known = ~k['grey']; nred = int(red.sum()); z0, z1 = HOLD_FIT['z_step_m']; st = HOLD_FIT['stages']

    def outline(h, yaw):
        sil = hold_silhouette(h, yaw); return 0.0 if sil is None else int((sil & red).sum()) / max(nred + int((sil & ~red & known).sum()), 1)
    coarse = [_hold_at(lit, hz, (0.0, 0.0), 0.0, np.pi / 4, st[:1], 2) for hz in np.arange(-HOLD_FIT['z_span_m'], HOLD_FIT['z_span_m'] + 1e-9, z0)]; cands = []
    for c in sorted(coarse, key=lambda r: -(r[0] + outline(r[1], r[2])))[:3]:          # the three depths that explain the lit face AND the whole outline best
        best = c
        for hz in c[1][2] + np.arange(-z0 + z1, z0 - 1e-9, z1):
            r = _hold_at(lit, hz, c[1][:2], c[2], np.pi / 10, st[1:2], 2)
            if r[0] > best[0]: best = r
        best = _hold_at(lit, best[1][2], best[1][:2], best[2], np.pi / 60, st[2:], 1); so = outline(best[1], best[2])
        cands.append({'h': best[1], 'yaw': best[2], 'iou': float(best[0]), 'outline': float(so), 'total': float(best[0] + so)})
    cands.sort(key=lambda x: -x['total']); top = cands[0]
    other = [x['total'] for x in cands[1:] if np.linalg.norm(x['h'][:2] - top['h'][:2]) >= HOLD_FIT['far_m'] or abs(x['h'][2] - top['h'][2]) >= HOLD_FIT['far_z_m']]
    margin = float(top['total'] - max(other)) if other else 1.0
    return {'h': top['h'], 'yaw': top['yaw'], 'iou': top['iou'], 'outline': top['outline'], 'margin': margin, 'pixels': n,
            'valid': bool(top['iou'] >= HOLD_FIT['iou_min'] and top['outline'] >= HOLD_FIT['outline_min'] and margin >= HOLD_FIT['margin_min'])}


# ---------------------------------------------------------------- target cube from a partial view
PARTIAL_FIT = {'pixels_min': 40, 'score_min': 0.90, 'margin_min': 0.10, 'far_m': 0.004, 'stages': ((0.036, 0.004, 6), (0.004, 0.001, 5), (0.001, 0.0005, 3))}


def target_partial_fit(img, cam, name):
    """Target cube centre when part of its top face is hidden: the 4 cm square on the plane z = 0.04 m that best explains the
    lit green pixels, scored only on pixels that are not an occluder (red cube, grey robot). `margin` is the score lost by
    the best placement at least `far_m` away; a small margin means the visible part does not pin the square down."""
    k = classes(img); lit = k['green_lit']; n = int(lit.sum())
    if n < PARTIAL_FIT['pixels_min']: return None
    v, u = np.nonzero(lit); X0, s = ray_plane(cam, u.mean() + PIXEL_CENTRE, v.mean() + PIXEL_CENTRE, TARGET_TOP_Z_M)
    if not s > 0: return None
    X, Y, ahead = plane_points(cam, TARGET_TOP_Z_M); sel = ahead & ((np.hypot(X - X0[0], Y - X0[1]) < 0.09) | lit)
    PX, PY, L = X[sel], Y[sel], lit[sel]; K = ~(k['red'] | k['grey'])[sel]
    best = (-1.0, X0[:2].copy(), 0.0); yaw_span = np.pi / 4; seen = []
    for span, st, ny in PARTIAL_FIT['stages']:
        yaws = best[2] + np.linspace(-yaw_span, yaw_span, ny, endpoint=False) + yaw_span / ny; yaw_span = yaw_span / ny * 1.5
        gx, gy = _grid(best[1], span, st); sc, arg = _square_scores(PX, PY, L, K, gx, gy, yaws); i = int(sc.argmax()); seen.append((gx, gy, sc))
        if sc[i] > best[0]: best = (float(sc[i]), np.array([gx[i], gy[i]]), float(arg[i]))
    gx = np.concatenate([a[0] for a in seen]); gy = np.concatenate([a[1] for a in seen]); sc = np.concatenate([a[2] for a in seen])
    far = np.hypot(gx - best[1][0], gy - best[1][1]) >= PARTIAL_FIT['far_m']; margin = float(best[0] - sc[far].max()) if far.any() else 0.0
    return {'xy': best[1], 'yaw': best[2], 'score': float(best[0]), 'margin': margin, 'pixels': n, 'camera': name,
            'valid': bool(best[0] >= PARTIAL_FIT['score_min'] and margin >= PARTIAL_FIT['margin_min'])}


# ---------------------------------------------------------------- one reading: images and TCP pose -> horizontal offset
ACT_MIN_M = 0.015                 # act only when the estimated offset between the cube centres is at least this
RESIDUAL_TOL_M = 0.010            # stop when the estimated offset is at most this (the oracle pilot's tolerance)
CARRY_Z_MIN_M = 0.058             # estimated height of the held cube's centre; resting on the target is 0.060. Lower: the cube is beside the target and a horizontal move would push it
STEPS_LEFT_MIN = 20               # 12 control steps plus 8 for the release (the oracle pilot needed at most 6 after the hand-back)
TARGET_JUMP_MAX_M = 0.005         # during control: a new valid target estimate this far from the one at the trigger means the target moved
ABSTAIN = ('NO_TARGET_ESTIMATE', 'NO_HOLD_ESTIMATE', 'LOW_CARRY', 'TOO_FEW_STEPS', 'ALIGNED')
DECISION = {'act_min_m': ACT_MIN_M, 'residual_tol_m': RESIDUAL_TOL_M, 'carry_z_min_m': CARRY_Z_MIN_M, 'steps_left_min': STEPS_LEFT_MIN, 'target_jump_max_m': TARGET_JUMP_MAX_M}


def reading(img_hand, tcp_pose, target=None):
    """What Vision-CENTER knows at one control step, from the hand image and the TCP pose alone. `target` (an earlier valid
    target estimate of the same control part) is used when the target is hidden in this image. No ground truth."""
    p = np.asarray(tcp_pose, float).reshape(-1); cam = cam_hand(p); tf = target_partial_fit(img_hand, cam, 'hand_camera'); hf = hold_fit(img_hand)
    fresh = bool(tf is not None and tf['valid']); txy = tf['xy'] if fresh else (None if target is None else np.asarray(target, float))
    out = {'target_fit': tf, 'hold_fit': hf, 'target_fresh': fresh, 'target_xy': txy, 'held_xyz': None, 'offset_xy': None, 'offset_norm_m': None}
    if hf is not None and hf['valid']:
        A = p[:3] + quat_R(p[3:7]) @ hf['h']; out['held_xyz'] = A
        if txy is not None: out['offset_xy'] = txy - A[:2]; out['offset_norm_m'] = float(np.linalg.norm(out['offset_xy']))       # the vector the TCP still has to move
    return out


def decide(rd, steps_left):
    """Act or abstain at the trigger. Abstaining leaves the policy's queue untouched."""
    if rd['hold_fit'] is None or not rd['hold_fit']['valid']: why = 'NO_HOLD_ESTIMATE'
    elif not rd['target_fresh']: why = 'NO_TARGET_ESTIMATE'
    elif rd['held_xyz'][2] < CARRY_Z_MIN_M: why = 'LOW_CARRY'
    elif steps_left < STEPS_LEFT_MIN: why = 'TOO_FEW_STEPS'
    elif rd['offset_norm_m'] < ACT_MIN_M: why = 'ALIGNED'
    else: why = 'ACT'
    return {'act': why == 'ACT', 'reason': why}


def summary(rd):
    """JSON-able record of a reading (no images)."""
    f = lambda x: None if x is None else [float(v) for v in np.asarray(x).reshape(-1)]; tf, hf = rd['target_fit'], rd['hold_fit']
    return {'target_fit': None if tf is None else {'xy': f(tf['xy']), 'yaw': tf['yaw'], 'score': tf['score'], 'margin': tf['margin'], 'pixels': tf['pixels'], 'valid': tf['valid']},
            'hold_fit': None if hf is None else {'h': f(hf['h']), 'yaw': hf['yaw'], 'iou': hf['iou'], 'outline': hf['outline'], 'margin': hf['margin'], 'pixels': hf['pixels'], 'valid': hf['valid']},
            'target_fresh': rd['target_fresh'], 'target_xy': f(rd['target_xy']), 'held_xyz': f(rd['held_xyz']), 'offset_xy': f(rd['offset_xy']), 'offset_norm_m': rd['offset_norm_m']}


# ================================================================ simulator side (not run in Phase A; mock-tested only)
ARMS = ('C0', 'VISION_CENTER', 'CENTER_ORACLE')
ROLE = 'vision_center_dev'; SET = 'vision_center_pilot'; RANGE = (50800, 50899); REPEAT = 0
SHADOW_ARMS = ('C0', 'VISION_CENTER'); SHADOW_ROLE = 'vision_center_shadow_used_scenes'
SHADOW_SETS = (('recovery_v1_diag', (50600, 50699)), ('recovery_v1_pilot_reserved', (50700, 50799)))     # already used; development data
INTERVENTION = 'vision_center_xy'
GATE = {'rescue_min': 4, 'net_min': 3, 'harm_max': 3}                                                   # VISION_CENTER against C0; the oracle pilot's engineering gate, unchanged
SHADOW_PASS = {'acts_min': 10, 'unnecessary_acts_max': 2, 'harm_max': 2, 'rescue_minus_harm_min': 3, 'offset_error_at_act_median_max_m': 0.003, 'offset_error_at_act_max_m': 0.008}
READINGS_PER_SCENE_BOUND = 1 + 12                # the reading at the trigger and one per control step


def pilot_seeds(manifest, used):
    ids = list(manifest['splits'][SET])
    if ids != list(range(RANGE[0], RANGE[1] + 1)): raise ValueError(f'{SET} is not the frozen range {RANGE[0]}-{RANGE[1]}')
    clash = {src: sorted(set(v) & set(ids))[:10] for src, v in used.items()}; clash = {k: v for k, v in clash.items() if v}
    if clash: raise ValueError(f'pilot seeds have a recorded use: {clash}. The manifest is not patched silently')
    return ids


def observation(r):
    """The only inputs of Vision-CENTER: the hand camera image and the TCP pose of the policy's own observation."""
    from .trees import as_numpy
    img = np.asarray(as_numpy(r.raw['sensor_data']['hand_camera']['rgb'])).reshape(IMG, IMG, 3); pose = np.asarray(as_numpy(r.raw['extra']['tcp_pose']), float).reshape(-1)[:7]
    return img, pose


def vision_center(r, frames=None):
    """At the trigger state. Abstain: nothing is touched (queue, history, counters) and the episode continues as the
    uninterrupted one. Act: the oracle pilot's bounded horizontal move (same per-step bound, target cap, step budget,
    held gripper command, zero vertical command), closed on the vision estimate; then the queue is cleared and the policy
    re-plans. Stops early when the estimate is lost, the target appears to have moved or the held cube is too low."""
    from . import recovery_v1_oracle as O
    if r.intervention is not None: raise ValueError('Invalid/repeated intervention')
    img, pose = observation(r); rd = reading(img, pose); d = decide(rd, r.horizon - r.t); log = [{'t': int(r.t), **summary(rd)}]
    if frames is not None: frames.append(img.copy())
    info = {'mode': 'VISION_CENTER', 'decision': d['reason'], 'acted': bool(d['act']), 'trigger_step': int(r.t), 'readings': log, 'steps_used': 0, 'end_reason': None, 'constants': dict(DECISION)}
    if not d['act']: return info
    grip = O.held_grip(r); scale = O.arm_scale(r); start = np.asarray(r.tcp_history[-1], float).copy(); target = rd['target_xy'].copy()
    r.intervention = INTERVENTION; r.intervention_t = r.t; r.policy.clear(); used = 0; path = 0.; why = None; trail = []
    while why is None:
        e = rd['offset_xy']; n = float(np.linalg.norm(e))
        if n <= RESIDUAL_TOL_M: why = 'within_tolerance'
        elif r.done: why = 'episode_ended'
        elif used >= O.MAX_STEPS: why = 'step_budget'
        else:
            cur = np.asarray(r.tcp_history[-1], float)[:2]; goal = cur + e * min(1., O.STEP_CMD_M / n); off = goal - start[:2]; m = float(np.linalg.norm(off))
            if m > O.TARGET_CAP_M: goal = start[:2] + off * (O.TARGET_CAP_M / m)
            step = goal - cur; s = float(np.linalg.norm(step))
            if s > O.STEP_CMD_M: step = step * (O.STEP_CMD_M / s); s = O.STEP_CMD_M
            reach = float(np.linalg.norm(cur + step - start[:2]))
            if s < O.MIN_CMD_M or reach > O.TARGET_CAP_M + 1e-9: why = 'target_cap'
            else:
                r.advance(O._xy_action(r, step, scale, grip)); used += 1; path += s
                trail.append({'commanded_xy_m': step.tolist(), 'target_distance_from_start_m': reach, 'estimated_offset_before_step_m': n, 'tcp_after': np.asarray(r.tcp_history[-1], float).tolist()})
                if r.done: why = 'episode_ended'; break
                img, pose = observation(r); rd = reading(img, pose, target=target); log.append({'t': int(r.t), **summary(rd)})
                if frames is not None: frames.append(img.copy())
                if rd['hold_fit'] is None or not rd['hold_fit']['valid']: why = 'estimate_lost'
                elif rd['target_fresh'] and float(np.linalg.norm(rd['target_fit']['xy'] - target)) > TARGET_JUMP_MAX_M: why = 'target_moved'
                elif rd['held_xyz'][2] < CARRY_Z_MIN_M: why = 'low_carry'
    final = np.asarray(r.tcp_history[-1], float)
    info.update(steps_used=used, end_reason=why, start_tcp=start.tolist(), final_tcp=final.tolist(), commanded_path_m=path, tcp_xy_displacement_m=float(np.linalg.norm(final[:2] - start[:2])),
                tcp_dz_m=float(final[2] - start[2]), gripper_command=grip, steps=trail, estimated_offset_before_m=log[0]['offset_norm_m'], estimated_offset_after_m=log[-1]['offset_norm_m'],
                bounds={'step_cmd_m': O.STEP_CMD_M, 'target_cap_m': O.TARGET_CAP_M, 'max_steps': O.MAX_STEPS})
    r.retreat_info = info; r.policy.clear(); return info


def _gt_error(info, gt):
    """After the fact: the reading at the trigger against the logged ground truth. Diagnosis only."""
    if gt is None: return None
    rd = info['readings'][0]; true = [gt['cubeB'][0] - gt['cubeA'][0], gt['cubeB'][1] - gt['cubeA'][1]]; n = float(np.hypot(*true))
    return {'true_offset_m': n, 'estimated_offset_m': rd['offset_norm_m'], 'offset_error_m': None if rd['offset_xy'] is None else float(np.hypot(rd['offset_xy'][0] - true[0], rd['offset_xy'][1] - true[1])),
            'target_error_m': None if not rd['target_fresh'] else float(np.hypot(rd['target_xy'][0] - gt['cubeB'][0], rd['target_xy'][1] - gt['cubeB'][1])),
            'unnecessary_act': bool(info['acted'] and n <= RESIDUAL_TOL_M), 'missed_large_offset': bool(not info['acted'] and n >= 0.02)}


def collect_scene(r, seed, c, probe=None, arms=ARMS, role=ROLE):
    """Trunk (never restored); if the trigger fired, every arm in `arms` from the one snapshot. `arms` is ARMS for the pilot
    and SHADOW_ARMS for the shadow pass on used scenes. Returns (record, traces, frames). A technical difference raises."""
    from . import recovery_v1_oracle as O
    import time
    arms = tuple(arms)
    if arms[0] != 'C0' or not set(arms) <= set(ARMS) or len(set(arms)) != len(arms): raise ValueError('arms must start with C0 and come from ARMS, once each')
    tol = c['audit']; started = time.perf_counter(); gate = O.OracleGate(); env_steps = 0
    r.reset(seed); r.repeat = REPEAT; initial = probe(r, None) if probe else None; z0 = initial['cubeA'][2] if initial else None; snap = None; traces = {}; frames = {}
    with O.Hooks(r, probe) as h:
        h.rows = trunk_rows = []; h.calls = trunk_calls = []
        while not r.done:
            if snap is None:
                rd = O.trigger_reading(r)
                if rd['fires']:
                    snap = {'s': r.snapshot(), 'tree': O.runtime_tree(r), 'reading': rd, 'queue_len': len(r.policy.queue), 'gt': O.gt_reading(trunk_rows[-1] if trunk_rows else initial, z0), 'last_grip': float(r.policy.last_action[-1])}
            r.step_base()
        env_steps += r.t
        trunk = {'policy_noise_repeat_id': REPEAT, 'success': int(r.y), 'steps': r.t, 'reason': r.reason, 'policy_calls': r.policy.inference_calls, 'inference_calls': trunk_calls}
        trunk_ref = O._ref(r); traces['trunk'] = trunk_rows
        if snap is None:
            cells = [{'arm': a, 'no_trigger': True, 'y': int(r.y), 'steps': r.t, 'reason': r.reason} for a in arms]
            return O.jsonable({'scene': int(seed), 'role': role, 'arms': list(arms), 'initial_state': initial, 'trunk': trunk, 'triggered': False, 'trigger': None, 'cells': cells,
                               'counts': {'continuations': 0, 'restores': 0, 'env_steps': env_steps, 'readings': 0}, 'oracle_reads': 0, 'wall_seconds': time.perf_counter() - started, 'completed': True}), traces, frames
        s = snap['s']; t = snap['reading']['step']; before = snap['gt']; cells = []; readings = 0; reading_seconds = 0.
        for arm in arms:
            tick = time.perf_counter(); r.restore(s); env_steps += len(s['actions']); cmp = O.compare_tree(snap['tree'], O.runtime_tree(r), tol); live = O.obs_diff(r.replay_raw, s['raw'])
            if not cmp['pass'] or not O.live_obs_ok(live, tol): raise RuntimeError(f'scene {seed} trigger {t}: the restored runtime differs from the snapshot ({cmp["failed"]}, live observation {live})')
            if O.trigger_reading(r) != snap['reading'] or len(r.policy.queue) != snap['queue_len']: raise RuntimeError(f'scene {seed} trigger {t}: the trigger reading is not reproduced after the restore')
            h.rows = rows = []; h.calls = calls = []; reads0 = gate.reads; info = None; r.repeat = REPEAT
            if arm == 'VISION_CENTER':
                frames[arm] = fr = []; t0 = time.perf_counter(); info = vision_center(r, fr); reading_seconds += time.perf_counter() - t0; readings += len(info['readings'])
                if not info['acted']:                                           # abstaining must leave the whole runtime as captured: simulator state, queue contents, history, counters, RNG
                    kept = O.compare_tree(snap['tree'], O.runtime_tree(r), tol)
                    if not kept['pass'] or r.intervention is not None or len(r.policy.queue) != snap['queue_len'] or r.t != t or calls: raise RuntimeError(f'scene {seed} trigger {t}: an abstaining Vision-CENTER changed the runtime ({kept["failed"]})')
            elif arm == 'CENTER_ORACLE':
                O.center_oracle(r, gate); info = O.clone(r.retreat_info)
            k = r.t - t; acted = bool(info and (arm == 'CENTER_ORACLE' or info['acted'])); cc = len(calls); q = len(r.policy.queue); early = acted and bool(r.done); reads = gate.reads - reads0
            if acted and (cc or q): raise RuntimeError(f'scene {seed} trigger {t} {arm}: policy inference during control or a queue left at the hand-back')
            if (reads > 0) != (arm == 'CENTER_ORACLE'): raise RuntimeError(f'scene {seed} trigger {t} {arm}: ground-truth cube poses were read {reads} times for control')
            if acted and info['gripper_command'] != snap['last_grip']: raise RuntimeError(f'scene {seed} trigger {t} {arm}: the gripper command of the snapshot was not held')
            after = O.gt_reading(rows[k - 1], z0) if (probe and acted and k) else (before if acted else None); res = r.complete(); env_steps += r.t - t; same = None
            if not acted:                                                       # C0 and an abstaining Vision-CENTER are the uninterrupted run
                same = O._same_run(trunk_ref, r, tol)
                if not same['match']: raise RuntimeError(f'scene {seed} trigger {t}: {arm} without intervention does not reproduce the uninterrupted trunk ({O.jsonable(same)})')
            cells.append({'arm': arm, 'no_trigger': False, 'y': int(res['success']), 'steps': res['steps'], 'reason': res['reason'], 'acted': acted, 'intervention': res['intervention'], 'intervention_step': res['intervention_step'],
                          'control': info, 'control_steps': k if acted else 0, 'ended_during_control': bool(early), 'oracle_reads': reads, 'first_inference_step': calls[0]['t'] if calls else None,
                          'reproduces_trunk': same, 'gt_effect_of_control': O.control_effect(before, after) if (probe and acted) else None,
                          'gt_release_after_trigger': O.release_reading([trunk_rows[t - 1] if t else initial] + rows) if probe else None,
                          'gt_reading_error': _gt_error(info, before) if arm == 'VISION_CENTER' else None, 'wall_seconds': time.perf_counter() - tick})
            traces[arm] = rows
    g = snap['gt']
    trig = {**snap['reading'], 'queue_len': snap['queue_len'], 'remaining_budget': r.horizon - t, 'held_gripper_command': snap['last_grip'],
            'gt_for_diagnosis_only': None if g is None else {**g, 'false_trigger': not (g['grasped'] and (g['cubeA_lift_m'] or 0.) >= 0.01)}}
    rec = {'scene': int(seed), 'role': role, 'arms': list(arms), 'initial_state': initial, 'trunk': trunk, 'triggered': True, 'trigger': trig, 'cells': cells,
           'counts': {'continuations': len(arms), 'restores': len(arms), 'env_steps': env_steps, 'readings': readings, 'reading_seconds': reading_seconds}, 'oracle_reads': gate.reads,
           'wall_seconds': time.perf_counter() - started, 'completed': True}
    rec = O.jsonable(rec); validate_scene(rec, c); return rec, traces, frames


def validate_scene(rec, c=None):
    from . import recovery_v1_oracle as O
    cells = rec['cells']; by = {x['arm']: x for x in cells}
    if [x['arm'] for x in cells] != list(rec['arms']) or rec['arms'][0] != 'C0': raise ValueError('a scene needs its declared arms, once each, C0 first')
    if c is not None and any(x['steps'] > c['env']['horizon'] for x in cells): raise ValueError('an arm exceeded the episode horizon')
    if not rec['triggered']:
        if any(not x['no_trigger'] or (x['y'], x['steps']) != (rec['trunk']['success'], rec['trunk']['steps']) for x in cells) or rec['oracle_reads']: raise ValueError('a scene without a trigger must carry the trunk outcome in every arm')
        return True
    x = by['C0']
    if (x['y'], x['steps']) != (rec['trunk']['success'], rec['trunk']['steps']) or x['acted'] or x['oracle_reads'] or not x['reproduces_trunk']['match']: raise ValueError('C0 does not reproduce the uninterrupted trunk')
    v = by.get('VISION_CENTER')
    if v is not None:
        ctl = v['control']
        if v['oracle_reads']: raise ValueError('ground truth reached Vision-CENTER')
        if ctl['acted'] != v['acted'] or (ctl['decision'] == 'ACT') != v['acted']: raise ValueError('Vision-CENTER decision and action disagree')
        if not v['acted'] and ((v['y'], v['steps']) != (x['y'], x['steps']) or v['intervention'] is not None): raise ValueError('an abstaining Vision-CENTER differs from C0')
        if v['acted']:
            if v['intervention'] != INTERVENTION or ctl['steps_used'] > O.MAX_STEPS or len(ctl['steps']) != ctl['steps_used'] or ctl['gripper_command'] != rec['trigger']['held_gripper_command']: raise ValueError('Vision-CENTER left its step bound or its record is inconsistent')
            if any(np.hypot(*s['commanded_xy_m']) > O.STEP_CMD_M + 1e-9 or s['target_distance_from_start_m'] > O.TARGET_CAP_M + 1e-9 for s in ctl['steps']): raise ValueError('a Vision-CENTER command exceeded a bound')
            if ctl['readings'][0]['offset_norm_m'] < ACT_MIN_M: raise ValueError('Vision-CENTER acted below its own threshold')
    o = by.get('CENTER_ORACLE')
    if o is not None and (not o['acted'] or o['oracle_reads'] < 1 or o['control']['steps_used'] > O.MAX_STEPS): raise ValueError('CENTER_ORACLE record is inconsistent')
    return True


def shadow_verdict(records):
    """Stage B1 on USED scenes: Vision-CENTER against C0 with ground truth as the after-the-fact label. Criteria fixed in the plan."""
    from . import recovery_v1_oracle as O
    trig = [r for r in records if r['triggered']]; cell = lambda r, a: next(x for x in r['cells'] if x['arm'] == a); v = [cell(r, 'VISION_CENTER') for r in trig]
    acts = [x for x in v if x['acted']]; err = [x['gt_reading_error']['offset_error_m'] for x in acts if x['gt_reading_error'] and x['gt_reading_error']['offset_error_m'] is not None]
    y0 = {r['scene']: cell(r, 'C0')['y'] for r in records}; y1 = {r['scene']: cell(r, 'VISION_CENTER')['y'] for r in records}; p = O.paired(y0, y1)
    m = {'scenes': len(records), 'triggered': len(trig), 'decisions': {k: sum(1 for x in v if x['control']['decision'] == k) for k in ('ACT',) + ABSTAIN}, 'acts': len(acts),
         'unnecessary_acts': sum(1 for x in acts if x['gt_reading_error'] and x['gt_reading_error']['unnecessary_act']), 'missed_large_offsets': sum(1 for x in v if x['gt_reading_error'] and x['gt_reading_error']['missed_large_offset']),
         'offset_error_at_act_m': {'n': len(err), 'median': float(np.median(err)) if err else None, 'max': float(np.max(err)) if err else None}, 'paired_VISION_CENTER_vs_C0': p,
         'end_reasons': {k: sum(1 for x in acts if x['control']['end_reason'] == k) for k in sorted({x['control']['end_reason'] for x in acts})}}
    k = SHADOW_PASS; crit = {'acts_at_least': len(acts) >= k['acts_min'], 'unnecessary_acts_at_most': m['unnecessary_acts'] <= k['unnecessary_acts_max'], 'harm_at_most': p['harm'] <= k['harm_max'],
                             'rescue_minus_harm_at_least': p['net'] >= k['rescue_minus_harm_min'], 'median_error_at_act': bool(err) and float(np.median(err)) <= k['offset_error_at_act_median_max_m'],
                             'max_error_at_act': bool(err) and float(np.max(err)) <= k['offset_error_at_act_max_m']}
    return {**m, 'criteria': crit, 'pass': bool(all(crit.values())), 'thresholds': dict(k), 'label': 'SHADOW_PASS' if all(crit.values()) else 'SHADOW_BLOCKED',
            'scope': 'used development scenes; decides only whether the new scenes are run. A failure stops the work: no constant is tuned and re-tried under this plan'}


def summarize(records, planned):
    """The pilot on new scenes: all planned scenes in every denominator."""
    from . import recovery_v1_oracle as O
    by = {r['scene']: r for r in records}
    if sorted(by) != sorted(int(s) for s in planned) or len(by) != len(records): raise ValueError('the records are not exactly the planned scenes; a partial run is not summarised')
    cell = lambda r, a: next(x for x in r['cells'] if x['arm'] == a); y = {a: {s: cell(r, a)['y'] for s, r in by.items()} for a in ARMS}; trig = [r for r in records if r['triggered']]
    v = [cell(r, 'VISION_CENTER') for r in trig]; acts = [x for x in v if x['acted']]; pv = O.paired(y['C0'], y['VISION_CENTER']); po = O.paired(y['C0'], y['CENTER_ORACLE'])
    err = [x['gt_reading_error']['offset_error_m'] for x in v if x['gt_reading_error'] and x['gt_reading_error']['offset_error_m'] is not None]
    stat = lambda a: {'n': len(a), 'min': float(np.min(a)), 'median': float(np.median(a)), 'max': float(np.max(a))} if a else {'n': 0}
    g = {'rescue_at_least_4': pv['rescue'] >= GATE['rescue_min'], 'net_at_least_3': pv['net'] >= GATE['net_min'], 'harm_at_most_3': pv['harm'] <= GATE['harm_max']}
    return {'role': ROLE, 'unit': 'scene', 'scenes': len(records), 'denominators': {'all': len(records), 'no_trigger': len(records) - len(trig), 'triggered': len(trig), 'vision_acted': len(acts), 'vision_abstained': len(v) - len(acts),
                                                                                    'c0_success': sum(y['C0'].values()), 'c0_failure': len(records) - sum(y['C0'].values())},
            'success': {a: {'k': sum(y[a].values()), 'n': len(records), 'wilson95': O.wilson(sum(y[a].values()), len(records))} for a in ARMS},
            'primary_VISION_CENTER_vs_C0': pv, 'reference_CENTER_ORACLE_vs_C0': po, 'VISION_CENTER_vs_CENTER_ORACLE': O.paired(y['CENTER_ORACLE'], y['VISION_CENTER']),
            'oracle_rescues_also_rescued_by_vision': {'both': len(set(pv['rescue_scenes']) & set(po['rescue_scenes'])), 'oracle': po['rescue']},
            'vision': {'decisions': {k: sum(1 for x in v if x['control']['decision'] == k) for k in ('ACT',) + ABSTAIN}, 'end_reasons': {k: sum(1 for x in acts if x['control']['end_reason'] == k) for k in sorted({x['control']['end_reason'] for x in acts})},
                       'unnecessary_acts': sum(1 for x in acts if x['gt_reading_error'] and x['gt_reading_error']['unnecessary_act']), 'missed_large_offsets': sum(1 for x in v if x['gt_reading_error'] and x['gt_reading_error']['missed_large_offset']),
                       'offset_error_at_trigger_m': stat(err), 'control_steps': stat([x['control_steps'] for x in acts]), 'tcp_xy_displacement_m': stat([x['control']['tcp_xy_displacement_m'] for x in acts]),
                       'target_cube_pushed_more_than_1mm': sum(1 for x in acts if x['gt_effect_of_control'] and x['gt_effect_of_control']['cubeB_xy_displacement_m'] > 0.001)},
            'followup_gate': {'criteria': g, 'label': 'VISION_GATE_PASS' if all(g.values()) else 'NO_VISION_GATE_PASS', 'thresholds': dict(GATE), 'scope': 'engineering gate, not a significance test'},
            'scope': ['One task, one policy, one noise repeat, 100 development scenes, simulator images with flat colours and exact camera parameters. No significance claim.',
                      'Vision-CENTER uses RGB and robot state only; ground truth appears only in after-the-fact error columns.', 'CENTER_ORACLE is the non-deployable reference of the earlier pilot, unchanged.']}


def projected_cost(*, seconds_per_env_step, seconds_per_reading, usd_per_hour, spent_usd, cap_usd, scenes, arms, fixed_seconds=600, reserve_usd=0.20, margin=1.25):
    """Worst case: every scene at the step bound and every Vision-CENTER control part at its 12 steps. Inputs are time and money only."""
    steps = 200 + arms * 200; hours = (scenes * (steps * seconds_per_env_step + READINGS_PER_SCENE_BOUND * seconds_per_reading) + fixed_seconds) * margin / 3600; total = spent_usd + hours * usd_per_hour + reserve_usd
    return {'projected_remaining_hours': hours, 'projected_total_usd': total, 'cap_usd': cap_usd, 'fits_cap': bool(total <= cap_usd),
            'inputs': {'seconds_per_env_step': seconds_per_env_step, 'seconds_per_reading': seconds_per_reading, 'usd_per_hour': usd_per_hour, 'spent_usd': spent_usd, 'scenes': scenes, 'arms': arms,
                       'steps_per_scene_bound': steps, 'readings_per_scene_bound': READINGS_PER_SCENE_BOUND, 'fixed_seconds': fixed_seconds, 'reserve_usd': reserve_usd, 'margin': margin},
            'rule': 'if the projection exceeds the cap the stage does not start; scenes, arms and bounds are not reduced'}
