"""Exact geometry for the C1 spatial occupancy audit (pure numpy, no simulator).

Every shape here is the simulator's own COLLISION geometry, read from the pinned upstream source (ManiSkill a4a4f92,
`assets/upstream_geometry/panda_hand_hull.json` lists the files and their sha256):

  cubes      boxes, half size 0.02 m                                   (envs/tasks/tabletop/stack_cube.py)
  fingers    four boxes per finger in the finger-link frame           (assets/robots/panda/panda_v2.urdf, panda_leftfinger /
                                                                        panda_rightfinger <collision>)
  hand       convex hull of collision/hand.stl (PhysX collides with the convex hull of a mesh)
  table      one box, top face at z = 0                               (utils/scene_builder/table/scene_builder.py)

Kinematics from the logged values only: the hand frame is the logged TCP pose moved 0.1034 m back along its own z
(panda_hand_tcp_joint, no rotation); each finger link is the hand frame moved 0.0584 m along z and +/- its logged joint
value along y (panda_finger_joint1 / 2). The arm links (panda_link0..7) need the arm joint angles, which the trace does
not record, so they are NOT represented here.

Quaternions are (w, x, y, z), as SAPIEN reports them. Units: metres.
Signed distance: > 0 the exact gap between two convex shapes, <= 0 overlap (minus the penetration depth from the
separating-axis test). Distances are evaluated at the logged poses only, i.e. once per control step.
"""
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HULL_FILE = ROOT / 'assets' / 'upstream_geometry' / 'panda_hand_hull.json'
UPSTREAM_COMMIT = 'a4a4f9272ad64b1564035874b605ceb687b63ed8'

CUBE_HALF_M = 0.02
HAND_TO_TCP_Z_M = 0.1034
FINGER_JOINT_Z_M = 0.0584
# (name, origin in the LEFT finger frame, roll about x (rad), full size). The right finger mirrors y and roll.
FINGER_BOXES = (('screw_mount', (0., 18.5e-3, 11e-3), 0., (22e-3, 15e-3, 20e-3)),
                ('carriage_sledge', (0., 6.8e-3, 2.2e-3), 0., (22e-3, 8.8e-3, 3.8e-3)),
                ('diagonal_finger', (0., 15.9e-3, 28.35e-3), math.pi / 6, (17.5e-3, 7e-3, 23.5e-3)),
                ('rubber_tip', (0., 7.58e-3, 45.25e-3), 0., (17.5e-3, 15.2e-3, 18.5e-3)))
TABLE_CENTER = (-0.12, 0., -0.9196429 / 2); TABLE_HALF = (1.209 / 2, 2.418 / 2, 0.9196429 / 2)   # actor rotated 90 deg about z
ROBOT_BASE_XYZ = (-0.615, 0., 0.)
SIM_STEPS_PER_CONTROL = 5                                            # sim_freq 100 / control_freq 20 (utils/structs/types.py)


# ---------------------------------------------------------------- rotations
def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / (np.linalg.norm(q) or 1.)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_x(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
def rot_z(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def slerp(q0, q1, f):
    q0 = np.asarray(q0, float) / np.linalg.norm(q0); q1 = np.asarray(q1, float) / np.linalg.norm(q1); d = float(q0 @ q1)
    if d < 0: q1, d = -q1, -d
    if d > 0.9995: q = q0 + f * (q1 - q0); return q / np.linalg.norm(q)
    th = math.acos(d); return (math.sin((1 - f) * th) * q0 + math.sin(f * th) * q1) / math.sin(th)


def rotation_angle_deg(q0, q1):
    """Angle of the rotation that takes orientation q0 to q1."""
    d = abs(float(np.dot(np.asarray(q0, float) / np.linalg.norm(q0), np.asarray(q1, float) / np.linalg.norm(q1))))
    return math.degrees(2 * math.acos(min(1., d)))


# ---------------------------------------------------------------- convex shapes
_BOX_TRI = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)   # index = 4*ix + 2*iy + iz


class Convex:
    """A convex polytope: world vertices V, outward triangles T, unique edges E, and the direction sets used by the
    separating-axis test (unique face normals N, unique edge directions D). Derived sets are computed once per local
    shape and rotated with it."""
    __slots__ = ('V', 'T', 'E', 'N', 'D', 'name')

    def __init__(self, V, T, name='', _derived=None):
        self.V = np.asarray(V, float); self.name = name
        if _derived is not None: self.T, self.E, self.N, self.D = _derived; return
        T = np.array(T, int); P = self.V[T]; n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        inward = np.einsum('ij,ij->i', n, P[:, 0] - self.V.mean(0)) < 0; T[inward] = T[inward][:, [0, 2, 1]]      # outward (convex)
        self.T = T; e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1); self.E = np.unique(e, axis=0)
        P = self.V[T]; self.N = _unique_dirs(_unit(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])))
        self.D = _unique_dirs(_unit(self.V[self.E[:, 1]] - self.V[self.E[:, 0]]))

    def moved(self, R, p, name=None):
        R = np.asarray(R, float)
        return Convex(self.V @ R.T + np.asarray(p, float), None, self.name if name is None else name, (self.T, self.E, self.N @ R.T, self.D @ R.T))

    def volume(self):
        P = self.V[self.T]; return float(abs(np.einsum('ij,ij->i', P[:, 0], np.cross(P[:, 1], P[:, 2])).sum()) / 6)

    def aabb(self): return self.V.min(0), self.V.max(0)


def _unique_dirs(n, tol=1e-9):
    """Unit vectors up to sign, duplicates removed (for separating-axis candidates)."""
    out = []
    for v in n:
        if not out or np.min(1 - np.abs(np.asarray(out) @ v)) > tol: out.append(v)
    return np.asarray(out).reshape(-1, 3)


_LOCAL_BOX = {}


def box(center, R, half, name=''):
    key = tuple(float(h) for h in half)
    if key not in _LOCAL_BOX: _LOCAL_BOX[key] = Convex(_SIGNS * np.asarray(key), _BOX_TRI)
    return _LOCAL_BOX[key].moved(R, center, name)


def cube(p, q, name='cube'): return box(p, quat_to_R(q), (CUBE_HALF_M,) * 3, name)


def table(): return box(TABLE_CENTER, np.eye(3), TABLE_HALF, 'table')


_HULL = None


def hand_hull_local():
    global _HULL
    if _HULL is None:
        d = json.loads(HULL_FILE.read_text(encoding='utf-8')); _HULL = Convex(d['vertices'], d['triangles'], 'hand')
    return _HULL


def gripper(tcp_p, tcp_q, finger_qpos):
    """Collision shapes of the hand and both fingers at a logged TCP pose and finger joint values. {name: Convex}"""
    R = quat_to_R(tcp_q); hand_p = np.asarray(tcp_p, float) - R @ np.array([0., 0., HAND_TO_TCP_Z_M]); out = {'hand': hand_hull_local().moved(R, hand_p)}
    for side, sgn, qj in (('left', 1., finger_qpos[0]), ('right', -1., finger_qpos[1])):
        f_p = hand_p + R @ np.array([0., sgn * qj, FINGER_JOINT_Z_M])
        for name, o, roll, size in FINGER_BOXES:
            o = np.array([o[0], sgn * o[1], o[2]]); Rl = R @ rot_x(sgn * roll)
            out[f'{side}_{name}'] = box(f_p + R @ o, Rl, np.asarray(size) / 2, f'{side}_{name}')
    return out


def finger_shapes(g, side): return [v for k, v in g.items() if k.startswith(side + '_')]


# ---------------------------------------------------------------- distances
def point_triangle(P, A, B, C):
    """Distance from every point P (n,3) to every triangle (A,B,C each (m,3)); (n,m). Ericson, Real-Time Collision Detection 5.1.5."""
    P = P[:, None, :]; A = A[None]; B = B[None]; C = C[None]
    ab = B - A; ac = C - A; ap = P - A; d1 = (ab * ap).sum(-1); d2 = (ac * ap).sum(-1)
    bp = P - B; d3 = (ab * bp).sum(-1); d4 = (ac * bp).sum(-1); cp = P - C; d5 = (ab * cp).sum(-1); d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) > 1e-300, va + vb + vc, 1e-300); v = vb / den; w = vc / den
    Q = A + ab * v[..., None] + ac * w[..., None]                                                     # interior
    def put(mask, X):
        nonlocal Q; Q = np.where(mask[..., None], X, Q)
    # edge regions (later assignments take priority, so vertices go last)
    den = (d4 - d3) + (d5 - d6); t = (d4 - d3) / np.where(np.abs(den) > 1e-300, den, 1e-300)
    put((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), B + (C - B) * t[..., None])
    t = d2 / np.where(np.abs(d2 - d6) > 1e-300, d2 - d6, 1e-300); put((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + ac * t[..., None])
    t = d1 / np.where(np.abs(d1 - d3) > 1e-300, d1 - d3, 1e-300); put((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + ab * t[..., None])
    put((d6 >= 0) & (d5 <= d6), np.broadcast_to(C, Q.shape)); put((d3 >= 0) & (d4 <= d3), np.broadcast_to(B, Q.shape)); put((d1 <= 0) & (d2 <= 0), np.broadcast_to(A, Q.shape))
    return np.linalg.norm(P - Q, axis=-1)


def segment_segment(P1, Q1, P2, Q2):
    """Distance between every segment P1Q1 (n,3) and every segment P2Q2 (m,3); (n,m). Ericson 5.1.9."""
    P1 = P1[:, None]; Q1 = Q1[:, None]; P2 = P2[None]; Q2 = Q2[None]
    d1 = Q1 - P1; d2 = Q2 - P2; r = P1 - P2; a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1); b = (d1 * d2).sum(-1); den = a * e - b * b
    a = np.broadcast_to(a, den.shape); e = np.broadcast_to(e, den.shape)
    s = np.where(den > 1e-18 * np.maximum(a * e, 1e-300), np.clip((b * f - c * e) / np.where(den > 0, den, 1.), 0, 1), 0.)
    t = (b * s + f) / np.where(e > 0, e, 1.)
    s = np.where(t < 0, np.clip(-c / np.where(a > 0, a, 1.), 0, 1), np.where(t > 1, np.clip((b - c) / np.where(a > 0, a, 1.), 0, 1), s)); t = np.clip(t, 0, 1)
    X = P1 + d1 * s[..., None]; Y = P2 + d2 * t[..., None]
    return np.linalg.norm(X - Y, axis=-1)


def _interval(V, axes): pr = V @ axes.T; return pr.min(0), pr.max(0)


def sat_overlap(A, B, return_axis=False):
    """Separating-axis test over both shapes' face normals and all edge-direction cross products.
    Returns the smallest interval overlap (< 0: separated along that axis; >= 0 on every axis: overlapping, value =
    penetration depth); with return_axis, also the unit axis that gives it, oriented from A to B."""
    cr = np.cross(A.D[:, None], B.D[None]).reshape(-1, 3); nn = np.linalg.norm(cr, axis=1); cr = cr[nn > 1e-9] / nn[nn > 1e-9, None]
    axes = np.concatenate([A.N, B.N, cr]); la, ha = _interval(A.V, axes); lb, hb = _interval(B.V, axes)
    ov = np.minimum(ha - lb, hb - la); i = int(np.argmin(ov))
    if not return_axis: return float(ov[i])
    n = axes[i] if (ha[i] - lb[i]) <= (hb[i] - la[i]) else -axes[i]
    return float(ov[i]), n


def _unit(v): return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def separation(A, B):
    """Exact distance between two separated convex polytopes: the minimum over vertex-face pairs both ways and edge-edge pairs."""
    da = point_triangle(A.V, B.V[B.T[:, 0]], B.V[B.T[:, 1]], B.V[B.T[:, 2]]).min()
    db = point_triangle(B.V, A.V[A.T[:, 0]], A.V[A.T[:, 1]], A.V[A.T[:, 2]]).min()
    de = segment_segment(A.V[A.E[:, 0]], A.V[A.E[:, 1]], B.V[B.E[:, 0]], B.V[B.E[:, 1]]).min()
    return float(min(da, db, de))


def signed_distance(A, B):
    ov = sat_overlap(A, B)
    return -ov if ov >= 0 else separation(A, B)


def signed_distance_many(As, B):
    return min(signed_distance(a, B) for a in As)


def aabb_gap(A, B):
    (la, ha), (lb, hb) = A.aabb(), B.aabb(); g = np.maximum(0, np.maximum(lb - ha, la - hb)); return float(np.linalg.norm(g))


# ---------------------------------------------------------------- 2D footprints (XY)
def hull2d(P):
    """Convex hull, counter-clockwise, of 2D points (Andrew's monotone chain)."""
    P = sorted(set(map(tuple, np.round(np.asarray(P, float)[:, :2], 12))))
    if len(P) <= 2: return np.asarray(P)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0: hi.pop()
        hi.append(p)
    return np.asarray(lo[:-1] + hi[:-1])


def footprint(shape): return hull2d(shape.V[:, :2])


def footprint_many(shapes): return hull2d(np.concatenate([s.V[:, :2] for s in shapes]))


def polygon_area(P):
    P = np.asarray(P, float)
    if len(P) < 3: return 0.
    x, y = P[:, 0], P[:, 1]; return float(0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)))


def clip_polygon(S, C):
    """Intersection of convex polygon S with convex polygon C (both counter-clockwise), Sutherland-Hodgman."""
    out = [tuple(p) for p in S]
    for i in range(len(C)):
        a, b = C[i], C[(i + 1) % len(C)]; inp = out; out = []
        if not inp: break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]; sp, sq = side(p), side(q)
            if sp >= 0: out.append(p)
            if (sp >= 0) != (sq >= 0):
                f = sp / (sp - sq); out.append((p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])))
    return np.asarray(out).reshape(-1, 2)


def overlap_area(P, Q): return polygon_area(clip_polygon(P, Q))


def inside_polygon(X, P):
    """Points X (n,2) inside the convex counter-clockwise polygon P."""
    P = np.asarray(P, float); e = np.roll(P, -1, 0) - P
    return np.all(e[:, 0] * (X[:, None, 1] - P[:, 1]) - e[:, 1] * (X[:, None, 0] - P[:, 0]) >= 0, axis=1)


def union_overlap_area(polys, Q, res=2.5e-4):
    """Area of convex polygon Q covered by the UNION of convex polygons `polys` (raster of Q at `res`; error about
    perimeter x res / 2). The union is not replaced by its convex hull, so the open space between two fingers is not counted."""
    Q = np.asarray(Q, float); lo = Q.min(0); hi = Q.max(0); xs = np.arange(lo[0] + res / 2, hi[0], res); ys = np.arange(lo[1] + res / 2, hi[1], res)
    X = np.stack(np.meshgrid(xs, ys, indexing='ij'), -1).reshape(-1, 2); X = X[inside_polygon(X, Q)]
    if not len(X): return 0.
    hit = np.zeros(len(X), bool)
    for P in polys:
        if len(P) >= 3: hit |= inside_polygon(X, P)
    return float(hit.sum() * res * res)


def union_gap(polys, Q):
    """Distance from convex polygon Q to the union of convex polygons (0 if any overlaps)."""
    return float(min(polygon_gap(P, Q) for P in polys))


def _pt_seg2(P, A, B):
    P = P[:, None]; A = A[None]; B = B[None]; d = B - A; t = np.clip(((P - A) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-300), 0, 1)
    return np.linalg.norm(P - (A + d * t[..., None]), axis=-1)


def polygon_gap(P, Q):
    """Distance between two convex polygons; 0 if they overlap or touch."""
    if overlap_area(P, Q) > 0: return 0.
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    return float(min(_pt_seg2(P, Q, np.roll(Q, -1, 0)).min(), _pt_seg2(Q, P, np.roll(P, -1, 0)).min()))


# ---------------------------------------------------------------- swept occupancy (voxel approximation)
def inside(shape, X, tol=0.):
    """Points X (n,3) inside a convex polytope (all outward face planes)."""
    P = shape.V[shape.T]; n = _unit(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])); off = np.einsum('ij,ij->i', n, P[:, 0])
    return np.all(X @ n.T - off <= tol, axis=1)


def swept_voxels(shape_seq, voxel=0.002):
    """Union of the voxels whose centre lies inside any shape of `shape_seq` (shapes along a path, sampled finely).
    Returns (set of integer voxel indices, voxel size). Volume = len(set) * voxel**3. Approximation: discretisation error
    is about (surface area * voxel / 2) per shape; tested against closed-form swept volumes of translating boxes."""
    occ = set()
    for s in shape_seq:
        lo, hi = s.aabb(); i0 = np.floor(lo / voxel).astype(int); i1 = np.ceil(hi / voxel).astype(int)
        g = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(i0, i1)], indexing='ij'), -1).reshape(-1, 3)
        if not len(g): continue
        m = inside(s, (g + 0.5) * voxel); occ.update(map(tuple, g[m]))
    return occ, voxel


def swept_volume(shape_seq, voxel=0.002):
    occ, v = swept_voxels(shape_seq, voxel); return len(occ) * v ** 3


def interpolate_poses(p0, q0, p1, q1, n=SIM_STEPS_PER_CONTROL):
    """Linear positions and slerped orientations at n sub-steps from pose 0 (excluded) to pose 1 (included). An
    ASSUMPTION used only for swept occupancy and potential-collision screening, never reported as observed."""
    return [((1 - f) * np.asarray(p0, float) + f * np.asarray(p1, float), slerp(q0, q1, f)) for f in np.arange(1, n + 1) / n]
