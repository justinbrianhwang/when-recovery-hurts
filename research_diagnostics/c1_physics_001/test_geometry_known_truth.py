"""Known-truth geometry tests for hci.spatial_geometry / hci.spatial_links (POST_HOC_EXPLORATORY verification).

Every expected value below is derived by hand in closed form (comments give the derivation); none is produced by
hci.spatial_geometry.signed_distance. Each case is evaluated by (1) the existing code path and (2) the independent
reference (geometry_reference: GJK / Qhull Minkowski facets / scipy rotations), and both are compared with the truth.

Tolerance: TOL_M = 1e-9 m. The inputs are float64 numbers of order 0.1 m, so one rounding is about 1e-17 m; the
algorithms do O(10^2) arithmetic operations plus a least-squares solve (reference GJK), which keeps the accumulated error
well below 1e-12 m. 1e-9 m leaves three orders of margin and is six orders below the 1 mm S6 tolerance, with which it has
nothing to do. Cases built from float32-logged C1 data use the cross-path tolerance TOL_CROSS_M (see run script).
"""
from __future__ import annotations

import math, sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402  (adds src to sys.path)
import geometry_reference as REF  # noqa: E402
from hci import spatial_geometry as G  # noqa: E402
from hci import spatial_links as SL  # noqa: E402

TOL_M = 1e-9
h = 0.02                       # cube half size of the task (0.02 m)
S2 = math.sqrt(2.0)


def rz(a): return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.]])
def ry(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1., 0], [-math.sin(a), 0, math.cos(a)]])
def rx(a): return np.array([[1., 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def case(cid, group, desc, truth, hA, RA, pA, hB, RB, pB):
    return dict(case=cid, group=group, description=desc, truth_m=truth, A=(np.asarray(hA, float), np.asarray(RA, float), np.asarray(pA, float)),
                B=(np.asarray(hB, float), np.asarray(RB, float), np.asarray(pB, float)))


def known_truth_cases():
    I = np.eye(3); C = []; hh = (h, h, h)
    # G1: two axis-aligned equal cubes, B shifted along +x by 2h + g: face-face gap g
    for g in (0.0, 0.0005, 0.001, 0.005):
        C.append(case(f'G1_x_gap_{g*1e3:g}mm', 'G1 axis-aligned gap', f'face-face along x, gap {g*1e3:g} mm', g, hh, I, (0, 0, 0), hh, I, (2 * h + g, 0, 0)))
    for g in (0.0005, 0.001):
        C.append(case(f'G1_z_gap_{g*1e3:g}mm', 'G1 axis-aligned gap', f'stacked along z, gap {g*1e3:g} mm, lateral shift 7 mm (faces still overlap)', g, hh, I, (0, 0, 0), hh, I, (0.007, -0.003, 2 * h + g)))
    # G1b: edge-edge (parallel vertical edges): offsets gx, gy beyond contact -> sqrt(gx^2 + gy^2); corner-corner -> g*sqrt(3)
    C.append(case('G1_edge_parallel', 'G1 axis-aligned gap', 'parallel vertical edges, gx=3 mm, gy=4 mm -> 5 mm', 0.005, hh, I, (0, 0, 0), hh, I, (2 * h + 0.003, 2 * h + 0.004, 0)))
    C.append(case('G1_corner_corner', 'G1 axis-aligned gap', 'corner to corner, 1 mm on each axis -> sqrt(3) mm', 0.001 * math.sqrt(3), hh, I, (0, 0, 0), hh, I, (2 * h + 0.001,) * 3))
    # G2: axis-aligned penetration p along x (overlap on y, z is 2h = 40 mm > p): minimum translation p
    for p in (0.0005, 0.001, 0.003):
        C.append(case(f'G2_x_pen_{p*1e3:g}mm', 'G2 axis-aligned penetration', f'overlap {p*1e3:g} mm along x', -p, hh, I, (0, 0, 0), hh, I, (2 * h - p, 0.002, -0.001)))
    C.append(case('G2_unequal_boxes_pen', 'G2 axis-aligned penetration', 'boxes 20x30x10 mm and 40 mm cube, overlap 2 mm along z (z is the smallest overlap)',
                  -0.002, (0.01, 0.015, 0.005), I, (0, 0, 0), hh, I, (0.003, 0.004, 0.005 + h - 0.002)))
    # G3: rotated about z by theta: the leftmost feature of B is a vertical edge at x_c - h (cos t + sin t); A's face at x = h
    for deg in (30.0, 45.0):
        t = math.radians(deg); g = 0.002; xc = h + h * (math.cos(t) + math.sin(t)) + g
        C.append(case(f'G3_edge_face_rot{deg:g}', 'G3 face-edge', f'B rotated {deg:g} deg about z, vertical edge to face, gap 2 mm', g, hh, I, (0, 0, 0), hh, rz(t), (xc, 0, 0)))
    # G3b: parallel faces of two boxes rotated together by the same rotation: gap invariant
    Rr = rz(0.3) @ ry(-0.4) @ rx(0.2); n = Rr[:, 0]
    C.append(case('G3_rotated_parallel_faces', 'G3 face-edge', 'both boxes rotated by the same rotation, B shifted 2h + 1.5 mm along the rotated x axis',
                  0.0015, hh, Rr, (0.1, -0.05, 0.2), hh, Rr, np.array((0.1, -0.05, 0.2)) + (2 * h + 0.0015) * n))
    # G4: crossing edges. A rotated 45 deg about z: edge parallel to z at x = h sqrt2 (y = 0). B rotated 45 deg about y at
    # (c, 0, 0): edge parallel to y at x = c - h sqrt2 (z = 0). The edges are perpendicular and cross in projection: d = c - 2 h sqrt2
    c = 2 * h * S2 + 0.001
    C.append(case('G4_edge_edge_crossing', 'G4 rotated', 'perpendicular crossing edges, 1 mm', 0.001, hh, rz(math.pi / 4), (0, 0, 0), hh, ry(math.pi / 4), (c, 0, 0)))
    # G4b: rotated penetration: A rotated 45 deg about z pokes its vertical edge (x = h sqrt2) into B's face x = c - h:
    # overlap along x is h sqrt2 + h - c; every other axis overlaps by much more for these values (checked: >= 5 mm)
    for p in (0.0005, 0.002):
        c = h * S2 + h - p
        C.append(case(f'G4_rotated_pen_{p*1e3:g}mm', 'G4 rotated', f'45 deg edge into a face, penetration {p*1e3:g} mm', -p, hh, rz(math.pi / 4), (0, 0, 0), hh, np.eye(3), (c, 0, 0)))
    # G4c: same crossing-edge configuration moved and rotated rigidly (whole scene): invariance of the distance
    Rw = rz(1.1) @ rx(0.7); pw = np.array([-0.3, 0.2, 0.05]); c = 2 * h * S2 + 0.0007
    C.append(case('G4_edge_edge_rigid_scene', 'G4 rotated', 'G4 crossing edges, whole scene rotated and moved, 0.7 mm', 0.0007,
                  hh, Rw @ rz(math.pi / 4), pw, hh, Rw @ ry(math.pi / 4), pw + Rw @ np.array([c, 0, 0])))
    return C


def existing(c):
    (hA, RA, pA), (hB, RB, pB) = c['A'], c['B']
    return G.signed_distance(G.box(pA, RA, hA), G.box(pB, RB, hB))


def reference(c):
    (hA, RA, pA), (hB, RB, pB) = c['A'], c['B']
    return REF.signed_distance_ref(REF.box_vertices(hA, RA, pA), REF.box_vertices(hB, RB, pB))


CASES = known_truth_cases()


@pytest.mark.parametrize('c', CASES, ids=[c['case'] for c in CASES])
def test_existing_matches_truth(c):
    assert abs(existing(c) - c['truth_m']) <= TOL_M


@pytest.mark.parametrize('c', CASES, ids=[c['case'] for c in CASES])
def test_reference_matches_truth(c):
    d, _ = reference(c); assert abs(d - c['truth_m']) <= TOL_M


def test_sign_convention():
    """Positive = gap, negative = penetration, ~0 = touching, for both paths."""
    for c in CASES:
        e = existing(c); r, _ = reference(c)
        if c['truth_m'] > TOL_M: assert e > 0 and r > 0
        elif c['truth_m'] < -TOL_M: assert e < 0 and r < 0
        else: assert abs(e) <= TOL_M and abs(r) <= TOL_M


def test_gjk_witness_points():
    # face-face gap 1 mm along x: witness points 1 mm apart along x, on the two faces
    VA = REF.box_vertices((h,) * 3); VB = REF.box_vertices((h,) * 3, np.eye(3), (2 * h + 0.001, 0, 0))
    d, pa, pb, ov = REF.gjk_distance(VA, VB)
    assert not ov and abs(d - 0.001) < TOL_M and abs(pa[0] - h) < TOL_M and abs(pb[0] - (h + 0.001)) < TOL_M and np.allclose(pb - pa, [0.001, 0, 0], atol=TOL_M)


def test_quaternion_order_wxyz():
    """SAPIEN quaternions are (w, x, y, z). 90 deg about z: (cos 45, 0, 0, sin 45) maps x to y."""
    q = (math.cos(math.pi / 4), 0., 0., math.sin(math.pi / 4))
    assert np.allclose(G.quat_to_R(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert np.allclose(REF.R_wxyz(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    # the same four numbers read as (x, y, z, w) would be a different rotation: the order matters and is detectable
    assert not np.allclose(G.quat_to_R((q[1], q[2], q[3], q[0])) @ [1, 0, 0], [0, 1, 0], atol=1e-3)


def test_local_pose_times_body_pose_and_shape_index():
    """spatial_links.posed: world = body (R_b, p_b) * local (R_l, p_l), shapes named body#index in the logged order.
    Body at (0.1, 0, 0.05) rotated 90 deg about z; shape #1 offset (0.03, 0, 0) in the body frame -> world centre
    (0.1, 0.03, 0.05) by hand. Shape #0 sits at the body origin."""
    q90 = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    geo = {'body': {'shapes': [{'type': 'PhysxCollisionShapeBox', 'local_pose': [0, 0, 0, 1, 0, 0, 0], 'half_size': [0.01, 0.01, 0.01]},
                               {'type': 'PhysxCollisionShapeBox', 'local_pose': [0.03, 0, 0, 1, 0, 0, 0], 'half_size': [0.005, 0.01, 0.002]}]}}
    S = SL.posed(geo, 'body', [0.1, 0, 0.05] + q90)
    assert [s.name for s in S] == ['body#0', 'body#1']
    assert np.allclose(S[1].V.mean(0), [0.1, 0.03, 0.05], atol=1e-12)
    # the long half-size (0.01, local y) becomes world x after 90 deg about z: extent in x is 0.02, in y 0.01
    ext = S[1].V.max(0) - S[1].V.min(0); assert np.allclose(ext, [0.02, 0.01, 0.004], atol=1e-12)
    Vref = REF.pose_vertices(REF.box_vertices([0.005, 0.01, 0.002]), [0.03, 0, 0, 1, 0, 0, 0], [0.1, 0, 0.05] + q90)
    assert np.allclose(np.sort(S[1].V, 0), np.sort(Vref, 0), atol=1e-12)


def test_convex_mesh_scale():
    """A ConvexMesh shape is scaled vertex-wise: unit-cube vertices with scale 0.02 is the task cube. Gap to a cube
    2h + 1 mm away along x: 1 mm by hand."""
    V = [list(v) for v in REF.box_vertices((1, 1, 1))]
    tri = G._BOX_TRI.tolist()
    geo = {'m': {'shapes': [{'type': 'PhysxCollisionShapeConvexMesh', 'local_pose': [0, 0, 0, 1, 0, 0, 0], 'scale': [h, h, h], 'vertices': V, 'triangles': tri}]},
           'c': {'shapes': [{'type': 'PhysxCollisionShapeBox', 'local_pose': [0, 0, 0, 1, 0, 0, 0], 'half_size': [h, h, h]}]}}
    A = SL.posed(geo, 'm', [0, 0, 0, 1, 0, 0, 0]); B = SL.posed(geo, 'c', [2 * h + 0.001, 0, 0, 1, 0, 0, 0])
    assert abs(SL.gap(A, B) - 0.001) <= TOL_M


def test_hand_hull_existing_vs_reference_random_poses():
    """No closed form for the hand hull: the two independent paths must agree (cross-path check, not known truth).
    Fixed random poses of the hand hull and a task cube, separated and overlapping; agreement within TOL_M."""
    rng = np.random.default_rng(20261122); hull = G.hand_hull_local(); n_sep = n_pen = 0
    for _ in range(60):
        q = rng.normal(size=4); q /= np.linalg.norm(q); p = rng.uniform(-0.06, 0.06, 3)
        A = hull.moved(G.quat_to_R(q), p); VA = np.asarray(hull.V) @ REF.R_wxyz(q).T + p
        e = G.signed_distance(A, G.box(np.zeros(3), np.eye(3), (h, h, h))); r, _ = REF.signed_distance_ref(VA, REF.box_vertices((h,) * 3))
        assert abs(e - r) <= TOL_M, (e, r); n_sep += e > 0; n_pen += e < 0
    assert n_sep > 5 and n_pen > 5
