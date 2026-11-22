"""Tests of the hull variant and the S4/S5 rules of hull_s45 on closed-form cases (no records needed)."""
import math, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hull_s45 as H  # noqa: E402
L = H.L; G = H.G

# A 2 cm cube as a ConvexMesh whose top face is split through an extra vertex pushed 3 mm DOWN (a dent): the logged
# triangulation is then not the convex hull of its vertices (the dent vertex is inside the hull).
CUBE = [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (-0.01, 0.01)]
DENT = [0., 0., 0.007]


def dented_mesh(scale=(1., 1., 1.)):
    V = CUBE + [DENT]; i = {tuple(v): k for k, v in enumerate(CUBE)}
    def q(x, y, z): return i[(x, y, z)]
    a, b, c, d = q(-.01, -.01, .01), q(.01, -.01, .01), q(.01, .01, .01), q(-.01, .01, .01); m = 8
    top = [[a, b, m], [b, c, m], [c, d, m], [d, a, m]]
    e, f, g, h = q(-.01, -.01, -.01), q(.01, -.01, -.01), q(.01, .01, -.01), q(-.01, .01, -.01)
    rest = [[e, g, f], [e, h, g], [e, f, b], [e, b, a], [f, g, c], [f, c, b], [g, h, d], [g, d, c], [h, e, a], [h, a, d]]
    return {'type': 'PhysxCollisionShapeConvexMesh', 'vertices': V, 'triangles': top + rest, 'scale': list(scale), 'local_pose': [0, 0, 0, 1, 0, 0, 0]}


def box_shape(half): return {'type': 'PhysxCollisionShapeBox', 'half_size': list(half), 'local_pose': [0, 0, 0, 1, 0, 0, 0]}


def test_hull_replaces_the_dented_triangulation_and_keeps_scale():
    d = dented_mesh(scale=(2., 1., 1.)); h = H.hull_of_shape(d)
    assert len(h['vertices']) == 8 and len(h['triangles']) == 12 and h['scale'] == [2., 1., 1.] and h['local_pose'] == d['local_pose']
    assert d['vertices'][-1] == DENT and len(d['triangles']) == 14                       # the input dict is not changed
    st = H.shape_stats({'b': {'shapes': [d]}})[0]
    assert st['max_logged_surface_depth_inside_hull_m'] == pytest.approx(0.003, abs=1e-12) and st['logged_triangles_not_hull_faces'] == 4


def test_distance_above_the_dent_logged_vs_hull_closed_form():
    """A 4 mm box whose bottom is 5 mm above the cube top, centred over the dent. Hull: 5 mm (closed form). Logged
    triangulation: the top face is missing, so the separated distance is measured to the dented faces and the rim and is
    larger (the logged path over-estimates separation; it can never be smaller than the hull distance here). Over a
    corner, away from the dent, both give the closed-form 5 mm."""
    mesh = dented_mesh(); hull = H.hull_of_shape(mesh)
    lg = L._local_shape(mesh); hl = L._local_shape(hull)
    above = G.box([0, 0, 0.01 + 0.005 + 0.002], np.eye(3), (0.002, 0.002, 0.002))
    assert G.signed_distance(hl, above) == pytest.approx(0.005, abs=1e-12)
    assert G.signed_distance(lg, above) > 0.005 + 1e-3
    corner = G.box([0.01 + 0.005 + 0.002, 0.01 + 0.005 + 0.002, 0], np.eye(3), (0.002, 0.002, 0.002))
    assert G.signed_distance(lg, corner) == pytest.approx(math.sqrt(2) * 0.005, abs=1e-12)
    assert G.signed_distance(hl, corner) == pytest.approx(math.sqrt(2) * 0.005, abs=1e-12)


def test_hull_geometry_leaves_boxes_and_planes_untouched():
    geo = {'panda_hand': {'shapes': [dented_mesh()]}, 'cubeA': {'shapes': [box_shape((0.02,) * 3)]}, 'scene-0_ground': {'shapes': [{'type': 'PhysxCollisionShapePlane', 'local_pose': [0, 0, 0, 1, 0, 0, 0]}]}}
    g = H.hull_geometry(geo)
    assert g['cubeA'] == geo['cubeA'] and g['scene-0_ground'] == geo['scene-0_ground'] and len(g['panda_hand']['shapes'][0]['triangles']) == 12
    assert len(geo['panda_hand']['shapes'][0]['triangles']) == 14


def _m(t, s4=None, pairs=()):
    m = {'t': t, 's5': [{'a': 'x', 'shape_a': 0, 'b': 'y', 'shape_b': 0, 'physx_min_separation_m': 0., 'reconstructed_m': r} for r in pairs]}
    if s4 is not None: m['s4'] = {'robot': s4}
    return m


def test_items_rules_match_spatial_items_semantics():
    rows = [_m(10, pairs=[0.0005]), _m(11, 0.003, [-0.002]), _m(12, 0.001, [0.0009])]
    it = H.items_from(rows, 10, 2, True)
    assert it[H.S4_KEY]['status'] == 'PASS' and it[H.S4_KEY]['min_distance_m'] == 0.001                  # the trigger row (t=10) is not a move row
    assert it[H.S5_KEY]['status'] == 'PASS' and it[H.S5_KEY]['n_pairs'] == 3 and it[H.S5_KEY]['max_penetration_m'] == 0.002
    assert H.items_from([_m(10), _m(11, 0.)], 10, 1, True)[H.S4_KEY]['status'] == 'FAIL'                  # > 0 is strict
    bad = H.items_from([_m(10, pairs=[0.0011])], None, 0, False)
    assert bad[H.S4_KEY]['status'] == 'NOT_APPLICABLE' and bad[H.S5_KEY]['status'] == 'FAIL' and bad[H.S5_KEY]['n_violations'] == 1
    assert H.items_from([_m(10, pairs=[None, 0.])], None, 0, False)[H.S5_KEY]['status'] == 'NOT_OBSERVABLE'
    assert H.items_from([_m(10)], None, 0, False)[H.S5_KEY]['status'] == 'NOT_OBSERVABLE'
