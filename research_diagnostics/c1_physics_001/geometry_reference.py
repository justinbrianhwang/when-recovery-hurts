"""Independent reference geometry for checking hci.spatial_geometry (POST_HOC_EXPLORATORY verification code).

It shares NO code with hci.spatial_geometry: no import of it, no vertex-triangle / edge-edge enumeration, no separating-axis
candidate list. The two paths used here are different algorithms:

  separated distance   GJK on the Minkowski difference A - B, with the support mapping on the vertex sets and an
                       exhaustive sub-simplex projection (closest point of a <= 4-point simplex to the origin by
                       enumerating its faces, edges and vertices with least-squares barycentric coordinates).
                       Witness points are returned.
  penetration depth    exact: the facets of the convex hull of the Minkowski difference {a_i - b_j} computed by Qhull
                       (scipy.spatial.ConvexHull, an external library). When the origin is inside, the penetration depth
                       (minimum translation distance) is the smallest distance from the origin to a facet plane.
                       This is NOT EPA (EPA is NOT_IMPLEMENTED here) and not the separating-axis test.
  poses                quaternions (w, x, y, z) converted with scipy.spatial.transform.Rotation (scalar-last input),
                       not with hci.spatial_geometry.quat_to_R.

Shapes are given as vertex arrays (n, 3) in metres; a convex shape is the convex hull of its vertices.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

EPS_REL = 1e-12


def R_wxyz(q):
    q = np.asarray(q, float); return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def box_vertices(half, R=np.eye(3), p=np.zeros(3)):
    h = np.asarray(half, float); V = np.array(list(itertools.product((-1, 1), repeat=3)), float) * h
    return V @ np.asarray(R, float).T + np.asarray(p, float)


def pose_vertices(V_local, local_pose, body_pose):
    """World vertices of a shape with local pose [x y z qw qx qy qz] in a body at pose [x y z qw qx qy qz]."""
    Rl = R_wxyz(local_pose[3:]); Rb = R_wxyz(body_pose[3:])
    return (np.asarray(V_local, float) @ Rl.T + np.asarray(local_pose[:3], float)) @ Rb.T + np.asarray(body_pose[:3], float)


def _support(V, d): i = int(np.argmax(V @ d)); return V[i]


def _closest_on_simplex(W):
    """Closest point to the origin of the convex hull of <= 4 points W (k, 3): (point, barycentric weights, used indices).
    Exhaustive: every non-empty subset, affine projection by least squares, kept only if all weights are >= 0."""
    k = len(W); best = None
    for r in range(1, k + 1):
        for S in itertools.combinations(range(k), r):
            P = W[list(S)]
            if r == 1: lam = np.array([1.]); x = P[0]
            else:
                # minimise |P0 + sum_j mu_j (Pj - P0)|^2
                D = (P[1:] - P[0]).T; mu, *_ = np.linalg.lstsq(D, -P[0], rcond=None); lam = np.concatenate([[1 - mu.sum()], mu]); x = P[0] + D @ mu
            if np.all(lam >= -1e-12):
                n = float(x @ x)
                if best is None or n < best[0] - 1e-30: best = (n, x, lam, S)
    return best[1], best[2], best[3]


def gjk_distance(VA, VB, max_iter=200, tol=1e-12):
    """Distance between the convex hulls of VA and VB, with witness points. Returns (distance, pa, pb, overlap_flag).
    overlap_flag True means the origin is in A - B (distance 0, shapes touch or overlap)."""
    VA = np.asarray(VA, float); VB = np.asarray(VB, float)
    d = VA.mean(0) - VB.mean(0)
    if not np.any(d): d = np.array([1., 0., 0.])
    a = _support(VA, -d); b = _support(VB, d); W = [a - b]; WA = [a]; WB = [b]; v = W[0]
    for _ in range(max_iter):
        vv = float(v @ v)
        if vv < 1e-30: return 0., None, None, True
        a = _support(VA, -v); b = _support(VB, v); w = a - b
        if vv - float(v @ w) <= tol * max(vv, 1e-30) or any(np.allclose(w, x, atol=1e-15) for x in W):
            break
        W.append(w); WA.append(a); WB.append(b)
        x, lam, S = _closest_on_simplex(np.array(W))
        W = [W[i] for i in S]; WA = [WA[i] for i in S]; WB = [WB[i] for i in S]; v = x
        if len(W) == 4 and float(v @ v) < 1e-24: return 0., None, None, True
    x, lam, S = _closest_on_simplex(np.array(W))
    pa = sum(l * WA[i] for l, i in zip(lam, S)); pb = sum(l * WB[i] for l, i in zip(lam, S))
    return float(np.linalg.norm(x)), pa, pb, False


def minkowski_hull(VA, VB):
    M = (np.asarray(VA, float)[:, None, :] - np.asarray(VB, float)[None, :, :]).reshape(-1, 3)
    return ConvexHull(M)


def origin_inside(hull, tol=0.):
    # Qhull facets: normal . x + offset <= 0 inside
    return bool(np.all(hull.equations[:, 3] <= tol))


def penetration_depth(VA, VB):
    """Exact minimum translation distance of two overlapping convex polytopes from the Qhull facets of A - B.
    Returns (depth >= 0, unit direction) or (None, None) if the origin is not inside A - B."""
    h = minkowski_hull(VA, VB)
    if not origin_inside(h, 1e-15): return None, None
    off = -h.equations[:, 3]; i = int(np.argmin(off)); return float(off[i]), h.equations[i, :3]


def signed_distance_ref(VA, VB):
    """> 0: GJK distance; <= 0: minus the Qhull penetration depth. Also returns the method used."""
    d, _, _, ov = gjk_distance(VA, VB)
    if not ov and d > 1e-12: return d, 'gjk'
    depth, _ = penetration_depth(VA, VB)
    if depth is None: return d, 'gjk_touching'
    return -depth, 'qhull_minkowski_facets'
