"""C: writes GEOMETRY_KNOWN_TRUTH_RESULTS.csv and GEOMETRY_LOGGED_MESH_CHECK.json (POST_HOC_EXPLORATORY verification).

1. Every known-truth case of test_geometry_known_truth (closed-form truth) through the existing path and the reference path.
2. The cross-path hand-hull check on random poses (no closed form; agreement only).
3. The collision shapes as logged in the C1 records: per shape, whether the logged triangles are the convex hull of the logged
   vertices (all vertices on or inside every face plane), the volume of the logged triangle mesh vs the hull, and whether
   the geometry is identical in every scene record. Then, for the logged hand mesh, existing vs reference distance to a task
   cube at random poses: this measures what the triangulation does to the existing calculator.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import geometry_reference as REF  # noqa: E402
import test_geometry_known_truth as K  # noqa: E402
from hci import spatial_geometry as G  # noqa: E402
from hci import spatial_links as SL  # noqa: E402


def main():
    out = C.OUT; rows = []
    for c in K.CASES:
        e = K.existing(c); r, how = K.reference(c)
        rows.append({'label': C.LABEL, 'case': c['case'], 'group': c['group'], 'description': c['description'], 'truth_mm': c['truth_m'] * 1e3,
                     'existing_mm': e * 1e3, 'reference_mm': r * 1e3, 'reference_method': how, 'existing_abs_err_m': abs(e - c['truth_m']), 'reference_abs_err_m': abs(r - c['truth_m']),
                     'tolerance_m': K.TOL_M, 'existing_pass': abs(e - c['truth_m']) <= K.TOL_M, 'reference_pass': abs(r - c['truth_m']) <= K.TOL_M,
                     'truth_source': 'closed form (see the comment of the case in test_geometry_known_truth.py)'})
    # logged C1 geometry
    first = None; identical = True; scenes = 0
    for p in C.POLICIES:
        for f in sorted((C.pass_dir(p) / 'scenes').glob('[0-9]*.json')):
            g = json.loads(f.read_text(encoding='utf-8'))['initial_state']['collision_geometry']; scenes += 1
            shapes = {b: [{k: v for k, v in s.items()} for s in e['shapes']] for b, e in g.items()}
            if first is None: first = (shapes, g)
            elif shapes != first[0]: identical = False
    geo = first[1]; shape_rows = []
    for body, e in geo.items():
        for i, d in enumerate(e['shapes']):
            if 'ConvexMesh' not in d['type']: shape_rows.append({'body': body, 'shape': i, 'type': d['type'], 'note': 'box: exact (8 vertices, 12 triangles built by the audit)'}); continue
            V = np.asarray(d['vertices'], float) * np.asarray(d['scale'], float); T = np.asarray(d['triangles']); H = ConvexHull(V)
            L = G.Convex(V, T); P = V[L.T]; n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); area2 = np.linalg.norm(n, axis=1); n = n / area2[:, None]; off = np.einsum('ij,ij->i', n, P[:, 0])
            viol = (V @ n.T - off).max(0)
            shape_rows.append({'body': body, 'shape': i, 'type': d['type'], 'vertices': len(V), 'triangles': len(T), 'hull_vertices': len(H.vertices), 'hull_facets_triangles': len(H.simplices),
                               'mesh_volume_m3': L.volume(), 'hull_volume_m3': float(H.volume), 'volume_ratio': L.volume() / H.volume,
                               'triangles_with_a_vertex_outside_their_plane_gt_1e-6m': int((viol > 1e-6).sum()), 'max_vertex_outside_a_triangle_plane_m': float(viol.max()),
                               'smallest_triangle_area_m2': float(area2.min() / 2), 'logged_triangles_are_hull_facets': bool((viol <= 1e-9).all())})
    # effect on the existing calculator for the logged hand mesh
    d = geo['panda_hand']['shapes'][0]; V = np.asarray(d['vertices'], float) * np.asarray(d['scale'], float); L = G.Convex(V, d['triangles'])
    rng = np.random.default_rng(20261122); diffs = []
    for _ in range(300):
        q = rng.normal(size=4); q /= np.linalg.norm(q); p = rng.uniform(-0.08, 0.08, 3)
        e = G.signed_distance(L.moved(G.quat_to_R(q), p), G.box(np.zeros(3), np.eye(3), (K.h,) * 3)); r, how = REF.signed_distance_ref(V @ REF.R_wxyz(q).T + p, REF.box_vertices((K.h,) * 3))
        diffs.append({'existing_m': e, 'reference_m': r, 'method': how})
    dd = np.array([x['existing_m'] - x['reference_m'] for x in diffs]); sep = np.array([x['reference_m'] > 0 for x in diffs])
    mesh = {'label': C.LABEL, 'scene_records_checked': scenes, 'geometry_identical_in_every_record': identical, 'shapes': shape_rows,
            'logged_hand_mesh_existing_minus_reference_random_poses': {'n': len(diffs), 'separated': int(sep.sum()), 'overlapping': int((~sep).sum()),
                                                                     'separated_max_abs_mm': float(np.abs(dd[sep]).max() * 1e3), 'separated_median_signed_mm': float(np.median(dd[sep]) * 1e3),
                                                                     'separated_share_existing_larger': float((dd[sep] > 1e-12).mean()),
                                                                     'overlapping_max_abs_mm': float(np.abs(dd[~sep]).max() * 1e3) if (~sep).any() else None,
                                                                     'sign_disagreements': int(sum((x['existing_m'] > 0) != (x['reference_m'] > 0) for x in diffs))},
            'reading': ('The PhysX-cooked hand mesh is logged as vertices plus a triangulation that is not exactly the convex hull of those vertices. PhysX collides with '
                        'a convex hull; hci.spatial_geometry enumerates the logged triangles (vertex-triangle, edge-edge, SAT on triangle normals) and therefore '
                        'differs from the hull of the same vertices by the amounts above. Boxes (cubes, fingers, camera, table) are unaffected.')}
    C.write_csv(out / 'GEOMETRY_KNOWN_TRUTH_RESULTS.csv', rows); C.jdump(out / 'GEOMETRY_LOGGED_MESH_CHECK.json', mesh)
    print(json.dumps({'cases': len(rows), 'existing_pass': sum(r['existing_pass'] for r in rows), 'reference_pass': sum(r['reference_pass'] for r in rows),
                      'mesh': {k: v for k, v in mesh.items() if k != 'shapes'}, 'convex_shapes': [s for s in shape_rows if 'vertices' in s]}, indent=1, default=str))


if __name__ == '__main__':
    main()
