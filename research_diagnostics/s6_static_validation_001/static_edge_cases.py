"""S6 B2 follow-up (added after the B4 sub-step result, which left a residual only in zero-impulse finger-box / cube
proximity pairs where PhysX reports a LARGER separation than the minimum distance): two boxes whose closest features are
edge-edge, edge-vertex or vertex-vertex, at rest, same PhysX configuration as static_two_box.py. The exact minimum
distance comes from the independent GJK reference (research_diagnostics/c1_physics_001/geometry_reference.py).

usage (WSL venv with sapien 3.0.3): python static_edge_cases.py --out <new folder>
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / 'c1_physics_001'))
import static_two_box as S  # noqa: E402
import geometry_reference as GR  # noqa: E402
import sapien  # noqa: E402
import sapien.physx as px  # noqa: E402

FINGER = np.array([0.01, 0.0035, 0.02])
CUBE = np.array([0.02, 0.02, 0.02])


def run(name, RA, pA, halfA, RB, pB, halfB):
    S.configure((0, 0, 0)); system = px.PhysxCpuSystem(); system.set_timestep(S.DT); scene = sapien.Scene([system])
    a = scene.create_actor_builder(); a.add_box_collision(half_size=halfA.tolist()); A = a.build_static(name='A'); A.set_pose(sapien.Pose(list(pA), S.quat_wxyz(RA)))
    b = scene.create_actor_builder(); b.add_box_collision(half_size=halfB.tolist(), density=1000); B = b.build(name='B'); B.set_pose(sapien.Pose(list(pB), S.quat_wxyz(RB)))
    VA = (np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * halfA) @ RA.T + pA
    VB = (np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * halfB) @ RB.T + pB
    dist, pa, pb, overlap = GR.gjk_distance(VA, VB); d = float(dist)
    if overlap: raise ValueError(f'{name}: boxes overlap; the case is mis-specified')
    system.step()
    pts = [(float(p.separation), [float(v) for v in p.normal], [float(v) for v in p.position]) for c in system.get_contacts() for p in c.points]
    seps = [p[0] for p in pts]
    return {'case': name, 'gjk_min_distance_mm': d * 1e3, 'listed_points': len(pts), 'sep_min_mm': min(seps) * 1e3 if seps else None, 'sep_max_mm': max(seps) * 1e3 if seps else None,
            'sep_min_minus_distance_mm': (min(seps) - d) * 1e3 if seps else None, 'normals': [p[1] for p in pts][:4]}


def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', required=True); a = a.parse_args(); out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    out.mkdir(parents=True); rows = []
    I = np.eye(3)
    for gap_mm in (2, 10, 20, 30):
        g = gap_mm / 1e3
        # face-face reference (expected equal)
        rows.append(run(f'face_face_{gap_mm}mm', I, np.zeros(3), CUBE, I, np.array([0.04 + g, 0, 0]), CUBE))
        # edge-edge: A rotated 45 deg about z (vertical edge toward +x), B rotated 45 deg about y (horizontal edge toward -x), crossing at right angles
        RA = S.rot((0, 0, 45)); RB = S.rot((0, 45, 0)); s = 0.02 * np.sqrt(2)
        rows.append(run(f'edge_edge_crossed_{gap_mm}mm', RA, np.zeros(3), CUBE, RB, np.array([2 * s + g, 0, 0]), CUBE))
        # edge-edge parallel offset: both rotated 45 deg about z, B shifted in y so their vertical edges are offset
        rows.append(run(f'edge_edge_offset_y_{gap_mm}mm', RA, np.zeros(3), CUBE, RA, np.array([2 * s + g, 0.015, 0]), CUBE))
        # finger-like thin box vs cube, finger corner toward the cube's vertical edge
        RF = S.rot((0, 0, 30)); rows.append(run(f'finger_corner_vs_cube_edge_{gap_mm}mm', RA, np.zeros(3), CUBE, RF, np.array([s + 0.012 + g, 0.01, 0.01]), FINGER))
        # vertex-vertex: both rotated so a vertex points at the other
        RV = S.rot((45, 35.26439, 0)); rows.append(run(f'vertex_vertex_{gap_mm}mm', RV, np.zeros(3), CUBE, RV.T, np.array([2 * 0.02 * np.sqrt(3) + g, 0, 0]), CUBE))
    with open(out / 'edge_cases.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}) for r in rows]
    (out / 'run.json').write_text(json.dumps({'sapien': sapien.__version__, 'cases': len(rows)}, indent=1))
    for r in rows: print(f"{r['case']:<38} gjk={r['gjk_min_distance_mm']:8.4f} n={r['listed_points']} sep_min={'NA' if r['sep_min_mm'] is None else round(r['sep_min_mm'], 4)} diff={'NA' if r['sep_min_minus_distance_mm'] is None else round(r['sep_min_minus_distance_mm'], 4)}")


if __name__ == '__main__':
    main()
