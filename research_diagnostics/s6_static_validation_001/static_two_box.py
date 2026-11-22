"""S6 measurement-validity extension B2: static and controlled two-box PhysX experiments, independent of every C1 record.

PhysX is configured exactly as ManiSkill a4a4f92 configures it (SceneConfig defaults: contact_offset 0.02, rest_offset 0,
PCM on, TGS on, 15 position / 1 velocity iterations, bounce threshold 2, sleep threshold 0.005, default material 0.3/0.3/0;
timestep 1/100 s). Each case builds a fresh scene with one static box A and one dynamic box B whose poses are set
exactly, steps the physics, and compares every reported contact point separation with the analytic distance of the
two boxes at the pose BEFORE the step and AFTER the step.

What is analytic: for face-face (B axis-aligned), edge-face and vertex-face (B rotated; A's face large enough to contain
the projection), the exact distance is (lowest support point of B along -n) - (A face), computed in closed form from the
box half sizes and the rotation. Penetration depth for aligned overlapping boxes is the overlap along x.

Cases (fixed before running):
  G   gap sweep, aligned faces, gravity 0, B at rest: d in {-3, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 20, 30, 39, 39.9, 40.1, 41, 45} mm
  R   rotated B (45 deg about z: edge-face; 45/35.26 deg: vertex-face; 30 deg about z: finger-like thin box), gap 0.5-20 mm
  M   moving B toward A at v in {0.05, 0.1, 0.2, 0.5} m/s, gravity 0, one step and five steps (one ManiSkill control step);
      separation compared with the gap before and after the LAST physics step
  P   persistence: B moved away by set_pose between steps (no velocity), gap after the move vs reported separation
  W   gravity on (-9.81), B resting near/on A's top face: what a resting contact reports
Outputs: CSV of every contact point and a JSON summary. Nothing here reads or writes C1 data.

usage (WSL venv with sapien 3.0.3): python static_two_box.py --out <new folder>
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path

import numpy as np
import sapien
import sapien.physx as px

DT = 0.01
HA = np.array([0.05, 0.05, 0.05])            # static box A: 10 cm cube, its +x face at x = 0
HB = np.array([0.02, 0.02, 0.02])            # dynamic box B: the 4 cm cube of StackCube
HF = np.array([0.01, 0.0035, 0.02])          # finger-like thin box


def configure(gravity=(0, 0, 0)):
    px.set_shape_config(contact_offset=0.02, rest_offset=0.0)
    px.set_body_config(solver_position_iterations=15, solver_velocity_iterations=1, sleep_threshold=0.005)
    px.set_scene_config(gravity=np.asarray(gravity, np.float32), bounce_threshold=2.0, enable_pcm=True, enable_tgs=True, enable_ccd=False,
                        enable_enhanced_determinism=False, enable_friction_every_iteration=True, cpu_workers=0)
    px.set_default_material(static_friction=0.3, dynamic_friction=0.3, restitution=0.0)


def quat_wxyz(R):
    """Rotation matrix -> quaternion [w, x, y, z]."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat(); return [w, x, y, z]


def rot(axis_angles):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('xyz', axis_angles, degrees=True).as_matrix() if axis_angles else np.eye(3)


def lowest_x(center, R, half):
    """min over B's vertices of x (the support point toward A along -x)."""
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * half
    return float((center + s @ R.T)[:, 0].min())


def build(gravity, posB, RB, halfB, velB=None):
    configure(gravity)
    system = px.PhysxCpuSystem(); system.set_timestep(DT); scene = sapien.Scene([system])
    a = scene.create_actor_builder(); a.add_box_collision(half_size=HA.tolist()); A = a.build_static(name='A'); A.set_pose(sapien.Pose([-HA[0], 0, 0]))
    b = scene.create_actor_builder(); b.add_box_collision(half_size=halfB.tolist(), density=1000); B = b.build(name='B')
    B.set_pose(sapien.Pose(list(posB), quat_wxyz(RB)))
    comp = B.find_component_by_type(px.PhysxRigidDynamicComponent)
    if velB is not None: comp.set_linear_velocity(list(velB))
    return system, scene, A, B, comp


def gap_of(B, halfB):
    p = B.get_pose(); from scipy.spatial.transform import Rotation
    w, x, y, z = p.q; R = Rotation.from_quat([x, y, z, w]).as_matrix(); return lowest_x(np.asarray(p.p, float), R, halfB) - 0.0


def contacts(system, case, step):
    out = []
    for c in system.get_contacts():
        names = [getattr(b.entity, 'name', '?') for b in c.bodies]
        for i, pt in enumerate(c.points):
            n = np.asarray(pt.normal, float)
            if names == ['A', 'B']: n = n
            out.append({'case': case, 'step': step, 'pair': '-'.join(names), 'point': i, 'separation_m': float(pt.separation),
                        'pos': [float(v) for v in pt.position], 'normal': [float(v) for v in n], 'impulse': [float(v) for v in pt.impulse]})
    return out


def run_case(case, gravity, posB, RB, halfB, steps=1, velB=None, move_between=None):
    system, scene, A, B, comp = build(gravity, posB, RB, halfB, velB)
    rows = []; gaps = [gap_of(B, halfB)]
    for k in range(steps):
        if move_between is not None and k == steps - 1 and k > 0:
            p = B.get_pose(); B.set_pose(sapien.Pose(list(np.asarray(p.p) + np.asarray(move_between)), p.q)); gaps[-1] = gap_of(B, halfB)
        system.step(); gaps.append(gap_of(B, halfB))
        rows = contacts(system, case, k + 1)
    seps = [r['separation_m'] for r in rows]
    res = {'case': case, 'steps': steps, 'gap_before_last_step_m': gaps[-2], 'gap_after_last_step_m': gaps[-1], 'listed_points': len(rows),
           'separation_min_m': min(seps) if seps else None, 'separation_max_m': max(seps) if seps else None,
           'impulse_norm_sum': float(sum(np.linalg.norm(r['impulse']) for r in rows)),
           'min_sep_minus_gap_before_mm': (min(seps) - gaps[-2]) * 1e3 if seps else None, 'min_sep_minus_gap_after_mm': (min(seps) - gaps[-1]) * 1e3 if seps else None,
           'B_velocity_after_m_s': [float(v) for v in comp.get_linear_velocity()]}
    return res, rows


def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', required=True); a = a.parse_args(); out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    out.mkdir(parents=True); results = []; points = []
    def add(r): results.append(r[0]); points.extend(r[1])
    # G: aligned gap sweep, B at rest, gravity 0
    for d_mm in (-3, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 20, 30, 39, 39.9, 40.1, 41, 45):
        d = d_mm / 1e3; add(run_case(f'G_aligned_gap_{d_mm}mm', (0, 0, 0), (d + HB[0], 0, 0), np.eye(3), HB))
    # R: rotated B, analytic gap from the support point
    for name, ang, half in (('edge_face_z45', (0, 0, 45), HB), ('vertex_face_x45_y35.26', (45, 35.26439, 0), HB), ('finger_z30', (0, 0, 30), HF), ('tilt_y10', (0, 10, 0), HB)):
        R = rot(ang)
        for d_mm in (0.5, 2, 5, 10, 20):
            d = d_mm / 1e3; cx = d - lowest_x(np.zeros(3), R, half); add(run_case(f'R_{name}_gap_{d_mm}mm', (0, 0, 0), (cx, 0, 0), R, half))
    # M: moving toward A, gravity 0; one step and five steps
    for v in (0.05, 0.1, 0.2, 0.5):
        for steps, g0_mm in ((1, 10), (5, 15)):
            g0 = g0_mm / 1e3; add(run_case(f'M_v{v}_steps{steps}_gap0_{g0_mm}mm', (0, 0, 0), (g0 + HB[0], 0, 0), np.eye(3), HB, steps=steps, velB=(-v, 0, 0)))
    # P: teleport away between steps (no velocity): separation after the last step vs the gap after the move
    for dx_mm in (1, 3, 10):
        add(run_case(f'P_teleport_plus{dx_mm}mm', (0, 0, 0), (0.002 + HB[0], 0, 0), np.eye(3), HB, steps=2, move_between=(dx_mm / 1e3, 0, 0)))
    # S: slow motion over many steps (PCM contact persistence): per step, separation minus the gap before / after that step,
    # with PCM on (ManiSkill) and off (diagnostic only). Linear drift toward A, and slow rotation of B about z (edge sweep).
    slow = []
    for pcm in (True, False):
        for kind, v, w in (('lin', 0.001, 0.0), ('lin', 0.005, 0.0), ('rot', 0.0, 0.05), ('rot', 0.0, 0.2)):
            configure((0, 0, 0)); px.set_scene_config(gravity=np.zeros(3, np.float32), bounce_threshold=2.0, enable_pcm=pcm, enable_tgs=True, enable_ccd=False,
                                                          enable_enhanced_determinism=False, enable_friction_every_iteration=True, cpu_workers=0)
            system = px.PhysxCpuSystem(); system.set_timestep(DT); scene = sapien.Scene([system])
            ab = scene.create_actor_builder(); ab.add_box_collision(half_size=HA.tolist()); A = ab.build_static(name='A'); A.set_pose(sapien.Pose([-HA[0], 0, 0]))
            bb = scene.create_actor_builder(); bb.add_box_collision(half_size=HB.tolist(), density=1000); B = bb.build(name='B')
            R0 = rot((0, 0, 10)); B.set_pose(sapien.Pose([0.008 - lowest_x(np.zeros(3), R0, HB), 0, 0], quat_wxyz(R0)))
            comp = B.find_component_by_type(px.PhysxRigidDynamicComponent); comp.set_linear_velocity([-v, 0, 0]); comp.set_angular_velocity([0, 0, w])
            for k in range(50):
                g0 = gap_of(B, HB); system.step(); g1 = gap_of(B, HB); rows = contacts(system, f'S_{kind}_v{v}_w{w}_pcm{int(pcm)}', k + 1)
                if rows:
                    s = min(r['separation_m'] for r in rows); slow.append({'case': f'S_{kind}_v{v}_w{w}_pcm{int(pcm)}', 'step': k + 1, 'gap_before_m': g0, 'gap_after_m': g1, 'sep_min_m': s,
                                                                         'sep_minus_before_mm': (s - g0) * 1e3, 'sep_minus_after_mm': (s - g1) * 1e3, 'B_sleeping_after_step': bool(comp.is_sleeping)})
    with open(out / 'slow_motion_steps.csv', 'w', newline='', encoding='utf-8') as f:
        w_ = csv.DictWriter(f, fieldnames=list(slow[0])); w_.writeheader(); [w_.writerow(r) for r in slow]
    for c in sorted({r['case'] for r in slow}):
        S = [r for r in slow if r['case'] == c]
        results.append({'case': c, 'steps': len(S), 'gap_before_last_step_m': S[-1]['gap_before_m'], 'gap_after_last_step_m': S[-1]['gap_after_m'], 'listed_points': None,
                        'separation_min_m': S[-1]['sep_min_m'], 'separation_max_m': None, 'impulse_norm_sum': None,
                        'min_sep_minus_gap_before_mm': max(abs(r['sep_minus_before_mm']) for r in S), 'min_sep_minus_gap_after_mm': max(abs(r['sep_minus_after_mm']) for r in S),
                        'B_velocity_after_m_s': None, 'note': 'S cases: the two difference fields are the MAX ABS over all 50 steps'})
    # W: gravity on; B resting on top of A (A's top face at z = 0 after shifting): place B 1 mm above A's top
    configure((0, 0, -9.81))
    for d_mm, steps in ((1, 1), (1, 5), (0, 5), (0, 50)):
        d = d_mm / 1e3; system, scene, A, B, comp = build((0, 0, -9.81), (-HA[0], 0, HA[2] + HB[2] + d), np.eye(3), HB)
        zs = [B.get_pose().p[2] - HB[2] - HA[2]]
        for k in range(steps): system.step(); zs.append(B.get_pose().p[2] - HB[2] - HA[2])
        rows = contacts(system, f'W_rest_gap{d_mm}mm_steps{steps}', steps); seps = [r['separation_m'] for r in rows]; points.extend(rows)
        results.append({'case': f'W_rest_gap{d_mm}mm_steps{steps}', 'steps': steps, 'gap_before_last_step_m': float(zs[-2]), 'gap_after_last_step_m': float(zs[-1]), 'listed_points': len(rows),
                        'separation_min_m': min(seps) if seps else None, 'separation_max_m': max(seps) if seps else None, 'impulse_norm_sum': float(sum(np.linalg.norm(r['impulse']) for r in rows)),
                        'min_sep_minus_gap_before_mm': (min(seps) - zs[-2]) * 1e3 if seps else None, 'min_sep_minus_gap_after_mm': (min(seps) - zs[-1]) * 1e3 if seps else None,
                        'B_velocity_after_m_s': [float(v) for v in comp.get_linear_velocity()]})
    with open(out / 'contact_points.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(points[0])); w.writeheader(); [w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in p.items()}) for p in points]
    with open(out / 'cases.csv', 'w', newline='', encoding='utf-8') as f:
        keys = list(dict.fromkeys(k for r in results for k in r)); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}) for r in results]
    import platform
    (out / 'run.json').write_text(json.dumps({'sapien': sapien.__version__, 'python': platform.python_version(), 'numpy': np.__version__, 'dt': DT, 'HA': HA.tolist(), 'HB': HB.tolist(), 'HF': HF.tolist(),
                                              'config': 'ManiSkill a4a4f92 SceneConfig defaults (see module docstring)', 'cases': len(results), 'points': len(points)}, indent=1), encoding='utf-8')
    fmt = lambda v, n=6: 'NA' if v is None else f'{v:.{n}f}'
    for r in results: print(f"{r['case']:<36} n={str(r['listed_points']):<4} sep_min_mm={fmt(None if r['separation_min_m'] is None else r['separation_min_m'] * 1e3, 5):<12} sep-before_mm={fmt(r['min_sep_minus_gap_before_mm']):<11} sep-after_mm={fmt(r['min_sep_minus_gap_after_mm']):<11}")


if __name__ == '__main__':
    main()
