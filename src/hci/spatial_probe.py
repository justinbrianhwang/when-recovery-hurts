"""Logging-only spatial probe for Vision-CENTER v2 Phase C1 (Option A). Not used by plan dd3af27e; a C1 plan uses it only
when its driver passes probe=gt_probe_spatial.

It extends hci.recovery_v1.probe (the ground-truth logger of every Vision-CENTER trace). The unchanged probe runs first and
its fields are never modified; this module only ADDS fields, all read from the simulator after the control step:

  spatial_schema              'c1_spatial_probe_v1'
  elapsed_steps               the environment's own control-step counter; sim_time_s = elapsed_steps / control_freq
  sim_time_s, sim_timestep_s, sim_substeps_per_control
  robot_qpos, robot_qvel      all 9 joints (7 arm + 2 finger), in robot.get_qpos() order (robot_joint_names in the initial row)
  link_pose                   {link name: [x, y, z, qw, qx, qy, qz]} world pose of every robot link, read from the simulator
  contacts                    every contact pair in the PhysX contact list of the LAST sub-step of this control step:
                              {'a', 'b': body names, 'shape_a', 'shape_b': collision-shape index in that body,
                               'points': [[position xyz, normal xyz, impulse xyz, separation], ...]} as PhysX gives them
                              (impulse applied to body a). PhysX lists pairs within the contact offset, so a pair may have
                              zero impulse and positive separation: that is proximity, not contact. Nothing is filtered.
                              Stored sorted by (a, b, shape_a, shape_b, points); see CONTACT_ORDER.
  f_hand_A, f_hand_B, f_B_table, f_hand_table, f_fingerL_table, f_fingerR_table
                              |sum of pair impulses| / time step over that list: the definition of the existing f_* fields
                              (mani_skill.utils.sapien_utils.get_pairwise_contact_impulse, same operations in the same order)
  f_links_A, f_links_B, f_links_table, f_links_other
                              {robot link: force} for every robot link with a non-zero force against the held cube / the
                              target / the table / any other scene body ({'link|body': force}, e.g. the ground)
  collision_geometry, robot_joint_names   initial row only (the probe call with action None): for every body in the scene
                              {'pose_at_this_row': world pose, 'shapes': every collision shape as PhysX holds it (type, pose
                              in the body frame, size, convex vertices and triangles)}; the static bodies (table, ground)
                              keep that pose, the others move (link_pose, cubeA/cubeA_q, cubeB/cubeB_q per row)

PhysX stores these values in float32. They are written as the shortest decimal that converts back to the same float32
(lossless for the engine's values), which keeps the trace smaller than a float64 repr would.

Not observable from this probe (stated, not worked around): contacts of the first four of the five PhysX sub-steps of a
control step, and any contact between two recorded steps. The simulator step loop is not changed.

Why it cannot change control: hci.pilot2b.Hooks calls the probe only AFTER Runtime.advance has stepped the simulator and
built the next observation; the returned row is only appended to the trace. Every call below is a getter (poses, joint
state, the contact list of the last sub-step, collision shapes); nothing is set, stepped, rendered or drawn at random.
An exception here propagates: the scene stops with an error and the run is INCOMPLETE (never a row without the fields).
"""
import numpy as np

SCHEMA = 'c1_spatial_probe_v1'
STEP_FIELDS = ('spatial_schema', 'elapsed_steps', 'sim_time_s', 'sim_timestep_s', 'sim_substeps_per_control', 'robot_qpos', 'robot_qvel', 'link_pose', 'contacts',
               'f_hand_A', 'f_hand_B', 'f_B_table', 'f_hand_table', 'f_fingerL_table', 'f_fingerR_table', 'f_links_A', 'f_links_B', 'f_links_table', 'f_links_other')
INITIAL_FIELDS = ('collision_geometry', 'robot_joint_names')
ADDED_FIELDS = STEP_FIELDS + INITIAL_FIELDS


def CONTACT_ORDER(c):
    """Storage order of the contact list. PhysX returns the same contacts in an order that is not reproducible between
    scene rebuilds although the simulated state is (measured on the pinned simulator: same sorted content, same pair
    orientation, different order). Sorting changes no value; the f_* sums are formed in the PhysX order, as upstream does."""
    return (c['a'], c['b'], c['shape_a'], c['shape_b'], c['points'])


def f32(x):
    """float32 values as the shortest decimals that convert back to the same float32 (str of np.float32, numpy >= 1.14)."""
    return [float(str(v)) for v in np.asarray(x, dtype=np.float32).reshape(-1)]


def _shape(s):
    """One PhysX collision shape: type, pose in its body frame and the size parameters PhysX holds."""
    import sapien.physx as px
    known = (px.PhysxCollisionShapeBox, px.PhysxCollisionShapeSphere, px.PhysxCollisionShapeCapsule, px.PhysxCollisionShapeCylinder, px.PhysxCollisionShapeConvexMesh, px.PhysxCollisionShapeTriangleMesh, px.PhysxCollisionShapePlane)
    if not isinstance(s, known): raise RuntimeError(f'unknown collision shape type {type(s).__name__}')
    lp = s.get_local_pose(); d = {'type': type(s).__name__, 'local_pose': f32(lp.p) + f32(lp.q)}
    if isinstance(s, px.PhysxCollisionShapeBox): d['half_size'] = f32(s.half_size)
    elif isinstance(s, px.PhysxCollisionShapeSphere): d['radius'] = f32(s.radius)[0]
    elif isinstance(s, (px.PhysxCollisionShapeCapsule, px.PhysxCollisionShapeCylinder)): d.update(radius=f32(s.radius)[0], half_length=f32(s.half_length)[0])
    elif isinstance(s, (px.PhysxCollisionShapeConvexMesh, px.PhysxCollisionShapeTriangleMesh)):
        d.update(scale=f32(s.scale), vertices=np.asarray(f32(s.get_vertices())).reshape(-1, 3).tolist(), triangles=np.asarray(s.get_triangles()).astype(int).tolist())
    elif isinstance(s, px.PhysxCollisionShapePlane): pass
    else: raise RuntimeError(f'unknown collision shape type {type(s).__name__}')
    return d


def pair_impulses(contacts):
    """Per contact: the summed impulse of its points, exactly as sapien_utils.compute_total_impulse forms it."""
    return [np.sum([p.impulse for p in c.points], axis=0) for c in contacts]


def pair_force(contacts, imps, a, b, dt):
    """= norm(sapien_utils.get_pairwise_contact_impulse(contacts, a, b)) / dt: same operations, same order."""
    total = np.zeros(3)
    for c, ci in zip(contacts, imps):
        e0, e1 = c.bodies[0].entity, c.bodies[1].entity
        if e0 == a and e1 == b: total += ci * 1
        elif e0 == b and e1 == a: total += ci * -1
    return float(np.linalg.norm(total) / dt)


def probe_spatial(u, t, action=None):
    from . import recovery_v1 as RV
    from .trees import as_numpy
    row = RV.probe(u, t, action)                                       # unchanged existing fields, computed first
    if getattr(u.scene, 'gpu_sim_enabled', False): raise RuntimeError('probe_spatial reads the CPU contact list; the GPU backend is not supported')
    contacts = u.scene.get_contacts(); dt = float(u.scene.px.timestep); ag = u.agent; imps = pair_impulses(contacts)
    ent = lambda o: o._bodies[0].entity
    A, B, table = ent(u.cubeA), ent(u.cubeB), ent(u.table_scene.table)
    lk = ag.robot.get_links(); links = {l.name: ent(l) for l in lk}; robot = set(links.values())
    others = sorted([e for e in u.scene.get_all_actors() if e not in robot and e not in (A, B, table)], key=lambda e: e.name)      # e.g. the ground plane
    name = {e: n for n, e in links.items()}; name.update({A: 'cubeA', B: 'cubeB', table: 'table'}); nm = lambda e: name.get(e, e.name)
    f = lambda a, b: pair_force(contacts, imps, a, b, dt)
    hand = links['panda_hand']; fl = ent(ag.finger1_link); fr = ent(ag.finger2_link)
    touching = {c.bodies[i].entity for c in contacts for i in (0, 1)}                 # a body in no contact pair has zero force with everything

    def shape_index(body, shape):
        for i, s in enumerate(body.collision_shapes):
            if s == shape: return i
        raise RuntimeError(f'contact shape not found among the collision shapes of {nm(body.entity)}')
    cl = sorted(({'a': nm(c.bodies[0].entity), 'b': nm(c.bodies[1].entity), 'shape_a': shape_index(c.bodies[0], c.shapes[0]), 'shape_b': shape_index(c.bodies[1], c.shapes[1]),
                  'points': [f32(p.position) + f32(p.normal) + f32(p.impulse) + f32(p.separation) for p in c.points]} for c in contacts), key=CONTACT_ORDER)
    per = lambda o: {n: v for n, e in links.items() if e in touching and (v := f(e, o)) > 0} if o in touching else {}
    steps = int(as_numpy(u.elapsed_steps).reshape(-1)[0]); cf = float(u.control_freq); sf = float(u.sim_freq)
    new = {'spatial_schema': SCHEMA, 'elapsed_steps': steps, 'sim_time_s': steps / cf, 'sim_timestep_s': dt, 'sim_substeps_per_control': sf / cf,
           'robot_qpos': f32(as_numpy(ag.robot.get_qpos())), 'robot_qvel': f32(as_numpy(ag.robot.get_qvel())),
           'link_pose': {l.name: f32(ep.p) + f32(ep.q) for l in lk for ep in (l._bodies[0].entity_pose,)}, 'contacts': cl,
           'f_hand_A': f(hand, A), 'f_hand_B': f(hand, B), 'f_B_table': f(B, table), 'f_hand_table': f(hand, table), 'f_fingerL_table': f(fl, table), 'f_fingerR_table': f(fr, table),
           'f_links_A': per(A), 'f_links_B': per(B), 'f_links_table': per(table),
           'f_links_other': {f'{n}|{nm(o)}': v for o in others for n, v in per(o).items()}}
    if action is None:                                                   # the initial row: static geometry once per scene
        bodies = {**links, 'cubeA': A, 'cubeB': B, 'table': table, **{nm(o): o for o in others}}
        geo = {}
        for n, e in bodies.items():
            comp = next((c for c in e.components if hasattr(c, 'collision_shapes')), None); ep = e.pose
            geo[n] = {'pose_at_this_row': f32(ep.p) + f32(ep.q), 'shapes': [] if comp is None else [_shape(s) for s in comp.collision_shapes]}
        new.update(collision_geometry=geo, robot_joint_names=[j.name for j in ag.robot.get_active_joints()])
    clash = set(new) & set(row)
    if clash: raise RuntimeError(f'probe_spatial would overwrite existing fields {sorted(clash)}')
    row.update(new); return row


def gt_probe_spatial(rt, z):
    """Drop-in for scripts/31_recovery_v1_oracle_xy.gt_probe (the probe the C1 driver passes)."""
    return probe_spatial(rt.env, rt.t, z)


# ---------------------------------------------------------------- logging ON/OFF comparison (pure; used by the driver's probe-check)
TIMING = ('wall_seconds',)                       # measured durations differ run to run; everything else must be identical


def _strip(rec, drop_initial=()):
    import json
    rec = json.loads(json.dumps(rec)); rec.pop('wall_seconds', None); rec.get('memory', {}).pop('seconds', None)
    for k in ('reading_seconds', 'memory_seconds'): rec.get('counts', {}).pop(k, None)
    for c in rec.get('cells', []): c.pop('wall_seconds', None)
    if rec.get('initial_state') is not None: rec['initial_state'] = {k: v for k, v in rec['initial_state'].items() if k not in drop_initial}
    return rec


def compare_on_off(off, on):
    """off, on = (record, traces, frames) of the same scene with the earlier logger and with the spatial logger.
    Bitwise: the record without durations, every frame array, every earlier trace field of every row; and the spatial
    fields present in every row of `on` and in no row of `off`."""
    import json
    (r0, t0, f0), (r1, t1, f1) = off, on
    dump = lambda x: json.dumps(x, sort_keys=True)
    record = dump(_strip(r0)) == dump(_strip(r1, ADDED_FIELDS))
    frames = set(f0) == set(f1) and all(len(f0[k]) == len(f1[k]) and all(np.asarray(x).dtype == np.asarray(y).dtype and np.array_equal(x, y) for x, y in zip(f0[k], f1[k])) for k in f0)
    traces = t0.keys() == t1.keys() and all(len(t0[k]) == len(t1[k]) and all(dump(x) == dump({kk: v for kk, v in y.items() if kk not in ADDED_FIELDS}) for x, y in zip(t0[k], t1[k])) for k in t0)
    rows_on = [y for k in t1 for y in t1[k]]; rows_off = [x for k in t0 for x in t0[k]]
    present = bool(rows_on) and all(set(STEP_FIELDS) <= set(y) and y['spatial_schema'] == SCHEMA for y in rows_on) and set(ADDED_FIELDS) <= set(r1.get('initial_state') or {})
    absent = not any(set(ADDED_FIELDS) & set(x) for x in rows_off) and not (set(ADDED_FIELDS) & set(r0.get('initial_state') or {}))
    out = {'record_equal_except_durations': record, 'frames_bitwise_equal': frames, 'earlier_trace_fields_equal': traces, 'spatial_fields_in_every_on_row': present, 'no_spatial_field_without_the_logger': absent}
    out['pass'] = bool(all(out.values())); return out


def rng_snapshot():
    """Global random states (python, numpy, torch CPU and CUDA), so both runs of a comparison start from the same states."""
    import random, torch
    s = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available(): s['cuda'] = torch.cuda.get_rng_state_all()
    return s


def rng_restore(s):
    import random, torch
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if 'cuda' in s: torch.cuda.set_rng_state_all(s['cuda'])


def rng_digest():
    import hashlib, pickle, random, torch
    h = hashlib.sha256(); h.update(pickle.dumps(random.getstate())); st = np.random.get_state(); h.update(pickle.dumps((st[0], st[1].tobytes(), st[2:])))
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for x in torch.cuda.get_rng_state_all(): h.update(x.numpy().tobytes())
    return h.hexdigest()
