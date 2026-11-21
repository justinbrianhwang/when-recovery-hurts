"""The spatial probe on the REAL pinned simulator (ManiSkill a4a4f92, SAPIEN 3.0.3, PhysX CPU), without policy, cameras or
GPU: one already-used audit scene (50400) driven by a fixed scripted action sequence that touches both cubes and the table.
SKIPPED where ManiSkill is not installed (e.g. the Windows development machine); runs in the Linux environment and in the
server's pytest step. Tolerance 0 everywhere."""
import hashlib, importlib.util, json, pickle, random
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('sapien')
pytest.importorskip('mani_skill')
from hci import spatial_probe as P
from hci import recovery_v1 as RV

SEED = 50400                                                             # vision_v1_select, already used (the C1 audit scenes)


def actions():
    """Down to the held cube's height, close, lift, move towards the target, lower: contacts with cubes and table."""
    return [[0, 0, -1, 1]] * 22 + [[0, 0, 0, -1]] * 8 + [[0, 0, 1, -1]] * 8 + [[1, 0, 0, -1]] * 10 + [[0, 1, -1, -1]] * 12 + [[0, 0, -1, 1]] * 6


@pytest.fixture(scope='module')
def sim():
    spec = importlib.util.spec_from_file_location('c1_sim_check', Path('scripts/37_c1_probe_sim_check.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    env = m.make_env(); yield m, env; env.close()


def test_logging_leaves_the_simulation_bit_identical_and_draws_no_random_number(sim):
    m, env = sim; acts = actions()
    runs = [m.run_branch(env, SEED, acts, on) for on in (False, True, False, True)]
    assert all(r[0] == runs[0][0] for r in runs) and len(runs[0][0]) == len(acts) + 1            # state, observation, reward, info after every step
    assert runs[1][2] == [] and runs[3][2] == []                                                    # every RNG state equal before and after every probe call
    dump = lambda rows: json.dumps(rows, sort_keys=True)
    assert dump(runs[1][1]) == dump(runs[3][1])                                                     # the record itself is reproducible (contact list stored sorted)


def test_forces_poses_and_contact_values_equal_what_the_engine_and_upstream_report(sim):
    m, env = sim
    from mani_skill.utils import sapien_utils as su
    env.reset(seed=SEED); ent = lambda o: o._bodies[0].entity; checked = 0; touched = set()
    for i, a in enumerate(actions(), 1):
        env.step(torch.as_tensor(np.asarray(a, np.float32)[None])); r = P.probe_spatial(env, i, a); base = RV.probe(env, i, a)
        assert {k: r[k] for k in base} == base and r['elapsed_steps'] == i and r['sim_substeps_per_control'] == 5.
        c = env.scene.get_contacts(); dt = float(env.scene.px.timestep)
        bodies = {**{l.name: ent(l) for l in env.agent.robot.get_links()}, 'cubeA': ent(env.cubeA), 'cubeB': ent(env.cubeB), 'table': ent(env.table_scene.table)}
        imps = P.pair_impulses(c)
        for ea in bodies.values():
            for eb in bodies.values():
                assert P.pair_force(c, imps, ea, eb, dt) == float(np.linalg.norm(su.get_pairwise_contact_impulse(c, ea, eb)) / dt); checked += 1
        for l in env.agent.robot.get_links():
            ref = np.concatenate([l.pose.p.numpy().reshape(-1), l.pose.q.numpy().reshape(-1)]).astype(np.float32)
            assert np.array_equal(np.asarray(r['link_pose'][l.name], np.float32), ref)
        pts = sorted(np.concatenate([p.position, p.normal, p.impulse, [p.separation]]).astype(np.float32).tobytes() for cc in c for p in cc.points)
        assert pts == sorted(np.asarray(q, np.float32).tobytes() for cc in r['contacts'] for q in cc['points'])     # lossless float32, nothing dropped or added
        assert np.array_equal(np.asarray(r['robot_qpos'], np.float32), env.agent.robot.get_qpos().numpy().reshape(-1))
        touched |= {(x['a'], x['b']) for x in r['contacts'] if any(any(v != 0 for v in q[6:9]) for q in x['points'])}
    assert checked > 10000 and {('cubeA', 'table'), ('cubeB', 'table')} <= touched and any('finger' in a or 'finger' in b for a, b in touched)


def test_the_initial_row_holds_the_collision_geometry_physx_uses(sim):
    m, env = sim; env.reset(seed=SEED); r = P.probe_spatial(env, 0, None); g = r['collision_geometry']
    assert r['robot_joint_names'] == [f'panda_joint{i}' for i in range(1, 8)] + ['panda_finger_joint1', 'panda_finger_joint2']
    for l in env.agent.robot.get_links():
        shapes = l._bodies[0].collision_shapes; assert len(g[l.name]['shapes']) == len(shapes)
        for s, d in zip(shapes, g[l.name]['shapes']):
            assert d['type'] == type(s).__name__
            if hasattr(s, 'get_vertices'): assert np.array_equal(np.asarray(d['vertices'], np.float32), np.asarray(s.get_vertices(), np.float32))
            if hasattr(s, 'half_size'): assert np.array_equal(np.asarray(d['half_size'], np.float32), np.asarray(s.half_size, np.float32))
    assert g['cubeA']['shapes'][0]['half_size'] == [.02, .02, .02] and g['camera_link']['shapes'][0]['type'] == 'PhysxCollisionShapeBox' and g['table']['shapes'][0]['type'] == 'PhysxCollisionShapeBox'
    assert {n for n in g if g[n]['shapes']} >= {f'panda_link{i}' for i in range(8)} | {'panda_hand', 'panda_leftfinger', 'panda_rightfinger', 'camera_link', 'cubeA', 'cubeB', 'table'}
