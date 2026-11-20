"""PROPOSAL, logging only. NOT used by C1 plan dd3af27e5484b9ba395820470d5adf5c7865a820514dd23964483c06ecd699a6.

An extension of hci.recovery_v1.probe (the ground-truth logger of every Vision-CENTER trace) that adds what the spatial
audit cannot derive from the current trace:
  robot_qpos            all 9 joint values (7 arm + 2 finger) -> arm link poses by forward kinematics from the URDF
  f_hand_A, f_hand_B    contact force hand - held cube, hand - target
  f_B_table             target - table
  f_hand_table, f_fingerL_table, f_fingerR_table
  f_links_B, f_links_table   {robot link name: force} for every robot link with a non-zero force against the target /
                        the table (sparse; an empty dict means: no robot link had a non-zero force in the last sub-step)
Forces use the same definition as the existing fields: |sum of contact impulses of the pair| / PhysX time step, from the
contacts of the LAST PhysX sub-step of the control step (as get_pairwise_contact_forces on the CPU backend).

Why it cannot change control: the probe is called by pilot2b.Hooks only AFTER Runtime.advance has stepped the simulator
and produced the next observation; its return value is appended to the trace rows. The added calls only read simulator
state: robot.get_qpos(), scene.get_contacts() (PhysX contact list of the last sub-step) and pure numpy over it. Existing
fields are produced by the unchanged hci.recovery_v1.probe first and are not modified. To take effect, the C1 driver
would pass probe=gt_probe_spatial instead of X.gt_probe; that is a change of a plan-hashed file and needs a new plan,
a new hash and an independent review.
"""
import numpy as np


def probe_spatial(u, t, action=None):
    from . import recovery_v1 as RV
    from .trees import as_numpy
    row = RV.probe(u, t, action)                                       # unchanged existing fields, computed first
    if getattr(u.scene, 'gpu_sim_enabled', False): raise RuntimeError('probe_spatial reads the CPU contact list; the GPU backend is not supported')
    from mani_skill.utils import sapien_utils
    contacts = u.scene.get_contacts(); dt = float(u.scene.px.timestep); ag = u.agent
    ent = lambda o: o._bodies[0].entity
    A, B, table = ent(u.cubeA), ent(u.cubeB), ent(u.table_scene.table); links = {l.name: ent(l) for l in ag.robot.get_links()}
    f = lambda a, b: float(np.linalg.norm(sapien_utils.get_pairwise_contact_impulse(contacts, a, b)) / dt)
    hand = links['panda_hand']; fl = ent(ag.finger1_link); fr = ent(ag.finger2_link)
    new = {'robot_qpos': [float(z) for z in as_numpy(ag.robot.get_qpos()).reshape(-1)],
           'f_hand_A': f(hand, A), 'f_hand_B': f(hand, B), 'f_B_table': f(B, table), 'f_hand_table': f(hand, table), 'f_fingerL_table': f(fl, table), 'f_fingerR_table': f(fr, table),
           'f_links_B': {n: v for n, e in links.items() if (v := f(e, B)) > 0}, 'f_links_table': {n: v for n, e in links.items() if (v := f(e, table)) > 0}}
    clash = set(new) & set(row)
    if clash: raise RuntimeError(f'probe_spatial would overwrite existing fields {sorted(clash)}')
    row.update(new); return row


def gt_probe_spatial(rt, z):
    """Drop-in for scripts/31_recovery_v1_oracle_xy.gt_probe (the probe the C1 driver passes)."""
    return probe_spatial(rt.env, rt.t, z)


ADDED_FIELDS = ('robot_qpos', 'f_hand_A', 'f_hand_B', 'f_B_table', 'f_hand_table', 'f_fingerL_table', 'f_fingerR_table', 'f_links_B', 'f_links_table')
