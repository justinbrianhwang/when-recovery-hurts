"""S6 sub-step instrumentation replay (work s6_substep_001). POST_HOC_VALIDATION; never changes the official S6 FAIL.

Re-runs USED C1 development scene-runs with the frozen C1 code, policies and arms (hci.vision_center_c1.collect_scene,
BASE_ARMS), twice per scene-run:
  OFF  exactly as C1 (no extra logging)
  ON   the same, with every physics sub-step logged: ManiSkillScene.step (the Python method that calls px.step once per
       sub-step) is wrapped on the scene instance; before and after each sub-step the poses of every robot link and of
       both cubes, the robot qpos/qvel and the arm controller's target qpos are read, and after it the full PhysX
       contact list (same serialisation as hci.spatial_probe: bodies, shape indices, points = position, normal, impulse,
       separation). The restore prefix replay is not logged (it reproduces the trunk); segments: trunk, then each arm
       from the trigger.
Checks: ON vs OFF trace and outcome identical (logging changes nothing); both vs the ORIGINAL C1 record (reported; if
they differ, nothing here claims to reproduce the original run).

Scene-runs fixed before running (rule: every scene-run listed in posthoc_physics_diagnostics_001/S6_EXTREME_CASES.csv,
plus the scene-runs of the two S5 FAIL events found by posthoc_hull_sensitivity_001): see SCENE_RUNS.
"""
from __future__ import annotations

import argparse, gzip, json, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

SCENE_RUNS = [('train_seed_1', 51124), ('train_seed_1', 51211), ('train_seed_1', 51248), ('train_seed_1', 51157), ('train_seed_1', 51143), ('train_seed_2', 51152),
              ('train_seed_2', 51272), ('train_seed_2', 51157), ('train_seed_2', 51267), ('train_seed_2', 51143), ('train_seed_1', 51270)]
POLICY_PATHS = {'train_seed_1': 'artifacts/policies/stack_v1_s1_d500_clean_resnet18_i080000', 'train_seed_2': 'artifacts/policies/stack_v1_s2_d500_clean_resnet18_i080000'}
C1_ROLE = 'vision_center_v2_c1_dev'


class SubstepLogger:
    """Wraps scene.step of one ManiSkill env instance and the Runtime reset/restore (segments). Reads only."""
    def __init__(self):
        self.segments = []; self.on = False; self._orig = {}

    def state(self, u):
        from hci.spatial_probe import f32
        from hci.trees import as_numpy
        ag = u.agent; lk = ag.robot.get_links()
        s = {'link_pose': {l.name: f32(ep.p) + f32(ep.q) for l in lk for ep in (l._bodies[0].entity_pose,)},
             'cubeA': f32(as_numpy(u.cubeA.pose.raw_pose)[0]), 'cubeB': f32(as_numpy(u.cubeB.pose.raw_pose)[0]),
             'qpos': f32(as_numpy(ag.robot.get_qpos())[0]), 'qvel': f32(as_numpy(ag.robot.get_qvel())[0])}
        arm = ag.controller.controllers.get('arm') if hasattr(ag.controller, 'controllers') else None
        tq = getattr(arm, '_target_qpos', None)
        s['arm_target_qpos'] = None if tq is None else f32(as_numpy(tq)[0])
        return s

    def contacts(self, u):
        from hci.spatial_probe import f32, CONTACT_ORDER
        ent = lambda o: o._bodies[0].entity
        lk = u.agent.robot.get_links(); name = {ent(l): l.name for l in lk}; name.update({ent(u.cubeA): 'cubeA', ent(u.cubeB): 'cubeB', ent(u.table_scene.table): 'table'})
        nm = lambda e: name.get(e, e.name)

        def idx(body, shape):
            for i, s in enumerate(body.collision_shapes):
                if s == shape: return i
            return -1
        return sorted(({'a': nm(c.bodies[0].entity), 'b': nm(c.bodies[1].entity), 'shape_a': idx(c.bodies[0], c.shapes[0]), 'shape_b': idx(c.bodies[1], c.shapes[1]),
                        'points': [f32(p.position) + f32(p.normal) + f32(p.impulse) + f32(p.separation) for p in c.points]} for c in u.scene.get_contacts()), key=CONTACT_ORDER)

    def _wrap(self, u):
        """Wrap scene.step of the CURRENT scene. ManiSkill may build a new scene object at every env.reset (try 1 wrapped
        only the first scene and logged nothing), so this runs after every reset and restore."""
        from hci.trees import as_numpy
        sc = u.scene
        if getattr(sc, '_substep_logger_wrapped', False): return
        orig_step = sc.step; log = self
        def step():
            if not log.on: return orig_step()
            pre = log.state(u); el = int(as_numpy(u.elapsed_steps).reshape(-1)[0]); seg = log.segments[-1]
            k = sum(1 for x in seg['substeps'] if x['control_step'] == el + 1)
            out = orig_step(); seg['substeps'].append({'control_step': el + 1, 'k': k, 'pre': pre, 'post': log.state(u), 'contacts': log.contacts(u)}); return out
        sc.step = step; sc._substep_logger_wrapped = True; self._wrapped.append((sc, orig_step))

    def install(self, r):
        from hci import simulation as SIM
        self._wrapped = []; log = self; u = r.env
        R = SIM.Runtime; self._orig = {'reset': R.reset, 'restore': R.restore}; o = self._orig
        def reset(rt, episode):
            log.on = False; out = o['reset'](rt, episode); log._wrap(rt.env); log.segments.append({'kind': 'trunk', 'substeps': []}); log.on = True; return out
        def restore(rt, s):
            log.on = False; out = o['restore'](rt, s); log._wrap(rt.env); log.segments.append({'kind': 'arm_from_trigger', 't': rt.t, 'substeps': []}); log.on = True; return out
        R.reset, R.restore = reset, restore

    def uninstall(self):
        from hci import simulation as SIM
        for k, v in self._orig.items(): setattr(SIM.Runtime, k, v)
        for sc, f in self._wrapped:
            sc.step = f; sc._substep_logger_wrapped = False
        self.on = False


def canon(x): return json.dumps(json.loads(json.dumps(x, default=float)), sort_keys=True)          # through a JSON round trip, as the records are stored


def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', required=True); a.add_argument('--c1-records', required=True, help='folder with pass_<policy>/scenes and traces of the ORIGINAL C1 run')
    a.add_argument('--upstream', default='third_party/ManiSkill'); a.add_argument('--device', default='cuda'); a.add_argument('--only', nargs='*'); a = a.parse_args()
    out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    out.mkdir(parents=True)
    from hci import vision_center_c1 as C1
    from hci import vision_v1 as VV
    from hci import spatial_probe as SP
    from hci.io import load_config
    c = load_config('configs/stackcube.yaml'); summary = {'work': 's6_substep_001', 'label': 'POST_HOC_VALIDATION', 'scene_runs': [], 'all_on_equals_off': True}
    todo = [sr for sr in SCENE_RUNS if not a.only or f'{sr[0]}:{sr[1]}' in a.only]
    for pid in sorted({p for p, _ in todo}):
        r = VV.runtime(c, POLICY_PATHS[pid], a.upstream, a.device)
        try:
            for _, seed in [sr for sr in todo if sr[0] == pid]:
                t0 = time.time(); e = {'policy': pid, 'scene': seed}
                try:
                    rec_off, tr_off, _ = C1.collect_scene(r, seed, c, probe=SP.gt_probe_spatial, arms=C1.BASE_ARMS, role=C1_ROLE, policy_id=pid)
                    lg = SubstepLogger(); lg.install(r)
                    try: rec_on, tr_on, _ = C1.collect_scene(r, seed, c, probe=SP.gt_probe_spatial, arms=C1.BASE_ARMS, role=C1_ROLE, policy_id=pid)
                    finally: lg.uninstall()
                    if sum(len(x['substeps']) for x in lg.segments) == 0: raise RuntimeError('the sub-step logger recorded nothing (scene.step was not wrapped)')
                    strip = lambda rec: {k: v for k, v in rec.items() if k not in ('wall_seconds',)} | {'cells': [{kk: vv for kk, vv in x.items() if kk != 'wall_seconds'} for x in rec['cells']]}
                    same = canon(tr_off) == canon(tr_on) and canon(strip(rec_off)['cells']) == canon(strip(rec_on)['cells'])
                    orig_rec = json.loads((Path(a.c1_records) / f'pass_{pid}' / 'scenes' / f'{seed}.json').read_text(encoding='utf-8'))
                    with gzip.open(Path(a.c1_records) / f'pass_{pid}' / orig_rec['trace'], 'rt', encoding='utf-8') as f: orig_tr = json.load(f)['traces']
                    same_orig = canon(orig_tr) == canon(tr_off); same_cells = [(x['arm'], x['y'], x['steps']) for x in orig_rec['cells']] == [(x['arm'], x['y'], x['steps']) for x in rec_off['cells']]
                    e.update(on_equals_off=bool(same), off_equals_original_trace=bool(same_orig), outcomes_equal_original=bool(same_cells), segments=[{'kind': s['kind'], 't': s.get('t'), 'substeps': len(s['substeps'])} for s in lg.segments],
                             seconds=time.time() - t0)
                    with gzip.open(out / f'substeps_{pid}_{seed}.json.gz', 'wt', encoding='utf-8') as f: json.dump({'policy': pid, 'scene': seed, 'arms': list(C1.BASE_ARMS), 'segments': lg.segments}, f)
                    with gzip.open(out / f'trace_on_{pid}_{seed}.json.gz', 'wt', encoding='utf-8') as f: json.dump({'traces': tr_on, 'record': rec_on}, f, default=float)
                    summary['all_on_equals_off'] &= bool(same)
                except Exception as ex:
                    e.update(error=repr(ex), traceback=traceback.format_exc()); summary['all_on_equals_off'] = False
                summary['scene_runs'].append(e); print(json.dumps({k: v for k, v in e.items() if k not in ('traceback', 'segments')}), flush=True)
        finally:
            r.env.close()
    (out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    if any('error' in e for e in summary['scene_runs']): raise SystemExit(3)


if __name__ == '__main__':
    main()
