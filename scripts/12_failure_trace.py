"""Per-step diagnostic trace of a few development episodes of one fixed policy (StackCube-v1 only).

Diagnostic logging only. Simulator ground truth (cube poses, grasp flag) is read AFTER each step and written to a
file; it is never passed to the policy or to a monitor. Each traced episode must reproduce the outcome recorded by
the reference evaluation run for the same seed, otherwise the script fails (the trace would not describe that run).

The stage labels are descriptions of what was observed (where the episode stopped making progress), not causes.
"""
from pathlib import Path
import argparse, json
import imageio.v2 as imageio
import numpy as np
from hci.io import load_config, read_json, write_json, new_run, finish_run
from hci.trees import as_numpy

NEAR_M = 0.03      # TCP within 3 cm of the cubeA centre counts as "reached the cube"
ALIGN_M = 0.02     # cubeA centre within 2 cm (xy) of the cubeB centre, the cube half-width
LIFT_M = 0.01      # cubeA centre raised 1 cm above its initial height


def classify(rows):
    """Furthest stage reached. rows: per-step dicts after each control step."""
    if not rows: return 'unreadable', {}
    z0 = rows[0]['cubeA'][2]
    near = [r['t'] for r in rows if r['tcp_to_cubeA'] <= NEAR_M]
    grasp = [r['t'] for r in rows if r['is_cubeA_grasped']]
    lift = [r['t'] for r in rows if r['cubeA'][2] - z0 >= LIFT_M]
    align = [r['t'] for r in rows if r['cubeA_to_cubeB_xy'] <= ALIGN_M and r['cubeA'][2] - z0 >= LIFT_M]
    on = [r['t'] for r in rows if r['is_cubeA_on_cubeB']]
    first = lambda v: v[0] if v else None
    facts = {'first_near_step': first(near), 'first_grasp_step': first(grasp), 'grasp_steps': len(grasp),
             'last_grasp_step': grasp[-1] if grasp else None, 'first_lift_step': first(lift),
             'first_aligned_step': first(align), 'first_on_cubeB_step': first(on), 'on_cubeB_steps': len(on),
             'min_tcp_to_cubeA_m': min(r['tcp_to_cubeA'] for r in rows),
             'min_cubeA_to_cubeB_xy_m': min(r['cubeA_to_cubeB_xy'] for r in rows),
             'max_cubeA_lift_m': max(r['cubeA'][2] - z0 for r in rows),
             'cubeA_xy_displacement_m': float(np.linalg.norm(np.array(rows[-1]['cubeA'][:2]) - np.array(rows[0]['cubeA'][:2]))),
             'cubeB_xy_displacement_m': float(np.linalg.norm(np.array(rows[-1]['cubeB'][:2]) - np.array(rows[0]['cubeB'][:2]))),
             'final_gripper_command': rows[-1]['action'][-1], 'final_is_grasped': bool(rows[-1]['is_cubeA_grasped'])}
    if rows[-1]['success']: return 'success', facts
    if not near: return 'approach', facts
    if not grasp or not lift: return 'grasp', facts
    if not align: return 'transport_align', facts
    return 'place_stack', facts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--policy', required=True); p.add_argument('--out', required=True)
    p.add_argument('--reference', required=True, help='finished 09_eval_policy run of the same policy (episodes.jsonl)')
    p.add_argument('--seeds', type=int, nargs='+', required=True); p.add_argument('--upstream', default='third_party/ManiSkill')
    p.add_argument('--videos', type=int, default=4, help='write a video for the first N traced episodes')
    a = p.parse_args(); c = load_config(a.config)
    if c['env']['id'] != 'StackCube-v1': p.error('This trace reads StackCube-v1 objects only')
    ref = {}
    for line in (Path(a.reference) / 'episodes.jsonl').read_text().splitlines():
        row = json.loads(line); ref[row['seed']] = row
    if read_json(Path(a.reference) / 'policy_eval.json')['policy'] != a.policy: p.error('Reference run used another policy')
    missing = [s for s in a.seeds if s not in ref]
    if missing: p.error(f'Seeds not in the reference run: {missing}')
    from hci.experiment import runtime
    out = new_run(a.out, 'failure-trace-development', c); r = runtime(c, a.policy, a.upstream); u = r.env.unwrapped
    summary = []

    def state(action):
        e = u.evaluate(); A = as_numpy(u.cubeA.pose.p)[0]; B = as_numpy(u.cubeB.pose.p)[0]; T = as_numpy(u.agent.tcp.pose.p)[0]
        flag = lambda k: bool(as_numpy(e[k]).reshape(-1)[0])
        return {'t': r.t, 'tcp': T.tolist(), 'cubeA': A.tolist(), 'cubeB': B.tolist(),
                'tcp_to_cubeA': float(np.linalg.norm(T - A)), 'cubeA_to_cubeB_xy': float(np.linalg.norm(A[:2] - B[:2])),
                'is_cubeA_grasped': flag('is_cubeA_grasped'), 'is_cubeA_on_cubeB': flag('is_cubeA_on_cubeB'),
                'is_cubeA_static': flag('is_cubeA_static'), 'success': flag('success'),
                'action': None if action is None else [float(x) for x in action]}

    def frame():
        return np.concatenate([as_numpy(r.raw['sensor_data'][n]['rgb'])[0] for n in sorted(r.raw['sensor_data'])], axis=1).astype(np.uint8)

    try:
        for i, seed in enumerate(a.seeds):
            r.reset(seed); rows = []; initial = state(None); frames = [frame()] if i < a.videos else None; old = type(r).advance

            def advance(action, old=old, rows=rows, frames=frames):
                z = old(r, action); rows.append(state(z))
                if frames is not None: frames.append(frame())
                return z
            r.advance = advance
            try: res = r.complete()
            finally: del r.advance
            if (int(res['success']), res['steps']) != (ref[seed]['success_once'], ref[seed]['steps']):
                raise RuntimeError(f'seed {seed}: traced outcome {res["success"]}/{res["steps"]} differs from the reference run')
            if frames is not None:
                with imageio.get_writer(str(out / f'seed_{seed}.mp4'), fps=20) as w:
                    for f in frames: w.append_data(f)
            label, facts = classify(rows)
            with (out / f'trace_{seed}.jsonl').open('x', encoding='utf-8') as f:
                f.write(json.dumps({'initial': initial}) + '\n')
                for row in rows: f.write(json.dumps(row) + '\n')
            summary.append({'seed': seed, 'success_once': int(res['success']), 'steps': res['steps'], 'stage_label': label, **facts})
            print(json.dumps(summary[-1]), flush=True)
        write_json(out / 'failure_trace.json', {'policy': a.policy, 'reference': a.reference, 'episodes': summary,
                   'thresholds': {'near_m': NEAR_M, 'align_xy_m': ALIGN_M, 'lift_m': LIFT_M},
                   'labels': {'approach': 'TCP never came within near_m of cubeA', 'grasp': 'reached cubeA but never grasped-and-lifted it',
                              'transport_align': 'lifted cubeA but never brought it over cubeB', 'place_stack': 'brought cubeA over cubeB without task success',
                              'success': 'task success at the end of the run'},
                   'scope': 'Observed furthest stage per episode. Descriptive; not a cause. Ground truth is logged only, never a policy input.'})
        finish_run(out, n=len(summary))
    finally:
        r.env.close()


if __name__ == '__main__':
    main()
