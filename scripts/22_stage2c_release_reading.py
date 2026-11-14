"""Stage 2C supplement, added AFTER the main Stage 2C tables were seen (run stage2c_001 was not changed).

Question that the main event table raised: in blocks where the cube was grasped at the checkpoint, the grasp flag
always stayed on during the retreat and turned off later. What was commanded and where was the cube when the flag
first turned off, in the Retreat-Retry arm (after control returned) and in the Continue arm (after the checkpoint)?

Reads the same saved traces and the event rows of stage2c_001. CPU only, no simulator, no model. Observations only:
"gripper command > 0 at the step where the flag turned off" is what was commanded, not why.
"""
from pathlib import Path
import argparse, csv, importlib.util, json
import numpy as np

_spec = importlib.util.spec_from_file_location('stage2c', Path(__file__).with_name('21_stage2c_offline_review.py'))
s2c = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s2c)


def first_release(d, start):
    """First row index >= start at which the grasp flag is off after having been on in the previous row (or at the start state)."""
    g = d['grasp']
    for i in range(max(start, 1), len(g)):
        if g[i - 1] and not g[i]: return i
    return None


def reading(d, i, t_ref):
    if i is None: return {'released': False}
    return {'released': True, 'steps_after_reference': int(d['t'][i] - t_ref), 'gripper_command_open': bool(d['grip'][i] > 0), 'cube_lift_m': float(d['lift'][i - 1]),
            'cubeA_to_cubeB_xy_m': float(d['xy'][i - 1]), 'over_cubeB_within_align': bool(d['xy'][i - 1] <= s2c.ALIGN_M), 'on_cubeB_flag': bool(d['on'][i] or d['on'][i - 1])}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--events', required=True); p.add_argument('--traces', required=True); p.add_argument('--scenes', required=True)
    p.add_argument('--out', required=True); a = p.parse_args()
    rows = [r for r in s2c.read_csv(a.events) if s2c.boolean(r['cp_grasped'])]; out = []
    for r in rows:
        scene, t, rep, k = int(r['scene']), int(r['checkpoint']), int(r['repeat']), int(r['retreat_steps_used'])
        z0 = float(s2c.read_json(Path(a.scenes) / f'{scene}.json')['initial_state']['cubeA'][2])
        with np.load(Path(a.traces) / f'{scene}.npz') as z:
            cp = z['trunk'][t - 1:t]; R = s2c.derived(np.concatenate([cp, z[f'b_{t}_{rep}_R']]), z0); C = s2c.derived(np.concatenate([cp, z[f'b_{t}_{rep}_C0']]), z0)
        # row 0 is the checkpoint state; R rows 1..k are the retreat, so the search starts at the first policy step after the return
        out.append({'scene': scene, 'checkpoint': t, 'repeat': rep, 'R_vs_C0': r['R_vs_C0'], **{'R_' + n: v for n, v in reading(R, first_release(R, k + 1), t + k).items()},
                    **{'C0_' + n: v for n, v in reading(C, first_release(C, 1), t).items()}})
    summary = {}
    for name in ('1to0', '0to1', '1to1', '0to0'):
        sel = [x for x in out if x['R_vs_C0'] == name]; row = {'blocks_grasped_at_checkpoint': len(sel), 'scenes': len({x['scene'] for x in sel})}
        for arm in ('R', 'C0'):
            rel = [x for x in sel if x[arm + '_released']]
            row[arm] = {'grasp_flag_turned_off': len(rel), 'gripper_command_open_at_that_step': sum(x[arm + '_gripper_command_open'] for x in rel),
                        'cube_within_2cm_xy_of_cubeB_just_before': sum(x[arm + '_over_cubeB_within_align'] for x in rel),
                        'on_cubeB_flag_at_that_step': sum(x[arm + '_on_cubeB_flag'] for x in rel),
                        'median_steps_after_reference': float(np.median([x[arm + '_steps_after_reference'] for x in rel])) if rel else 'NA',
                        'median_cube_lift_m_just_before': float(np.median([x[arm + '_cube_lift_m'] for x in rel])) if rel else 'NA',
                        'median_cubeA_to_cubeB_xy_m_just_before': float(np.median([x[arm + '_cubeA_to_cubeB_xy_m'] for x in rel])) if rel else 'NA'}
        summary[name] = row
    res = {'nature': 'Supplementary reading added after the main Stage 2C tables were seen. Descriptive; no cause is attached.',
           'reference': {'R': 'step at which control returned to the policy (checkpoint + retreat steps used)', 'C0': 'checkpoint'},
           'sets': 'R_vs_C0 = C0 outcome to R outcome, blocks whose cube was grasped at the checkpoint only', 'summary': summary}
    Path(a.out).mkdir(parents=True, exist_ok=False)
    with (Path(a.out) / 'grasp_release_rows.csv').open('x', newline='', encoding='utf-8') as f:
        fields = []
        for x in out:
            for n in x:
                if n not in fields: fields.append(n)
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{n: x.get(n, '') for n in fields} for x in out])
    s2c.write_json(Path(a.out) / 'grasp_release_summary.json', res); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
