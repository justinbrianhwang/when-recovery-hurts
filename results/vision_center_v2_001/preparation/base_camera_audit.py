"""Vision-CENTER v2, Phase A: error of the base-camera target estimate on every frame of the 500 training demonstrations
(development data; the target does not move in them). v1's full-view centroid against v2's largest-connected-region
estimate. Ground truth (the stored cube state) is the label. No simulator, no policy.
Usage (repository root): python results/vision_center_v2_001/preparation/base_camera_audit.py <demo .h5> <out.json>"""
import json, sys
import numpy as np
sys.path.insert(0, 'src')
from hci import vision_center as V1
from hci import vision_center_v2 as V2
import h5py

DEMOS, OUT = sys.argv[1], sys.argv[2]; cb = V1.cam_base(); rows = {'v1': [], 'v2': [], 'v2_any_ratio': []}; frames = 0; first = {'v1': [], 'v2': []}
with h5py.File(DEMOS, 'r') as f:
    ks = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
    for k in ks:
        g = f[k]; bi = g['obs/sensor_data/base_camera/rgb'][:]; B = g['env_states/actors/cubeB'][:, :2].astype(float); got = {'v1': None, 'v2': None}
        for i in range(len(bi)):
            frames += 1; a = V1.target_full_view(bi[i], cb, 'base_camera'); b = V2.target_base(bi[i])
            if a is not None and a['valid']:
                e = float(np.linalg.norm(a['xy'] - B[i])); rows['v1'].append(e)
                if got['v1'] is None: got['v1'] = e
            if b is not None:
                e = float(np.linalg.norm(b['xy'] - B[i])); rows['v2_any_ratio'].append([e, b['area_ratio']])
                if b['valid']:
                    rows['v2'].append(e)
                    if got['v2'] is None: got['v2'] = e
        for n in got: first[n].append(got[n])
st = lambda a: {'n': len(a), 'median_mm': round(float(np.median(a)) * 1000, 2), 'p90_mm': round(float(np.percentile(a, 90)) * 1000, 2), 'p99_mm': round(float(np.percentile(a, 99)) * 1000, 2), 'max_mm': round(float(np.max(a)) * 1000, 2),
                'over_5mm': int(sum(x > 0.005 for x in a)), 'over_10mm': int(sum(x > 0.010 for x in a))}
r = np.array(rows['v2_any_ratio']); bands = [{'ratio_from': lo, 'ratio_to': hi, **st(r[(r[:, 1] >= lo) & (r[:, 1] < hi), 0].tolist())} for lo, hi in ((0.0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.25)) if ((r[:, 1] >= lo) & (r[:, 1] < hi)).any()]
summary = {'demonstrations': len(ks), 'frames': frames, 'v1_full_view_centroid_accepted': st(rows['v1']), 'v2_largest_region_accepted': st(rows['v2']), 'v2_by_area_ratio_before_the_acceptance_test': bands,
           'demonstrations_with_an_accepted_v2_estimate': sum(1 for x in first['v2'] if x is not None), 'base_view': V2.BASE_VIEW, 'u_base_m': V2.U_BASE_M,
           'note': 'the arm, the gripper and the carried cube occlude the target in part of the frames; an occluded view fails the area test. Motion-planner demonstrations, not policy rollouts'}
with open(OUT, 'x', encoding='utf-8') as fo: json.dump({'status': 'OFFLINE, development data; ground truth is a label', 'summary': summary, 'errors_m': {'v1': rows['v1'], 'v2': rows['v2']}}, fo)
print(json.dumps(summary, indent=1))
