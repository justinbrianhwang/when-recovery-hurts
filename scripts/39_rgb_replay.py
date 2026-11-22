"""RGB replay of the three README GIF scenes (work id rgb_replay_001). Presentation only: no new scene, no new controller,
no criterion, no result of any experiment is changed or added.

Why: the Phase B pass of Vision-CENTER v2 (vision_center_v2_001) stored RGB only up to the trigger and during the
vision-controlled move, so the README GIFs show grey "NO RGB RECORDED" panels for the rest of each episode. Its traces
hold no arm joint positions, so the stored state cannot be re-rendered. This script runs the SAME scene again with the
SAME frozen plan (its code hashes are checked), policy, collector (hci.vision_center_v2.collect_scene, unchanged) and
arms, and additionally keeps both cameras' frames after every simulator step of the trunk and of every arm.

The replayed frames may be shown only if the replay reproduces the stored record: every per-step trace row (TCP, cube
poses, finger opening, executed action, ...) and every cell outcome is compared with the stored files. Any difference
is written to compare.json and the script exits 2; it never edits or replaces the stored records.

usage (server, repository root, venv active):
  python scripts/39_rgb_replay.py --phase-b <folder with plan.json, shadow/, pilot/> --out runs/rgb_replay_001/replay [--cases ...]
"""
from pathlib import Path
import argparse, gzip, importlib.util, json, sys, time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hci.io import read_json, write_json, sha256, runtime_info
from hci import vision_center_v2 as V2
from hci import simulation as SIM
from hci.trees import as_numpy

_spec = importlib.util.spec_from_file_location('vc2_driver', Path(__file__).with_name('33_vision_center_v2.py')); M = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(M)

CASES = {'recovery_success': ('shadow', 50632), 'unsafe_intervention': ('pilot', 51004), 'memory_failure': ('pilot', 51009)}   # as in scripts/make_research_gifs.py
TRACE_KEYS_SKIP = ()            # every key of every stored row is compared


class Recorder:
    """Class-level wrap of Runtime.reset / restore / advance (installed before the collector's own Hooks, which wrap
    type(r).advance and therefore call this). Reads r.raw only; never changes the runtime."""
    def __init__(self):
        self.segments = []; self._orig = {}

    @staticmethod
    def frame(r):
        sd = r.raw['sensor_data']; img = lambda k: np.asarray(as_numpy(sd[k]['rgb'])).reshape(128, 128, 3).astype(np.uint8)
        return np.concatenate([img('base_camera'), img('hand_camera')], 1)

    def __enter__(self):
        R = SIM.Runtime; self._orig = {k: getattr(R, k) for k in ('reset', 'restore', 'advance')}; rec = self; o = self._orig

        def reset(r, episode):
            out = o['reset'](r, episode); rec.segments.append({'kind': 'reset', 't': [r.t], 'frames': [rec.frame(r)]}); return out

        def restore(r, s):
            out = o['restore'](r, s); rec.segments.append({'kind': 'restore', 't': [r.t], 'frames': [rec.frame(r)]}); return out

        def advance(r, action):
            z = o['advance'](r, action); seg = rec.segments[-1]; seg['t'].append(r.t); seg['frames'].append(rec.frame(r)); return z
        R.reset, R.restore, R.advance = reset, restore, advance; return self

    def __exit__(self, *exc):
        for k, v in self._orig.items(): setattr(SIM.Runtime, k, v)


def maxdiff(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): return float('inf')
        return max([maxdiff(a[k], b[k]) for k in a] + [0.])
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) != len(b): return float('inf')
    if a is None or b is None: return 0. if a is b else float('inf')
    if isinstance(a, str) or isinstance(b, str): return 0. if a == b else float('inf')
    try:
        x = np.asarray(a, float); y = np.asarray(b, float)
        if x.shape != y.shape: return float('inf')
        return float(np.max(np.abs(x - y))) if x.size else 0.
    except (TypeError, ValueError):
        return 0. if a == b else float('inf')


def compare(old_rec, old_traces, old_frames, rec, traces, arm_frames):
    cmp = {'arms_equal': old_rec['arms'] == rec['arms'], 'trigger_step': [old_rec['trigger']['step'] if old_rec['triggered'] else None, rec['trigger']['step'] if rec['triggered'] else None], 'cells': {}, 'traces': {}, 'stored_frames': {}}
    for a, b in zip(old_rec['cells'], rec['cells']):
        keys = ('arm', 'y', 'steps', 'reason', 'acted', 'intervention', 'intervention_step', 'control_steps')
        cmp['cells'][a['arm']] = {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)}
        ca, cb = a.get('control'), b.get('control')
        if (ca or cb) and maxdiff(ca, cb) != 0.: cmp['cells'][a['arm']]['control_record_maxdiff'] = maxdiff(ca, cb)
    for k in old_traces:
        A, B = old_traces[k], traces.get(k, [])
        cmp['traces'][k] = {'rows': [len(A), len(B)], 'maxdiff': max([maxdiff(x, y) for x, y in zip(A, B)] + [0.]) if len(A) == len(B) else None}
    t0 = rec['trigger']['step'] if rec['triggered'] else None
    for k, arr in old_frames.items():
        if k == 'trunk': new = arm_frames['trunk'][:len(arr)]
        elif k in arm_frames: new = arm_frames[k][t0:t0 + len(arr)]
        else: cmp['stored_frames'][k] = 'not an arm (no comparison)'; continue
        if arr.shape[2] == 128: new = new[:, :, 128:]                         # v1 stored the hand camera only
        d = np.abs(arr.astype(int) - new.astype(int)) if new.shape == arr.shape else None
        cmp['stored_frames'][k] = {'n': int(len(arr)), 'max_abs_pixel_diff': None if d is None else int(d.max()), 'pixels_differing': None if d is None else int((d > 0).sum())}
    ok = (cmp['arms_equal'] and cmp['trigger_step'][0] == cmp['trigger_step'][1] and all(not v for v in cmp['cells'].values())
          and all(v['rows'][0] == v['rows'][1] and v['maxdiff'] == 0. for v in cmp['traces'].values()))
    cmp['trajectory_reproduced_exactly'] = bool(ok); return cmp


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase-b', required=True); p.add_argument('--out', required=True); p.add_argument('--cases', nargs='*', default=list(CASES))
    p.add_argument('--upstream', default='third_party/ManiSkill'); p.add_argument('--device', default='cuda')
    a = p.parse_args(); root = Path(a.phase_b); out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists: never overwritten (use a new folder)')
    plan_path = root / 'plan.json'; plan, c = M.bound(plan_path)                 # refuses if any code file or the policy differs from the frozen Phase B plan
    out.mkdir(parents=True); summary = {'work': 'rgb_replay_001', 'phase_b_plan_sha256': sha256(plan_path), 'policy': M.POLICY, 'cases': {}, 'runtime': runtime_info()}
    from hci import vision_v1 as V
    r = V.runtime(c, M.POLICY['path'], a.upstream, a.device); all_ok = True
    try:
        for name in a.cases:
            stage, seed = CASES[name]; tick = time.perf_counter()
            old_rec = read_json(root / stage / 'scenes' / f'{seed}.json')
            with gzip.open(root / stage / old_rec['trace'], 'rt', encoding='utf-8') as f: old_traces = json.load(f)['traces']
            with np.load(root / stage / old_rec['frames']) as z: old_frames = {k: z[k] for k in z.files}
            with Recorder() as R:
                rec, traces, _frames = V2.collect_scene(r, seed, c, probe=M.X.gt_probe, arms=old_rec['arms'], role=old_rec['role'])
            segs = R.segments
            if [s['kind'] for s in segs] != ['reset'] + ['restore'] * (len(rec['arms']) if rec['triggered'] else 0): raise RuntimeError(f'{name}: unexpected reset/restore sequence {[s["kind"] for s in segs]}')
            arm_frames = {'trunk': np.stack(segs[0]['frames'])}; steps = {'trunk': segs[0]['t']}
            if segs[0]['t'] != list(range(len(segs[0]['t']))): raise RuntimeError(f'{name}: trunk steps are not 0..T')
            t0 = rec['trigger']['step']
            for arm, s in zip(rec['arms'], segs[1:]):
                if s['t'] != list(range(t0, t0 + len(s['t']))): raise RuntimeError(f'{name} {arm}: steps are not consecutive from the trigger')
                arm_frames[arm] = np.concatenate([arm_frames['trunk'][:t0], np.stack(s['frames'])]); steps[arm] = [0, s['t'][-1]]   # step t at index t
            cmp = compare(old_rec, old_traces, old_frames, rec, traces, arm_frames); all_ok &= cmp['trajectory_reproduced_exactly']
            d = out / name; d.mkdir()
            np.savez_compressed(d / f'frames_{seed}.npz', **arm_frames)
            with gzip.open(d / f'trace_{seed}.json.gz', 'wt', encoding='utf-8') as f: json.dump({'seed': seed, 'traces': traces}, f)
            write_json(d / f'{seed}.json', rec); write_json(d / 'compare.json', cmp)
            summary['cases'][name] = {'stage': stage, 'scene': seed, 'arms': rec['arms'], 'frames_file': f'{name}/frames_{seed}.npz', 'frames_sha256': sha256(d / f'frames_{seed}.npz'),
                                      'frames_layout': 'per arm: index t = observation after step t (index 0 = after reset); arms share the trunk frames up to the trigger; 128 x 256, base_camera left, hand_camera right',
                                      'steps_per_arm': {k: int(len(v)) - 1 for k, v in arm_frames.items()}, 'stored_record_sha256': sha256(root / stage / 'scenes' / f'{seed}.json'),
                                      'trajectory_reproduced_exactly': cmp['trajectory_reproduced_exactly'], 'wall_seconds': time.perf_counter() - tick}
            print(json.dumps({name: {k: summary['cases'][name][k] for k in ('trajectory_reproduced_exactly', 'steps_per_arm', 'wall_seconds')}}), flush=True)
    finally:
        r.env.close()
    summary['all_reproduced_exactly'] = bool(all_ok); write_json(out / 'replay_summary.json', summary)
    if not all_ok: print('MISMATCH: see compare.json; the replayed frames must not be shown as the stored episodes'); raise SystemExit(2)


if __name__ == '__main__':
    main()
