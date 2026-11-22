"""Run hull_s45.scene_task for every C1 scene record of both policies (400 scene-runs). One gzipped JSON per scene in a
new work folder (git-ignored, never overwritten). A scene that raises is written as <policy>_<scene>.error.json and the
run stays INCOMPLETE; nothing is dropped.

usage: python -P research_diagnostics/c1_hull_sensitivity_001/run_all.py --work runs/c1_hull_sensitivity_001/work [--workers 4] [--scenes 51157 ...]
"""
from __future__ import annotations

import argparse, gzip, json, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hull_s45 as H  # noqa: E402
C = H.C


def one(task):
    policy, scene, work = task
    try:
        r = H.scene_task((policy, scene))
        with gzip.open(Path(work) / f'{policy}_{scene}.json.gz', 'xt', encoding='utf-8') as f: json.dump(r, f, default=float)
        return policy, scene, None
    except Exception as e:
        (Path(work) / f'{policy}_{scene}.error.json').write_text(json.dumps({'policy': policy, 'scene': scene, 'error': repr(e), 'traceback': traceback.format_exc()}, indent=1), encoding='utf-8')
        return policy, scene, repr(e)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--work', required=True); p.add_argument('--workers', type=int, default=4); p.add_argument('--scenes', nargs='*', type=int)
    a = p.parse_args(); work = Path(a.work)
    if work.exists(): raise SystemExit(f'{work} exists: never overwritten (use a new folder)')
    work.mkdir(parents=True)
    tasks = [(pol, int(f.stem), str(work)) for pol in C.POLICIES for f in sorted((C.pass_dir(pol) / 'scenes').glob('*.json')) if f.stem.isdigit() and (not a.scenes or int(f.stem) in a.scenes)]
    errs = sorted(f.name for pol in C.POLICIES for f in (C.pass_dir(pol) / 'scenes').glob('*.error.json'))
    print(json.dumps({'tasks': len(tasks), 'run_error_records': errs, 'workers': a.workers}), flush=True)
    t0 = time.time(); done = 0; failed = []
    with ProcessPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(one, t) for t in tasks]):
            pol, sc, err = fut.result(); done += 1
            if err: failed.append((pol, sc, err))
            if done % 20 == 0 or err: print(f'{done}/{len(tasks)} {time.time() - t0:.0f}s failed={len(failed)}' + (f' {pol} {sc} {err}' if err else ''), flush=True)
    (work / 'RUN_DONE.json').write_text(json.dumps({'tasks': len(tasks), 'failed': failed, 'seconds': time.time() - t0, 'complete': not failed}, indent=1), encoding='utf-8')
    print('DONE', len(tasks), 'failed', len(failed), f'{time.time() - t0:.0f}s', flush=True)
    if failed: raise SystemExit(3)


if __name__ == '__main__':
    main()
