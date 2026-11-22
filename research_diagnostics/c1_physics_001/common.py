"""Shared paths, hashing and read-only loaders for the C1 post-hoc physics diagnostics (POST_HOC_EXPLORATORY).

Nothing here writes into an input folder, runs a simulator or reaches a controller. Every loaded scene record and trace
is checked against the sha256 that the record itself stores (the run's own provenance), so a changed input is an error,
never a silently different result.
"""
from __future__ import annotations

import csv, gzip, hashlib, json, os, sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

FROZEN_PLAN_SHA256 = '6b3f04fea1ea5346d6f0725b6e49ae9ce04c4fb132f37450243a807024e5dbf7'
RUN = REPO / 'runs' / 'vision_center_v2_c1_001_server' / 'extracted' / 'runs' / 'vision_center_v2_c1_001'
ARCHIVE = REPO / 'runs' / 'vision_center_v2_c1_001_server' / 'vision_center_v2_c1_out.tgz'
ARCHIVE_SHA256 = 'c3960c8db0b0dcfa7a9da56cf602056c57c691bcbe06bea22151e359782b60f2'
PLAN = REPO / 'results' / 'vision_center_v2_c1_001' / 'preparation_v3' / 'plan.json'
OUT = REPO / 'results' / 'vision_center_v2_c1_001' / 'posthoc_physics_diagnostics_001'
# The existing analysis outputs (read-only). Durable copy: the synced Drive folder of the C1 development run.
DRIVE_RUN = Path(os.environ.get('C1_DRIVE_RUN', r'G:\내 드라이브\Independent Research\Robotics\Robot Harm\results\vision_center_v2_c1_001\development_run_001'))
MECH = DRIVE_RUN / 'mechanism_c1_001'
TRAJ = DRIVE_RUN / 'trajectory_audit_001'
POLICIES = ('train_seed_1', 'train_seed_2')
SOURCE_KIND = 'C1_ORIGINAL'
LABEL = 'POST_HOC_EXPLORATORY'

# Files that must not change during this work (checked at start and end).
PROTECTED_CODE = ['src/hci/spatial_probe.py', 'src/hci/spatial_links.py', 'src/hci/spatial_geometry.py', 'src/hci/trajectory_audit.py',
                  'src/hci/recovery_v1_oracle.py', 'src/hci/vision_center_v2.py', 'src/hci/vision_center.py', 'src/hci/vision_center_c1.py',
                  'src/hci/simulation.py', 'configs/stackcube.yaml', 'scripts/34_vision_center_c1.py', 'scripts/35_c1_trajectory_audit.py',
                  'scripts/38_c1_spatial_audit_v2.py', 'mechanism_analysis/s6_validity.py', 'mechanism_analysis/run_analysis.py',
                  'results/vision_center_v2_c1_001/preparation_v3/plan.json']


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(chunk): h.update(b)
    return h.hexdigest()


def rel(p):
    """Repo-relative POSIX path, or a path relative to the Drive run folder; never a personal absolute path."""
    p = Path(p)
    for base, tag in ((REPO, ''), (DRIVE_RUN, 'DRIVE:development_run_001/')):
        try: return tag + p.resolve().relative_to(base.resolve()).as_posix()
        except ValueError: pass
    return p.name


def jdump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=_py) + '\n', encoding='utf-8', newline='\n')


def _py(v):
    import numpy as np
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, Path): return rel(v)
    raise TypeError(type(v))


def write_csv(path, rows, fields=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ['empty'])
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='raise'); w.writeheader()
        for r in rows: w.writerow({k: ('NA' if r.get(k) is None else r.get(k)) for k in fields})


def pass_dir(policy): return RUN / f'pass_{policy}'


@lru_cache(maxsize=None)
def record(policy, scene):
    """Scene record with its own sha256 (a dict; do not mutate)."""
    p = pass_dir(policy) / 'scenes' / f'{scene}.json'
    return json.loads(p.read_text(encoding='utf-8')), sha256(p)


def traces(policy, scene):
    """All traces of one scene, after checking the trace file against the sha256 stored in its record."""
    rec, _ = record(policy, scene); p = pass_dir(policy) / rec['trace']
    h = sha256(p)
    if h != rec['trace_sha256']: raise ValueError(f'{policy} {scene}: trace sha256 {h} != recorded {rec["trace_sha256"]}')
    with gzip.open(p, 'rt', encoding='utf-8') as f: return json.load(f)['traces'], h


def cell(rec, arm): return next((c for c in rec['cells'] if c['arm'] == arm), None)


def protected_hashes():
    return {c: sha256(REPO / c) for c in PROTECTED_CODE}


def run_folder_hashes():
    """sha256 of every file of the C1 run folder (records, traces, frames, run-level files)."""
    return {p.relative_to(RUN).as_posix(): sha256(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
