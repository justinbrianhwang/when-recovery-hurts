"""Per-job cost of the shared instance (B1 calibration, B3 generalization, B4 S6 sub-step), from the job/step timestamps in
logs/ras_exec.status and the process wall times. Method (declared before the end of the run): the instance cost is the
Vast credit difference; setup time (install, bootstrap) is split evenly over the three jobs; every later interval is split
over the jobs in proportion to their running process count in that interval.

usage: python -P research_diagnostics/ras_exec_001/cost_split.py --credit-before X --credit-after Y --intervals <json> --out <json>
  intervals json: {"setup": [start, end], "jobs": {"B1": [[start, end], ...], "B3": [...], "B4": [...]}, "instance": [created, destroyed]}  (ISO UTC)
"""
import argparse, json
from datetime import datetime
from pathlib import Path


def t(s): return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def main():
    a = argparse.ArgumentParser(); a.add_argument('--credit-before', type=float, required=True); a.add_argument('--credit-after', type=float, required=True)
    a.add_argument('--intervals', required=True); a.add_argument('--out', required=True); a = a.parse_args()
    iv = json.loads(Path(a.intervals).read_text(encoding='utf-8')); total = a.credit_before - a.credit_after
    t0, t1 = map(t, iv['instance']); span = t1 - t0; rate = total / span
    jobs = {k: [(t(s), t(e)) for s, e in v] for k, v in iv['jobs'].items()}; cost = {k: 0.0 for k in jobs}
    marks = sorted({t0, t1, *[x for v in jobs.values() for se in v for x in se]})
    for lo, hi in zip(marks, marks[1:]):
        mid = (lo + hi) / 2; act = {k: sum(1 for s, e in v if s <= mid < e) for k, v in jobs.items()}; n = sum(act.values())
        share = {k: (act[k] / n if n else 1 / len(jobs)) for k in jobs}
        for k in jobs: cost[k] += rate * (hi - lo) * share[k]
    res = {'credit_before': a.credit_before, 'credit_after': a.credit_after, 'instance_total_usd': total, 'instance_hours': span / 3600, 'usd_per_hour_effective': rate * 3600,
           'per_job_usd': cost, 'caps_usd': {'B1': 2.5, 'B3': 1.5, 'B4': 1.0}, 'within_caps': {k: cost[k] <= c for k, c in {'B1': 2.5, 'B3': 1.5, 'B4': 1.0}.items()}, 'cap_note': 'B1 cap raised from 2.00 to 2.50 by the user during the run (approvals/cap_change_B1_2026-11-22.json)',
           'method': __doc__.split('Method (declared before the end of the run): ')[1].split('\n\nusage')[0]}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
