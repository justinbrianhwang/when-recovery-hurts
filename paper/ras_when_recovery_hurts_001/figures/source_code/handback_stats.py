"""Descriptive hand-back statistics from the raw scene records (read-only; POST HOC, descriptive).

For every V2 branch that moved: steps from the end of the move to the first gripper-open command of the re-planned
policy (`first_open_command_after_move`), and whether the policy re-inferred at the first step after the move
(`first_inference_step`). Also the Check 4 status of the development harm case 51157 (seed 1) from CHECK4_ALL_MOVES.csv.
usage: python -P paper/ras_when_recovery_hurts_001/figures/source_code/handback_stats.py
"""
from __future__ import annotations

import csv, json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]; OUT = REPO / 'paper/ras_when_recovery_hurts_001/data/handback_stats.json'
SRC = {'C1': (REPO / 'runs/vision_center_v2_c1_001_server/extracted/runs/vision_center_v2_c1_001', range(51100, 51300)),
       'calibration': (REPO / 'runs/ras_exec_001/download_final/runs/ras_v2_confirm_001/calibration', range(30000, 30400)),
       'test': (REPO / 'runs/ras_exec_001/download_test/runs/ras_v2_confirm_001/test', range(90000, 90400))}


def main():
    res = {}
    for st, (root, seeds) in SRC.items():
        for p in ('train_seed_1', 'train_seed_2'):
            gaps = []; reinfer = Counter(); ends = Counter(); paths = []; disp = []; missing = 0
            for s in seeds:
                r = json.loads((root / f'pass_{p}/scenes/{s}.json').read_text(encoding='utf-8'))
                c = next(x for x in r['cells'] if x['arm'] == 'VISION_CENTER_V2')
                if not c.get('acted'): continue
                t_end = r['trigger']['step'] + c['control_steps']; ctl = c['control'] or {}
                ends[ctl.get('end_reason')] += 1
                if ctl.get('commanded_path_m') is not None: paths.append(ctl['commanded_path_m']); disp.append(ctl['tcp_xy_displacement_m'])
                fo = c.get('first_open_command_after_move'); fi = c.get('first_inference_step')
                if fo is None: missing += 1
                else: gaps.append(fo - t_end)
                reinfer['first_inference_at_end_of_move' if fi == t_end else f'other({None if fi is None else fi - t_end})'] += 1
            g = np.array(gaps, float); P = np.array(paths); D = np.array(disp)
            res[f'{st}|{p}'] = {'acted': len(gaps) + missing, 'no_open_command_after_move': missing,
                                'steps_to_first_open_after_move': {'min': float(g.min()), 'median': float(np.median(g)), 'max': float(g.max())} if len(g) else None,
                                'reinference': dict(reinfer), 'end_reason': dict(ends),
                                'net_xy_over_commanded_path': {'median': float(np.median(D / P)), 'min': float((D / P).min()), 'max': float((D / P).max())} if len(P) else None}
    with open(REPO / 'results/vision_center_v2_c1_001/posthoc_physics_diagnostics_001/CHECK4_ALL_MOVES.csv', newline='', encoding='utf-8') as f:
        row = next(x for x in csv.DictReader(f) if x['policy'] == 'train_seed_1' and x['scene'] == '51157' and x['arm'] == 'VISION_CENTER_V2')
    res['check4_51157_seed1_V2'] = {k: row[k] for k in ('recomputed_status', 'first_step_cos', 'first_step_act_mm', 'first_step_cmd_mm', 'step2plus_min_cos')}
    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
