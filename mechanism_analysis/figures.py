"""Case figures from stored Phase B original records only: the stored RGB frames, the post-hoc ground-truth paths of the
same run's trace, the online estimates the arm recorded, and a decision / control / resume / opening timeline.
No frame, contact or joint value is synthesised; the local CPU replay (another host) is never drawn here."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mechanism as M

ARM_STYLE = {'C0': ('#6b6b6b', 'C0 (reference)'), 'VISION_CENTER_V1': ('#2f6db3', 'v1'), 'VISION_CENTER_V2': ('#d1701c', 'v2'), 'CENTER_ORACLE': ('#2e8b57', 'oracle')}


def case_figures(run, res, sel, out):
    by = {}
    for r in res['rows']: by.setdefault(r['scene'], {})[r['arm']] = r
    stage_of = {r['scene']: r['stage'] for r in res['rows']}; made = {}
    for scene, why in sorted(sel.items()):
        if scene not in by: made[str(scene)] = {'status': 'NOT_FOUND', 'why': why}; continue
        stage = stage_of[scene]; d = run.root / run.stages[stage]; rec = M.read_json(d / 'scenes' / f'{scene}.json')
        tr, _ = run.trace(stage, rec); fr = np.load(d / rec['frames']); f = out / f'case_{scene}.png'
        draw(rec, tr, fr, by[scene], why, f); made[str(scene)] = {'status': 'MADE', 'file': f'figures/{f.name}', 'why': why, 'stage': stage, 'source_kind': run.kind,
                                                                  'record_sha256': M.sha256(d / 'scenes' / f'{scene}.json'), 'frames_sha256': rec['frames_sha256'], 'trace_sha256': rec['trace_sha256']}
    return made


def draw(rec, tr, fr, rows, why, path):
    T = rec['trigger']['step'] if rec.get('triggered') else None; trunk = M.rows_by_t(tr['trunk'])
    fig = plt.figure(figsize=(15, 10.5)); gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.25, 0.9], hspace=0.45)
    fig.suptitle(f'Scene {rec["scene"]} ({rec["role"]}) - PHASE_B_ORIGINAL, policy train_seed_1 - selected: {why}', fontsize=11)
    # A: stored RGB frames (the only images used)
    keys = list(fr.keys()); ax_n = 0
    for k, idx, title in (('trunk', T, f'trunk frame at the trigger (t={T})'), ('VISION_CENTER_V2', 0, 'v2 frame at the trigger (stored)'), ('trunk', max(0, (T or 0) - 10), f'trunk frame t={max(0, (T or 0) - 10)}')):
        ax = fig.add_subplot(gs[0, ax_n]); ax_n += 1
        if k in keys and idx is not None and idx < fr[k].shape[0]: ax.imshow(fr[k][idx]); ax.set_title(f'STORED RGB: {title}\n(left base camera, right hand camera)', fontsize=8)
        else: ax.text(0.5, 0.5, f'no stored frame\n({k}, index {idx})', ha='center', va='center', fontsize=9); ax.set_title('STORED RGB: not available', fontsize=8)
        ax.axis('off')
    ax = fig.add_subplot(gs[0, 3]); ax.axis('off')
    lines = ['Recorded decisions (online):']
    for arm, r in rows.items():
        lines.append(f"{ARM_STYLE.get(arm, ('k', arm))[1].split(' ')[0]}: {r['decision'] or '-'}")
        lines.append(f"   steps {r['control_steps']}, y={r['y_arm']}, vs C0 {r.get('outcome_vs_c0') or '-'}")
        if r.get('target_src'): lines.append(f"   target {r['target_src']}, held {r['hold_src']}"); lines.append(f"   offset est {fmt(r['est_offset_m'])} / true {fmt(r['true_offset_m'])} mm")
    lines.append(''); lines.append('offsets in XY; "true" = POST-HOC GT')
    ax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=6.5)
    # B: XY paths, ground truth of this run's trace, with online estimates
    ax = fig.add_subplot(gs[1, 0:2]); ax.set_title('XY paths around the trigger - POST-HOC GROUND TRUTH (this run\'s trace); markers: ONLINE ESTIMATES', fontsize=9)
    lo = (T or 0) - 10
    for arm, r in rows.items():
        c, lab = ARM_STYLE.get(arm, ('k', arm)); br = M.rows_by_t(tr.get(arm) or [])
        ts = sorted(t for t in br if T is None or t <= T + max(r['control_steps'], 1) + 25)
        if not ts: continue
        P = np.array([br[t]['tcp'][:2] for t in ts]); A = np.array([br[t]['cubeA'][:2] for t in ts]); B = np.array([br[t]['cubeB'][:2] for t in ts])
        ax.plot(A[:, 0] * 1e3, A[:, 1] * 1e3, '-', color=c, lw=1.6, label=f'{lab}: held cube (GT)'); ax.plot(B[:, 0] * 1e3, B[:, 1] * 1e3, ':', color=c, lw=1.2, label=f'{lab}: target cube (GT)')
        if r['control_steps'] and (T + r['control_steps']) in br:
            e = br[T + r['control_steps']]; ax.plot(e['cubeA'][0] * 1e3, e['cubeA'][1] * 1e3, 's', color=c, ms=6, mfc='none')
    pre = [trunk[t] for t in sorted(trunk) if lo <= t <= (T or 0)]
    if pre:
        A = np.array([x['cubeA'][:2] for x in pre]); ax.plot(A[:, 0] * 1e3, A[:, 1] * 1e3, '-', color='k', lw=0.8, alpha=0.5, label='trunk before trigger: held cube (GT)')
    if T in trunk:
        ax.plot(trunk[T]['cubeB'][0] * 1e3, trunk[T]['cubeB'][1] * 1e3, 'k*', ms=12, label='target at trigger (GT)')
    for arm in ('VISION_CENTER_V1', 'VISION_CENTER_V2'):
        cell = next((x for x in rec['cells'] if x['arm'] == arm), None); rd0 = (((cell or {}).get('control') or {}).get('readings') or [None])[0]
        if not rd0: continue
        c = ARM_STYLE[arm][0]
        if rd0.get('target_xy'): ax.plot(rd0['target_xy'][0] * 1e3, rd0['target_xy'][1] * 1e3, 'x', color=c, ms=10, mew=2, label=f'{ARM_STYLE[arm][1]}: target estimate ({rd0.get("target_src") or "frame"})')
        if rd0.get('held_xyz'): ax.plot(rd0['held_xyz'][0] * 1e3, rd0['held_xyz'][1] * 1e3, '+', color=c, ms=10, mew=2, label=f'{ARM_STYLE[arm][1]}: held estimate')
        mt = rd0.get('memory_target')
        if mt and mt.get('t_last') in trunk:
            g = trunk[mt['t_last']]['cubeB']; ax.plot(g[0] * 1e3, g[1] * 1e3, 'o', color=c, mfc='none', ms=9, label=f'{ARM_STYLE[arm][1]}: GT target at memory t_last={mt["t_last"]}')
            circ = plt.Circle((mt['xy'][0] * 1e3, mt['xy'][1] * 1e3), mt['u'] * 1e3, color=c, fill=False, ls='--', lw=0.8); ax.add_patch(circ)
    ax.set_xlabel('x (mm, world)'); ax.set_ylabel('y (mm, world)'); ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=0.25); ax.legend(fontsize=6, loc='best', ncol=2)
    ax.text(0.01, 0.01, 'dashed circle: the memory\'s own bound u (an uncertainty radius, NOT a verified safe region)', transform=ax.transAxes, fontsize=6)
    # B2: offset over time per arm (GT)
    ax = fig.add_subplot(gs[1, 2:4]); ax.set_title('XY offset target - held cube over time (POST-HOC GROUND TRUTH); vertical: events', fontsize=9)
    for arm, r in rows.items():
        c, lab = ARM_STYLE.get(arm, ('k', arm)); br = M.rows_by_t(tr.get(arm) or [])
        ts = sorted(br); ax.plot(ts, [M.norm(M.gt_offset(br[t])) * 1e3 for t in ts], color=c, lw=1.3, label=lab)
    ts = sorted(t for t in trunk if lo <= t <= (T or 0)); ax.plot(ts, [M.norm(M.gt_offset(trunk[t])) * 1e3 for t in ts], color='k', lw=0.8, alpha=0.5, label='trunk')
    ax.axhline(M.RESIDUAL_TOL_M * 1e3, color='k', ls=':', lw=0.8); ax.text(ax.get_xlim()[0], M.RESIDUAL_TOL_M * 1e3, ' residual tolerance 10 mm', fontsize=6, va='bottom')
    if T: ax.axvline(T, color='r', lw=0.8)
    ax.set_xlabel('trace row t (state after t steps)'); ax.set_ylabel('mm'); ax.set_ylim(0, None); ax.grid(alpha=0.25); ax.legend(fontsize=7)
    # C: timeline
    ax = fig.add_subplot(gs[2, :]); ax.set_title('Timeline per arm (trace rows): control interval, policy resume, first open command, GT release, success', fontsize=9)
    evs = [T or 0]; labels = []
    for i, (arm, r) in enumerate(rows.items()):
        c, lab = ARM_STYLE.get(arm, ('k', arm)); y = len(rows) - i
        if r['t_control_start'] is not None: ax.plot([r['t_control_start'], r['t_control_end']], [y, y], color=c, lw=6, solid_capstyle='butt')
        for k, mk, ms, dy in (('t_policy_resume', 'v', 8, 0.18), ('t_first_open', 'o', 8, 0.0), ('t_release_gt', 'x', 9, -0.18), ('t_success', '*', 11, 0.0)):
            if r.get(k) is not None: ax.plot(r[k], y + dy, mk, color=c, ms=ms, mew=2)
        evs += [r.get(k) for k in ('t_control_end', 't_policy_resume', 't_first_open', 't_release_gt', 't_success') if r.get(k) is not None]
        labels.append((y, f"{lab}: {r['decision'] or '-'} -> y={r['y_arm']}"))
    lo_x = (T or 0) - 12; hi_x = max(evs) + 6; ax.set_xlim(lo_x, hi_x)
    for y, s_ in labels: ax.text(lo_x + 0.3, y, s_, fontsize=7, va='center')
    if T: ax.axvline(T, color='r', lw=0.8); ax.text(T, len(rows) + 0.6, ' trigger', color='r', fontsize=7)
    ax.set_yticks([]); ax.set_xlabel('trace row t'); ax.set_ylim(0.3, len(rows) + 1)
    ax.plot([], [], 'kv', label='policy resume'); ax.plot([], [], 'ko', label='first open command'); ax.plot([], [], 'kx', label='GT release'); ax.plot([], [], 'k*', label='success'); ax.legend(fontsize=7, loc='lower left', ncol=4)
    fig.text(0.01, 0.005, 'Sources: STORED RGB and trace of the same Phase B original run (record sha in results.json). The local CPU replay (another host) is not drawn here; '
             'its 3D occupancy figures are in logging_instrumentation_option_a/spatial_audit/cases_final and differ from this run by up to 5.6-11 mm on full trunks.', fontsize=6.5)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.06, wspace=0.25); fig.savefig(path, dpi=110); plt.close(fig)


def fmt(x): return '-' if x is None else f'{x * 1e3:.1f}'
