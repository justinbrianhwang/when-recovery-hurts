"""Spatial occupancy, clearance, contact and expected-vs-actual audit of stored Vision-CENTER records. Post-hoc only.

Reads scene records, traces and stored frames; runs no simulator, no policy, no controller; writes only into a NEW
output folder (refused if it exists, or if it is inside / contains the records folder).

usage:
  python scripts/36_c1_spatial_audit.py metrics <records_root> <out> [--workers 4] [--voxel 0.003]
        every triggered scene of every stage folder (a folder holding scenes/ and traces/) under <records_root>
  python scripts/36_c1_spatial_audit.py cases <records_root> <out>
        figures (PNG) and the multi-panel GIF for the fixed representative cases, with the numbers they drew
  python scripts/36_c1_spatial_audit.py verify <records_root> <cases_out>
        compares the drawn numbers with the raw trace (own JSON parse) and checks distances by surface sampling
Exit codes: 0 complete, 3 incomplete (an error in any scene, recorded in errors.json; nothing is dropped silently).
"""
import argparse, csv, gzip, hashlib, io, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hci import spatial_audit as SA, spatial_geometry as G, spatial_figures as F, trajectory_audit as T

CODE = ['scripts/36_c1_spatial_audit.py', 'src/hci/spatial_audit.py', 'src/hci/spatial_geometry.py', 'src/hci/spatial_figures.py', 'src/hci/trajectory_audit.py',
        'assets/upstream_geometry/panda_hand_hull.json', 'assets/upstream_geometry/panda_hand_collision.stl']
# Representative cases, named in the request; arms shown per case fixed before any figure was drawn.
CASES = [('50632_rescue', 'shadow', 50632, ('C0', 'VISION_CENTER_V2'), 'v2 rescue: C0 releases off-centre and fails; v2 centres the held cube'),
         ('51004_push_vs_abstain', 'pilot', 51004, ('VISION_CENTER_V1', 'VISION_CENTER_V2'), 'v1 moves and pushes the target; v2 abstains (SIDE_CONTACT_RISK)'),
         ('51009_stale_memory', 'pilot', 51009, ('VISION_CENTER_V2', 'CENTER_ORACLE'), 'stale target memory; v2 abstains (LOW_CARRY); oracle moves'),
         ('50892_side_contact_risk', 'shadow', 50892, ('VISION_CENTER_V1', 'VISION_CENTER_V2'), 'side-contact risk: v1 moves and stops (target_moved); v2 abstains'),
         ('50959_base_camera_error', 'pilot', 50959, ('VISION_CENTER_V2', 'CENTER_ORACLE'), 'v2 acts on a base-camera memory estimate 5.1 mm off; oracle for reference'),
         ('50702_target_moved_memory', 'shadow', 50702, ('C0', 'VISION_CENTER_V2'), 'target moved before the trigger; v2 acts on remembered target')]
ARM_C = {'C0': '#555555', 'VISION_CENTER_V1': '#1f77b4', 'VISION_CENTER_V2': '#d62728', 'CENTER_ORACLE': '#ff7f0e'}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def safe_sha(p):
    """sha256, or MISSING / UNREADABLE (the scene itself is then reported as an error by one_scene)."""
    try: return sha(p)
    except FileNotFoundError: return 'MISSING'
    except OSError as e: return f'UNREADABLE: {e!r}'


def stages(root, only=None):
    root = Path(root); out = sorted(p for p in root.iterdir() if (p / 'scenes').is_dir() and (p / 'traces').is_dir())
    if only:
        missing = set(only) - {p.name for p in out}
        if missing: raise SystemExit(f'stage folders not found: {sorted(missing)}')
        out = [p for p in out if p.name in only]
    return out


def guard(out, root):
    out = Path(out).resolve(); root = Path(root).resolve()
    if out.exists(): raise SystemExit(f'Refused overwrite: {out}')
    if out == root or root in out.parents or out in root.parents: raise SystemExit('Refused: the output folder must be outside the records folder')
    out.mkdir(parents=True)
    return out


def jdump(p, obj): Path(p).write_bytes((json.dumps(obj, indent=1, sort_keys=True, default=_jd) + '\n').encode('utf-8'))
def _jd(v): return v.item() if isinstance(v, np.generic) else str(v)


# ---------------------------------------------------------------- metrics over every intervention
STEP_COLS = ['t', 'phase', 'offset_xy_true', 'd_A_B', 'd_fingerL_A', 'd_fingerR_A', 'd_hand_A', 'd_fingerL_B', 'd_fingerR_B', 'd_hand_B', 'exact_hand_B', 'exact_fingerL_B', 'exact_fingerR_B',
             'd_A_table', 'd_B_table', 'd_gripper_table', 'height_gap_Abottom_Btop', 'fp_overlap_A_B_mm2', 'fp_gap_A_B', 'fp_overlap_gripper_B_mm2', 'fp_gap_gripper_B',
             'B_disp_xy_from_initial', 'B_disp_z_from_initial', 'B_rot_deg_from_initial'] + [f'force_{p}' for p in SA.RECORDED_PAIRS] + \
            [f'subpose_min_d_{p}' for p in SA.SCREEN_PAIRS] + [f'potential_collision_{p}' for p in SA.SCREEN_PAIRS] + [f'proximity_{p}' for p in ('A_B', 'fingerL_B', 'fingerR_B', 'hand_B')]


def one_scene(args):
    stage_dir, seed, voxel = args
    try:
        rec, tr, _ = T.load_scene(stage_dir, seed, frames=False)
        if sha(Path(stage_dir) / rec['trace']) != rec['trace_sha256']: raise ValueError('trace sha256 differs from the record')
        if not rec['triggered']: return {'seed': seed, 'triggered': False}
        out = {'seed': seed, 'triggered': True, 'policy': rec.get('policy_id'), 'arms': {}}
        for arm in rec['arms']:
            M = SA.branch_metrics(rec, tr, arm); e = SA.expected_vs_actual(rec, tr, arm, M); sw = SA.swept_occupancy(rec, tr, arm, voxel)
            contact = {p: [(m[dk], m[f'force_{p}']) for m in M if m.get(f'force_{p}') is not None] for p, dk in SA.CONTACT_PAIRS_GEOM.items()}
            out['arms'][arm] = {'steps': [{k: m.get(k) for k in STEP_COLS} for m in M], 'eva': e, 'swept': sw, 'contact': contact}
        out['requirements'] = SA.requirements_check(rec, tr)['fields_present_in_this_record']
        return out
    except Exception as ex:
        import traceback
        return {'seed': seed, 'error': repr(ex), 'traceback': traceback.format_exc()}


def cmd_metrics(a):
    sds = stages(a.records, a.stages); out = guard(a.out, a.records); t_start = time.time(); errors = []; summary = {'stages': {}}
    from multiprocessing import Pool
    inputs = {}
    all_contact = {p: [] for p in SA.CONTACT_PAIRS_GEOM}; inter_rows = []; eva_all = []; req_fields = {}
    with gzip.open(out / 'per_step_metrics.csv.gz', 'wt', newline='', encoding='utf-8', compresslevel=6) as gz:
        w = csv.writer(gz); w.writerow(['stage', 'scene', 'arm'] + STEP_COLS)
        for sd in sds:
            seeds = sorted(int(p.stem) for p in (sd / 'scenes').glob('*.json') if p.stem.isdigit())
            for s in seeds: inputs[f'{sd.name}/scenes/{s}.json'] = safe_sha(sd / 'scenes' / f'{s}.json'); inputs[f'{sd.name}/traces/trace_{s}.json.gz'] = safe_sha(sd / 'traces' / f'trace_{s}.json.gz')
            jobs = [(str(sd), s, a.voxel) for s in seeds]
            if a.workers > 1:
                with Pool(a.workers) as pool: res = pool.map(one_scene, jobs, chunksize=2)
            else: res = [one_scene(j) for j in jobs]
            st = {'scenes': len(seeds), 'triggered': 0, 'errors': 0, 'interventions': 0, 'cells': 0}
            for r in res:
                if 'error' in r: errors.append({'stage': sd.name, **r}); st['errors'] += 1; continue
                if not r['triggered']: continue
                st['triggered'] += 1
                for k, v in r['requirements'].items(): req_fields.setdefault(k, set()).add(v)
                for arm, d in r['arms'].items():
                    st['cells'] += 1; acted = d['eva']['acted']; st['interventions'] += int(acted)
                    for m in d['steps']: w.writerow([sd.name, r['seed'], arm] + [_fmt(m.get(k)) for k in STEP_COLS])
                    for p, v in d['contact'].items(): all_contact[p] += v
                    eva_all.append({'stage': sd.name, **d['eva'], 'swept': d['swept']}); inter_rows.append(_inter_row(sd.name, r['seed'], arm, d))
            summary['stages'][sd.name] = st
    jdump(out / 'expected_vs_actual.json', eva_all)
    with open(out / 'per_cell_summary.csv', 'w', newline='', encoding='utf-8') as f:
        cols = sorted({k for r in inter_rows for k in r}, key=lambda k: (not k.startswith(('stage', 'scene', 'arm', 'acted')), k)); w = csv.DictWriter(f, cols); w.writeheader(); [w.writerow(r) for r in inter_rows]
    cv = contact_table(all_contact); jdump(out / 'contact_geometry_validation.json', cv)
    tally = {}
    for e in eva_all:
        for k, v in e['expected'].items(): tally.setdefault(e['arm'], {}).setdefault(k, {}).setdefault(v['status'], 0); tally[e['arm']][k][v['status']] += 1
    summary.update(status='COMPLETE' if not errors else 'INCOMPLETE', errors=len(errors), expected_vs_actual_tally=tally, thresholds={'PROXIMITY_M': SA.PROXIMITY_M, 'CONTACT_GEOM_TOL_M': SA.CONTACT_GEOM_TOL_M, 'FAR_M': SA.FAR_M, 'voxel_m': a.voxel, 'label': SA.EXPLORATORY},
                   spec_thresholds={'STEP_CMD_M': T.STEP_CMD_M, 'TARGET_CAP_M': T.TARGET_CAP_M, 'MAX_STEPS': T.MAX_STEPS, 'DZ_TOL_M': T.DZ_TOL_M, 'PUSH_TOL_M': T.PUSH_TOL_M, 'SLIP_TOL_M': T.SLIP_TOL_M, 'REVERT_FRACTION_MAX': T.REVERT_FRACTION_MAX},
                   record_fields_present={k: sorted(v) for k, v in req_fields.items()}, requirements=SA.REQUIREMENTS, wall_seconds=round(time.time() - t_start, 1))
    jdump(out / 'summary.json', {k: v for k, v in summary.items() if k != 'wall_seconds'}); jdump(out / 'timing.json', {'wall_seconds': summary['wall_seconds'], 'workers': a.workers})
    if errors: jdump(out / 'errors.json', errors)
    manifest(out, inputs)
    print(json.dumps({'status': summary['status'], 'stages': summary['stages'], 'errors': len(errors), 'wall_seconds': summary['wall_seconds']}))
    return 0 if not errors else 3


def _fmt(v):
    if isinstance(v, float): return repr(v)
    return '' if v is None else v


def _inter_row(stage, seed, arm, d):
    e = d['eva']; r = {'stage': stage, 'scene': seed, 'arm': arm, 'acted': e['acted'], 'control_steps': e['control_steps']}
    for k, v in e['expected'].items(): r[k] = v['status']
    act = e['actual']
    for k in ('true_offset_trigger_m', 'true_offset_end_of_move_m', 'measured_tcp_xy_displacement_norm_m', 'commanded_step_norm_sum_m', 'target_displacement_trigger_to_end_m', 'first_open_command_step', 'control_end_step', 'final_success_reported_not_judged'):
        r[k] = act.get(k)
    for ph in ('control', 'policy_after'):
        for p, v in (act.get(f'min_clearance_{ph}') or {}).items(): r[f'min_d_{p}_{ph}'] = v
        for p, v in (act.get(f'observed_contacts_{ph}') or {}).items(): r[f'obs_contact_steps_{p}_{ph}'] = v
        for p, v in (act.get(f'potential_collision_steps_{ph}') or {}).items(): r[f'potential_steps_{p}_{ph}'] = v
    sw = d['swept']
    for k in ('gripper_swept_volume_cm3', 'held_cube_swept_volume_cm3', 'union_swept_volume_cm3', 'xy_swept_footprint_mm2', 'xy_swept_overlap_with_target_footprint_mm2'): r[f'swept_{k}'] = sw.get(k)
    return r


def contact_table(all_contact):
    out = {}
    for p, v in all_contact.items():
        if not v: out[p] = {'n_steps': 0}; continue
        d = np.array([x[0] for x in v]); f = np.array([x[1] for x in v]); on = f > 0; q = lambda x, k: float(np.quantile(x, k)) if len(x) else None
        out[p] = {'n_steps': len(d), 'n_contact': int(on.sum()), 'd_when_contact_m': {k: q(d[on], v) for k, v in (('min', 0), ('p01', .01), ('p50', .5), ('p99', .99), ('max', 1))},
                  'd_when_no_contact_m': {k: q(d[~on], v) for k, v in (('min', 0), ('p01', .01), ('p50', .5))},
                  'contact_with_gap_above_tol': int((on & (d > SA.CONTACT_GEOM_TOL_M)).sum()), 'contact_with_gap_above_2mm': int((on & (d > .002)).sum()),
                  'no_contact_with_overlap_beyond_tol': int((~on & (d < -SA.CONTACT_GEOM_TOL_M)).sum()), 'no_contact_within_tol': int((~on & (np.abs(d) <= SA.CONTACT_GEOM_TOL_M)).sum())}
    return out


def manifest(out, inputs):
    code = {p: sha(ROOT / p) for p in CODE}
    outs = {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json',)}
    jdump(out / 'MANIFEST.json', {'what': 'spatial audit outputs; inputs are stored records only (no simulator run)', 'code_sha256': code, 'input_sha256': inputs, 'output_sha256': outs,
                                  'upstream_geometry': json.loads((ROOT / 'assets/upstream_geometry/panda_hand_hull.json').read_text())['upstream_file_sha256']})


# ---------------------------------------------------------------- representative cases: figures and GIF
def case_data(root, stage, seed):
    d = Path(root) / stage; rec, tr, fr = T.load_scene(d, seed)
    for k in ('trace', 'frames'):
        if rec.get(k) and sha(d / rec[k]) != rec[k + '_sha256']: raise SystemExit(f'{k} of scene {seed} does not match its recorded sha256')
    return d, rec, tr, fr


def cmd_cases(a):
    out = guard(a.out, a.records); plt = F._plt(); allp = {}; inputs = {}
    from PIL import Image
    for name, stage, seed, arms, title in CASES:
        d, rec, tr, fr = case_data(a.records, stage, seed); inputs.update({f'{stage}/scenes/{seed}.json': sha(d / 'scenes' / f'{seed}.json'), f'{stage}/{rec["trace"]}': rec['trace_sha256'], f'{stage}/{rec["frames"]}': rec['frames_sha256']})
        M = {arm: SA.branch_metrics(rec, tr, arm) for arm in arms}; lims = F.view_limits(rec, tr, arms); t0 = rec['trigger']['step']; cdir = out / name; cdir.mkdir()
        sw = {arm: F.swept_footprint(rec, tr, arm) for arm in arms}; P = {'case': name, 'scene': seed, 'stage': stage, 'arms': list(arms), 'trigger': t0}
        # 2D occupancy
        fig, axs = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 5.4), dpi=110)
        P['topdown'] = {}
        for ax, arm in zip(np.atleast_1d(axs), arms):
            P['topdown'][arm] = F.draw_topdown(ax, rec, tr, arm, None, lims, sw[arm]); ks = F.key_steps(rec, tr, arm)
            ax.set_title(f"{F.NAME[arm]}: footprints at key steps {', '.join(f'{k} {v}' for k, v in ks.items())}", fontsize=7.5, loc='left')
        F.topdown_legend(np.atleast_1d(axs)[0]); fig.suptitle(f'scene {seed} ({stage}) | {title}\nXY occupancy and clearance, true scale; {F.POSTHOC}', fontsize=8.5)
        fig.tight_layout(); fig.savefig(cdir / 'xy_occupancy.png', metadata={'Software': None}); plt.close(fig)
        # 3D at the end of the move (or trigger)
        fig = plt.figure(figsize=(5.2 * len(arms), 5), dpi=110); P['3d'] = {}
        for i, arm in enumerate(arms):
            ks = F.key_steps(rec, tr, arm); q = ks.get('end_of_move', ks['trigger']); ax = fig.add_subplot(1, len(arms), i + 1, projection='3d')
            p = F.draw_3d(ax, rec, tr, arm, q, lims); P['3d'][arm] = p; ax.set_title(f"{F.NAME[arm]} at step {q}", fontsize=8, loc='left'); ax.text2D(0.02, -0.02, p['text'], transform=ax.transAxes, fontsize=7, family='monospace', va='top')
        fig.suptitle(f'scene {seed}: 3D collision shapes (cubes, finger boxes, hand hull edges faint), paths from trigger - 3\n{F.POSTHOC}', fontsize=8); fig.savefig(cdir / 'clearance_3d.png', metadata={'Software': None}); plt.close(fig)
        # time series
        fig, axs = plt.subplots(4, 1, figsize=(9, 8), dpi=110, sharex=True); P['series'] = {}
        tb = max(M[arm][-1]['t'] for arm in arms); ta = M[arms[0]][0]['t']
        for arm in arms:
            D = F.series_data(rec, tr, arm, M[arm]); P['series'][arm] = D; F.draw_series(axs, D, arm, ARM_C[arm])
        F.series_axes_style(axs, (ta, tb), trig=t0); axs[0].legend(fontsize=6, ncol=2); axs[1].legend(fontsize=5.5, ncol=3, loc='upper right')
        fig.suptitle(f'scene {seed}: behaviour over time (shade = move; black = trigger); {F.POSTHOC}; + = controller online estimate', fontsize=8); fig.tight_layout(); fig.savefig(cdir / 'timeseries.png', metadata={'Software': None}); plt.close(fig)
        # multi-panel GIF
        frames, shown = gif_frames(plt, rec, tr, fr, arms, M, lims, sw, title, stage)
        ims = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
        evs = {v for arm in arms for v in F.key_steps(rec, tr, arm).values()}; dur = [900 if s in evs else 180 for s, _ in shown]; dur[-1] = 2200
        ims[0].save(cdir / 'multipanel.gif', save_all=True, append_images=ims[1:], duration=dur, loop=0, optimize=True)
        P['gif_steps'] = [s for s, _ in shown]; P['gif_frame_provenance'] = [p for _, p in shown]
        P['expected_vs_actual'] = {arm: SA.expected_vs_actual(rec, tr, arm, M[arm]) for arm in arms}; P['swept'] = {arm: SA.swept_occupancy(rec, tr, arm, 0.002) for arm in arms}
        jdump(cdir / 'plotted.json', P); allp[name] = {k: sha(cdir / k) for k in ('xy_occupancy.png', 'clearance_3d.png', 'timeseries.png', 'multipanel.gif', 'plotted.json')}
        print(name, 'gif frames', len(frames), round((cdir / 'multipanel.gif').stat().st_size / 1e6, 2), 'MB', flush=True)
    manifest(out, inputs); return 0


def gif_frames(plt, rec, tr, fr, arms, M, lims, sw, title, stage, W=1600, H=1000, DPI=100):
    t0 = rec['trigger']['step']; ends = [m[-1]['t'] for m in M.values()]
    ev = [v for arm in arms for k, v in F.key_steps(rec, tr, arm).items() if k != 'last']; ta = max(0, t0 - 8); tb = min(max(ends), max(ev) + 12)
    D = {arm: F.series_data(rec, tr, arm, M[arm]) for arm in arms}; frames = []; shown = []
    for t in range(ta, tb + 1):
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI); fig.patch.set_facecolor('white')
        fig.text(.01, .985, f"scene {rec['scene']} ({stage}, used development scene) | {title}", fontsize=9.5, weight='bold', va='top')
        fig.text(.01, .958, f"simulation step t = {t} (trigger {t0}); every panel shows step t.  A: stored RGB only.  B, C, D: {F.POSTHOC}; + = controller online estimate", fontsize=8, va='top', color='#7b1e1e')
        prov = {}
        for i, arm in enumerate(arms):
            y = .54 - i * .40
            img, why = F.frame_at(rec, fr, arm, t); prov[arm] = why; ax = fig.add_axes([.01, y, .29, .33])
            if img is None: ax.imshow(np.full((128, 256, 3), 175, np.uint8)); ax.text(128, 64, 'NO RGB RECORDED\nfor this arm and step', ha='center', va='center', fontsize=9, weight='bold')
            else: ax.imshow(img, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([]); [s.set_color(ARM_C[arm]) or s.set_linewidth(2.5) for s in ax.spines.values()]
            ax.set_title(f"A{i + 1} {F.NAME[arm]}: base | wrist camera", fontsize=8, loc='left', color=ARM_C[arm], weight='bold'); ax.text(0, 140, why, fontsize=6.5, va='top')
            bx = fig.add_axes([.32, y, .20, .33]); F.draw_topdown(bx, rec, tr, arm, t, lims, sw[arm]); bx.set_title(f'B{i + 1} top view (true scale)', fontsize=8, loc='left'); bx.tick_params(labelsize=6)
            cx = fig.add_axes([.535, y + .045, .20, .31], projection='3d'); p = F.draw_3d(cx, rec, tr, arm, t, lims); cx.set_title(f'C{i + 1} 3D collision shapes', fontsize=8, loc='left')
            fig.text(.545, y - .005, p['text'], fontsize=6.3, family='monospace', va='bottom', bbox=dict(fc='white', ec='#ccc', alpha=.9))
        axs = [fig.add_axes([.785, .77 - j * .19, .205, .15]) for j in range(4)]
        for arm in arms: F.draw_series(axs, D[arm], arm, ARM_C[arm], t=t)
        F.series_axes_style(axs, (ta, tb), now=t, trig=t0); [plt.setp(ax.get_xticklabels(), visible=False) for ax in axs[:-1]]
        axs[0].set_title('D behaviour over time (shade = move)', fontsize=8, loc='left'); axs[0].legend(fontsize=5.3, ncol=1, loc='upper right')
        fig.text(.01, .012, 'red = held cube, green = target, slate = finger boxes (dotted: hand hull), orange dashed = risk zone (target + 5 mm, exploratory), grey = swept footprint of fingers + held cube (interpolated),\n'
                 'blue x = commanded target, purple + = controller online estimate.  Arm links not shown: arm joint angles are not logged.  '
                 'D, gap panel: | recorded contact held-target, v recorded contact fingers-target, diamond potential collision (geometry).', fontsize=6.5, va='bottom')
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig); shown.append((t, prov))
    return frames, shown


# ---------------------------------------------------------------- independent check of the drawn numbers
def cmd_verify(a):
    root = Path(a.records); cases = Path(a.out); report = {'cases': {}, 'failures': []}
    for name, stage, seed, arms, _ in CASES:
        P = json.loads((cases / name / 'plotted.json').read_text(encoding='utf-8'))
        with gzip.open(root / stage / 'traces' / f'trace_{seed}.json.gz', 'rt', encoding='utf-8') as f: raw = json.load(f)['traces']
        rec = json.loads((root / stage / 'scenes' / f'{seed}.json').read_text(encoding='utf-8')); t0 = rec['trigger']['step']; r = {'max_abs_diff_m': 0.}
        for arm in arms:
            rows = {x['t']: x for x in raw['trunk'] if x['t'] <= t0}; rows.update({x['t']: x for x in raw[arm]})
            td = P['topdown'][arm]; dx = max(abs(td['tcp_xy'][i][j] - rows[q]['tcp'][j]) for i, q in enumerate(td['steps']) for j in (0, 1))
            dB = max(abs(td['target_xy_true'][j] - rows[td['steps'][-1]]['cubeB'][j]) for j in (0, 1))
            cell = next(c for c in rec['cells'] if c['arm'] == arm); cmdl = ((cell.get('control') or {}).get('steps') or []) if cell.get('acted') else []
            dc = max([0.] + [abs(td['commanded'][j][2][q] - (rows[t0 + j]['tcp'][q] + x['commanded_xy_m'][q])) for j, x in enumerate(cmdl) for q in (0, 1)])
            three = P['3d'][arm]; d3 = max(abs(three['tcp_mm'][-1][j] / 1e3 - rows[three['t']]['tcp'][j]) for j in range(3))
            S = P['series'][arm]; s_off = max(abs(S['offset_true_m'][i] - math.hypot(rows[q]['cubeB'][0] - rows[q]['cubeA'][0], rows[q]['cubeB'][1] - rows[q]['cubeA'][1])) for i, q in enumerate(S['t']))
            raw_ab = lambda q: None if rows[q].get('f_A_B') is None else bool(rows[q]['f_A_B'] > 0)
            raw_fb = lambda q: None if (rows[q].get('f_B_left') is None or rows[q].get('f_B_right') is None) else bool(rows[q]['f_B_left'] > 0 or rows[q]['f_B_right'] > 0)
            s_obs = all(S['obs_A_B'][i] is raw_ab(q) for i, q in enumerate(S['t'])) and all(S['obs_fingers_B'][i] is raw_fb(q) for i, q in enumerate(S['t']))   # three-state: True / False / None
            s_w = max(abs(S['width_m'][i] - sum(rows[q]['finger_qpos'])) for i, q in enumerate(S['t']))
            # distances by surface sampling (independent of the exact algorithm): sampled >= exact - eps, and close to it
            samp = []; bounds = []                                         # values above FAR_M are AABB lower bounds, checked as bounds only
            for i, q in list(enumerate(S['t']))[::max(1, len(S['t']) // 12)]:
                Ex = S['d_A_B'][i]
                if 0 < Ex <= SA.FAR_M: samp.append((Ex, _sampled_cube_gap(rows[q])))
                elif Ex > SA.FAR_M: bounds.append((Ex, _sampled_cube_gap(rows[q], 41)))
            pen = [_penetration_check(rows[q], S['d_A_B'][i]) for i, q in enumerate(S['t']) if S['d_A_B'][i] <= 0]      # every overlapping step
            sd = max([0.] + [abs(e - s) for e, s in samp]); below = any(s < e - 1e-9 for e, s in samp) or any(s < e - 1e-9 for e, s in bounds)
            r.setdefault('lower_bound_checks', 0); r['lower_bound_checks'] += len(bounds)
            pen_bad = [p for p in pen if not p['ok']]
            if pen_bad: report['failures'].append(f'{name} {arm}: {len(pen_bad)} overlapping steps fail the independent penetration check')
            r[arm] = {'penetration_checks': len(pen), 'penetration_checks_failed': len(pen_bad), 'tcp_path_max_diff_m': dx, 'target_centre_diff_m': dB, 'commanded_target_max_diff_m': dc, 'tcp_3d_diff_m': d3, 'offset_series_max_diff_m': s_off, 'contact_flags_match_raw': s_obs,
                      'finger_width_max_diff_m': s_w, 'held_target_gap_sampled_checks': len(samp), 'held_target_gap_sampled_max_diff_m': sd, 'sampled_below_exact': below}
            for k in ('tcp_path_max_diff_m', 'target_centre_diff_m', 'commanded_target_max_diff_m', 'tcp_3d_diff_m', 'offset_series_max_diff_m', 'finger_width_max_diff_m'):
                if r[arm][k] > 1e-9: report['failures'].append(f'{name} {arm} {k} = {r[arm][k]}')
            if not s_obs: report['failures'].append(f'{name} {arm}: contact flags differ from the raw forces')
            if below or sd > 0.0005: report['failures'].append(f'{name} {arm}: sampled held-target gap disagrees with the exact gap (max diff {sd})')
            gp = P['gif_frame_provenance']; rgb_steps = [s for s, p in zip(P['gif_steps'], gp) if p[arm].startswith('recorded')]
            r[arm]['gif_steps_with_recorded_rgb'] = len(rgb_steps); r[arm]['gif_steps_without_rgb'] = len(gp) - len(rgb_steps)
        report['cases'][name] = r
    report['status'] = 'PASS' if not report['failures'] else 'FAIL'
    out = cases / 'verification.json'
    if out.exists(): raise SystemExit(f'Refused overwrite: {out}')
    jdump(out, report); print(json.dumps({'status': report['status'], 'failures': report['failures']}, indent=1)); return 0 if report['status'] == 'PASS' else 3


def _penetration_check(row, d, n_dirs=4000, push=3e-4):
    """Independent two-sided check of a reported overlap d <= 0 between the cubes (depth = -d):
    (a) no sampled direction (Fibonacci sphere) has a smaller interval overlap than the depth: the overlap along any
        direction is exact for that direction, so a smaller one would prove the depth too large;
    (b) moving the target by depth + 0.3 mm along the reported axis separates the cubes (point-to-solid-box sampling),
        so the depth is not too small."""
    A = G.cube(row['cubeA'], row['cubeA_q']); B = G.cube(row['cubeB'], row['cubeB_q']); depth, axis = G.sat_overlap(A, B, return_axis=True)
    k = np.arange(n_dirs) + .5; ph = np.arccos(1 - 2 * k / n_dirs); th = np.pi * (1 + 5 ** .5) * k; U = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
    pa = A.V @ U.T; pb = B.V @ U.T; ov = np.minimum(pa.max(0) - pb.min(0), pb.max(0) - pa.min(0)); smallest = float(ov.min())
    moved = {**row, 'cubeB': (np.asarray(row['cubeB']) + axis * (depth + push)).tolist()}; gap = _sampled_cube_gap(moved, 41)
    ok = bool(abs(-d - depth) < 1e-12 and smallest >= depth - 1e-9 and gap > 0)
    return {'reported_depth_m': -d, 'sat_depth_m': depth, 'smallest_sampled_direction_overlap_m': smallest, 'gap_after_push_m': gap, 'ok': ok}


def _sampled_cube_gap(row, n=81):
    """Independent of the vertex-face / edge-edge algorithm: each cube's surface sampled on an n x n grid per face
    (0.5 mm spacing at n = 81), and every sample's distance to the other SOLID cube in closed form (clamp in the cube's
    frame); the minimum over both directions. Never below the exact gap; above it by at most about half a grid diagonal."""
    u = np.linspace(-G.CUBE_HALF_M, G.CUBE_HALF_M, n); a, b = np.meshgrid(u, u); a = a.ravel(); b = b.ravel(); h = np.full_like(a, G.CUBE_HALF_M)
    L = np.concatenate([np.stack(x, 1) for s in (1, -1) for x in ((s * h, a, b), (a, s * h, b), (a, b, s * h))])
    def to_box(P, c, q):
        Q = (P - np.asarray(c)) @ G.quat_to_R(q); return float(np.linalg.norm(Q - np.clip(Q, -G.CUBE_HALF_M, G.CUBE_HALF_M), axis=1).min())
    A = np.asarray(row['cubeA']) + L @ G.quat_to_R(row['cubeA_q']).T; B = np.asarray(row['cubeB']) + L @ G.quat_to_R(row['cubeB_q']).T
    return min(to_box(A, row['cubeB'], row['cubeB_q']), to_box(B, row['cubeA'], row['cubeA_q']))


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument('cmd', choices=('metrics', 'cases', 'verify')); p.add_argument('records'); p.add_argument('out')
    p.add_argument('--workers', type=int, default=4); p.add_argument('--voxel', type=float, default=0.003)
    p.add_argument('--stages', nargs='*', default=None, help='stage folder names (default: every folder with scenes/ and traces/)'); a = p.parse_args(argv)
    return {'metrics': cmd_metrics, 'cases': cmd_cases, 'verify': cmd_verify}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
