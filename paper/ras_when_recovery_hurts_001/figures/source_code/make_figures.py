"""Vector figures F1-F9 of the manuscript, drawn from the original (read-only) records and result tables.

Every input is listed in INPUTS and its SHA256 is written to figures/FIGURE_PROVENANCE.json together with the
SHA256 of every output PDF. Camera images are the stored 128x256 RGB frames, shown unmodified (nearest-neighbour).
No value is typed by hand: counts come from data/recount.json (recount_all.py) or from the raw records.

usage: python -P paper/ras_when_recovery_hurts_001/figures/source_code/make_figures.py
"""
from __future__ import annotations

import csv, gzip, hashlib, json, sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = Path(__file__).resolve().parents[4]
PAPER = REPO / 'paper/ras_when_recovery_hurts_001'
OUT = PAPER / 'figures'
C1 = REPO / 'runs/vision_center_v2_c1_001_server/extracted/runs/vision_center_v2_c1_001'
DIAG = REPO / 'results/vision_center_v2_c1_001/posthoc_physics_diagnostics_001'
HULL = REPO / 'results/vision_center_v2_c1_001/posthoc_hull_sensitivity_001'
B4 = REPO / 'results/ras_exec_001/b4_s6_substep/S6_SUBSTEP_SUMMARY.json'
STATIC = REPO / 'results/ras_submission_readiness_001/s6_static_validation/cases.csv'
RECOUNT = PAPER / 'data/recount.json'
USED: dict[str, str] = {}

# Okabe-Ito (CVD-safe) in a fixed order; text stays black/grey.
COL = {'c0': '#7F7F7F', 'v2': '#0072B2', 'single': '#56B4E9', 'oracle': '#E69F00', 'rescue': '#009E73', 'harm': '#D55E00',
       'v1': '#CC79A7', 'abl': '#999999', 'ink': '#222222', 'muted': '#666666', 'grid': '#DDDDDD'}
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 6.8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#444444', 'axes.linewidth': 0.6, 'lines.linewidth': 1.4})
W1, W2 = 3.5, 7.2  # single / double column width (inches)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def use(p: Path) -> Path:
    USED[str(Path(p).relative_to(REPO)).replace('\\', '/')] = sha(p); return Path(p)


def rcsv(p):
    with open(use(p), newline='', encoding='utf-8') as f: return list(csv.DictReader(f))


def scene(pid, s):
    d = C1 / f'pass_{pid}'; r = json.loads(use(d / 'scenes' / f'{s}.json').read_text(encoding='utf-8'))
    t = json.loads(gzip.open(use(d / r['trace'])).read()); fr = np.load(use(d / r['frames'])); return r, t['traces'], fr


def cell(r, arm): return next(c for c in r['cells'] if c['arm'] == arm)


def panel(ax, s): ax.text(-0.14, 1.07, s, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom', ha='right')


def save(fig, name):
    p = OUT / f'{name}.pdf'
    if p.exists(): p.unlink()
    fig.savefig(p, bbox_inches='tight', pad_inches=0.02); plt.close(fig); return p


R = json.loads(use(RECOUNT).read_text(encoding='utf-8'))
POL = ('train_seed_1', 'train_seed_2'); PLAB = {'train_seed_1': 'Policy seed 1', 'train_seed_2': 'Policy seed 2'}


def pick_rescue():  # first C1 seed-1 scene (ascending) in which C0 failed and V2 succeeded
    for s in range(51100, 51300):
        r = json.loads((C1 / 'pass_train_seed_1/scenes' / f'{s}.json').read_text(encoding='utf-8'))
        if cell(r, 'C0')['y'] == 0 and cell(r, 'VISION_CENTER_V2')['y'] == 1: return s
    raise RuntimeError('no rescue scene')


# ------------------------------------------------------------------ F1 problem: rescue vs harm, real frames
def f1():
    fig = plt.figure(figsize=(W2, 2.3)); gs = fig.add_gridspec(2, 6, width_ratios=[1.15, 0.13, 1, 1, 1, 1], hspace=0.3, wspace=0.06)
    ax = fig.add_subplot(gs[:, 0]); panel(ax, 'a')
    t1, t2 = R['test']['train_seed_1'], R['test']['train_seed_2']
    m = [[t1['success_ref'] - t1['harm'] + t2['success_ref'] - t2['harm'], t1['harm'] + t2['harm']],
         [t1['rescue'] + t2['rescue'], t1['ref_failures'] - t1['rescue'] + t2['ref_failures'] - t2['rescue']]]
    lab = [['both succeed', 'HARM'], ['RESCUE', 'both fail']]; fc = [['#F2F2F2', '#FBE3D6'], ['#D5EFE6', '#F2F2F2']]
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j, 1 - i), 1, 1, fc=fc[i][j], ec='white', lw=2))
            ax.text(j + .5, 1.58 - i, f'{m[i][j]}', ha='center', va='center', fontsize=10.5, fontweight='bold', color=COL['ink'])
            cc = COL['harm'] if lab[i][j] == 'HARM' else COL['rescue'] if lab[i][j] == 'RESCUE' else COL['muted']
            ax.text(j + .5, 1.27 - i, lab[i][j], ha='center', va='center', fontsize=6.2, color=cc)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect('equal')
    ax.xaxis.tick_top(); ax.xaxis.set_label_position('top'); ax.set_xticks([.5, 1.5], ['success', 'failure']); ax.set_yticks([1.5, .5], ['success', 'failure'], rotation=90, va='center')
    ax.set_xlabel('recovery arm V2', fontsize=7); ax.set_ylabel('no-recovery baseline C0', fontsize=7); ax.tick_params(length=0, labelsize=6.3)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(1, -0.12, 'Final Test: 2 policies x\n400 paired scenes', ha='center', va='top', fontsize=6, color=COL['muted'])
    rs = pick_rescue()
    rows = [('b', rs, 'rescue'), ('c', 51157, 'harm')]
    for k, (pl, s, kind) in enumerate(rows):
        r, tr, fr = scene('train_seed_1', s); c0, v2 = cell(r, 'C0'), cell(r, 'VISION_CENTER_V2')
        shots = [(fr['trunk'][-1], f'trigger, t={r["trigger"]["step"]}'), (fr['VISION_CENTER_V2'][-1], f'V2 move ends, t={r["trigger"]["step"] + v2["control_steps"]}'),
                 (fr['after_C0'][-1], f'C0 final: {"success" if c0["y"] else "failure"}'), (fr['after_VISION_CENTER_V2'][-1], f'V2 final: {"success" if v2["y"] else "failure"}')]
        for j, (img, cap) in enumerate(shots):
            a_ = fig.add_subplot(gs[k, j + 2]); a_.imshow(img, interpolation='nearest'); a_.set_xticks([]); a_.set_yticks([])
            for sp in a_.spines.values(): sp.set_visible(True); sp.set_color('#BBBBBB'); sp.set_linewidth(.5)
            col = COL['ink']
            if j == 3: col = COL['rescue'] if kind == 'rescue' else COL['harm']
            a_.set_title(cap, fontsize=6.0, pad=2, color=col)
            if j == 0:
                a_.text(-0.05, 1.12, pl, transform=a_.transAxes, fontweight='bold', fontsize=9, ha='right', va='bottom')
                a_.set_ylabel(f'{s}: {kind}', fontsize=6.3, color=COL['rescue'] if kind == 'rescue' else COL['harm'])
    fig.text(0.6, 0.0, 'Frames: base camera (left half) | wrist camera (right half), stored 128x256 RGB; C1 development scenes, policy seed 1.', ha='center', fontsize=5.8, color=COL['muted'])
    return save(fig, 'F1_problem_rescue_vs_harm'), {'rescue_scene': rs, 'harm_scene': 51157, 'matrix': m}


# ------------------------------------------------------------------ F2 method schematic (no data)
def f2():
    fig, ax = plt.subplots(figsize=(W2, 2.55)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 38)
    def box(x, y, w, h, txt, fc='#F4F7FB', ec='#4A6A8A', fs=5.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec=ec, lw=.8))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=COL['ink'], linespacing=1.25)
    def arr(x0, y0, x1, y1, c='#4A6A8A'): ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=c, lw=.9, shrinkA=0, shrinkB=0))
    box(0.5, 21, 16, 14, 'Frozen diffusion policy\n(ResNet18 encoder,\n2 training seeds)\nC0: run unchanged')
    box(0.5, 3, 16, 14, 'Observations\nbase RGB, wrist RGB,\nrobot state\n(no object pose,\nno success label)')
    box(20, 21, 15.5, 14, 'Trigger\nfirst gripper-open\ncommand while the\ncube is held')
    box(20, 3, 15.5, 14, 'Memory\nhand / base / hold\nestimates, accepted\nafter 3 confirmations')
    box(39, 10, 19.5, 18, 'Decision\nACT if target and held\ncube are estimated,\nmemory not contradicted,\noffset > reading bound u;\notherwise ABSTAIN', fc='#EEF6FF')
    box(62, 21, 17, 14, 'ACT: bounded XY move\n<= 12 steps of 5 mm,\ntarget capped at 30 mm,\nno Z or gripper change', fc='#E7F4EE', ec='#2E7D5B')
    box(62, 3, 17, 14, 'ABSTAIN\nbranch equals C0\n(same release,\nsame trajectory)', fc='#F2F2F2', ec='#888888')
    box(82.5, 10, 17, 18, 'Frozen policy resumes\n(release, then the task\ncontinues to step 200)\n\npaired outcome\nY_g vs Y_0, same scene')
    arr(17, 28, 20, 28); arr(17, 10, 20, 10); arr(35.8, 28, 39, 23); arr(35.8, 10, 39, 15)
    arr(58.8, 23, 62, 28, '#2E7D5B'); arr(58.8, 15, 62, 10, '#888888'); arr(79.3, 28, 82.5, 23); arr(79.3, 10, 82.5, 15)
    ax.text(50, 0.0, 'Each triggered scene is branched from one simulator snapshot: C0, V2 and every ablation start from the identical state (paired design).', ha='center', fontsize=5.8, color=COL['muted'])
    return save(fig, 'F2_method'), {}


# ------------------------------------------------------------------ F3 statistics: calibration certificate vs Test efficacy
def f3():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={'width_ratios': [1.05, 1.15, 1.0], 'wspace': 0.55})
    ax = axs[0]; panel(ax, 'a')
    for i, pid in enumerate(POL):
        c = R['calibration'][pid]; up = c['harm_cp95_upper_POSTHOC']; rate = c['harm_rate']
        ax.errorbar(i, rate * 100, yerr=[[rate * 100], [(up - rate) * 100]], fmt='o', ms=4.5, color=COL['v2'], capsize=3, lw=1)
        ax.text(i, up * 100 + 0.35, f'{c["harm"]}/{c["ref_successes"]}\np={c["harm_p_B"]:.1e}', ha='center', fontsize=6.2)
    ax.axhline(5, color=COL['harm'], ls='--', lw=1); ax.text(1.45, 5.15, 'h* = 5%', color=COL['harm'], fontsize=6.3, ha='right')
    ax.set_xticks([0, 1], ['seed 1', 'seed 2']); ax.set_xlim(-.6, 1.6); ax.set_ylim(0, 6); ax.set_ylabel('conditional harm (%)')
    ax.set_title('Calibration: harm certificate', fontsize=7.5); ax.grid(axis='y', color=COL['grid'], lw=.5)
    ax = axs[1]; panel(ax, 'b')
    for i, pid in enumerate(POL):
        t = R['test'][pid]; n = t['n']
        ax.bar(i - .18, t['success_ref'] / n * 100, .34, color=COL['c0'], label='C0' if i == 0 else None)
        ax.bar(i + .18, t['success_arm'] / n * 100, .34, color=COL['v2'], label='V2' if i == 0 else None)
        ax.text(i - .18, t['success_ref'] / n * 100 + .4, t['success_ref'], ha='center', fontsize=6.2)
        ax.text(i + .18, t['success_arm'] / n * 100 + .4, t['success_arm'], ha='center', fontsize=6.2)
    ax.set_ylim(80, 100); ax.set_xticks([0, 1], ['seed 1', 'seed 2']); ax.set_ylabel('task success (% of 400)')
    ax.legend(frameon=False, loc='upper left', ncol=2); ax.set_title('Final Test: success', fontsize=7.5); ax.grid(axis='y', color=COL['grid'], lw=.5)
    ax = axs[2]; panel(ax, 'c')
    for i, pid in enumerate(POL):
        t = R['test'][pid]
        ax.barh(i + .17, t['rescue'], .32, color=COL['rescue'], label='rescue' if i == 0 else None)
        ax.barh(i - .17, t['harm'], .32, color=COL['harm'], label='harm' if i == 0 else None)
        ax.text(t['rescue'] + .6, i + .17, f"{t['rescue']}", va='center', fontsize=6.2); ax.text(t['harm'] + .6, i - .17, f"{t['harm']}", va='center', fontsize=6.2)
        ax.text(t['rescue'] + 5.5, i + .17, f"p={t['eff_p_B']:.1e}", va='center', fontsize=5.8, color=COL['muted'])
    ax.set_yticks([0, 1], ['seed 1', 'seed 2']); ax.set_xlim(0, 48); ax.set_xlabel('discordant scenes'); ax.legend(frameon=False, loc='upper right')
    ax.set_title('Final Test: rescue vs harm', fontsize=7.5); ax.invert_yaxis()
    return save(fig, 'F3_statistics_calibration_vs_test'), {}


# ------------------------------------------------------------------ F4 memory sources and ablation chain (C1 development)
ARMS = [('C0', 'C0'), ('VISION_CENTER_V1', 'V1 (wrist only)'), ('V2_SINGLE_FRAME', 'V2 single frame'), ('V2_NO_HAND_MEMORY', 'V2 - hand memory'),
        ('V2_NO_BASE_MEMORY', 'V2 - base memory'), ('V2_NO_HOLD_MEMORY', 'V2 - hold memory'), ('V2_RULES_OFF', 'V2 rules off'), ('VISION_CENTER_V2', 'V2 (full)'), ('CENTER_ORACLE', 'oracle (privileged)')]


def f4():
    src = {p: Counter() for p in POL}; dec = {p: Counter() for p in POL}
    for pid in POL:
        for s in range(51100, 51300):
            r = json.loads(use(C1 / f'pass_{pid}/scenes/{s}.json').read_text(encoding='utf-8'))
            if not r['triggered']: continue
            c = cell(r, 'VISION_CENTER_V2')['control'] or {}; dec[pid][c.get('decision', 'none')] += 1
            rd = (c.get('readings') or [{}])[0]; src[pid][rd.get('target_src') or 'none'] += 1
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw={'width_ratios': [1, 1.6], 'wspace': 0.55})
    ax = axs[0]; panel(ax, 'a'); keys = sorted(set(src['train_seed_1']) | set(src['train_seed_2']), key=lambda k: -(src['train_seed_1'][k] + src['train_seed_2'][k]))
    y = np.arange(len(keys))
    for i, pid in enumerate(POL):
        ax.barh(y + (.18 if i == 0 else -.18), [src[pid][k] for k in keys], .34, color=COL['v2'] if i == 0 else COL['single'], label=PLAB[pid])
    ax.set_yticks(y, [k.replace('_', ' ') for k in keys]); ax.invert_yaxis(); ax.set_xlabel('triggered scenes'); ax.legend(frameon=False, loc='lower right')
    ax.set_title('Target source at the first V2 reading', fontsize=7.5)
    ax = axs[1]; panel(ax, 'b'); c1 = R['C1_development']; x = np.arange(len(ARMS))
    for i, pid in enumerate(POL):
        vals = [c1[pid]['C0']['success'] if a == 'C0' else c1[pid][a]['success_arm'] for a, _ in ARMS]
        ax.plot(x, vals, 'o-' if i == 0 else 's--', color=COL['v2'] if i == 0 else COL['single'], ms=4, lw=1, label=PLAB[pid])
        for xi, (a, _) in zip(x, ARMS):
            if a in ('C0',): continue
            e = c1[pid][a]; ax.text(xi + (.0), vals[xi] + (1.6 if i == 0 else -3.2), f"+{e['rescue']}/-{e['harm']}", fontsize=5.4, ha='center', color=COL['muted'])
    ax.set_xticks(x, [l for _, l in ARMS], rotation=35, ha='right'); ax.set_ylabel('success of 200 scenes'); ax.set_ylim(155, 195)
    ax.grid(axis='y', color=COL['grid'], lw=.5); ax.legend(frameon=False, loc='lower right'); ax.set_title('Ablation chain (labels: rescue/harm vs C0)', fontsize=7.5)
    return save(fig, 'F4_memory_and_ablation'), {'target_src_first_reading': {p: dict(src[p]) for p in POL}, 'decision': {p: dict(dec[p]) for p in POL}}


# ------------------------------------------------------------------ F5 spatial: 51157 harm case (XY occupancy, 3D boxes, TCP path)
def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def cube_corners(p, q, h=0.02):
    c = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)]); return c @ quat_R(q).T + np.asarray(p)


def cube_faces(P):
    idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]; return [[P[i] for i in f] for f in idx]


def footprint(P):
    xy = P[:, :2]; from scipy.spatial import ConvexHull
    return xy[ConvexHull(xy).vertices]


def f5():
    r, tr, fr = scene('train_seed_1', 51157); t0 = r['trigger']['step']; v2 = tr['VISION_CENTER_V2']; c0 = tr['C0']; n = cell(r, 'VISION_CENTER_V2')['control_steps']
    fig = plt.figure(figsize=(W2, 2.75)); gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.35)
    ax = fig.add_subplot(gs[0]); panel(ax, 'a')
    st0 = v2[0]; ste = next(s for s in v2 if s['t'] == t0 + n)
    for s, ls, lab in ((st0, '-', f't={t0} (trigger)'), (ste, '--', f't={t0 + n} (end of move)')):
        for key, colr in (('cubeA', COL['v2']), ('cubeB', COL['c0'])):
            ax.add_patch(Polygon(footprint(cube_corners(s[key], s[key + '_q'])) * 1000, closed=True, fc='none', ec=colr, ls=ls, lw=1.1))
        ax.plot([], [], ls=ls, color=COL['ink'], lw=1, label=lab)
    for tr_, colr, lab in ((c0, COL['c0'], 'TCP, C0'), (v2, COL['harm'], 'TCP, V2')):
        P = np.array([s['tcp'] for s in tr_ if s['t'] <= t0 + n + 12]) * 1000; ax.plot(P[:, 0], P[:, 1], color=colr, lw=1.1, label=lab)
    tg = cell(r, 'VISION_CENTER_V2')['control']['readings'][0]['target_xy']; ax.plot(tg[0] * 1000, tg[1] * 1000, 'x', color=COL['oracle'], ms=6, mew=1.5, label='V2 target estimate')
    ax.text(st0['cubeB'][0] * 1000 + 14, st0['cubeB'][1] * 1000 - 14, 'B', ha='center', va='center', fontsize=7, color=COL['c0']); ax.text(st0['cubeA'][0] * 1000 + 12, st0['cubeA'][1] * 1000 + 10, 'A', ha='center', va='center', fontsize=7, color=COL['v2'])
    ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.legend(frameon=False, fontsize=5.6, loc='upper left', bbox_to_anchor=(0, -0.25), ncol=2)
    ax.set_title('XY occupancy (cube footprints)', fontsize=7.5)
    ax = fig.add_subplot(gs[1], projection='3d'); ax.text2D(-0.02, 1.0, 'b', transform=ax.transAxes, fontweight='bold', fontsize=9)
    for key, colr in (('cubeA', COL['v2']), ('cubeB', COL['c0'])):
        ax.add_collection3d(Poly3DCollection(cube_faces(cube_corners(st0[key], st0[key + '_q']) * 1000), fc=colr, alpha=.25, ec=colr, lw=.4))
    for tr_, colr in ((c0, COL['c0']), (v2, COL['harm'])):
        P = np.array([s['tcp'] for s in tr_]) * 1000; ax.plot(P[:, 0], P[:, 1], P[:, 2], color=colr, lw=1)
    L = np.array([st0['link_pose'][k][:3] for k in sorted(st0['link_pose']) if k.startswith('panda_link')]) * 1000
    ax.plot(L[:, 0], L[:, 1], L[:, 2], 'o-', color='#555555', ms=2, lw=.8)
    ax.set_xlabel('x (mm)', labelpad=-8, fontsize=6); ax.set_ylabel('y (mm)', labelpad=-8, fontsize=6); ax.set_zlabel(''); ax.tick_params(pad=-4, labelsize=5)
    ax.view_init(elev=22, azim=-60); ax.set_title('3D at t=%d (z up, 0-600 mm): cubes,\nTCP paths, robot link frames' % t0, fontsize=7)
    ax = fig.add_subplot(gs[2]); panel(ax, 'c')
    for tr_, colr, lab in ((c0, COL['c0'], 'C0'), (v2, COL['harm'], 'V2')):
        T = [s['t'] for s in tr_]; ax.plot(T, [s['tcp'][2] * 1000 for s in tr_], color=colr, lw=1.1, label=f'TCP z, {lab}')
        ax.plot(T, [s['cubeA'][2] * 1000 for s in tr_], color=colr, lw=.8, ls=':', label=f'cube A z, {lab}')
    ax.axvspan(t0, t0 + n, color=COL['harm'], alpha=.08, lw=0); ax.text(t0 + n + 1, 128, 'V2 move', ha='left', fontsize=6, color=COL['harm'])
    ax.set_xlabel('control step'); ax.set_ylabel('height (mm)'); ax.legend(frameon=False, fontsize=5.4, loc='center right', bbox_to_anchor=(1.0, 0.62)); ax.set_title('Heights after the trigger', fontsize=7.5)
    g = cell(r, 'VISION_CENTER_V2')['gt_effect_of_control']
    return save(fig, 'F5_spatial_51157'), {'trigger': t0, 'control_steps': n, 'gt_effect_of_control': g}


# ------------------------------------------------------------------ F6 control: commanded vs actual (Check 4)
def f6():
    rows = [x for x in rcsv(DIAG / 'CHECK4_STEPWISE.csv') if x['policy'] == 'train_seed_1' and x['scene'] == '51157' and x['arm'] == 'VISION_CENTER_V2']
    mv = rcsv(DIAG / 'CHECK4_ALL_MOVES.csv')
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={'wspace': 0.5})
    ax = axs[0]; panel(ax, 'a'); k = [int(x['step']) for x in rows]
    ax.plot(k, [float(x['cmd_norm_mm']) for x in rows], 's-', color=COL['c0'], ms=3, lw=1, label='commanded |dxy|')
    ax.plot(k, [float(x['act_norm_mm']) for x in rows], 'o-', color=COL['v2'], ms=3, lw=1, label='actual |dxy|')
    ax.plot(k, [float(x['progress_along_cmd_mm']) for x in rows], '^:', color=COL['rescue'], ms=3, lw=1, label='progress along cmd')
    ax.set_xlabel('move step'); ax.set_ylabel('TCP displacement per step (mm)'); ax.legend(frameon=False, fontsize=5.8); ax.set_title('Scene 51157, seed 1, V2', fontsize=7.5)
    ax = axs[1]; panel(ax, 'b'); ax.plot(k, [float(x['cos_act_cmd']) for x in rows], 'o-', color=COL['v2'], ms=3, lw=1)
    ax.set_ylim(-1.05, 1.05); ax.axhline(0, color=COL['grid'], lw=.6); ax.set_xlabel('move step'); ax.set_ylabel('cos(actual, commanded)'); ax.set_title('Direction agreement', fontsize=7.5)
    ax = axs[2]; panel(ax, 'c'); a = np.array([float(x['first_step_cos']) for x in mv if x['first_step_cos'] not in ('', 'NA')])
    b = np.array([float(x['step2plus_min_cos']) for x in mv if x['step2plus_min_cos'] not in ('', 'NA')])
    bins = np.linspace(min(0.0, float(a.min())), 1, 26); ax.hist(a, bins, color=COL['oracle'], alpha=.85, label=f'step 1 (n={len(a)})'); ax.hist(b, bins, histtype='step', color=COL['v2'], lw=1.2, label=f'min over steps 2+ (n={len(b)})')
    ax.axvline(0.9, color=COL['harm'], ls='--', lw=.8); ax.text(0.885, 250, 'Check 4\nthreshold\ncos = 0.9', ha='right', va='center', fontsize=5.6, color=COL['harm']); ax.set_yscale('log')
    ax.set_xlabel('cos(actual, commanded)'); ax.set_ylabel('acted branches (log)'); ax.legend(frameon=False, fontsize=5.6, loc='center left', bbox_to_anchor=(0, .42)); ax.set_title(f'All {len(mv)} C1 moves (post hoc)', fontsize=7.5)
    return save(fig, 'F6_control_check4'), {'n_moves': len(mv), 'step1_cos_below_0.9': int((a < .9).sum()), 'step2plus_min_cos_below_0.9': int((b < .9).sum()), 'n_step1': len(a), 'n_step2plus': len(b), 'scene_51157_step1_cos': float(rows[0]['cos_act_cmd'])}


# ------------------------------------------------------------------ F7 physics: S6, static check, sub-step timing (B4)
def f7():
    g = [x for x in rcsv(DIAG / 'S6_ERRORS_BY_GROUP.csv') if x['grouping'] == 'kind' and x['policy'].startswith('both')]
    g.sort(key=lambda x: -int(x['n'])); sb = json.loads(use(B4).read_text(encoding='utf-8')); st = rcsv(STATIC)
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw={'width_ratios': [1.35, 1, 1], 'wspace': 0.55})
    ax = axs[0]; panel(ax, 'a'); y = np.arange(len(g))
    ax.barh(y, [float(x['over_tol_fraction']) * 100 for x in g], color=COL['harm'], height=.6)
    for i, x in enumerate(g): ax.text(float(x['over_tol_fraction']) * 100 + .4, i, f"{int(x['over_tol']):,}/{int(x['n']):,}", va='center', fontsize=5.4)
    ax.set_yticks(y, [x['level'] for x in g]); ax.invert_yaxis(); ax.set_xlim(0, 42); ax.set_xlabel('rows over 1 mm (%)'); ax.set_title('S6: logged separation vs\nreconstructed gap (official FAIL)', fontsize=7.2)
    ax = axs[1]; panel(ax, 'b'); band = ['static <1 mm/s', 'slow 1-10 mm/s', 'moving 10-100 mm/s', 'fast >100 mm/s']
    L = sb['groups']['last']; post = [L[f'ALL | {b}']['over_1mm_post'] / L[f'ALL | {b}']['n'] * 100 for b in band]; pre = [L[f'ALL | {b}']['over_1mm_pre'] / L[f'ALL | {b}']['n'] * 100 for b in band]
    x = np.arange(4); ax.bar(x - .2, post, .38, color=COL['c0'], label='end-of-step pose'); ax.bar(x + .2, pre, .38, color=COL['v2'], label='pre-sub-step pose')
    ax.set_xticks(x, ['<1', '1-10', '10-100', '>100'], fontsize=6.5); ax.set_xlabel('relative speed (mm/s)'); ax.set_ylabel('pairs over 1 mm (%)'); ax.legend(frameon=False, fontsize=5.8, loc='upper left')
    a = L['ALL | ALL']; ax.set_title(f"B4 sub-step alignment\n{a['over_1mm_post']:,} -> {a['over_1mm_pre']:,} of {a['n']:,}", fontsize=7.2)
    ax = axs[2]; panel(ax, 'c'); stl = [x for x in st if x['min_sep_minus_gap_before_mm'] not in ('', 'NA')]; gb = np.array([float(x['gap_before_last_step_m']) * 1000 for x in stl]); d = np.array([float(x['min_sep_minus_gap_before_mm']) for x in stl])
    ax.scatter(gb, d, s=9, color=COL['v2'], ec='white', lw=.3); ax.axhline(0, color=COL['grid'], lw=.6)
    ax.set_xlabel('gap before last sub-step (mm)'); ax.set_ylabel('PhysX sep. - gap (mm)'); ax.set_title(f'Static known-geometry cases\n({len(stl)} with listed points of {len(st)})', fontsize=7.2)
    return save(fig, 'F7_physics_s6_b4'), {'b4_all': a, 'static_cases': len(st), 'static_with_points': len(stl), 'static_max_abs_mm': float(np.abs(d).max())}


# ------------------------------------------------------------------ F8 hull sensitivity: S4 and S5 unchanged; 51152 oracle overlap
def f8():
    br = [x for x in rcsv(HULL / 'S45_BRANCH_RESULTS.csv') if x['acted'] == 'True' and x['S4_min_distance_mm_hull'] not in ('NA', '')]
    ms = rcsv(HULL / 'MESH_SHAPE_STATS.csv')
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.4})
    ax = axs[0]; panel(ax, 'a'); arms = sorted({x['arm'] for x in br}); armcol = {'CENTER_ORACLE': COL['oracle'], 'VISION_CENTER_V2': COL['v2']}
    for a in arms:
        d = np.array([float(x['S4_min_distance_mm_hull']) for x in br if x['arm'] == a])
        ax.scatter(np.full(len(d), arms.index(a)) + np.random.default_rng(0).uniform(-.25, .25, len(d)), d, s=3, color=armcol.get(a, COL['abl']), alpha=.6, lw=0)
    neg = [x for x in br if float(x['S4_min_distance_mm_hull']) < 0]
    for x in neg: ax.annotate(f"{x['policy'].replace('train_', '')} {x['scene']} {x['arm']}\n{float(x['S4_min_distance_mm_hull']):.2f} mm", (arms.index(x['arm']), float(x['S4_min_distance_mm_hull'])), (arms.index(x['arm']) + 1.2, -3.2), fontsize=5.6, arrowprops=dict(arrowstyle='-', lw=.5, color=COL['harm']), color=COL['harm'])
    ax.axhline(0, color=COL['harm'], lw=.7, ls='--'); ax.set_ylim(-5, None); ax.set_xticks(range(len(arms)), [a.replace('VISION_CENTER_', 'VC_').replace('_', ' ') for a in arms], rotation=35, ha='right', fontsize=5.8)
    ax.set_ylabel('S4 min. robot-cube B distance (mm)'); ax.set_title(f'S4 on convex hulls: {len(br)} acted branches,\nchange vs logged mesh = 0 in all', fontsize=7)
    ax = axs[1]; panel(ax, 'b'); ms = [x for x in ms]; y = np.arange(len(ms))
    ax.barh(y, [(1 - float(x['volume_ratio_logged_to_hull'])) * 100 for x in ms], color=COL['single'], height=.6)
    ax.set_yticks(y, [x['body'].replace('panda_', '') for x in ms]); ax.invert_yaxis(); ax.set_xlabel('hull volume excess (%)'); ax.set_title('Logged mesh vs convex hull', fontsize=7.5)
    return save(fig, 'F8_hull_sensitivity'), {'acted_branches': len(br), 'negative_S4': [(x['policy'], x['scene'], x['arm'], x['S4_min_distance_mm_hull']) for x in neg], 'S4_changed': sum(1 for x in br if x['S4_status_changed'] == 'True')}


# ------------------------------------------------------------------ F9 generalization to a second architecture (B3, development)
def f9():
    b3 = R['B3_development']; c1 = R['C1_development']; arms = [('V2_SINGLE_FRAME', 'V2 single frame'), ('VISION_CENTER_V2', 'V2'), ('CENTER_ORACLE', 'oracle')]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.25), gridspec_kw={'width_ratios': [1, 1.2], 'wspace': 0.45})
    ax = axs[0]; panel(ax, 'a'); x = np.arange(4); vals = [b3['C0']] + [b3[a]['success_arm'] for a, _ in arms]
    ax.bar(x, vals, color=[COL['c0'], COL['single'], COL['v2'], COL['oracle']], width=.6)
    for i, v in enumerate(vals): ax.text(i, v + 2, v, ha='center', fontsize=6.5)
    ax.set_xticks(x, ['C0'] + [l for _, l in arms], fontsize=6.5); ax.set_ylabel('success of 200 scenes'); ax.set_ylim(0, 110); ax.set_title('PlainConv policy (dev seeds 51300-51499)', fontsize=7.2)
    ax = axs[1]; panel(ax, 'b'); groups = [('C1 seed 1\n(ResNet18)', c1['train_seed_1']), ('C1 seed 2\n(ResNet18)', c1['train_seed_2']), ('B3\n(PlainConv)', b3)]
    for i, (lab, d) in enumerate(groups):
        for j, (a, _) in enumerate(arms):
            e = d[a]; xx = i + (j - 1) * .26
            ax.bar(xx, e['rescue'], .24, color=[COL['single'], COL['v2'], COL['oracle']][j], label=_ if i == 0 else None)
            ax.bar(xx, -e['harm'], .24, color=[COL['single'], COL['v2'], COL['oracle']][j], hatch='////', ec='white', lw=0)
            ax.text(xx, e['rescue'] + .8, e['rescue'], ha='center', fontsize=5.6); ax.text(xx, -e['harm'] - 3.2, e['harm'], ha='center', fontsize=5.6)
    ax.axhline(0, color=COL['ink'], lw=.6); ax.set_xticks(range(3), [g for g, _ in groups], fontsize=6.3); ax.set_ylabel('rescues (up) / harms (down, hatched)')
    ax.set_ylim(-12, 52); ax.legend(frameon=False, fontsize=5.8, loc='upper left'); ax.set_title('Rescue and harm vs C0 by policy', fontsize=7.2)
    return save(fig, 'F9_generalization_plainconv'), {}


def main():
    OUT.mkdir(parents=True, exist_ok=True); notes = {}
    for f in (f1, f2, f3, f4, f5, f6, f7, f8, f9):
        p, info = f(); notes[p.name] = {'sha256': sha(p), 'info': info}; print(p.name, json.dumps(info, default=str)[:400])
    script = Path(__file__); prov = {'generator': str(script.relative_to(REPO)).replace('\\', '/'), 'generator_sha256': sha(script),
                                     'matplotlib': matplotlib.__version__, 'inputs_sha256': dict(sorted(USED.items())), 'outputs': notes}
    (OUT / 'FIGURE_PROVENANCE.json').write_text(json.dumps(prov, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
