"""Supplementary figures S1-S3 from the original records (read-only); provenance in figures/FIGURE_PROVENANCE_SUPP.json.

S1  scene 51157 (policy seed 1, development): every stored V2 control frame and C0/V2 frames after the hand-back
S2  V2 decision at the trigger (ACT or abstention reason) in every stage, from the raw scene records
S3  V2 memory timeline for 51157: steps at which hand, base and hold observations were accepted

usage: python -P paper/ras_when_recovery_hurts_001/figures/source_code/make_supp_figures.py
"""
from __future__ import annotations

import hashlib, json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]; OUT = REPO / 'paper/ras_when_recovery_hurts_001/figures'
SRC = {'C1': (REPO / 'runs/vision_center_v2_c1_001_server/extracted/runs/vision_center_v2_c1_001', 'pass_{p}', range(51100, 51300)),
       'Calibration': (REPO / 'runs/ras_exec_001/download_final/runs/ras_v2_confirm_001/calibration', 'pass_{p}', range(30000, 30400)),
       'Final Test': (REPO / 'runs/ras_exec_001/download_test/runs/ras_v2_confirm_001/test', 'pass_{p}', range(90000, 90400))}
POL = ('train_seed_1', 'train_seed_2'); USED = {}
plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6})


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def use(p):
    USED[str(Path(p).relative_to(REPO)).replace('\\', '/')] = sha(p); return Path(p)


def save(fig, name):
    p = OUT / f'{name}.pdf'
    if p.exists(): p.unlink()
    fig.savefig(p, bbox_inches='tight', pad_inches=.02); plt.close(fig); return p


def s1():
    d = SRC['C1'][0] / 'pass_train_seed_1'; r = json.loads(use(d / 'scenes/51157.json').read_text(encoding='utf-8')); fr = np.load(use(d / r['frames']))
    v2 = next(c for c in r['cells'] if c['arm'] == 'VISION_CENTER_V2'); t0 = r['trigger']['step']
    ctrl = fr['VISION_CENTER_V2']; after_v2 = fr['after_VISION_CENTER_V2']; after_c0 = fr['after_C0']
    st_v2 = v2['frames_after_trigger']['steps']; c0 = next(c for c in r['cells'] if c['arm'] == 'C0'); st_c0 = c0['frames_after_trigger']['steps']
    pick = lambda steps, want: [min(range(len(steps)), key=lambda i: abs(steps[i] - w)) for w in want]
    want = [130, 140, 150, 170, 190, 200]
    fig, axs = plt.subplots(4, 7, figsize=(7.2, 3.3), gridspec_kw={'hspace': .45, 'wspace': .05})
    for a in axs.flat: a.axis('off')
    for j in range(14):
        a = axs[j // 7, j % 7]; a.axis('off')
        if j < len(ctrl): a.imshow(ctrl[j], interpolation='nearest'); a.set_title(f'V2 move t={t0 + j}', fontsize=5.5, pad=1)
    for row, (frames, steps, lab) in enumerate(((after_c0, st_c0, 'C0'), (after_v2, st_v2, 'V2')), start=2):
        idx = sorted(set(pick(steps, want)))
        axs[row, 0].axis('off'); axs[row, 0].text(.5, .5, f'{lab}\nafter the\nhand-back', ha='center', va='center', fontsize=7, transform=axs[row, 0].transAxes)
        for k, i in enumerate(idx):
            a = axs[row, k + 1]; a.axis('off'); a.imshow(frames[i], interpolation='nearest'); a.set_title(f'{lab} t={steps[i]}', fontsize=5.5, pad=1)
    return save(fig, 'S1_frames_51157'), {'control_frames': int(len(ctrl)), 'c0_steps_shown': sorted({st_c0[i] for i in pick(st_c0, want)}), 'v2_steps_shown': sorted({st_v2[i] for i in pick(st_v2, want)})}


def s2():
    counts = {}
    for stage, (root, sub, seeds) in SRC.items():
        for p in POL:
            c = Counter()
            for s in seeds:
                r = json.loads(use(root / sub.format(p=p) / 'scenes' / f'{s}.json').read_text(encoding='utf-8'))
                if not r['triggered']: c['no trigger'] += 1; continue
                ctl = next(x for x in r['cells'] if x['arm'] == 'VISION_CENTER_V2')['control'] or {}; c[ctl.get('decision', 'none')] += 1
            counts[f'{stage} | {p}'] = dict(c)
    keys = ['ACT', 'ALIGNED', 'ALIGNED_WITHIN_UNCERTAINTY', 'LOW_CARRY', 'NO_HOLD_ESTIMATE', 'NO_TARGET_ESTIMATE', 'SIDE_CONTACT_RISK', 'MEMORY_CONTRADICTED', 'TOO_FEW_STEPS', 'no trigger']
    extra = sorted({k for v in counts.values() for k in v} - set(keys)); keys += extra
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); groups = list(counts); x = np.arange(len(keys)); w = .8 / len(groups)
    cols = ['#0072B2', '#56B4E9', '#009E73', '#8CD3B9', '#D55E00', '#F0B48A']
    for i, g in enumerate(groups):
        n = sum(counts[g].values()); ax.bar(x + (i - (len(groups) - 1) / 2) * w, [counts[g].get(k, 0) / n * 100 for k in keys], w, color=cols[i % 6], label=g.replace('train_seed_', 'policy seed '))
    ax.set_xticks(x, [k.replace('_', ' ').lower() for k in keys], rotation=30, ha='right', fontsize=6.5); ax.set_ylabel('% of scenes'); ax.set_yscale('log'); ax.set_ylim(.1, 100)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc='upper right'); ax.grid(axis='y', color='#DDDDDD', lw=.5)
    return save(fig, 'S2_decisions_by_stage'), counts


def s3():
    d = SRC['C1'][0] / 'pass_train_seed_1'; r = json.loads(use(d / 'scenes/51157.json').read_text(encoding='utf-8')); m = r['memory']; t0 = r['trigger']['step']
    fig, ax = plt.subplots(figsize=(7.2, 1.5))
    for i, (k, lab, c) in enumerate((('hand_steps', 'hand (wrist camera)', '#0072B2'), ('base_steps', 'base (base camera)', '#56B4E9'), ('hold_steps', 'hold (held cube)', '#E69F00'))):
        st = m.get(k, []); ax.scatter(st, np.full(len(st), i), s=6, color=c, marker='|'); ax.text(-3, i, f'{lab} ({len(st)})', ha='right', va='center', fontsize=6.5)
    ax.axvline(t0, color='#D55E00', lw=.8, ls='--'); ax.text(t0 + 1, 2.4, f'trigger t={t0}', fontsize=6.5, color='#D55E00')
    ax.set_yticks([]); ax.set_xlim(0, 125); ax.set_ylim(-.6, 2.8); ax.set_xlabel('control step'); ax.spines['left'].set_visible(False)
    return save(fig, 'S3_memory_timeline_51157'), {'log': m.get('log')}


def main():
    notes = {}
    for f in (s1, s2, s3):
        p, info = f(); notes[p.name] = {'sha256': sha(p), 'info': info}; print(p.name, json.dumps(info)[:600])
    s = Path(__file__)
    (OUT / 'FIGURE_PROVENANCE_SUPP.json').write_text(json.dumps({'generator': str(s.relative_to(REPO)).replace('\\', '/'), 'generator_sha256': sha(s), 'inputs_sha256': dict(sorted(USED.items())), 'outputs': notes}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
