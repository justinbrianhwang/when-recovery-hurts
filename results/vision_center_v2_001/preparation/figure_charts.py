"""Vision-CENTER v2, Phase A, charts from the offline analysis files of this folder (no new computation).
  fig2  when each estimate was available before the trigger, 15 failed diagnostic episodes with real video
  fig3  base-camera target error on the demonstrations: v1 centroid against the v2 largest-region estimate
  fig4  executed centring moves of run2 (v1 and oracle): offset and carry height at the trigger, pushed targets marked
Usage (repository root): python results/vision_center_v2_001/preparation/figure_charts.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P = Path('results/vision_center_v2_001/preparation'); F = P / 'figures'; F.mkdir(exist_ok=True)
INK = '#1f2328'; MUTED = '#6b7280'; GRID = '#e5e7eb'; BLUE = '#2f6fde'; ORANGE = '#d9822b'; GREY = '#8b95a1'; RED = '#c2362f'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
rd = lambda n: json.load(open(P / n, encoding='utf-8'))

# ---------------------------------------------------------------- fig2
V = [x for x in rd('memory_replay.json')['video_failed_diagnostic_episodes'] if x['triggered']]
order = {'ACT': 0, 'NO_TARGET_ESTIMATE': 1, 'NO_HOLD_ESTIMATE': 2}; V.sort(key=lambda x: (order.get(x['v1'], 9), x['seed']))
short = lambda s: s.replace('_ESTIMATE', '').replace('_', ' ').lower()
fig, ax = plt.subplots(figsize=(11, 0.46 * len(V) + 1.3))
for j, x in enumerate(V):
    y = len(V) - 1 - j; tl = x['timeline']; T = x['trigger_step']
    for key, dy, col in (('hand', 0.24, BLUE), ('base', 0.0, ORANGE), ('hold', -0.24, GREY)):
        ts = [t['t'] for t in tl if t[key]]; ax.vlines(ts, y + dy - 0.09, y + dy + 0.09, color=col, lw=2.2 if key == 'hold' else 1.4)
    ax.plot([T], [y], marker='|', color=INK, ms=16, mew=1.6); ax.text(-3, y, str(x['seed']), ha='right', va='center')
    same = x['v1'] == x['v2']; ax.text(132, y, f"{short(x['v1'])}  →  {short(x['v2'])}", va='center', color=MUTED if same else INK, fontweight='normal' if same else 'bold')
ax.set_xlim(-1, 131); ax.set_ylim(-0.7, len(V) - 0.3); ax.set_yticks([]); ax.set_xlabel('step of the episode (the black mark is the trigger: the policy is about to open the gripper)'); ax.spines['left'].set_visible(False)
ax.text(132, len(V) - 0.45, 'v1 at the trigger  →  v2 at the trigger', color=MUTED, va='bottom')
for k, (lab, col) in enumerate((('hand camera: target fit accepted', BLUE), ('base camera: target estimate accepted', ORANGE), ('hand camera: held-cube fit accepted (run every 4th step)', GREY))):
    ax.text(0.0, 1.10 - 0.035 * k, '█ ' + lab, color=col, transform=ax.transAxes, va='bottom')
fig.suptitle('Before the trigger, the base camera sees the target in every episode; the hand camera often does not', x=0.01, ha='left', fontsize=11, y=0.995)
fig.subplots_adjust(left=0.06, right=0.70, top=0.84, bottom=0.10); fig.savefig(F / 'fig2_observation_timeline_video_episodes.png', dpi=150); plt.close(fig)

# ---------------------------------------------------------------- fig3
B = rd('base_camera_audit.json'); e1 = np.sort(np.array(B['errors_m']['v1']) * 1000); e2 = np.sort(np.array(B['errors_m']['v2']) * 1000)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for e, col, lab in ((e1, GREY, 'v1 centroid of all lit-green pixels'), (e2, BLUE, 'v2 largest connected region')):
    share = 1 - np.arange(1, len(e) + 1) / len(e); ax.step(e, np.maximum(share, 1e-5), where='post', color=col, lw=2); ax.text(*((22, 0.07) if col == GREY else (7.5, 1.5e-4)), lab, color=col, va='bottom')
ax.set_yscale('log'); ax.set_xlim(0, 70); ax.set_ylim(1e-5, 1); ax.axvline(5, color=GRID, lw=1); ax.text(5.6, 0.4, 'bound used by v2: 5 mm', color=MUTED)
ax.set_xlabel('error of the target position, mm'); ax.set_ylabel('share of accepted frames with a larger error'); ax.grid(axis='y', color=GRID, lw=0.6)
s1, s2 = B['summary']['v1_full_view_centroid_accepted'], B['summary']['v2_largest_region_accepted']
fig.suptitle(f"Base camera, 500 demonstrations: errors over 10 mm fall from\n{s1['over_10mm']:,} of {s1['n']:,} accepted frames to {s2['over_10mm']} of {s2['n']:,}", x=0.01, ha='left', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(F / 'fig3_base_camera_error.png', dpi=150); plt.close(fig)

# ---------------------------------------------------------------- fig4
G = [x for x in rd('ground_truth_audit.json')['scenes'] if x['triggered']]
fig, ax = plt.subplots(figsize=(7.2, 4.2)); seen = set()
for key, mk in (('oracle', 'o'), ('v1', 's')):
    L = [x for x in G if f'{key}_pushed_target_m' in x and -5 < x['true_gap_m'] * 1000]
    ok = [x for x in L if x[f'{key}_pushed_target_m'] <= 0.001]; bad = [x for x in L if x[f'{key}_pushed_target_m'] > 0.001]
    ax.scatter([x['true_offset_m'] * 1000 for x in ok], [x['true_gap_m'] * 1000 for x in ok], s=22, marker=mk, facecolor='none', edgecolor=BLUE if key == 'v1' else GREY, lw=1.1, label=f"{'v1' if key == 'v1' else 'oracle'} move, target not pushed ({len(ok)})")
    ax.scatter([x['true_offset_m'] * 1000 for x in bad], [x['true_gap_m'] * 1000 for x in bad], s=60, marker=mk, color=RED, label=f"{'v1' if key == 'v1' else 'oracle'} move, target pushed over 1 mm ({len(bad)})", zorder=3)
    for x in bad:
        if x['scene'] not in seen: ax.annotate(str(x['scene']), (x['true_offset_m'] * 1000, x['true_gap_m'] * 1000), xytext=(6, 6), textcoords='offset points', color=RED); seen.add(x['scene'])
ax.axvline(40, color=GRID, lw=1); ax.axhline(3, color=GRID, lw=1); ax.fill_between([40, 60], -5, 3, color=RED, alpha=0.06, lw=0); ax.text(40.6, -4.3, 'v2 abstains here:\nbeside the target and\nnot clearly above it', color=MUTED, va='bottom')
ax.set_xlim(0, 60); ax.set_ylim(-5, 13); ax.set_xlabel('true horizontal offset between the cube centres at the trigger, mm'); ax.set_ylabel('held-cube bottom above the target top face, mm'); ax.legend(frameon=False, loc='upper left', fontsize=8)
fig.suptitle('Executed centring moves on the used scenes: the target was pushed only\nfrom beside it (one more pushed oracle move, at -26 mm, is off the chart)', x=0.01, ha='left', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(F / 'fig4_carry_height_and_pushed_target.png', dpi=150); plt.close(fig)
print('saved', sorted(p.name for p in F.glob('*.png')))
