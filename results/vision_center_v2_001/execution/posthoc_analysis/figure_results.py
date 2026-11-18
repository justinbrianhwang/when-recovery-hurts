"""Vision-CENTER v2 execution, one figure from the stored result files (no new computation).
Usage (repository root): python results/vision_center_v2_001/execution/posthoc_analysis/figure_results.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E = Path('results/vision_center_v2_001/execution'); P = json.load(open(E / 'posthoc_analysis' / 'posthoc_analysis.json', encoding='utf-8')); B2 = P['B2_pilot_new_scenes']; B1 = P['B1_shadow_used_scenes']
INK = '#1f2328'; MUTED = '#6b7280'; GRID = '#e5e7eb'; BLUE = '#2f6fde'; GREY = '#8b95a1'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={'width_ratios': [1, 1.25]})
names = [('C0', 'no intervention'), ('V1', 'Vision-CENTER v1 (single frame)'), ('V2', 'Vision-CENTER v2 (memory)'), ('OR', 'CENTER_ORACLE (reference)')]
for i, (k, lab) in enumerate(names):
    v = B2['success'][k]; y = len(names) - 1 - i; ax[0].barh(y, v, height=0.5, color=BLUE if k == 'V2' else GREY); ax[0].text(v + 1, y, f'{v} / 200', va='center'); ax[0].text(101, y + 0.42, lab, va='bottom', color=MUTED)
ax[0].set_xlim(100, 205); ax[0].set_ylim(-0.6, len(names) - 0.1); ax[0].set_yticks([]); ax[0].spines['left'].set_visible(False); ax[0].set_xlabel('successful scenes of 200 (axis starts at 100)')
ax[0].set_title('Stage B2: success by arm', loc='left', fontsize=10)
orr = B2['oracle_rescues']; by = B2['v2_acts_by_target_source']
parts = [('current frame', by['current_frame']['oracle_rescue_among_them'], BLUE), ('hand-camera memory', by['hand_memory_past_only']['oracle_rescue_among_them'] + by['hand_memory_last_seen_at_trigger']['oracle_rescue_among_them'], BLUE),
         ('base-camera memory', by['base_memory_past_only']['oracle_rescue_among_them'] + by['base_memory_last_seen_at_trigger']['oracle_rescue_among_them'], BLUE)]
miss = {}
for m in orr['missed_by_v2']: miss[m['v2_decision']] = miss.get(m['v2_decision'], 0) + 1
short = {'LOW_CARRY': 'abstained: held cube too low', 'ALIGNED_WITHIN_UNCERTAINTY': 'abstained: offset inside the error bound', 'NO_HOLD_ESTIMATE': 'abstained: held-cube fit rejected', 'NO_TARGET_ESTIMATE': 'abstained: target never seen'}
parts += [(short.get(k, k), v, GREY) for k, v in sorted(miss.items(), key=lambda kv: -kv[1])]
for i, (lab, v, col) in enumerate(parts):
    y = len(parts) - 1 - i; ax[1].barh(y, v, height=0.5, color=col); ax[1].text(v + 0.15, y, str(v), va='center'); ax[1].text(-0.2, y, ('rescued by v2, ' if col == BLUE else 'missed, ') + lab, va='center', ha='right')
ax[1].set_xlim(0, 11); ax[1].set_ylim(-0.6, len(parts) - 0.4); ax[1].set_yticks([]); ax[1].spines['left'].set_visible(False); ax[1].set_xlabel('scenes')
ax[1].set_title(f"The {orr['n']} scenes the oracle rescues: v2 rescues {orr['also_v2']} (v1: {orr['also_v1']})", loc='left', fontsize=10)
fig.suptitle('Vision-CENTER v2 on 200 new scenes: 19 rescues, 0 harm; 10 of the 19 use a remembered target position', x=0.01, ha='left', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.subplots_adjust(wspace=0.95); fig.savefig(E / 'posthoc_analysis' / 'fig_b2_results.png', dpi=150); print('saved')
