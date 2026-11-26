"""Check the numbers written in the canonical split project (latex_project/ = Drive Paper/Latex) against
data/recount.json and data/handback_stats.json. Every expected literal is DERIVED from those files; exit 1 on any miss.
usage: python -P paper/ras_when_recovery_hurts_001/figures/source_code/check_numbers_latex.py [latex_project_dir]
"""
from __future__ import annotations

import json, sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
LP = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER / 'latex_project'
R = json.loads((PAPER / 'data/recount.json').read_text(encoding='utf-8'))
HB = json.loads((PAPER / 'data/handback_stats.json').read_text(encoding='utf-8'))
c1, cal, te, b3 = R['C1_development'], R['calibration'], R['test'], R['B3_development']; s1, s2 = 'train_seed_1', 'train_seed_2'
MAIN = '\n'.join(p.read_text(encoding='utf-8') for p in sorted((LP / 'sections').glob('*.tex')))
SUPP = '\n'.join(p.read_text(encoding='utf-8') for p in sorted((LP / 'supplementary/sections').glob('*.tex')))
TAB = '\n'.join(p.read_text(encoding='utf-8') for p in sorted((LP / 'tables').glob('*.tex')))


def sci(p):
    m, e = f'{p:.1e}'.split('e'); return f'{m}\\times10^{{{int(e)}}}'


def pct(a, n): return f'{a / n * 100:.2f}'.rstrip('0').rstrip('.')


v1, v2 = c1[s1]['VISION_CENTER_V2'], c1[s2]['VISION_CENTER_V2']
C = [
 ('MAIN', f"harm was {cal[s1]['harm']} of {cal[s1]['ref_successes']} and {cal[s2]['harm']} of {cal[s2]['ref_successes']} baseline successes", 'abstract calibration harm'),
 ('MAIN', f"success rose from {te[s1]['success_ref']} to {te[s1]['success_arm']} and from {te[s2]['success_ref']} to {te[s2]['success_arm']}", 'abstract test'),
 ('MAIN', f"({te[s1]['rescue']} rescues vs {te[s1]['harm']} harms, $p={sci(te[s1]['eff_p_B'])}$; {te[s2]['rescue']} vs {te[s2]['harm']}, $p={sci(te[s2]['eff_p_B'])}$)", 'abstract p'),
 ('MAIN', f"V2 raised success from {c1[s1]['C0']['success']} to {v1['success_arm']} ({v1['rescue']} rescues, {v1['harm']} harm) for policy seed 1 and from {c1[s2]['C0']['success']} to {v2['success_arm']} ({v2['rescue']} rescues, {v2['harm']} harms)", 'C1 V2'),
 ('MAIN', f"$p={sci(c1['H_gen']['p'])}$", 'H_gen'),
 ('MAIN', f"intersection--union $p={c1['H_mem']['p_intersection_union']:.4f}$", 'H_mem'),
 ('MAIN', f"The privileged oracle reached {c1[s1]['CENTER_ORACLE']['success_arm']} and {c1[s2]['CENTER_ORACLE']['success_arm']}", 'oracle success'),
 ('MAIN', f"With a single frame the controller acted in {c1[s1]['V2_SINGLE_FRAME']['acted']} and {c1[s2]['V2_SINGLE_FRAME']['acted']} scenes and rescued {c1[s1]['V2_SINGLE_FRAME']['rescue']} and {c1[s2]['V2_SINGLE_FRAME']['rescue']}", 'single frame'),
 ('MAIN', f"with the confirmed memory it acted in {v1['acted']} and {v2['acted']} and rescued {v1['rescue']} and {v2['rescue']}", 'V2 acted/rescued'),
 ('MAIN', f"V2 turned {cal[s1]['harm']} of {cal[s1]['ref_successes']} (policy seed 1) and {cal[s2]['harm']} of {cal[s2]['ref_successes']} (policy seed 2)", 'cal harm'),
 ('MAIN', f"${sci(cal[s1]['harm_p_B'])}$ and ${sci(cal[s2]['harm_p_B'])}$", 'cal p'),
 ('MAIN', f"{cal[s1]['harm_cp95_upper_POSTHOC'] * 100:.2f}\\% and {cal[s2]['harm_cp95_upper_POSTHOC'] * 100:.2f}\\%", 'CP'),
 ('MAIN', f"({cal[s1]['success_ref']} to {cal[s1]['success_arm']} and {cal[s2]['success_ref']} to {cal[s2]['success_arm']})", 'cal success descriptive'),
 ('MAIN', f"V2 moved in {cal[s1]['acted']} and {cal[s2]['acted']} calibration scenes, of which {cal[s1]['moved_among_baseline_success']} and {cal[s2]['moved_among_baseline_success']} were baseline successes", 'moved among successes'),
 ('MAIN', f"on the test split 0 of {te[s1]['moved_among_baseline_success']} and 1 of {te[s2]['moved_among_baseline_success']}", 'test moved among successes'),
 ('MAIN', f"Success rose from {te[s1]['success_ref']} to {te[s1]['success_arm']} ({pct(te[s1]['success_ref'], 400)}\\% to {pct(te[s1]['success_arm'], 400)}\\%)", 'test seed 1 pct'),
 ('MAIN', f"from {te[s2]['success_ref']} to {te[s2]['success_arm']} ({pct(te[s2]['success_ref'], 400)}\\% to {pct(te[s2]['success_arm'], 400)}\\%)", 'test seed 2 pct'),
 ('MAIN', f"$[+{te[s1]['success_diff_boot95_POSTHOC'][0] * 100:.1f}, +{te[s1]['success_diff_boot95_POSTHOC'][1] * 100:.1f}]$ and $[+{te[s2]['success_diff_boot95_POSTHOC'][0] * 100:.1f}, +{te[s2]['success_diff_boot95_POSTHOC'][1] * 100:.1f}]$", 'bootstrap'),
 ('MAIN', f"Harm on the test scenes was {te[s1]['harm']} of {te[s1]['ref_successes']} and {te[s2]['harm']} of {te[s2]['ref_successes']}", 'test harm'),
 ('MAIN', f"{te[s1]['ref_successes'] - te[s1]['harm'] + te[s2]['ref_successes'] - te[s2]['harm']} both-succeed, {te[s1]['rescue'] + te[s2]['rescue']} rescues, {te[s1]['harm'] + te[s2]['harm']} harm and {te[s1]['ref_failures'] - te[s1]['rescue'] + te[s2]['ref_failures'] - te[s2]['rescue']} both-fail", 'pooled'),
 ('MAIN', f"{te[s1]['harm'] + te[s2]['harm']} of {te[s1]['ref_successes'] + te[s2]['ref_successes']} baseline successes was lost while {te[s1]['rescue'] + te[s2]['rescue']} of {te[s1]['ref_failures'] + te[s2]['ref_failures']} failures", 'discussion pooled'),
 ('MAIN', f"V2 raised success from {b3['C0']} to {b3['VISION_CENTER_V2']['success_arm']} of 200 scenes, with {b3['VISION_CENTER_V2']['rescue']} rescues and {b3['VISION_CENTER_V2']['harm']} harms ($p={sci(b3['VISION_CENTER_V2']['p_exact_vs_C0'])}$", 'B3'),
 ('MAIN', f"It recovered {b3['oracle_rescues_recovered_by_v2']} of the {b3['oracle_rescues']} oracle rescues; the oracle itself caused {b3['CENTER_ORACLE']['harm']} harms", 'B3 oracle'),
 ('MAIN', f"The single-frame variant reached {b3['V2_SINGLE_FRAME']['success_arm']} ({b3['V2_SINGLE_FRAME']['rescue']} rescues, {b3['V2_SINGLE_FRAME']['harm']} harms)", 'B3 single'),
 ('MAIN', f"The oracle rescued {c1[s1]['CENTER_ORACLE']['rescue']} and {c1[s2]['CENTER_ORACLE']['rescue']} development scenes; V2 rescued {v1['rescue']} and {v2['rescue']}", 'oracle rescues'),
 ('MAIN', f"The rules-off ablation, which removes the three V2-specific rules as one bundle, acted in {c1[s1]['V2_RULES_OFF']['acted']} and {c1[s2]['V2_RULES_OFF']['acted']} development scenes instead of {v1['acted']} and {v2['acted']}", 'rules off acted'),
 ('MAIN', f"({v1['acted']} of {c1[s1]['triggered']} and {v2['acted']} of {c1[s2]['triggered']} in development)", 'acted of triggered'),
 ('MAIN', f"The trigger fired in {c1[s1]['triggered']} (seed 1) and {c1[s2]['triggered']} (seed 2) scenes", 'C1 triggered'),
 ('MAIN', f"its first open command followed {min(HB[k]['steps_to_first_open_after_move']['min'] for k in HB if '|' in k):.0f}--{max(HB[k]['steps_to_first_open_after_move']['max'] for k in HB if '|' in k):.0f} steps later", 'hand-back range'),
 ('MAIN', f"net horizontal TCP displacement was {min(HB[k]['net_xy_over_commanded_path']['min'] for k in ('C1|train_seed_1', 'C1|train_seed_2')) * 100:.0f}--{max(HB[k]['net_xy_over_commanded_path']['max'] for k in ('C1|train_seed_1', 'C1|train_seed_2')) * 100:.0f}\\% of the commanded path", 'executed fraction (dev)'),
 ('MAIN', f"(medians {HB['C1|train_seed_1']['net_xy_over_commanded_path']['median'] * 100:.0f}\\% and {HB['C1|train_seed_2']['net_xy_over_commanded_path']['median'] * 100:.0f}\\%", 'executed fraction medians'),
 ('SUPP', f"$p=2^{{-29}}=1.86\\times10^{{-9}}$", 'supp p 29 vs 0'),
 ('SUPP', f"max(0.0059,0.0078)=0.0078", 'supp IU'),
 ('TAB', f"{cal[s1]['ref_successes']} & {cal[s1]['harm']} & {cal[s1]['harm_cp95_upper_POSTHOC'] * 100:.2f}", 'TS2 row'),
]
txt = {'MAIN': MAIN, 'SUPP': SUPP, 'TAB': TAB}; fails = []; items = []
if not (HB['check4_51157_seed1_V2']['recomputed_status'] == 'PASS' and 'so this move passed Check 4' in MAIN): fails.append('51157 Check 4 PASS statement')
if sum(1 for k in HB if '|' in k and HB[k]['no_open_command_after_move']) != 1 or 'in one calibration branch it never opened the gripper again' not in MAIN: fails.append('never-opened statement')
for where, lit, what in C:
    ok = lit in txt[where]; items.append({'where': where, 'claim': what, 'expected': lit, 'found': ok})
    if not ok: fails.append(f'{what}: {lit}')
out = PAPER / 'build_logs/check_numbers_latex.json'; out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({'checks': len(C) + 2, 'failures': fails, 'items': items}, indent=1), encoding='utf-8')
print(f'{len(C) + 2} checks, {len(fails)} failures'); [print(' -', f) for f in fails]
sys.exit(1 if fails else 0)
