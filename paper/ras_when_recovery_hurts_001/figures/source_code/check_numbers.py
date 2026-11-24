"""Check that every headline number written in the manuscripts equals the recount (data/recount.json).

Each claim is a (file, literal text, value derived from recount.json) triple: the literal must occur in the file AND the
derived string must equal the literal. Exit code 1 on any mismatch. Writes build_logs/check_numbers.json.
usage: python -P paper/ras_when_recovery_hurts_001/figures/source_code/check_numbers.py
"""
from __future__ import annotations

import json, sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]
R = json.loads((PAPER / 'data/recount.json').read_text(encoding='utf-8'))
c1, cal, te, b3 = R['C1_development'], R['calibration'], R['test'], R['B3_development']
s1, s2 = 'train_seed_1', 'train_seed_2'


def sci(p):
    m, e = f'{p:.1e}'.split('e'); return f'{m}\\times10^{{{int(e)}}}'


def pct(a, n): return f'{a / n * 100:.2f}'.rstrip('0').rstrip('.')


main, supp, abst = 'manuscripts/main.tex', 'manuscripts/supplementary.tex', 'manuscripts/abstract.txt'
C = [
    (main, f"harm was {cal[s1]['harm']} of {cal[s1]['ref_successes']} and {cal[s2]['harm']} of {cal[s2]['ref_successes']} baseline successes", 'abstract: calibration harm'),
    (main, f"success rose from {te[s1]['success_ref']} to {te[s1]['success_arm']} and from {te[s2]['success_ref']} to {te[s2]['success_arm']}", 'abstract: test success'),
    (main, f"({te[s1]['rescue']} rescues vs {te[s1]['harm']} harms, $p={sci(te[s1]['eff_p_B'])}$; {te[s2]['rescue']} vs {te[s2]['harm']}, $p={sci(te[s2]['eff_p_B'])}$)", 'abstract: test discordant'),
    (main, f"V2 raised success from {c1[s1]['C0']['success']} to {c1[s1]['VISION_CENTER_V2']['success_arm']} ({c1[s1]['VISION_CENTER_V2']['rescue']} rescues, {c1[s1]['VISION_CENTER_V2']['harm']} harm)", 'C1 seed 1'),
    (main, f"from {c1[s2]['C0']['success']} to {c1[s2]['VISION_CENTER_V2']['success_arm']} ({c1[s2]['VISION_CENTER_V2']['rescue']} rescues, {c1[s2]['VISION_CENTER_V2']['harm']} harms)", 'C1 seed 2'),
    (main, f"$p={sci(c1['H_gen']['p'])}$", 'H_gen p'),
    (main, f"intersection--union $p={c1['H_mem']['p_intersection_union']:.4f}$", 'H_mem p'),
    (main, f"V2 turned {cal[s1]['harm']} of {cal[s1]['ref_successes']} (policy seed 1) and {cal[s2]['harm']} of {cal[s2]['ref_successes']} (policy seed 2)", 'calibration harm'),
    (main, f"${sci(cal[s1]['harm_p_B'])}$ and ${sci(cal[s2]['harm_p_B'])}$", 'calibration p'),
    (main, f"{cal[s1]['harm_cp95_upper_POSTHOC'] * 100:.2f}\\% and {cal[s2]['harm_cp95_upper_POSTHOC'] * 100:.2f}\\%", 'CP bounds'),
    (main, f"V2 moved in {cal[s1]['acted']} and {cal[s2]['acted']} scenes", 'calibration acted'),
    (main, f"1 harm in {cal[s1]['moved_among_baseline_success']} and 0 in {cal[s2]['moved_among_baseline_success']}", 'calibration moved among successes'),
    (main, f"({cal[s1]['success_ref']} to {cal[s1]['success_arm']} and {cal[s2]['success_ref']} to {cal[s2]['success_arm']})", 'calibration success (descriptive)'),
    (main, f"Success rose from {te[s1]['success_ref']} to {te[s1]['success_arm']} ({pct(te[s1]['success_ref'], 400)}\\% to {pct(te[s1]['success_arm'], 400)}\\%)", 'test seed 1 %'),
    (main, f"from {te[s2]['success_ref']} to {te[s2]['success_arm']} ({pct(te[s2]['success_ref'], 400)}\\% to {pct(te[s2]['success_arm'], 400)}\\%)", 'test seed 2 %'),
    (main, f"$[+{te[s1]['success_diff_boot95_POSTHOC'][0] * 100:.1f}, +{te[s1]['success_diff_boot95_POSTHOC'][1] * 100:.1f}]$ and $[+{te[s2]['success_diff_boot95_POSTHOC'][0] * 100:.1f}, +{te[s2]['success_diff_boot95_POSTHOC'][1] * 100:.1f}]$", 'bootstrap'),
    (main, f"Harm on the test scenes was {te[s1]['harm']} of {te[s1]['ref_successes']} and {te[s2]['harm']} of {te[s2]['ref_successes']}", 'test harm'),
    (main, f"{te[s1]['ref_successes'] - te[s1]['harm'] + te[s2]['ref_successes'] - te[s2]['harm']} both-succeed, {te[s1]['rescue'] + te[s2]['rescue']} rescues, {te[s1]['harm'] + te[s2]['harm']} harm and {te[s1]['ref_failures'] - te[s1]['rescue'] + te[s2]['ref_failures'] - te[s2]['rescue']} both-fail", 'pooled matrix'),
    (main, f"{te[s1]['harm'] + te[s2]['harm']} of {te[s1]['ref_successes'] + te[s2]['ref_successes']} baseline successes was lost while {te[s1]['rescue'] + te[s2]['rescue']} of {te[s1]['ref_failures'] + te[s2]['ref_failures']} failures", 'discussion pooled'),
    (main, f"V2 raised success from {b3['C0']} to {b3['VISION_CENTER_V2']['success_arm']} of 200 scenes, with {b3['VISION_CENTER_V2']['rescue']} rescues and {b3['VISION_CENTER_V2']['harm']} harms ($p={sci(b3['VISION_CENTER_V2']['p_exact_vs_C0'])}$", 'B3'),
    (main, f"It recovered {b3['oracle_rescues_recovered_by_v2']} of the {b3['oracle_rescues']} oracle rescues; the oracle itself caused {b3['CENTER_ORACLE']['harm']} harms", 'B3 oracle'),
    (main, f"The single-frame variant reached {b3['V2_SINGLE_FRAME']['success_arm']} ({b3['V2_SINGLE_FRAME']['rescue']} rescues, {b3['V2_SINGLE_FRAME']['harm']} harms)", 'B3 single'),
    (main, f"The oracle rescued {c1[s1]['CENTER_ORACLE']['rescue']} and {c1[s2]['CENTER_ORACLE']['rescue']} development scenes; V2 rescued {c1[s1]['VISION_CENTER_V2']['rescue']} and {c1[s2]['VISION_CENTER_V2']['rescue']}", 'oracle vs V2 rescues'),
    (main, f"The privileged oracle reached {c1[s1]['CENTER_ORACLE']['success_arm']} and {c1[s2]['CENTER_ORACLE']['success_arm']}", 'oracle success'),
    (main, f"the controller acts in {c1[s1]['V2_SINGLE_FRAME']['acted']}--{c1[s2]['V2_SINGLE_FRAME']['acted']} scenes and rescues {c1[s1]['V2_SINGLE_FRAME']['rescue']} and {c1[s2]['V2_SINGLE_FRAME']['rescue']}", 'single frame acted/rescue'),
    (main, f"it acts in {c1[s2]['VISION_CENTER_V2']['acted']}--{c1[s1]['VISION_CENTER_V2']['acted']} scenes and rescues {c1[s1]['VISION_CENTER_V2']['rescue']} and {c1[s2]['VISION_CENTER_V2']['rescue']}", 'V2 acted/rescue'),
    (main, f"The rules-off ablation acted in {c1[s1]['V2_RULES_OFF']['acted']} and {c1[s2]['V2_RULES_OFF']['acted']} development scenes instead of {c1[s1]['VISION_CENTER_V2']['acted']} and {c1[s2]['VISION_CENTER_V2']['acted']}", 'rules off'),
    (main, f"({c1[s1]['VISION_CENTER_V2']['acted']} of {c1[s1]['triggered']} and {c1[s2]['VISION_CENTER_V2']['acted']} of {c1[s2]['triggered']} in development)", 'acted of triggered'),
    (main, f"The trigger fired in {c1[s1]['triggered']} (seed 1) and {c1[s2]['triggered']} (seed 2) scenes", 'C1 triggered'),
    (supp, f"$\\max(0.0059,0.0078)=0.0078$", 'IU (literal)'),
]
fails = []; out = []
texts = {f: (PAPER / f).read_text(encoding='utf-8') for f in {x[0] for x in C}}
m1, m2 = c1[s1]['V2_vs_SINGLE_FRAME']['p_B'], c1[s2]['V2_vs_SINGLE_FRAME']['p_B']
if f'{m1:.4f}' != '0.0059' or f'{m2:.4f}' != '0.0078': fails.append('IU literal values differ from recount')
for f, lit, what in C:
    ok = lit in texts[f]; out.append({'file': f, 'claim': what, 'expected_text': lit, 'found': ok})
    if not ok: fails.append(f'{what}: not found in {f}: {lit}')
(PAPER / 'build_logs').mkdir(exist_ok=True)
(PAPER / 'build_logs/check_numbers.json').write_text(json.dumps({'checks': len(C) + 1, 'failures': fails, 'items': out}, indent=1), encoding='utf-8')
print(f'{len(C) + 1} checks, {len(fails)} failures'); [print(' -', x) for x in fails]
sys.exit(1 if fails else 0)
