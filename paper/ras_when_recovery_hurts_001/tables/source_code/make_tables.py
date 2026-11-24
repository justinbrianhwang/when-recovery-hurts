"""LaTeX tables of the manuscript, generated from data/recount.json and the (read-only) result files.

usage: python -P paper/ras_when_recovery_hurts_001/tables/source_code/make_tables.py
Writes tables/T*.tex and tables/TABLE_PROVENANCE.json (input and output SHA256).
"""
from __future__ import annotations

import csv, hashlib, json
from pathlib import Path

from scipy.stats import beta

REPO = Path(__file__).resolve().parents[4]
PAPER = REPO / 'paper/ras_when_recovery_hurts_001'; OUT = PAPER / 'tables'
USED = {}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def use(p):
    p = Path(p); USED[str(p.relative_to(REPO)).replace('\\', '/')] = sha(p); return p


def jl(p): return json.loads(use(p).read_text(encoding='utf-8'))


def cp(k, n):  # two-sided 95% Clopper-Pearson, descriptive
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k); return lo, hi


def pfmt(p):
    if p >= 0.001: return f'{p:.4f}'.rstrip('0') if p < 1 else '1'
    m, e = f'{p:.1e}'.split('e'); return f'${m}\\times10^{{{int(e)}}}$'


R = jl(PAPER / 'data/recount.json'); POL = ('train_seed_1', 'train_seed_2')
ARMS = [('VISION_CENTER_V1', 'V1 (wrist camera only)'), ('V2_SINGLE_FRAME', 'V2, single frame'), ('V2_NO_HAND_MEMORY', 'V2 without hand memory'),
        ('V2_NO_BASE_MEMORY', 'V2 without base memory'), ('V2_NO_HOLD_MEMORY', 'V2 without hold memory'), ('V2_RULES_OFF', 'V2, abstention rules off'),
        ('VISION_CENTER_V2', '\\textbf{V2 (proposed)}'), ('CENTER_ORACLE', 'Oracle centring (privileged)')]


def t_dev():
    c = R['C1_development']; L = ['\\begin{tabular}{lrrrrrrrr}', '\\toprule',
         ' & \\multicolumn{4}{c}{Policy seed 1 (C0: %d/200)} & \\multicolumn{4}{c}{Policy seed 2 (C0: %d/200)}\\\\' % (c['train_seed_1']['C0']['success'], c['train_seed_2']['C0']['success']),
         '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}', 'Arm & Succ. & Resc. & Harm & Acted & Succ. & Resc. & Harm & Acted\\\\', '\\midrule']
    for a, lab in ARMS:
        row = [lab]
        for p in POL:
            e = c[p][a]; row += [str(e['success_arm']), str(e['rescue']), str(e['harm']), str(e['acted'])]
        L.append(' & '.join(row) + '\\\\')
    L += ['\\bottomrule', '\\end{tabular}']; return '\n'.join(L)


def t_tests():
    c = R['C1_development']
    L = ['\\begin{tabular}{llrrl}', '\\toprule', 'Hypothesis & Comparison & Discordant & $p$ (one-sided) & Holm, $\\alpha=0.05$\\\\', '\\midrule']
    v = c['train_seed_2']['VISION_CENTER_V2']
    L.append(f"$H_{{\\mathrm{{gen}}}}$ & V2 vs C0, policy seed 2 & {v['rescue']} vs {v['harm']} & {pfmt(c['H_gen']['p'])} & {'rejected' if c['H_gen']['rejected'] else 'not rejected'}\\\\")
    m1, m2 = c['train_seed_1']['V2_vs_SINGLE_FRAME'], c['train_seed_2']['V2_vs_SINGLE_FRAME']
    L.append(f"$H_{{\\mathrm{{mem}}}}$ & V2 vs single frame, seed 1 & {m1['v2_only']} vs {m1['single_only']} & {pfmt(m1['p_B'])} & \\multirow{{2}}{{*}}{{{'rejected' if c['H_mem']['rejected'] else 'not rejected'} (IU $p$ = {pfmt(c['H_mem']['p_intersection_union'])})}}\\\\")
    L.append(f" & V2 vs single frame, seed 2 & {m2['v2_only']} vs {m2['single_only']} & {pfmt(m2['p_B'])} & \\\\")
    L += ['\\bottomrule', '\\end{tabular}']; return '\n'.join(L)


def t_confirm():
    L = ['\\begin{tabular}{llrrrrrrl}', '\\toprule', 'Stage (seeds) & Policy & Trig. & Acted & C0 & V2 & Resc. & Harm & Pre-registered result\\\\', '\\midrule']
    for stage, lab in (('calibration', 'Calibration (30000--30399)'), ('test', 'Final Test (90000--90399)')):
        for i, p in enumerate(POL):
            e = R[stage][p]
            if stage == 'calibration':
                res = f"harm {e['harm']}/{e['ref_successes']}, $p$ = {pfmt(e['harm_p_B'])}: {'certified' if e['harm_certified'] else 'not certified'}"
            else:
                res = f"{e['rescue']} vs {e['harm']}, $p$ = {pfmt(e['eff_p_B'])}: {'$H_0$ rejected' if e['efficacy_rejected'] else 'not rejected'}"
            L.append(f"{lab if i == 0 else ''} & seed {i + 1} & {e['triggered']} & {e['acted']} & {e['success_ref']} & {e['success_arm']} & {e['rescue']} & {e['harm']} & {res}\\\\")
        if stage == 'calibration': L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}']; return '\n'.join(L)


def t_b3():
    b = R['B3_development']; L = ['\\begin{tabular}{lrrrrr}', '\\toprule', 'Arm (PlainConv policy) & Success/200 & 95\\% CI & Acted & Rescue & Harm\\\\', '\\midrule']
    lo, hi = cp(b['C0'], 200); L.append(f"C0 & {b['C0']} & {lo * 100:.0f}--{hi * 100:.0f}\\% & -- & -- & --\\\\")
    for a, lab in (('V2_SINGLE_FRAME', 'V2, single frame'), ('VISION_CENTER_V2', '\\textbf{V2 (unchanged)}'), ('CENTER_ORACLE', 'Oracle centring (privileged)')):
        e = b[a]; lo, hi = cp(e['success_arm'], 200)
        L.append(f"{lab} & {e['success_arm']} & {lo * 100:.0f}--{hi * 100:.0f}\\% & {e['acted']} & {e['rescue']} & {e['harm']}\\\\")
    L += ['\\bottomrule', '\\end{tabular}']; return '\n'.join(L)


def t_physics():
    D = REPO / 'results/vision_center_v2_c1_001/posthoc_physics_diagnostics_001'; H = REPO / 'results/vision_center_v2_c1_001/posthoc_hull_sensitivity_001'
    allrow = next(x for x in csv.DictReader(open(use(D / 'S6_ERRORS_BY_GROUP.csv'), encoding='utf-8')) if x['grouping'] == 'all')
    hs = jl(H / 'S45_HULL_SENSITIVITY_SUMMARY.json'); b4 = jl(REPO / 'results/ras_exec_001/b4_s6_substep/S6_SUBSTEP_SUMMARY.json')['groups']['last']['ALL | ALL']
    st = list(csv.DictReader(open(use(REPO / 'results/ras_submission_readiness_001/s6_static_validation/cases.csv'), encoding='utf-8')))
    stl = [x for x in st if x['min_sep_minus_gap_before_mm'] not in ('', 'NA')]; mx = max(abs(float(x['min_sep_minus_gap_before_mm'])) for x in stl)
    s4 = hs['status']['S4_logged_to_hull']; s5 = hs['status']['S5_logged_to_hull']
    rows = [('S4: robot--target clearance during the move', f"{s4.get('PASS -> PASS', 0) + s4.get('FAIL -> FAIL', 0)} acted branches; FAIL {s4.get('FAIL -> FAIL', 0)} (oracle arm, seed 2, scene 51152, $-2.16$\\,mm)", 'FAIL in 1 branch, none in V2'),
            ('S5: reconstructed penetration $\\le 1$\\,mm', f"{s5.get('FAIL -> FAIL', 0)} of {hs['status']['branches']} branches FAIL (2 fast-cube events)", 'FAIL in 2 events'),
            ('Hull sensitivity of S4/S5', 'logged meshes replaced by convex hulls: 0 status changes', 'unchanged'),
            ('S6: logged PhysX separation vs reconstructed gap', f"{int(allrow['over_tol']):,} of {int(allrow['n']):,} rows over 1\\,mm", '\\textbf{official FAIL}'),
            ('S6 static, known geometry', f"{len(stl)} cases with listed points (of {len(st)}): $|\\Delta| \\le {mx:.2f}$\\,mm", 'separation definition confirmed'),
            ('S6 sub-step logging (B4)', f"pre-sub-step pose: {b4['over_1mm_post']:,} $\\to$ {b4['over_1mm_pre']:,} of {b4['n']:,} pairs over 1\\,mm", 'residual unexplained')]
    L = ['\\begin{tabular}{p{0.28\\linewidth}p{0.40\\linewidth}p{0.18\\linewidth}}', '\\toprule', 'Check & Result & Status\\\\', '\\midrule']
    L += [f'{a} & {b} & {c}\\\\' for a, b, c in rows]; L += ['\\bottomrule', '\\end{tabular}']; return '\n'.join(L)


def t_decisions():
    d = jl(PAPER / 'figures/FIGURE_PROVENANCE_SUPP.json')['outputs']['S2_decisions_by_stage.pdf']['info']
    keys = ['ACT', 'ALIGNED', 'ALIGNED_WITHIN_UNCERTAINTY', 'LOW_CARRY', 'NO_HOLD_ESTIMATE', 'NO_TARGET_ESTIMATE', 'SIDE_CONTACT_RISK', 'MEMORY_CONTRADICTED', 'TOO_FEW_STEPS', 'no trigger']
    cols = list(d); BS = chr(92); NL = BS + BS
    L = [BS + 'begin{tabular}{l' + 'r' * len(cols) + '}', BS + 'toprule',
         'Decision & ' + ' & '.join(c.split(' | ')[0] + (' s1' if c.endswith('1') else ' s2') for c in cols) + NL, BS + 'midrule']
    for k in keys:
        L.append(k.replace('_', ' ').lower() + ' & ' + ' & '.join(str(d[c].get(k, 0)) for c in cols) + NL)
    L.append(BS + 'midrule'); L.append('total & ' + ' & '.join(str(sum(d[c].values())) for c in cols) + NL)
    L += [BS + 'bottomrule', BS + 'end{tabular}']; return chr(10).join(L)


def t_descriptive():
    BS = chr(92); NL = BS + BS
    L = [BS + 'begin{tabular}{llrrrrl}', BS + 'toprule', 'Stage & Policy & $n_+$ & Harm & CP 95' + BS + '% upper & Moved among $Y_0{=}1$ & Bootstrap 95' + BS + '% CI of success difference' + NL, BS + 'midrule']
    for stage, lab in (('calibration', 'Calibration'), ('test', 'Final Test')):
        for i, p in enumerate(POL):
            e = R[stage][p]; b = e['success_diff_boot95_POSTHOC']
            L.append(f"{lab if i == 0 else ''} & seed {i + 1} & {e['ref_successes']} & {e['harm']} & {e['harm_cp95_upper_POSTHOC'] * 100:.2f}" + BS + f"% & {e['moved_among_baseline_success']} & [{b[0] * 100:+.2f}, {b[1] * 100:+.2f}] pp" + NL)
    L += [BS + 'bottomrule', BS + 'end{tabular}']; return chr(10).join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True); outs = {}
    for name, fn in (('T2_development_arms', t_dev), ('T3_development_tests', t_tests), ('T4_calibration_and_test', t_confirm), ('T5_physics_validity', t_physics), ('T6_generalization_plainconv', t_b3), ('TS1_decisions_by_stage', t_decisions), ('TS2_descriptive_posthoc', t_descriptive)):
        p = OUT / f'{name}.tex'; p.write_text(fn() + '\n', encoding='utf-8'); outs[p.name] = sha(p); print(p.name)
    s = Path(__file__); (OUT / 'TABLE_PROVENANCE.json').write_text(json.dumps({'generator': str(s.relative_to(REPO)).replace('\\', '/'), 'generator_sha256': sha(s), 'inputs_sha256': USED, 'outputs_sha256': outs}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
