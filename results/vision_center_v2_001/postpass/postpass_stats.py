"""Vision-CENTER v2, Phase C0 (post-pass): paired statistics recomputed from the stored scene records.

Reads results/vision_center_v2_001/execution/{pilot,shadow}/scenes/*.json only. Nothing is simulated, no record is changed.
Every number here is EXPLORATORY: stage B2 is development data (the gate thresholds were fixed before the run, the tests
below were not), stage B1 is the data v2 was designed on. Run from the repository root:

    python results/vision_center_v2_001/postpass/postpass_stats.py
"""
import json, math, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import beta, binom, norm
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[3]; EXE = ROOT / 'results/vision_center_v2_001/execution'; OUT = Path(__file__).resolve().parent / 'postpass_stats.json'
ARMS = ('C0', 'VISION_CENTER_V1', 'VISION_CENTER_V2', 'CENTER_ORACLE')
EXPECTED_B2 = {'C0': 165, 'VISION_CENTER_V1': 173, 'VISION_CENTER_V2': 184, 'CENTER_ORACLE': 190}


def load(stage, lo, hi):
    recs = [json.loads((EXE / stage / 'scenes' / f'{s}.json').read_text(encoding='utf-8')) for s in range(lo, hi + 1)]
    assert [r['scene'] for r in recs] == list(range(lo, hi + 1)) and all(r['completed'] for r in recs)
    return recs


def cell(r, a): return next((x for x in r['cells'] if x['arm'] == a), None)


def wilson(k, n, conf=0.95):
    z = norm.ppf(1 - (1 - conf) / 2); p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0., c - h), min(1., c + h)]


def cp(k, n, conf=0.95):
    """Clopper-Pearson two-sided interval."""
    a = (1 - conf) / 2
    return [0. if k == 0 else float(beta.ppf(a, k, n - k + 1)), 1. if k == n else float(beta.ppf(1 - a, k + 1, n - k))]


def cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper limit."""
    return 1. if k == n else float(beta.ppf(conf, k + 1, n - k))


def tango(b, c, n, conf=0.95):
    """Tango (1998) score interval for the paired difference of proportions (b - c) / n; b, c = discordant counts."""
    z = norm.ppf(1 - (1 - conf) / 2)

    def stat(d):
        # constrained MLE of p21 (the 'c' cell probability) under difference d
        A = 2 * n; B = -b - c + (2 * n - b + c) * d; C = -c * d * (1 - d); q = (math.sqrt(max(B * B - 4 * A * C, 0.)) - B) / (2 * A)
        v = n * (2 * q + d * (1 - d))
        return (b - c - n * d) / math.sqrt(v) if v > 0 else (math.inf if b - c - n * d > 0 else -math.inf if b - c - n * d < 0 else 0.)
    est = (b - c) / n; eps = 1e-10
    lo = -1. if stat(-1 + eps) < z else brentq(lambda d: stat(d) - z, -1 + eps, est) if stat(est) < z else est
    hi = 1. if stat(1 - eps) > -z else brentq(lambda d: stat(d) + z, est, 1 - eps)
    return [float(lo), float(hi)]


def newcombe(b, c, a, d, conf=0.95):
    """Newcombe (1998) method 10 for the paired difference; a = both succeed, d = both fail."""
    n = a + b + c + d; p1 = (a + b) / n; p2 = (a + c) / n; l1, u1 = wilson(a + b, n, conf); l2, u2 = wilson(a + c, n, conf); diff = (b - c) / n
    A = (a + b) * (c + d) * (a + c) * (b + d); phi = 0. if A == 0 else (a * d - b * c) / math.sqrt(A)
    if phi > 0 and a * d - b * c > 0: phi = max(a * d - b * c - n / 2, 0) / math.sqrt(A)
    lo = diff - math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2); hi = diff + math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    return [float(lo), float(hi)]


def mcnemar_exact(b, c):
    """Two-sided exact conditional (binomial) test on the discordant scenes; also the one-sided p for b > c."""
    m = b + c
    if m == 0: return {'discordant': 0, 'p_two_sided': 1., 'p_one_sided_arm_better': 1.}
    one = float(binom.sf(b - 1, m, 0.5)); two = float(min(1., 2 * min(binom.cdf(min(b, c), m, 0.5), binom.sf(max(b, c) - 1, m, 0.5))))
    return {'discordant': m, 'p_two_sided': two, 'p_one_sided_arm_better': one}


def bootstrap(y0, y1, seed=20261118, draws=20000):
    """Percentile bootstrap over scenes for the difference; with zero harm scenes it can never draw a harm, so its lower limit is optimistic."""
    rng = np.random.default_rng(seed); d = np.asarray(y1, float) - np.asarray(y0, float); idx = rng.integers(0, len(d), (draws, len(d)))
    return [float(x) for x in np.quantile(d[idx].mean(1), [0.025, 0.975])]


def paired(recs, ref, arm):
    y0 = np.array([cell(r, ref)['y'] for r in recs]); y1 = np.array([cell(r, arm)['y'] for r in recs]); n = len(recs)
    a = int(((y0 == 1) & (y1 == 1)).sum()); b = int(((y0 == 0) & (y1 == 1)).sum()); c = int(((y0 == 1) & (y1 == 0)).sum()); d = int(((y0 == 0) & (y1 == 0)).sum())
    n_pos = a + c; n_neg = b + d
    # joint statement on rescue and harm rates at 95 % (Bonferroni: each at 97.5 %), giving a conservative interval for the net gain per scene
    r_lo = cp(b, n, 0.975)[0]; h_hi = cp(c, n, 0.975)[1]
    return {'reference': ref, 'arm': arm, 'n': n, 'table': {'both_succeed': a, 'rescue': b, 'harm': c, 'both_fail': d}, 'reference_success': a + c, 'arm_success': a + b,
            'difference_pp': 100 * (b - c) / n,
            'difference_ci95_pp': {'tango_score': [100 * x for x in tango(b, c, n)], 'newcombe_method10': [100 * x for x in newcombe(b, c, a, d)], 'bootstrap_percentile': [100 * x for x in bootstrap(y0, y1)]},
            'exact_paired_test': mcnemar_exact(b, c),
            'rescue': {'count': b, 'given_reference_failure': b / n_neg if n_neg else None, 'denominator': n_neg, 'clopper_pearson95': cp(b, n_neg) if n_neg else None, 'per_scene': b / n, 'per_scene_clopper_pearson95': cp(b, n)},
            'harm': {'count': c, 'given_reference_success': c / n_pos if n_pos else None, 'denominator': n_pos, 'upper95_one_sided': cp_upper(c, n_pos) if n_pos else None, 'upper97_5_one_sided': cp_upper(c, n_pos, 0.975) if n_pos else None,
                     'rule_of_three': 3 / n_pos if (n_pos and c == 0) else None, 'per_scene_upper95_one_sided': cp_upper(c, n),
                     'binomial_p_against_epsilon_0_05': float(binom.cdf(c, n_pos, 0.05)) if n_pos else None},
            'net': {'count': b - c, 'per_scene_pp': 100 * (b - c) / n, 'ci95_pp_tango': [100 * x for x in tango(b, c, n)],
                    'conservative_bonferroni_lower_pp': 100 * (r_lo - h_hi), 'conservative_note': 'rescue lower limit minus harm upper limit, each Clopper-Pearson at 97.5 % two-sided'},
            'rescue_scenes': [r['scene'] for r, u, v in zip(recs, y0, y1) if u == 0 and v == 1], 'harm_scenes': [r['scene'] for r, u, v in zip(recs, y0, y1) if u == 1 and v == 0]}


def acted_on_success(recs, arm):
    """Harm can only arise where the arm acts. Scenes the policy solved alone and in which the arm moved the robot."""
    acted = [r for r in recs if r['triggered'] and cell(r, arm)['acted']]; pos = [r for r in acted if cell(r, 'C0')['y'] == 1]; lost = sum(1 for r in pos if cell(r, arm)['y'] == 0)
    return {'acts': len(acted), 'acts_on_reference_success': len(pos), 'harm_among_them': lost, 'upper95_one_sided': cp_upper(lost, len(pos)) if pos else None,
            'acts_on_reference_failure': len(acted) - len(pos), 'rescues': sum(1 for r in acted if cell(r, 'C0')['y'] == 0 and cell(r, arm)['y'] == 1)}


def stage(recs, arms):
    n = len(recs); succ = {a: {'k': int(sum(cell(r, a)['y'] for r in recs)), 'n': n} for a in arms}
    for a in arms: succ[a]['wilson95'] = wilson(succ[a]['k'], n); succ[a]['clopper_pearson95'] = cp(succ[a]['k'], n)
    out = {'scenes': n, 'triggered': sum(1 for r in recs if r['triggered']), 'success': succ, 'paired': {}, 'wilson_overlap': {}}
    pairs = [('C0', a) for a in arms[1:]] + ([('VISION_CENTER_V1', 'VISION_CENTER_V2')] if 'VISION_CENTER_V2' in arms else []) + ([('VISION_CENTER_V2', 'CENTER_ORACLE')] if 'CENTER_ORACLE' in arms else [])
    for ref, arm in pairs:
        out['paired'][f'{arm}_vs_{ref}'] = paired(recs, ref, arm); w0 = succ[ref]['wilson95']; w1 = succ[arm]['wilson95']
        out['wilson_overlap'][f'{arm}_vs_{ref}'] = {'reference': w0, 'arm': w1, 'overlap': bool(w1[0] <= w0[1] and w0[0] <= w1[1]), 'gap_pp': 100 * (w1[0] - w0[1])}
    out['acted_on_success'] = {a: acted_on_success(recs, a) for a in arms if a in ('VISION_CENTER_V1', 'VISION_CENTER_V2', 'CENTER_ORACLE')}
    return out


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [False] * len(ps)
    for j, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - j): out[i] = True
        else: break
    return out


def planning():
    """Sample sizes for a later confirmatory run. Planning arithmetic only; no data beyond the observed rates enters."""
    zero = lambda eps, delta, m: int(math.ceil(math.log(delta / m) / math.log1p(-eps)))

    def kmax(n_pos, eps, alpha):
        k = -1
        while binom.cdf(k + 1, n_pos, eps) <= alpha: k += 1
        return k

    def pass_prob(n_pos, eps, alpha, true_h):
        k = kmax(n_pos, eps, alpha); return 0. if k < 0 else float(binom.cdf(k, n_pos, true_h))
    rows = []
    for n_ep in (200, 400):
        for base in (0.80, 0.825, 0.85):
            n_pos = int(round(n_ep * base))
            for m in (1, 2, 4):
                a = 0.05 / m; rows.append({'calibration_scenes': n_ep, 'assumed_base_success': base, 'n_plus': n_pos, 'candidates_holm_worst_case': m, 'alpha_per_test': a, 'max_harm_count_that_passes': kmax(n_pos, 0.05, a),
                                           'pass_probability_if_true_harm': {str(h): pass_prob(n_pos, 0.05, a, h) for h in (0.0, 0.01, 0.02, 0.03)}})
    # exact one-sided paired test, v2 against C0: probability that b rescues out of m discordant reach p <= alpha when the true discordant split is (rescue, harm)
    def power(n, p_r, p_h, alpha=0.025, draws=200000, seed=7):
        rng = np.random.default_rng(seed); x = rng.multinomial(n, [p_r, p_h, 1 - p_r - p_h], size=draws); b = x[:, 0]; c = x[:, 1]
        return float((binom.sf(b - 1, b + c, 0.5) <= alpha).mean())
    pw = [{'scenes': n, 'true_rescue_per_scene': pr, 'true_harm_per_scene': ph, 'power_one_sided_0_025': power(n, pr, ph)} for n in (200, 400) for pr, ph in ((0.095, 0.0), (0.05, 0.005), (0.05, 0.01), (0.03, 0.01))]
    return {'zero_harm_n_plus_needed_epsilon_0_05_delta_0_05': {str(m): zero(0.05, 0.05, m) for m in (1, 2, 3, 4, 6)}, 'harm_certificate': rows, 'paired_superiority_power_monte_carlo': pw,
            'note': 'm = number of non-Continue candidates (times policies, if a joint statement over policies is wanted) in the Holm family; worst case uses alpha/m for every test'}


def main():
    b2 = load('pilot', 50900, 51099); b1 = load('shadow', 50600, 50899)
    s2 = stage(b2, ARMS); s1 = stage(b1, ARMS[:3])
    got = {a: s2['success'][a]['k'] for a in ARMS}; stored = json.loads((EXE / 'pilot/result.json').read_text(encoding='utf-8'))
    checks = {'b2_success_counts_equal_expected': got == EXPECTED_B2, 'b2_counts': got,
              'equal_to_stored_result_json': all(stored['success'][a]['k'] == got[a] for a in ARMS) and stored['primary_V2_vs_C0']['rescue_scenes'] == s2['paired']['VISION_CENTER_V2_vs_C0']['rescue_scenes']
              and stored['primary_V2_vs_C0']['harm'] == s2['paired']['VISION_CENTER_V2_vs_C0']['harm']['count'],
              'record_files': {'pilot': len(b2), 'shadow': len(b1)},
              'records_digest_sha256': {st: hashlib.sha256(''.join(hashlib.sha256((EXE / st / 'scenes' / f'{r["scene"]}.json').read_bytes()).hexdigest() for r in recs).encode()).hexdigest() for st, recs in (('pilot', b2), ('shadow', b1))}}
    fam = ['VISION_CENTER_V2_vs_C0', 'VISION_CENTER_V2_vs_VISION_CENTER_V1', 'VISION_CENTER_V1_vs_C0']; ps = [s2['paired'][k]['exact_paired_test']['p_two_sided'] for k in fam]
    out = {'work_id': 'vision_center_v2_001', 'phase': 'C0 post-pass', 'status': 'EXPLORATORY. Stage B2 is development data: the scenes were new, the gate was fixed in advance, but these tests, intervals and their family were chosen after the result. '
           'Stage B1 is design data. Neither is calibration or Test.',
           'unit': 'initial scene (one seed = one scene = one paired observation across arms)', 'checks': checks, 'stage_B2_new_scenes_50900_51099': s2, 'stage_B1_used_scenes_50600_50899_design_data': s1,
           'multiplicity_exploratory': {'family': fam, 'p_two_sided': ps, 'holm_reject_at_0_05': holm(ps), 'note': 'descriptive; the family was not declared before the run'},
           'planning_for_a_confirmatory_run': planning()}
    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8', newline='\n'); print(json.dumps({'checks': checks}, indent=1))
    for k, v in s2['paired'].items():
        print(k, v['table'], 'diff %.1f pp' % v['difference_pp'], 'tango', [round(x, 1) for x in v['difference_ci95_pp']['tango_score']], 'newcombe', [round(x, 1) for x in v['difference_ci95_pp']['newcombe_method10']],
              'boot', [round(x, 1) for x in v['difference_ci95_pp']['bootstrap_percentile']], 'p2 %.2e' % v['exact_paired_test']['p_two_sided'], 'harm upper95 %.4f' % v['harm']['upper95_one_sided'], 'cons net lower %.1f' % v['net']['conservative_bonferroni_lower_pp'])
    for k, v in s2['wilson_overlap'].items(): print('wilson', k, [round(x, 4) for x in v['reference']], [round(x, 4) for x in v['arm']], v['overlap'])
    print('acted', json.dumps(s2['acted_on_success'])); print('B1 acted', json.dumps(s1['acted_on_success']))
    for k, v in s1['paired'].items(): print('B1', k, v['table'], [round(x, 1) for x in v['difference_ci95_pp']['tango_score']], 'p2 %.2e' % v['exact_paired_test']['p_two_sided'])
    print(json.dumps(out['planning_for_a_confirmatory_run']['zero_harm_n_plus_needed_epsilon_0_05_delta_0_05']))
    for r in out['planning_for_a_confirmatory_run']['harm_certificate']:
        if r['assumed_base_success'] == 0.825: print(r['calibration_scenes'], r['n_plus'], r['candidates_holm_worst_case'], r['max_harm_count_that_passes'], {k: round(v, 3) for k, v in r['pass_probability_if_true_harm'].items()})
    for r in out['planning_for_a_confirmatory_run']['paired_superiority_power_monte_carlo']: print(r)


if __name__ == '__main__': main()
