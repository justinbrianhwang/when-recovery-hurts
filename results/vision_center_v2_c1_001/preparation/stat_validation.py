"""Vision-CENTER v2 Phase C1: check of the two primary tests BEFORE any C1 rollout. Simulation only; no experimental data.

Tests under check (hci.vision_center_c1.primary_tests):
  H_gen  train seed 2, v2 against C0: exact conditional paired test on the discordant scenes, one-sided
  H_mem  v2 against V2_SINGLE_FRAME with both policies: one such test per policy, combined by intersection-union (larger p)
  Holm over the two at familywise alpha 0.05.
The draft of Phase C0 pooled the discordant pairs of both policies for H_mem; that test is simulated next to it to show
what the change does. The 200 scenes are shared by the two policies: outcomes of the same scene are made dependent through a
scene-level difficulty, and no test here treats the 400 runs as independent.

    python results/vision_center_v2_c1_001/preparation/stat_validation.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from hci import vision_center_c1 as C1

OUT = Path(__file__).resolve().parent / 'stat_validation.json'; N = 200; REPS = 40000; ALPHA = C1.ALPHA
PT = np.array([[C1.exact_one_sided(b, c) for c in range(N + 1)] for b in range(N + 1)])      # p-value table


def discordant(rng, rates, rho):
    """rates[k] = (P(only the arm succeeds), P(only the reference succeeds)) for comparison k; the same scenes carry all of
    them. A scene is 'hard' with probability 1/2; discordance is concentrated there by `rho` (0: none, 1: all)."""
    hard = rng.random((REPS, N)) < 0.5; out = []
    for pb, pc in rates:
        f = np.where(hard, 1 + rho, 1 - rho); u = rng.random((REPS, N)); b = (u < pb * f).sum(1); c = ((u >= pb * f) & (u < (pb + pc) * f)).sum(1); out.append((b, c))
    return out


def run(rng, name, gen, mem1, mem2, rho, true_nulls):
    (bg, cg), (b1, c1), (b2, c2) = discordant(rng, [gen, mem1, mem2], rho); p_gen = PT[bg, cg]; p1 = PT[b1, c1]; p2 = PT[b2, c2]; p_iut = np.maximum(p1, p2); p_pool = PT[b1 + b2, c1 + c2]
    lo = np.minimum(p_gen, p_iut); hi = np.maximum(p_gen, p_iut); first = lo <= ALPHA / 2; second = first & (hi <= ALPHA); rej_gen = np.where(p_gen <= p_iut, first, second); rej_mem = np.where(p_iut < p_gen, first, second)
    lo2 = np.minimum(p_gen, p_pool); hi2 = np.maximum(p_gen, p_pool); f2 = lo2 <= ALPHA / 2; s2 = f2 & (hi2 <= ALPHA); rej_pool = np.where(p_pool < p_gen, f2, s2)
    false = np.zeros(REPS, bool)
    if 'H_gen' in true_nulls: false |= rej_gen
    if 'H_mem' in true_nulls: false |= rej_mem
    se = lambda x: float(np.sqrt(max(x * (1 - x), 1e-12) / REPS))
    r = {'scenario': name, 'rates_only_arm_only_reference': {'H_gen_seed2_v2_vs_C0': gen, 'H_mem_seed1_v2_vs_single_frame': mem1, 'H_mem_seed2_v2_vs_single_frame': mem2}, 'scene_dependence_rho': rho, 'true_nulls': sorted(true_nulls),
         'reject_H_gen': float(rej_gen.mean()), 'reject_H_mem_intersection_union': float(rej_mem.mean()), 'reject_H_mem_pooled_draft': float(rej_pool.mean()),
         'familywise_error': float(false.mean()) if true_nulls else None, 'familywise_error_mc_se': se(float(false.mean())) if true_nulls else None,
         'pooled_draft_rejects_although_H_mem_is_false': float(rej_pool.mean()) if 'H_mem' in true_nulls else None}
    return r


def main():
    if OUT.exists(): raise SystemExit(f'Refused overwrite: {OUT}')
    rng = np.random.default_rng(20261118); z = (0.02, 0.02); s = (0.095, 0.0); m = (0.055, 0.0); out = []
    for rho in (0.0, 0.8):
        out.append(run(rng, 'global null: no arm better anywhere', z, z, z, rho, {'H_gen', 'H_mem'}))
        out.append(run(rng, 'global null, least favourable for the exact test (only-arm = only-reference = 5 %)', (0.05, 0.05), (0.05, 0.05), (0.05, 0.05), rho, {'H_gen', 'H_mem'}))
        out.append(run(rng, 'memory helps with train seed 1 only (H_mem false), v2 helps with seed 2 (H_gen true)', s, m, z, rho, {'H_mem'}))
        out.append(run(rng, 'memory helps strongly with train seed 1 only, least favourable second policy (5 % / 5 %)', s, (0.10, 0.0), (0.05, 0.05), rho, {'H_mem'}))
        out.append(run(rng, 'memory helps with both, v2 not better than C0 with seed 2 (H_gen false)', (0.03, 0.03), m, m, rho, {'H_gen'}))
        out.append(run(rng, 'alternative at the stage-B2 rates (9.5 % against C0, 5.5 % against single frame, no loss)', s, m, m, rho, set()))
        out.append(run(rng, 'alternative at half the stage-B2 effect, with losses of 0.5 %', (0.0475, 0.005), (0.0275, 0.005), (0.0275, 0.005), rho, set()))
        out.append(run(rng, 'alternative: stage-B2 rates with seed 1, half effect with seed 2', (0.0475, 0.005), m, (0.0275, 0.005), rho, set()))
    worst = max(r['familywise_error'] for r in out if r['familywise_error'] is not None)
    res = {'work_id': C1.WORK, 'what': 'simulation check of the primary tests; no experimental data', 'scenes': N, 'replications': REPS, 'alpha_familywise': ALPHA, 'largest_familywise_error': worst,
           'bound_holds': bool(worst <= ALPHA + 3 * 0.0011), 'exact_test_examples': {'11_against_0': C1.exact_one_sided(11, 0), '19_against_0': C1.exact_one_sided(19, 0), '5_against_0': C1.exact_one_sided(5, 0), '6_against_0': C1.exact_one_sided(6, 0), '8_against_1': C1.exact_one_sided(8, 1)},
           'smallest_one_sided_win_that_rejects': {'at_0.025_no_loss': next(b for b in range(1, 30) if C1.exact_one_sided(b, 0) <= 0.025), 'at_0.05_no_loss': next(b for b in range(1, 30) if C1.exact_one_sided(b, 0) <= 0.05),
                                                   'at_0.025_one_loss': next(b for b in range(1, 40) if C1.exact_one_sided(b, 1) <= 0.025), 'at_0.025_two_losses': next(b for b in range(1, 40) if C1.exact_one_sided(b, 2) <= 0.025)},
           'scenarios': out}
    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n')
    print('largest FWER', worst, res['bound_holds'], res['smallest_one_sided_win_that_rejects'])
    for r in out: print(f"rho={r['scene_dependence_rho']} | {r['scenario'][:78]:78s} | gen {r['reject_H_gen']:.3f} mem(IUT) {r['reject_H_mem_intersection_union']:.3f} mem(pooled) {r['reject_H_mem_pooled_draft']:.3f} FWER {r['familywise_error']}")


if __name__ == '__main__': main()
