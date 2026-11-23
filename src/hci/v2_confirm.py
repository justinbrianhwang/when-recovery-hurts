"""Confirmatory Calibration / Final Test logic for the Vision-CENTER V2 cycle (work id ras_v2_confirm_001).

PREPARED, NOT RUN. Nothing here runs a simulator by itself; scripts/40_v2_confirm.py drives it. It changes no controller,
policy, threshold or earlier result. Protocol: Amendment 017 FINAL PROPOSAL (results/ras_submission_readiness_001/),
which is NOT in force until the user approves it; the runner refuses to freeze without an approval record that names the
amendment file by sha256, and nothing in this repository writes that record.

Fixed here (to be frozen by hash):
  * candidate: VISION_CENTER_V2 only; arms run per scene: C0 and VISION_CENTER_V2 from one snapshot of one episode
    (hci.vision_center_c1.collect_scene, unchanged); CENTER_ORACLE, V2S and every ablation are not candidates and not run.
  * harm H = P(Y_V2 = 0 | Y_C0 = 1); certificate per policy: n_plus = C0 successes, k = harms,
    p = BinomialCDF(k; n_plus, epsilon) (hci.statistics.binomial_safe_p), Holm over the declared family at delta.
    n_plus = 0 never passes. The family is the set of (candidate, policy) pairs declared before calibration.
  * a JOINT statement over the two policies is made only if every member of the family passes (Holm controls the
    familywise error of the per-member statements, so the joint statement inherits it); per-policy statements are reported
    separately.
  * efficacy (Final Test only, separate from the certificate): exact one-sided paired test V2 vs C0 per policy
    (hci.vision_center_c1.exact_one_sided on discordant scenes), Holm over the policies at alpha_efficacy.
  * calibration results never change the candidate, the policies, epsilon, delta, the tests or the rules.
"""
from __future__ import annotations

import hashlib, json, math, os
from datetime import datetime, timezone
from pathlib import Path

from .statistics import binomial_safe_p, holm_reject, paired_metrics, wilson_interval

WORK = 'ras_v2_confirm_001'
CANDIDATE = 'VISION_CENTER_V2'
ARMS = ('C0', CANDIDATE)
POLICY_IDS = ('train_seed_1', 'train_seed_2')
SPLITS = {'calibration': (30000, 30399), 'test': (90000, 90399)}
EPSILON = 0.05; DELTA = 0.05; ALPHA_EFFICACY = 0.05
APPROVAL = 'RUN_FROZEN_TEST'
ROLE = {'calibration': 'ras_v2_confirm_calibration', 'test': 'ras_v2_confirm_test'}


def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- seeds
def split_seeds(splits_json, name):
    """The seeds of a declared split, checked against the fixed range (no gap, no duplicate, no extra)."""
    if name not in SPLITS: raise ValueError(f'unknown split {name}')
    seeds = [int(s) for s in json.loads(Path(splits_json).read_text(encoding='utf-8'))['splits'][name]]
    lo, hi = SPLITS[name]
    if seeds != list(range(lo, hi + 1)): raise ValueError(f'{name}: the declared seeds are not exactly {lo}..{hi}')
    return seeds


def seeds_disjoint(sets):
    """{name: [seeds]} -> pairs of names that share a seed (must be empty)."""
    names = sorted(sets); out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(sets[a]) & set(sets[b]): out.append((a, b, sorted(set(sets[a]) & set(sets[b]))[:5]))
    return out


# ---------------------------------------------------------------- family and tests
def family(policies):
    pol = list(policies)
    if not pol or len(set(pol)) != len(pol) or not set(pol) <= set(POLICY_IDS): raise ValueError('the family needs distinct declared policy ids')
    return [{'candidate': CANDIDATE, 'policy': p} for p in pol]


def outcomes(records):
    """{scene: (y_C0, y_V2, v2_acted)} from scene records; every record must carry both arms once."""
    out = {}
    for r in records:
        cells = {c['arm']: c for c in r['cells']}
        if list(cells) != list(ARMS) or len(r['cells']) != len(ARMS): raise ValueError(f"scene {r['scene']}: arms {list(cells)} are not {list(ARMS)}")
        if r['scene'] in out: raise ValueError(f"scene {r['scene']} appears twice")
        out[int(r['scene'])] = (int(cells['C0']['y']), int(cells[CANDIDATE]['y']), bool(cells[CANDIDATE].get('acted')))
    return out


def certificate(outcomes_by_policy, fam, epsilon=EPSILON, delta=DELTA):
    """Harm certificate per family member (exact binomial, Holm over the family). Returns per-member results and the joint
    statement. The p-value tests H0: harm rate > epsilon (rejecting it certifies harm <= epsilon)."""
    rows = []
    for m in fam:
        o = outcomes_by_policy[m['policy']]; y0 = [o[s][0] for s in sorted(o)]; yg = [o[s][1] for s in sorted(o)]
        n_plus = sum(y0); k = sum(1 for a, b in zip(y0, yg) if a == 1 and b == 0)
        p = binomial_safe_p(k, n_plus, epsilon); moved = sum(1 for s in o if o[s][0] == 1 and o[s][2])
        rows.append({**m, 'scenes': len(o), 'n_plus': n_plus, 'k_harm': k, 'p_value': p, 'harm_rate': (k / n_plus) if n_plus else None,
                     'harm_wilson95': list(wilson_interval(k, n_plus)) if n_plus else None, 'c0_successes_where_v2_moved': moved,
                     'harm_among_moved': sum(1 for s in o if o[s][0] == 1 and o[s][1] == 0 and o[s][2])})
    rej = holm_reject([r['p_value'] for r in rows], delta) if rows else []
    for r, x in zip(rows, rej): r['certified'] = bool(x and r['n_plus'] > 0)
    return {'epsilon': epsilon, 'delta': delta, 'test': 'exact binomial, p = BinomialCDF(k; n_plus, epsilon); Holm over the declared family at delta; n_plus = 0 never passes',
            'null_hypothesis': 'harm rate > epsilon', 'members': rows, 'joint_statement_all_policies': bool(rows) and all(r['certified'] for r in rows)}


def exact_one_sided(b, c):
    n = int(b) + int(c)
    if n == 0: return 1.0
    return float(sum(math.comb(n, j) for j in range(int(b), n + 1)) / 2 ** n)


def efficacy(outcomes_by_policy, policies, alpha=ALPHA_EFFICACY):
    """Final Test only: exact one-sided paired test, V2 succeeds in more scenes than C0, per policy; Holm over the policies."""
    rows = []
    for p in policies:
        o = outcomes_by_policy[p]; b = sum(1 for s in o if o[s][0] == 0 and o[s][1] == 1); c = sum(1 for s in o if o[s][0] == 1 and o[s][1] == 0)
        m = paired_metrics([o[s][0] for s in sorted(o)], [o[s][1] for s in sorted(o)])
        rows.append({'policy': p, 'v2_only': b, 'c0_only': c, 'p_one_sided': exact_one_sided(b, c), 'success_c0': m['baseline_success_rate'], 'success_v2': m['success_rate'], 'gain_pp': m['gain_pp']})
    rej = holm_reject([r['p_one_sided'] for r in rows], alpha) if rows else []
    for r, x in zip(rows, rej): r['rejected_null'] = bool(x)
    return {'alpha': alpha, 'test': 'exact conditional paired test on discordant scenes, one-sided (V2 > C0); Holm over the policies', 'per_policy': rows}


def power_table(n_plus, family_size, epsilon=EPSILON, delta=DELTA, true_rates=(0.0, 0.005, 0.01, 0.02, 0.03, 0.04)):
    """Largest k that passes for a single member at the most stringent Holm level, and the pass probability at true harm rates."""
    from scipy.stats import binom
    level = delta / family_size; kmax = -1
    while binomial_safe_p(kmax + 1, n_plus, epsilon) <= level: kmax += 1
    return {'n_plus': n_plus, 'family_size': family_size, 'holm_first_level': level, 'largest_k_that_passes': kmax,
            'pass_probability': {f'{r:.3f}': (float(binom.cdf(kmax, n_plus, r)) if kmax >= 0 else 0.0) for r in true_rates}}


# ---------------------------------------------------------------- completeness
def check_complete(records_dir, seeds):
    """A run is analysable only if every planned scene has exactly one record and no error file exists."""
    d = Path(records_dir); errors = sorted(p.name for p in (d / 'scenes').glob('*.error.json'))
    have = sorted(int(p.stem) for p in (d / 'scenes').glob('*.json') if p.stem.isdigit())
    problems = []
    if errors: problems.append(f'error records: {errors[:5]}')
    if have != sorted(seeds): problems.append(f'records {len(have)} != planned {len(seeds)}; missing {sorted(set(seeds) - set(have))[:5]}, extra {sorted(set(have) - set(seeds))[:5]}')
    if problems: raise SystemExit('INCOMPLETE: ' + '; '.join(problems) + ' — a partial run is never summarised')
    return have


# ---------------------------------------------------------------- freeze, ledger, markers
def manifest(files, extra):
    """{path: sha256} of every frozen file plus the declared design."""
    missing = [f for f in files if not Path(f).is_file()]
    if missing: raise FileNotFoundError(f'frozen inputs missing: {missing}')
    return {'work': WORK, 'created_utc': now(), 'files_sha256': {str(f): sha256(f) for f in sorted(set(files))}, **extra}


def verify_manifest(man):
    changed = [f for f, h in man['files_sha256'].items() if not Path(f).is_file() or sha256(f) != h]
    if changed: raise SystemExit(f'Refused: frozen inputs changed: {changed}')
    return True


def require_amendment_approval(approval_path, amendment_path):
    """An approval record written by the user side: {'approved_by': 'user', 'amendment_sha256': <sha of the amendment file>,
    'scope': [...]} . Nothing in this repository writes it."""
    p = Path(approval_path)
    if not p.is_file(): raise SystemExit(f'Refused: no amendment approval record ({p}); Amendment 017 is not in force')
    a = json.loads(p.read_text(encoding='utf-8'))
    if a.get('approved_by') != 'user' or a.get('amendment_sha256') != sha256(amendment_path): raise SystemExit('Refused: the approval record does not name this amendment file by sha256')
    return a


def create_exclusive(path, payload):
    """Create a ledger or marker file exactly once; a second creation fails (operational lock, not access control)."""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'x', encoding='utf-8') as f: json.dump(payload, f, indent=1)
    return p


def require_test_approval(token):
    if token != APPROVAL: raise PermissionError('Explicit final Test approval token is required')
    return True


# ---------------------------------------------------------------- Final Test environment and host equivalence (added before the Test freeze)
HOST_CHECK_SEEDS = (51103, 51157, 51253)          # USED C1 development scenes; never a calibration or test seed


def live_environment(upstream_dir='third_party/ManiSkill'):
    """What the running host is: Python, GPU and driver (nvidia-smi), the full pip freeze, the upstream commit."""
    import platform, subprocess, sys
    pf = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True, text=True)
    gpu, drv = ([x.strip() for x in smi.stdout.strip().splitlines()[0].split(',')] if smi.returncode == 0 and smi.stdout.strip() else (None, None))
    up = subprocess.run(['git', '-C', upstream_dir, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    return {'python': platform.python_version(), 'gpu': gpu, 'driver': drv, 'upstream_commit': up, 'pip_freeze': sorted(l.strip() for l in pf.splitlines() if l.strip())}


def check_environment(live, expected):
    """Problems (empty list = the host equals the frozen environment): driver, GPU, Python minor version, upstream commit,
    and the pip freeze line by line."""
    p = []
    if live.get('driver') != expected['driver']: p.append(f"driver {live.get('driver')} != {expected['driver']}")
    if live.get('gpu') != expected['gpu']: p.append(f"gpu {live.get('gpu')} != {expected['gpu']}")
    if not str(live.get('python', '')).startswith(expected['python_minor'] + '.'): p.append(f"python {live.get('python')} is not {expected['python_minor']}.x")
    if live.get('upstream_commit') != expected['upstream_commit']: p.append(f"upstream {live.get('upstream_commit')} != {expected['upstream_commit']}")
    a, b = set(live.get('pip_freeze', [])), set(expected['pip_freeze'])
    if a != b: p.append(f'pip freeze differs: missing {sorted(b - a)[:5]}, extra {sorted(a - b)[:5]}')
    return p


def compare_runs(ref_dir, new_dir, policies, seeds):
    """Host equivalence: for every (policy, seed) the trunk, C0 and V2 trace rows (canonical JSON) and the C0/V2 outcomes of a
    new runner pass equal the reference pass. Returns (all_identical, rows)."""
    import gzip
    rows = []; ok = True
    canon = lambda x: json.dumps(x, sort_keys=True)
    for pid in policies:
        for s in seeds:
            rd, nd = Path(ref_dir) / f'pass_{pid}', Path(new_dir) / f'pass_{pid}'
            r0 = json.loads((rd / 'scenes' / f'{s}.json').read_text(encoding='utf-8')); r1 = json.loads((nd / 'scenes' / f'{s}.json').read_text(encoding='utf-8'))
            with gzip.open(rd / r0['trace'], 'rt', encoding='utf-8') as f: t0 = json.load(f)['traces']
            with gzip.open(nd / r1['trace'], 'rt', encoding='utf-8') as f: t1 = json.load(f)['traces']
            c0 = {c['arm']: (c['y'], c['steps'], c.get('acted')) for c in r0['cells']}; c1 = {c['arm']: (c['y'], c['steps'], c.get('acted')) for c in r1['cells']}
            e = {'policy': pid, 'scene': s, 'traces_identical': all(canon(t0.get(k)) == canon(t1.get(k)) for k in ('trunk',) + ARMS), 'cells_equal': c0 == c1}
            e['identical'] = e['traces_identical'] and e['cells_equal']; ok &= e['identical']; rows.append(e)
    return bool(ok and rows), rows
