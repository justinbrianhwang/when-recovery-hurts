"""Expected-set gate for a C1 run folder (review F2). Read-only: it never writes into the run folder, never creates or edits
a ledger, and never patches a plan or registry.

C1_ORIGINAL: the expected policies, scenes and arms come from the frozen plan file itself and the two registries it hashes
             (seed registry, arm registry); the plan file's own SHA256 must equal the frozen plan hash, and the run's
             provenance records (gates, ledger copy, authorization, no budget stop) must name that plan.
MOCK:        a separate small expected manifest (kind MOCK_EXPECTED_MANIFEST). It is checked with the same completeness
             rules; it may not name a C1 research scene, the frozen plan hash, or a run folder with a ledger.

Every problem is a string that starts with an upper-case code (for example MISSING_SCENES), so a block can be told apart
from an unrelated failure (an import error never produces these codes).
"""
from __future__ import annotations

import json, re
from pathlib import Path

import mechanism as M

PROVENANCE = ('scenes.json', 'probe_check/probe_check.json', 'audit/audit.json', 'budget_gate.json', 'budget_gate_at_start.json', 'ledger.json')
HEX64 = re.compile(r'^[0-9a-f]{64}$')
FROZEN_PLAN_SHA256 = '6b3f04fea1ea5346d6f0725b6e49ae9ce04c4fb132f37450243a807024e5dbf7'      # the frozen C1 plan; deliberately not a CLI option (review f1)
SCENE_FILE = re.compile(r'^[0-9]+(\.error)?\.json$')


def _list_of(x, typ):
    return isinstance(x, list) and len(x) > 0 and all(isinstance(v, typ) and not isinstance(v, bool) for v in x) and len(set(x)) == len(x)


def expected_from_plan(plan_path, repo, frozen_sha):
    """The expected sets of a C1_ORIGINAL run, derived from the plan file and the registries (never from a CLI string)."""
    plan_path = Path(plan_path); repo = Path(repo); p = []
    if not plan_path.is_file(): return None, [f'PLAN_MISSING: {plan_path}']
    h = M.sha256(plan_path)
    if h != frozen_sha: p.append(f'WRONG_PLAN: the plan file {plan_path.name} has sha256 {h}, not the frozen plan {frozen_sha}')
    plan = M.read_json(plan_path)
    if plan.get('pass') is not True: p.append('WRONG_PLAN: the plan file does not record pass = true')
    if plan.get('work_id') != 'vision_center_v2_c1_001': p.append(f"WRONG_PLAN: work_id {plan.get('work_id')!r}")
    sr, ar = plan.get('seed_registry') or {}, plan.get('arm_registry') or {}
    sf, af = repo / str(sr.get('path')), repo / str(ar.get('path'))
    scenes = arms = None
    if not sf.is_file(): p.append(f'SEED_REGISTRY_MISSING: {sr.get("path")}')
    elif M.sha256(sf) != sr.get('sha256'): p.append(f'SEED_REGISTRY_CHANGED: {sr.get("path")} differs from the plan')
    else:
        scenes = M.read_json(sf).get('splits', {}).get(sr.get('set'))
        if not _list_of(scenes, int) or len(scenes) != sr.get('n') or [scenes[0], scenes[-1]] != sr.get('first_last'): p.append('SEED_REGISTRY_INCONSISTENT: the registry set does not match the plan (n, first and last)')
    if not af.is_file(): p.append(f'ARM_REGISTRY_MISSING: {ar.get("path")}')
    elif M.sha256(af) != ar.get('sha256'): p.append(f'ARM_REGISTRY_CHANGED: {ar.get("path")} differs from the plan')
    else:
        reg = [a.get('id') for a in M.read_json(af).get('arms', []) if a.get('included') is True]
        arms = ar.get('arms')
        if reg != arms or not _list_of(arms, str): p.append(f'ARM_REGISTRY_INCONSISTENT: registry arms {reg} vs plan arms {arms}')
    pol = (plan.get('policies') or {}).get('order')
    if not _list_of(pol, str): p.append(f'WRONG_PLAN: policy order {pol!r}')
    exp = {'source': 'C1 plan file and the registries it hashes', 'plan_file': plan_path.as_posix(), 'plan_sha256': h, 'record_plan_sha256': frozen_sha, 'policies': pol, 'scenes': scenes, 'arms': arms,
           'role': sr.get('set'), 'require_policy_id': True, 'provenance_required': True, 'collector_result': 'result.json', 'spatial_fields': True,
           'seed_registry_sha256': sr.get('sha256'), 'arm_registry_sha256': ar.get('sha256')}
    return exp, p


def expected_from_mock(path, repo, frozen_sha):
    """A MOCK expected manifest: the same completeness rules on a small synthetic set."""
    path = Path(path); p = []
    if not path.is_file(): return None, [f'MOCK_MANIFEST_MISSING: {path}']
    m = M.read_json(path)
    if m.get('kind') != 'MOCK_EXPECTED_MANIFEST': p.append(f"MOCK_MANIFEST_INVALID: kind {m.get('kind')!r}")
    if not _list_of(m.get('policies'), str): p.append('MOCK_MANIFEST_INVALID: policies must be a non-empty list of distinct strings')
    if not _list_of(m.get('scenes'), int): p.append('MOCK_MANIFEST_INVALID: scenes must be a non-empty list of distinct integers')
    if not _list_of(m.get('arms'), str) or any(a not in (M.REF, M.ORACLE) + M.VISION_ARMS for a in m.get('arms') or []): p.append('MOCK_MANIFEST_INVALID: arms must be distinct registry arm names')
    # review f1: a MOCK manifest is small in scenes only. Arms, policies, the collector summary and the spatial fields follow
    # the plan, so a reduced manifest cannot define away an arm, a policy, the cross-check or S6.
    plan_p = Path(repo) / 'results/vision_center_v2_c1_001/preparation_v3/plan.json'; arm_p = Path(repo) / 'configs/vision_center_c1_arms.json'
    reg_arms = [a.get('id') for a in M.read_json(arm_p).get('arms', []) if a.get('included') is True] if arm_p.is_file() else None
    reg_pol = (M.read_json(plan_p).get('policies') or {}).get('order') if plan_p.is_file() else None
    if reg_arms is None or reg_pol is None: p.append('MOCK_REFERENCE_UNAVAILABLE: the arm registry or the plan policy order cannot be read')
    else:
        if m.get('arms') != reg_arms: p.append(f"MOCK_MANIFEST_REDUCED: arms {m.get('arms')} are not the registry arms {reg_arms}")
        if m.get('policies') != reg_pol: p.append(f"MOCK_MANIFEST_REDUCED: policies {m.get('policies')} are not the plan order {reg_pol}")
    if not m.get('collector_result'): p.append('MOCK_MANIFEST_REDUCED: no collector summary named (the cross-check would be skipped)')
    elif m.get('collector_result') != 'result.json': p.append(f"MOCK_MANIFEST_INVALID: collector_result must be result.json in the run folder, not {m.get('collector_result')!r}")      # review f4
    if m.get('spatial_fields') is not True: p.append('MOCK_MANIFEST_REDUCED: spatial_fields is not true (S6 would be skipped)')
    if m.get('require_policy_id', True) is not True: p.append('MOCK_MANIFEST_REDUCED: require_policy_id may not be switched off')
    rp = m.get('record_plan_sha256')
    if not isinstance(rp, str) or not HEX64.match(rp): p.append('MOCK_MANIFEST_INVALID: record_plan_sha256 must be a 64-hex string')
    elif rp == frozen_sha: p.append('MOCK_USES_RESEARCH_PLAN: a MOCK manifest may not name the frozen C1 plan')
    sf = Path(repo) / 'configs/vision_center_c1_splits.json'
    if sf.is_file() and isinstance(m.get('scenes'), list):
        research = set(M.read_json(sf)['splits']['vision_center_v2_c1_dev']); clash = sorted(set(m['scenes']) & research)
        if clash: p.append(f'MOCK_USES_RESEARCH_SEEDS: {clash[:10]}')
    exp = {'source': f'MOCK expected manifest {path.name}', 'plan_file': None, 'plan_sha256': None, 'record_plan_sha256': rp, 'policies': m.get('policies'), 'scenes': m.get('scenes'), 'arms': m.get('arms'),
           'role': m.get('role'), 'require_policy_id': m.get('require_policy_id', True) is not False, 'provenance_required': False, 'collector_result': m.get('collector_result'),
           'spatial_fields': bool(m.get('spatial_fields', False)), 'manifest_sha256': M.sha256(path)}
    return exp, p


def _short(xs, n=10): xs = sorted(xs, key=str); return f'{xs[:n]}{" ..." if len(xs) > n else ""} ({len(xs)})'


def check_run(root, exp, kind):
    """Compare the run folder with the expected sets, independently of what the records say about themselves."""
    root = Path(root); p = []
    if exp is None or any(exp.get(k) is None for k in ('policies', 'scenes', 'arms')): return ['EXPECTED_SET_UNAVAILABLE: the gate has no expected policies, scenes or arms']
    if not root.is_dir(): return [f'RUN_MISSING: {root}']
    want = [f'pass_{pid}' for pid in exp['policies']]; have = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith('pass_'))
    if set(want) - set(have): p.append(f'MISSING_POLICY_PASS: {sorted(set(want) - set(have))}')
    if set(have) - set(want): p.append(f'EXTRA_POLICY_PASS: {sorted(set(have) - set(want))}')
    if kind == 'MOCK' and (root / 'ledger.json').exists(): p.append('MOCK_USES_LEDGER: a MOCK run folder holds a ledger.json')
    scenes = {int(s) for s in exp['scenes']}; arms = list(exp['arms']); ids = {}
    for pid in exp['policies']:
        d = root / f'pass_{pid}' / 'scenes'
        if not d.is_dir(): continue
        odd = sorted(f.name for f in d.iterdir() if not (f.is_file() and SCENE_FILE.match(f.name)))
        if odd: p.append(f'UNEXPECTED_FILE_IN_SCENES pass_{pid}: {_short(odd)} (only <id>.json and <id>.error.json, lower case)')
        errs = sorted(f.name for f in d.glob('*.error.json'))
        if errs: p.append(f'ERROR_RECORD in pass_{pid}: {_short(errs)}')
        files = sorted(f for f in d.glob('*.json') if not f.name.endswith('.error.json')); seen = {}
        if not files: p.append(f'EMPTY_PASS: pass_{pid}/scenes holds no scene record')
        for f in files:
            try: rec = M.read_json(f)
            except (ValueError, UnicodeDecodeError) as e: p.append(f'UNREADABLE_RECORD pass_{pid}/{f.name}: {e!r}'); continue
            if not isinstance(rec, dict): p.append(f'MALFORMED_RECORD pass_{pid}/{f.name}: not an object'); continue
            s = rec.get('scene')
            if not isinstance(s, int) or isinstance(s, bool): p.append(f'MALFORMED_RECORD pass_{pid}/{f.name}: scene id {s!r}'); continue
            if f.stem != str(s): p.append(f'SCENE_FILE_MISMATCH pass_{pid}/{f.name}: holds scene {s}')
            if s in seen: p.append(f'DUPLICATE_SCENE pass_{pid}: scene {s} in {seen[s]} and {f.name}')
            seen[s] = f.name; where = f'pass_{pid}/{f.name}'
            if rec.get('arms') != arms: p.append(f'ARMS_DIFFER_FROM_REGISTRY {where}: {rec.get("arms")} vs {arms}')
            cells = rec.get('cells') if isinstance(rec.get('cells'), list) else []
            got = [c.get('arm') if isinstance(c, dict) else None for c in cells]
            if sorted(map(str, got)) != sorted(arms): p.append(f'CELLS_DIFFER_FROM_REGISTRY {where}: {got}')
            if rec.get('completed') is not True: p.append(f'INCOMPLETE_RECORD {where}')
            if exp.get('role') is not None and rec.get('role') != exp['role']: p.append(f"WRONG_ROLE {where}: {rec.get('role')!r}")
            if exp['require_policy_id'] and rec.get('policy_id') != pid: p.append(f"WRONG_POLICY {where}: policy_id {rec.get('policy_id')!r}")
            if rec.get('plan_sha256') != exp['record_plan_sha256']: p.append(f"WRONG_PLAN_IN_RECORD {where}: {rec.get('plan_sha256')!r}")
            tp = d.parent / str(rec.get('trace'))
            inside = lambda q: q.resolve().is_relative_to(d.parent.resolve()) and not Path(str(rec.get('trace'))).is_absolute()
            if rec.get('trace') and not inside(tp): p.append(f'TRACE_OUTSIDE_RUN {where}: {rec.get("trace")!r}')      # review f2: every input lies inside the hashed run folder
            elif not rec.get('trace') or not tp.is_file(): p.append(f'TRACE_MISSING {where}')
            elif M.sha256(tp) != rec.get('trace_sha256'): p.append(f'TRACE_SHA_MISMATCH {where}')
            fp = rec.get('frames')
            if not fp: p.append(f'FRAMES_MISSING {where}: no frames entry (the collector stores the frames of every scene)')
            elif Path(str(fp)).is_absolute() or not (d.parent / str(fp)).resolve().is_relative_to(d.parent.resolve()): p.append(f'FRAMES_OUTSIDE_RUN {where}: {fp!r}')
            elif not (d.parent / str(fp)).is_file(): p.append(f'FRAMES_MISSING {where}')
            elif M.sha256(d.parent / str(fp)) != rec.get('frames_sha256'): p.append(f'FRAMES_SHA_MISMATCH {where}')      # review f3: both kinds
        got_ids = set(seen); ids[pid] = got_ids
        if scenes - got_ids: p.append(f'MISSING_SCENES in pass_{pid}: {_short(scenes - got_ids)}')
        if got_ids - scenes: p.append(f'EXTRA_SCENES in pass_{pid}: {_short(got_ids - scenes)}')
    if len(ids) > 1 and len({frozenset(v) for v in ids.values()}) > 1: p.append(f'POLICY_SCENE_SETS_DIFFER: {{{", ".join(f"{k}: {len(v)}" for k, v in ids.items())}}} (same count does not mean the same scenes)')
    if exp['provenance_required']: p += check_provenance(root, exp)
    cr = exp.get('collector_result')
    if cr and not (root / cr).is_file(): p.append(f'COLLECTOR_RESULT_MISSING: {cr} (the collector summary of a finished run)')
    return p


def check_provenance(root, exp):
    """The run's own records of the approved procedure, read only (C1_ORIGINAL)."""
    root = Path(root); p = []; h = exp['record_plan_sha256']; rd = {}
    for n in PROVENANCE:
        f = root / n
        if not f.is_file(): p.append(f'PROVENANCE_MISSING: {n}'); continue
        rd[n] = M.read_json(f)
        if rd[n].get('plan_sha256') != h: p.append(f'PROVENANCE_OTHER_PLAN: {n}')
    stops = sorted(str(x.relative_to(root)) for x in root.rglob('budget_stop.json'))
    if stops: p.append(f'RUN_STOPPED: {stops}')
    au = root / 'authorization.json'
    if not au.is_file() or M.read_json(au).get('plan_sha256') != h: p.append('AUTHORIZATION_MISSING_OR_OTHER_PLAN')
    if 'probe_check/probe_check.json' in rd and rd['probe_check/probe_check.json'].get('pass') is not True: p.append('GATE_NOT_PASSED: probe_check')
    for n, k in (('scenes.json', 'pass'), ('audit/audit.json', 'pass'), ('budget_gate.json', 'fits_cap'), ('budget_gate_at_start.json', 'fits_cap')):
        if n in rd and rd[n].get(k) is not True: p.append(f'GATE_NOT_PASSED: {n}')
    if 'audit/audit.json' in rd and list(rd['audit/audit.json'].get('arms', [])) != list(exp['arms']): p.append('GATE_NOT_PASSED: audit arms differ from the registry')
    lg = rd.get('ledger.json')
    if lg is not None and (lg.get('set') != exp['role'] or list(lg.get('arms', [])) != list(exp['arms']) or lg.get('seed_manifest_sha256') != exp['seed_registry_sha256'] or lg.get('arm_registry_sha256') != exp['arm_registry_sha256']):
        p.append('LEDGER_DIFFERS_FROM_PLAN')
    return p


def row_coverage(rows, exp, stage_of):
    """After the analysis: every expected (policy, scene, arm) has exactly one row (a dropped branch is a problem)."""
    p = []
    for pid in exp['policies']:
        got = [(r['scene'], r['arm']) for r in rows if r['policy'] == pid and r['stage'] == stage_of(pid)]
        want = {(int(s), a) for s in exp['scenes'] for a in exp['arms']}
        if len(got) != len(set(got)): p.append(f'DUPLICATE_ROWS for {pid}')
        if want - set(got): p.append(f'MISSING_ROWS for {pid}: {_short(want - set(got))}')
        if set(got) - want: p.append(f'EXTRA_ROWS for {pid}: {_short(set(got) - want)}')
    return p


def crosscheck(fun, ref):
    """Independent recount: the analysis' OVERALL counts against the collector's own paired summary (O.paired layout).
    ref: {(stage, policy, arm): {'n', 'rescue', 'harm', 'net', 'reference_success', 'arm_success'}}."""
    rows, p = [], []
    d = {(x['stage'], x['policy'], x['arm'], x['step']): x['numerator'] for x in fun}
    pairs = (('n', 'overall_scenes_with_complete_outcomes'), ('reference_success', 'overall_c0_success'), ('arm_success', 'overall_success'), ('rescue', 'overall_rescue_of_c0_failure'), ('harm', 'overall_harm_of_c0_success'), ('net', 'overall_net_rescue_minus_harm'))
    for (stage, pol, arm), c in sorted(ref.items()):
        for ck, step in pairs:
            a = d.get((stage, pol, arm, step)); ok = a is not None and c.get(ck) == a
            rows.append({'stage': stage, 'policy': pol, 'arm': arm, 'collector_field': ck, 'collector_value': c.get(ck), 'analysis_step': step, 'analysis_value': a, 'match': ok})
            if not ok: p.append(f'CROSSCHECK_MISMATCH {stage} {pol} {arm} {ck}: collector {c.get(ck)} vs analysis {a}')
    if not ref: p.append('CROSSCHECK_UNAVAILABLE: no collector summary to compare with')
    return rows, p
