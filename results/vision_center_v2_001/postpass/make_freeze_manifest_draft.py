"""Vision-CENTER v2, Phase C0 (post-pass): DRAFT freeze manifest of the candidate as it was executed.

A record of hashes, not a freeze: `hci freeze` is not called, no frozen directory is created, nothing is locked.
Refuses to overwrite its output. Run from the repository root:

    python results/vision_center_v2_001/postpass/make_freeze_manifest_draft.py
"""
import json, sys, hashlib, subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]; HERE = Path(__file__).resolve().parent; OUT = HERE / 'V2_FREEZE_MANIFEST_DRAFT.json'
sys.path.insert(0, str(ROOT / 'src'))
from hci import vision_center as V1, vision_center_v2 as V2, recovery_v1_oracle as O

PLAN = 'results/vision_center_v2_001/preparation/plan.json'; PLAN_SHA = '1b4ea3bb65bd4cfdffc13a610fee010d568a5837e632701d4b0d265e6ba66e86'; COMMIT = 'fd23aa2'
EXE = 'results/vision_center_v2_001/execution'
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
digest = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()
git = lambda *a: subprocess.check_output(['git', '-C', str(ROOT), *a], text=True).strip()


def files(paths): return {p: sha(p) for p in sorted(paths)}


def const(mod, names): return {n: (list(v) if isinstance(v, tuple) else v) for n in names for v in [getattr(mod, n)]}


def main():
    if OUT.exists(): raise SystemExit(f'Refused overwrite: {OUT}')
    plan = json.loads((ROOT / PLAN).read_text(encoding='utf-8')); head = git('rev-parse', 'HEAD'); exec_commit = git('rev-parse', COMMIT)
    code_now = files(plan['code'].keys()); changed = sorted(k for k, v in plan['code'].items() if code_now[k] != v)
    src = files(p.relative_to(ROOT).as_posix() for p in (ROOT / 'src/hci').glob('*.py'))
    tests = files(['tests/test_vision_center.py', 'tests/test_vision_center_v2.py', 'tests/test_recovery_v1_oracle.py'])
    tracked = sorted(set(code_now) | set(src)); diff = git('diff', '--name-only', exec_commit, '--', *tracked); tests_diff = git('diff', '--name-only', exec_commit, '--', *sorted(tests))
    pol = ROOT / plan['base_policy']['path']; res = json.loads((ROOT / EXE / 'pilot/result.json').read_text(encoding='utf-8')); sh = json.loads((ROOT / EXE / 'shadow/shadow_result.json').read_text(encoding='utf-8'))
    rec = lambda st, lo, hi: hashlib.sha256(''.join(sha(f'{EXE}/{st}/scenes/{s}.json') for s in range(lo, hi + 1)).encode()).hexdigest()
    v1_names = ['ACT_MIN_M', 'RESIDUAL_TOL_M', 'CARRY_Z_MIN_M', 'STEPS_LEFT_MIN', 'TARGET_TOP_Z_M', 'CUBE_HALF_M', 'IMG', 'PIXEL_CENTRE']
    rules = {'v2_memory': V2.MEMORY, 'v2_decision': V2.DECISION, 'v2_abstain_reasons': list(V2.ABSTAIN), 'v2_memory_error_reporting_m': V2.MEMORY_ERROR_M, 'v2_intervention': V2.INTERVENTION, 'v2_noise_repeat': V2.REPEAT,
             'v1_constants': const(V1, [n for n in v1_names if hasattr(V1, n)]), 'v1_abstain_reasons': list(V1.ABSTAIN),
             'move_bounds': {'step_cmd_m': O.STEP_CMD_M, 'target_cap_m': O.TARGET_CAP_M, 'max_steps': O.MAX_STEPS, 'min_cmd_m': O.MIN_CMD_M}, 'trigger': plan['semantics']['trigger'],
             'gate_B2': V2.GATE, 'shadow_pass_B1': V2.SHADOW_PASS, 'arms': list(V2.ARMS)}
    man = {
        'document': 'V2_FREEZE_MANIFEST_DRAFT', 'status': 'DRAFT. Not a freeze. `hci freeze` was not called, no frozen directory exists, no calibration or Test input is selected. It records what a later freeze of this candidate would have to match.',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'work_id': V2.WORK, 'candidate': 'VISION_CENTER_V2 as executed in Phase B (intervention id vision_center_v2_xy), with VISION_CENTER_V1 as the unchanged baseline and CENTER_ORACLE as the non-deployable reference',
        'identity': {'approved_plan': PLAN, 'approved_plan_sha256_expected': PLAN_SHA, 'approved_plan_sha256_now': sha(PLAN), 'execution_commit': exec_commit, 'head_at_draft': head,
                     'files_changed_since_execution_commit': diff.splitlines(), 'code_files_changed_since_plan': changed, 'upstream_commit': plan['upstream_commit'], 'upstream_lock': files(['upstream.lock.json'])},
        'base_policy': {'path': plan['base_policy']['path'], 'weights.pt': sha(pol.relative_to(ROOT).as_posix() + '/weights.pt'), 'policy_meta.json': sha(pol.relative_to(ROOT).as_posix() + '/policy_meta.json'), 'train_seed': 1, 'iteration': 80000, 'weights': 'EMA',
                        'source_checkpoint_sha256': plan['base_policy']['source_checkpoint_sha256'], 'pretrained_encoder': files(['artifacts/pretrained/resnet18-f37072fd.pth']), 'rule': 'frozen; not retrained; v2 is not adapted to any other policy'},
        'not_part_of_the_candidate': {'second_policy_for_generalisation_only': {'path': 'artifacts/policies/stack_v1_s2_d500_clean_resnet18_i080000', **files(['artifacts/policies/stack_v1_s2_d500_clean_resnet18_i080000/weights.pt', 'artifacts/policies/stack_v1_s2_d500_clean_resnet18_i080000/policy_meta.json'])}},
        'environment': {'config': files(['configs/stackcube.yaml']), 'config_digest': plan['config']['config_sha256'], 'env': plan['config']['env'], 'policy_io': plan['config']['policy'], 'audit_tolerances': plan['config']['audit'],
                        'requirements': files(p.relative_to(ROOT).as_posix() for p in (ROOT / 'requirements').glob('*.txt')), 'build': files(['pyproject.toml', 'Dockerfile']),
                        'execution_runtime': res['runtime'], 'execution_host': 'Vast machine 146456 (France), RTX 4090, AMD Ryzen 7 7800X3D, driver 595.99.02; thread limit 1',
                        'server_logs': files([f'{EXE}/server_logs/vc2_versions.log', f'{EXE}/server_logs/vc2_pytest.log', f'{EXE}/doctor/doctor.json', f'{EXE}/dry_run_server.json']),
                        'gap': 'no `pip freeze` of the execution server is in the backup, only the package versions in vc2_versions.log and the runtime block of result.json. A real freeze needs the full pip freeze of the host it will run on.',
                        'host_dependence': 'uninterrupted outcomes differ between hosts in 16 of 300 scenes (same totals). Calibration and Test must each run in one session on one host, every arm from the same snapshot.'},
        'code': {'plan_code_files': code_now, 'all_equal_to_plan': not changed, 'src_hci': src, 'tests': tests, 'tests_changed_since_execution_commit': tests_diff.splitlines(), 'tests_note': 'Phase C0 changed one assertion in tests/test_recovery_v1_oracle.py: the seed-use scan also accepts the post-pass folder (offline audit of used records). No code under src/ or scripts/ changed.', 'rules_sha256_recorded_in_result': res['rules_sha256']},
        'rules': {**rules, 'rules_digest_sha256': digest(rules), 'frozen_since': 'the plan hash above; no constant was changed after an outcome'},
        'seed_registries': files(['configs/splits.json', 'configs/policy_eval_splits.json', 'configs/vision_v1_splits.json', 'configs/recovery_v1_splits.json', 'configs/vision_center_splits.json', 'configs/vision_center_v2_splits.json']),
        'evidence_development_only': {'ledger': files(['artifacts/ledger/vision_center_v2_pilot.json']), 'stage_B2': {**files([f'{EXE}/pilot/result.json', f'{EXE}/result_local_recomputed.json']), 'scene_records_digest_sha256': rec('pilot', 50900, 51099), 'label': res['followup_gate']['label'],
                                                                                                                            'success': {a: res['success'][a]['k'] for a in V2.ARMS}},
                                      'stage_B1': {**files([f'{EXE}/shadow/shadow_result.json']), 'scene_records_digest_sha256': rec('shadow', 50600, 50899), 'label': sh['label']},
                                      'archive_file_list': files([f'{EXE}/archive_file_sha256.txt']), 'server_backup_archive_sha256': 'b4611a2fe0dd18f68454fb068e9afd61d191cb1d168846475f6fbffd5dddb6b4', 'cost': files([f'{EXE}/cost.json']),
                                      'scope': 'development data. Not calibration, not Test, no statistical guarantee.'},
        'analysis_scripts': files([f'{EXE}/posthoc_analysis/posthoc_analysis.py', f'{EXE}/posthoc_analysis/figure_results.py', 'results/vision_center_v2_001/postpass/postpass_stats.py', 'results/vision_center_v2_001/postpass/memory_risk.py',
                                   'results/vision_center_v2_001/postpass/seed_and_split_audit.py', 'results/vision_center_v2_001/postpass/next_phase_plan_draft.py']),
        'analysis_outputs': files([f'{EXE}/posthoc_analysis/posthoc_analysis.json', 'results/vision_center_v2_001/postpass/postpass_stats.json', 'results/vision_center_v2_001/postpass/memory_risk.json', 'results/vision_center_v2_001/postpass/seed_and_split_audit.json', 'results/vision_center_v2_001/postpass/component_attribution_from_logs.json']),
        'open_before_a_real_freeze': ['protocol amendment for the new research cycle (the v0.3 freeze code expects selector heads, a candidate registry and a calibration evaluation; v2 has none)',
                                      'decision on the memory safety issues (keep v2 as the candidate, or a separate v2s candidate with its own development evidence)',
                                      'candidate list and multiplicity family, epsilon, delta', 'calibration and Test seed sets and their single-use ledgers', 'pre-registered analysis script and its hash', 'full pip freeze of the execution host',
                                      'policy seeds covered by the claim'],
        'how_to_check': 'recompute every sha256 listed here; `git diff --name-only fd23aa2 -- <code files>` must be empty; hci.vision_center_v2.MEMORY / DECISION / GATE must equal `rules`'}
    problems = [k for k, v in {'plan hash differs': man['identity']['approved_plan_sha256_now'] != PLAN_SHA, 'code changed since plan': bool(changed), 'files changed since execution commit': bool(diff),
                               'policy weights differ from plan': man['base_policy']['weights.pt'] != plan['base_policy']['weights.pt'], 'result weights differ': res['weights_sha256'] != man['base_policy']['weights.pt'],
                               'result plan hash differs': res['plan_sha256'] != PLAN_SHA or sh['plan_sha256'] != PLAN_SHA, 'rules hash in result differs from code': res['rules_sha256'] != {k: code_now[k] for k in res['rules_sha256']}}.items() if v]
    man['consistency_checks'] = {'problems': problems, 'pass': not problems}
    OUT.write_text(json.dumps(man, indent=1), encoding='utf-8', newline='\n'); print(json.dumps({'pass': not problems, 'problems': problems, 'rules_digest': man['rules']['rules_digest_sha256'], 'files': len(code_now) + len(src)}))


if __name__ == '__main__': main()
