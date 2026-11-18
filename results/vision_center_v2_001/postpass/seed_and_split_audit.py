"""Vision-CENTER v2, Phase C0 (post-pass): seed-level audit. No simulator, no rollout.

1. Candidate development scenes 51100-51299: any overlap with a declared seed set, the demonstration scenes, or a seed
   named in a stored record or ledger. (The placement check, identical initial cube positions, needs the simulator and is
   planned as the first server step of the next phase.)
2. The protocol's calibration (30000-30399) and test (90000-90399) sets: is any of their seeds named in a stored record,
   run folder or ledger; does a frozen directory, a TEST_STARTED marker or a calibration evaluation exist.
3. The second Vision-V1 policy (train seed 2): files, hashes, records.

    python results/vision_center_v2_001/postpass/seed_and_split_audit.py
"""
import json, re, sys, importlib.util, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]; OUT = Path(__file__).resolve().parent / 'seed_and_split_audit.json'
sys.path.insert(0, str(ROOT / 'src'))
CAND = (51100, 51299); CAL = (30000, 30399); TEST = (90000, 90399)
RANGES = {'candidate_51100_51299': CAND, 'calibration_30000_30399': CAL, 'test_90000_90399': TEST}
PAT = re.compile(r'"(?:seed|scene|episode|episode_seed|scene_seed)":\s*(\d+)\b')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def driver():
    import os; os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location('v2_driver', ROOT / 'scripts/33_vision_center_v2.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def scan():
    hits = {k: {} for k in RANGES}; n = 0; roots = ['results', 'runs', 'artifacts/ledger', 'artifacts/manifests', 'logs', 'validation']
    for root in roots:
        for p in sorted((ROOT / root).rglob('*')) if (ROOT / root).is_dir() else []:
            if p.suffix not in ('.json', '.jsonl') or not p.is_file() or 'postpass' in p.parts: continue
            n += 1; nums = {int(x) for x in PAT.findall(p.read_text(encoding='utf-8', errors='replace'))}
            for k, (lo, hi) in RANGES.items():
                f = sorted(x for x in nums if lo <= x <= hi)
                if f: hits[k][p.relative_to(ROOT).as_posix()] = {'count': len(f), 'first': f[:5]}
    return n, roots, hits


def main():
    m = driver(); used = m.used_ledger(); used['vision_center_v2/vision_center_v2_pilot'] = list(json.loads((ROOT / m.MANIFEST).read_text())['splits']['vision_center_v2_pilot'])
    cand = set(range(CAND[0], CAND[1] + 1)); overlap = {k: sorted(cand & set(int(x) for x in v))[:10] for k, v in used.items()}; overlap = {k: v for k, v in overlap.items() if v}
    allused = sorted({int(x) for v in used.values() for x in v}); near = [x for x in allused if 51000 <= x <= 51400]
    n, roots, hits = scan()
    cal = set(range(CAL[0], CAL[1] + 1)); tst = set(range(TEST[0], TEST[1] + 1))
    declared = {k: {'n': len(v), 'calibration_overlap': len(cal & set(map(int, v))), 'test_overlap': len(tst & set(map(int, v)))} for k, v in used.items()}
    frozen = [p.relative_to(ROOT).as_posix() for p in ROOT.glob('frozen*')]; marker = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('TEST_STARTED.json') if '.git' not in p.parts]
    sel = [p.relative_to(ROOT).as_posix() for d in ('runs', 'results') for p in (ROOT / d).rglob('selection.json') if 'calibration' in p.read_text(encoding='utf-8', errors='replace')]
    pol = {}
    for s in (1, 2):
        d = ROOT / f'artifacts/policies/stack_v1_s{s}_d500_clean_resnet18_i080000'; meta = json.loads((d / 'policy_meta.json').read_text())
        res = json.loads((ROOT / f'results/vision_v1_001/run2/select_seed{s}/result.json').read_text())
        pol[f'train_seed_{s}'] = {'path': d.relative_to(ROOT).as_posix(), 'weights.pt': sha(d / 'weights.pt'), 'policy_meta.json': sha(d / 'policy_meta.json'), 'meta_weights_sha256_equal': meta['weights_sha256'] == sha(d / 'weights.pt'),
                                  'train_seed': meta['train_seed'], 'iteration': meta['iteration'], 'weights': meta['weights'], 'architecture_id': meta['architecture_id'], 'observation_schema': meta['observation_schema'],
                                  'source_checkpoint_sha256': meta['source_checkpoint_sha256'], 'upstream_commit': meta['upstream_commit'], 'num_demos': meta['args']['num_demos'], 'demo_path': meta['args']['demo_path'],
                                  'select_result_keys': sorted(res.keys())[:40], 'select_result': {k: res[k] for k in res if k in ('success_once', 'n', 'k', 'scenes', 'set', 'weights_sha256', 'policy', 'wilson95', 'success')}}
    selj = json.loads((ROOT / 'results/vision_v1_001/run2/selection.json').read_text())
    out = {'work_id': 'vision_center_v2_001', 'phase': 'C0 post-pass', 'candidate_scenes': {'range': list(CAND), 'n': len(cand), 'overlap_with_declared_or_used_sets': overlap, 'sets_checked': {k: len(v) for k, v in used.items()},
                                                                                         'used_or_declared_seeds_between_51000_and_51400': [near[0], near[-1], len(near)] if near else None,
                                                                                         'named_in_a_stored_record': hits['candidate_51100_51299'], 'seed_level_pass': not overlap and not hits['candidate_51100_51299'],
                                                                                         'placement_check': 'NOT DONE (needs the simulator): first server step of the next phase, against every placement on record; an identical placement fails the step'},
           'record_scan': {'files_scanned': n, 'roots': roots, 'pattern': PAT.pattern},
           'calibration_and_test': {'calibration_range': list(CAL), 'test_range': list(TEST), 'seeds_named_in_a_stored_record': {'calibration': hits['calibration_30000_30399'], 'test': hits['test_90000_90399']},
                                    'declared_sets_overlapping': {k: v for k, v in declared.items() if v['calibration_overlap'] or v['test_overlap']}, 'frozen_directories': frozen, 'TEST_STARTED_markers': marker,
                                    'selection_files_from_a_calibration_evaluation': sel},
           'vision_v1_policies': pol, 'vision_v1_selection': selj}
    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8', newline='\n')
    print(json.dumps({k: out[k] for k in ('candidate_scenes', 'record_scan', 'calibration_and_test')}, indent=1)[:5000]); print(json.dumps(pol, indent=1)[:3000])


if __name__ == '__main__': main()
