"""REPRODUCIBILITY_AND_SHA256_MANIFEST.json for revision round 001 (run from the repository root)."""
import hashlib, json
from pathlib import Path
P = Path(__file__).resolve().parents[1]; R = P / 'revision_round_001'; REPO = P.parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rel = lambda p: str(Path(p).relative_to(REPO)).replace('\\', '/')
man = {
 'canonical_latex_project': {rel(f): sha(f) for f in sorted((P / 'latex_project').rglob('*')) if f.is_file()},
 'revision_outputs': {rel(f): sha(f) for f in sorted(R.rglob('*')) if f.is_file() and f.name != 'REPRODUCIBILITY_AND_SHA256_MANIFEST.json' and 'flat_export' not in f.parts},
 'flat_export': json.loads((R / 'flat_export/FLAT_PACKAGE_MANIFEST.json').read_text(encoding='utf-8')),
 'evidence_inputs': {rel(f): sha(f) for f in [P / 'data/recount.json', P / 'data/handback_stats.json', P / 'figures/FIGURE_PROVENANCE.json', P / 'figures/FIGURE_PROVENANCE_SUPP.json', P / 'tables/TABLE_PROVENANCE.json']},
 'generators': {rel(f): sha(f) for f in sorted((P / 'figures/source_code').glob('*.py')) + [P / 'tables/source_code/make_tables.py', P / 'export_flat_from_latex.py', P / 'make_latex_project.py']},
 'frozen_artifacts_unchanged': {'calibration_manifest': '01b70a065f82bcd80ccc031061e73bfdd3f86a302123387fbc578a4bbad3ef5e', 'final_test_manifest': '94c5a591554efa99f821dd2d7ea1099619afec790ee92f995e95ee8a92161c5f'},
 'pre_revision_backup': 'Drive Paper/manuscript_package/revision_round_001/backup_Latex_before_revision (47 files; = git 6027f0e latex_project)',
}
(R / 'REPRODUCIBILITY_AND_SHA256_MANIFEST.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
print(len(man['canonical_latex_project']), 'canonical files;', len(man['revision_outputs']), 'revision outputs')
