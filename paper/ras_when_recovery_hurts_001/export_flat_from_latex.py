"""Derive the flat Elsevier upload package FROM the canonical split project (latex_project/ = Drive Paper/Latex).

Direction of derivation: latex_project/ (canonical, edited) -> this script -> <out>/flat/ (no subfolders) -> compiled PDFs,
main.bbl, elsarticle-num.bst -> <out>/RAS_source_flat.zip + MANIFEST.json. The canonical files are only read.
usage: python -P paper/ras_when_recovery_hurts_001/export_flat_from_latex.py --out <dir> [--tectonic <path>]
"""
from __future__ import annotations

import argparse, hashlib, json, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path

P = Path(__file__).resolve().parent; LP = P / 'latex_project'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--tectonic', default=shutil.which('tectonic')); a = ap.parse_args()
    out = Path(a.out); flat = out / 'flat'
    if flat.exists(): raise SystemExit(f'{flat} exists; use a new output folder')
    flat.mkdir(parents=True)
    m = (LP / 'main.tex').read_text(encoding='utf-8')
    m = m.replace('\\graphicspath{{figures/}}', '\\graphicspath{{./}}').replace('\\newcommand{\\tabdir}{tables/}', '\\newcommand{\\tabdir}{}')
    m = re.sub(r'\\input\{sections/([^}]+)\}', r'\\input{\1}', m)
    s = (LP / 'supplementary/supplementary.tex').read_text(encoding='utf-8')
    s = s.replace('\\graphicspath{{figures/}{../figures/}}', '\\graphicspath{{./}}')
    s = re.sub(r'\\IfFileExists\{tables/[^\n]*\n', '\\\\newcommand{\\\\tabdir}{}\n', s)
    s = re.sub(r'\\IfFileExists\{supplementary/sections/[^\n]*\n', '\\\\newcommand{\\\\secdir}{}\n', s)
    for t, name in ((m, 'main.tex'), (s, 'supplementary.tex')):
        if 'sections/' in t.replace('%', '') and name == 'main.tex' and re.search(r'\\input\{sections/', t): raise ValueError('path left')
        (flat / name).write_text(t, encoding='utf-8')
    for f in (LP / 'sections').glob('*.tex'): shutil.copy2(f, flat)
    for f in (LP / 'supplementary/sections').glob('*.tex'): shutil.copy2(f, flat)
    for f in (LP / 'figures').glob('*.pdf'): shutil.copy2(f, flat)
    for f in (LP / 'tables').glob('*.tex'): shutil.copy2(f, flat)
    shutil.copy2(LP / 'references.bib', flat)
    names = [f.name for f in flat.iterdir()]
    if len(names) != len(set(n.lower() for n in names)): raise ValueError('file-name clash in the flat package')
    status = 'NOT_COMPILED'; pdfs = {}
    if a.tectonic:
        r = subprocess.run([a.tectonic, '-X', 'bundle', 'cat', 'elsarticle-num.bst'], capture_output=True)
        if r.returncode == 0 and r.stdout.startswith(b'%%'): (flat / 'elsarticle-num.bst').write_bytes(r.stdout)
        ok = True; log = out / 'flat_build.log'
        with open(log, 'w', encoding='utf-8') as lg:
            for doc in ('main', 'supplementary'):
                rr = subprocess.run([a.tectonic, '--keep-intermediates', '--keep-logs', f'{doc}.tex'], cwd=flat, capture_output=True, text=True)
                lg.write(f'$ tectonic {doc}.tex\n{rr.stdout}\n{rr.stderr}\n[exit {rr.returncode}]\n'); ok &= rr.returncode == 0
        status = 'COMPILED' if ok else 'COMPILE_ERROR'
        for doc in ('main', 'supplementary'):
            pdf = flat / f'{doc}.pdf'
            if pdf.exists():
                dst = out / f'flat_{doc}.pdf'; shutil.copy2(pdf, dst)
                import pymupdf; pdfs[dst.name] = {'sha256': sha(dst), 'pages': pymupdf.open(dst).page_count}
        for f in list(flat.iterdir()):
            if f.stem in ('main', 'supplementary') and f.suffix in ('.aux', '.log', '.blg', '.out', '.pdf'): f.unlink()
    files = sorted(flat.iterdir())
    if any(f.is_dir() for f in files): raise ValueError('subfolder in flat package')
    zp = out / 'RAS_source_flat.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, f.name)
    man = {'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'derived_from': 'latex_project (canonical)', 'status': status,
           'zip_sha256': sha(zp), 'files': {f.name: sha(f) for f in files}, 'has_bbl': (flat / 'main.bbl').exists(), 'has_bst': (flat / 'elsarticle-num.bst').exists(), 'pdfs': pdfs}
    (out / 'FLAT_PACKAGE_MANIFEST.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
    print(json.dumps({k: man[k] for k in ('status', 'has_bbl', 'has_bst', 'pdfs')}, indent=1), len(files), 'files')
    sys.exit(0 if status == 'COMPILED' else 2)


if __name__ == '__main__':
    main()
