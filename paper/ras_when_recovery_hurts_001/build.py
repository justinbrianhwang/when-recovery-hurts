"""Build the flat Elsevier source package, compile main and Supplementary PDFs, zip, and write the manifest.

Steps (each logged to build_logs/):
  1. check_numbers.py must pass (manuscript numbers equal data/recount.json).
  2. Assemble submission/RAS_source_flat/ with NO subfolders (Elsevier LaTeX upload rule): main.tex, supplementary.tex,
     references.bib, figure PDFs, table .tex files, elsarticle-num.bst (copied from the TeX installation when present).
     Paths in the .tex files are rewritten to the flat layout; the sources in manuscripts/ are not modified.
  3. Compile with tectonic, or latexmk, or pdflatex + bibtex (first engine found). No engine -> exit code 2 and status
     NOT_COMPILED in the manifest (nothing is hidden).
  4. Copy main.bbl into the package (Elsevier asks for the .bbl), zip the folder, write
     submission/FINAL_PDF_AND_SUPPLEMENT_MANIFEST.json (SHA256, page counts, engine, status).
usage: python -P paper/ras_when_recovery_hurts_001/build.py [--render-dir DIR]
"""
from __future__ import annotations

import argparse, hashlib, json, shutil, subprocess, sys, time, zipfile
from pathlib import Path

P = Path(__file__).resolve().parent
FLAT = P / 'submission/RAS_source_flat'; LOGS = P / 'build_logs'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd, cwd, log):
    with open(log, 'a', encoding='utf-8') as f:
        f.write(f'\n$ {" ".join(map(str, cmd))}\n'); f.flush()
        r = subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=900)
        f.write(f'[exit {r.returncode}]\n')
    return r.returncode


def mask_logs():
    """Replace the home folder and the user name in every build log (personal paths stay out of the repository)."""
    home = str(Path.home()); user = Path.home().name
    for f in LOGS.glob('*.log'):
        s = f.read_text(encoding='utf-8', errors='replace')
        for h in (home, home.replace('\\', '/'), home.replace('\\', '\\\\')): s = s.replace(h, '~')
        s = s.replace(user, '<user>'); f.write_text(s, encoding='utf-8')


def engine():
    for name in ('tectonic', 'latexmk', 'pdflatex'):
        p = shutil.which(name)
        if p: return name, p
    return None, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--render-dir'); a = ap.parse_args()
    LOGS.mkdir(exist_ok=True); stamp = time.strftime('%Y%m%dT%H%M%S'); log = LOGS / f'build_{stamp}.log'
    if run([sys.executable, '-P', P / 'figures/source_code/check_numbers.py'], P, log) != 0:
        print('check_numbers failed; see', log); sys.exit(1)
    if FLAT.exists(): shutil.rmtree(FLAT)  # derived build folder, regenerated from the sources each time
    FLAT.mkdir(parents=True)
    for name in ('main.tex', 'supplementary.tex'):
        s = (P / 'manuscripts' / name).read_text(encoding='utf-8')
        s = s.replace('\\graphicspath{{../figures/}{./}}', '\\graphicspath{{./}}').replace('\\newcommand{\\tabdir}{../tables/}', '\\newcommand{\\tabdir}{}')
        if '../' in s: raise ValueError(f'relative path left in {name}')
        (FLAT / name).write_text(s, encoding='utf-8')
    shutil.copy2(P / 'manuscripts/references.bib', FLAT)
    for f in sorted((P / 'figures').glob('*.pdf')): shutil.copy2(f, FLAT)
    for f in sorted((P / 'tables').glob('T*.tex')): shutil.copy2(f, FLAT)
    name, exe = engine(); status = 'NOT_COMPILED: no TeX engine (tectonic, latexmk or pdflatex) found'; pdfs = {}
    if name:
        kp = shutil.which('kpsewhich')
        if kp:
            r = subprocess.run([kp, 'elsarticle-num.bst'], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip(): shutil.copy2(r.stdout.strip(), FLAT)
        elif name == 'tectonic':  # no kpsewhich: take the .bst from Tectonic's own bundle
            r = subprocess.run([exe, '-X', 'bundle', 'cat', 'elsarticle-num.bst'], capture_output=True)
            if r.returncode != 0 or not r.stdout.startswith(b'%%'): raise RuntimeError('could not obtain elsarticle-num.bst from the Tectonic bundle')
            (FLAT / 'elsarticle-num.bst').write_bytes(r.stdout)
        ok = True
        for doc in ('main', 'supplementary'):
            if name == 'tectonic': ok &= run([exe, '--keep-intermediates', '--keep-logs', f'{doc}.tex'], FLAT, log) == 0
            elif name == 'latexmk': ok &= run([exe, '-pdf', '-interaction=nonstopmode', '-halt-on-error', f'{doc}.tex'], FLAT, log) == 0
            else:
                ok &= run([exe, '-interaction=nonstopmode', '-halt-on-error', f'{doc}.tex'], FLAT, log) == 0
                if doc == 'main': ok &= run([shutil.which('bibtex') or 'bibtex', doc], FLAT, log) == 0
                for _ in range(2): ok &= run([exe, '-interaction=nonstopmode', '-halt-on-error', f'{doc}.tex'], FLAT, log) == 0
        status = 'COMPILED' if ok else 'COMPILE_ERROR (see log)'
        out = P / 'submission'
        for doc in ('main', 'supplementary'):
            pdf = FLAT / f'{doc}.pdf'
            if pdf.exists():
                dst = out / ('RAS_main.pdf' if doc == 'main' else 'RAS_supplementary.pdf'); shutil.copy2(pdf, dst)
                try:
                    import pymupdf; d = pymupdf.open(dst); pages = d.page_count
                    if a.render_dir:
                        rd = Path(a.render_dir); rd.mkdir(parents=True, exist_ok=True)
                        for i, pg in enumerate(d): pg.get_pixmap(dpi=110).save(rd / f'{doc}_p{i + 1:02d}.png')
                except ImportError: pages = None
                pdfs[dst.name] = {'sha256': sha(dst), 'pages': pages}
        for junk in FLAT.iterdir():  # keep only upload files (+ .bbl) in the package
            if junk.suffix in ('.aux', '.log', '.blg', '.out', '.fls', '.fdb_latexmk', '.spl', '.pdf') and junk.stem in ('main', 'supplementary'): junk.unlink()
    files = sorted(FLAT.iterdir())
    if any(f.is_dir() for f in files): raise ValueError('subfolder in the flat package')
    zp = P / 'submission/RAS_source_flat.zip'
    if zp.exists(): zp.unlink()
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, f.name)
    man = {'built_at_utc': stamp, 'status': status, 'engine': name, 'engine_path_name': Path(exe).name if exe else None,
           'source_zip': {'file': zp.name, 'sha256': sha(zp), 'files': {f.name: sha(f) for f in files}}, 'pdfs': pdfs,
           'has_bbl': (FLAT / 'main.bbl').exists(), 'log': log.name}
    (P / 'submission/FINAL_PDF_AND_SUPPLEMENT_MANIFEST.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
    mask_logs()
    print(json.dumps({k: man[k] for k in ('status', 'engine', 'pdfs', 'has_bbl')}, indent=1))
    sys.exit(0 if status == 'COMPILED' else 2)


if __name__ == '__main__':
    main()
