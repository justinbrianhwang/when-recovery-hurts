"""Check the README research GIFs against the raw Phase B records, without the code that made them (no hci import).

  1. source files: sha256 of trace and frames equal the sha256 stored in the scene record; scene json hash as listed.
  2. Panel A, every step of the 1280x720 GIF: where a frame is shown, the stored raw frame closest to the GIF crop is the
     frame of that very step (trunk frame t up to the trigger, the arm's control frame t - trigger after it); where none
     was stored, the panel is the uniform grey placeholder.
  3. numbers and events shown: recomputed from the raw trace rows and the scene record.

  With <replay_dir> (GIFs made from the RGB re-run, scripts/39_rgb_replay.py): 1b. the re-run's trace rows equal the stored
  rows exactly and its frames equal every stored frame of the same step; 2. every Panel A crop is closest to the re-run
  frame of its own step (the last frame of an ended episode is held).

usage: python scripts/verify_research_gifs.py <phase_b_root> <gif_dir> [<replay_dir>]   -> writes <gif_dir>/VERIFICATION.json (refuses overwrite)
"""
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
W, H = 1280, 720
# Panel A image rectangles in the 1280x720 frame (axes [0.01, 0.53 - 0.40 i, 0.33, 0.33], image 2:1 centred in the box)
def box(i):
    x0, x1 = 0.01 * W, 0.34 * W; ytop, ybot = H * (1 - (0.53 - 0.40 * i + 0.33)), H * (1 - (0.53 - 0.40 * i)); h = (x1 - x0) / 2; y0 = ytop + ((ybot - ytop) - h) / 2
    return int(round(x0)) + 4, int(round(y0)) + 4, int(round(x1)) - 4, int(round(y0 + h)) - 4


COLOUR = {'C0': '#555555', 'VISION_CENTER_V1': '#1f77b4', 'VISION_CENTER_V2': '#d62728', 'CENTER_ORACLE': '#ff7f0e'}


def crop(im, i, arm):
    """Inside of the coloured frame (axes spines in the arm's colour) around Panel A row i."""
    a = np.asarray(im.convert('RGB'), float); x0, y0, x1, y1 = box(i); r = a[y0 - 20:y1 + 20, x0 - 10:x1 + 10]
    c = np.array([int(COLOUR[arm][k:k + 2], 16) for k in (1, 3, 5)]); m = np.all(np.abs(r - c) < 30, -1)
    rows_, cols_ = np.where(m.sum(1) > 0.6 * m.shape[1])[0], np.where(m.sum(0) > 0.6 * m.shape[0])[0]
    if len(rows_) < 2 or len(cols_) < 2: raise ValueError(f'frame of row {i} not found')
    def centre(idx):                                      # centre line of the first and the last spine run
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1); return runs[0].mean(), runs[-1].mean()
    (t, bt), (l, rt) = centre(rows_), centre(cols_)
    return r, (t, bt, l, rt)


INSET = 6


def compare(cr, f):
    """Mean absolute pixel difference between the GIF crop and raw frame f, mapped onto the image extent, borders excluded."""
    r, (t, bt, l, rt) = cr; f = np.asarray(f)
    if f.shape[1] == 128: f = np.concatenate([np.full_like(f, 150), f], 1)
    h, w = int(round(bt - t)), int(round(rt - l)); up = np.asarray(Image.fromarray(f.astype(np.uint8)).resize((w, h), Image.NEAREST), float)
    y0, x0 = int(round(t)), int(round(l)); g = r[y0:y0 + h, x0:x0 + w]
    return float(np.mean(np.abs(g[INSET:-INSET, INSET:-INSET] - up[INSET:-INSET, INSET:-INSET])))


def main(root, gdir, rdir=None):
    gdir = Path(gdir); out = gdir / 'VERIFICATION.json'
    if out.exists(): raise SystemExit(f'Refused overwrite: {out}')
    man = json.loads((gdir / 'MANIFEST.json').read_text(encoding='utf-8')); res = {'what': 'independent check of the README GIFs against the raw Phase B records', 'cases': {}}; ok_all = True
    for name, m in man['cases'].items():
        d = Path(root) / m['stage']; scene = d / 'scenes' / f"{m['scene']}.json"; rec = json.loads(scene.read_text(encoding='utf-8')); chk = {}
        chk['source_sha256'] = all([sha(d / rec['trace']) == rec['trace_sha256'] == m['source_files']['trace']['sha256'], sha(d / rec['frames']) == rec['frames_sha256'] == m['source_files']['frames']['sha256'],
                                    sha(scene) == m['source_files']['scene']['sha256']])
        with gzip.open(d / rec['trace'], 'rt', encoding='utf-8') as f: tr = json.load(f)['traces']
        with np.load(d / rec['frames']) as z: fr = {k: z[k] for k in z.files}
        t0 = rec['trigger']['step']; cells = {c['arm']: c for c in rec['cells']}; ta, tb = m['steps_shown']; rows = m['arms_shown']; rp = None
        if rdir is not None:                                                     # 1b. the re-run against the stored record
            rd = Path(rdir) / name; seed = m['scene']
            with gzip.open(rd / f'trace_{seed}.json.gz', 'rt', encoding='utf-8') as f: rtr = json.load(f)['traces']
            with np.load(rd / f'frames_{seed}.npz') as z: rp = {k: z[k] for k in z.files}
            same_rows = set(rtr) == set(tr) and all(len(rtr[k]) == len(tr[k]) and all(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(rtr[k], tr[k])) for k in tr)
            fd = {'trunk': int(np.abs(rp['trunk'][:len(fr['trunk'])].astype(int) - fr['trunk'].astype(int)).max())}
            for k in fr:
                if k == 'trunk' or k not in rp: continue
                new = rp[k][t0:t0 + len(fr[k])]; new = new[:, :, 128:] if fr[k].shape[2] == 128 else new; fd[k] = int(np.abs(new.astype(int) - fr[k].astype(int)).max())
            lens = {a: len(rp[a]) - 1 == cells[a]['steps'] for a in rows}
            chk['replay'] = {'trace_rows_identical_json': bool(same_rows), 'max_abs_pixel_diff_to_stored_frames': fd, 'frames_cover_whole_episode': lens}
        # 2. Panel A
        gif = Image.open(gdir / 'hires' / f'{name}_1280x720.gif'); assert gif.n_frames == tb - ta + 1, (name, gif.n_frames)
        shown = mism = grey = grey_bad = 0; bad = []; errs = []
        for n, t in enumerate(range(ta, tb + 1)):
            gif.seek(n); im = gif.copy()
            for i, a in enumerate(rows):
                if rp is not None:
                    cand = {(a, q): rp[a][q] for q in range(len(rp[a]))}; exp = (a, min(t, len(rp[a]) - 1))
                else:
                    cand = {('trunk', q): fr['trunk'][q] for q in range(len(fr['trunk']))} | ({(a, j): fr[a][j] for j in range(len(fr[a]))} if a in fr else {})
                    exp = ('trunk', t) if t <= t0 else ((a, t - t0) if a in fr and t - t0 < len(fr[a]) else None)
                cr = crop(im, i, a)
                if exp is None:
                    grey += 1; rr_, (t_, b_, l_, r_) = cr; c = rr_[int(t_) + INSET:int(b_) - INSET, int(l_) + INSET:int(r_) - INSET]; frac = np.mean(np.all(np.abs(c - 175) < 3, -1))
                    if frac < 0.85: grey_bad += 1; bad.append((t, a, 'placeholder expected', round(float(frac), 3)))
                    continue
                step = (lambda k: k[1]) if rp is not None else (lambda k: k[1] if k[0] == 'trunk' else k[1] + t0)
                shown += 1; near = [k for k in cand if (k[0] == 'trunk') == (exp[0] == 'trunk') and abs(step(k) - step(exp)) <= 3] + [exp]
                err = {k: compare(cr, cand[k]) for k in set(near)}; best = min(err, key=err.get); errs.append(err[exp])
                if best != exp and err[best] < err[exp] - 1e-9: mism += 1; bad.append((t, a, 'closest raw frame', list(best), 'expected', list(exp)))
        chk['panel_A'] = {'frames_shown': shown, 'closest_raw_frame_is_the_step_s_own': shown - mism, 'placeholders': grey, 'placeholders_not_grey': grey_bad, 'max_mean_abs_pixel_difference_to_its_raw_frame': round(max(errs), 2) if errs else None, 'problems': bad[:10]}
        # 3. numbers and events from raw rows
        def rows_of(a): return [r for r in tr['trunk'] if r['t'] <= t0] + tr[a]
        def row(a, t): return next(r for r in rows_of(a) if r['t'] == t)
        off = lambda r: float(np.hypot(r['cubeB'][0] - r['cubeA'][0], r['cubeB'][1] - r['cubeA'][1]))
        num = {'true offset at trigger [POST-HOC]': f"{off(row(rows[0], t0)) * 1e3:.1f} mm"}; ev = {}
        for a in rows:
            c = cells[a]; k = c['control_steps'] if c.get('acted') else 0; rr = rows_of(a)
            if k:
                push = float(np.hypot(*(np.array(row(a, t0 + k)['cubeB'][:2]) - row(a, t0)['cubeB'][:2])))
                mx = max(float(np.hypot(*(np.array(row(a, q)['cubeB'][:2]) - row(a, t0)['cubeB'][:2]))) for q in range(t0, t0 + k + 1))
                num[f"{ {'C0': 'C0', 'VISION_CENTER_V1': 'v1', 'VISION_CENTER_V2': 'v2', 'CENTER_ORACLE': 'oracle'}[a]} target pushed during move [POST-HOC]"] = (push, mx)
                num[f"{ {'C0': 'C0', 'VISION_CENTER_V1': 'v1', 'VISION_CENTER_V2': 'v2', 'CENTER_ORACLE': 'oracle'}[a]} offset after move [POST-HOC]"] = f"{off(row(a, t0 + k)) * 1e3:.1f} mm"
            succ = next((r['t'] for r in rr if r['success']), None); opn = next((r['t'] for r in rr if r['t'] > t0 + k and r['action'][-1] > 0), None)
            ev[a] = {'control_steps_equals_logged_steps': (not k) or k == len(c['control']['steps']), 'success_step': succ, 'outcome_y_matches_trace': bool(c['y']) == (succ is not None),
                     'episode_length': rr[-1]['t'], 'length_matches_record': rr[-1]['t'] == c['steps'], 'first_open_command': opn,
                     'events_in_gif_match': all(any(q == e[0] for e in m['events'][a]) for q in [x for x in (t0, opn, succ, c['steps']) if x is not None])}
        shown_num = m['numbers_shown']; nmatch = {}
        for k, v in num.items():
            if isinstance(v, tuple):
                txt = shown_num.get(k, ''); val = float(txt.split(' mm')[0]) if txt else None; nmatch[k] = {'shown': txt, 'raw_end_of_move_mm': round(v[0] * 1e3, 2), 'raw_max_during_move_mm': round(v[1] * 1e3, 2), 'ok': val is not None and abs(val - v[1] * 1e3) < 0.051}
            else: nmatch[k] = {'shown': shown_num.get(k), 'raw': v, 'ok': shown_num.get(k) == v}
        g = cells['VISION_CENTER_V2'].get('gt_reading_error') or {}
        if g: nmatch['v2 estimate error at trigger [POST-HOC]'] = {'shown': shown_num.get('v2 estimate error at trigger [POST-HOC]'), 'raw': f"{g['offset_error_m'] * 1e3:.1f} mm (own bound {g['bound_m'] * 1e3:.0f} mm)"}; nmatch['v2 estimate error at trigger [POST-HOC]']['ok'] = nmatch['v2 estimate error at trigger [POST-HOC]']['shown'] == nmatch['v2 estimate error at trigger [POST-HOC]']['raw']
        if m['scene'] == 51009:
            mt = cells['VISION_CENTER_V2']['control']['readings'][0]['memory_target']; b = row('VISION_CENTER_V2', t0)['cubeB']; b_last = row('VISION_CENTER_V2', mt['t_last'])['cubeB']
            nmatch['memory'] = {'t_last': mt['t_last'], 'remembered_vs_true_target_at_trigger_mm': round(float(np.hypot(mt['xy'][0] - b[0], mt['xy'][1] - b[1])) * 1e3, 2),
                                'target_moved_since_last_view_mm': round(float(np.hypot(b[0] - b_last[0], b[1] - b_last[1])) * 1e3, 2), 'ok': True}
        chk['numbers'] = nmatch; chk['events'] = ev
        rp_ok = rp is None or (chk['replay']['trace_rows_identical_json'] and all(v == 0 for v in chk['replay']['max_abs_pixel_diff_to_stored_frames'].values()) and all(chk['replay']['frames_cover_whole_episode'].values()) and grey == 0)
        chk['pass'] = bool(rp_ok and chk['source_sha256'] and mism == 0 and grey_bad == 0 and all(v['ok'] for v in nmatch.values()) and all(all(x for k, x in e.items() if isinstance(x, bool)) for e in ev.values()))
        ok_all &= chk['pass']; res['cases'][name] = chk; print(name, 'PASS' if chk['pass'] else 'FAIL', chk['panel_A'])
    res['all_pass'] = ok_all; res['verifier_sha256'] = sha(__file__); res['gif_sha256'] = {p.relative_to(gdir).as_posix(): sha(p) for p in sorted(gdir.rglob('*.gif'))}
    out.write_bytes(json.dumps(res, indent=1, default=str).encode('utf-8')); return ok_all


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None) else 1)
