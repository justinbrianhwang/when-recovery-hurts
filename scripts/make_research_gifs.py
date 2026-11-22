"""Research-figure GIFs for the README, from stored Phase B records of Vision-CENTER v2 (work vision_center_v2_001).

Reads stored records only: scene json, per-step trace, recorded RGB frames. Runs no simulator, uses no new scene, changes no
controller. Every panel shows the same simulation step t.

  Panel A  recorded RGB as stored (base | wrist camera). Steps without a stored frame show a grey "NO RGB RECORDED" panel;
           nothing is interpolated, re-rendered or reconstructed.
  Panel B  top view of the logged TCP, held cube and target (POST-HOC GROUND TRUTH) and the controller's own estimates (x).
  Panel C  true XY offset and finger opening per step, intervention phase shaded, events marked.
  Panel D  decisions, reasons, key numbers and the event list, all read from the record.

usage: python scripts/make_research_gifs.py <phase_b_root> <out_dir> [<replay_dir>]
  <phase_b_root>  folder holding shadow/ and pilot/ (git-ignored server backup)
  <out_dir>       new folder; refused if it exists (no overwrite)
  <replay_dir>    optional: output of scripts/39_rgb_replay.py. Used only if every scene's re-run reproduced the stored
                  record exactly (every trace row and cell) and names the same stored record by sha256; Panel A then
                  shows the re-run's frames at every step instead of grey panels.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hci import trajectory_audit as T

# Fixed before any figure was made. Rescue: the lowest-seed v2 rescue (rule `representative_v2_rescue` of the trajectory audit).
# The other two scenes were named in the request.
CASES = {
    'recovery_success': {'stage': 'shadow', 'seed': 50632, 'rows': ('C0', 'VISION_CENTER_V2'),
                         'title': 'v2 rescue: C0 releases off-centre and fails; v2 centres the held cube, the policy succeeds'},
    'unsafe_intervention': {'stage': 'pilot', 'seed': 51004, 'rows': ('VISION_CENTER_V1', 'VISION_CENTER_V2'),
                            'title': 'v1 moves and pushes the target 5.7 mm, then stops; v2 abstains (SIDE_CONTACT_RISK)'},
    'memory_failure': {'stage': 'pilot', 'seed': 51009, 'rows': ('VISION_CENTER_V2', 'CENTER_ORACLE'),
                       'title': 'stale target memory (37.7 mm off after an unseen push); v2 abstains (LOW_CARRY)'},
}
NAME = {'C0': 'C0 (no intervention)', 'VISION_CENTER_V1': 'v1', 'VISION_CENTER_V2': 'v2', 'CENTER_ORACLE': 'oracle (privileged; reference, not ground truth)'}
SHORT = {'C0': 'C0', 'VISION_CENTER_V1': 'v1', 'VISION_CENTER_V2': 'v2', 'CENTER_ORACLE': 'oracle'}
W, H, DPI = 1280, 720, 100
README_W = 960
MS, MS_EVENT, MS_FIRST, MS_LAST = 160, 1100, 900, 2600
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def colour(arm): return T.ARM_STYLE[arm][0]


def readings(rec, arm):
    c = T.cell(rec, arm); return ((c or {}).get('control') or {}).get('readings') or []


def scene_data(root, case):
    d = Path(root) / case['stage']; rec, tr, fr = T.load_scene(d, case['seed'])
    for k in ('trace', 'frames'):
        if sha(d / rec[k]) != rec[k + '_sha256']: raise SystemExit(f'{k} of scene {case["seed"]} does not match its recorded sha256')
    S = {a: T.series(rec, tr, a) for a in case['rows']}
    E = {a: T.events(rec, S[a], a) for a in case['rows']}
    return d, rec, tr, fr, S, E


def window(rec, case, E):
    t0 = rec['trigger']['step']; start = t0 - 20
    mem = (T.cell(rec, 'VISION_CENTER_V2')['control'].get('memory_log') and readings(rec, 'VISION_CENTER_V2')[0].get('memory_target')) or None
    if case['seed'] == 51009 and mem: start = min(start, mem['t_last'] - 8)
    ends = [v for a in case['rows'] for v in (E[a]['control_end'], E[a]['release_cmd'], E[a]['success']) if v is not None]
    return max(0, start), min(200, max(ends) + 15)


def limits(rec, case, S, ta, tb, mem):
    """Axis ranges, fixed for the whole GIF: a square top view around the cubes from shortly before the trigger (earlier
    approach paths are clipped), and an offset range from 10 steps before the trigger."""
    t0 = rec['trigger']['step']; pts = []
    for a in case['rows']:
        s = S[a]; i = T.at(s, max(ta, t0 - 6))
        pts += [s['tcp'][i:, :2], s['A'][i:, :2], s['B'][T.at(s, ta):, :2]]
    pts = np.concatenate(pts) * 1e3
    if mem: pts = np.vstack([pts, np.array(mem['xy']) * 1e3])
    c = (pts.min(0) + pts.max(0)) / 2; half = (pts.max(0) - pts.min(0)).max() / 2 + 30
    off = max(np.linalg.norm(T.offset_xy(S[a]), axis=1)[T.at(S[a], max(ta, t0 - 10)):].max() for a in case['rows']) * 1e3
    return (c[0] - half, c[0] + half, c[1] - half, c[1] + half), (ta, tb), max(20, off * 1.4)


REPLAY = None      # {case_name: {arm: frames}} from scripts/39_rgb_replay.py, set by build() only after its checks pass


def frame_for(rec, fr, arm, t, rp=None):
    """(image, provenance) of the frame of `arm` at step t, or (None, why). With `rp` (the replayed frames of this scene,
    index t = observation after step t), every step comes from the re-run whose trajectory equals the stored record; after
    the arm's episode ended its last frame is held and marked."""
    t0 = rec['trigger']['step']; c = T.cell(rec, arm)
    if rp is not None:
        a = rp[arm]; last = len(a) - 1
        return a[min(t, last)], 'RGB re-run (rgb_replay_001), same trajectory as the record' + (f'; ended at step {last}, last frame held' if t > last else '')
    if t <= t0: return fr['trunk'][t], 'recorded RGB, shared prefix (same frame for every arm)'
    if arm in fr and t - t0 < len(fr[arm]):
        a = fr[arm][t - t0]
        if a.shape[1] == 128:
            g = np.full_like(a, 150); return np.concatenate([g, a], 1), 'recorded RGB, recovery move (v1 stored the wrist camera only)'
        return a, 'recorded RGB, recovery move'
    return None, 'NO RGB RECORDED: trace only (Panels B, C)' + (' (episode ended)' if t > c['steps'] else '')


def state(S, a, t):
    """Index of step t in the arm's series, or the last index if the episode already ended (marked by the caller)."""
    return T.at(S[a], t) if t <= S[a]['t'][-1] else len(S[a]['t']) - 1


def events_of(rec, E, a):
    c = T.cell(rec, a); t0 = rec['trigger']['step']; acted = c.get('acted') and c['control_steps'] > 0
    ev = [(t0, 'trigger: ' + ('intervention starts' if acted else 'abstained, policy continues'))]
    if acted:
        ev += [(E[a]['control_end'], f"control ends ({c['control'].get('end_reason')})"), (E[a]['control_end'] + 1, 'policy resumes')]
    if E[a]['release_cmd'] is not None: ev.append((E[a]['release_cmd'], 'first open command after the move' if acted else 'first open command'))
    if E[a]['success'] is not None and E[a]['success'] != c['steps']: ev.append((E[a]['success'], 'success'))
    ev.append((c['steps'], 'success: episode ends' if E[a]['success'] == c['steps'] else f"episode ends: {'SUCCESS' if c['y'] else 'FAIL'}"))
    return sorted(set(ev), key=lambda e: (e[0], 'episode' in e[1] or 'success' in e[1]))


def draw(case_name, case, rec, fr, S, E, t, lims, memory):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    t0 = rec['trigger']['step']; rows = case['rows']; (x0, x1, y0, y1), (ta, tb), _ = lims
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI); fig.patch.set_facecolor('white')
    fig.text(0.01, 0.975, f"scene {rec['scene']} (stage {'B1' if case['stage'] == 'shadow' else 'B2'}, used development scene)  |  policy train seed 1  |  {case['title']}", fontsize=9.5, weight='bold', va='top')
    fig.text(0.01, 0.945, f"simulation step t = {t}   (trigger at {t0}; every panel shows step t)", fontsize=9, va='top', color='#222')
    fig.text(0.99, 0.945, ('A: RGB re-run, trajectory identical to the record' if REPLAY else 'A: recorded RGB as stored') + '    B, C: POST-HOC GROUND TRUTH from the logged simulator state (not a controller input); x = controller estimate', fontsize=7.5, va='top', ha='right', color='#a00000')
    # Panel A
    for i, a in enumerate(rows):
        ax = fig.add_axes([0.01, 0.53 - i * 0.40, 0.33, 0.33]); img, why = frame_for(rec, fr, a, t, (REPLAY or {}).get(case_name))
        if img is None:
            ax.imshow(np.full((128, 256, 3), 175, np.uint8)); ax.text(128, 64, 'NO RGB RECORDED\nfor this arm and step\n(trace only: Panels B, C)', ha='center', va='center', fontsize=10, weight='bold')
        else:
            ax.imshow(img, interpolation='nearest')
            if img[:, :128].std() == 0: ax.text(64, 64, 'base camera\nnot recorded\nfor v1', ha='center', va='center', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(colour(a)); s.set_linewidth(2.5)
        ax.set_title(f"A{i + 1}  {NAME[a]}", fontsize=9, loc='left', color=colour(a), weight='bold', pad=14)
        ax.text(0, -4, why, fontsize=7, va='bottom'); ax.text(64, 131, 'base camera', ha='center', va='top', fontsize=7); ax.text(192, 131, 'wrist camera', ha='center', va='top', fontsize=7)
    # Panel B
    ax = fig.add_axes([0.40, 0.46, 0.25, 0.44])
    for a in rows:
        s = S[a]; i = state(S, a, t); j0 = T.at(s, ta); cl = colour(a)
        ax.plot(s['tcp'][j0:i + 1, 0] * 1e3, s['tcp'][j0:i + 1, 1] * 1e3, ':', color=cl, lw=1)
        ax.plot(s['A'][j0:i + 1, 0] * 1e3, s['A'][j0:i + 1, 1] * 1e3, '-', color=cl, lw=1.6)
        ax.plot(s['B'][j0:i + 1, 0] * 1e3, s['B'][j0:i + 1, 1] * 1e3, '--', color=cl, lw=1.2)
        for X, fill, ls in ((s['A'][i], True, '-'), (s['B'][i], False, '--')):
            ax.add_patch(Rectangle(((X[0] - .02) * 1e3, (X[1] - .02) * 1e3), 40, 40, fill=fill, alpha=.18 if fill else 1, ec=cl, fc=cl, ls=ls, lw=1.2))
        ax.plot(*(s['tcp'][i, :2] * 1e3), 'o', ms=4, color=cl)
        for r in readings(rec, a):
            if r['t'] <= t and r.get('target_xy') is not None: ax.plot(r['target_xy'][0] * 1e3, r['target_xy'][1] * 1e3, 'x', color=cl, ms=6, mew=1.6)
    if memory and t >= memory['t_last']:
        ax.plot(memory['xy'][0] * 1e3, memory['xy'][1] * 1e3, 'P', color=colour('VISION_CENTER_V2'), ms=9, mfc='white', mew=1.6)
        ax.annotate(f"v2 remembered target\n(last seen step {memory['t_last']})", (memory['xy'][0] * 1e3, memory['xy'][1] * 1e3), xytext=(6, 8), textcoords='offset points', fontsize=7)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.25)
    ax.set_xlabel('world x [mm]', fontsize=8); ax.set_ylabel('world y [mm]', fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title('B  top view (POST-HOC GROUND TRUTH)', fontsize=9, loc='left', weight='bold')
    ax.text(0.01, 0.01, 'solid = held cube (fill: footprint)\ndashed = target cube   dotted = TCP\nx = target as estimated by the controller', transform=ax.transAxes, fontsize=6.5, va='bottom', bbox=dict(fc='white', ec='#ccc', alpha=.85))
    # Panel C
    c1 = fig.add_axes([0.40, 0.21, 0.58, 0.17]); c2 = fig.add_axes([0.40, 0.06, 0.58, 0.11], sharex=c1)
    for a in rows:
        s = S[a]; j0 = T.at(s, ta); j1 = state(S, a, min(tb, t)); off = np.linalg.norm(T.offset_xy(s), axis=1) * 1e3; cl = colour(a)
        c1.plot(s['t'][j0:j1 + 1], off[j0:j1 + 1], color=cl, lw=1.6, label=f'{SHORT[a]} true offset')
        est = [(r['t'], r['offset_norm_m'] * 1e3) for r in readings(rec, a) if r['t'] <= t and r.get('offset_norm_m') is not None]
        if est: c1.plot(*zip(*est), 'x', color=cl, ms=5, mew=1.4, label=f'{SHORT[a]} estimate')
        c2.plot(s['t'][j0:j1 + 1], s['width'][j0:j1 + 1] * 1e3, color=cl, lw=1.4)
        op = s['t'][j0:j1 + 1][s['action'][j0:j1 + 1, -1] > 0]; c2.plot(op, np.full(len(op), 82 - 6 * rows.index(a)), '|', color=cl, ms=6)
        c = T.cell(rec, a)
        if c.get('acted') and c['control_steps'] > 0 and t > t0:                      # shaded only up to the current step
            for axx in (c1, c2): axx.axvspan(t0 + .5, min(t0 + c['control_steps'], t) + .5, color=cl, alpha=.10, lw=0)
    if memory:
        sv = S['VISION_CENTER_V2']; ts = [q for q in range(memory['t_last'], min(t0, t) + 1)]
        if ts: c1.plot(ts, [np.linalg.norm(np.array(memory['xy']) - sv['B'][T.at(sv, q), :2]) * 1e3 for q in ts], color='#6a3d9a', lw=1.6, ls='-.', label='v2 memory error (remembered vs true target)')
    c1.axhline(10, color='#888', lw=.8, ls=':'); c1.text(ta + .5, 10.5, 'residual tolerance 10 mm', fontsize=6.5, color='#555')
    for axx in (c1, c2): axx.axvline(t0, color='k', lw=1); axx.axvline(t, color='#d4a000', lw=1.6); axx.grid(alpha=.25); axx.tick_params(labelsize=7); axx.set_xlim(ta, tb)
    c1.set_ylim(0, lims[2]); c1.set_ylabel('XY offset [mm]', fontsize=8); c1.legend(fontsize=6.5, ncol=3, loc='upper right', framealpha=.9)
    c1.set_title('C  true XY offset, held cube to target, per step (POST-HOC); x = estimate; shade = move; black = trigger; yellow = now', fontsize=7.8, loc='left', weight='bold')
    c2.set_ylim(0, 86); c2.set_ylabel('finger\nopening [mm]', fontsize=7.5); c2.set_xlabel('simulation step', fontsize=8)
    c2.text(ta + .5, 70, '| = open command (gripper)', fontsize=6.5, color='#444'); plt.setp(c1.get_xticklabels(), visible=False)
    # Panel D
    lines = [('D  decisions and events (from the record)', 'k', 'bold'),
             (f"trigger reading: open command pending, close run {rec['trigger']['close_run']}, width {rec['trigger']['finger_width_m'] * 1e3:.1f} mm", '#222', 'normal')]
    for a in rows:
        c = T.cell(rec, a); ctl = c.get('control') or {}
        dec = ctl.get('decision') or ('ACT' if c.get('acted') else 'no intervention'); lines.append((f"{SHORT[a]}: {dec}; {c['control_steps']} move steps; {'SUCCESS' if c['y'] else 'FAIL'} ({c['steps']} steps)", colour(a), 'bold'))
        for q, what in events_of(rec, E, a): lines.append((f"   {'>' if q <= t else ' '} step {q:>3}  {what}", colour(a) if q <= t else '#999', 'normal'))
    for k, v in case['numbers']: lines.append((f"{k}: {v}", '#222', 'normal'))
    y = 0.915
    for txt, cl, wt in lines:
        fig.text(0.665, y, txt, fontsize=7.3, color=cl, weight=wt, va='top', family='monospace' if txt.startswith('   ') else None); y -= 0.024
    fig.canvas.draw(); im = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return im


def numbers(rec, tr, case, S):
    c = {a: T.check_arm(rec, tr, a) for a in case['rows']}; a0 = case['rows'][0]; t0 = rec['trigger']['step']
    out = [('true offset at trigger [POST-HOC]', f"{np.linalg.norm(T.offset_xy(S[a0])[T.at(S[a0], t0)]) * 1e3:.1f} mm")]
    for a in case['rows']:
        ch = c[a]
        if ch.get('7_no_slip_no_push', {}).get('target_xy_displacement_m') is not None:
            out.append((f"{SHORT[a]} target pushed during move [POST-HOC]", f"{ch['7_no_slip_no_push']['target_xy_displacement_m'] * 1e3:.1f} mm ({ch['7_no_slip_no_push']['status']})"))
        if ch.get('3_command_reduces_true_error', {}).get('true_offset_after_m') is not None:
            out.append((f"{SHORT[a]} offset after move [POST-HOC]", f"{ch['3_command_reduces_true_error']['true_offset_after_m'] * 1e3:.1f} mm"))
    g = T.cell(rec, 'VISION_CENTER_V2').get('gt_reading_error') or {}
    if g: out.append(('v2 estimate error at trigger [POST-HOC]', f"{g['offset_error_m'] * 1e3:.1f} mm (own bound {g['bound_m'] * 1e3:.0f} mm)"))
    return out, c


def load_replay(rdir, root):
    """Frames of scripts/39_rgb_replay.py, accepted only if each scene's re-run reproduced the stored record exactly."""
    global REPLAY
    rdir = Path(rdir); summ = json.loads((rdir / 'replay_summary.json').read_text(encoding='utf-8')); REPLAY = {}
    if summ.get('all_reproduced_exactly') is not True: raise SystemExit('Refused: the replay did not reproduce every stored record')
    for name, case in CASES.items():
        e = summ['cases'][name]; cmp = json.loads((rdir / name / 'compare.json').read_text(encoding='utf-8'))
        if not (e['trajectory_reproduced_exactly'] and cmp['trajectory_reproduced_exactly']) or e['scene'] != case['seed']: raise SystemExit(f'Refused: {name} not reproduced')
        if sha(Path(root) / case['stage'] / 'scenes' / f"{case['seed']}.json") != e['stored_record_sha256']: raise SystemExit(f'Refused: {name} replay names another record')
        fp = rdir / e['frames_file']
        if sha(fp) != e['frames_sha256']: raise SystemExit(f'Refused: {fp} changed')
        with np.load(fp) as z: REPLAY[name] = {k: z[k] for k in z.files}
    return summ


def build(root, out):
    from PIL import Image
    out = Path(out)
    if out.exists(): raise SystemExit(f'Refused overwrite: {out}')
    (out / 'hires').mkdir(parents=True); man = {'what': 'README research GIFs from stored Phase B records' + ('; Panel A from the RGB re-run rgb_replay_001 (trajectories identical to the records)' if REPLAY else '; no simulator run'), 'code_sha256': {}, 'cases': {}}
    for p in ('scripts/make_research_gifs.py', 'src/hci/trajectory_audit.py') + (('scripts/39_rgb_replay.py',) if REPLAY else ()): man['code_sha256'][p] = sha(Path(__file__).resolve().parents[1] / p)
    for name, case in CASES.items():
        d, rec, tr, fr, S, E = scene_data(root, case); case = {**case}; case['numbers'], checks = numbers(rec, tr, case, S)
        ta, tb = window(rec, case, E); t0 = rec['trigger']['step']
        mem = readings(rec, 'VISION_CENTER_V2')[0].get('memory_target') if case['seed'] == 51009 else None
        lims = limits(rec, case, S, ta, tb, mem)
        evsteps = {q for a in case['rows'] for q, _ in events_of(rec, E, a)}
        frames, durs, shown = [], [], []
        for t in range(ta, tb + 1):
            frames.append(draw(name, case, rec, fr, S, E, t, lims, mem)); durs.append(MS_EVENT if t in evsteps else MS)
            shown.append({'t': t, **{a: frame_for(rec, fr, a, t, (REPLAY or {}).get(name))[1] for a in case['rows']}})
        durs[0], durs[-1] = MS_FIRST, MS_LAST
        hi_imgs = [Image.fromarray(f) for f in frames]; small = [im.resize((README_W, README_W * H // W), Image.LANCZOS) for im in hi_imgs]
        for ims, path in ((small, out / f'{name}.gif'), (hi_imgs, out / 'hires' / f'{name}_1280x720.gif')):
            q = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in ims]
            q[0].save(path, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True)
        key = t0 if name == 'memory_failure' else E[case['rows'][0] if name == 'unsafe_intervention' else 'VISION_CENTER_V2']['control_end']      # key frame: end of the move, or the trigger
        hi_imgs[key - ta].save(out / f'{name}.png', optimize=True)
        man['cases'][name] = {
            'scene': case['seed'], 'stage': case['stage'], 'arms_shown': list(case['rows']), 'steps_shown': [ta, tb], 'trigger_step': t0, 'key_frame_step': key,
            'events': {a: events_of(rec, E, a) for a in case['rows']}, 'frame_provenance_per_step': shown,
            'source_files': {k: {'path': f"runs/vision_center_v2_001/server_backup/extracted/runs/vision_center_v2_001/{case['stage']}/{rec[k]}", 'sha256': rec[k + '_sha256']} for k in ('trace', 'frames')}
                            | {'scene': {'path': f"runs/vision_center_v2_001/server_backup/extracted/runs/vision_center_v2_001/{case['stage']}/scenes/{case['seed']}.json", 'sha256': sha(d / 'scenes' / f"{case['seed']}.json")}},
            'spec_checks': {a: {k: v['status'] for k, v in checks[a].items()} for a in case['rows']},
            'numbers_shown': dict(case['numbers']),
            'outputs': {p.name: sha(p) for p in (out / f'{name}.gif', out / f'{name}.png', out / 'hires' / f'{name}_1280x720.gif')},
        }
        print(name, 'frames', len(frames), {p: round((out / p).stat().st_size / 1e6, 2) for p in (f'{name}.gif', f'hires/{name}_1280x720.gif')})
    (out / 'MANIFEST.json').write_bytes(json.dumps(man, indent=1, default=str).encode('utf-8'))


if __name__ == '__main__':
    if len(sys.argv) == 4: load_replay(sys.argv[3], sys.argv[1])
    build(sys.argv[1], sys.argv[2])
