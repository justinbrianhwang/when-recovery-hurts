"""Phase C1 Option A: post-hoc spatial audit of records written with the spatial logger (hci.spatial_probe). Stored records
only; no simulator, no controller input, never writes into an input folder.

  metrics   every arm branch of every scene: per-step clearances of the arm links, hand, camera and fingers to both cubes
            and the table (exact, from the simulator's own collision shapes), observed contact vs proximity per category
            from the PhysX contact list, PhysX separation vs reconstruction, whole-robot swept volume of the move
            (approximation), items S1-S6 (EXPLORATORY); optionally the logged shapes against the fixed URDF / STL
  cases     the six representative cases: XY occupancy (B), 3D occupancy (C), physical-behaviour timeline (D) and a
            multi-panel GIF (A RGB | B | C | D), every panel at the same simulation step

Input is either C1 records (--records <pass folder with scenes/ and traces/>) or the local replay of stored Phase B actions
(--replay <scripts/37 replay folder> --phase-b <Phase B execution folder>). A replay record is the stored Phase B record
(trigger, decisions, commands, RGB of the ORIGINAL run) with the replayed trace (LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS):
its positions differ from the original run by the amounts in the replay summary; it is a test of the instrumentation and
the audit, never evidence about the original run or about C1. The earlier E1-E10 results are recomputed from the
ORIGINAL Phase B traces with the unchanged hci.spatial_audit and reported next to S1-S6, never replaced by them.
"""
from pathlib import Path
import argparse, csv, gzip, hashlib, importlib.util, io, json, math, sys, time
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from hci import spatial_links as L
from hci import spatial_geometry as G
from hci import spatial_audit as SA
from hci import trajectory_audit as T

_s36 = importlib.util.spec_from_file_location('c1_spatial_audit_v1', REPO / 'scripts' / '36_c1_spatial_audit.py'); A36 = importlib.util.module_from_spec(_s36); _s36.loader.exec_module(A36)
CASES = A36.CASES; ARM_C = A36.ARM_C
NAME = {'C0': 'C0 (no intervention)', 'VISION_CENTER_V1': 'v1', 'VISION_CENTER_V2': 'v2', 'CENTER_ORACLE': 'oracle (privileged reference)'}
POSTHOC = 'POST-HOC GROUND TRUTH from the logged simulator state: never a controller input'
GC = {'arm_links': '#7f8c8d', 'hand': '#2c3e50', 'camera': '#8e44ad', 'fingers': '#34495e'}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jdump(p, x): Path(p).write_text(json.dumps(x, indent=1, sort_keys=True, default=float), encoding='utf-8', newline='\n')


def guard(out, *inputs):
    out = Path(out).resolve()
    if out.exists(): raise SystemExit(f'Refused overwrite: {out}')
    for i in inputs:
        if i is None: continue
        i = Path(i).resolve()
        if out == i or i in out.parents or out in i.parents: raise SystemExit(f'Refused: output {out} inside or around input {i}')
    out.mkdir(parents=True); return out


# ---------------------------------------------------------------- loading
def load_c1(records, seed):
    d = Path(records); rec = json.loads((d / 'scenes' / f'{seed}.json').read_text(encoding='utf-8'))
    for k in ('trace', 'frames'):                                        # review a9: a changed trace or frame file is an error, never audited
        if rec.get(k) and sha(d / rec[k]) != rec[k + '_sha256']: raise ValueError(f'{k} of scene {seed} does not match its recorded sha256')
    with gzip.open(d / rec['trace'], 'rt', encoding='utf-8') as f: tr = json.load(f)['traces']
    fr = dict(np.load(d / rec['frames'])) if rec.get('frames') else None
    return rec, tr, fr, {'source': 'C1 record', 'record_sha256': sha(d / 'scenes' / f'{seed}.json'), 'trace_sha256': rec['trace_sha256']}


def load_replay(replay, phase_b, stage, seed):
    pb = Path(phase_b) / stage; rec = json.loads((pb / 'scenes' / f'{seed}.json').read_text(encoding='utf-8'))
    for k in ('trace', 'frames'):
        if rec.get(k) and sha(pb / rec[k]) != rec[k + '_sha256']: raise SystemExit(f'{k} of Phase B scene {seed} does not match its recorded sha256')
    tp = Path(replay) / 'traces' / f'trace_{seed}.json.gz'
    with gzip.open(tp, 'rt', encoding='utf-8') as f: d = json.load(f)
    if d.get('label') != 'LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS': raise SystemExit(f'{tp} is not a local replay trace')
    rp = d['traces']; trig = rec['trigger']['step'] if rec.get('triggered') else None; orig_rec = json.loads(json.dumps(rec))
    tr = {'trunk': rp['trunk'][1:]}
    for arm, rows in rp.items():
        if arm != 'trunk': tr[arm] = [r for r in rows if trig is not None and r['t'] > trig]
    rec = json.loads(json.dumps(rec)); rec['initial_state'] = rp['trunk'][0]; rec['source'] = 'LOCAL_CPU_REPLAY_OF_RECORDED_ACTIONS (positions: replay; decisions, commands, RGB: original Phase B run)'
    with gzip.open(pb / rec['trace'], 'rt', encoding='utf-8') as f: orig = json.load(f)['traces']
    fr = dict(np.load(pb / rec['frames'])) if rec.get('frames') else None
    return rec, tr, fr, {'source': rec['source'], 'replay_trace_sha256': sha(tp), 'phase_b_record_sha256': sha(pb / 'scenes' / f'{seed}.json'), 'phase_b_trace_sha256': rec['trace_sha256'], 'original_traces': orig, 'original_record': orig_rec}


def scenes_of(a):
    if a.records: return [(None, int(p.stem)) for p in sorted((Path(a.records) / 'scenes').glob('*.json')) if not p.name.endswith('.error.json')]
    out = []
    for p in sorted((Path(a.replay) / 'traces').glob('trace_*.json.gz')):
        s = int(p.name[6:-8]); stage = next(st for st in ('shadow', 'pilot') if (Path(a.phase_b) / st / 'scenes' / f'{s}.json').exists()); out.append((stage, s))
    return out


def load(a, stage, seed): return load_c1(a.records, seed) if a.records else load_replay(a.replay, a.phase_b, stage, seed)


def check_coverage(rec, tr):
    """Records mode (review a17): the trunk covers steps 1..trunk.steps and every arm steps trigger+1..its last step, consecutively."""
    want = lambda lo, hi: list(range(lo, hi + 1)); got = lambda rows: [int(r['t']) for r in rows]
    if got(tr.get('trunk') or []) != want(1, int(rec['trunk']['steps'])): raise ValueError(f"scene {rec['scene']}: the trunk trace does not cover steps 1..{rec['trunk']['steps']}")
    if rec.get('triggered'):
        t0 = int(rec['trigger']['step'])
        for c in rec['cells']:
            if got(tr.get(c['arm']) or []) != want(t0 + 1, int(c['steps'])): raise ValueError(f"scene {rec['scene']} {c['arm']}: the trace does not cover steps {t0 + 1}..{c['steps']}")


def arms_of(rec, tr):
    """The arm branches of a triggered scene; a scene that never triggered is audited as its trunk (review a15)."""
    return ['trunk'] + [c['arm'] for c in rec['cells'] if c['arm'] in tr] if rec.get('triggered') else ['trunk']      # review a18: the full trunk once, then each arm from trigger - 10


# ---------------------------------------------------------------- metrics
STEP_KEYS = ['t', 'phase', *[f'd_{g}_{t}' for g in L.GROUPS for t in L.TARGETS], 'd_robot_cubeB', 'robot_lowest_z', 'table_top_z', *[f'observed_contact_{k}' for k in L.CATEGORIES], *[f'listed_{k}' for k in L.CATEGORIES]]


def summarise(b, cell):
    M = b['metrics']; out = {'control_steps': b['control_steps'], 'robot_swept_volume_control_cm3': None if b['robot_swept_volume_control_m3'] is None else b['robot_swept_volume_control_m3'] * 1e6, 'swept_note': b['swept_note'], 'phases': {}}
    for ph in ('trunk', 'before_trigger', 'control', 'policy_after'):
        P = [m for m in M if m['phase'] == ph and not (ph == 'control' and m['t'] == b['trigger'])]
        if not P: continue
        e = {'steps': len(P)}
        for g in L.GROUPS:
            for t in L.TARGETS:
                v = [m[f'd_{g}_{t}'] for m in P if m[f'd_{g}_{t}'] is not None]; e[f'min_d_{g}_{t}_mm'] = min(v) * 1e3 if v else None
        e['robot_lowest_z_below_table_top_mm'] = max(0., max(m['table_top_z'] - m['robot_lowest_z'] for m in P)) * 1e3
        for k in L.CATEGORIES:
            e[f'steps_observed_contact_{k}'] = sum(m[f'observed_contact_{k}'] for m in P); e[f'steps_listed_proximity_only_{k}'] = sum(m[f'listed_{k}'] and not m[f'observed_contact_{k}'] for m in P)
        out['phases'][ph] = e
    out['spatial_items'] = L.spatial_items(b, bool(cell and cell.get('acted') and b['control_steps'] > 0)); return out


def cmd_metrics(a):
    out = guard(a.out, a.records, a.replay); t0 = time.time(); res = {}; errors = {}; rows_csv = []; sep = []; inputs = {}; geo_check = None
    if a.records:                                                        # a scene that failed in the run is reported, never dropped (review a5)
        for p in sorted((Path(a.records) / 'scenes').glob('*.error.json')): errors[p.name.split('.')[0]] = {'error': 'the run wrote an error record for this scene', 'file': p.name}
    expected = None; require_spatial = False; without = []; plan_arms = None
    if getattr(a, 'plan', None):                                         # review a9: the scenes the plan names; a missing record is an error
        pl = json.loads(Path(a.plan).read_text(encoding='utf-8')); lo, hi = pl['seed_registry']['range']; expected = set(range(int(lo), int(hi) + 1))
        require_spatial = 'spatial_logging' in pl                     # review a12: a plan with spatial logging requires the fields in every record
        plan_arms = list(pl['arm_registry']['arms']) if 'arm_registry' in pl else None
        found = {sd for _, sd in scenes_of(a)}
        for sd in sorted(expected - found):
            if str(sd) not in errors: errors[str(sd)] = {'error': 'no scene record for a scene of the plan'}
        for sd in sorted(found - expected): errors[str(sd)] = {'error': 'a scene record outside the plan\'s seed range (review a10)'}
    for stage, seed in scenes_of(a):
        try:
            if str(seed) in errors: continue                             # already an error (e.g. outside the plan); never audited
            sc_rows = []; sc_sep = []
            rec, tr, fr, meta = load(a, stage, seed); inputs[str(seed)] = {k: v for k, v in meta.items() if k not in ('original_traces', 'original_record')}; sc = {'stage': stage, 'source': meta['source'], 'arms': {}}
            if a.records:                                                # review a10: a completed record of this scene, of the given plan
                if rec.get('completed') is not True or rec.get('scene') != int(seed): raise ValueError(f'scene {seed}: not a completed record of this scene')
                if getattr(a, 'plan', None) and rec.get('plan_sha256') != sha(a.plan): raise ValueError(f'scene {seed}: the record belongs to another plan ({rec.get("plan_sha256")})')
            geo = (rec.get('initial_state') or {}).get('collision_geometry')
            if a.urdf and geo_check is None and geo: geo_check = L.geometry_against_urdf(geo, a.urdf)
            sc['requirements'] = L.requirements_v2(rec, tr)
            if require_spatial:                                          # review a14: every scene (also one that never triggered) carries the fields
                ini = rec.get('initial_state') or {}
                if not (set(L.SP.ADDED_FIELDS) <= set(ini) and L.has_spatial(tr.get('trunk') or [])): raise ValueError(f'scene {seed}: the plan logs spatial fields, the initial state or the trunk has none')
            arms_rec = [c.get('arm') for c in rec.get('cells') or []]      # review a16: the record's arms, complete and once each
            if a.records and (not arms_rec or len(set(arms_rec)) != len(arms_rec) or arms_rec[0] != 'C0' or (plan_arms is not None and arms_rec != plan_arms)):
                raise ValueError(f'scene {seed}: the record does not list the planned arms once each ({arms_rec})')
            missing = [c['arm'] for c in rec['cells'] if c['arm'] not in tr] if rec.get('triggered') else []
            if a.records and not missing: check_coverage(rec, tr)
            if missing: raise ValueError(f'scene {seed}: no trace for the recorded arms {missing}')       # an arm is never audited away (review a8)
            for arm in arms_of(rec, tr):
                b = L.branch(rec, tr, arm); cell = T.cell(rec, arm) if (rec.get('triggered') and arm != 'trunk') else None
                if b is None:
                    if require_spatial: raise ValueError(f'scene {seed} {arm}: the plan logs spatial fields, the record has none')
                    sc['arms'][arm] = {'status': 'NOT_OBSERVABLE', 'why': 'no spatial fields'}; without.append(f'{seed}/{arm}'); continue
                sc['arms'][arm] = summarise(b, cell)
                if rec.get('triggered') and arm != 'trunk' and meta.get('original_traces') is not None and arm in meta['original_traces']:
                    sc['arms'][arm]['E1_E10_original_run_unchanged_audit'] = e_status(SA.expected_vs_actual(meta['original_record'], meta['original_traces'], arm))
                for m in b['metrics']:
                    sc_rows.append([seed, arm] + [m[k] for k in STEP_KEYS]); sc_sep += [{'scene': seed, 'arm': arm, 't': m['t'], **s} for s in m['separation_vs_reconstruction']]
            rows_csv += sc_rows; sep += sc_sep                              # review a13: a scene enters the aggregates only when every arm succeeded
            res[str(seed)] = sc; print(json.dumps({'scene': seed, 'arms': list(sc['arms'])}), flush=True)
        except Exception as e:
            import traceback
            errors[str(seed)] = {'error': repr(e), 'traceback': traceback.format_exc()}; print(f'ERROR scene {seed}: {e!r}', flush=True)
    with gzip.open(out / 'per_step_spatial_metrics.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['scene', 'arm'] + STEP_KEYS); w.writerows(rows_csv)
    s_obs = [x for x in sep if x['observed']]; s_sep = [x for x in sep if x['physx_min_separation_m'] > 0]          # the filter of S6 (review a7)
    val = {'observed_shape_pair_steps': len(s_obs), 'observed_with_reconstructed_gap_above_1mm': sum(x['reconstructed_m'] > L.CONTACT_GEOM_TOL_M for x in s_obs),
           'observed_reconstructed_gap_mm': pct([x['reconstructed_m'] * 1e3 for x in s_obs]), 'separated_shape_pair_steps': len(s_sep),
           'abs_physx_separation_minus_reconstruction_mm': pct([abs(x['physx_min_separation_m'] - x['reconstructed_m']) * 1e3 for x in s_sep]),
           'by_pair_kind': {k: pct([abs(x['physx_min_separation_m'] - x['reconstructed_m']) * 1e3 for x in s_sep if L.category(x['a'], x['b']) == k]) for k in sorted({L.category(x['a'], x['b']) for x in s_sep})}}
    jdump(out / 'spatial_summary.json', res); jdump(out / 'contact_geometry_validation_v2.json', val); jdump(out / 'geometry_against_urdf.json', geo_check)
    if errors: jdump(out / 'errors.json', errors)
    # no scene read: INCOMPLETE (review a3)
    st_ = 'INCOMPLETE' if (errors or not res) else ('COMPLETE' if (expected is not None or not a.records) else 'COMPLETE_FOR_THE_SCENES_FOUND (no --plan: the expected scene set was not checked)')
    if without and st_ != 'INCOMPLETE': st_ += f' | WITHOUT_SPATIAL_FIELDS: {len(without)} arm branches NOT_OBSERVABLE (records written without hci.spatial_probe)'
    man = {'status': st_, 'scenes': len(res), 'expected_scenes': None if expected is None else len(expected), 'inputs': inputs, 'outputs': {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}, 'seconds': time.time() - t0, 'label': 'post-hoc; stored records only'}
    jdump(out / 'MANIFEST.json', man); print(json.dumps({'status': man['status'], 'scenes': len(res), 'errors': len(errors), 'validation': {k: v for k, v in val.items() if k != 'by_pair_kind'}}, default=float))
    return 3 if (errors or not res) else 0


def e_status(ev):
    """The E1-E10 statuses of the unchanged hci.spatial_audit.expected_vs_actual (its other entries are context)."""
    items = ev.get('expected', ev)                                      # the E items sit under 'expected'; the other keys are context
    out = {k: v['status'] for k, v in items.items() if k[:1] == 'E' and isinstance(v, dict) and 'status' in v}
    if not out: raise ValueError('no E1-E10 status found in the output of hci.spatial_audit.expected_vs_actual')
    return out


def pct(v):
    if not v: return None
    v = np.asarray(v, float); return {'n': int(len(v)), 'min': float(v.min()), 'median': float(np.median(v)), 'p99': float(np.percentile(v, 99)), 'max': float(v.max())}


# ---------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt': 'c1', 'font.size': 8}); return plt


def lims_of(rows):
    B = np.array([r['cubeB'][:2] for r in rows]); A = np.array([r['cubeA'][:2] for r in rows]); P = np.concatenate([A, B]); c = P.mean(0); h = max(.07, float(np.abs(P - c).max()) + .05)
    return (c[0] - h, c[0] + h), (c[1] - h, c[1] + h)


def draw_xy(ax, rows, geo, t, lims, cmds):
    """Panel B: per-shape XY footprints of every robot shape and both cubes at step t; TCP path up to t; commanded targets."""
    r = next(x for x in rows if x['t'] == t); W = L.world(r, geo)
    for n, S in W.items():
        g = L.group_of(n); col = GC.get(g, None)
        for s in S:
            P = G.hull2d(s.V[:, :2]) * 1e3
            if n == 'cubeB': ax.fill(*P.T, fc='#1e8449', alpha=.35, ec='#1e8449', lw=1.2)
            elif n == 'cubeA': ax.fill(*P.T, fc='#c0392b', alpha=.35, ec='#c0392b', lw=1.2)
            elif n == 'table': continue
            elif col: ax.fill(*P.T, fc=col, alpha=.12 if g == 'arm_links' else .25, ec=col, lw=.6)
    path = np.array([x['tcp'][:2] for x in rows if x['t'] <= t]) * 1e3; ax.plot(*path.T, '-', color='k', lw=1)
    for s, c in cmds:
        if s <= t: ax.plot(c[0] * 1e3, c[1] * 1e3, '+', color='#1f5fbf', ms=5, mew=1)
    ax.set_xlim(lims[0][0] * 1e3, lims[0][1] * 1e3); ax.set_ylim(lims[1][0] * 1e3, lims[1][1] * 1e3); ax.set_aspect('equal'); ax.set_xlabel('x [mm]', fontsize=7); ax.set_ylabel('y [mm]', fontsize=7)


def draw_3d(ax, rows, geo, t, lims):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    r = next(x for x in rows if x['t'] == t); W = L.world(r, geo); (x0, x1), (y0, y1) = lims
    for n, S in W.items():
        if n == 'table': continue
        g = L.group_of(n); col = {'cubeA': '#c0392b', 'cubeB': '#1e8449'}.get(n, GC.get(g, '#999'))
        for s in S:
            if s.V[:, 0].max() < x0 - .05 or s.V[:, 0].min() > x1 + .05 or s.V[:, 1].max() < y0 - .05 or s.V[:, 1].min() > y1 + .05: continue
            ax.add_collection3d(Poly3DCollection(s.V[s.T] * 1e3, facecolor=col, edgecolor='none', alpha=.55 if n in ('cubeA', 'cubeB') else .18))
    path = np.array([x['tcp'] for x in rows if x['t'] <= t]) * 1e3; ax.plot(*path.T, '-', color='k', lw=1)
    ax.set_xlim(x0 * 1e3, x1 * 1e3); ax.set_ylim(y0 * 1e3, y1 * 1e3); ax.set_zlim(0, 220); ax.set_box_aspect(((x1 - x0), (y1 - y0), .22)); ax.view_init(22, -60); ax.tick_params(labelsize=5)


SER = (('d_fingers_cubeB', 'fingers', GC['fingers']), ('d_hand_cubeB', 'hand', GC['hand']), ('d_camera_cubeB', 'camera', GC['camera']), ('d_arm_links_cubeB', 'arm links', GC['arm_links']))
CONTACT_ROWS = ('fingers-cubeB', 'hand-cubeB', 'camera-cubeB', 'arm_links-cubeB', 'cubeA-cubeB', 'fingers-table', 'hand-table', 'arm_links-table', 'cubeB-table')


def timeline(axs, M, rows, rec, arm, t=None, xlim=None):
    """Panel D: gaps robot -> target, observed contacts (filled) vs listed proximity (hollow), target displacement and
    XY offset, commanded steps; trigger, end of move, first policy inference and first open command."""
    ts = [m['t'] for m in M]
    for k, lab, col in SER: axs[0].plot(ts, [None if m[k] is None else min(60., m[k] * 1e3) for m in M], color=col, lw=1, label=lab)
    axs[0].set_ylabel('gap to target\n[mm, cap 60]', fontsize=6.5); axs[0].axhline(0, color='r', lw=.6); axs[0].legend(fontsize=4.8, loc='center right', ncol=2, handlelength=1)
    for i, k in enumerate(CONTACT_ROWS):
        o = [m['t'] for m in M if m[f'observed_contact_{k}']]; p = [m['t'] for m in M if m[f'listed_{k}'] and not m[f'observed_contact_{k}']]
        axs[1].plot(o, [i] * len(o), 's', ms=2.6, color='#c0392b'); axs[1].plot(p, [i] * len(p), 's', ms=2.6, mfc='none', mec='#aaa', mew=.5)
    axs[1].set_yticks(range(len(CONTACT_ROWS))); axs[1].set_yticklabels(CONTACT_ROWS, fontsize=5.2); axs[1].set_ylim(-.7, len(CONTACT_ROWS) - .3)
    ini = rec['initial_state']; B0 = np.asarray(ini['cubeB'][:2])
    axs[2].plot([r['t'] for r in rows], [np.hypot(*(np.asarray(r['cubeB'][:2]) - B0)) * 1e3 for r in rows], color='#1e8449', lw=1, label='target moved')
    axs[2].plot([r['t'] for r in rows], [np.hypot(r['cubeB'][0] - r['cubeA'][0], r['cubeB'][1] - r['cubeA'][1]) * 1e3 for r in rows], color='#c0392b', lw=1, label='XY offset held-target')
    axs[2].set_ylabel('[mm]', fontsize=6.5); axs[2].legend(fontsize=5, loc='upper right')
    c = T.cell(rec, arm); trig = rec['trigger']['step']; steps = ((c or {}).get('control') or {}).get('steps') or []
    axs[3].bar([trig + 1 + i for i in range(len(steps))], [float(np.hypot(*s['commanded_xy_m'])) * 1e3 if s.get('commanded_xy_m') else 0. for s in steps], color='#1f5fbf', width=.8)
    axs[3].set_ylabel('commanded\nstep [mm]', fontsize=6.5); axs[3].set_xlabel('simulation step', fontsize=6.5)
    ev = {'trigger': trig, 'end of move': trig + (c['control_steps'] if c and c.get('acted') else 0), 'policy resumes': (c or {}).get('first_inference_step'), 'first open': (c or {}).get('first_open_command_after_move')}
    for ax in axs:
        for k, v in ev.items():
            if v is not None: ax.axvline(v, color={'trigger': 'k', 'end of move': '#1f5fbf', 'policy resumes': '#7d3c98', 'first open': '#e67e22'}[k], lw=.7, ls='--')
        if t is not None: ax.axvline(t, color='r', lw=1)
        ax.tick_params(labelsize=5.5); ax.set_xlim(*(xlim or (ts[0] - .5, ts[-1] + .5)))          # one step range for all four panels
    return ev


def commands(rec, arm, rows):
    c = T.cell(rec, arm); trig = rec['trigger']['step']; out = []; byt = {r['t']: r for r in rows}
    for i, s in enumerate(((c or {}).get('control') or {}).get('steps') or []):
        r = byt.get(trig + i)
        if r is not None and s.get('commanded_xy_m'): out.append((trig + i + 1, (r['tcp'][0] + s['commanded_xy_m'][0], r['tcp'][1] + s['commanded_xy_m'][1])))
    return out


def frame(rec, fr, arm, t):
    from hci import spatial_figures as F
    return F.frame_at(rec, fr, arm, t)


def cmd_cases(a):
    out = guard(a.out, a.records, a.replay); plt = _plt(); from PIL import Image; allp = {}; inputs = {}; skipped = []
    for name, stage, seed, arms, title in CASES:
        rec, tr, fr, meta = load(a, stage, seed); inputs[name] = {k: v for k, v in meta.items() if k not in ('original_traces', 'original_record')}; cdir = out / name; cdir.mkdir(); P = {'case': name, 'scene': seed, 'source': meta['source'], 'arms': {}}
        B = {arm: L.branch(rec, tr, arm) for arm in arms}
        if any(b is None for b in B.values()): jdump(cdir / 'NOT_OBSERVABLE.json', {'why': 'record without spatial fields'}); skipped.append(name); continue
        geo = rec['initial_state']['collision_geometry']; trig = rec['trigger']['step']; lims = lims_of([r for b in B.values() for r in b['rows']])
        for arm in arms:
            c = T.cell(rec, arm); P['arms'][arm] = {'summary': summarise(B[arm], c), 'E1_E10_original_run_unchanged_audit': e_status(SA.expected_vs_actual(meta['original_record'], meta['original_traces'], arm)) if meta.get('original_traces') else None}
        # B and C at the end of the move (or the trigger + 3), D over the branch
        fig = plt.figure(figsize=(6.2 * len(arms), 9.6), dpi=110)
        for i, arm in enumerate(arms):
            b = B[arm]; q = trig + max(b['control_steps'], 3); q = min(q, b['rows'][-1]['t'])
            ax = fig.add_subplot(3, len(arms), i + 1); draw_xy(ax, b['rows'], geo, q, lims, commands(rec, arm, b['rows'])); ax.set_title(f'B {NAME[arm]}: XY occupancy at step {q} (true scale)', fontsize=8, loc='left', color=ARM_C[arm])
            ax = fig.add_subplot(3, len(arms), len(arms) + i + 1, projection='3d'); draw_3d(ax, b['rows'], geo, q, lims); ax.set_title(f'C {NAME[arm]}: 3D collision shapes at step {q}', fontsize=8, loc='left', color=ARM_C[arm])
        fig.suptitle(f"scene {seed} | {title}\n{meta['source']}\n{POSTHOC}. Shapes: the simulator's own collision shapes (arm links grey, hand dark, camera purple, cubes red/green).", fontsize=7.5)
        fig.savefig(cdir / 'occupancy_xy_3d.png', metadata={'Software': None}); plt.close(fig)
        fig, axs = plt.subplots(4, len(arms), figsize=(6.2 * len(arms), 7.5), dpi=110, sharex='col'); axs = np.atleast_2d(axs).reshape(4, len(arms))
        for i, arm in enumerate(arms):
            P['arms'][arm]['events'] = timeline(list(axs[:, i]), B[arm]['metrics'], B[arm]['rows'], rec, arm); axs[0, i].set_title(f'D {NAME[arm]}', fontsize=8, loc='left', color=ARM_C[arm])
        axs[0, 0].legend(fontsize=5.5, loc='upper right'); fig.suptitle(f'scene {seed}: physical behaviour (filled = observed contact, non-zero impulse; hollow = listed by PhysX within the contact offset, zero impulse)\n{POSTHOC}', fontsize=7.5)
        fig.tight_layout(); fig.savefig(cdir / 'timeline.png', metadata={'Software': None}); plt.close(fig)
        frames, shown = gif_frames(plt, rec, fr, arms, B, geo, lims, title, meta)
        ims = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
        ev = {v for arm in arms for v in P['arms'][arm]['events'].values() if v is not None}; dur = [900 if s in ev else 200 for s, _ in shown]; dur[-1] = 2200
        ims[0].save(cdir / 'multipanel.gif', save_all=True, append_images=ims[1:], duration=dur, loop=0, optimize=True)
        P['gif_steps'] = [s for s, _ in shown]; P['gif_rgb_provenance'] = [p for _, p in shown]; jdump(cdir / 'plotted.json', P)
        allp[name] = {p.name: sha(p) for p in sorted(cdir.iterdir())}; print(name, len(frames), 'frames', round((cdir / 'multipanel.gif').stat().st_size / 1e6, 2), 'MB', flush=True)
    jdump(out / 'MANIFEST.json', {'inputs': inputs, 'outputs': allp, 'skipped_cases_without_spatial_fields': skipped, 'status': 'INCOMPLETE' if skipped else 'COMPLETE'})
    return 3 if skipped else 0                                          # review a12: a requested case that could not be drawn is not success


def gif_frames(plt, rec, fr, arms, B, geo, lims, title, meta, W=1600, H=980, DPI=100):
    trig = rec['trigger']['step']; ends = []
    for arm in arms:
        c = T.cell(rec, arm); ends.append(max(trig + (c['control_steps'] if c.get('acted') else 0), (c.get('first_open_command_after_move') or trig) ) + 12)
    ta = max(B[arms[0]]['rows'][0]['t'], trig - 6); tb = min(min(b['rows'][-1]['t'] for b in B.values()), max(ends)); frames = []; shown = []
    for t in range(ta, tb + 1):
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI); fig.patch.set_facecolor('white')
        fig.text(.01, .985, f"scene {rec['scene']} | {title}", fontsize=9.5, weight='bold', va='top')
        fig.text(.01, .962, f"simulation step t = {t} (trigger {trig}); every panel shows step t.  A: stored RGB of the original run only.  B, C, D: {POSTHOC}.", fontsize=7.5, va='top', color='#7b1e1e')
        fig.text(.01, .943, meta['source'], fontsize=7, va='top', color='#555')
        fig.text(.01, .926, 'B, C colours: held cube red, target green, fingers and hand dark grey, wrist camera purple, arm links light grey; black = TCP path up to t; blue + = commanded targets.  '
                 'D: dashed lines = trigger (black), end of move (blue), policy resumes (purple), first open (orange); red = step t.', fontsize=6.8, va='top', color='#333')
        prov = []
        for i, arm in enumerate(arms):
            y = .50 - i * .44; img, why = frame(rec, fr, arm, t); prov.append(why); ax = fig.add_axes([.01, y + .06, .27, .30])
            if img is None: ax.imshow(np.full((128, 256, 3), 175, np.uint8)); ax.text(128, 64, 'NO RGB RECORDED\nfor this arm and step', ha='center', va='center', fontsize=9, weight='bold')
            else: ax.imshow(img, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([]); [s.set_color(ARM_C[arm]) or s.set_linewidth(2.5) for s in ax.spines.values()]
            ax.set_title(f'A{i + 1} {NAME[arm]}: base | wrist camera', fontsize=8, loc='left', color=ARM_C[arm], weight='bold'); ax.text(0, 142, why, fontsize=6.3, va='top')
            rows = B[arm]['rows']; tt = min(t, rows[-1]['t'])
            bx = fig.add_axes([.30, y + .02, .20, .36]); draw_xy(bx, rows, geo, tt, lims, commands(rec, arm, rows)); bx.set_title(f'B{i + 1} XY occupancy (true scale)', fontsize=8, loc='left'); bx.tick_params(labelsize=5.5)
            cx = fig.add_axes([.50, y + .02, .22, .38], projection='3d'); draw_3d(cx, rows, geo, tt, lims); cx.set_title(f'C{i + 1} 3D collision shapes', fontsize=8, loc='left')
            axs = [fig.add_axes([.765, y + .30 - j * .085, .225, .07]) for j in range(4)]; timeline(axs, B[arm]['metrics'], rows, rec, arm, t=t, xlim=(ta - .5, tb + .5))
            for ax in axs[:-1]: ax.set_xticklabels([])
            axs[0].set_title(f'D{i + 1} {NAME[arm]}', fontsize=7.5, loc='left', color=ARM_C[arm])
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig); shown.append((t, prov))
    return frames, shown


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest='cmd', required=True)
    for n in ('metrics', 'cases'):
        q = s.add_parser(n); q.add_argument('--out', required=True); g = q.add_mutually_exclusive_group(required=True); g.add_argument('--records'); g.add_argument('--replay')
        q.add_argument('--phase-b', default=str(REPO / 'runs' / 'vision_center_v2_001' / 'server_backup' / 'extracted' / 'runs' / 'vision_center_v2_001'))     # the full Phase B records, with frames
        q.add_argument('--urdf'); q.add_argument('--plan', help='C1 plan: its seed range is the expected scene set (records mode)')
    a = p.parse_args()
    if a.replay is None: a.phase_b = None
    raise SystemExit({'metrics': cmd_metrics, 'cases': cmd_cases}[a.cmd](a))


if __name__ == '__main__':
    main()
