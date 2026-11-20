"""Figures and the multi-panel GIF of the spatial audit (matplotlib Agg, PIL). Stored records only.

Fixed meaning of every colour and line style for the whole figure or GIF (never changed during playback):
  held cube (A)  red          target cube (B)  green        gripper (hand + fingers, collision shapes)  slate
  TCP path       black solid  commanded target (TCP before step + command)  blue x and arrow
  controller online estimate of the target centre  purple +   (an estimate, not ground truth)
  contact-risk zone  orange dashed = target footprint grown by PROXIMITY_M (exploratory)
  swept XY footprint of the finger boxes + held cube during the move  grey fill (interpolated sub-poses: an approximation;
                 the hand is drawn as an outline only, its swept area is in the metrics)
Every geometric quantity is POST-HOC GROUND TRUTH from the logged simulator state; RGB panels show stored frames only.
Each drawing function returns the numbers it drew ('plotted'), which a separate check compares with the raw trace.
"""
import numpy as np
from . import spatial_geometry as G
from . import spatial_audit as SA
from . import trajectory_audit as T

COL = {'A': '#c0392b', 'B': '#1e8449', 'grip': '#34495e', 'tcp': '#000000', 'cmd': '#1f5fbf', 'est': '#7d3c98', 'risk': '#e67e22', 'swept': '#95a5a6', 'held_est': '#d35400'}
NAME = {'C0': 'C0 (no intervention)', 'VISION_CENTER_V1': 'v1', 'VISION_CENTER_V2': 'v2', 'CENTER_ORACLE': 'oracle (privileged reference)'}
POSTHOC = 'geometry = POST-HOC GROUND TRUTH from the logged state (not a controller input)'


def _plt():
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'svg.hashsalt': 'spatial-audit', 'path.simplify': False})
    return plt


def rows_of(rec, tr, arm):
    t0 = rec['trigger']['step']; return {r['t']: r for r in tr['trunk'] if r['t'] <= t0} | {r['t']: r for r in tr[arm]}


def key_steps(rec, tr, arm):
    c = T.cell(rec, arm); t0 = rec['trigger']['step']; s = T.series(rec, tr, arm); ev = T.events(rec, s, arm); k = c['control_steps'] if c.get('acted') else 0
    out = {'trigger': t0, 'end_of_move': t0 + k if k else None, 'first_open': ev['release_cmd'], 'last': int(s['t'][-1])}
    return {n: v for n, v in out.items() if v is not None}


def dilate(poly, r, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False); circ = np.stack([np.cos(a), np.sin(a)], 1) * r
    return G.hull2d((np.asarray(poly)[:, None, :] + circ[None]).reshape(-1, 2))


def _poly(ax, P, s=1e3, **kw):
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(np.asarray(P) * s, closed=True, **kw))


def swept_footprint(rec, tr, arm):
    """Polygons whose union is the XY swept footprint of the finger collision boxes + held cube at interpolated sub-poses
    of the move (each box separately; no hull across the gap between the fingers)."""
    c = T.cell(rec, arm)
    if not (c.get('acted') and c['control_steps']): return []
    t0 = rec['trigger']['step']; R = rows_of(rec, tr, arm); out = []
    for t in range(t0 + 1, t0 + c['control_steps'] + 1):
        for f in np.arange(1, G.SIM_STEPS_PER_CONTROL + 1) / G.SIM_STEPS_PER_CONTROL:
            S = SA.shapes(SA._interp_row(R[t - 1], R[t], f)); out += [(t, G.footprint(s)) for s in S['fingerL'] + S['fingerR']] + [(t, G.footprint(S['A']))]   # (step, polygon)
    return out


def estimates(rec, arm):
    c = T.cell(rec, arm); rd = ((c or {}).get('control') or {}).get('readings') or []
    return [(r['t'], r['target_xy'], r.get('held_xyz'), r.get('offset_norm_m')) for r in rd if r.get('target_xy') is not None]


def commanded(rec, tr, arm):
    """(step, TCP xy before the step, commanded target xy) per control step. Commands are per-step delta targets from the
    current TCP, so they are drawn as arrows from where the TCP was, never summed into a path."""
    c = T.cell(rec, arm); R = rows_of(rec, tr, arm); t0 = rec['trigger']['step']; out = []
    for j, x in enumerate(((c.get('control') or {}).get('steps') or []) if c.get('acted') else []):
        b = np.array(R[t0 + j]['tcp'][:2]); out.append((t0 + j + 1, b.tolist(), (b + np.array(x['commanded_xy_m'])).tolist()))
    return out


def view_limits(rec, tr, arms, pad=0.035):
    t0 = rec['trigger']['step']; P = []
    for a in arms:
        R = rows_of(rec, tr, a)
        for t, r in R.items():
            if t >= t0 - 3: P += [r['cubeA'][:2], r['cubeB'][:2], r['tcp'][:2]]
    P = np.array(P); c = (P.min(0) + P.max(0)) / 2; h = (P.max(0) - P.min(0)).max() / 2 + pad
    return (c[0] - h, c[0] + h, c[1] - h, c[1] + h)


# ---------------------------------------------------------------- top view
def draw_topdown(ax, rec, tr, arm, t, lims, swept=None, upto=True):
    """Top view at step t (or the key steps when t is None). Returns the plotted numbers."""
    R = rows_of(rec, tr, arm); t0 = rec['trigger']['step']; ks = key_steps(rec, tr, arm); plotted = {'arm': arm, 't': t}
    t_last = max(R) if t is None else min(t, max(R)); ts = [q for q in sorted(R) if min(t0 - 3, t_last) <= q <= t_last]
    if swept:
        for q, P in swept:
            if t is None or q <= t: _poly(ax, P, fc=COL['swept'], ec='none', alpha=.06)       # never ahead of the step shown
    r_now = R[t_last]; S = SA.shapes(r_now); fb = G.footprint(S['B'])
    _poly(ax, dilate(fb, SA.PROXIMITY_M), fill=False, ec=COL['risk'], ls='--', lw=1.1)
    Bi = rec.get('initial_state');
    if Bi: _poly(ax, G.footprint(G.cube(Bi['cubeB'], Bi['cubeB_q'])), fill=False, ec=COL['B'], ls=':', lw=1)
    shown = [t_last] if t is not None else [v for v in ks.values()]
    for q in shown:
        Sq = SA.shapes(R[q]); a = 1. if q == t_last else .45
        _poly(ax, G.footprint(Sq['B']), fc=COL['B'], ec=COL['B'], alpha=.18 * a, lw=1.2)
        _poly(ax, G.footprint(Sq['A']), fc=COL['A'], ec=COL['A'], alpha=.25 * a, lw=1.2)
        for s in Sq['fingerL'] + Sq['fingerR']: _poly(ax, G.footprint(s), fill=False, ec=COL['grip'], lw=.9 if q == t_last else .5, alpha=a)
        if q == t_last: _poly(ax, G.footprint(Sq['hand']), fill=False, ec=COL['grip'], lw=.6, ls=':', alpha=.8)
    tcp = np.array([R[q]['tcp'][:2] for q in ts]); ax.plot(tcp[:, 0] * 1e3, tcp[:, 1] * 1e3, '-', color=COL['tcp'], lw=1.2)
    Bc = np.array([R[q]['cubeB'][:2] for q in ts]); ax.plot(Bc[-1, 0] * 1e3, Bc[-1, 1] * 1e3, 'o', color=COL['B'], ms=3.5)
    plotted.update(tcp_xy=tcp.tolist(), target_xy_true=Bc[-1].tolist(), steps=ts, footprint_B=fb.tolist(), footprint_A=G.footprint(S['A']).tolist())
    cm = [x for x in commanded(rec, tr, arm) if t is None or x[0] <= t]
    for q, b, g in cm:
        ax.annotate('', xy=(g[0] * 1e3, g[1] * 1e3), xytext=(b[0] * 1e3, b[1] * 1e3), arrowprops=dict(arrowstyle='->', color=COL['cmd'], lw=.8))
        ax.plot(g[0] * 1e3, g[1] * 1e3, 'x', color=COL['cmd'], ms=4)
    plotted['commanded'] = cm
    es = [e for e in estimates(rec, arm) if t is None or e[0] <= t]
    for q, xy, held, _ in es: ax.plot(xy[0] * 1e3, xy[1] * 1e3, '+', color=COL['est'], ms=7, mew=1.4)
    plotted['estimated_target_xy'] = [(q, xy) for q, xy, _, _ in es]
    if 'end_of_move' in ks and (t is None or t >= ks['end_of_move']):
        e = np.array(R[ks['end_of_move']]['tcp'][:2]) * 1e3; ax.plot(*e, 's', mfc='white', mec=COL['tcp'], ms=5)
    s0 = np.array(R[t0]['tcp'][:2]) * 1e3; ax.plot(*s0, 'o', mfc='white', mec=COL['tcp'], ms=5)
    ax.set_xlim(lims[0] * 1e3, lims[1] * 1e3); ax.set_ylim(lims[2] * 1e3, lims[3] * 1e3); ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.25)
    ax.set_xlabel('world x [mm]'); ax.set_ylabel('world y [mm]')
    return plotted


def topdown_legend(ax):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(fc=COL['A'], alpha=.35, label='held cube footprint'), Patch(fc=COL['B'], alpha=.3, label='target footprint'), Line2D([], [], color=COL['B'], ls=':', label='target at episode start'),
         Line2D([], [], color=COL['grip'], label='finger footprints (collision boxes)'), Line2D([], [], color=COL['grip'], ls=':', lw=.6, label='hand footprint (collision hull)'), Line2D([], [], color=COL['risk'], ls='--', label=f'risk zone: target + {SA.PROXIMITY_M * 1e3:.0f} mm (exploratory)'),
         Patch(fc=COL['swept'], alpha=.4, label='swept footprint, fingers + held cube (interpolated)'), Line2D([], [], color=COL['tcp'], label='TCP path (o start, square end of move)'),
         Line2D([], [], color=COL['cmd'], marker='x', ls='', label='commanded target per step (arrow from TCP)'), Line2D([], [], color=COL['est'], marker='+', ls='', ms=7, label='controller estimate of target centre')]
    ax.legend(handles=h, fontsize=6.3, loc='upper left', framealpha=.92)


# ---------------------------------------------------------------- 3D
def _wire(ax, S, color, lw=.8, alpha=1.):
    for e in S.E:
        P = S.V[e] * 1e3; ax.plot(P[:, 0], P[:, 1], P[:, 2], color=color, lw=lw, alpha=alpha)


def _box_edges(ax, S, color, lw=1., alpha=1.):
    """Only the 12 box edges (not the triangulation diagonals)."""
    V = S.V * 1e3
    for i in range(8):
        for b in (1, 2, 4):
            j = i ^ b
            if j > i: ax.plot(*V[[i, j]].T, color=color, lw=lw, alpha=alpha)


def draw_3d(ax, rec, tr, arm, t, lims, z_range=(0, 140)):
    R = rows_of(rec, tr, arm); t0 = rec['trigger']['step']; t_last = min(t, max(R)) if t is not None else max(R); ts = [q for q in sorted(R) if min(t0 - 3, t_last) <= q <= t_last]
    S = SA.shapes(R[t_last]); m = SA.row_metrics(R[t_last])
    _box_edges(ax, S['A'], COL['A'], 1.2); _box_edges(ax, S['B'], COL['B'], 1.2)
    for g in S['gripper'][1:]: _box_edges(ax, g, COL['grip'], .6)
    _wire(ax, S['hand'], COL['grip'], lw=.25, alpha=.25)
    tcp = np.array([R[q]['tcp'] for q in ts]) * 1e3; A = np.array([R[q]['cubeA'] for q in ts]) * 1e3; B = np.array([R[q]['cubeB'] for q in ts]) * 1e3
    ax.plot(*tcp.T, color=COL['tcp'], lw=1.1); ax.plot(*A.T, color=COL['A'], lw=1., ls='--'); ax.plot(*B.T, color=COL['B'], lw=1., ls='--')
    ax.set_xlim(lims[0] * 1e3, lims[1] * 1e3); ax.set_ylim(lims[2] * 1e3, lims[3] * 1e3); ax.set_zlim(*z_range)
    try: ax.set_box_aspect(((lims[1] - lims[0]) * 1e3, (lims[3] - lims[2]) * 1e3, z_range[1] - z_range[0]))
    except Exception: pass
    ax.set_xlabel('x [mm]', labelpad=-6); ax.set_ylabel('y [mm]', labelpad=-6); ax.set_zlabel('z [mm]', labelpad=-6); ax.tick_params(labelsize=5.5, pad=-2); ax.view_init(elev=22, azim=-58)
    dmin = min(m['d_fingerL_B'], m['d_fingerR_B'], m['d_hand_B'])
    txt = (f"min surface gap  held-target {m['d_A_B'] * 1e3:+.1f} mm\n  gripper-target {dmin * 1e3:+.1f} mm\n"
           f"height gap (A bottom - B top) {m['height_gap_Abottom_Btop'] * 1e3:+.1f} mm")
    return {'arm': arm, 't': t_last, 'tcp_mm': tcp.tolist(), 'A_mm': A.tolist(), 'B_mm': B.tolist(), 'd_A_B': m['d_A_B'], 'd_gripper_B': dmin, 'height_gap': m['height_gap_Abottom_Btop'], 'text': txt}


# ---------------------------------------------------------------- time series
def series_data(rec, tr, arm, metrics):
    s = T.series(rec, tr, arm); t0 = rec['trigger']['step']; ts = [m['t'] for m in metrics]; idx = [T.at(s, q) for q in ts]
    off = np.linalg.norm(T.offset_xy(s), axis=1)[idx]; Bd = np.array([m.get('B_disp_xy_from_initial', np.nan) for m in metrics])
    est = [(q, o) for q, _, _, o in estimates(rec, arm) if o is not None]
    c = T.cell(rec, arm); k = c['control_steps'] if c.get('acted') else 0
    R = rows_of(rec, tr, arm); act = np.array([np.linalg.norm(np.subtract(R[q]['tcp'][:2], R[q - 1]['tcp'][:2])) if q - 1 in R else np.nan for q in ts])
    cmd = {q: float(np.linalg.norm(np.subtract(g, b))) for q, b, g in commanded(rec, tr, arm)}
    return {'t': ts, 'offset_true_m': off.tolist(), 'offset_est': est, 'B_disp_xy_m': Bd.tolist(), 'width_m': s['width'][idx].tolist(), 'open_cmd': (s['action'][idx, -1] > 0).tolist(),
            'd_A_B': [m['d_A_B'] for m in metrics], 'd_fingers_B': [min(m['d_fingerL_B'], m['d_fingerR_B']) for m in metrics], 'd_hand_B': [m['d_hand_B'] for m in metrics],
            'obs_A_B': [m['observed_contact_A_B'] for m in metrics],                                    # True / False / None (not recorded)
            'obs_fingers_B': [None if (m['observed_contact_fingerL_B'] is None or m['observed_contact_fingerR_B'] is None) else bool(m['observed_contact_fingerL_B'] or m['observed_contact_fingerR_B']) for m in metrics],
            'potential_any': [any(m[f'potential_collision_{p}'] for p in ('fingerL_B', 'fingerR_B', 'hand_B')) for m in metrics],
            'tcp_step_actual_m': act.tolist(), 'tcp_step_commanded_m': cmd, 'control': (t0, t0 + k)}


def draw_series(axs, D, arm, color, t=None, label=None):
    """Four stacked axes: XY offset, clearances to the target, target displacement and finger opening, step sizes."""
    ts = np.array(D['t']); m = ts <= (t if t is not None else ts[-1]); L = label or NAME.get(arm, arm)
    a0, a1, a2, a3 = axs
    a0.plot(ts[m], np.array(D['offset_true_m'])[m] * 1e3, color=color, lw=1.4, label=f'{L} true')
    e = [(q, o) for q, o in D['offset_est'] if t is None or q <= t]
    if e: a0.plot(*zip(*[(q, o * 1e3) for q, o in e]), '+', color=color, ms=6, mew=1.2, label=f'{L} online estimate')
    a1.plot(ts[m], np.array(D['d_A_B'])[m] * 1e3, color=color, lw=1.3, label=f'{L} held-target')
    a1.plot(ts[m], np.array(D['d_fingers_B'])[m] * 1e3, color=color, lw=1., ls='--', label=f'{L} fingers-target')
    a1.plot(ts[m], np.array(D['d_hand_B'])[m] * 1e3, color=color, lw=.9, ls=':', label=f'{L} hand-target')
    isT = lambda v: np.array([x is True for x in v], bool); isN = lambda v: np.array([x is None for x in v], bool)
    ob = ts[m][isT(D['obs_A_B'])[m]]; a1.plot(ob, np.full(len(ob), -4.), '|', color=color, ms=5)
    of = ts[m][isT(D['obs_fingers_B'])[m]]; a1.plot(of, np.full(len(of), -7.), 'v', color='#b03a2e', ms=4)
    nr = ts[m][(isN(D['obs_A_B']) | isN(D['obs_fingers_B']))[m]]; a1.plot(nr, np.full(len(nr), -2.), '.', color='#999', ms=2)    # contact not recorded
    pc = ts[m][np.array(D['potential_any'])[m]]; a1.plot(pc, np.full(len(pc), -10.), 'd', color=COL['risk'], ms=3)
    a2.plot(ts[m], np.array(D['B_disp_xy_m'])[m] * 1e3, color=color, lw=1.3); a2b = a2
    a3.plot(ts[m], np.array(D['tcp_step_actual_m'])[m] * 1e3, color=color, lw=1.1)
    cq = [(q, v * 1e3) for q, v in sorted(D['tcp_step_commanded_m'].items()) if t is None or q <= t]
    if cq: a3.plot(*zip(*cq), 'x', color=color, ms=4)
    c0, c1 = D['control']
    if c1 > c0 and (t is None or t > c0):
        for ax in axs: ax.axvspan(c0 + .5, min(c1, t if t is not None else c1) + .5, color=color, alpha=.08, lw=0)
    return {'t': ts[m].tolist()}


def series_axes_style(axs, xlim, now=None, trig=None):
    a0, a1, a2, a3 = axs
    a0.axhline(T.RESIDUAL_TOL_M * 1e3, color='#888', lw=.7, ls=':'); a0.set_ylabel('XY offset\n[mm]')
    a1.axhline(0, color='#555', lw=.6); a1.axhline(SA.PROXIMITY_M * 1e3, color=COL['risk'], lw=.7, ls='--'); a1.set_ylabel('surface gap\nto target [mm]'); a1.set_ylim(-12, 40)
    a1.text(xlim[0] + .3, -12, '| v recorded contact   diamond potential (geometry)   grey dot: contact not recorded', fontsize=5.6, va='bottom', color='#333')
    a2.set_ylabel('target moved\nfrom start [mm]'); a3.set_ylabel('TCP XY step\n[mm]'); a3.set_xlabel('simulation step')
    a3.set_title('line = measured TCP step, x = commanded step', fontsize=6, loc='right', pad=2)
    for ax in axs:
        ax.set_xlim(*xlim); ax.grid(alpha=.25); ax.tick_params(labelsize=6)
        if trig is not None: ax.axvline(trig, color='k', lw=.8)
        if now is not None: ax.axvline(now, color='#d4a000', lw=1.4)


# ---------------------------------------------------------------- RGB
def frame_at(rec, fr, arm, t):
    """(image 128x256 or None, provenance). Stored frames only; nothing is interpolated or re-rendered."""
    if fr is None: return None, 'NO RGB STORED for this scene'
    t0 = rec['trigger']['step']; c = T.cell(rec, arm)
    if t <= t0: return fr['trunk'][t], 'recorded RGB (shared prefix)'
    fa = (c or {}).get('frames_after_trigger') or {}
    if fa.get('same_as'): return frame_at(rec, fr, fa['same_as'], t)[0], f"recorded RGB of {fa['same_as']} (abstained: identical trajectory, checked)"
    if fa.get('key') in fr and t in fa.get('steps', []):
        return fr[fa['key']][fa['steps'].index(t)], 'recorded RGB'
    if arm in fr and t - t0 < len(fr[arm]):
        a = fr[arm][t - t0]
        if a.shape[1] == 128: return np.concatenate([np.full_like(a, 150), a], 1), 'recorded RGB (v1 stored the wrist camera only)'
        return a, 'recorded RGB (during the move)'
    return None, 'NO RGB RECORDED for this step (trace only)'
