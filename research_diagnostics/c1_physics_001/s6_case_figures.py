"""Figures of the rule-selected S6 extreme cases (S6_EXTREME_CASES.csv), from the raw trace rows (POST_HOC_EXPLORATORY).

For each case: the two collision shapes projected on the plane spanned by the contact normal of the minimum-separation
point and a horizontal axis perpendicular to it, at row t (solid) and at f = 0.8 of the control step (dashed,
constant-velocity ASSUMPTION), with every PhysX point of the pair (marker) and its normal. Panel text: PhysX separation,
reconstruction at t and at f = 0.8, logged-velocity back-step, record / trace sha256 prefixes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import diagnose_s6_offline as D  # noqa: E402
import geometry_reference as REF  # noqa: E402


def outline(P2):
    from scipy.spatial import ConvexHull
    h = ConvexHull(P2); v = list(h.vertices) + [h.vertices[0]]; return P2[v]


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    e = pd.read_csv(C.OUT / 'S6_EXTREME_CASES.csv'); n = len(e); cols = 5; rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.3 * rows), dpi=110); axs = np.atleast_2d(axs)
    for ax, (_, r) in zip(axs.ravel(), e.iterrows()):
        rec, _ = C.record(r.policy, int(r.scene)); tr, _ = C.traces(r.policy, int(r.scene)); geo = rec['initial_state']['collision_geometry']
        rows_ = tr[r.arm]; i = next(k for k, x in enumerate(rows_) if int(x['t']) == int(r.t)); r1, r0 = rows_[i], rows_[i - 1]
        a, b = r.shape_pair.split('|'); (na, sa), (nb, sb) = a.split('#'), b.split('#'); sa, sb = int(sa), int(sb)
        pts = [p for c in r1['contacts'] if {(c['a'], c['shape_a']), (c['b'], c['shape_b'])} == {(na, sa), (nb, sb)} for p in c['points']]
        pmin = min(pts, key=lambda p: p[9]); nrm = np.asarray(pmin[3:6], float); h = np.cross(nrm, [0, 0, 1.]); h = h / np.linalg.norm(h) if np.linalg.norm(h) > 1e-6 else np.array([1., 0, 0])
        B = np.stack([h, nrm]); origin = np.asarray(pmin[0:3], float)
        moving = lambda nm: nm in r1['link_pose'] or nm.startswith('cube')
        for f, ls in ((1.0, '-'), (0.8, '--')):
            for nm, si, col in ((na, sa, '#2a6fdb'), (nb, sb, '#c0392b')):
                pose = D.interp(D.body_pose(r0, geo, nm), D.body_pose(r1, geo, nm), f) if moving(nm) else D.body_pose(r1, geo, nm)
                d = geo[nm]['shapes'][si]; V = REF.pose_vertices(D.local_vertices(d), d['local_pose'], pose)
                P2 = (V - origin) @ B.T * 1e3; O = outline(P2); ax.plot(O[:, 0], O[:, 1], ls=ls, color=col, lw=1.1, label=f'{nm}#{si} at {"t" if f == 1 else "f=0.8"}')
        for p in pts:
            x = (np.asarray(p[0:3]) - origin) @ B.T * 1e3; v = np.asarray(p[3:6]) @ B.T
            ax.plot(*x, 'k.', ms=5); ax.annotate('', xy=x + v * 4, xytext=x, arrowprops=dict(arrowstyle='->', color='k', lw=.7))
        span = 25; ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_aspect('equal'); ax.tick_params(labelsize=6); ax.grid(alpha=.2)
        ax.set_title(f'{r.policy[-6:]} {int(r.scene)} {r.arm[:14]} t={int(r.t)}\n{r.shape_pair}\n{r.why}', fontsize=6.2)
        ax.text(0.02, 0.02, f'PhysX sep {r.physx_sep_m*1e3:.2f} mm | recon t {r.stored_recon_m*1e3:.2f}\nrecon f0.8 {(r.physx_sep_m - r["diff_at_f0.8_m"])*1e3:.2f} | back-step {(r.physx_sep_m - r.diff_backstep_logged_velocity_m)*1e3:.2f}\n'
                f'{r.impulse_group}, |v_rel| {r.speed_rel_mps*1e3:.0f} mm/s\nrecord {r.record_sha256[:10]} trace {r.trace_sha256[:10]}', transform=ax.transAxes, fontsize=5.6, va='bottom',
                bbox=dict(facecolor='white', alpha=.8, lw=0))
        ax.legend(fontsize=4.8, loc='upper right')
    for ax in axs.ravel()[n:]: ax.axis('off')
    fig.suptitle('A2: S6 extreme cases (rule-selected), projected on (horizontal, contact normal) through the minimum-separation point; mm. Solid: logged row t; dashed: f = 0.8 (ASSUMPTION). POST_HOC_EXPLORATORY', fontsize=8)
    fig.tight_layout(); fig.savefig(C.OUT / 'figures' / 's6_extreme_cases.png'); plt.close(fig); print('ok', n)


if __name__ == '__main__':
    main()
