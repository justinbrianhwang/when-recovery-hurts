"""Draw README figures from recorded measurements in results/. No simulation is run here."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                     'savefig.facecolor': SURFACE})


def rows(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))


def restore_divergence():
    r = rows(ROOT / 'results/stage0/restore_divergence.csv')
    x = [int(v['step_after_snapshot']) for v in r]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for key, name, color in (('stackcube_set_state', 'StackCube', BLUE), ('pickcube_set_state', 'PickCube', ORANGE)):
        y = [float(v[key]) for v in r]
        ax.plot(x, y, color=color, lw=2, marker='o', ms=6, mec=SURFACE, mew=1.5, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(8, 0), textcoords='offset points', va='center', color=INK)
    ax.axhline(1e-6, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.annotate('audit tolerance 1e-6', (1, 1e-6), xytext=(0, 6), textcoords='offset points', color=MUTED)
    ax.annotate('prefix-replay restore: exactly 0.0 at every step, both tasks\n(cannot be drawn on a log axis)',
                (4.5, 1.2e-8), ha='center', color=INK, fontsize=10)
    ax.set_yscale('log'); ax.set_ylim(3e-9, 3e-2); ax.set_xlim(0.7, 9.6); ax.set_xticks(x)
    ax.set_xlabel('control steps after the snapshot'); ax.set_ylabel('max |state difference| vs. unrestored rollout')
    ax.set_title('Restoring with set_state perturbs the simulation', loc='left', color=INK, fontsize=13, pad=12)
    ax.grid(True, axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT / 'assets/restore_divergence.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    restore_divergence()
    print('figures written to assets/')
