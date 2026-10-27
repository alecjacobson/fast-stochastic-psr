"""Plot every predeclared calibration rank, with all three seeds retained."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'results/deflated_variance'


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for row, (label, prefix) in enumerate([('Unfiltered', ''), ('One filtering step', 'filtered/')]):
        for col, case in enumerate('AB'):
            r = json.loads((ROOT / f'{prefix}pilot/{case}/results.json').read_text())
            ax = axes[row, col]
            for rank in [0] + r['ranks']:
                values = np.array([[next(c['metrics']['contrast_relative_l2'] * 100
                                         for c in run['checkpoints']
                                         if (c['rank'], c['count']) == (rank, count))
                                    for count in r['looks']] for run in r['runs']])
                line, = ax.loglog(r['looks'], np.median(values, axis=0),
                                 '--' if rank == 0 else '-o', markersize=4,
                                 label='Raw moments' if rank == 0 else f'Rank {rank}')
                ax.fill_between(r['looks'], values.min(axis=0), values.max(axis=0),
                                color=line.get_color(), alpha=.18)
            ax.axhline(1, color='black', lw=1, ls=':')
            ax.set_title(f'{label}: {case}, N = {r["N"]:,}')
            ax.grid(alpha=.2, which='both')
            ax.legend(fontsize=8, ncol=2)
            if col == 0:
                ax.set_ylabel('Variance error / reference spatial contrast (%)')
            if row == 1:
                ax.set_xlabel('Independent evaluation draws (pilot/action cost excluded)')
    fig.suptitle('Fixed 20,000-query slices: calibration medians and three-seed ranges', fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / 'convergence.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
