"""Show observed SDD accuracy checkpoints without extrapolating convergence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    root = Path(root)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for n, color, h2path in [
            (196608, '#1764a2', 'tuned/h2-gpu-196608-0.json'),
            (796825, '#c14c25', 'eight-large/h2-streamed-796825-0.json')]:
        d = json.loads((root/'eagle/sdd'/f'{n}.json').read_text())
        h2 = json.loads((root/'eagle'/h2path).read_text())
        if not h2['independent_audit']['passes']:
            raise ValueError('H2 comparison must pass its independent audit')
        for config in d['configurations']:
            label = f"SDD N={n:,}, eta={config['eta']}"
            for ax, metric in zip(axes, ['max_probability_error', 'max_variance_error_over_prior']):
                rows = [r for r in config['history'] if r[metric] is not None]
                ax.loglog([r['solve_seconds'] for r in rows], [r[metric] for r in rows],
                          ('--o' if config['eta'] == .03 else '-o'), color=color, label=label)
        for ax in axes:
            ax.axvline(h2['total_seconds'], color=color, linestyle=':', alpha=.7,
                       label=f'H² complete workload, N={n:,}')
    for ax, gate, title in zip(axes, [1e-3, 1e-4], ['Probability error', 'Variance error / prior']):
        ax.axhline(gate, color='black', linestyle='--', label='Acceptance threshold')
        ax.set(xlabel='SDD solver seconds (compression/compile excluded)', ylabel=title)
        ax.grid(True, which='both', alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Observed accuracy checkpoints • no extrapolated SDD completion time', fontsize=11)
    fig.tight_layout()
    fig.savefig(root/'sdd-checkpoints.png', dpi=180)
    fig.savefig(root/'sdd-checkpoints.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='reproduction/results/poissonrecon_large')
    main(p.parse_args().root)
