"""Plot accuracy and empirical ranks, without claiming an end-to-end speedup."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def main():
    root = Path('reproduction/results/nonperiodic')
    studies = {name: [json.loads(line) for line in (root/name/'regional.jsonl').read_text().splitlines()]
               for name in ['pilot', 'scaling', 'local', 'stress']}
    main_rows = studies['pilot'] + studies['scaling']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for region, label in [('separated', 'Separated box'), ('near_surface', 'Surface box'),
                          ('whole_object', 'Whole-object box')]:
        rows = [r for r in main_rows if r['region'] == region and r['order'] == 7]
        axes[0, 0].loglog([r['n'] for r in rows], [r['max_variance_error_over_prior'] for r in rows], 'o-', label=label)
        axes[0, 1].semilogx([r['n'] for r in rows],
                           [r['whitened_node_spectral_ranks']['0.0001'] for r in rows], 'o-', label=label)
    axes[0, 0].axhline(1e-4, color='black', linestyle='--', label='Variance gate')
    axes[0, 0].set(title='Fixed order 7 (343 coefficients)', ylabel='Max variance error / prior variance')
    axes[0, 1].set(title='Dense diagnostic rank, order 7', ylabel='Singular values > 10⁻⁴ × largest')
    for study, label, style in [('local', 'Local, ℓ=.3, noise=.01', '-'),
                               ('stress', 'Local, ℓ=.15, noise=.0001', '--')]:
        for order in [5, 7]:
            rows = [r for r in studies[study] if r['region'] == 'near_surface' and r['order'] == order]
            axes[1, 0].loglog([r['n'] for r in rows], [r['max_variance_error_over_prior'] for r in rows],
                              'o'+style, label=f'{label}, order {order}')
    axes[1, 0].axhline(1e-4, color='black', linestyle=':')
    axes[1, 0].set(title='Smaller surface box: halfwidth .1', ylabel='Max variance error / prior variance')
    reference = json.loads((root/'pilot/original_reference.json').read_text())
    x = [q[0] for q in reference['query']]
    for row in reference['rows']:
        axes[1, 1].plot(x, row['ray_variance'], label=f"Original SPSR {row['grid_size']}³")
    axes[1, 1].axvline(-.7, color='grey', linestyle=':')
    axes[1, 1].axvline(.7, color='grey', linestyle=':')
    axes[1, 1].set(title='Unmodified original reference (kernel changes with grid)',
                   xlabel='Position along sphere diameter', ylabel='Reported variance')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xlabel('Observations N')
        ticks = [128, 256, 512, 1024] if ax is axes[1, 0] else [128, 256, 512, 1024, 2048, 4096]
        ax.set_xticks(ticks, labels=[str(t) for t in ticks], fontsize=9)
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle('Non-periodic variance feasibility: accuracy and rank, not a scalability claim')
    fig.savefig(root/'pilot.png', dpi=160)


if __name__ == '__main__':
    main()
