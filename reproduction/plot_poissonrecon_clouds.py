"""Show the complete scan extents and preselected contact; plot a fixed decimation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    for ax, name in zip(axes, ['eagle', 'torso']):
        path = Path(data)/(name+'.npy')
        meta = json.loads(path.with_suffix('.json').read_text())
        cloud = np.load(path, mmap_mode='r')
        ids = np.linspace(0, len(cloud)-1, min(len(cloud), 80000), dtype=int)
        xyz = (cloud[ids, :3]-[1, 0, 0])@np.array(meta['rotation']).T/meta['coordinate_scale']+meta['anchor_source']
        order = np.argsort(xyz[:, 1])
        xyz = xyz[order]
        ax.scatter(xyz[:, 0], xyz[:, 2], c=xyz[:, 1], s=.15, cmap='bone', rasterized=True)
        a = meta['anchor_source']
        ax.scatter([a[0]], [a[2]], c='#d04a2f', marker='*', s=85, label='Fixed contact')
        ax.set_aspect('equal')
        ax.set_title(f'{name.title()}: {len(cloud):,} oriented points')
        ax.set_xlabel('Source x'); ax.set_ylabel('Source z')
        ax.legend(loc='lower left', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('PoissonRecon inputs • 80,000 displayed samples each • all points available to solvers')
    fig.tight_layout()
    fig.savefig(output, dpi=180)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default=str(Path.home()/'.cache/fast-stochastic-psr/poissonrecon'))
    p.add_argument('--output', default='reproduction/results/poissonrecon_large/clouds.png')
    a = p.parse_args(); main(a.data, a.output)
