"""Numerical comparisons and paired scientific figures; shared scales per pair."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduction/results/matched'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                     'mathtext.fontset': 'stix', 'font.size': 14,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})


def compare(case):
    out = OUT / case
    data = np.load(out / 'input.npz')
    backend = 'compatible'
    base, fast = [np.load(out / (name + '.npz')) for name in ['original', backend]]
    a, b = [json.loads((out / (name + '.json')).read_text()) for name in ['original', backend]]
    for key in ['points_sha256', 'normals_sha256', 'queries_sha256']:
        assert a[key] == b[key]
    result = {'N': a['N'], 'M': a['M'], 'points_sha256': a['points_sha256'], 'nufft_backend': backend, 'fields': {}}
    for key in ['mean', 'variance', 'probability']:
        diff = fast[key] - base[key]
        result['fields'][key] = {'relative_l2': float(np.linalg.norm(diff) / np.linalg.norm(base[key])),
                                 'max_abs': float(np.max(np.abs(diff))),
                                 'rms_abs': float(np.sqrt(np.mean(diff ** 2)))}
    result['variance_error_over_reference_range'] = float(np.max(np.abs(fast['variance'] - base['variance'])) / np.ptp(base['variance']))
    result['first_reconstruction_seconds'] = {
        'released_api': sum(a['seconds'][key] for key in ['kernel_setup_first', 'data_mean_first', 'mean_first', 'variance_first']),
        'nufft': sum(b['seconds'][key] for key in ['fit_first', 'interpolant_spectrum_setup', 'offset_first', 'mean_first', 'cached_variance_first'])}
    result['first_reconstruction_speedup'] = result['first_reconstruction_seconds']['released_api'] / result['first_reconstruction_seconds']['nufft']
    result['mean_seconds'] = {'original_cached': a['mean_warm_median'], 'nufft': b['mean_warm_median']}
    if 'mean_repeat' in a['seconds']:
        result['released_api_mean_seconds'] = a['seconds']['mean_repeat']
        result['released_api_mean_speedup'] = a['seconds']['mean_repeat'] / b['mean_warm_median']
        result['released_api_variance_seconds'] = a['seconds']['variance_repeat']
        result['released_api_all_fields_speedup'] = (a['seconds']['mean_repeat'] + a['seconds']['variance_repeat']) / (b['mean_warm_median'] + b['seconds']['cached_variance_repeat'])
    result['mean_speedup'] = a['mean_warm_median'] / b['mean_warm_median']
    result['variance_seconds'] = {name: meta['seconds']['cached_variance_repeat'] for name, meta in [('original_cached', a), ('nufft', b)]}
    result['query_all_fields_seconds'] = {name: meta['mean_warm_median'] + meta['seconds']['cached_variance_repeat'] for name, meta in [('original_cached', a), ('nufft', b)]}
    result['all_fields_speedup'] = result['query_all_fields_seconds']['original_cached'] / result['query_all_fields_seconds']['nufft']
    (out / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(case, json.dumps(result, indent=2))

    # Same input cloud; each method receives two identical planes, kernels,
    # training covariance, hyperparameters and observation-noise convention.
    fig = plt.figure(figsize=(17, 5.7), facecolor='white')
    cloudax = fig.add_axes([.002, .13, .238, .77])
    cloudpath = out / 'cloud_render.png'
    if cloudpath.exists():
        cloudax.imshow(plt.imread(cloudpath))
    cloudax.axis('off')
    col_x = [.258, .376, .494, .633, .751, .869]
    row_y = [.56, .145]
    width, height = .112, .334
    axes = {}
    maps = ['RdBu', 'plasma', 'viridis']
    titles = ['Mean', 'Variance', r'$\mathbb{P}(x\in\Omega)$']
    for row, h in enumerate(data['heights']):
        for col, key in enumerate(['mean', 'variance', 'probability']):
            # Match upstream's linear per-image normalization, shared across
            # compared methods. Probability always has physical [0,1] limits.
            lo, hi = (0, 1) if key == 'probability' else (float(min(base[key][row].min(), fast[key][row].min())), float(max(base[key][row].max(), fast[key][row].max())))
            norm = Normalize(lo, hi)
            for method, values in enumerate([base, fast]):
                c = method * 3 + col
                ax = fig.add_axes([col_x[c], row_y[row], width, height])
                ax.imshow(values[key][row], origin='lower', extent=[data['axis'][0], data['axis'][-1]] * 2,
                          cmap=maps[col], norm=norm, interpolation='bilinear', aspect='equal')
                if col == 0:
                    ax.contour(data['axis'], data['axis'], values[key][row], levels=[0], colors='white', linewidths=.75)
                if row == 0:
                    ax.set_title(titles[col], fontsize=15, pad=5)
                if col == 0:
                    ax.text(.03, .04, f'$y={h:g}$', color='white', transform=ax.transAxes, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(.55); spine.set_color('#333')
                axes[row, c] = ax
            # Per-slice scales prevent the tiny but real low-density variance
            # reduction from disappearing; values are labeled, never hidden.
            cbax = fig.add_axes([col_x[3 + col] + .01, row_y[row] - .03, width - .02, .009])
            cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=maps[col]), cax=cbax, orientation='horizontal', ticks=[lo, hi])
            cb.ax.tick_params(labelsize=8, length=2, pad=1)
            cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g' if col == 2 else '%.2e'))
            cb.outline.set_linewidth(.4)
    if (out / 'cloud_projection.json').exists():
        anchors = json.loads((out / 'cloud_projection.json').read_text())['anchors']
        for row, anchor in enumerate(anchors):
            fig.add_artist(ConnectionPatch(xyA=anchor, coordsA=cloudax.transAxes, xyB=(0, .5), coordsB=axes[row, 0].transAxes, linewidth=.65, color='#444', zorder=0))
    fig.text(.008, .985, f'({case}) ' + ('Released scan' if case == 'A' else 'Denser scan, same cameras'), va='top', fontsize=20)
    fig.text(.12, .10, f'{a["N"]:,} observations', ha='center', fontsize=15)
    fig.text(.43, .95, 'Original GeoSPSR', ha='center', fontsize=19)
    fig.text(.80, .95, 'NUFFT · matched kernel', ha='center', fontsize=19)
    if case == 'A':
        fig.text(.43, .035, 'Verified against the untouched API', ha='center', fontsize=13)
        fig.text(.80, .035, f'Mean error: {100 * result["fields"]["mean"]["relative_l2"]:.3f}% · probability RMS: {100 * result["fields"]["probability"]["rms_abs"]:.3f} pp', ha='center', fontsize=13)
    else:
        fig.text(.43, .035, f'Mean, cached fit: {a["mean_warm_median"] * 1000:.1f} ms', ha='center', fontsize=14)
        fig.text(.80, .035, f'Mean: {b["mean_warm_median"] * 1000:.1f} ms  ({result["mean_speedup"]:.1f}×)', ha='center', fontsize=14)
    fig.text(.5, -.018, 'Identical inputs and slices · deterministic variances · color scales shared within each comparison', ha='center', fontsize=12, color='#444')
    for ext in ['png', 'pdf', 'svg']:
        path = ROOT / 'assets' / f'scorpion-matched-{case}.{ext}'
        fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=.06)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)

    # Separate error plots use absolute differences, not independently rescaled
    # originals. Probability errors are reported in percentage points.
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    for col, key in enumerate(['mean', 'variance', 'probability']):
        diff = np.abs(base[key] - fast[key]) * (100 if key == 'probability' else 1)
        for row in range(2):
            im = axs[row, col].imshow(diff[row], origin='lower', cmap='magma', vmin=0, vmax=diff.max())
            axs[row, col].axis('off')
            if row == 0:
                axs[row, col].set_title(key + (' (percentage points)' if key == 'probability' else ' absolute error'))
        fig.colorbar(im, ax=axs[:, col], shrink=.7)
    fig.savefig(out / 'errors.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    for case in ['A', 'B']:
        if (OUT / case / 'compatible.npz').exists():
            compare(case)
