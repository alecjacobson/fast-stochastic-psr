"""Plot accuracy-qualified Torso scaling, preserving the failed variance trial."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np


def main(args):
    root = Path(args.output)
    paths = [Path('reproduction/results/poissonrecon_large/torso/final')/f'h2-streamed-{n}-0.json'
             for n in (77824, 196608)]
    paths.append(root/'torso/strong-preconditioner/h2-streamed-524288-0.json')
    failed_path = root/'torso/half-million/h2-streamed-524288-0.json'
    records = [json.loads(p.read_text()) for p in paths]
    failed = json.loads(failed_path.read_text())
    if not all(r['independent_audit']['passes'] for r in records):
        raise ValueError('selected scaling points must pass the independent audit')
    if failed['independent_audit']['passes']:
        raise ValueError('expected failed variance trial is no longer failing')
    keys = ('seed', 'point_cloud_sha256', 'lengthscale', 'amplitude', 'relative_noise', 'halfwidth', 'queries', 'batches')
    for r in records+[failed]:
        for key in keys:
            if r['arguments'][key] != records[0]['arguments'][key]:
                raise ValueError(f'workload contract mismatch: {key}')
        if r['device'] != records[0]['device'] or r['cpu_quota'] != records[0]['cpu_quota']:
            raise ValueError('hardware contract mismatch')
        audit = r['independent_audit']
        if hashlib.sha256(Path(audit['reference']).read_bytes()).hexdigest() != audit['reference_sha256']:
            raise ValueError('independent reference artifact changed')
    n = np.array([r['arguments']['n'] for r in records])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), layout='constrained')
    for ax in axes:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(n))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.1f}k'))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlabel('Torso observations')
        ax.grid(alpha=.2, which='major')
    ax = axes[0]
    ax.set_yscale('log')
    ax.plot(n, [r['total_seconds'] for r in records], 'o-', label='Passes both output gates')
    ax.scatter([n[-1]], [failed['total_seconds']], color='tab:red', marker='x', s=75, label='Fails variance gate')
    for r in records:
        ax.annotate(f"{r['total_seconds']:.1f} s", (r['arguments']['n'], r['total_seconds']),
                    xytext=(-12, -17) if r is records[-1] else (0, 10),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.set_ylabel('Setup + mean + variance + queries (s)')
    ax.set_title('Complete million-query workload')
    ax.legend(fontsize=8, loc='upper left')
    ax = axes[1]
    ax.plot(n, [r['solve_info']['mean']['max_iterations'] for r in records], 'o-', label='Selected preconditioner')
    ax.scatter([n[-1]], [failed['solve_info']['mean']['max_iterations']], color='tab:red', marker='x', s=75)
    for r in records+[failed]:
        ax.annotate(str(r['solve_info']['mean']['max_iterations']),
                    (r['arguments']['n'], r['solve_info']['mean']['max_iterations']),
                    xytext=(0, 7), textcoords='offset points', ha='center', fontsize=9)
    ax.set_ylim(0, 76)
    ax.set_ylabel('Mean PCG iterations')
    ax.set_title('Stronger AFN at 524k; eight variance steps')
    ax = axes[2]
    for key, gate, label, marker in [('max_probability_error', 1e-3, 'CDF error / gate', 'o'),
                                    ('max_variance_error_over_prior', 1e-4, 'Variance error / gate', 's')]:
        ax.semilogy(n, [r['independent_audit'][key]/gate for r in records], marker+'-', label=label)
        ax.scatter([n[-1]], [failed['independent_audit'][key]/gate], color='tab:red', marker='x', s=75)
    ax.axhline(1., color='black', linestyle=':', label='Acceptance limit')
    ax.set_ylabel('Independent 17-probe error / tolerance')
    ax.set_title('Probability alone misses this failure')
    ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('Torso subset scaling: fixed physical box, model and 1,048,576 probability queries\n'
                 'Single runs; offline full-kernel audits excluded; no full-Torso or competing-method win claimed', fontsize=12)
    fig.savefig(root/'torso-scaling.png', dpi=170)
    fig.savefig(root/'torso-scaling.pdf')
    summary = dict(selected=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                   n=r['arguments']['n'], seconds=r['total_seconds'],
                                   mean_iterations=r['solve_info']['mean']['max_iterations'],
                                   retained_pool_gib=r['cupy_pool_total_bytes']/2**30,
                                   independent_audit=r['independent_audit']) for p, r in zip(paths, records)],
                   failed=dict(path=str(failed_path), sha256=hashlib.sha256(failed_path.read_bytes()).hexdigest(),
                               seconds=failed['total_seconds'], independent_audit=failed['independent_audit']),
                   scope='Nested subsets; configuration adapts to output accuracy and memory. '
                         'No full-Torso or competing-method speedup measured.')
    (root/'torso-summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reproduction/results/regional_accuracy')
    main(parser.parse_args())
