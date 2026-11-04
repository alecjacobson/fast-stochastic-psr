"""Sequential, isolated-process repetitions for the fixed contact-region case."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for n in args.n:
        for rep in range(args.repeats):
            # Rotate execution order to reduce systematic warm/hot-device bias.
            methods = args.methods[rep % len(args.methods):]+args.methods[:rep % len(args.methods)]
            for method in methods:
                stem = f'{method}-{n}-{rep}'
                path = root/(stem+'.json')
                if path.exists():
                    continue
                cmd = [sys.executable, '-m', 'reproduction.benchmark_hierarchical_variance',
                       '--n', str(n), '--method', method, '--variance-iterations', str(args.variance_iterations),
                       '--output', str(path)]
                for key in ['lengthscale', 'halfwidth', 'h2_tolerance', 'leaf_size', 'rank', 'neighbors',
                            'variance_refinements', 'seed', 'scan', 'normal_noise', 'sparse_cutoff',
                            'sparse_builder', 'order', 'cross_tolerance', 'geometry', 'point_cloud', 'cross_block_size', 'preconditioner_backend', 'interaction_budget_mib', 'rhs_block_size']:
                    value = getattr(args, key)
                    if value is not None:
                        cmd += ['--'+key.replace('_', '-'), str(value)]
                if args.release_native:
                    cmd += ['--release-native']
                with (root/(stem+'.log')).open('w') as log:
                    try:
                        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                        status = dict(command=cmd, returncode=result.returncode)
                    except subprocess.TimeoutExpired:
                        status = dict(command=cmd, timeout_seconds=args.timeout)
                (root/(stem+'.status.json')).write_text(json.dumps(status, indent=2)+'\n')
                print(stem, status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', nargs='+', type=int, default=[8192, 16384, 32768, 65536, 73728])
    parser.add_argument('--methods', nargs='+', default=['h2-gpu', 'dense-float32'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--variance-iterations', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=180)
    for name in ['lengthscale', 'halfwidth', 'h2-tolerance', 'normal-noise', 'sparse-cutoff', 'cross-tolerance']:
        parser.add_argument('--'+name, type=float)
    for name in ['leaf-size', 'rank', 'neighbors', 'variance-refinements', 'seed', 'order', 'cross-block-size', 'interaction-budget-mib', 'rhs-block-size']:
        parser.add_argument('--'+name, type=int)
    parser.add_argument('--geometry', choices=['ellipsoid', 'sphere', 'eagle', 'torso'])
    parser.add_argument('--point-cloud')
    parser.add_argument('--preconditioner-backend', choices=['native', 'prefix-kd'])
    parser.add_argument('--release-native', action='store_true')
    parser.add_argument('--scan', choices=['full', 'front'])
    parser.add_argument('--sparse-builder', choices=['gpu', 'cpu'])
    parser.add_argument('--output', default='reproduction/results/hierarchical_variance/scaling')
    main(parser.parse_args())
