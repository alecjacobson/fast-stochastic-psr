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
    parser.add_argument('--output', default='reproduction/results/hierarchical_variance/scaling')
    main(parser.parse_args())
