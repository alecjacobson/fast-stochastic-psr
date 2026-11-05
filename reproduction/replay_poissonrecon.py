"""Replay explicitly selected real-scan timings and independent audits sequentially."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from reproduction.poissonrecon_data import sha256


def replay(args):
    source, output, data = map(Path, [args.source, args.output, args.data])
    if source.resolve() == output.resolve():
        raise ValueError('Use a separate output directory to preserve the published measurements')
    selection = json.loads((source/'selection.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    (output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    hashes, failures = {}, []
    paths = list(dict.fromkeys(p for group in selection['series'] for p in group['records']))
    for relative in paths:
        original = json.loads((source/relative).read_text())
        options = dict(original['arguments'])
        dataset, n = options['geometry'], options['n']
        cloud = data/(dataset+'.npy')
        if dataset not in hashes:
            hashes[dataset] = sha256(cloud)
            (output/'data').mkdir(exist_ok=True)
            shutil.copyfile(cloud.with_suffix('.json'), output/'data'/(dataset+'.json'))
        if hashes[dataset] != options['point_cloud_sha256']:
            raise ValueError('Prepared data differs from selected record: '+dataset)
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        options.update(point_cloud=str(cloud), output=str(target))
        command = [sys.executable, '-m', 'reproduction.benchmark_hierarchical_variance']
        for key, value in options.items():
            if key == 'point_cloud_sha256' or value is None or value is False:
                continue
            command.append('--'+key.replace('_', '-'))
            if value is not True:
                command.append(str(value))
        if not target.exists():
            with target.with_suffix('.log').open('w') as log:
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    status = dict(command=command, returncode=result.returncode)
                except subprocess.TimeoutExpired:
                    status = dict(command=command, timeout_seconds=args.timeout)
            target.with_suffix('.status.json').write_text(json.dumps(status, indent=2)+'\n')
            if status.get('returncode') != 0:
                failures.append(relative)
                continue
        existing = json.loads(target.read_text())
        if any(existing['arguments'].get(key) != value for key, value in options.items()):
            raise ValueError('Existing output has different arguments: '+str(target))
        status_path = target.with_suffix('.status.json')
        if status_path.exists() and json.loads(status_path.read_text()).get('returncode') != 0:
            raise ValueError('Existing run failed; preserve it and use a fresh output directory')
        random_queries = original['independent_audit']['queries']-9
        reference = output/dataset/'references'/f'{n}-q{random_queries}.json'
        subprocess.run([sys.executable, '-m', 'reproduction.audit_hierarchical_variance', str(target),
                        '--reference', str(reference), '--random-queries', str(random_queries),
                        '--reference-preconditioner', 'prefix-kd'], check=True)
        if not json.loads(target.read_text())['independent_audit']['passes']:
            failures.append(relative)
    if failures:
        raise RuntimeError('No complete accepted replay; inspect '+repr(failures))
    for pair in selection.get('comparisons', []):
        target = output/pair['output']
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'reproduction.compare_hierarchical_application',
                        str(output/pair['candidate']), str(output/pair['baseline']),
                        '--output', str(target)], check=True)
        if not json.loads(target.read_text())['passes']:
            failures.append(pair['output'])
    if failures:
        raise RuntimeError('Full-query replay comparisons failed; inspect '+repr(failures))
    subprocess.run([sys.executable, '-m', 'reproduction.plot_poissonrecon_large', '--root', str(output)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='reproduction/results/poissonrecon_large')
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', default=str(Path.home()/'.cache/fast-stochastic-psr/poissonrecon'))
    parser.add_argument('--timeout', type=float, default=3600, help='Per production run; offline audits can take longer')
    replay(parser.parse_args())
