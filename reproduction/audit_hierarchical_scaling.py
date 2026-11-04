"""Audit every completed repetition against shared full-kernel references."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(root):
    root = Path(root)
    for directory in ['scaling', 'large', 'controls']:
        for path in sorted((root/directory).glob('*.json')):
            record = json.loads(path.read_text())
            if 'field' not in record or 'independent_audit' in record:
                continue
            n = record['arguments']['n']
            cmd = [sys.executable, '-m', 'reproduction.audit_hierarchical_variance', str(path),
                   '--reference', str(root/'references'/f'{n}.json')]
            if n > 73728:
                cmd += ['--random-queries', '8']
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='reproduction/results/hierarchical_variance')
    main(parser.parse_args().root)
