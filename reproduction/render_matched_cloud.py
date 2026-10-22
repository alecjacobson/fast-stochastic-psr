"""Reuse the scientific point renderer on the identical A/B scan inputs."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--case', choices=['A', 'B'], required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
# The shared renderer is executed with explicit input files and point radius.
import sys
sys.argv = [sys.argv[0], '--input', str(root / 'reproduction/results/matched' / a.case / 'input.npz'),
            '--radius', '.04' if a.case == 'A' else '.016']
exec(compile((root / 'reproduction/render_million_cloud.py').read_text(),
             str(root / 'reproduction/render_million_cloud.py'), 'exec'))
