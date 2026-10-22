"""Compare complete demo outputs with and without prepared query caches."""
import json
import argparse
from pathlib import Path
import numpy as np


def read_mesh(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith('v '):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith('f '):
            faces.append(line)
    return np.array(vertices), faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent / 'results/cached_queries')
    parser.add_argument('--reference', default='applications-ordinary')
    parser.add_argument('--candidate', default='applications-cached')
    parser.add_argument('--output', default='application_cache_check.json')
    args = parser.parse_args()
    root = args.root
    ordinary, cached = root / args.reference, root / args.candidate
    a, b = [json.loads((p / 'applications.json').read_text()) for p in [ordinary, cached]]
    result = {'scope': 'single first calls in the full scorpion application demonstration; same samples and input; not a repeated-run performance benchmark',
              'reference': str(ordinary), 'candidate': str(cached),
              'seconds_ordinary': a['seconds'], 'seconds_cached': b['seconds'], 'fields': {}, 'meshes': {}}
    aa, bb = [np.load(p / 'applications.npz') for p in [ordinary, cached]]
    for key in ['transmittance', 'scores', 'collision', 'volume_mean', 'volume_std']:
        x, y = aa[key], bb[key]
        result['fields'][key] = {'relative_l2': float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30)),
                                 'max_abs': float(np.max(np.abs(y - x)))}
    for eta in [0, 1, 2]:
        x, fx = read_mesh(ordinary / f'hitbox_eta{eta}.obj')
        y, fy = read_mesh(cached / f'hitbox_eta{eta}.obj')
        result['meshes'][str(eta)] = {
            'same_vertex_count': len(x) == len(y), 'identical_faces': fx == fy, 'vertices': len(x),
            'max_vertex_displacement': float(np.linalg.norm(y - x, axis=1).max()) if len(x) == len(y) and fx == fy else None,
            'domain_clipping_unchanged': a['hitboxes'][eta]['touches_domain_boundary'] == b['hitboxes'][eta]['touches_domain_boundary']}
    (root / args.output).write_text(json.dumps(result, indent=2) + '\n')
    assert all(v['relative_l2'] < 1e-4 for v in result['fields'].values())
    assert all(v['identical_faces'] and v['max_vertex_displacement'] < .001 for v in result['meshes'].values())


if __name__ == '__main__':
    main()
