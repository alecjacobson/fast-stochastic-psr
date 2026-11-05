"""Prepare the upstream oriented point sets without resampling their geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCES = {
    'eagle': ('https://www.cs.jhu.edu/~misha/Code/PoissonRecon/eagle.points.ply',
              '468e41b3679f775b98fe8cc79132cb99e8d6dfc4dcc1bb684e571a90e326e478'),
    'torso': ('https://www.cs.jhu.edu/~misha/Code/PoissonRecon/torso.zip',
              'ad7955bb71fe4a042e125b407beed7eeabb79a2062157b28ea955cb621e66524'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_vertices(path):
    types = {'float': '<f4', 'double': '<f8', 'uchar': 'u1', 'int': '<i4', 'uint': '<u4'}
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('not a PLY file')
        properties, count, vertex = [], None, False
        while True:
            line = f.readline().decode('ascii').strip().split()
            if not line:
                raise ValueError('incomplete PLY header')
            if line[0] == 'format' and line[1] != 'binary_little_endian':
                raise ValueError('expected upstream binary little-endian PLY')
            if line[0] == 'element':
                vertex = line[1] == 'vertex'
                if vertex:
                    count = int(line[2])
            if line[0] == 'property' and vertex:
                if line[1] not in types:
                    raise ValueError('unsupported vertex property')
                properties.append((line[2], types[line[1]]))
            if line[0] == 'end_header':
                break
        return np.memmap(path, dtype=np.dtype(properties), mode='r', offset=f.tell(), shape=(count,))


def prepare(name, source, output, seed=12):
    source, output = Path(source), Path(output)
    digest = sha256(source)
    if digest != SOURCES[name][1]:
        raise ValueError('upstream PLY checksum mismatch')
    rows = read_vertices(source)
    x = np.column_stack([rows[k] for k in ['x', 'y', 'z']]).astype(float)
    v = np.column_stack([rows[k] for k in ['nx', 'ny', 'nz']]).astype(float)
    lengths = np.linalg.norm(v, axis=1)
    if not np.isfinite(x).all() or not np.isfinite(v).all() or np.any(lengths <= 0):
        raise ValueError('nonfinite points or invalid supplied normals')
    v /= lengths[:, None]
    lower, upper = x.min(axis=0), x.max(axis=0)
    extent = float(np.max(upper-lower))
    # Fixed geometric rule, selected before any solver timing or accuracy check.
    target = (lower+upper)/2 + np.array([.35*extent, 0., 0.])
    anchor_index = int(np.argmin(np.sum((x-target)**2, axis=1)))
    anchor, normal = x[anchor_index].copy(), v[anchor_index].copy()
    tangent = np.eye(3)[np.argmin(abs(normal))]
    tangent -= normal*(tangent@normal)
    tangent /= np.linalg.norm(tangent)
    rotation = np.column_stack((normal, tangent, np.cross(normal, tangent)))
    permutation = np.random.default_rng(seed).permutation(len(x))
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = np.lib.format.open_memmap(output, mode='w+', dtype=np.float64, shape=(len(x), 6))
    for start in range(0, len(x), 65536):
        idx = permutation[start:start+65536]
        normalized[start:start+len(idx), :3] = (x[idx]-anchor)@rotation*(2/extent)+[1, 0, 0]
        normalized[start:start+len(idx), 3:] = v[idx]@rotation
    normalized.flush(); del normalized
    metadata = dict(dataset=name, source_url=SOURCES[name][0], source_ply_sha256=digest,
                    source_points=len(x), normalized_sha256=sha256(output), permutation_seed=seed,
                    source_bounds=[lower.tolist(), upper.tolist()], coordinate_scale=2/extent,
                    anchor_source_index=anchor_index, anchor_source=anchor.tolist(),
                    normal_source=normal.tolist(), rotation=rotation.tolist(),
                    normal_length_min=float(lengths.min()), normal_length_max=float(lengths.max()),
                    preprocessing='Preserve supplied positions and normal directions; normalize normal lengths; rigid contact frame; global longest bounding-box side = 2 before rotation; deterministic nested permutation.',
                    contact_policy='Nearest original sample to bbox center plus .35 longest-side along original +x; anchor maps to (1,0,0), normal maps to +x. Mean-level anchoring only, no scalar conditioning.',
                    ground_truth='No ground-truth inside labels; torso envelope is not used as geometry truth or model conditioning.')
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)


def load_cloud(path, n, expected_hash=None):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    digest = sha256(path)
    if digest != metadata['normalized_sha256'] or (expected_hash and digest != expected_hash):
        raise ValueError('normalized point cloud checksum mismatch')
    rows = np.load(path, mmap_mode='r', allow_pickle=False)
    if rows.ndim != 2 or rows.shape[1] != 6 or not 1 <= n <= len(rows):
        raise ValueError('invalid point cloud shape or requested count')
    return np.array(rows[:n, :3]), np.array(rows[:n, 3:]), metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name', choices=SOURCES); p.add_argument('source'); p.add_argument('output')
    p.add_argument('--seed', type=int, default=12)
    a = p.parse_args(); prepare(a.name, a.source, a.output, a.seed)
