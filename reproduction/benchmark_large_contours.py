"""Refine two mean-zero slice contours from the saved three-million-point fits."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import contourpy
from reproduction.common_rhs_integrity import array_digest
from reproduction.contour_distance import curve_statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--fits', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import cupy as cp
    from geospsr.spectral import FourierPoisson
    from reproduction.gpu_audit import GpuProcessAudit
    from reproduction.posterior import sync
    root = Path(__file__).resolve().parent / 'results'
    a.output.mkdir(parents=True, exist_ok=True)
    r = {'complete': False, 'scope': 'saved-fit zero contours on two slices only',
         'grids': [257, 513, 1025, 2049], 'heights': [-.35, -.75],
         'domain': [-float(np.pi/2), float(np.pi/2)], 'methods': {}}
    curves = {}
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    def save():
        (a.output / 'results.json').write_text(json.dumps(r, indent=2) + '\n')
    try:
        source = np.load(a.source / 'points.npy')
        manifest = json.loads((root / 'large_inputs/scorpion_3000000/manifest.json').read_text())
        r['source_sha256'] = array_digest(source)
        if r['source_sha256'] != manifest['arrays']['points']['payload_sha256']:
            raise ValueError('source mismatch')
        for name in ['half200', 'half250', 'sdd']:
            raw = json.loads((root / 'large_fit' / name / 'results.json').read_text())
            weights = np.load(a.fits / name / 'weights.npy')
            digest = array_digest(weights)
            if not raw['complete'] or digest != raw['weights_sha256']:
                raise ValueError('weights mismatch')
            info = r['methods'][name] = {'weights_sha256': digest, 'grids': {}}
            op = FourierPoisson(50)
            sync(); start = time.perf_counter()
            coeff = op.aggregate(source, weights)
            sync(); info['aggregation_seconds'] = time.perf_counter() - start
            with np.load(root / 'large_fit' / name / 'fields.npz') as f:
                offset = float(f['offset'])
                old = cp.asnumpy(op.evaluate(f['query'], coeff)[0]) - offset
                error = float(np.max(np.abs(old - f['mean'])))
            info['replay_max_abs_error'] = error
            if error > 1e-6:
                raise ValueError('saved mean replay failed')
            for size in r['grids']:
                axis = np.linspace(*r['domain'], size, dtype=np.float32)
                xx, zz = np.meshgrid(axis, axis)
                for height_index, height in enumerate(r['heights']):
                    query = np.stack([xx.ravel(), np.full(xx.size, height, np.float32), zz.ravel()], axis=1)
                    sync(); start = time.perf_counter()
                    value = cp.asnumpy(op.evaluate(query, coeff)[0]).reshape(size, size) - offset
                    sync(); seconds = time.perf_counter() - start
                    extracted = contourpy.contour_generator(x=axis, y=axis, z=value,
                                                            name='serial', line_type='Separate').lines(0.)
                    if not extracted:
                        raise ValueError('empty zero contour')
                    key = f'{name}_{size}_{height_index}'
                    for i, curve in enumerate(extracted):
                        curves[f'{key}_{i}'] = curve
                    info['grids'][f'{size}_{height_index}'] = {
                        **curve_statistics(extracted), 'query_seconds': seconds,
                        'negative_grid_fraction': float(np.mean(value <= 0)),
                        'field_sha256': array_digest(value)}
                    # Large fields outside Git; compact contour vertices are published.
                    np.save(a.output / f'{key}.npy', value)
                    print(key, info['grids'][f'{size}_{height_index}'], flush=True)
                    save()
            del coeff, op, weights
            cp.get_default_memory_pool().free_all_blocks()
        np.savez_compressed(a.output / 'contours.npz', **curves)
        r['complete'] = True
    finally:
        r['gpu_process_audit'] = audit.stop()
        save()


if __name__ == '__main__':
    main()
