"""Recompute checkpoint accuracy from saved fields and extracted polylines."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from reproduction.common_rhs_integrity import array_digest
from reproduction.contour_distance import curve_distance, curve_statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); root = a.run
    raw = json.loads((root / 'results.json').read_text())
    if not raw['complete']:
        raise ValueError('run incomplete')
    archive = np.load(root / 'contours.npz')
    started = time.perf_counter(); count = 0; seen_keys = set()
    def snapshot(prefix, h, record):
        nonlocal count
        curves = {}; fields = {}
        for size in raw['grids']:
            key = f'{prefix}_{h}_{size}'
            field = np.load(root / f'{key}.npy', mmap_mode='r')
            assert field.shape == (size, size)
            assert np.isfinite(field).all()
            assert array_digest(field) == record[str(size)]['field_sha256']
            count += 1
            keys = [f'{key}_{i}' for i in range(record[str(size)]['components'])]
            seen_keys.update(keys)
            curves[size] = [archive[k] for k in keys]
            fields[size] = field
            for name, value in curve_statistics(curves[size]).items():
                np.testing.assert_allclose(value, record[str(size)][name], rtol=1e-12)
        low, fine = raw['grids']
        if curves[low] and curves[fine]:
            dist = curve_distance(curves[low], curves[fine])
            for k in ['hausdorff_lower', 'hausdorff_upper']:
                np.testing.assert_allclose(dist[k], record['refinement'][k], rtol=1e-12)
        else:
            dist = None
            assert record['refinement'] is None
        stable = all(record[str(low)][k] == record[str(fine)][k] for k in ['components', 'open_components'])
        assert stable == record['stable_counts']
        return fields, curves, dist, stable
    references = {h: snapshot('reference', h, raw['reference'][str(h)]) for h in range(4)}
    cumulative = 0.
    for checkpoint in raw['checkpoints']:
        iteration = checkpoint['iteration']
        actual = {}
        for h in range(4):
            record = checkpoint['planes'][str(h)]
            fields, curves, refinement, stable = snapshot(f'checkpoint{iteration}', h, record)
            ref_fields, ref_curves, ref_refinement, ref_stable = references[h]
            fine = raw['grids'][-1]
            rel = float(np.linalg.norm(fields[fine].astype('d') - ref_fields[fine]) /
                        np.linalg.norm(ref_fields[fine].astype('d')))
            np.testing.assert_allclose(rel, record['mean_relative_l2'], rtol=1e-12)
            d = curve_distance(curves[fine], ref_curves[fine]) if curves[fine] and ref_curves[fine] else None
            if d is not None:
                for k in ['hausdorff_lower', 'hausdorff_upper']:
                    np.testing.assert_allclose(d[k], record['distance'][k], rtol=1e-12)
            else:
                assert record['distance'] is None
            match = all(curve_statistics(curves[fine])[k] == curve_statistics(ref_curves[fine])[k]
                        for k in ['components', 'open_components'])
            assert match == record['reference_counts_match']
            actual[h] = {name: bool(rel <= raw['mean_gate'] and match and stable and ref_stable
                and d is not None and d['hausdorff_upper'] <= target
                and refinement is not None and refinement['hausdorff_upper'] <= target
                and ref_refinement is not None and ref_refinement['hausdorff_upper'] <= target)
                for name, target in raw['boundary_targets'].items()}
            assert actual[h] == record['quality']
        for label, ids in [('calibration', [0, 1]), ('holdout', [2, 3])]:
            assert checkpoint[f'{label}_quality'] == {k: all(actual[h][k] for h in ids) for k in raw['boundary_targets']}
        cumulative += checkpoint['aggregation_centering_seconds'] + checkpoint['calibration_seconds']
        np.testing.assert_allclose(checkpoint['calibration_path_seconds'],
            raw['seconds']['method_setup'] + checkpoint['update_seconds'] + cumulative, rtol=1e-12)
        assert array_digest(np.load(root / f'coefficients_{iteration}.npy', mmap_mode='r')) == checkpoint['coefficients_sha256']
    assert set(archive.files) == seen_keys
    result = {'complete': True, 'field_arrays_verified': count,
              'checkpoints_verified': len(raw['checkpoints']),
              'results_sha256': hashlib.sha256((root/'results.json').read_bytes()).hexdigest(),
              'contours_sha256': hashlib.sha256((root/'contours.npz').read_bytes()).hexdigest(),
              'seconds': time.perf_counter() - started,
              'scope': 'saved-field hashes, norms, polyline metrics, quality gates and calibration cost accounting; not independent solver/model validation'}
    (root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
