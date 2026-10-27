"""Oracle checkpoint accuracy on calibration and held-out mean-zero slices."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import contourpy
from reproduction.common_rhs_integrity import array_digest
from reproduction.contour_distance import curve_distance, curve_statistics

GRIDS = [1025, 2049]
HEIGHTS = [-.4, -.7, -.5, -.8]
TARGETS = {'coarse': .001 * np.pi, 'strict': .0001 * np.pi}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--method', choices=['half200', 'sdd'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import cupy as cp
    from geospsr.spectral import FourierPoisson
    from reproduction.gpu_audit import GpuProcessAudit
    from reproduction.posterior import sync
    root = Path(__file__).resolve().parent / 'results'
    a.output.mkdir(parents=True, exist_ok=True)
    r = {'complete': False, 'method': a.method, 'seed': a.seed,
         'grids': GRIDS, 'heights': HEIGHTS, 'boundary_targets': TARGETS,
         'mean_gate': .002, 'checkpoints': [], 'seconds': {},
         'scope': 'oracle mean/2D boundary accuracy, not posterior applications or deployable stopping',
         'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    archive = {}
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    work_start = time.perf_counter()
    def save():
        (a.output / 'results.json').write_text(json.dumps(r, indent=2) + '\n')
    def elapsed(start):
        sync()
        return time.perf_counter() - start
    def snapshot(op, coefficients, offset, h):
        result = {}; fields = {}; curves = {}
        for size in GRIDS:
            axis = np.linspace(-np.pi/2, np.pi/2, size, dtype=np.float32)
            xx, zz = np.meshgrid(axis, axis)
            query = np.stack([xx.ravel(), np.full(xx.size, HEIGHTS[h], np.float32), zz.ravel()], axis=1)
            value = cp.asnumpy(op.evaluate(query, coefficients)[0]).reshape(size, size) - offset
            contour = contourpy.contour_generator(x=axis, y=axis, z=value,
                                                  name='serial', line_type='Separate').lines(0.)
            fields[size] = value; curves[size] = contour
            result[str(size)] = {**curve_statistics(contour), 'field_sha256': array_digest(value)}
        result['refinement'] = (curve_distance(curves[GRIDS[0]], curves[GRIDS[1]])
                                if all(curves.values()) else None)
        result['stable_counts'] = all(result[str(GRIDS[0])][key] == result[str(GRIDS[1])][key]
                                     for key in ['components', 'open_components'])
        return result, fields, curves
    def store_snapshot(prefix, h, fields, curves):
        for size, field in fields.items():
            np.save(a.output / f'{prefix}_{h}_{size}.npy', field)
            for i, curve in enumerate(curves[size]):
                archive[f'{prefix}_{h}_{size}_{i}'] = curve
    try:
        begin = time.perf_counter()
        source = np.load(a.source / 'points.npy'); normals = np.load(a.source / 'normals.npy')
        manifest = json.loads((root / 'large_inputs/scorpion_3000000/manifest.json').read_text())
        r['source_hashes'] = {k: array_digest(v) for k, v in [('points', source), ('normals', normals)]}
        if any(v != manifest['arrays'][k]['payload_sha256'] for k, v in r['source_hashes'].items()):
            raise ValueError('source hash mismatch')
        r['N'] = len(source)
        r['seconds']['source_load_verify'] = elapsed(begin)
        begin = time.perf_counter()
        ref_raw = json.loads((root / 'large_fit/half250/results.json').read_text())
        ref_weights = np.load(a.reference / 'weights.npy')
        r['reference_weights_sha256'] = array_digest(ref_weights)
        if r['reference_weights_sha256'] != ref_raw['weights_sha256']:
            raise ValueError('reference hash mismatch')
        ref_op = FourierPoisson(50)
        ref_coeff = ref_op.aggregate(source, ref_weights)
        ref_offset = float(np.load(root / 'large_fit/half250/fields.npz')['offset'])
        references = {}; r['reference'] = {}
        for h in range(4):
            stats, fields, curves = snapshot(ref_op, ref_coeff, ref_offset, h)
            references[h] = (fields, curves)
            r['reference'][str(h)] = stats
            store_snapshot('reference', h, fields, curves)
        r['seconds']['reference_prepare_verify_and_save'] = elapsed(begin)
        # Fitting the saved oracle is not included in preparation above.
        r['reference_fit_provenance'] = {'commit': '3befc70',
            'historical_fit_through_mean_seconds': ref_raw['fit_through_mean_seconds'],
            'qualification': 'historical contended fit, not added to a current-machine timing ratio'}
        del ref_coeff, ref_op, ref_weights
        cp.get_default_memory_pool().free_all_blocks()
        cumulative_calibration = 0.
        def checkpoint(iteration, weights, update_seconds):
            nonlocal cumulative_calibration
            entry = {'iteration': iteration, 'update_seconds': update_seconds, 'planes': {}}
            sync(); start = time.perf_counter()
            coeff = op.aggregate(source, weights)
            offset = float(op.evaluate(source, coeff).mean())
            entry['aggregation_centering_seconds'] = elapsed(start)
            retained = {}
            for label, ids in [('calibration', [0, 1]), ('holdout', [2, 3])]:
                start = time.perf_counter()
                for h in ids:
                    stats, fields, curves = snapshot(op, coeff, offset, h)
                    ref_fields, ref_curves = references[h]
                    fine = GRIDS[-1]
                    diff = fields[fine].astype('d') - ref_fields[fine]
                    stats['mean_relative_l2'] = float(np.linalg.norm(diff) / np.linalg.norm(ref_fields[fine].astype('d')))
                    stats['distance'] = curve_distance(curves[fine], ref_curves[fine]) if curves[fine] and ref_curves[fine] else None
                    stats['reference_counts_match'] = all(
                        stats[str(fine)][k] == r['reference'][str(h)][str(fine)][k]
                        for k in ['components', 'open_components'])
                    ref_stats = r['reference'][str(h)]
                    stats['quality'] = {name: bool(
                        stats['mean_relative_l2'] <= .002 and stats['reference_counts_match']
                        and stats['stable_counts'] and ref_stats['stable_counts']
                        and stats['distance'] is not None and stats['distance']['hausdorff_upper'] <= target
                        and stats['refinement'] is not None and stats['refinement']['hausdorff_upper'] <= target
                        and ref_stats['refinement'] is not None and ref_stats['refinement']['hausdorff_upper'] <= target)
                        for name, target in TARGETS.items()}
                    entry['planes'][str(h)] = stats
                    retained[h] = (fields, curves)
                entry[f'{label}_seconds'] = elapsed(start)
                entry[f'{label}_quality'] = {name: all(entry['planes'][str(h)]['quality'][name] for h in ids)
                                            for name in TARGETS}
            cumulative_calibration += entry['aggregation_centering_seconds'] + entry['calibration_seconds']
            entry['calibration_path_seconds'] = r['seconds']['method_setup'] + update_seconds + cumulative_calibration
            entry['with_selected_confirmation_seconds'] = entry['calibration_path_seconds'] + entry['holdout_seconds']
            entry['with_saved_oracle_preparation_seconds'] = entry['with_selected_confirmation_seconds'] + r['seconds']['reference_prepare_verify_and_save']
            entry['offset'] = offset
            start = time.perf_counter()
            for h, (fields, curves) in retained.items():
                store_snapshot(f'checkpoint{iteration}', h, fields, curves)
            coeff_host = cp.asnumpy(coeff)
            np.save(a.output / f'coefficients_{iteration}.npy', coeff_host)
            entry['coefficients_sha256'] = array_digest(coeff_host)
            entry['artifact_seconds'] = elapsed(start)
            r['checkpoints'].append(entry)
            save()
            print(a.method, a.seed, iteration, entry['calibration_path_seconds'],
                  entry['calibration_quality'], entry['holdout_quality'], flush=True)
        start = time.perf_counter()
        op = FourierPoisson(50)
        if a.method == 'half200':
            from geospsr.half_spectral import HalfCubeSpectrum, HalfCubeKernel
            kernel = HalfCubeKernel(source, HalfCubeSpectrum(200), .02)
            rhs = cp.asarray(normals)
            r['seconds']['method_setup'] = elapsed(start)
            observations = [4, 8, 16, 32, 64]
            r['scheduled_checkpoints'] = observations + ['final']
            callback_seconds = 0.
            solve_start = time.perf_counter()
            def observe(iteration, weights):
                nonlocal callback_seconds
                if iteration not in observations:
                    return
                sync(); begin = time.perf_counter()
                checkpoint(iteration, weights, begin - solve_start - callback_seconds)
                callback_seconds += elapsed(begin)
            weights, info = kernel.solve(rhs, rtol=1e-5, maxiter=1000, callback=observe)
            r['finite_solve'] = info
            r['seconds']['updates'] = elapsed(solve_start) - callback_seconds
            if not r['checkpoints'] or r['checkpoints'][-1]['iteration'] != info['iterations']:
                checkpoint(info['iterations'], weights, r['seconds']['updates'])
        else:
            from geospsr.iterative import StochasticDualDescent
            from geospsr.kernels import Matern32Kernel, ProductKernel
            from reproduction.matched_teaser import own_jax_array
            k = Matern32Kernel(.01, .1)
            solver = StochasticDualDescent(source, normals, ProductKernel(k, k, k),
                                           batch=512, lr=5*512/len(source), seed=a.seed)
            solver.compile()
            r['seconds']['method_setup'] = elapsed(start)
            r['scheduled_checkpoints'] = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
            updates = 0.
            for iteration in r['scheduled_checkpoints']:
                start = time.perf_counter()
                value = solver.advance(iteration - solver.iterations)
                weights = own_jax_array(value)
                updates += elapsed(start)
                checkpoint(iteration, weights, updates)
            r['seconds']['updates'] = updates
        np.save(a.output / 'weights.npy', cp.asnumpy(weights))
        r['weights_sha256'] = array_digest(np.load(a.output / 'weights.npy', mmap_mode='r'))
        np.savez_compressed(a.output / 'contours.npz', **archive)
        r['complete'] = True
    except Exception as exc:
        r['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        r['seconds']['instrumented_work_wall'] = time.perf_counter() - work_start
        r['gpu_process_audit'] = audit.stop()
        save()


if __name__ == '__main__':
    main()
