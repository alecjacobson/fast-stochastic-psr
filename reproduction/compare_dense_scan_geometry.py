"""Compare the winning dense-scan regime with original SPSR and exact shape.

Numerical algorithms/models differ. This is a geometry/uncertainty profile
comparison, not a pointwise-equivalence or probability-calibration test.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.hierarchical import HierarchicalMaternSystem
from geospsr.hierarchical_gpu import GPUHierarchicalSystem
from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.benchmark_hierarchical_variance import data


def main(path, destination):
    import gpytoolbox as gpy
    import matplotlib.pyplot as plt
    a = json.loads(Path(path).read_text())['arguments']
    out = Path(destination); out.mkdir(parents=True, exist_ok=True)
    x, normals, axes = data(a['n'], a['seed'], a['geometry'])
    model = FreeSpacePoissonMatern(a['lengthscale'], a['amplitude'])
    noise = model.amplitude*a['relative_noise']
    radius = np.linspace(.94, 1.06, 301)
    vradius = np.array([.94, .97, .99, .995, 1., 1.005, 1.01, 1.03, 1.06])
    queries = np.concatenate([radius[:, None]*np.eye(3)[d]*axes[d] for d in range(3)])
    vqueries = np.concatenate([vradius[:, None]*np.eye(3)[d]*axes[d] for d in range(3)])
    gs = np.full(3, 16); h = np.full(3, 3.2/15); corner = np.full(3, -1.6)
    start = perf_counter()
    original_mean, original_variance, _ = gpy.stochastic_poisson_surface_reconstruction(
        x, normals, gs=gs, h=h, corner=corner, output_variance=True, solve_subspace_dim=0,
        sigma_n=np.sqrt(noise), sigma=np.sqrt(model.amplitude))
    original_seconds = perf_counter()-start
    w = gpy.fd_interpolate(queries, gs=gs, h=h, corner=corner)
    om, ov = np.asarray(w@original_mean).ravel(), np.asarray(w@original_variance).ravel()
    start = perf_counter()
    system = GPUHierarchicalSystem(HierarchicalMaternSystem(
        model, x, noise, tolerance=1e-5, leaf_size=40, preconditioner_rank=8, neighbors=24))
    alpha = system.solve(normals, tolerance=1e-8)
    level = float(np.einsum('mnd,nd->m', model.cross_covariance([[1, 0, 0]], x), alpha)[0])
    mean = np.concatenate([np.einsum('mnd,nd->m', model.cross_covariance(queries[i:i+32], x), alpha)
                           for i in range(0, len(queries), 32)])-level
    c = model.cross_covariance(vqueries, x)
    rhs = c.transpose(1, 2, 0).reshape(len(x), -1)
    z = system.solve(rhs, tolerance=1e-8)
    variance = model.prior_variance-np.sum(rhs*z, axis=0).reshape(3, len(vqueries)).sum(axis=0)
    vm = np.einsum('mnd,nd->m', c, alpha)-level
    if np.any(variance <= 0):
        raise RuntimeError('nonpositive direct profile variance')
    candidate_seconds = perf_counter()-start
    fig, plots = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    rows = []
    for d, axis in enumerate('xyz'):
        sl, vl = slice(d*len(radius), (d+1)*len(radius)), slice(d*len(vradius), (d+1)*len(vradius))
        row = dict(axis=axis)
        for name, values in [('Original 2022', om[sl]), ('Free-space model', mean[sl])]:
            crossings = np.flatnonzero(values[:-1]*values[1:] < 0)
            roots = [float(radius[i]-values[i]*(radius[i+1]-radius[i])/(values[i+1]-values[i])) for i in crossings]
            roots += radius[values == 0].tolist()
            row[name] = dict(normalized_surface_crossings=roots, mean=values.tolist())
        plots[0, d].plot(radius, ndtr(-om[sl]/np.sqrt(ov[sl])), label='Original 2022, 16³ grid', color='C1')
        plots[0, d].plot(vradius, ndtr(-vm[vl]/np.sqrt(variance[vl])), 'o-', label='Free-space, direct profile samples', color='C0', markersize=3)
        plots[1, d].plot(radius, ov[sl], color='C1')
        plots[1, d].plot(vradius, variance[vl], 'o-', color='C0', markersize=3)
        plots[1, d].set_yscale('log')
        plots[0, d].set_title(f'{axis} axis'); plots[1, d].set_xlabel('radius / true surface radius')
        for plot in plots[:, d]:
            plot.axvline(1, color='k', linestyle=':')
        rows.append(row)
    plots[0, 0].set_ylabel('Marginal inside probability')
    plots[1, 0].set_ylabel('Reported variance (model-dependent scale)')
    plots[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'profiles.png', dpi=160)
    package = Path(inspect.getfile(gpy.stochastic_poisson_surface_reconstruction)).parent
    result = dict(arguments=a, axes=axes.tolist(), gs=gs.tolist(), h=h.tolist(), corner=corner.tolist(),
                  sigma_n=float(np.sqrt(noise)), sigma=float(np.sqrt(model.amplitude)), solve_subspace_dim=0,
                  gpytoolbox_version=importlib.metadata.version('gpytoolbox'),
                  original_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')},
                  candidate_source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ['geospsr/nonperiodic.py', 'geospsr/hierarchical.py', 'geospsr/hierarchical_gpu.py',
                     'reproduction/compare_dense_scan_geometry.py']},
                  original_seconds=original_seconds, candidate_profile_seconds=candidate_seconds,
                  candidate_h2_tolerance=1e-5, candidate_solve_tolerance=1e-8, candidate_level=level,
                  normalized_radius=radius.tolist(), variance_normalized_radius=vradius.tolist(), rows=rows,
                  original_variance=ov.tolist(), candidate_variance=variance.tolist(), candidate_profile_mean=vm.tolist(),
                  interpretation='Same observations and normals, different kernels/boundaries/gauges. Original published variance shift preserved. Candidate variance unshifted; its x contact is anchored by construction. Coarse original grid is not a convergence result; neither profile establishes calibration or a speed comparison.')
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({row['axis']: {k: v['normalized_surface_crossings'] for k, v in row.items() if isinstance(v, dict)} for row in rows}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record'); parser.add_argument('--output', required=True)
    a = parser.parse_args(); main(a.record, a.output)
