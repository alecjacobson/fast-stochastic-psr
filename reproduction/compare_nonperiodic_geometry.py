"""Phenomenological comparison with unmodified 2022 SPSR on known geometry.

Different probabilistic models and boundary/gauge conventions: no pointwise
equivalence or uncertainty-calibration claim. Shape labels support geometry and
occupancy diagnostics, not a ground-truth implicit-field variance.
"""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import DenseFreeSpacePosterior, FreeSpacePoissonMatern
from reproduction.benchmark_hierarchical_variance import data


def main():
    import gpytoolbox as gpy
    import matplotlib.pyplot as plt
    out = Path('reproduction/results/hierarchical_variance/geometry')
    out.mkdir(parents=True, exist_ok=True)
    x, normals, axes = data(512, 12, 'ellipsoid')
    t = np.linspace(.5, 1.5, 201)
    q = np.concatenate([t[:, None]*np.eye(3)[d]*axes[d] for d in range(3)])
    gs = np.full(3, 16); h = np.full(3, 3.2/15); corner = np.full(3, -1.6)
    begin = perf_counter()
    mean, variance, _ = gpy.stochastic_poisson_surface_reconstruction(
        x, normals, gs=gs, h=h, corner=corner, output_variance=True,
        solve_subspace_dim=0, sigma_n=.01, sigma=.05)
    original_seconds = perf_counter()-begin
    interpolate = gpy.fd_interpolate(q, gs=gs, h=h, corner=corner)
    original_mean = np.asarray(interpolate@mean).ravel()
    original_var = np.asarray(interpolate@variance).ravel()
    model = FreeSpacePoissonMatern(.3, .0025)
    posterior = DenseFreeSpacePosterior(model, x, normals, .000025)
    new_mean, new_var = posterior.predict(q)
    level = float(posterior.predict([[1, 0, 0]])[0][0])
    new_mean -= level
    fig, plots = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    rows = []
    for d, axis in enumerate('xyz'):
        sl = slice(d*len(t), (d+1)*len(t))
        row = dict(axis=axis)
        for name, mu, var, color in [('Original 2022', original_mean[sl], original_var[sl], 'C1'),
                                    ('Free-space model', new_mean[sl], new_var[sl], 'C0')]:
            p = ndtr(-mu/np.sqrt(var))
            plots[0, d].plot(t, p, label=name, color=color)
            plots[1, d].plot(t, var, color=color)
            crossings = np.flatnonzero(mu[:-1]*mu[1:] <= 0)
            roots = [float(t[i]-mu[i]*(t[i+1]-t[i])/(mu[i+1]-mu[i])) for i in crossings]
            row[name] = dict(mean=mu.tolist(), variance=var.tolist(), inside_probability=p.tolist(),
                             normalized_surface_crossings=roots,
                             shape_label_brier=float(np.mean((p-(t<1))**2)))
        plots[0, d].set_title(f'{axis} axis'); plots[0, d].axvline(1, color='k', linestyle=':')
        plots[1, d].axvline(1, color='k', linestyle=':'); plots[1, d].set_xlabel('radius / true surface radius')
        rows.append(row)
    plots[0, 0].set_ylabel('marginal inside probability'); plots[1, 0].set_ylabel('reported marginal variance')
    plots[0, 0].legend(); fig.tight_layout(); fig.savefig(out/'profiles.png', dpi=160)
    package = Path(inspect.getfile(gpy.stochastic_poisson_surface_reconstruction)).parent
    record = dict(n=512, seed=12, geometry='ellipsoid', axes=axes.tolist(), grid_size=gs.tolist(),
                  h=h.tolist(), corner=corner.tolist(), sigma_n=.01, sigma=.05, solve_subspace_dim=0,
                  gpytoolbox_version=importlib.metadata.version('gpytoolbox'), original_seconds=original_seconds,
                  original_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')},
                  candidate_lengthscale=.3, candidate_amplitude=.0025, candidate_noise_variance=.000025,
                  candidate_level=level, normalized_radius=t.tolist(), rows=rows,
                  interpretation='Published variance shift and mean centering preserved for original. Candidate uses deterministic contact-point mean level, with unshifted variance. Different models: geometry/probability profiles only, no pointwise-equivalence or calibration claim.')
    (out/'comparison.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({r['axis']: {k: v['normalized_surface_crossings'] for k, v in r.items() if isinstance(v, dict)} for r in rows}), flush=True)


if __name__ == '__main__':
    main()
