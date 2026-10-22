"""CPU-only spectral workspace/mass design check; not a performance benchmark."""
import json
from pathlib import Path
import numpy as np
from geospsr.spectral_blocks import axis_weights, block_audit


def main():
    root = Path(__file__).resolve().parent / 'results/spectral_blocks'
    root.mkdir(parents=True, exist_ok=True)
    # spectral_axis uses NumPy only. Importing its module requires the normal
    # runtime dependencies; this audit performs no GPU operations.
    from geospsr.spectral import spectral_axis
    normalization = []
    for ell in [.005, .01, .05, .1, .5, 1., 10.]:
        n, production = spectral_axis(200, ell)
        ratio = 1/(1+np.exp(-np.sqrt(3)/ell))
        np.testing.assert_allclose(production, axis_weights(n, ell)*ratio, rtol=2e-12)
        normalization.append({'lengthscale': ell, 'production_over_normalized_axis': ratio,
                              'product3_diagonal_deficit': 1-ratio**3})
    (root/'normalization-diagnostic.json').write_text(json.dumps(normalization, indent=2)+'\n')
    rows = [block_audit(core, extent, lengthscale)
            for lengthscale in [.01, .005]
            for core, extent in [(50, 200), (100, 200), (150, 200), (100, 400), (150, 400)]]
    (root/'design.json').write_text(json.dumps(rows, indent=2)+'\n')
    table = '\n'.join(f"| {r['lengthscale']} | {r['core']} | {r['extent']} | "
                      f"{r['retained_modes']:,} / {r['full_cube_modes']:,} | "
                      f"{100*r['omitted_inside_cube_diagonal']/.1**3:.5f} | "
                      f"{100*r['omitted_outside_cube_diagonal']/.1**3:.5f} |"
                      for r in rows)
    (root/'README.md').write_text('''# Positive spectral block design audit

CPU-only design check for [issue #22](https://github.com/alecjacobson/fast-stochastic-psr/issues/22).
Keep a complete core cube and six disjoint positive/negative slabs, each with
one high-frequency coordinate. Every scalar/cross mode is included if its
cutoff is no larger than the core. There are no negative inclusion-exclusion
weights and no overlap. Multiple-high-axis corners are omitted deliberately.

An independent closed-form normalization of the infinite periodic Matérn
spectrum gives the following **prior diagonal mass**. These percentages are
not posterior covariance errors or acceptance gates. Noise regularization,
observation geometry and query directions still determine downstream error.

| Length scale | Core | Extent | Retained / cube modes | Omitted corners % | Outside cube % |
| --- | ---: | ---: | ---: | ---: | ---: |
''' + table + '''

Lower mode storage does not prove lower NUFFT workspace or faster solves:
the GPU algorithm would repeat source spreading across blocks, with padded
FFT grids and plan overhead. It also needs coherent prior sampling and an
omitted-spectrum estimator that includes the excluded corners, not just the
outside-cube tail. No GPU backend or speedup is implemented here.

The normalization check also exposes an inherited mismatch at longer scales:
the production spectrum is smaller than the variance-normalized analytic
spectrum by `1/(1+exp(-sqrt(3)/ell))` per axis. At ell=1 this leaves a 38.66%
3D diagonal deficit even with an infinite Fourier band. At the .01/.005 scales
in the table, the factor rounds to one in float64. The existing production
code is unchanged; [issue #24](https://github.com/alecjacobson/fast-stochastic-psr/issues/24)
tracks correction, released-code compatibility and affected tail diagnostics.
`normalization-diagnostic.json` records the coefficient-ratio check. The block
mass formula uses the normalized analytic law.

Tests enumerate small integer-frequency sets to verify coverage, disjointness,
symmetry and mass conservation, and compare the independent one-dimensional
normalization to production at the audited short scales. An explicit Fourier
sum also reconstructs the analytic covariance at long scales. Reproduce with:

```bash
./reproduction/run_env.sh -m reproduction.audit_spectral_blocks
```
''')


if __name__ == '__main__':
    main()
