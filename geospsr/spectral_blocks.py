"""CPU design audit for positive anisotropic product-Matern spectra.

No NUFFT backend, sampler or posterior-error guarantee is implemented here.
Blocks include the complete core cube and at most one high-frequency axis.
Bounds are inclusive; paired positive/negative slabs preserve symmetry.
"""
import numpy as np


def axis_weights(frequencies, lengthscale=.01, variance=.1):
    """Independent normalized periodic Matern-3/2 Fourier coefficients.

    sum_n (n*n+a*a)^-2 = pi*coth(pi*a)/(2*a^3)
                           + pi^2*csch(pi*a)^2/(2*a^2).
    This closed sum normalizes the infinite discrete law, not a finite cube.
    This independent formula also detects the inherited released spectrum's
    normalization defect (issue #24); the current default corrects that defect.
    """
    if lengthscale <= 0 or variance <= 0:
        raise ValueError('lengthscale and variance must be positive')
    a = np.sqrt(3) / (2*np.pi*lengthscale)
    z = np.pi*a
    denominator = -np.expm1(-2*z)
    coth = (1 + np.exp(-2*z)) / denominator
    csch2 = 4*np.exp(-2*z) / denominator**2
    total = np.pi*coth/(2*a**3) + np.pi**2*csch2/(2*a**2)
    return variance / total / (np.asarray(frequencies, dtype=float)**2 + a*a)**2


def one_high_axis_blocks(core, extent):
    """Seven disjoint boxes, or one if core == extent; all coefficients positive."""
    if core != int(core) or extent != int(extent) or not 0 <= core <= extent:
        raise ValueError('require integer 0 <= core <= extent')
    core, extent = int(core), int(extent)
    blocks = [tuple([(-core, core)]*3)]
    if core < extent:
        for axis in range(3):
            for interval in [(-extent, -core-1), (core+1, extent)]:
                block = [(-core, core)]*3
                block[axis] = interval
                blocks.append(tuple(block))
    return blocks


def block_audit(core, extent, lengthscale=.01, variance=.1):
    blocks = one_high_axis_blocks(core, extent)
    low = axis_weights(np.arange(-core, core+1), lengthscale, variance).sum()
    # Sum the slabs directly instead of subtracting nearly equal sums.
    extra = 2*axis_weights(np.arange(core+1, extent+1), lengthscale, variance).sum()
    outer = low+extra
    retained = low**3 + 3*extra*low**2
    omitted_corners = 3*extra**2*low + extra**3
    fraction = min(outer/variance, 1.)
    outside = variance**3 * -np.expm1(3*np.log(fraction))
    modes = sum(int(np.prod([hi-lo+1 for lo, hi in block])) for block in blocks)
    return {'core': core, 'extent': extent, 'lengthscale': lengthscale,
            'variance_per_axis': variance, 'blocks': blocks, 'retained_modes': modes,
            'full_cube_modes': (2*extent+1)**3, 'retained_prior_diagonal': retained,
            'omitted_inside_cube_diagonal': omitted_corners,
            'omitted_outside_cube_diagonal': float(outside),
            'omitted_total_diagonal_fraction': float((outside+omitted_corners)/variance**3),
            'scope': 'prior diagonal mass only; no posterior quality or timing claim'}
