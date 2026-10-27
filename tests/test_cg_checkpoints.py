"""Observing CG must preserve the solve, including convergence at a checkpoint."""
import cupy as cp
import numpy as np
import pytest
from geospsr.half_spectral import HalfCubeKernel, HalfCubeSpectrum


def test_callback_preserves_iterates_and_final_convergence():
    rng = np.random.default_rng(562)
    x = rng.uniform(-.2, .2, (37, 3))
    y = rng.normal(size=(37, 3))
    k = HalfCubeKernel(x, HalfCubeSpectrum(5, lengthscale=.2, dtype='float64',
                                         eps=1e-11, upsampfac=2.), .00001)
    expected, expected_info = k.solve(y, rtol=1e-9)
    snapshots = []
    actual, info = k.solve(y, rtol=1e-9,
                           callback=lambda iteration, value: snapshots.append((iteration, value.copy())))
    assert info == expected_info and info['converged']
    np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected))
    assert [i for i, _ in snapshots] == list(range(1, info['iterations'] + 1))
    np.testing.assert_array_equal(cp.asnumpy(snapshots[-1][1]), cp.asnumpy(actual))
    assert float(cp.linalg.norm(snapshots[0][1] - actual)) > 0
    halfway = info['iterations'] // 2
    prefix, _ = k.solve(y, rtol=1e-9, maxiter=halfway)
    np.testing.assert_array_equal(cp.asnumpy(snapshots[halfway - 1][1]), cp.asnumpy(prefix))


def test_callback_error_propagates_and_pcg_is_explicitly_unsupported():
    k = HalfCubeKernel(np.zeros((2, 3)), HalfCubeSpectrum(1), .02)
    def fail(iteration, value):
        raise RuntimeError('checkpoint failed')
    with pytest.raises(RuntimeError, match='checkpoint failed'):
        k.solve(np.ones((2, 3)), callback=fail)
    with pytest.raises(ValueError, match='unpreconditioned'):
        k.solve(np.ones((2, 3)), callback=fail, preconditioner=lambda x: x)
