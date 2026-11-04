import numpy as np
import pytest

from geospsr.nonperiodic import FreeSpacePoissonMatern


@pytest.mark.parametrize('columns', [None, 7])
def test_full_kernel_two_dimensional_tiles_match_independent_dense(columns):
    pytest.importorskip('cupy')
    from geospsr.nonperiodic_dense_gpu import full_kernel_matmul
    rng = np.random.default_rng(172)
    x = rng.normal(size=(513, 3))*.2
    model = FreeSpacePoissonMatern(.17, .8)
    rhs = rng.normal(size=(len(x),) if columns is None else (len(x), columns))
    expected = (model.vector_covariance(x, x)+.03*np.eye(len(x)))@rhs
    actual = full_kernel_matmul(model, x, .03, rhs, block_size=128, workspace_bytes=32768)
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)
