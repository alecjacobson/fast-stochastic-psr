import importlib.util
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec('cupy') is None, reason='GPU runtime required')


def test_gpu_contact_queries_match_augmented_gaussian_and_shortcut_mean():
    import cupy as cp
    from geospsr.nonperiodic import FreeSpacePoissonMatern, DenseFreeSpacePosterior
    from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
    from geospsr.scalar_contacts import ContactConditionedDensePosterior
    from reproduction.partial_direct_gpu import DirectContactQueries, cross_rhs
    rng = np.random.default_rng(45)
    x, vectors = rng.normal(size=(31, 3)), rng.normal(size=(31, 3))
    contacts, q = x[:4].copy(), np.r_[x[:2], x[:2]+1e-10, rng.normal(size=(7, 3))]
    model = FreeSpacePoissonMatern(.48, 1/3)
    expected_cross = model.cross_covariance(q, x).transpose(1, 2, 0).reshape(len(x), -1)
    np.testing.assert_allclose(cp.asnumpy(cross_rhs(model, q, x)), expected_cross, atol=1e-15, rtol=0)
    cpu = ContactConditionedDensePosterior(DenseFreeSpacePosterior(model, x, vectors, .04),
                                         contacts, np.zeros(4), .01**2)
    gpu = DirectContactQueries(model, x, vectors, contacts,
                              GPUDenseMaternSystem(model, x, .04, dtype='float64'), dense=True)
    mean, variance = gpu.predict(q, batch_size=3)
    np.testing.assert_allclose((mean, variance), cpu.predict(q), atol=1e-13, rtol=0)
    shortcut_mean, shortcut_variance = gpu.predict(q, batch_size=3, compute_variance=False)
    np.testing.assert_allclose(shortcut_mean, mean, atol=1e-14, rtol=0)
    assert np.all(shortcut_variance >= variance-1e-14)
