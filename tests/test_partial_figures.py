"""Check the contact-folded mean used by the illustration display grid."""
import numpy as np
import pytest


def test_folded_mean_matches_independent_augmented_posterior():
    pytest.importorskip('cupy')
    from geospsr.nonperiodic import FreeSpacePoissonMatern, DenseFreeSpacePosterior
    from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
    from geospsr.scalar_contacts import ContactConditionedDensePosterior
    from reproduction.partial_direct_gpu import DirectContactQueries
    from reproduction.compute_partial_figures import mean_coefficients, mean_field
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(96, 3))*.3
    y = rng.normal(size=x.shape)
    q = np.vstack((x[:5], rng.normal(size=(33, 3))*.5))
    model = FreeSpacePoissonMatern(.48, 1/3)
    system = GPUDenseMaternSystem(model, x, .04, dtype='float64')
    posterior = DirectContactQueries(model, x, y, x[:5], system, dense=True)
    exact = ContactConditionedDensePosterior(DenseFreeSpacePosterior(model,x,y,.04),x[:5],np.zeros(5),.0001)
    mean = mean_field(posterior,q,mean_coefficients(posterior),batch_size=7)
    np.testing.assert_allclose(mean,exact.predict(q)[0],rtol=1e-10,atol=2e-11)
