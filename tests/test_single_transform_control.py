import numpy as np


def test_cross_components_share_the_same_finite_spectrum():
    import cupy as cp
    from geospsr.spectral import FourierPoisson
    from reproduction.single_transform_control import cross_values_single
    model=FourierPoisson(5,.03,.0025**(1/3),dtype='float64',eps=1e-11)
    points=np.random.default_rng(7).normal(size=(23,3))
    expected=cp.asnumpy(model.cross_values(points))
    actual=cp.asnumpy(cross_values_single(model,points))
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-13)
