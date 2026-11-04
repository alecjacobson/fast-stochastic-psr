import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
import pytest

from geospsr.spectral import FourierPoisson
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.variance_series import prepare_variance_series, FourierSeriesEvaluator


@pytest.mark.parametrize('ell',[.01,.2,2.])
def test_float32_dense_baseline_against_independent_analytic_formula(ell):
    from reproduction.dense_covariance import analytic_covariance_rows32
    x=np.random.default_rng(731).uniform(-8,8,(39,3)).astype('f')
    actual=analytic_covariance_rows32(x,x,lengthscale=ell)
    expected=analytic_covariance_rows64(x,x,lengthscale=ell)
    np.testing.assert_allclose(actual.get(),expected.get(),atol=2e-9,rtol=3e-5)
    np.testing.assert_allclose(cp.diag(actual).get(),.001,atol=1e-9,rtol=0)


def test_variance_series_matches_dense_factor_at_new_queries():
    rng = np.random.default_rng(726)
    x = rng.uniform(-1, 1, (11,3)).astype('f')
    q = rng.uniform(-1, 1, (29,3)).astype('f')
    a = analytic_covariance_rows64(x,x,lengthscale=.2)
    a.flat[::12] += .02
    chol = cp.linalg.cholesky(a)
    op = FourierPoisson(2, lengthscale=.2, dtype='float64', eps=1e-12)
    c = op.cross_values((q[:,None].astype('d')-x[None].astype('d')).reshape(-1,3)).reshape(len(q),len(x),3)
    z = cla.solve_triangular(chol, c.transpose(1,0,2).reshape(len(x),-1), lower=True)
    coeff, record = prepare_variance_series(x,chol,op,batch=4)
    expected = record['prior_variance']-(z*z).sum(0).reshape(-1,3).sum(1)
    actual = FourierSeriesEvaluator(coeff).evaluate(q)[0]
    np.testing.assert_allclose(actual.get(), expected.get(), atol=1e-14, rtol=1e-10)
    assert record['functions'] == 3*len(x)
    assert record['variance_cutoff'] == 4


def test_tabulated_cross_matches_independent_scipy_interpolation():
    from scipy.interpolate import RegularGridInterpolator
    from reproduction.tabulated_cross import TabulatedCrossKernel
    rng=np.random.default_rng(73)
    op=FourierPoisson(2,lengthscale=.2,dtype='float64',eps=1e-12)
    table=TabulatedCrossKernel(op,16)
    x=rng.uniform(-np.pi,np.pi,(11,3)).astype('f')
    q=rng.uniform(-np.pi,np.pi,(9,3)).astype('f')
    knots=table.knots.get()
    interp=RegularGridInterpolator((knots,)*3,table.values.get().reshape(16,16,16,3))
    delta=(q[:,None].astype('d')-x[None].astype('d')+np.pi)%(2*np.pi)-np.pi
    expected=interp(delta.reshape(-1,3)).reshape(9,11,3)
    np.testing.assert_allclose(table.rows(q,x).get(),expected,atol=1e-17,rtol=1e-11)


def test_packed_mean_and_variance_preserve_small_spatial_contrast():
    from geospsr.cached_spectral import CachedFourierEvaluator
    c=cp.zeros((3,7,7,7),dtype='complex128')
    c[0,4,3,3]=c[0,2,3,3]=1e-4
    c[1,3,3,3]=1e-8
    c[1,3,4,3]=c[1,3,2,3]=1e-11
    c[2,3,3,3]=2.  # Odd final batch, constant-only zero-amplitude channel.
    q=cp.asarray(np.random.default_rng(29).uniform(-np.pi,np.pi,(101,3)),dtype='f')
    cache=CachedFourierEvaluator(c,oversampling=2.,batch=1)
    actual=cache.evaluate(q).get()
    q=q.get().astype('d')
    expected=np.stack([2e-4*np.cos(q[:,0]),1e-8+2e-11*np.cos(q[:,1]),np.full(len(q),2.)])
    assert np.linalg.norm(actual[0]-expected[0])/np.linalg.norm(expected[0]) < 1e-5
    assert np.linalg.norm(actual[1]-expected[1])/np.linalg.norm(expected[1]-1e-8) < 1e-4
    np.testing.assert_array_equal(actual[2],expected[2])


def test_spatial_moment_grid_matches_scipy_and_rejects_outside_domain():
    import pytest
    from scipy.ndimage import map_coordinates as scipy_map
    from reproduction.spatial_moment_grid import SpatialMomentGrid
    rng=np.random.default_rng(37)
    values=rng.normal(size=(2,12,12,12))
    q=rng.uniform(-.8,.8,(51,3)).astype('f')
    grid=SpatialMomentGrid(values,[-1]*3,[1]*3)
    actual=grid.evaluate(q)
    coords=((q.astype('d')+1)*11/2).T
    for i in range(2):
        expected=scipy_map(values[i],coords,order=3,mode='mirror')
        np.testing.assert_allclose(actual[i].get(),expected,atol=1e-12,rtol=1e-11)
    with pytest.raises(ValueError):grid.evaluate([[1.01,0,0]])


def test_periodic_spatial_grid_matches_scipy_across_seam():
    from scipy.ndimage import map_coordinates as scipy_map
    from reproduction.spatial_moment_grid import SpatialMomentGrid
    values=np.random.default_rng(38).normal(size=(2,12,12,12))
    q=np.array([[-1,0,.99],[1,0,.99],[.999,.998,-.999]])
    grid=SpatialMomentGrid(values,[-1]*3,[1]*3,periodic=True)
    actual=grid.evaluate(q)
    for i in range(2):
        expected=scipy_map(values[i],((q+1)*6).T,order=3,mode='grid-wrap')
        np.testing.assert_allclose(actual[i].get(),expected,atol=1e-12,rtol=1e-11)
        np.testing.assert_allclose(actual[i][0].get(),actual[i][1].get(),atol=1e-12)
