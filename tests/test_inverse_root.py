import numpy as np
import pytest
from geospsr.inverse_root import FixedChebyshevInverseRoot, periodic_matern_axis, spatial_box_spectral_bound


@pytest.mark.parametrize('condition',[1.,2.,50.,1000.])
def test_fixed_inverse_root_matches_dense_and_preserves_covariance(condition):
    rng=np.random.default_rng(917)
    orthogonal=np.linalg.qr(rng.normal(size=(23,23)))[0]
    eigenvalues=np.geomspace(.02,.02*condition,23)
    matrix=(orthogonal*eigenvalues)@orthogonal.T
    model=FixedChebyshevInverseRoot(eigenvalues[0],eigenvalues[-1],1e-7)
    # A deterministic matrix of all basis RHS verifies the complete covariance,
    # avoiding the sampling error of a finite ensemble as a correctness test.
    calls=[]
    value=model.apply(lambda x:matrix@x,np.eye(23),progress=lambda i,n:calls.append(i))
    expected=(orthogonal/np.sqrt(eigenvalues))@orthogonal.T
    assert np.linalg.norm(value-expected)/np.linalg.norm(expected)<1e-7
    inverse=(orthogonal/eigenvalues)@orthogonal.T
    assert np.linalg.norm(value@value.T-inverse)/np.linalg.norm(inverse)<2.1e-7
    assert calls==list(range(1,model.degree+1))
    a,b=rng.normal(size=(2,23,3))
    np.testing.assert_allclose(model.apply(lambda x:matrix@x,2*a-b),
                               2*model.apply(lambda x:matrix@x,a)-model.apply(lambda x:matrix@x,b),rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('ell',[.003,.03,1.])
def test_spatial_envelope_covers_dense_spectrum_with_seams_duplicates_and_clusters(ell):
    rng=np.random.default_rng(12)
    x=np.concatenate([rng.uniform(-np.pi,np.pi,(35,3)),rng.normal(0,.01,(35,3)),
                      [[-np.pi+1e-8,0,0],[np.pi-1e-8,0,0],[0,0,0],[0,0,0]]])
    distance=np.abs(x[:,None]-x[None]);distance=np.minimum(distance,2*np.pi-distance)
    matrix=periodic_matern_axis(distance,ell).prod(2)+.00001*np.eye(len(x))
    for grid in [8,16,32]:
        bound=spatial_box_spectral_bound(x,noise_variance=.00001,grid=grid,lengthscale=ell)
        assert np.linalg.eigvalsh(matrix)[-1]<=bound['upper']*(1+1e-12)
        assert matrix.sum(1).max()<=bound['upper']*(1+1e-12)
        assert bound['lower']==.00001
    values=periodic_matern_axis(np.linspace(0,np.pi,1001),ell)
    assert np.all(np.diff(values)<=1e-13)
    assert values[0]==pytest.approx(.1)


def test_limits_and_invalid_inputs():
    with pytest.raises(ValueError,match='required degree'):
        FixedChebyshevInverseRoot(.02,3000,max_degree=2)
    for args in [(0,1), (2,1), (1,np.inf)]:
        with pytest.raises(ValueError):FixedChebyshevInverseRoot(*args)
    model=FixedChebyshevInverseRoot(1,2)
    with pytest.raises(ValueError):model.apply(lambda x:x,np.ones(3,dtype=int))
    with pytest.raises(ValueError):model.apply(lambda x:x,np.full(3,np.nan))
    with pytest.raises(ValueError):spatial_box_spectral_bound(np.zeros((0,3)))
