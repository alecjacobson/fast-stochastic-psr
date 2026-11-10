import numpy as np
import pytest
from scipy.linalg import block_diag

from geospsr.nonperiodic import FreeSpacePoissonMatern


def fixture():
    rng=np.random.default_rng(129)
    x=rng.normal(size=(64,3))*.03;y=rng.normal(size=x.shape)
    return FreeSpacePoissonMatern(.015,1/3),x,y,x[::8],rng.normal(size=(11,3))*.03


def augmented(model,x,y,c,q,noise,scalar_noise):
    a=model.vector_covariance(x,x)+noise*np.eye(len(x))
    bc=model.cross_covariance(c,x).transpose(0,2,1).reshape(len(c),-1)
    bq=model.cross_covariance(q,x).transpose(0,2,1).reshape(len(q),-1)
    joint=np.block([[block_diag(a,a,a),bc.T],[bc,model.scalar_covariance(c,c)+scalar_noise*np.eye(len(c))]])
    cross=np.column_stack((bq,model.scalar_covariance(q,c)))
    mean=cross@np.linalg.solve(joint,np.r_[y.T.ravel(),np.zeros(len(c))])
    var=model.prior_variance-np.sum(cross*np.linalg.solve(joint,cross.T).T,axis=1)
    return mean,var


@pytest.mark.parametrize('scalar_noise',[7.5e-9,2e-6])
def test_shared_moments_honor_scalar_noise(scalar_noise):
    from reproduction.partial_competitor_common import conditioned_moments,projected_moments
    model,x,y,c,q=fixture();noise=.02**2
    a=model.vector_covariance(x,x)+noise*np.eye(len(x))
    cross=model.cross_covariance(np.vstack((c,q)),x).transpose(1,2,0).reshape(len(x),-1)
    rhs=np.column_stack((cross,y));z=np.linalg.solve(a,rhs)
    got=conditioned_moments(model,c,q,rhs,z,np,scalar_variance=scalar_noise)
    expected=augmented(model,x,y,c,q,noise,scalar_noise)
    np.testing.assert_allclose(got,expected,atol=1e-12,rtol=0)
    got=projected_moments(model,c,q,cross,np.linalg.inv(a),np.linalg.solve(a,y),scalar_variance=scalar_noise)
    np.testing.assert_allclose(got,expected,atol=1e-12,rtol=0)


def test_component_cross_matches_independent_kernel():
    cp=pytest.importorskip('cupy')
    from reproduction.partial_direct_gpu import cross_rhs
    model,x,y,c,q=fixture();q=np.vstack((q,x[:2],x[:2]+1e-12,[[10,20,30]]))
    expected=model.cross_covariance(q,x)
    full=cp.asnumpy(cross_rhs(model,q,x))
    for d in range(3):
        got=cp.asnumpy(cross_rhs(model,q,x,component=d))
        np.testing.assert_allclose(got,expected[:,:,d].T,atol=1e-15,rtol=1e-12)
        np.testing.assert_allclose(got,full[:,d*len(q):(d+1)*len(q)],atol=1e-15,rtol=0)
    with pytest.raises(ValueError,match='component'):cross_rhs(model,q,x,component=3)


@pytest.mark.parametrize('method',['plain','plain-assembled64','corrected','float64'])
def test_bounded_dense_moments_against_augmented_gaussian(method):
    pytest.importorskip('cupy')
    from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
    from reproduction.benchmark_observed_fidelity import component_moments
    model,x,y,c,q=fixture();noise=.02**2;scalar_noise=7.5e-9
    system=GPUDenseMaternSystem(model,x,noise,dtype='float64' if method=='float64' else 'float32',
        kernel_dtype='float64' if method=='plain-assembled64' else None)
    mean,var,_=component_moments(system,model,x,y,c,q,scalar_noise,method)
    expected=augmented(model,x,y,c,q,noise,scalar_noise)
    np.testing.assert_allclose((mean,var),expected,atol=1e-7 if method.startswith('plain') else 1e-12,rtol=0)
