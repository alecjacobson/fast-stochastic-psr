import numpy as np
import pytest
from scipy.linalg import block_diag

from geospsr.nonperiodic import FreeSpacePoissonMatern


@pytest.mark.parametrize('preconditioner',['prior','triangular-prior','schur-diagonal'])
@pytest.mark.parametrize('cross_storage',['dual','single'])
def test_coupled_action_and_marginals_against_augmented_gaussian(preconditioner,cross_storage):
    cp=pytest.importorskip('cupy')
    from reproduction.coupled_observation_gpu import CoupledObservationSystem
    rng=np.random.default_rng(128);x=rng.normal(size=(48,3))*.03;y=rng.normal(size=x.shape)
    c=x[::6];q=rng.normal(size=(9,3))*.03;targets=rng.normal(size=len(c))*.001
    model=FreeSpacePoissonMatern(.015,1/3);noise=.0004;scalar_noise=7.5e-9
    a=model.vector_covariance(x,x)+noise*np.eye(len(x));aa=cp.asarray(a)
    class Normal:
        def matmul(self,b):return aa.astype(cp.float32)@b.astype(cp.float32)
        def accurate_matmul(self,b):return aa@b
        def precondition(self,b):return cp.linalg.solve(aa,b)
    system=CoupledObservationSystem(Normal(),model,x,c,scalar_noise,preconditioner=preconditioner,cross_storage=cross_storage)
    bc=model.cross_covariance(c,x).transpose(0,2,1).reshape(len(c),-1)
    joint=np.block([[block_diag(a,a,a),bc.T],[bc,model.scalar_covariance(c,c)+scalar_noise*np.eye(len(c))]])
    b=rng.normal(size=(len(joint),5))
    np.testing.assert_allclose(cp.asnumpy(system.accurate_matmul(cp.asarray(b))),joint@b,atol=1e-14,rtol=1e-13)
    assert np.linalg.eigvalsh(joint).min()>0
    cross=np.column_stack((model.cross_covariance(q,x).transpose(0,2,1).reshape(len(q),-1),model.scalar_covariance(q,c)))
    expected_mean=cross@np.linalg.solve(joint,np.r_[y.T.ravel(),targets])
    expected_var=model.prior_variance-np.sum(cross*np.linalg.solve(joint,cross.T).T,axis=1)
    mean,var=system.predict(y,q,iterations=24,refinements=1,scalar_targets=targets)
    np.testing.assert_allclose(mean,expected_mean,atol=1e-11,rtol=0)
    np.testing.assert_allclose(var,expected_var,atol=1e-14,rtol=0)
    assert np.all(var>0)
    with pytest.raises(ValueError,match='scalar noise'):CoupledObservationSystem(Normal(),model,x,c,0.)
