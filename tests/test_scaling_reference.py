"""Check the scalable reference against independent dense double precision."""
import numpy as np
import pytest


def test_refined_moments_against_dense_oracle():
    cp=pytest.importorskip('cupy')
    from cupyx.scipy.linalg import solve_triangular
    from reproduction.benchmark_eagle_comparison import cholesky_in_place
    from reproduction.scaling_reference import refined_moments
    rng=np.random.default_rng(81);n=48;noise=.003
    w=rng.normal(size=(n,12));a=w@w.T+noise*np.eye(n)
    c=rng.normal(size=(7,n,3))*1e-4;y=rng.normal(size=(n,3))
    factor=cholesky_in_place(cp.asarray(a,dtype='f'))
    def solve(b):
        z=solve_triangular(factor,cp.asarray(b,dtype='f'),lower=True)
        return solve_triangular(factor,z,lower=True,trans='T').astype('d')
    gpu=cp.asarray(a)
    mean,var,info=refined_moments(solve,lambda z:gpu@z,c,y,1.,noise)
    alpha=np.linalg.solve(a,y);z=np.linalg.solve(a,c.transpose(1,0,2).reshape(n,-1)).reshape(n,-1,3)
    mr=np.einsum('qnd,nd->q',c,alpha);vr=1.-(c.transpose(1,0,2)*z).sum((0,2))
    np.testing.assert_allclose(mean,mr,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(var,vr,rtol=1e-12)
    assert info['relative_residual_history'][-1]<1e-9
