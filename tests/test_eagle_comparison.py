"""Numerical checks for the cross-model benchmark's new shared machinery."""
import numpy as np
import pytest

from reproduction.benchmark_eagle_comparison import assess


def test_probability_gate_includes_mean_level_error_and_variance():
    mean=np.array([-.001,.002,0.]);variance=np.full(3,1e-6)
    assert assess(mean,variance,mean,variance,1.)['passes']
    changed=mean.copy();changed[-1]=.001
    assert not assess(changed,variance,mean,variance,1.)['passes']
    assert not assess(mean,np.array([-1.,1.,1.]),mean,variance,1.)['passes']
    assert not assess(mean,np.full(3,.1),mean,variance,1.)['passes']


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_in_place_cholesky_against_independent_cpu(dtype):
    cp=pytest.importorskip('cupy')
    from cupyx.scipy.linalg import solve_triangular
    from reproduction.benchmark_eagle_comparison import cholesky_in_place
    rng=np.random.default_rng(7);b=rng.normal(size=(32,32));matrix=b@b.T+.3*np.eye(32)
    rhs=rng.normal(size=(32,7));gpu=cp.asarray(matrix,dtype=dtype);pointer=gpu.data.ptr
    lower=cholesky_in_place(gpu)
    assert lower.data.ptr==pointer
    z=solve_triangular(lower,cp.asarray(rhs,dtype=dtype),lower=True)
    result=solve_triangular(lower,z,lower=True,trans='T').get()
    np.testing.assert_allclose(result,np.linalg.solve(matrix,rhs),rtol=2e-5 if dtype=='float32' else 1e-12,
                               atol=2e-5 if dtype=='float32' else 1e-12)
