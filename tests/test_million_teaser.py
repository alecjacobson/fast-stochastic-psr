"""Dense-reference checks for the matrix-free teaser's statistical fields."""
import cupy as cp
import numpy as np
from reproduction.million_teaser import FinitePosterior


def test_finite_posterior_mean_and_variance_against_dense():
    rng=np.random.default_rng(5)
    x=rng.uniform(-1,1,(12,3)).astype('f')
    y=rng.normal(size=(12,3)).astype('f')
    q=rng.uniform(-1,1,(9,3)).astype('f')
    model=FinitePosterior(x,y,cutoff=2,lengthscale=.1,noise_variance=.001)
    op=model.op
    diff=(cp.asarray(x)[:,None]-cp.asarray(x)[None]).reshape(-1,3)
    K=op.evaluate(diff,op.rho)[0].reshape(len(x),len(x))+.001*cp.eye(len(x),dtype='f')
    alpha=cp.linalg.solve(K,cp.asarray(y))
    cross=op.cross_values((cp.asarray(q)[:,None]-cp.asarray(x)[None]).reshape(-1,3)).reshape(len(q),len(x),3)
    mean=cp.einsum('qnd,nd->q',cross,alpha)-model.offset
    np.testing.assert_allclose(cp.asnumpy(model.mean(q)),cp.asnumpy(mean),atol=2e-6,rtol=1e-4)
    n2=(op.ns**2).sum(0)
    prior=cp.sum(cp.where(n2>0,op.rho/cp.where(n2>0,n2,1),0))
    rhs=cross.transpose(1,0,2).reshape(len(x),-1)
    solved=cp.linalg.solve(K,rhs).reshape(len(x),len(q),3).transpose(1,0,2)
    expected=prior-cp.sum(cross*solved,axis=(1,2))
    actual,draw_mean,records=model.variance(q,count=2048,batch=64,seed=62,rtol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(actual),cp.asnumpy(expected),rtol=.12,atol=1e-6)
    assert all(r['converged'] for r in records)
    assert float(cp.max(cp.abs(draw_mean)/cp.sqrt(expected)))<.1
