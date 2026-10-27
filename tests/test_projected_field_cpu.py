import numpy as np
import pytest
pytest.importorskip('finufft')
from reproduction.projected_field_cpu import prepare


@pytest.mark.parametrize('rank',[0,10,17])
def test_cpu_nufft_preparation_against_independent_spatial_and_fourier_controls(rank):
    rng=np.random.default_rng(5408);x=rng.uniform(-np.pi,np.pi,(19,3));q=rng.uniform(-np.pi,np.pi,(17,3))
    y=rng.normal(size=x.shape);a=prepare(x,q,y,rank=rank,cutoff=3,lengthscale=.2,noise=.0002)
    op=a['operator'];n=len(x);field=a['sampler']
    new=np.vstack([q,rng.normal(size=(5,3))]);phase=np.exp(1j*(new@op.freq.T))
    prior=((phase*op.scalar)@phase.conj().T).real
    cross=np.sin((new[:,None]-x[None,:])@op.freq.T)
    cross=[cross@(op.scalar*op.freq[:,d]) for d in range(3)]
    # Independent image sum for the vector covariance.
    delta=(x[:,None]-x[None,:])/(2*np.pi);kernel=np.zeros_like(delta)
    for k in range(-20,21):
        t=np.abs(delta+k);kernel+=(1+op.a*t)*np.exp(-op.a*t)
    matrix=.1**3*np.prod(kernel/op.normalization,axis=2)+.0002*np.eye(n)
    target=prior-sum(c@np.linalg.solve(matrix,c.T) for c in cross)
    lift=(phase@a['lift_coefficients'].T).real
    l=(phase@a['cross_coefficients'].T).real
    actual_g=field.scalar_gram+field.remainder_root@field.remainder_root.T
    covariance=prior-l@lift.T-lift@l.T+lift@actual_g@lift.T
    residual=[c.T-matrix@a['weights'][d].T@lift.T for d,c in enumerate(cross)]
    error=sum(r.T@np.linalg.solve(matrix,r) for r in residual)
    np.testing.assert_allclose(covariance-target,error,rtol=1e-9,atol=1e-13)
    if rank==17:np.testing.assert_allclose(covariance[:17,:17],target[:17,:17],rtol=1e-8,atol=1e-13)
    alpha=np.linalg.solve(matrix,y)
    mean=sum(c@alpha[:,d] for d,c in enumerate(cross))-a['offset']
    np.testing.assert_allclose(op.evaluate(new,a['mean_coefficients'][None,:])[0]-a['offset'],mean,rtol=1e-9,atol=1e-13)
    np.testing.assert_allclose(lift[:17],a['basis']*np.sqrt(a['eigenvalues']),rtol=1e-9,atol=1e-12)
