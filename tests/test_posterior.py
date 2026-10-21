from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from geospsr.gp import (compute_covariance, compute_mean, compute_variance,
    evaluate_karhunen_loeve_kernel, f_eigenvalues_from_v_eigenvalues,
    f_gamma_from_v_xi, poisson_cross_covariances, sample_pathwise_conditioning)
from geospsr.utils import trunc_Zd
from geospsr.sdd import sdd_solve


def problem():
    freq=trunc_Zd(1)
    ev=jnp.ones(len(freq))*.1
    fev=f_eigenvalues_from_v_eigenvalues(freq,ev)
    kv=lambda x,y: evaluate_karhunen_loeve_kernel(x-y,freq,ev)
    kfv=lambda x,y: poisson_cross_covariances(x-y,freq,ev)
    kf=lambda x,y: evaluate_karhunen_loeve_kernel(x-y,freq,fev)
    x=jax.random.uniform(jax.random.key(2),(3,3),minval=-2,maxval=2)
    q=jax.random.uniform(jax.random.key(3),(4,3),minval=-2,maxval=2)
    y=jax.random.normal(jax.random.key(4),(3,3))
    return freq,ev,fev,kv,kfv,kf,x,q,y


def test_covariance_diagonal_and_psd():
    freq,ev,fev,kv,kfv,kf,x,q,y=problem()
    cov=compute_covariance(q,x,kv,kfv,kf,.1)
    var=compute_variance(q,x,kv,kfv,jnp.sum(fev**2),.1)
    np.testing.assert_allclose(jnp.diag(cov),var,atol=2e-7)
    assert float(jnp.linalg.eigvalsh(cov).min())>-1e-7
    prior=jax.vmap(jax.vmap(kf,(None,0)),(0,None))(q,q)
    assert float(jnp.linalg.eigvalsh(prior-cov).min())>-1e-7


def test_pathwise_empirical_moments():
    freq,ev,fev,kv,kfv,kf,x,q,y=problem()
    xi=jax.random.normal(jax.random.key(8),(8192,3,len(freq),2))
    gamma=f_gamma_from_v_xi(xi,freq,ev)
    s=sample_pathwise_conditioning(q,x,y,kv,kfv,.1,xi,gamma,freq,ev,freq,fev,4,4,4,noise_key=jax.random.key(9))
    mean=compute_mean(q,x,y,kv,kfv,.1)
    cov=compute_covariance(q,x,kv,kfv,kf,.1)
    np.testing.assert_allclose(s.mean(0),mean,atol=.015)
    np.testing.assert_allclose(np.cov(np.asarray(s).T),cov,atol=.006)


def test_sdd_and_cholesky_solve_same_noise_model():
    *_,kv,kfv,kf,x,q,y=problem()
    K=jax.vmap(jax.vmap(kv,(None,0)),(0,None))(x,x)+.3*jnp.eye(len(x))
    alpha=sdd_solve(x,y,kv,.3,jax.random.key(0),bs=3,lr=.05,iterations=1500,polyak=.1)
    np.testing.assert_allclose(K@alpha,y,atol=1e-4,rtol=1e-4)
