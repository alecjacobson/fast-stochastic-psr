import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from geospsr.spectral import FourierPoisson
from geospsr.gp import poisson_cross_covariances, evaluate_karhunen_loeve_sample, f_gamma_from_v_xi, f_eigenvalues_from_v_eigenvalues
from geospsr.kernels import Matern32Kernel, ProductKernel
from geospsr.utils import trunc_Zd


def test_nufft_cross_sum_and_spectrum():
    op=FourierPoisson(cutoff=4,lengthscale=.04,eps=1e-6)
    k=Matern32Kernel(.04,.1); kv=ProductKernel(k,k,k)
    freq=trunc_Zd(4); ev=jax.vmap(kv.spectral_density)(freq)**.5
    np.testing.assert_allclose(cp.asnumpy(op.rho).ravel(),np.asarray(ev**2),rtol=2e-5)
    rng=np.random.default_rng(0)
    x=rng.uniform(-np.pi,np.pi,(13,3)).astype('f'); q=rng.uniform(-np.pi,np.pi,(17,3)).astype('f')
    alpha=rng.normal(size=(13,3)).astype('f')
    exact=jax.vmap(jax.vmap(lambda a,b: poisson_cross_covariances(a-b,freq,ev),(None,0)),(0,None))(jnp.asarray(q),jnp.asarray(x))
    expected=np.einsum('qnd,nd->q',exact,alpha)
    coeff=op.aggregate(x,alpha)
    actual=cp.asnumpy(op.evaluate(q,coeff))[0]
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-10)
    shifted=cp.asnumpy(op.evaluate(q+np.float32(2*np.pi),coeff))[0]
    np.testing.assert_allclose(shifted,expected,rtol=3e-4,atol=3e-10)


def test_nufft_joint_prior_matches_upstream():
    op=FourierPoisson(cutoff=2,lengthscale=.04)
    freq=trunc_Zd(2); ev=jnp.sqrt(jnp.asarray(cp.asnumpy(op.rho).ravel()))
    xi=jax.random.normal(jax.random.key(0),(2,3,len(freq),2))
    gamma=f_gamma_from_v_xi(xi,freq,ev)
    fev=f_eigenvalues_from_v_eigenvalues(freq,ev)
    q=jax.random.uniform(jax.random.key(1),(11,3))
    v,f=op.prior_coefficients(np.asarray(xi))
    actual=cp.asnumpy(op.evaluate(np.asarray(q),f))
    expected=jax.vmap(jax.vmap(lambda x,g: evaluate_karhunen_loeve_sample(x,freq,fev,g),(0,None)),(None,0))(q,gamma)
    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-8)


def test_fourier_training_matvec_and_solve():
    from geospsr.spectral import FourierKernel
    op=FourierPoisson(cutoff=3,lengthscale=.1,eps=1e-6)
    rng=np.random.default_rng(32)
    x=rng.uniform(-np.pi,np.pi,(23,3)).astype('f')
    b=cp.asarray(rng.normal(size=(23,3)).astype('f'))
    phi=cp.exp(1j*(cp.asarray(x)@op.ns.reshape(3,-1)))
    dense=(phi*op.rho.ravel())@phi.conj().T
    dense=dense.real+.02*cp.eye(len(x),dtype='f')
    K=FourierKernel(x,op,.02)
    np.testing.assert_allclose(cp.asnumpy(K(b)),cp.asnumpy(dense@b),rtol=1e-5,atol=1e-7)
    alpha,info=K.solve(b,rtol=1e-5)
    assert info['converged']
    np.testing.assert_allclose(cp.asnumpy(dense@alpha),cp.asnumpy(b),rtol=1e-4,atol=2e-5)


@pytest.mark.parametrize('rng_backend,upsampfac', [('cpu',2.0),('gpu',1.25)])
def test_full_vector_residual_sampler_moments(rng_backend,upsampfac):
    from reproduction.posterior import Posterior
    from geospsr.gp import compute_covariance, compute_mean, evaluate_karhunen_loeve_kernel
    rng=np.random.default_rng(12)
    x=rng.uniform(-1,1,(8,3)).astype('f'); y=rng.normal(size=(8,3)).astype('f')
    q=rng.uniform(-1,1,(4,3)).astype('f')
    model=Posterior(x,y,lengthscale=.1,variance=.1,noise_variance=.001,cutoff=2,prior_cutoff=1,upsampfac=upsampfac)
    model.draw(4096,seed=46,batch=128,exact=True,rng_backend=rng_backend)
    samples=cp.asnumpy(model.samples(q,batch=128))
    k=Matern32Kernel(.1,.1);kv=ProductKernel(k,k,k);freq=trunc_Zd(2)
    ev=jax.vmap(kv.spectral_density)(freq)**.5
    fev=f_eigenvalues_from_v_eigenvalues(freq,ev)
    kfv=lambda a,b:poisson_cross_covariances(a-b,freq,ev)
    kf=lambda a,b:evaluate_karhunen_loeve_kernel(a-b,freq,fev)
    cov=np.asarray(compute_covariance(jnp.asarray(q),jnp.asarray(x),kv,kfv,kf,.001))
    expected=np.asarray(compute_mean(jnp.asarray(q),jnp.asarray(x),jnp.asarray(y),kv,kfv,.001))-float(model.offset)
    np.testing.assert_allclose(samples.mean(0),expected,atol=.0015)
    np.testing.assert_allclose(np.cov(samples.T),cov,atol=1e-5,rtol=.1)
