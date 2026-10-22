"""Long-scale analytic/Fourier consistency and explicit released compatibility."""
import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geospsr.kernels import Matern32Kernel, Matern32_reference
from geospsr.spectral import FourierPoisson, spectral_axis
from geospsr.spectral_blocks import axis_weights
from geospsr.training_spectrum import FourierCovariance


def analytic_product(x, y, ell, variance=.1):
    difference = (x[:, None].astype('d')-y[None].astype('d')+np.pi)%(2*np.pi)-np.pi
    rt3 = np.sqrt(3.)
    def helper(angle):
        w = rt3*(np.abs(angle/(2*np.pi))-.5)
        z = rt3/2
        return (np.exp((w-z)/ell)+np.exp((-w-z)/ell))*(
            2*ell+rt3/np.tanh(z/ell)-2*w*np.tanh(w/ell))
    return np.prod(variance*helper(difference)/helper(0.),axis=2)


def explicit_scalar_model(q, x, ell, cutoff, normalization):
    ns = np.stack(np.meshgrid(*[np.arange(-cutoff,cutoff+1)]*3,indexing='ij'),-1).reshape(-1,3)
    rho = axis_weights(ns,ell).prod(1)
    if normalization == 'released':
        rho /= (1+np.exp(-np.sqrt(3)/ell))**3
    norm = (ns*ns).sum(1)
    nonzero = norm > 0
    rho, ns, norm = rho[nonzero],ns[nonzero],norm[nonzero]
    cross = np.einsum('qnf,fd->qnd',np.sin((q[:,None].astype('d')-x[None].astype('d'))@ns.T),
                      rho[:,None]*ns/norm[:,None])
    prior = np.cos((q[:,None].astype('d')-q[None].astype('d'))@ns.T)@(rho/norm)
    return cross,prior


def test_jax_corrected_spectrum_matches_analytic_and_reference_formula():
    with jax.experimental.enable_x64():
        ns = jnp.arange(-200,201,dtype=jnp.float64)
        for ell in [.01,.1,.5,1.,10.]:
            actual = Matern32Kernel(ell,.1)
            released = Matern32Kernel(ell,.1,normalization='released')
            expected = axis_weights(np.asarray(ns),ell)
            np.testing.assert_allclose(actual.spectral_density(ns),expected,rtol=3e-12)
            np.testing.assert_allclose(Matern32_reference(ell).spectral_density(ns)*.1,expected,rtol=3e-12)
            np.testing.assert_allclose(released.spectral_density(ns),
                                       spectral_axis(200,ell,normalization='released')[1],rtol=3e-12)
            angles = jnp.asarray([0.,.01,.1,1.,np.pi])
            analytic = jax.vmap(lambda a: actual(a,0.))(angles)
            old_analytic = jax.vmap(lambda a: released(a,0.))(angles)
            np.testing.assert_array_equal(analytic,old_analytic)
            reference = jax.vmap(lambda a: Matern32_reference(ell)(a,0.))(angles)*.1
            np.testing.assert_allclose(analytic,reference,atol=2e-16,rtol=3e-12)
    for constructor in [Matern32Kernel,FourierPoisson,FourierCovariance]:
        with pytest.raises(ValueError,match='normalization'):
            constructor(normalization='typo')


@pytest.mark.parametrize('normalization',['analytic','released'])
def test_long_scale_posterior_and_coherent_draws_against_independent_dense_model(normalization):
    from reproduction.posterior import Posterior
    from reproduction.spectral_posterior import SpectralPosterior
    rng = np.random.default_rng(918)
    x = rng.uniform(-1.5,1.5,(13,3)).astype('f')
    y = rng.normal(size=x.shape).astype('f')
    q = rng.uniform(-1.5,1.5,(5,3)).astype('f')
    ell,noise,cutoff = 1.,.0002,3
    model = Posterior(x,y,lengthscale=ell,noise_variance=noise,cutoff=cutoff,
                      prior_cutoff=2,normalization=normalization)
    cross,prior = explicit_scalar_model(q,x,ell,cutoff,normalization)
    source_cross,_ = explicit_scalar_model(x,x,ell,cutoff,normalization)
    k = analytic_product(x,x,ell)+noise*np.eye(len(x))
    alpha = np.linalg.solve(k,y.astype('d'))
    offset = np.einsum('qnd,nd->q',source_cross,alpha).mean()
    expected_mean = np.einsum('qnd,nd->q',cross,alpha)-offset
    cov = prior.copy()
    for d in range(3):
        cov -= cross[:,:,d]@np.linalg.solve(k,cross[:,:,d].T)
    np.testing.assert_allclose(cp.asnumpy(model.mean(q)),expected_mean,atol=1e-8,rtol=4e-4)
    model.draw(4096,seed=312,batch=128,rng_backend='gpu',joint_solve=True)
    fields = cp.asnumpy(model.samples(q,batch=128))
    assert np.all(np.abs(fields.mean(0)-expected_mean) < 5*np.sqrt(np.diag(cov)/len(fields)))
    assert np.linalg.norm(np.cov(fields.T)-cov)/np.linalg.norm(cov) < .07
    # A finite prior/inverse and a centered hybrid must propagate the choice
    # through all bands, including a partial final batch.
    finite = SpectralPosterior(x,y,training_cutoff=4,cutoff=2,lengthscale=ell,
                               noise_variance=noise,normalization=normalization)
    finite.draw(3,seed=48,batch=2,retain_components=True,retain_rhs=True,solve_cutoff=3)
    from geospsr.training_spectrum import finite_covariance_rows
    inverse = FourierCovariance(3,ell,normalization=normalization)
    matrix = finite_covariance_rows(x,x,inverse)+noise*cp.eye(len(x),dtype='f')
    rhs = (finite.sample_rhs-cp.asarray(y)[None]).transpose(1,0,2).reshape(len(x),-1)
    exact = cp.linalg.solve(matrix,rhs).reshape(len(x),3,3).transpose(1,0,2)+finite.alpha[None]
    assert float(cp.linalg.norm(exact-finite.sample_weights)/cp.linalg.norm(exact)) < 4e-5
