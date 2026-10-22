import cupy as cp
import numpy as np
from reproduction.spectral_posterior import SpectralPosterior
from reproduction.benchmark_nufft_batches import prior_rhs
from geospsr.spectral import FourierKernel


def test_staged_prior_preserves_draws_and_batched_solution():
    rng = np.random.default_rng(91)
    x = rng.normal(0,.2,(23,3)).astype('f')
    y = rng.normal(size=x.shape).astype('f')
    model = SpectralPosterior(x,y,training_cutoff=4,cutoff=2,lengthscale=.1,noise_variance=.001)
    model.draw(3,seed=41,batch=1,retain_rhs=True,retain_components=True)
    random = cp.random.RandomState(41)
    root = cp.sqrt(model.training.rho)
    denom = cp.maximum((model.op.ns**2).sum(0),1)
    staged = [prior_rhs(model,random,root,denom) for _ in range(3)]
    rhs = cp.stack([v[0] for v in staged])
    prior = cp.stack([v[1] for v in staged])
    np.testing.assert_allclose(cp.asnumpy(rhs),cp.asnumpy(model.sample_rhs),atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(prior),cp.asnumpy(model.sample_prior_coeff),atol=1e-9,rtol=1e-6)
    kernel = FourierKernel(x,model.training,.001,rhs_count=9)
    weights,info = kernel.solve(rhs.transpose(1,0,2).reshape(len(x),9),rtol=1e-5)
    assert info['converged']
    coeff = model.op.aggregate(x,weights.reshape(len(x),3,3).transpose(1,0,2))+prior
    q = rng.normal(0,.2,(41,3)).astype('f')
    expected = model.samples(q)
    actual = model.op.evaluate(q,coeff)-model.offset
    np.testing.assert_allclose(cp.asnumpy(actual),cp.asnumpy(expected),atol=1e-8,rtol=5e-5)
    model.draw(3,seed=41,batch=1,retain_rhs=True,retain_components=True,initial_mean=True)
    np.testing.assert_allclose(cp.asnumpy(model.sample_rhs),cp.asnumpy(rhs),atol=1e-7,rtol=1e-6)
    # Different CG starting points stop at different approximations. Check each
    # against an independent dense solve; pointwise relative error near zeros
    # is not the solver's normwise stopping criterion.
    from geospsr.training_spectrum import finite_covariance_rows
    matrix = finite_covariance_rows(x,x,model.training)+.001*cp.eye(len(x),dtype='f')
    exact_weights = cp.linalg.solve(matrix,rhs.transpose(1,0,2).reshape(len(x),9))
    exact_coeff = model.op.aggregate(x,exact_weights.reshape(len(x),3,3).transpose(1,0,2))+prior
    exact_fields = model.op.evaluate(q,exact_coeff)-model.offset
    for candidate in [expected,model.samples(q)]:
        assert float(cp.linalg.norm(candidate-exact_fields)/cp.linalg.norm(exact_fields)) < 5e-5
    warm_weights = model.sample_weights.transpose(1,0,2).reshape(len(x),9)
    b = rhs.transpose(1,0,2).reshape(len(x),9)
    assert bool(cp.all(cp.linalg.norm(matrix@warm_weights-b,axis=0)/cp.linalg.norm(b,axis=0) < 3e-5))


def test_cg_starting_solution_uses_original_rhs_tolerance_and_is_immutable():
    from geospsr.training_spectrum import FourierCovariance, finite_covariance_rows
    rng = np.random.default_rng(12)
    x = rng.normal(0,.15,(33,3)).astype('f')
    op = FourierCovariance(5,lengthscale=.1)
    kernel = FourierKernel(x,op,.0001)
    matrix = finite_covariance_rows(x,x,op) + .0001*cp.eye(len(x),dtype='f')
    rhs = cp.asarray(rng.normal(size=(len(x),3)),dtype='f')
    exact = cp.linalg.solve(matrix,rhs)
    initial = exact*.99999
    saved = initial.copy()
    solution,info = kernel.solve(rhs,rtol=1e-4,x0=initial)
    assert info['converged']
    assert info['iterations'] <= 2
    np.testing.assert_array_equal(cp.asnumpy(initial),cp.asnumpy(saved))
    residual = cp.linalg.norm(matrix@solution-rhs,axis=0)/cp.linalg.norm(rhs,axis=0)
    assert bool(cp.all(residual < 2e-4))
