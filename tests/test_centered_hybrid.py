import cupy as cp
import numpy as np
import pytest
from reproduction.spectral_posterior import SpectralPosterior
from geospsr.training_spectrum import FourierCovariance,finite_covariance_rows


def test_centered_hybrid_preserves_mean_and_shared_draws_against_dense_inverse():
    rng = np.random.default_rng(552)
    x = rng.normal(0,.2,(17,3)).astype('f')
    y = rng.normal(size=x.shape).astype('f')
    q = rng.normal(0,.2,(23,3)).astype('f')
    model = SpectralPosterior(x,y,training_cutoff=4,cutoff=2,lengthscale=.1,noise_variance=.0004)
    mean,offset = model.mean(q).copy(),model.offset.copy()
    model.draw(7,seed=119,batch=3,retain_components=True,retain_rhs=True)
    rhs,prior = model.sample_rhs.copy(),model.sample_prior_coeff.copy()
    model.draw(7,seed=119,batch=3,retain_components=True,retain_rhs=True,solve_cutoff=2)
    np.testing.assert_array_equal(cp.asnumpy(rhs),cp.asnumpy(model.sample_rhs))
    np.testing.assert_array_equal(cp.asnumpy(prior),cp.asnumpy(model.sample_prior_coeff))
    np.testing.assert_array_equal(cp.asnumpy(mean),cp.asnumpy(model.mean(q)))
    np.testing.assert_array_equal(cp.asnumpy(offset),cp.asnumpy(model.offset))
    op = FourierCovariance(2,lengthscale=.1)
    matrix = finite_covariance_rows(x,x,op)+.0004*cp.eye(len(x),dtype='f')
    centered = (rhs-cp.asarray(y)[None]).transpose(1,0,2).reshape(len(x),-1)
    exact = cp.linalg.solve(matrix,centered).reshape(len(x),7,3).transpose(1,0,2)+model.alpha[None]
    assert float(cp.linalg.norm(exact-model.sample_weights)/cp.linalg.norm(exact)) < 3e-5
    coeff = model.op.aggregate(x,exact)+prior
    fields = model.op.evaluate(q,coeff)-offset
    assert float(cp.linalg.norm(fields-model.samples(q))/cp.linalg.norm(fields)) < 3e-5
    assert all(r['centered_fluctuations'] and r['solve_cutoff']==2 and r['prior_cutoff']==4
               for r in model.sample_solve_records)
    with pytest.raises(ValueError,match='zero start'):
        model.draw(1,solve_cutoff=2,initial_mean=True)
    for cutoff in [1,5,2.5]:
        with pytest.raises(ValueError,match='cutoff'):
            model.draw(1,solve_cutoff=cutoff)


def test_centered_hybrid_sampler_moments_match_independent_joint_covariance():
    rng = np.random.default_rng(148)
    x = rng.uniform(-.6,.6,(9,3)).astype('f')
    y = rng.normal(size=x.shape).astype('f')
    q = rng.uniform(-.6,.6,(4,3)).astype('f')
    noise = .0001
    model = SpectralPosterior(x,y,training_cutoff=4,cutoff=2,lengthscale=.08,noise_variance=noise)
    wide = finite_covariance_rows(x,x,model.training,dtype='d')+noise*cp.eye(len(x))
    narrow = finite_covariance_rows(x,x,FourierCovariance(2,lengthscale=.08),dtype='d')+noise*cp.eye(len(x))
    cross = model.op.cross_values((q[:,None]-x[None]).reshape(-1,3)).reshape(len(q),len(x),3).astype('d')
    norm2 = (model.op.ns**2).sum(0)
    rho = cp.where(norm2>0,model.op.rho/cp.maximum(norm2,1),0)
    covariance = model.op.evaluate((q[:,None]-q[None]).reshape(-1,3),rho)[0].reshape(len(q),len(q)).astype('d')
    for component in range(3):
        c = cross[:,:,component].T
        a = cp.linalg.solve(narrow,c)
        covariance += -c.T@a-a.T@c+a.T@wide@a
    model.draw(4096,seed=71,batch=64,solve_cutoff=2)
    samples = cp.asnumpy(model.samples(q,batch=64))
    expected_mean,expected_cov = cp.asnumpy(model.mean(q)),cp.asnumpy(covariance)
    se = np.sqrt(np.diag(expected_cov)/len(samples))
    assert np.all(np.abs(samples.mean(0)-expected_mean) < 5*se+2e-6)
    assert np.linalg.norm(np.cov(samples.T)-expected_cov)/np.linalg.norm(expected_cov) < .09


def test_centered_block_inverse_on_fixed_cube_prior_rhs():
    from geospsr.block_spectral import BlockSpectrum,BlockKernel
    from reproduction.block_reference import block_covariance_rows
    from reproduction.spectral_posterior import solve_centered_fluctuations
    rng = np.random.default_rng(52)
    x = rng.normal(0,.2,(17,3)).astype('f')
    y = rng.normal(size=x.shape).astype('f')
    model = SpectralPosterior(x,y,training_cutoff=4,cutoff=2,lengthscale=.1,noise_variance=.0004)
    model.draw(3,seed=71,batch=1,retain_components=True,retain_rhs=True)
    rhs = model.sample_rhs.transpose(1,0,2).reshape(len(x),9)
    saved = rhs.copy()
    kernel = BlockKernel(x,BlockSpectrum(2,4,lengthscale=.1),.0004,rhs_count=9)
    weights,info = solve_centered_fluctuations(kernel,rhs,model.normals,model.alpha)
    assert info['converged'] and info['centered_fluctuations']
    matrix = block_covariance_rows(x,x,2,4,.1)+.0004*cp.eye(len(x))
    centered = rhs-cp.tile(model.normals,(1,3))
    exact = cp.linalg.solve(matrix,centered.astype('d'))+cp.tile(model.alpha,(1,3))
    assert float(cp.linalg.norm(weights-exact)/cp.linalg.norm(exact))<3e-5
    np.testing.assert_array_equal(cp.asnumpy(saved),cp.asnumpy(rhs))
