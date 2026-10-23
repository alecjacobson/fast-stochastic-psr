from contextlib import closing
import cupy as cp
import numpy as np
import pytest
from geospsr.spectral import spectral_axis
from reproduction.half_posterior import HalfCubePosterior


@pytest.mark.parametrize('packed,initial',[(True,False),(False,True)])
@pytest.mark.parametrize('options',[{}, {'basis_dtype':'float16','recompute_features':True,'apply_row_batch':11}])
def test_preconditioned_stream_matches_prior_dense_solve_and_fields(packed,initial,options):
    rng=np.random.default_rng(222)
    x=rng.uniform(-.5,.5,(23,3)).astype('f');y=rng.normal(size=x.shape).astype('f')
    q=rng.uniform(-.5,.5,(11,3)).astype('f')
    model=HalfCubePosterior(x,y,training_cutoff=5,cutoff=2,lengthscale=.12,noise_variance=.0001,rtol=3e-6)
    model.draw(5,seed=19,batch=2,pack_rhs=packed,initial_mean=initial,retain_components=True,retain_rhs=True)
    rhs=model.sample_rhs.copy();prior=model.sample_prior_coeff.copy();fields=cp.asnumpy(model.samples(q));mean=model.mean(q).copy()
    pre=model.prepare_conditioning_preconditioner(rank=9,seed=41,row_batch=7,**options)
    assert model.prepare_conditioning_preconditioner(rank=9,seed=41,row_batch=7,**options) is pre
    coefficients=[];weights=[]
    with closing(model.draw_batches(5,seed=19,batch=2,pack_rhs=packed,initial_mean=initial,
                                   preconditioned=True,retain_components=True,retain_rhs=True)) as stream:
        for b in stream:
            sl=slice(b['start'],b['stop'])
            np.testing.assert_array_equal(cp.asnumpy(b['rhs']),cp.asnumpy(rhs[sl]))
            np.testing.assert_array_equal(cp.asnumpy(b['prior']),cp.asnumpy(prior[sl]))
            coefficients.append(b['coefficients']);weights.append(b['weights'])
            assert b['record']['preconditioned'] and b['record']['conditioning_preconditioner']['rank']==9
    actual=cp.asnumpy(model.op.evaluate(q,cp.concatenate(coefficients))-model.offset)
    assert np.linalg.norm(actual-fields)/np.linalg.norm(fields-fields.mean(0))<5e-4
    assert np.linalg.norm(np.cov(actual.T)-np.cov(fields.T))/np.linalg.norm(np.cov(fields.T))<1e-4
    np.testing.assert_array_equal(cp.asnumpy(model.mean(q)),cp.asnumpy(mean))
    modes,axis=spectral_axis(5,.12,.1);matrix=np.ones((len(x),len(x)))
    for d in range(3):matrix*=np.cos((x[:,None,d].astype('d')-x[None,:,d].astype('d'))[...,None]*modes)@axis
    matrix+=.0001*np.eye(len(x))
    exact=np.linalg.solve(matrix,cp.asnumpy(rhs).transpose(1,0,2).reshape(len(x),-1))
    solved=cp.asnumpy(cp.concatenate(weights)).transpose(1,0,2).reshape(len(x),-1)
    assert np.linalg.norm(solved-exact)/np.linalg.norm(exact)<3e-5
    assert model._draw_preconditioner is None and model.conditioning_preconditioner is pre
    model.draw(1,seed=4)
    assert 'preconditioned' not in model.sample_solve_records[0]


def test_preconditioner_lifetime_validation_and_early_close():
    rng=np.random.default_rng(333);x=rng.uniform(-1,1,(13,3)).astype('f')
    model=HalfCubePosterior(x,x,training_cutoff=4,cutoff=2,lengthscale=.2)
    with pytest.raises(ValueError,match='prepare'):
        next(model.draw_batches(2,preconditioned=True))
    assert not model._drawing and model._draw_preconditioner is None
    pre=model.prepare_conditioning_preconditioner(rank=5)
    with pytest.raises(ValueError,match='full coupled'):
        next(model.draw_batches(2,preconditioned=True,solve_cutoff=3))
    with closing(model.draw_batches(3,batch=2,pack_rhs=True,preconditioned=True)) as stream:
        next(stream)
        with pytest.raises(RuntimeError,match='active stream'):model.clear_conditioning_preconditioner()
        with pytest.raises(RuntimeError,match='active stream'):model.prepare_conditioning_preconditioner(rank=5)
    assert not model._drawing and model._packed_kernel is None and model._draw_preconditioner is None
    assert model.conditioning_preconditioner is pre
    with pytest.raises(ValueError):model.prepare_conditioning_preconditioner(rank=14)
    assert model.conditioning_preconditioner is pre
    model.clear_conditioning_preconditioner()
    assert model.conditioning_preconditioner is None
