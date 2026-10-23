from contextlib import closing
import gc
import weakref
import cupy as cp
import numpy as np
import pytest
from reproduction.half_posterior import HalfCubePosterior
from reproduction.streaming_statistics import OnlineMoments,OnlineEvents,view_scores


@pytest.mark.parametrize('pack,band',[(False,None),(True,None),(False,3),(True,3)])
def test_stream_matches_retained_draws_and_releases_state(pack,band):
    rng=np.random.default_rng(136)
    x=rng.uniform(-1,1,(13,3)).astype('f');y=rng.normal(size=x.shape).astype('f')
    q=rng.uniform(-1,1,(5,3)).astype('f')
    model=HalfCubePosterior(x,y,training_cutoff=5,cutoff=2,lengthscale=.1,noise_variance=.0002)
    options=dict(count=7,seed=37,batch=3,retain_components=True,retain_rhs=True,pack_rhs=pack,solve_cutoff=band)
    model.draw(**options)
    expected={k:getattr(model,v).copy() for k,v in [('coefficients','sample_coeff'),('weights','sample_weights'),('prior','sample_prior_coeff'),('rhs','sample_rhs')]}
    fields=cp.asnumpy(model.samples(q));moments=OnlineMoments(covariance=True)
    model.prepare_query_cache()
    with closing(model.draw_batches(**options)) as batches:
        for result in batches:
            sl=slice(result['start'],result['stop'])
            for key in expected:
                np.testing.assert_allclose(cp.asnumpy(result[key]),cp.asnumpy(expected[key][sl]),rtol=5e-5,atol=1e-9)
            moments.update(cp.asnumpy(model.op.evaluate(q,result['coefficients'])-model.offset))
            assert model.sample_coeff is None and model.sample_cache is None
            with pytest.raises(RuntimeError,match='already active'):
                next(model.draw_batches(1))
    assert not model._drawing
    assert model._packed_kernel is None and model._fluctuation_kernel is None
    np.testing.assert_allclose(moments.summary()['mean'],fields.mean(0,dtype='d'),rtol=1e-4,atol=1e-9)
    np.testing.assert_allclose(moments.summary()['covariance'],np.cov(fields.T),rtol=5e-4,atol=1e-12)
    with closing(model.draw_batches(count=5,batch=2,pack_rhs=True,solve_cutoff=3)) as batches:
        next(batches)
    assert not model._drawing and model._packed_kernel is None
    model.draw(1)
    assert model.sample_coeff.shape[0]==1


def test_online_statistics_equal_full_ensemble_and_bounds_cover_score_ranges():
    rng=np.random.default_rng(6)
    values=1e4+rng.normal(size=(101,7))
    samples=rng.integers(0,2,size=(101,3,9))
    diagonal=OnlineMoments();covariance=OnlineMoments(covariance=True);events=OnlineEvents()
    for start in range(0,len(values),13):
        diagonal.update(values[start:start+13]);covariance.update(values[start:start+13]);events.update(samples[start:start+13])
    np.testing.assert_allclose(diagonal.summary()['variance'],values.var(0,ddof=1),rtol=1e-11)
    np.testing.assert_allclose(covariance.summary()['covariance'],np.cov(values.T),rtol=1e-10,atol=1e-12)
    summary=events.summary(looks=4,family_size=30)
    np.testing.assert_array_equal(summary['successes'],samples.sum(0))
    assert summary['simultaneous_radius']==np.sqrt(np.log(2*30*4/.05)/(2*101))
    score=view_scores(summary,np.arange(9))
    assert np.all(score['lower']<=score['scores']) and np.all(score['scores']<=score['upper'])
    with pytest.raises(ValueError,match='binary'):
        events.update(np.ones((2,3,9))*.5)
    with pytest.raises(ValueError,match='family'):
        events.summary(family_size=1)
    with pytest.raises(ValueError,match='binary'):
        events.update(np.empty((2,0)))


def test_numeric_uncertain_length_and_view_regret():
    from reproduction.paired_applications import quality
    probability=np.array([[.5,.5,.5,.5],[1.,.5,.5,0.]])
    result=view_scores({'probability':probability,'lower':probability,'upper':probability},np.arange(4.))
    np.testing.assert_array_equal(result['scores'],[3.,2.])
    np.testing.assert_array_equal(result['lower'],result['scores'])
    # Valid monotone transmittance curves can reverse a legacy Boolean tie.
    probability=np.array([[1.,.5,0.,0.],[.5,.5,0.,0.]])
    result=view_scores({'probability':probability,'lower':probability,'upper':probability},np.arange(4.))
    np.testing.assert_array_equal(result['scores'],[1.,1.5])
    assert result['selected']==1
    # Pointwise probability 1/2 uses one all-positive and one all-negative draw.
    rays=np.stack([np.ones((2,4)),np.zeros((2,4))])
    reference={'rays':rays,'collision':np.array([[0.],[1.]])}
    candidate={'rays':rays.copy(),'collision':reference['collision']}
    check=quality(reference,candidate,np.arange(4.))
    assert check['quality_gate_version']==3
    assert check['compatibility_pass']


def test_stream_discards_previous_coefficients_and_live_storage_stays_bounded():
    rng=np.random.default_rng(33)
    x=rng.normal(size=(9,3)).astype('f');y=rng.normal(size=x.shape).astype('f')
    model=HalfCubePosterior(x,y,training_cutoff=5,cutoff=2,lengthscale=.1)
    previous=None;live=[]
    with closing(model.draw_batches(128,batch=8,pack_rhs=True)) as batches:
        for block in batches:
            gc.collect()
            if previous is not None: assert previous() is None
            previous=weakref.ref(block['coefficients'])
            live.append(cp.get_default_memory_pool().used_bytes())
            assert block['coefficients'].shape[0]==8
            assert model.sample_coeff is None
            del block
    assert max(live[1:])<live[0]+1024**2
    assert not model._drawing
