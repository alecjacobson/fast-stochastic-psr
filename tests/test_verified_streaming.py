"""Same finite-cube law, including centered hybrids and partial sample batches."""
from contextlib import closing
import cupy as cp
import numpy as np
import pytest
from geospsr.spectral import spectral_axis
from reproduction.half_posterior import HalfCubePosterior


def dense(x,cutoff):
    modes,axis=spectral_axis(cutoff,.12,.1)
    matrix=np.ones((len(x),len(x)))
    for d in range(3):
        matrix*=np.cos((x[:,None,d].astype('d')-x[None,:,d].astype('d'))[...,None]*modes)@axis
    return matrix+.0001*np.eye(len(x))


@pytest.mark.parametrize('pack,hybrid,preconditioned,axis',[
    (False,None,False,0),(True,None,True,1),(False,3,False,2),(True,3,False,0)])
def test_verified_stream_matches_dense_target(pack,hybrid,preconditioned,axis):
    rng=np.random.default_rng(234);x=rng.uniform(-.4,.4,(23,3)).astype('f');y=rng.normal(size=x.shape).astype('f')
    model=HalfCubePosterior(x,y,training_cutoff=5,cutoff=2,lengthscale=.12,
                           noise_variance=.0001,rtol=3e-6,axis=axis,refine_solves=True)
    mean=np.linalg.solve(dense(x,5),y)
    assert np.linalg.norm(cp.asnumpy(model.alpha)-mean)/np.linalg.norm(mean)<2e-5
    assert model.mean_solve['refinement']['converged']
    if preconditioned:model.prepare_conditioning_preconditioner(rank=9)
    fields=[];expected=[]
    with closing(model.draw_batches(3,seed=19,batch=2,pack_rhs=pack,solve_cutoff=hybrid,
            preconditioned=preconditioned,retain_components=True,retain_rhs=True)) as batches:
        for block in batches:
            n=block['stop']-block['start'];rhs=cp.asnumpy(block['rhs']).transpose(1,0,2).reshape(len(x),-1)
            reference=np.linalg.solve(dense(x,hybrid or 5),rhs if hybrid is None else rhs-np.tile(y,(1,n)))
            if hybrid is not None:reference+=np.tile(cp.asnumpy(model.alpha),(1,n))
            weights=cp.asnumpy(block['weights']).transpose(1,0,2).reshape(len(x),-1)
            assert np.linalg.norm(weights-reference)/np.linalg.norm(reference)<2e-5
            assert block['record']['verification_cutoff']==(hybrid or 5)
            assert block['record']['refinement']['converged']
            if hybrid:assert block['record']['centered_fluctuations']
            q=cp.asarray(x[:7])
            reference=cp.asarray(reference.reshape(len(x),n,3).transpose(1,0,2),dtype='f')
            expected.append(cp.asnumpy(model.op.evaluate(q,model.op.aggregate(model.x,reference)+block['prior'])))
            fields.append(cp.asnumpy(model.op.evaluate(q,block['coefficients'])))
    a,b=np.concatenate(fields),np.concatenate(expected)
    assert np.linalg.norm(a-b)/np.linalg.norm(b-b.mean(0))<5e-4
    assert np.linalg.norm(np.cov(a.T)-np.cov(b.T))/np.linalg.norm(np.cov(b.T))<1e-4
    assert not model._drawing and model._packed_kernel is None and model._draw_preconditioner is None
    # A hybrid draw must not leave the public solve using the narrower verifier.
    actual=cp.asnumpy(model.solve(cp.asarray(y)))
    assert model._verification_kernel.op.cutoff==5
    assert np.linalg.norm(actual-mean)/np.linalg.norm(mean)<2e-5


def test_native_failure_repaired_and_failed_verification_rejected(monkeypatch):
    rng=np.random.default_rng(52);x=rng.uniform(-.4,.4,(17,3)).astype('f')
    original=HalfCubePosterior._make_kernel
    def inaccurate_initial(self,count):
        kernel=original(self,count);solve=kernel.solve
        def injected(rhs,rtol,**kw):
            if rtol==self.rtol:return cp.zeros_like(rhs),{'converged':False,'injected_failure':True}
            return solve(rhs,rtol=1e-7,**kw)
        kernel.solve=injected
        return kernel
    monkeypatch.setattr(HalfCubePosterior,'_make_kernel',inaccurate_initial)
    model=HalfCubePosterior(x,x,training_cutoff=4,cutoff=2,lengthscale=.12,refine_solves=True)
    assert not model.mean_solve['native_solve']['converged']
    assert model.mean_solve['refinement']['corrections']==1
    # A native success cannot override a failed independent verifier.
    monkeypatch.setattr(model,'_verification_action',lambda weights,band:cp.zeros_like(weights))
    with pytest.raises(RuntimeError):
        with closing(model.draw_batches(1,pack_rhs=True)) as stream:next(stream)
    assert not model._drawing and model._packed_kernel is None
