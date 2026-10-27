import cupy as cp
import numpy as np
import pytest
from geospsr.spectral import spectral_axis
from geospsr.spectral_tail import analytic_covariance_rows64
from geospsr.training_spectrum import finite_covariance_rows,FourierCovariance
from reproduction.analytic_posterior import AnalyticPosterior


def problem():
    rng=np.random.default_rng(4691)
    x=rng.normal(0,.2,(13,3)).astype('f'); x[1]=x[0]
    y=rng.normal(size=x.shape); q=rng.normal(0,.2,(7,3)).astype('f')
    model=AnalyticPosterior(x,y,cutoff=2,fit_cutoff=5,lengthscale=.3,
                            noise_variance=.0002,bound_grid=16,leaf_size=8)
    a=analytic_covariance_rows64(x,x,.3).get()+.0002*np.eye(len(x))
    return x,y,q,model,a


def test_complete_paired_draws_and_partial_batch_match_dense_principal_root():
    x,y,q,model,a=problem()
    low=finite_covariance_rows(x,x,FourierCovariance(2,.3),dtype='float64').get()
    eig,vec=np.linalg.eigh(a-low); root=(vec*np.sqrt(eig))@vec.T
    reference_alpha=np.linalg.solve(a,y)
    np.testing.assert_allclose(model.alpha.get(),reference_alpha,rtol=1e-7,atol=1e-7)
    counts=[]
    for block in model.draw_batches(5,seed=4692,batch=2,retain_components=True):
        p=block['stop']-block['start']; counts.append(p)
        rhs=np.tile(y,(1,p))-block['vector_values'].get()-root@block['tail_white'].get()
        weights=np.linalg.solve(a,rhs)
        ref=model.op.aggregate(model.x,cp.asarray(weights.reshape(len(x),p,3).transpose(1,0,2)))+block['prior']
        actual=model.op.evaluate(q,block['coefficients'])
        expected=model.op.evaluate(q,ref)
        assert float(cp.linalg.norm(actual-expected)/cp.linalg.norm(expected))<1e-7
        assert np.max(np.linalg.norm(a@block['weights'].get()-block['rhs'].get(),axis=0)/np.linalg.norm(block['rhs'].get(),axis=0))<1e-8
    assert counts==[2,2,1]
    assert model.batch_size==1


def test_streamed_posterior_moments_match_independent_analytic_conditioning():
    x,y,q,model,a=problem()
    # Direct trigonometric prior features, with independent normal coefficients
    # for each real/imaginary component. No sampler output defines the target.
    modes,rho=spectral_axis(2,.3)
    ns=np.stack(np.meshgrid(modes,modes,modes,indexing='ij')).reshape(3,-1).T
    weights=np.sqrt((rho[:,None,None]*rho[None,:,None]*rho[None,None,:]).ravel())
    phaseq=q.astype('d')@ns.T; phasex=x.astype('d')@ns.T
    norm2=(ns*ns).sum(1); denom=np.where(norm2==0,1,norm2)
    vf=np.concatenate([np.cos(phasex),np.sin(phasex)],axis=1)*np.tile(weights,2)
    sf=[np.concatenate([np.sin(phaseq),-np.cos(phaseq)],axis=1)*np.tile(weights*ns[:,d]/denom,2) for d in range(3)]
    cross=[f@vf.T for f in sf]
    alpha=np.linalg.solve(a,y)
    mean=sum(c@alpha[:,d] for d,c in enumerate(cross))-float(model.offset)
    covariance=sum(f@f.T-c@np.linalg.solve(a,c.T) for f,c in zip(sf,cross))
    samples=np.concatenate([(model.op.evaluate(q,b['coefficients'])-model.offset).get()
                            for b in model.draw_batches(8192,seed=4693,batch=64)])
    mean_se=np.sqrt(np.diag(covariance)/len(samples))
    assert np.max(np.abs(samples.mean(0)-mean)/mean_se)<6
    cov_se=np.sqrt((covariance**2+np.diag(covariance)[:,None]*np.diag(covariance)[None,:])/(len(samples)-1))
    assert np.max(np.abs(np.cov(samples.T)-covariance)/cov_se)<6


def test_invalid_sampler_parameters_are_rejected():
    with pytest.raises(ValueError):AnalyticPosterior(np.zeros((3,3)),np.zeros((2,3)))
    with pytest.raises(ValueError):AnalyticPosterior(np.zeros((3,3)),np.zeros((3,3)),cutoff=5,fit_cutoff=2)
