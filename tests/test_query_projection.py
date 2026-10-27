import numpy as np
import pytest
from reproduction.query_projection import prior_basis,assemble_projected


@pytest.mark.parametrize('rank',[0,5,17])
@pytest.mark.parametrize('shift',[0.,.03])
def test_projected_identity_and_psd_bound_against_independent_dense_features(rank,shift):
    rng=np.random.default_rng(5101);n,m,p=23,17,13;noise=.002
    v=rng.normal(size=(n,p));fs=[rng.normal(size=(m,p))*.1*s for s in [1.,1e-14,0.]]
    for f in fs:f[1]=f[0];f[2]=f[0]+1e-7*f[2]
    a=v@v.T+noise*np.eye(n);prior=sum(f@f.T for f in fs)
    q,tail,info=prior_basis(prior,rank)
    xs=[v@f.T for f in fs];ws=[np.linalg.solve(a+shift*np.eye(n),x@q) for x in xs]
    rs=[x@q-a@w for x,w in zip(xs,ws)]
    l=sum(x.T@w for x,w in zip(xs,ws));h=sum(w.T@r for w,r in zip(ws,rs))
    upper=sum(r.T@r for r in rs)/noise
    result,bound,record=assemble_projected(prior,q,l,h,tail,upper)
    target=prior-sum(x.T@np.linalg.solve(a,x) for x in xs)
    residuals=[x-a@w@q.T for x,w in zip(xs,ws)]
    error=sum(r.T@np.linalg.solve(a,r) for r in residuals)
    allowance=1e-11*np.linalg.norm(prior)
    np.testing.assert_allclose(result-target,error,atol=allowance,rtol=1e-9)
    assert np.linalg.eigvalsh(bound-error).min()>=-allowance
    assert np.linalg.norm(error)<=record['covariance_error_frobenius_upper']+allowance
    assert not info['automatic']


def test_exact_zero_tail_and_residual_limits_and_rank_selection():
    prior=np.diag([1.,.01,1e-12,0.]);q,tail,info=prior_basis(prior,rtol=1e-7)
    assert q.shape==(4,2) and info['automatic']
    for r in [0,2,4]:
        q,tail,_=prior_basis(prior,r)
        s,upper,record=assemble_projected(prior,q,np.zeros((4,r)),np.zeros((r,r)),tail,np.zeros((r,r)))
        np.testing.assert_array_equal(s,prior);np.testing.assert_allclose(upper,tail)
        assert record['eta'] is None
    q=np.eye(4);u=np.eye(4)*1e-10
    _,upper,record=assemble_projected(prior,q,np.zeros((4,4)),np.zeros((4,4)),np.zeros((4,4)),u)
    np.testing.assert_array_equal(upper,u);assert record['eta'] is None
    with pytest.raises(ValueError):prior_basis(np.diag([1.,-1e-3]),1)
    with pytest.raises(ValueError):assemble_projected(prior,q*2,q,q,prior,q)


def test_gpu_projected_query_matches_dense_features_and_full_rank_limit():
    import cupy as cp
    from geospsr.spectral import FourierPoisson,spectral_axis
    from geospsr.spectral_tail import analytic_covariance_rows64
    from reproduction.query_posterior import AnalyticConditioner
    from reproduction.query_projection import ProjectedQueryPosterior
    rng=np.random.default_rng(5102);x=rng.normal(0,.1,(19,3)).astype('f');y=rng.normal(size=x.shape)
    query=rng.normal(0,.1,(9,3));query[1]=query[0];query[2]=query[0]+1e-7
    op=FourierPoisson(3,.2,dtype='float64',eps=1e-12);noise=.0002
    condition=AnalyticConditioner(x,.2,noise_variance=noise)
    alpha=condition.solve(cp.asarray(y));coeff=op.aggregate(x,alpha);offset=op.evaluate(x,coeff).mean()
    axis,rho=spectral_axis(3,.2);ns=np.stack(np.meshgrid(axis,axis,axis,indexing='ij')).reshape(3,-1).T
    scale=np.sqrt((rho[:,None,None]*rho[None,:,None]*rho[None,None,:]).ravel());norm2=np.sum(ns**2,1);denom=np.where(norm2>0,norm2,1)
    vx=x.astype('d')@ns.T;fq=query@ns.T
    vf=np.concatenate([np.cos(vx),np.sin(vx)],1)*np.tile(scale,2)
    sf=[np.concatenate([np.sin(fq),-np.cos(fq)],1)*np.tile(scale*ns[:,d]/denom,2) for d in range(3)]
    a=analytic_covariance_rows64(x,x,.2).get()+noise*np.eye(len(x));cross=[f@vf.T for f in sf]
    target=sum(f@f.T-c@np.linalg.solve(a,c.T) for f,c in zip(sf,cross))
    for rank in [0,4,9]:
        model=ProjectedQueryPosterior(x,query,op,condition.solve,condition.action,coeff,offset,noise_min=noise,rank=rank,batch=2)
        error=model.covariance.get()-target
        assert np.linalg.eigvalsh(model.upper.get()-error).min()>=-1e-10*np.linalg.norm(model.prior.get())
        if rank==9:
            assert np.linalg.norm(error)/np.linalg.norm(target)<1e-7
            draws=model.draw(16384,seed=5103).get();se=np.sqrt(np.diag(target)/len(draws))
            assert np.max(np.abs(draws.mean(0)-model.mean.get())/se)<6
            np.testing.assert_allclose(draws[:,0],draws[:,1],atol=1e-8,rtol=0)
        model.clear_vectors()
