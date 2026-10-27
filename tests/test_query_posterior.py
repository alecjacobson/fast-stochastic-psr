import cupy as cp
import numpy as np
import pytest
from geospsr.spectral import FourierPoisson,spectral_axis
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.query_posterior import AnalyticConditioner,FixedQueryPosterior,principal_root


@pytest.mark.parametrize('solver',['analytic_cg','defect'])
def test_query_covariance_and_joint_draws_against_independent_dense_features(solver):
    rng=np.random.default_rng(4801)
    x=rng.normal(0,.1,(19,3)).astype('f');x[1]=x[0]
    q=rng.normal(0,.1,(9,3));q[1]=q[0];q[2]=q[0]+1e-7
    y=rng.normal(size=x.shape)
    ell=.2;noise=.0002
    op=FourierPoisson(3,ell,dtype='float64',eps=1e-12)
    condition=AnalyticConditioner(x,ell,noise_variance=noise,solver=solver,fit_cutoff=12)
    alpha=condition.solve(cp.asarray(y))
    coeff=op.aggregate(x,alpha);offset=op.evaluate(x,coeff).mean()
    model=FixedQueryPosterior(x,q,op,condition.solve,coeff,offset,batch=2)
    modes,rho=spectral_axis(3,ell)
    ns=np.stack(np.meshgrid(modes,modes,modes,indexing='ij')).reshape(3,-1).T
    weights=np.sqrt((rho[:,None,None]*rho[None,:,None]*rho[None,None,:]).ravel())
    norm2=(ns*ns).sum(1);denom=np.where(norm2>0,norm2,1)
    vx=x.astype('d')@ns.T;fq=q@ns.T
    vf=np.concatenate([np.cos(vx),np.sin(vx)],axis=1)*np.tile(weights,2)
    sf=[np.concatenate([np.sin(fq),-np.cos(fq)],axis=1)*np.tile(weights*ns[:,d]/denom,2) for d in range(3)]
    a=analytic_covariance_rows64(x,x,ell).get()+noise*np.eye(len(x))
    c=[f@vf.T for f in sf]
    covariance=sum(f@f.T-cross@np.linalg.solve(a,cross.T) for f,cross in zip(sf,c))
    mean=sum(c[d]@np.linalg.solve(a,y)[:,d] for d in range(3))-float(offset)
    assert np.linalg.norm(model.covariance.get()-covariance)/np.linalg.norm(covariance)<1e-7
    np.testing.assert_allclose(model.mean.get(),mean,rtol=1e-7,atol=1e-10)
    assert np.linalg.norm(model.factor.get()@model.factor.get().T-covariance)/np.linalg.norm(covariance)<1e-7
    draws=model.draw(16384,seed=4802).get()
    se=np.sqrt(np.diag(covariance)/len(draws))
    assert np.max(np.abs(draws.mean(0)-mean)/se)<6
    cse=np.sqrt((covariance**2+np.diag(covariance)[:,None]*np.diag(covariance)[None,:])/(len(draws)-1))
    assert np.max(np.abs(np.cov(draws.T)-covariance)/cse)<6
    np.testing.assert_allclose(draws[:,0],draws[:,1],rtol=0,atol=1e-8)


def test_psd_budget_rejects_real_indefiniteness_and_reports_roundoff_projection():
    with pytest.raises(ValueError,match='indefinite'):principal_root(cp.diag(cp.asarray([1.,-1e-5])))
    _,root,record=principal_root(cp.diag(cp.asarray([1.,-1e-15])))
    assert record['negative_eigenvalues']==1 and record['maximum_projection']==1e-15
    np.testing.assert_allclose(root.get(),np.diag([1.,0.]))
    with pytest.raises(ValueError,match='asymmetric'):principal_root(cp.asarray([[1.,.01],[0.,1.]]))


def test_tiny_and_unequal_rhs_keep_relative_accuracy_through_defect_correction():
    rng=np.random.default_rng(4803);x=rng.normal(0,.1,(17,3)).astype('f')
    rhs=rng.normal(size=(len(x),3))*[1e-20,1.,0.]
    condition=AnalyticConditioner(x,.2,noise_variance=.0002,solver='defect',fit_cutoff=12)
    value=condition.solve(cp.asarray(rhs)).get()
    a=analytic_covariance_rows64(x,x,.2).get()+.0002*np.eye(len(x))
    denom=np.linalg.norm(rhs,axis=0);denom=np.where(denom>0,denom,1)
    assert np.max(np.linalg.norm(a@value-rhs,axis=0)/denom)<1e-10
    np.testing.assert_array_equal(value[:,2],0)
