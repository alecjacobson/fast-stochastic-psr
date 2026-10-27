import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from reproduction.event_probability_budget import (
    sign_probability_budget, prefix_probability_budget, scanning_score_intervals, mean_solve_budget)


@pytest.mark.parametrize('centered',[False,True])
@pytest.mark.parametrize('noise',[1e-5,.02])
def test_mean_solve_budget_against_independent_nonstationary_feature_model(centered,noise):
    rng=np.random.default_rng(5204);n,m,p,d=23,17,13,3
    v=rng.normal(size=(n,p));f=rng.normal(size=(d,m,p))*.1
    center=rng.normal(size=(d,7,p))*.2;weight=np.arange(1,8)/28
    a=v@v.T+noise*np.eye(n);y=rng.normal(size=(n,d))
    approximate=np.linalg.solve(a+.01*np.eye(n),y);exact=np.linalg.solve(a,y)
    residual=y-a@approximate
    cross=np.einsum('dmp,np->dmn',f,v)
    error=np.einsum('dmn,nd->m',cross,approximate-exact)
    center_std=0.
    if centered:
        c=np.einsum('dmp,np->dmn',center,v)
        error-=weight@np.einsum('dmn,nd->m',c,approximate-exact)
        center_std=weight@np.sqrt(np.sum(center**2,axis=(0,2)))
    bound=mean_solve_budget(np.sum(f*f,axis=(0,2)),np.linalg.norm(residual,axis=0),noise,center_std)
    assert np.all(np.abs(error)<=bound)
    np.testing.assert_array_equal(mean_solve_budget([0.,1.],[0.,0.,0.],noise,center_std),[0.,0.])


def test_mean_budget_tiny_residuals_and_invalid_inputs():
    bound=mean_solve_budget(1.,[1e-200,1e-200,1e-200],1.,1.)
    np.testing.assert_allclose(bound,2*np.sqrt(3)*1e-200,rtol=1e-15,atol=0)
    for args in [(1.,[-1.],1.),(1.,[1.],0.),(-1.,[1.],1.),(1.,[],1.),(1.,[1.],1.,-1.)]:
        with pytest.raises(ValueError):mean_solve_budget(*args)


@pytest.mark.parametrize('ratio',[1e-12,1e-7,.001,.1,.9])
def test_zero_mean_gaussian_sign_formula(ratio):
    # For Y=X+eta with independent centered normals, rho=sqrt(1-ratio).
    # Their sign mismatch probability is acos(rho)/pi.
    exact=np.arccos(np.sqrt(1-ratio))/np.pi
    bound=sign_probability_budget(0.,1.,ratio)['node_bound']
    assert exact<=bound<=1
    assert bound<.01 if ratio<=1e-7 else bound>0


def test_known_approximate_marginal_dominates_old_density_bound():
    rng=np.random.default_rng(5203)
    s=np.exp(rng.uniform(-25,5,200));u=s*rng.uniform(0,.99,200)
    mu=rng.normal(size=200)*np.sqrt(s);d=rng.uniform(0,.1,200)*np.sqrt(s)
    new=sign_probability_budget(mu,s,u,d)['node_bound']
    old=sign_probability_budget(mu,s,u,d,method='lower_exact_density')['node_bound']
    assert np.all(new<=old+1e-14)


def test_general_correlated_factor_coupling_without_psd_covariance_order():
    # X=A z, Y=B z. Cov(Y)-Cov(X) is indefinite; centered differences are
    # correlated with both fields. Only row norms of B-A enter the new bound.
    a=np.array([[1.,0.],[0.,2.]])
    b=np.array([[1.1,.02],[.01,1.8]])
    assert np.linalg.eigvalsh(b@b.T-a@a.T).min()<0
    assert np.linalg.eigvalsh(b@b.T-a@a.T).max()>0
    s=np.sum(b*b,axis=1);u=np.sum((b-a)**2,axis=1)
    bound=sign_probability_budget(np.zeros(2),s,u)['node_bound']
    rho=np.sum(a*b,axis=1)/np.sqrt(np.sum(a*a,axis=1)*s)
    np.testing.assert_array_less(np.arccos(rho)/np.pi,bound)


@pytest.mark.parametrize('mu,delta,v,e',[(.2,.1,.3,.01),(-2.,-.3,.7,.001),(.05,-.1,.01,.0001)])
def test_unknown_mean_interval_against_independent_scalar_integration(mu,delta,v,e):
    # mu is the approximate mean; the true mean is mu-delta.
    def density(x):return np.exp(-.5*((x-(mu-delta))/np.sqrt(v))**2)/np.sqrt(2*np.pi*v)
    negative=quad(lambda x:density(x)*ndtr((x+delta)/np.sqrt(e)),-np.inf,0,epsabs=1e-13)[0]
    positive=quad(lambda x:density(x)*ndtr((-x-delta)/np.sqrt(e)),0,np.inf,epsabs=1e-13)[0]
    bound=sign_probability_budget(mu,v+e,e,abs(delta))['node_bound']
    assert negative+positive<=bound+1e-13


def test_correlated_gaussian_coupling_duplicate_nodes_and_prefix_events():
    rng=np.random.default_rng(5201)
    a=rng.normal(size=(5,3));a[1]=a[0]
    noise=.002*rng.normal(size=(5,2));noise[1]=noise[0]
    mu=np.array([.1,.1,-.4,.7,-.2]);shift=np.array([.001,.001,-.002,0.,.0003])
    v=np.sum(a*a,axis=1);e=np.sum(noise*noise,axis=1)
    nodes=sign_probability_budget(mu,v+e,e,np.abs(shift))['node_bound']
    assert nodes[0]==nodes[1]
    count=200000
    x=rng.normal(size=(count,3))@a.T+mu-shift
    y=x+shift+rng.normal(size=(count,2))@noise.T
    exact_prefix=np.logical_and.accumulate(x>0,axis=1)
    approximate_prefix=np.logical_and.accumulate(y>0,axis=1)
    disagreement=(exact_prefix!=approximate_prefix).mean(0)
    # Conservative six-sigma sampling allowance; correlations across nodes
    # are retained and no independence assumption is used by the union bound.
    assert np.all(disagreement<=prefix_probability_budget(nodes)+6/np.sqrt(count))


def test_zero_error_degenerate_variance_and_tiny_scales():
    np.testing.assert_array_equal(sign_probability_budget([0,1],[0,1],[0,0])['node_bound'],[0,0])
    np.testing.assert_array_equal(sign_probability_budget([2,0],[0,0],[0,0],1)['node_bound'],[0,1])
    assert sign_probability_budget(0,1,1)['node_bound']==1
    assert sign_probability_budget(0,1,2)['node_bound']==1
    # A deterministic mean shift with identical positive variance.
    bound=sign_probability_budget(.2,.5,0.,.1)['node_bound']
    exact=abs(ndtr(.2/np.sqrt(.5))-ndtr(.1/np.sqrt(.5)))
    assert exact<=bound<1
    normal=sign_probability_budget(2.,1.,1e-9,.01)['node_bound']
    tiny=sign_probability_budget(2e-150,1e-300,1e-309,1e-152)['node_bound']
    np.testing.assert_allclose(tiny,normal,rtol=1e-12)
    # The finite threshold grid retains a positive 30-sigma perturbation tail.
    assert 0<sign_probability_budget(1e100,1e-300,1e-310)['node_bound']<1e-190


def test_score_enclosure_including_nonuniform_grid_and_threshold_edges():
    p=np.array([[0.,.05,.5,.95,1.],[.04,.06,.94,.96,.97]])
    b=np.array([.01,.02,.03,.04,.05]);x=np.array([0.,.1,.3,.8,1.])
    result=scanning_score_intervals(p,x,b,.01)
    rng=np.random.default_rng(5202)
    for _ in range(200):
        prob=rng.uniform(result['probability_lower'],result['probability_upper'])
        score=np.trapezoid(((prob>=.05)&(prob<=.95)).astype(float),x,axis=-1)
        assert np.all(score>=result['score_lower']) and np.all(score<=result['score_upper'])
    np.testing.assert_allclose(prefix_probability_budget([.1,.2,.8]),[.1,.3,1.],rtol=1e-15)


def test_invalid_inputs_rejected():
    for args in [(0,-1,0),(0,1,-1),(np.nan,1,0),(0,1,0,-1)]:
        with pytest.raises(ValueError):sign_probability_budget(*args)
    with pytest.raises(ValueError):sign_probability_budget(0,1,.1,z_grid=[0,1])
    with pytest.raises(ValueError):prefix_probability_budget([1.1])
    with pytest.raises(ValueError):scanning_score_intervals([[.2,.3]],[1,0],0,.1)
