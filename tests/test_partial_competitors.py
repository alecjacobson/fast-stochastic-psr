import numpy as np
from scipy.linalg import solve_triangular

from geospsr.nonperiodic import FreeSpacePoissonMatern, DenseFreeSpacePosterior
from geospsr.scalar_contacts import ContactConditionedDensePosterior
from reproduction.partial_competitor_common import conditioned_moments, projected_moments, gates


def test_shared_whitened_and_corrected_moments_match_dense_oracle():
    rng = np.random.default_rng(1141)
    x, y = rng.normal(size=(23,3)), rng.normal(size=(23,3))
    contacts, queries = x[:5], rng.normal(size=(11,3))
    model = FreeSpacePoissonMatern(.48, 1/3)
    base = DenseFreeSpacePosterior(model, x, y, .04)
    truth = ContactConditionedDensePosterior(base, contacts, np.zeros(5), .01**2).predict(queries)
    c = model.cross_covariance(np.concatenate((contacts,queries)), x).transpose(1,2,0).reshape(len(x),-1)
    rhs = np.column_stack((c,y))
    w = solve_triangular(base.factor, rhs, lower=True)
    np.testing.assert_allclose(conditioned_moments(model,contacts,queries,rhs,w,np,whitened=True),truth,atol=2e-14,rtol=0)
    z = solve_triangular(base.factor.T,w,lower=False)
    np.testing.assert_allclose(conditioned_moments(model,contacts,queries,rhs,z,np),truth,atol=2e-14,rtol=0)
    u,coef=np.linalg.qr(c,mode='reduced')
    zu=np.linalg.solve(model.vector_covariance(x,x)+.04*np.eye(len(x)),u)
    np.testing.assert_allclose(projected_moments(model,contacts,queries,coef,u.T@zu,u.T@z[:,-3:]),truth,atol=2e-14,rtol=0)
    a = model.vector_covariance(x,x)+.04*np.eye(len(x))
    perturbed = z+1e-4*rng.normal(size=z.shape)
    residual = rhs-a@perturbed
    result = conditioned_moments(model,contacts,queries,rhs,perturbed,np,residual=residual)
    # The correction cancels first-order solve error in every joint moment.
    plain = conditioned_moments(model,contacts,queries,rhs,perturbed,np)
    assert np.max(abs(np.array(result)-truth)) < np.max(abs(np.array(plain)-truth))/100


def test_output_gates_reject_invalid_or_inaccurate_variances():
    model=FreeSpacePoissonMatern(.48,1/3);fit=dict(log_scale=0.,bias=-1.)
    ref=dict(mean=np.array([.01,.02]),variance=np.array([.001,.002]))
    assert gates(ref['mean'],ref['variance'],ref,model,fit)['passes']
    assert not gates(ref['mean'],ref['variance']*2,ref,model,fit)['passes']
    assert not gates(ref['mean'],np.array([np.nan,.002]),ref,model,fit)['passes']
    assert not gates(ref['mean'],np.array([-.001,.002]),ref,model,fit)['passes']
    # A steep validation calibration can fail even when raw CDF and variance pass.
    steep=dict(log_scale=np.log(10.),bias=10*ref['mean'][0]/np.sqrt(ref['variance'][0]))
    result=gates(ref['mean']+1e-5,ref['variance'],ref,model,steep)
    assert result['cdf'] < 1e-3 < result['calibrated_cdf']
    assert not result['passes']
