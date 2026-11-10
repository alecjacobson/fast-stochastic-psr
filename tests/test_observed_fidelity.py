import numpy as np
import pytest

from geospsr.nonperiodic import FreeSpacePoissonMatern, DenseFreeSpacePosterior
from geospsr.scalar_contacts import ContactConditionedDensePosterior
from reproduction.observed_fidelity import FidelityFamily, fidelity


def test_contact_noise_family_matches_independent_posterior():
    pytest.importorskip('cupy')
    rng=np.random.default_rng(7124)
    x,y=rng.normal(size=(40,3))*.03,rng.normal(size=(40,3))
    c,q=rng.normal(size=(9,3))*.03,rng.normal(size=(13,3))*.03
    model=FreeSpacePoissonMatern(.03,1/3)
    configs=[(4,1e-6*model.prior_variance),(9,1e-4*model.prior_variance)]
    family=FidelityFamily(model,x,y,c,.02**2,configs)
    mean,var=family.predict(q,batch=5)
    for k,(count,noise) in enumerate(configs):
        truth=ContactConditionedDensePosterior(DenseFreeSpacePosterior(model,x,y,.02**2),c[:count],np.zeros(count),noise)
        np.testing.assert_allclose((mean[:,k],var[:,k]),truth.predict(q),atol=2e-13,rtol=0)
    np.testing.assert_allclose(family.mean(q,batch=7),mean,atol=2e-13,rtol=0)


def test_resolution_gate_detects_displacement_missing_and_reversed_normals():
    t=np.linspace(-.03,.03,65)
    panel=dict(fit_offsets=t,spacing=.003,fit_normals=np.tile([0,0,1.],(100,1)))
    values=np.broadcast_to(t-.001,(100,65));g=panel['fit_normals'].copy()
    result,err,angle=fidelity(values,g,panel)
    assert result['passes'];np.testing.assert_allclose(err,.001);np.testing.assert_allclose(angle,0)
    result,_,_=fidelity(values+.02,g,panel)
    assert not result['checks']['p95'] and not result['checks']['median']
    result,err,_=fidelity(np.ones_like(values),g,panel)
    assert not result['checks']['missing'];np.testing.assert_allclose(err,.03)
    result,_,angle=fidelity(values,-g,panel)
    assert not result['checks']['normal_median'];np.testing.assert_allclose(angle,180)


def test_streamed_contact_reference_matches_direct_conditioning():
    cp=pytest.importorskip('cupy')
    from reproduction.observed_fidelity_large import StreamingReference
    from reproduction.compute_partial_figures import mean_field
    rng=np.random.default_rng(19124)
    x,y=rng.normal(size=(48,3))*.03,rng.normal(size=(48,3))
    c,q=rng.normal(size=(13,3))*.03,rng.normal(size=(11,3))*.03
    model=FreeSpacePoissonMatern(.03,1/3);scalar_noise=1e-6*model.prior_variance
    streamed=StreamingReference(model,x,y,c,.02**2,scalar_noise,block=5)
    actual=streamed.predict(q)
    expected=ContactConditionedDensePosterior(DenseFreeSpacePosterior(model,x,y,.02**2),c,np.zeros(len(c)),scalar_noise).predict(q)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    streamed.release_factor()
    np.testing.assert_allclose(mean_field(streamed,q,(streamed.coefficients,streamed.update.weights)),actual[0],atol=1e-12,rtol=0)
    assert all(h[-1]['relative_residual'] <= 1e-10 for h in streamed.histories)


def test_fused_mean_matches_full_covariance_at_coincident_near_and_far_points():
    pytest.importorskip('cupy')
    from reproduction.fused_mean import FusedMean
    rng=np.random.default_rng(22124)
    x=rng.normal(size=(317,3))*.03;c=x[:23]
    a=rng.normal(size=x.shape)*100;w=rng.normal(size=len(c))*100
    q=np.vstack((x[:4],x[:4]+1e-10,rng.normal(size=(19,3))*.1,rng.normal(size=(7,3))*100))
    model=FreeSpacePoissonMatern(.015,1/3)
    q=np.vstack((q,np.array([[z/model.kappa,0,0] for z in [49.99,50,50.01]])))
    truth=np.einsum('mnd,nd->m',model.cross_covariance(q,x),a)+model.scalar_covariance(q,c)@w
    np.testing.assert_allclose(FusedMean(model,x,c,a,w)(q,batch=11),truth,atol=1e-13,rtol=0)
