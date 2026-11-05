import numpy as np
import pytest
from reproduction.density_protocol import lengthscale, original_parameters, replace_original_floor


@pytest.mark.parametrize('n', [2048, 8192, 32768, 131072])
def test_density_lumped_normalization_matches_physical_units(n):
    from gpytoolbox import compactly_supported_normal
    ell = lengthscale(n); h = .8942957229*ell
    p = original_parameters([h]*3, .0025, .000025)
    peak = compactly_supported_normal(np.zeros((1,3)), n=3, sigma=1.5*h)[0]
    np.testing.assert_allclose(p['sigma']**2*peak, .0025)
    np.testing.assert_allclose(p['sigma_n']**2*p['sigma']**2, .000025)
    rho = np.array([100.,200.,300.]); cross = np.array([[.03,.01],[.02,.04],[.01,.02]])
    y = np.arange(3.)
    c = p['covariance_scale']; physical_inverse = 1/(c*rho+.000025)
    native_inverse = 1/(rho+p['sigma_n']**2)
    np.testing.assert_allclose((c*cross).T@(physical_inverse*y),cross.T@(native_inverse*y))
    prior = np.eye(2)
    np.testing.assert_allclose(c*prior-(c*cross).T@(physical_inverse[:,None]*(c*cross)),
                               c*(prior-cross.T@(native_inverse[:,None]*cross)))
    raw = np.array([0.,.01,.02])+p['native_noise_variance']+1e-6
    np.testing.assert_allclose(replace_original_floor(raw,p), np.array([0.,.01,.02])+.000026)


def test_density_law():
    assert lengthscale(8192) == .12
    assert lengthscale(32768) == .06
    with pytest.raises(ValueError): lengthscale(0)


def test_native_original_covariance_scales_with_controlled_amplitude():
    import gpytoolbox as gpy
    rng=np.random.default_rng(1); normal=rng.normal(size=(16,3));normal/=np.linalg.norm(normal,axis=1)[:,None]
    points=.2*normal;h=np.full(3,.15);means=[];variances=[]
    for amplitude in [.0025,.005]:
        p=original_parameters(h,amplitude,amplitude*.01)
        mean,variance,_=gpy.stochastic_poisson_surface_reconstruction(points,normal,
            gs=np.array([8,8,8]),h=h,corner=np.full(3,-.5),output_variance=True,
            sigma=p['sigma'],sigma_n=p['sigma_n'])
        variance=replace_original_floor(variance,p)
        np.testing.assert_allclose(variance.min(),p['scalar_variance_floor'],atol=1e-13)
        means.append(mean);variances.append(variance-p['scalar_variance_floor'])
    np.testing.assert_allclose(means[0],means[1],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(variances[1],2*variances[0],rtol=1e-9,atol=1e-13)
