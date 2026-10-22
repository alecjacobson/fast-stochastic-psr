"""Paired backend comparisons must preserve draws and reject bad event outputs."""
import cupy as cp
import numpy as np

from reproduction.paired_applications import events, quality, uncertain_objects, application_geometry, field_quality
from reproduction.posterior import Posterior


def test_retained_draw_components_reconstruct_same_function():
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, (13, 3)).astype('f')
    normals = rng.normal(size=x.shape).astype('f')
    q = rng.uniform(-1, 1, (9, 3)).astype('f')
    model = Posterior(x, normals, cutoff=2, prior_cutoff=2)
    model.draw(7, seed=5, batch=3, retain_components=True, retain_rhs=True)
    weights = model.sample_weights.transpose(1, 0, 2).reshape(len(x), -1)
    rhs = model.sample_rhs.transpose(1, 0, 2).reshape(len(x), -1)
    assert float(cp.linalg.norm(model.K @ weights - rhs) / cp.linalg.norm(rhs)) < 2e-5
    reconstructed = model.op.aggregate(x, model.sample_weights) + model.sample_prior_coeff
    actual = model.op.evaluate(q, reconstructed) - model.offset
    np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(model.samples(q, batch=3)),
                               rtol=3e-5, atol=2e-8)
    q2 = q * .6 + .1
    expected = model.op.evaluate(q2, model.sample_coeff)
    np.testing.assert_allclose(cp.asnumpy(model.op.evaluate_batched(q2, model.sample_coeff, batch=3)),
                               cp.asnumpy(expected), rtol=3e-5, atol=2e-8)
    model.draw(2, seed=6, batch=2)
    assert model.sample_weights is None and model.sample_prior_coeff is None


def test_joint_draw_solve_preserves_common_random_functions():
    rng = np.random.default_rng(52)
    x = rng.uniform(-1, 1, (19, 3)).astype('f')
    q = rng.uniform(-1, 1, (12, 3)).astype('f')
    model = Posterior(x, rng.normal(size=x.shape).astype('f'), cutoff=3, upsampfac=1.25)
    model.draw(11, seed=3, batch=4, rng_backend='gpu', retain_rhs=True)
    original = model.samples(q).copy()
    rhs = model.sample_rhs.copy()
    model.draw(11, seed=3, batch=4, rng_backend='gpu', retain_rhs=True, joint_solve=True)
    np.testing.assert_array_equal(cp.asnumpy(model.sample_rhs), cp.asnumpy(rhs))
    np.testing.assert_allclose(cp.asnumpy(model.samples(q)), cp.asnumpy(original), rtol=2e-5, atol=2e-8)


def test_posterior_query_cache_and_redraw_invalidation():
    rng = np.random.default_rng(18)
    x = rng.uniform(-1, 1, (13, 3)).astype('f')
    q = rng.uniform(-1, 1, (9, 3)).astype('f')
    model = Posterior(x, rng.normal(size=x.shape).astype('f'), cutoff=3, prior_cutoff=3)
    model.draw(5, seed=3, batch=2)
    mean, samples = model.mean(q), model.samples(q)
    sizes = model.prepare_query_cache(batch=2)
    assert sizes['sample_cache_bytes'] > 0
    np.testing.assert_allclose(cp.asnumpy(model.mean(q)), cp.asnumpy(mean), atol=2e-8, rtol=3e-5)
    np.testing.assert_allclose(cp.asnumpy(model.samples(q)), cp.asnumpy(samples), atol=2e-8, rtol=3e-5)
    q2 = q + .1
    expected = model.op.evaluate(q2, model.sample_coeff) - model.offset
    np.testing.assert_allclose(cp.asnumpy(model.samples(q2)), cp.asnumpy(expected), atol=2e-8, rtol=3e-5)
    model.draw(3, seed=4, batch=2)
    assert model.sample_cache is None and model.mean_cache is not None
    assert model.samples(q).shape == (3, len(q))


def test_joint_events_and_quality_gate():
    fields = cp.asarray([[1., -1., 1., 1., 1., 1.], [1., 1., 1., -1., 1., 1.]])
    rays = cp.asnumpy(events(fields, (2, 3), 'rays'))
    np.testing.assert_array_equal(rays[0], [[1, 0, 0], [1, 1, 1]])
    collision = cp.asnumpy(events(fields, (2, 3), 'collision'))
    np.testing.assert_array_equal(collision, [[1, 0], [0, 1]])
    ref = {'rays': rays, 'collision': collision}
    assert quality(ref, ref, np.arange(3))['compatibility_pass']
    bad = {'rays': np.zeros_like(rays), 'collision': np.zeros_like(collision)}
    assert not quality(ref, bad, np.arange(3))['compatibility_pass']


def test_local_probability_error_cannot_hide_in_small_rms():
    rays = np.ones((128, 1, 1000), dtype=np.float32)
    reference = {'rays': rays, 'collision': np.tile([0., 1.], (128, 1))}
    changed = rays.copy()
    changed[:16, 0, -1] = 0
    candidate = {**reference, 'rays': changed}
    result = quality(reference, candidate, np.arange(1000))
    assert result['rays']['rms_probability_difference'] < .01
    assert not result['compatibility_pass']
    assert not result['uncertainty_coverage_pass']


def test_collision_calibration_crosses_surface_in_both_directions():
    _, objects, _ = application_geometry(np.zeros((1, 3), dtype='f'))
    for sign in [-1, 1]:
        mean = lambda q: cp.asarray(q[:, 0] * sign)
        probes = uncertain_objects(mean, objects, prior_std=.01)
        values = cp.asnumpy(mean(probes.reshape(-1, 3))).reshape(len(probes), -1).min(1)
        np.testing.assert_allclose(values, np.linspace(-.02, .02, len(probes)), atol=2e-5)


def test_field_gate_detects_lost_uncertainty_even_with_large_mean():
    rng = np.random.default_rng(87)
    fields = cp.asarray(100 + rng.normal(size=(128, 64)) * .001, dtype='f')
    assert field_quality(fields, fields)['pass']
    nearly_same_mean = 100 + (fields - 100) * .5
    result = field_quality(fields, nearly_same_mean)
    assert result['field_relative_l2'] < 1e-5
    assert not result['pass']
