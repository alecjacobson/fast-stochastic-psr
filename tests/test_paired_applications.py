"""Paired backend comparisons must preserve draws and reject bad event outputs."""
import cupy as cp
import numpy as np

from reproduction.paired_applications import events, quality
from reproduction.posterior import Posterior


def test_retained_draw_components_reconstruct_same_function():
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, (13, 3)).astype('f')
    normals = rng.normal(size=x.shape).astype('f')
    q = rng.uniform(-1, 1, (9, 3)).astype('f')
    model = Posterior(x, normals, cutoff=2, prior_cutoff=2)
    model.draw(7, seed=5, batch=3, retain_components=True)
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
