from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

from geospsr.regional_bounds import compare_moment_fields, _restrict, _exp_negative_upper


def field(mean=0., variance=1., order=1):
    m = np.zeros((order,)*3)
    c = np.zeros_like(m)
    m[0, 0, 0] = mean
    c[0, 0, 0] = 2.-variance
    return SimpleNamespace(center=np.zeros(3), halfwidth=np.ones(3),
                           mean=m, correction=c, prior_variance=2.)


def test_constants_and_different_orders_levels():
    a, b = field(mean=.2, variance=.7), field(mean=.3, variance=.8, order=5)
    result = compare_moment_fields(a, b, candidate_level=.4, reference_level=.5,
                                  probability_tolerance=.1, variance_tolerance=.1)
    assert result['status'] == 'bounded'
    expected = abs(ndtr(.2/np.sqrt(.7))-ndtr(.2/np.sqrt(.8)))
    assert expected <= result['probability_error_bound'] < .025
    assert .1 <= result['variance_error_bound'] < .1+1e-14
    # A shift to the mean and its level leaves the probability unchanged.
    b.correction[0, 0, 0] = a.correction[0, 0, 0]
    result = compare_moment_fields(a, b, candidate_level=.4, reference_level=.5)
    assert result['probability_error_bound'] < 1e-14


def test_negative_variance_is_inconclusive_and_never_clipped():
    a, b = field(variance=-.1), field()
    result = compare_moment_fields(a, b, max_leaves=8)
    assert result['status'] == 'inconclusive'
    assert result['variance_lower_bound'] < 0
    assert result['probability_error_bound'] is None
    from reproduction.bound_hierarchical_application import inspect_leaf_probes
    probes = inspect_leaf_probes(result, dict(field=vars(a), level=0.), dict(field=vars(b), level=0.))
    assert probes['invalid_queries'] == 27
    assert probes['observed_probability_difference'] is None


def test_subdivision_recovers_positive_variance_and_tightens_probability():
    a, b = field(order=3), field(order=3)
    # v(x) = .1+(x-.5)^2 = .85 - x + .5*T2(x).
    for f in (a, b):
        f.correction[0, 0, 0] = 2.-.85
        f.correction[1, 0, 0] = 1.
        f.correction[2, 0, 0] = -.5
    a.mean[0, 0, 0] = 1e-5
    initial = compare_moment_fields(a, b, max_leaves=1)
    assert initial['probability_error_bound'] is None
    result = compare_moment_fields(a, b, max_leaves=128)
    assert result['status'] == 'bounded'
    assert result['variance_lower_bound'] > 0
    assert result['leaves'] > 1


def test_subdivision_uses_local_cdf_sensitivity():
    a, b = field(order=2), field(order=2)
    # h differs mainly in the tails, where Phi is insensitive.
    a.mean[1, 0, 0] = 10.01
    b.mean[1, 0, 0] = 10.
    initial = compare_moment_fields(a, b, max_leaves=1)
    assert initial['probability_error_bound'] > .001
    result = compare_moment_fields(a, b, max_leaves=256)
    assert result['status'] == 'bounded'
    assert result['probability_error_bound'] <= .001


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('side', [-1, 1])
def test_restriction_contains_exact_rational_polynomial(axis, side):
    # Independent recurrence at rational points, with exact coefficients.
    rng = np.random.default_rng(12)
    coefficients = rng.integers(-20, 21, size=(1, 7, 5, 3)).astype(float)/16
    lower, upper = _restrict((coefficients, coefficients), axis, side)

    def evaluate(c, x):
        basis = []
        for n, t in zip(c.shape, x):
            ts = [Fraction(1), t]
            for _ in range(2, n):
                ts.append(2*t*ts[-1]-ts[-2])
            basis.append(ts[:n])
        return sum(Fraction(float(c[idx]))*basis[0][idx[0]]*basis[1][idx[1]]*basis[2][idx[2]]
                   for idx in np.ndindex(c.shape))

    for point in ((Fraction(-1), Fraction(1, 3), Fraction(1)),
                  (Fraction(1, 7), Fraction(-2, 5), Fraction(3, 11))):
        original = list(point)
        original[axis] = (original[axis]+side)/2
        exact = evaluate(coefficients[0], original)
        # Coefficient interval radius gives an independent evaluation enclosure.
        midpoint = (lower[0]+upper[0])/2
        radius = sum(max(abs(Fraction(float(l))-Fraction(float(m))),
                         abs(Fraction(float(u))-Fraction(float(m))))
                     for l, m, u in zip(lower.ravel(), midpoint.ravel(), upper.ravel()))
        assert abs(evaluate(midpoint, point)-exact) <= radius


def test_random_polynomials_dense_query_crosscheck():
    rng = np.random.default_rng(512)
    a, b = field(order=5), field(order=3)
    a.mean += rng.normal(scale=.0001, size=a.mean.shape)
    a.correction += rng.normal(scale=.001, size=a.mean.shape)
    b.mean += rng.normal(scale=.0001, size=b.mean.shape)
    b.correction += rng.normal(scale=.001, size=b.mean.shape)
    result = compare_moment_fields(a, b, max_leaves=32)
    points = rng.uniform(-1, 1, (10000, 3))
    evaluate = np.polynomial.chebyshev.chebval3d
    ma, mb = [evaluate(*points.T, f.mean) for f in (a, b)]
    va, vb = [f.prior_variance-evaluate(*points.T, f.correction) for f in (a, b)]
    assert np.max(np.abs(ma-mb)) <= result['shifted_mean_error_bound']
    assert np.max(np.abs(va-vb)) <= result['variance_error_bound']
    assert min(va.min(), vb.min()) >= result['variance_lower_bound']
    assert np.max(np.abs(ndtr(-ma/np.sqrt(va))-ndtr(-mb/np.sqrt(vb)))) <= result['probability_error_bound']


def test_interior_violation_is_not_accepted():
    a, b = field(order=3), field(order=1)
    # Zero difference at the box corners; interior discrepancy > tolerance.
    a.mean[0, 0, 0], a.mean[2, 0, 0] = .01, -.01
    result = compare_moment_fields(a, b, max_leaves=64)
    assert result['status'] == 'inconclusive'
    assert result['probability_error_bound'] >= ndtr(.02)-.5 > .001
    from reproduction.bound_hierarchical_application import inspect_leaf_probes
    observed = inspect_leaf_probes(result, dict(field=vars(a), level=0.),
                                   dict(field=vars(b), level=0.))
    assert observed['observed_above_tolerance']
    assert abs(observed['normalized_probe'][0]) < .5


def test_exponential_enclosure_against_high_precision():
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 60
        for x in (0., 1e-12, .01, 1., 8., 40., 100., 1e6):
            assert Decimal(_exp_negative_upper(x)) >= (-Decimal(x)).exp()


def test_arithmetic_overflow_fails_closed():
    a, b = field(mean=1e308), field(mean=-1e308)
    with pytest.raises(FloatingPointError):
        compare_moment_fields(a, b)


def test_very_small_variance_does_not_underflow_to_false_acceptance():
    a, b = field(), field()
    for f in (a, b):
        f.prior_variance = 1e-300
        f.correction.fill(0.)
    a.mean[0, 0, 0] = 1e-150
    result = compare_moment_fields(a, b, max_leaves=1)
    assert result['status'] == 'inconclusive'
    assert result['probability_error_bound'] >= ndtr(1.)-.5


@pytest.mark.parametrize('invalid', ['box', 'width', 'coefficients', 'level', 'leaves', 'tolerance'])
def test_invalid_input(invalid):
    a, b, kwargs = field(), field(), {}
    if invalid == 'box':
        a.center[0] = 1.
    elif invalid == 'width':
        a.halfwidth[0] = 0.
    elif invalid == 'coefficients':
        a.mean[0, 0, 0] = np.nan
    elif invalid == 'level':
        kwargs['candidate_level'] = np.inf
    elif invalid == 'leaves':
        kwargs['max_leaves'] = 0
    elif invalid == 'tolerance':
        kwargs['probability_tolerance'] = np.nan
    with pytest.raises(ValueError):
        compare_moment_fields(a, b, **kwargs)
