"""Whole-box comparisons of represented Chebyshev moment polynomials.

Bounds concern the real polynomials whose coefficients are the stored binary64
numbers. They do NOT include cache construction error, exact-GP error, CDF
calibration, or floating-point evaluation of queries (CPU or GPU).

Arithmetic enclosures use IEEE-754 binary64 operations followed by nextafter,
including every reduction step. This assumes round-to-nearest, correctly
rounded basic arithmetic/sqrt, and gradual underflow (no fast-math/FTZ).
No transcendental function's rounding is assumed. Restriction matrices are
constructed with exact rational arithmetic before enclosing their conversion.
"""
from fractions import Fraction
from functools import lru_cache
import heapq

import numpy as np


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _add(a, b):
    return _down(a[0] + b[0]), _up(a[1] + b[1])


def _sub(a, b):
    lo, hi = _down(a[0] - b[1]), _up(a[1] - b[0])
    # Equal point intervals subtract exactly, even for zero/subnormal inputs.
    same = (a[0] == a[1]) & (b[0] == b[1]) & (a[0] == b[0])
    return np.where(same, 0., lo), np.where(same, 0., hi)


def _mul(a, b):
    products = np.stack([a[i]*b[j] for i in range(2) for j in range(2)])
    lo, hi = _down(products.min(axis=0)), _up(products.max(axis=0))
    zero = ((a[0] == 0) & (a[1] == 0)) | ((b[0] == 0) & (b[1] == 0))
    return np.where(zero, 0., lo), np.where(zero, 0., hi)


def _sum_up(x):
    """Pairwise upper sum of nonnegative numbers, rounding at every step."""
    x = np.asarray(x, dtype=float).ravel()
    if not len(x):
        return 0.
    while len(x) > 1:
        if len(x) % 2:
            x = np.r_[x, 0.]
        a, b = x[::2], x[1::2]
        x = np.where(a == 0, b, np.where(b == 0, a, _up(a+b)))
    return float(x[0])


def _magnitude(poly):
    return _sum_up(np.maximum(np.abs(poly[0]), np.abs(poly[1])))


def _range(poly):
    lo, hi = (p.ravel() for p in poly)
    tail = _sum_up(np.maximum(np.abs(lo[1:]), np.abs(hi[1:])))
    return float(_down(lo[0]-tail)), float(_up(hi[0]+tail))


@lru_cache(maxsize=64)
def _restriction(order, side):
    """Columns expand T_k((t + side)/2) in the T_j(t) basis."""
    a, b = Fraction(side, 2), Fraction(1, 2)
    columns = [[Fraction(1)] + [Fraction(0)]*(order-1)]
    if order > 1:
        columns.append([a, b] + [Fraction(0)]*(order-2))
    for k in range(1, order-1):
        current = columns[k]
        nxt = [2*a*c - p for c, p in zip(current, columns[k-1])]
        nxt[1] += 2*b*current[0]
        for j in range(1, order):
            nxt[j-1] += b*current[j]
            if j+1 < order:
                nxt[j+1] += b*current[j]
        columns.append(nxt)
    exact = list(zip(*columns))
    value = np.array(exact, dtype=float)
    lo, hi = value.copy(), value.copy()
    for index in np.ndindex(value.shape):
        f = Fraction(float(value[index]))
        q = exact[index[0]][index[1]]
        if f > q:
            lo[index] = _down(value[index])
        elif f < q:
            hi[index] = _up(value[index])
    lo.flags.writeable = hi.flags.writeable = False
    return lo, hi


def _restrict(polys, axis, side):
    """Restrict all polynomials to a half-box along one normalized axis."""
    # polys have shape (polynomial, x, y, z); put the split axis last.
    source = tuple(np.moveaxis(p, axis+1, -1) for p in polys)
    order = source[0].shape[-1]
    matrix = _restriction(order, side)
    result = (np.zeros_like(source[0]), np.zeros_like(source[1]))
    for k in range(order):
        if not np.any(source[0][..., k]) and not np.any(source[1][..., k]):
            continue
        term = _mul(tuple(p[..., k, None] for p in source),
                    tuple(m[:, k] for m in matrix))
        result = _add(result, term)
    return tuple(np.moveaxis(p, -1, axis+1) for p in result)


def _exp_negative_upper(t):
    """exp(-t) <= 1 / sum_{k=0}^{20} t**k/k!, for t >= 0.

    Clip downward to avoid overflow; all partial terms/sums are lower bounds.
    """
    t = min(t, 64.)
    term, total = 1., 1.
    for k in range(1, 21):
        term = max(0., float(_down(_down(term*t)/k)))
        total = max(1., float(_down(total+term)))
    return min(1., float(_up(1./total)))


def _box_bounds(polys, prior):
    h_a, h_b, v_a, v_b, dh, dv = [tuple(p[i] for p in polys) for i in range(6)]
    h_ranges, v_ranges = [_range(p) for p in (h_a, h_b)], [_range(p) for p in (v_a, v_b)]
    vmin, vmax = min(r[0] for r in v_ranges), max(r[1] for r in v_ranges)
    eps_h, eps_v = _magnitude(dh), _magnitude(dv)
    result = dict(shifted_mean_error_bound=eps_h, variance_error_bound=eps_v,
                  variance_error_over_prior_bound=float(_up(eps_v/prior)),
                  variance_lower_bound=vmin, probability_error_bound=None)
    if vmin <= 0 or not np.isfinite(vmin):
        return result
    hlo, hhi = min(r[0] for r in h_ranges), max(r[1] for r in h_ranges)
    hmin = max(0., hlo, -hhi)
    hmax = max(abs(hlo), abs(hhi))
    # Enclose derivatives on the convex hull of both (level-mean, variance)
    # ranges. 0.4 > 1/sqrt(2*pi), 0.121 > 1/(2*sqrt(2*pi*e)).
    sqrtv = _down(np.sqrt(vmin))
    exponent_lower = max(0., float(_down(_down(_down(hmin*hmin)/vmax)*.5)))
    decay = _exp_negative_upper(exponent_lower)
    mean_derivative = _up(_up(.4/sqrtv)*decay)
    denominator = _down(vmin*sqrtv)
    if denominator > 0:
        variance_derivative = min(_up(.121/vmin),
                                  _up(_up(_up(.2*hmax)/denominator)*decay))
    else:
        variance_derivative = _up(.121/vmin)
    result['probability_error_bound'] = float(min(1., _up(
        _up(mean_derivative*eps_h) + _up(variance_derivative*eps_v))))
    return result


def _field_arrays(field, shape):
    result = []
    for key in ('mean', 'correction'):
        raw = np.asarray(getattr(field, key), dtype=float)
        if raw.ndim != 3 or min(raw.shape) < 1 or not np.isfinite(raw).all():
            raise ValueError('field coefficients must be finite nonempty 3D arrays')
        padded = np.zeros(shape)
        padded[tuple(slice(0, d) for d in raw.shape)] = raw
        result.append(padded)
    return result


@np.errstate(over='raise', invalid='raise', divide='raise', under='ignore')
def compare_moment_fields(candidate, reference, *, candidate_level=0., reference_level=0.,
                          probability_tolerance=1e-3, variance_tolerance=1e-4,
                          max_leaves=256, max_depth=24):
    """Bound differences everywhere in a shared physical box.

    Returns ``bounded`` only when positive variances and both tolerances hold
    over a complete partition. ``inconclusive`` is NOT a demonstrated failure.
    Variance error is normalized by the reference prior variance. Unequal
    polynomial orders and separately anchored probability levels are supported.
    """
    for field in (candidate, reference):
        center, width = np.asarray(field.center), np.asarray(field.halfwidth)
        if (center.shape != (3,) or width.shape != (3,) or
                not np.isfinite(center).all() or not np.isfinite(width).all() or np.any(width <= 0)):
            raise ValueError('fields must have finite centers and positive halfwidths')
        if not np.isfinite(field.prior_variance) or field.prior_variance <= 0:
            raise ValueError('prior variances must be finite and positive')
    if not (np.array_equal(candidate.center, reference.center) and
            np.array_equal(candidate.halfwidth, reference.halfwidth)):
        raise ValueError('fields must use identical physical boxes')
    if not all(np.isfinite(x) for x in (candidate_level, reference_level)):
        raise ValueError('probability levels must be finite')
    if not all(np.isfinite(x) and x > 0 for x in (probability_tolerance, variance_tolerance)):
        raise ValueError('tolerances must be finite and positive')
    if (not isinstance(max_leaves, int) or max_leaves < 1 or
            not isinstance(max_depth, int) or max_depth < 0):
        raise ValueError('max_leaves must be positive and max_depth nonnegative integers')
    shapes = [np.asarray(getattr(f, k)).shape for f in (candidate, reference)
              for k in ('mean', 'correction')]
    if any(len(s) != 3 or min(s) < 1 for s in shapes):
        raise ValueError('field coefficients must be nonempty 3D arrays')
    shape = tuple(max(s[i] for s in shapes) for i in range(3))
    means, variances = [], []
    for field, level in ((candidate, candidate_level), (reference, reference_level)):
        mean, correction = _field_arrays(field, shape)
        constant = np.zeros(shape)
        constant[0, 0, 0] = level
        means.append(_sub((constant, constant), (mean, mean)))
        constant = np.zeros(shape)
        constant[0, 0, 0] = field.prior_variance
        variances.append(_sub((constant, constant), (correction, correction)))
    polys = means + variances + [_sub(*means), _sub(*variances)]
    polys = tuple(np.stack([p[i] for p in polys]) for i in range(2))
    prior = reference.prior_variance
    initial = _box_bounds(polys, prior)

    def score(bound):
        if bound['probability_error_bound'] is None:
            return np.inf
        return max(bound['probability_error_bound']/probability_tolerance,
                   bound['variance_error_over_prior_bound']/variance_tolerance)

    def acceptable(bound):
        return (bound['probability_error_bound'] is not None and
                bound['probability_error_bound'] <= probability_tolerance and
                bound['variance_error_over_prior_bound'] <= variance_tolerance)

    queue, finished, serial = [], [], 0

    def insert(poly, depth, bounds):
        nonlocal serial
        serial += 1
        if acceptable(bounds) or depth >= max_depth:
            finished.append((depth, bounds))
        else:
            heapq.heappush(queue, (-score(bounds), serial, depth, poly, bounds))

    insert(polys, 0, initial)
    while queue and len(queue)+len(finished) < max_leaves:
        _, _, depth, parent, _ = heapq.heappop(queue)
        # Split where the represented functions vary most. This heuristic does
        # not enter the bound, but avoids subdividing inactive coordinates.
        magnitude = np.maximum(np.abs(parent[0]), np.abs(parent[1]))
        scales = np.array([np.sqrt(prior), np.sqrt(prior), prior, prior, np.sqrt(prior), prior])
        weighted = magnitude / scales[:, None, None, None]
        variation = [np.sum(np.take(weighted, range(1, shape[a]), axis=a+1)) for a in range(3)]
        axis = int(np.argmax(variation))
        for side in (-1, 1):
            child = _restrict(parent, axis, side)
            insert(child, depth+1, _box_bounds(child, prior))
    leaves = finished + [(item[2], item[4]) for item in queue]
    aggregate = {k: (None if any(b[k] is None for _, b in leaves) else
                     (min if k == 'variance_lower_bound' else max)(b[k] for _, b in leaves))
                 for k in initial}
    return dict(status='bounded' if all(acceptable(b) for _, b in leaves) else 'inconclusive',
                **aggregate, initial_box_bounds=initial, leaves=len(leaves),
                deepest_leaf=max(d for d, _ in leaves),
                probability_tolerance=probability_tolerance, variance_tolerance=variance_tolerance,
                scope='Real polynomials represented by the stored binary64 coefficients; '
                      'excludes construction, exact-GP, calibration and query-evaluation error.',
                rounding='Outward binary64 intervals; assumes IEEE-754 round-to-nearest '
                         'basic arithmetic/sqrt and gradual underflow.')
