"""Hausdorff brackets for extracted polylines, not continuous implicit curves."""
import numpy as np
from scipy.spatial import cKDTree


def sample_curves(curves, spacing):
    """Cover every segment by samples at gaps <= spacing, preserving endpoints."""
    if spacing <= 0:
        raise ValueError('spacing must be positive')
    samples = []
    largest_gap = 0.
    for curve in curves:
        curve = np.asarray(curve, dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
            raise ValueError('expected polylines with at least two 2D vertices')
        if not np.isfinite(curve).all():
            raise ValueError('nonfinite contour')
        vectors = np.diff(curve, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        pieces = np.maximum(1, np.ceil(lengths / spacing).astype(int))
        largest_gap = max(largest_gap, float(np.max(lengths / pieces)))
        samples.extend(curve[i] + np.arange(n)[:, None] / n * vectors[i]
                       for i, n in enumerate(pieces))
        samples.append(curve[-1:])
    if not samples:
        raise ValueError('empty level set: Hausdorff distance is undefined here')
    return np.concatenate(samples), largest_gap / 2


def curve_distance(first, second, spacing=1e-4):
    """Symmetric distance bracket from exact nearest-neighbor sample distances.

    Each point on a source polyline is within source_cover of a source sample.
    A nearest target sample overestimates distance to its polyline by at most
    target_cover. Thus directed distance lies in [max_sample-target_cover,
    max_sample+source_cover]. These bounds exclude floating-point roundoff.
    """
    a, ra = sample_curves(first, spacing)
    b, rb = sample_curves(second, spacing)
    ab = cKDTree(b).query(a, eps=0, workers=1)[0]
    ba = cKDTree(a).query(b, eps=0, workers=1)[0]
    lower = max(0., float(ab.max()) - rb, float(ba.max()) - ra)
    upper = max(float(ab.max()) + ra, float(ba.max()) + rb)
    return {'hausdorff_lower': lower, 'hausdorff_upper': upper,
            'sampled_symmetric_max': max(float(ab.max()), float(ba.max())),
            'sampled_max_location': (a[int(ab.argmax())] if ab.max() >= ba.max()
                                     else b[int(ba.argmax())]).tolist(),
            'sample_counts': [len(a), len(b)], 'cover_radii': [ra, rb]}


def curve_statistics(curves):
    return {'components': len(curves),
            'open_components': sum(not np.array_equal(c[0], c[-1]) for c in curves),
            'vertices': sum(len(c) for c in curves),
            'length': sum(float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) for c in curves)}
