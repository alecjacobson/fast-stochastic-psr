"""Exact finite-cross-kernel variance series prepared from dense Cholesky.

This is a solver/evaluator hybrid: it removes observation count from later
marginal queries, but retains the dense preparation cost. No sampling or
rank truncation is involved. Other correction-factor producers can use the
same FourierSquareAccumulator independently.
"""
import cupy as cp
import cupyx.scipy.linalg as cla
from geospsr.fourier_moments import FourierSquareAccumulator


def prepare_variance_series(points, cholesky, operator, batch=8):
    n = len(points)
    if operator.dtype != cp.dtype('float64') or cholesky.dtype != cp.float64:
        raise ValueError('variance preparation requires double precision')
    if cholesky.shape != (n, n) or n < 1 or batch != int(batch) or batch < 1:
        raise ValueError('invalid factor shape or batch')
    acc = FourierSquareAccumulator(operator.shape[0]//2, xp=cp)
    plans = {}
    for start in range(0, n, batch):
        count = min(batch, n-start)
        rhs = cp.zeros((n, count), dtype='d')
        rhs[start+cp.arange(count), cp.arange(count)] = 1.
        weights = cla.solve_triangular(cholesky.T, rhs, lower=False)
        if count not in plans:
            plans[count] = operator.plan(1, points, count)
        sources = plans[count].execute(cp.ascontiguousarray(weights.T, dtype='complex128'))
        # Each scalar whitened source yields one function per vector component.
        for d in range(3):
            acc.add(-1j*sources*operator.multiplier[d][None])
    norm2 = (operator.ns**2).sum(0)
    prior = float(cp.sum(cp.where(norm2 > 0, operator.rho/cp.where(norm2 > 0, norm2, 1), 0)))
    return acc.coefficients(prior, -1), {'functions':acc.count, 'prior_variance':prior,
        'product_fft_width':acc.fft_width,
        'variance_cutoff':2*(operator.shape[0]//2),
        'scope':'full analytic observation Cholesky and finite scalar/cross kernel; no Monte Carlo'}


class FourierSeriesEvaluator:
    """Evaluate arbitrary centered real Fourier series without kernel setup."""
    def __init__(self, coefficients, eps=1e-11, upsampfac=2.):
        self.coefficients = cp.ascontiguousarray(coefficients, dtype='complex128')
        if self.coefficients.ndim == 3:
            self.coefficients = self.coefficients[None]
        self.shape = self.coefficients.shape[1:]
        self.dtype, self.cdtype = cp.dtype('float64'), cp.dtype('complex128')
        self.eps, self.upsampfac = eps, upsampfac

    def evaluate(self, points):
        from geospsr.spectral import FourierPoisson
        plan = FourierPoisson.plan(self, 2, points, len(self.coefficients))
        return plan.execute(self.coefficients).real
