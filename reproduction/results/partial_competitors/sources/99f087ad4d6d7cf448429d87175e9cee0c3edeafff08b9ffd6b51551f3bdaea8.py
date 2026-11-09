"""Optional HiGP H² kernel actions and batched preconditioned CG.

The native extension uses a spatial tree and nested low-rank far interactions,
an algebraic/kernel-independent FMM-style representation. This wrapper is not
an analytic Laplace FMM and does not approximate the Poisson kernel by wrapping.
Build the pinned extension with reproduction/setup_higp.py.
"""
import numpy as np
from time import perf_counter

from .nonperiodic import _points


class HierarchicalMaternSystem:
    def __init__(self, model, points, noise_variance, tolerance=1e-8,
                 preconditioner_rank=50, neighbors=40, threads=4, seed=12,
                 leaf_size=400, build_hierarchy=True, preconditioner_backend='native'):
        if preconditioner_backend == 'native':
            import psr_higp_cext as native
        elif preconditioner_backend == 'prefix-kd':
            import psr_higp_kd_cext as native
        else:
            raise ValueError('unknown preconditioner backend')
        self.model = model
        self.points = _points(points).copy()
        n = len(self.points)
        if n < 2 or not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError('at least two points and positive scalar noise variance required')
        if not 0 < tolerance < 1 or threads < 1 or leaf_size < 1:
            raise ValueError('invalid tolerance, thread count or leaf size')
        if preconditioner_rank < 1 or neighbors < 1:
            raise ValueError('preconditioner rank and neighbor count must be positive')
        self.noise_variance = float(noise_variance)
        # Native code reads raw C buffers; do not pass transposed/strided views.
        self.coordinates = np.ascontiguousarray(self.points.T)
        kw = dict(data=self.coordinates, kernel_type=2, l=model.lengthscale,
                  f=np.sqrt(model.amplitude), s=self.noise_variance,
                  nthreads=int(threads), seed=int(seed))
        start = perf_counter()
        self.operator = (native.krnlmat.setup(**kw, use_h2=1, h2_tol=tolerance,
                                             h2_leaf_nmax=int(leaf_size)) if build_hierarchy else None)
        hierarchy_seconds = perf_counter()-start
        start = perf_counter()
        self.preconditioner = native.precond.setup_afn(
            **kw, rank=min(int(preconditioner_rank), n-1), lfil=min(int(neighbors), n-1))
        self.setup_timings = dict(hierarchy_seconds=hierarchy_seconds,
                                  preconditioner_seconds=perf_counter()-start,
                                  preconditioner_backend=preconditioner_backend)
        self.last_solve = None

    def _apply(self, native_operator, value):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim not in (1, 2) or value.shape[0] != len(self.points):
            raise ValueError('RHS must have shape (n,) or (n,k)')
        if not np.isfinite(value).all():
            raise ValueError('RHS must be finite')
        if value.ndim == 1:
            return native_operator.matmul(np.ascontiguousarray(value))
        if not value.shape[1]:
            return np.empty_like(value)
        result = native_operator.matmul(np.ascontiguousarray(value.T))
        return np.asarray(result).reshape(value.shape[1], value.shape[0]).T.copy()

    def matmul(self, value):
        return self._apply(self.operator, value)

    def precondition(self, value):
        return self._apply(self.preconditioner, value)

    def solve(self, rhs, tolerance=1e-7, max_iterations=500):
        solution, info = batched_pcg(self.matmul, self.precondition, rhs,
                                     tolerance=tolerance, max_iterations=max_iterations)
        self.last_solve = info
        if not info['converged']:
            raise RuntimeError(f'PCG did not converge: {info}')
        return solution


def batched_pcg(matmul, precondition, rhs, tolerance=1e-7, max_iterations=500, xp=np,
                return_residual=False):
    """Independent PCG recurrences sharing native matrix-matrix actions.

    Unlike block-CG, dependent columns do not require a small Gram inverse.
    Deflate finished columns and verify convergence with fresh residuals.
    """
    b = xp.asarray(rhs, dtype=float)
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    if b.ndim != 2 or not xp.isfinite(b).all() or not 0 < tolerance < 1 or max_iterations < 1:
        raise ValueError('invalid RHS or PCG stopping parameters')
    x = xp.zeros_like(b)
    norms = xp.linalg.norm(b, axis=0)
    indices = xp.flatnonzero(norms > 0)
    iterations = xp.zeros(b.shape[1], dtype=int)
    r = b[:, indices].copy()
    z = precondition(r)
    p = z.copy()
    rz = xp.sum(r*z, axis=0)
    for iteration in range(max_iterations):
        if not len(indices):
            break
        ap = matmul(p)
        pap = xp.sum(p*ap, axis=0)
        if xp.any(pap <= 0) or xp.any(rz <= 0):
            raise np.linalg.LinAlgError('nonpositive curvature in PCG')
        alpha = rz/pap
        x[:, indices] += p*alpha
        r -= ap*alpha
        iterations[indices] = iteration+1
        done = xp.linalg.norm(r, axis=0) <= tolerance*norms[indices]
        replaced = done.copy()
        if xp.any(done):
            # Fresh residuals prevent a recurrence from claiming false accuracy.
            r[:, done] = b[:, indices[done]] - matmul(x[:, indices[done]])
            done = xp.linalg.norm(r, axis=0) <= tolerance*norms[indices]
        keep = ~done
        indices, r, p, rz = indices[keep], r[:, keep], p[:, keep], rz[keep]
        if not len(indices):
            break
        z = precondition(r)
        new_rz = xp.sum(r*z, axis=0)
        p = z + p*xp.where(replaced[keep], 0., new_rz/rz)
        rz = new_rz
    residual = b-matmul(x)
    relative = xp.linalg.norm(residual, axis=0)/xp.where(norms > 0, norms, 1.)
    host = np.asarray if xp is np else xp.asnumpy
    info = dict(converged=bool(xp.all(relative <= tolerance)),
                iterations=host(iterations).tolist(), relative_residuals=host(relative).tolist(),
                max_iterations=int(host(iterations).max(initial=0)), tolerance=tolerance)
    result = (x[:, 0] if vector else x), info
    if return_residual:
        return (*result, residual[:, 0] if vector else residual)
    return result
