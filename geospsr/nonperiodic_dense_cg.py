"""Strong GPU dense-action PCG control using the same AFN preconditioner.

Cache the matrix when it fits; otherwise regenerate bounded row blocks. This
separates hierarchical compression from preconditioning and regional caching.
"""
import numpy as np

from .hierarchical import HierarchicalMaternSystem
from .hierarchical_gpu import GPUAFNPreconditioner, GPUHierarchicalSystem
from .nonperiodic_dense_gpu import _KERNEL


class GPUDenseCGSystem(GPUHierarchicalSystem):
    def __init__(self, model, points, noise_variance, dtype='float32', rank=50,
                 neighbors=40, threads=4, seed=12, block_size=1024, cache=True,
                 preconditioner_backend='native'):
        import cupy as cp
        self.native = HierarchicalMaternSystem(model, points, noise_variance,
                                               preconditioner_rank=rank, neighbors=neighbors,
                                               threads=threads, seed=seed, build_hierarchy=False,
                                               preconditioner_backend=preconditioner_backend)
        self.model, self.points = model, self.native.points
        self.noise_variance = noise_variance
        self.preconditioner_gpu = GPUAFNPreconditioner(self.native.preconditioner)
        self.dtype = np.dtype(dtype)
        self.block_size = block_size
        self.coordinates = cp.asarray(self.points, dtype=self.dtype)
        self.kernel = cp.RawKernel(_KERNEL.replace('TYPE', 'float' if self.dtype == np.float32 else 'double'), 'covariance')
        n = len(self.points)
        available = cp.cuda.runtime.memGetInfo()[0]
        self.matrix = self._rows(0, n) if cache and n*n*self.dtype.itemsize < available-1024**3 else None
        self.matrix_cached = self.matrix is not None
        self.device_matrix_bytes = self.preconditioner_gpu.nbytes+(self.matrix.nbytes if self.matrix_cached else 0)
        self.last_solve = None
        cp.cuda.get_current_stream().synchronize()

    def _rows(self, start, nr):
        import cupy as cp
        n = len(self.points)
        a = cp.empty((nr, n), dtype=self.dtype)
        scalar = self.dtype.type
        self.kernel(((nr*n+255)//256,), (256,),
                    (self.coordinates, a, n, start, nr, scalar(self.model.kappa),
                     scalar(self.model.amplitude), scalar(self.noise_variance)))
        return a

    def matmul(self, rhs):
        import cupy as cp
        b = cp.asarray(rhs, dtype=self.dtype)
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        if self.matrix_cached:
            result = (self.matrix@b).astype(cp.float64)
        else:
            result = cp.empty(b.shape, dtype=cp.float64)
            for start in range(0, len(b), self.block_size):
                nr = min(self.block_size, len(b)-start)
                result[start:start+nr] = self._rows(start, nr)@b
        return result[:, 0] if vector else result

    def accurate_matmul(self, rhs):
        """Float64 accumulation against the stored (possibly float32) kernel."""
        import cupy as cp
        b = cp.asarray(rhs, dtype=cp.float64)
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        result = cp.empty_like(b)
        for start in range(0, len(b), self.block_size):
            nr = min(self.block_size, len(b)-start)
            rows = self.matrix[start:start+nr] if self.matrix_cached else self._rows(start, nr)
            result[start:start+nr] = rows.astype(cp.float64)@b
        return result[:, 0] if vector else result

    def project(self, basis, vectors, variance_iterations=8, mean_tolerance=1e-6,
                variance_refinements=1):
        """Fast low-precision inner steps, reliable high-precision correction.

        The final quadratic form uses a float64 residual after the same fixed
        variance-step budget as H2, followed by one reliable correction sweep
        to compensate float32 action error. Only mean columns are subsequently
        refined to a tight residual; over-solving variance weakens the control.
        """
        import cupy as cp
        from .hierarchical import batched_pcg
        if not 0 <= variance_refinements <= 4:
            raise ValueError('variance_refinements must be between zero and four')
        b = cp.asarray(np.column_stack((basis, vectors)))
        z, inner, _ = batched_pcg(self.matmul, self.precondition, b, tolerance=1e-10,
                                  max_iterations=variance_iterations, xp=cp, return_residual=True)
        r = b-self.accurate_matmul(z)
        history = [inner]
        norms = cp.linalg.norm(b[:, -3:], axis=0)
        for step in range(4):
            if step >= variance_refinements and bool(cp.all(cp.linalg.norm(r[:, -3:], axis=0) <= mean_tolerance*norms)):
                break
            columns = slice(None) if step < variance_refinements else slice(-3, None)
            dz, inner = batched_pcg(self.matmul, self.precondition, r[:, columns], tolerance=.003,
                                    max_iterations=variance_iterations, xp=cp)
            history.append(inner)
            z[:, columns] += dz
            r[:, columns] = b[:, columns]-self.accurate_matmul(z[:, columns])
        relative = cp.linalg.norm(r[:, -3:], axis=0)/norms
        self.last_solve = dict(refinement_steps=len(history)-1, inner=history,
                              mean_relative_residuals=cp.asnumpy(relative).tolist(),
                              variance_relative_residuals=cp.asnumpy(cp.linalg.norm(r[:, :-3], axis=0)/cp.linalg.norm(b[:, :-3], axis=0)).tolist(),
                              mean_converged=bool(cp.all(relative <= mean_tolerance)),
                              variance_step_budget=variance_iterations,
                              variance_refinement_sweeps=variance_refinements,
                              residual_scope='float64 accumulation of stored kernel entries')
        if not self.last_solve['mean_converged']:
            raise RuntimeError(f'dense mixed precision refinement did not converge: {self.last_solve}')
        u, zu, ru = b[:, :-3], z[:, :-3], r[:, :-3]
        gram = u.T@zu+zu.T@ru
        return cp.asnumpy(.5*(gram+gram.T)), cp.asnumpy(u.T@z[:, -3:])
