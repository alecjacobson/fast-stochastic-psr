"""Memory-efficient GPU dense reference for the free-space model.

Factors in place to avoid claiming a hierarchical win over an avoidable second
N-by-N allocation. A small number of solve/projection RHS remain separate.
"""
import numpy as np

from .nonperiodic import _points


_KERNEL = r'''
extern "C" __global__ void covariance(const TYPE* x, TYPE* a, long long n,
    long long begin, long long nr, TYPE kappa, TYPE amplitude, TYPE noise) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=nr*n)return;
    long long i=begin+t/n,j=t%n;
    TYPE d=0;
    for(int k=0;k<3;k++){TYPE v=x[3*i+k]-x[3*j+k];d+=v*v;}
    TYPE z=sqrt(d)*kappa;
    a[t]=amplitude*(1+z)*exp(-z)+(i==j?noise:0);
}
'''


def full_kernel_matmul(model, points, noise_variance, rhs, block_size=512):
    """Independent float64 full-kernel action with bounded row storage."""
    import cupy as cp
    x = cp.asarray(points, dtype=cp.float64)
    b = cp.asarray(rhs, dtype=cp.float64)
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    n = len(x)
    block_size = min(block_size, max(1, (512*2**20)//(n*8)))
    out = cp.empty_like(b)
    kernel = cp.RawKernel(_KERNEL.replace('TYPE', 'double'), 'covariance')
    for start in range(0, n, block_size):
        nr = min(block_size, n-start)
        a = cp.empty((nr, n), dtype=cp.float64)
        kernel(((nr*n+255)//256,), (256,),
               (x, a, n, start, nr, model.kappa, model.amplitude, noise_variance))
        out[start:start+nr] = a@b
    result = cp.asnumpy(out)
    return result[:, 0] if vector else result


class GPUDenseMaternSystem:
    def __init__(self, model, points, noise_variance, dtype='float32'):
        import cupy as cp
        from cupy.cuda import cusolver, cublas, device
        self.points = _points(points).copy()
        self.model = model
        self.noise_variance = float(noise_variance)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('dtype must be float32 or float64')
        if not len(self.points) or not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError('nonempty observations and positive noise variance required')
        self.coordinates = cp.asarray(self.points, dtype=self.dtype)
        n = len(self.points)
        # Fortran order lets triangular solves reuse this allocation.
        self.factor = cp.empty((n, n), dtype=self.dtype, order='F')
        typ = 'float' if self.dtype == np.float32 else 'double'
        self.kernel = cp.RawKernel(_KERNEL.replace('TYPE', typ), 'covariance')
        scalar = self.dtype.type
        self.kernel(((n*n+255)//256,), (256,),
                    (self.coordinates, self.factor, n, 0, n, scalar(model.kappa),
                     scalar(model.amplitude), scalar(noise_variance)))
        handle = device.get_cusolver_handle()
        if self.dtype == np.float32:
            potrf, size = cusolver.spotrf, cusolver.spotrf_bufferSize
        else:
            potrf, size = cusolver.dpotrf, cusolver.dpotrf_bufferSize
        fill = cublas.CUBLAS_FILL_MODE_LOWER
        length = size(handle, fill, n, self.factor.data.ptr, n)
        workspace = cp.empty(length, dtype=self.dtype)
        info = cp.empty(1, dtype=cp.int32)
        potrf(handle, fill, n, self.factor.data.ptr, n, workspace.data.ptr, length, info.data.ptr)
        error = int(info.item())
        if error:
            raise np.linalg.LinAlgError(f'GPU Cholesky failed with info={error}')
        self.factor_bytes = self.factor.nbytes
        self.factor_workspace_bytes = workspace.nbytes

    def solve(self, rhs):
        import cupy as cp
        from cupyx.scipy.linalg import solve_triangular
        b = cp.asarray(rhs, dtype=self.dtype)
        w = solve_triangular(self.factor, b, lower=True, check_finite=False)
        x = solve_triangular(self.factor, w, lower=True, trans='T', check_finite=False)
        return cp.asnumpy(x).astype(float)

    def project(self, basis, vectors):
        import cupy as cp
        from cupyx.scipy.linalg import solve_triangular
        b = cp.asarray(np.column_stack((basis, vectors)), dtype=self.dtype)
        w = solve_triangular(self.factor, b, lower=True, check_finite=False).astype(cp.float64)
        u, y = w[:, :-3], w[:, -3:]
        return cp.asnumpy(u.T@u), cp.asnumpy(u.T@y)

    def exact_matmul(self, rhs, block_size=512):
        """Independent float64 full-kernel action, tiled to avoid dense storage."""
        return full_kernel_matmul(self.model, self.points, self.noise_variance, rhs, block_size)

    def refined_solve(self, rhs, tolerance=1e-9, max_steps=5):
        b = np.asarray(rhs, dtype=float)
        x = self.solve(b)
        norm = np.linalg.norm(b, axis=0)
        for _ in range(max_steps):
            r = b-self.exact_matmul(x)
            if np.all(np.linalg.norm(r, axis=0) <= tolerance*norm):
                self.refinement_info = dict(steps=_, relative_residuals=(np.linalg.norm(r, axis=0)/norm).tolist())
                return x
            x += self.solve(r)
        raise RuntimeError('mixed precision reference refinement did not converge')
