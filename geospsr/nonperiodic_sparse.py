"""Distance-truncated GPU Matérn actions as a strong short-range control.

This approximates the original full kernel; it is not a covariance taper and
does not guarantee positive definiteness. CG curvature failures are explicit,
and posterior outputs require an independent full-kernel audit. In a fixed
domain with fixed radius, the neighbor count can grow with N: sparsity alone
does not imply asymptotically linear work or storage.
"""
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .hierarchical import HierarchicalMaternSystem
from .hierarchical_gpu import GPUAFNPreconditioner, GPUHierarchicalSystem


_FILL = r'''
extern "C" __global__ void fill_sparse(
    const double* x, const int* ptr, const int* columns, double* values,
    double kappa, double amplitude, double noise, int n) {
    int i=blockIdx.x;
    if(i>=n) return;
    for(int p=ptr[i]+threadIdx.x;p<ptr[i+1];p+=blockDim.x) {
        int j=columns[p]; double d=0;
        for(int k=0;k<3;k++) {double v=x[3*i+k]-x[3*j+k]; d+=v*v;}
        double z=sqrt(d)*kappa;
        values[p]=amplitude*(1+z)*exp(-z)+(i==j ? noise : 0.);
    }
}
'''


_RADIUS = r'''
extern "C" __global__ void radius_graph(
    const double* x, const long long* cells, const int* offsets,
    int ny, int nz, int nx, double radius2, int n,
    int* counts, const int* ptr, int* columns, int fill) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    long long cell=cells[i];
    int iz=cell%nz, iy=(cell/nz)%ny, ix=cell/(ny*nz);
    double a=x[3*i], b=x[3*i+1], c=x[3*i+2];
    int count=0, position=fill ? ptr[i] : 0;
    // Cell IDs and the coordinates within each cell are ordered, so each row
    // is generated sorted and unique without a global COO/CSR sorting step.
    for(int u=max(0,ix-1);u<=min(nx-1,ix+1);u++)
    for(int v=max(0,iy-1);v<=min(ny-1,iy+1);v++)
    for(int w=max(0,iz-1);w<=min(nz-1,iz+1);w++) {
        int bin=(u*ny+v)*nz+w;
        for(int j=offsets[bin];j<offsets[bin+1];j++) {
            double dx=a-x[3*j], dy=b-x[3*j+1], dz=c-x[3*j+2];
            if(dx*dx+dy*dy+dz*dz<=radius2) {
                if(fill) columns[position+count]=j;
                count++;
            }
        }
    }
    if(!fill) counts[i]=count;
}
'''


def _gpu_graph(points, radius):
    import cupy as cp
    x = cp.asarray(points)
    bins = cp.floor((x-cp.min(x, axis=0))/radius).astype(cp.int64)
    sizes = cp.asnumpy(bins.max(axis=0)+1)
    nx, ny, nz = map(int, sizes)
    if nx*ny*nz > max(1000000, 64*len(x)):
        raise ValueError('uniform grid too large; use the CPU tree builder for sparse/elongated domains')
    ids = (bins[:, 0]*ny+bins[:, 1])*nz+bins[:, 2]
    permutation = cp.argsort(ids)
    ids = ids[permutation]; x = x[permutation]
    offsets = cp.r_[cp.zeros(1, dtype=cp.int32), cp.cumsum(cp.bincount(ids, minlength=nx*ny*nz), dtype=cp.int32)]
    counts = cp.empty(len(x), dtype=cp.int32)
    kernel = cp.RawKernel(_RADIUS, 'radius_graph')
    common = (x, ids, offsets, ny, nz, nx, radius*radius, len(x), counts)
    # Empty arrays are legal unused pointer arguments in the counting pass.
    empty = cp.empty(0, dtype=cp.int32)
    kernel(((len(x)+127)//128,), (128,), common+(empty, empty, 0))
    indptr64 = cp.r_[cp.zeros(1, dtype=cp.int64), cp.cumsum(counts, dtype=cp.int64)]
    nnz = int(indptr64[-1])
    _check_capacity(nnz, len(x))
    ptr = indptr64.astype(cp.int32)
    columns = cp.empty(nnz, dtype=cp.int32)
    kernel(((len(x)+127)//128,), (128,), common+(ptr, columns, 1))
    return x, permutation, ptr, columns, cp.asnumpy(counts)


def _check_capacity(nnz, n):
    import cupy as cp
    if nnz >= 2**31 or n >= 2**31:
        raise ValueError('sparse control requires 32-bit CSR indexing')
    if nnz*12+(n+1)*4 > .75*cp.cuda.runtime.memGetInfo()[0]:
        raise MemoryError(f'sparse graph requires {nnz*12+(n+1)*4} bytes before solve workspace')


class GPUSparseMaternSystem(GPUHierarchicalSystem):
    def __init__(self, model, points, noise_variance, cutoff=12., rank=8,
                 neighbors=24, threads=4, seed=12, block_size=256, graph_backend='gpu',
                 preconditioner_backend='native', rhs_block_size=None):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
        if not np.isfinite(cutoff) or cutoff <= 0 or block_size < 1:
            raise ValueError('cutoff and block size must be positive')
        start = perf_counter()
        self.native = HierarchicalMaternSystem(model, points, noise_variance,
                                               preconditioner_rank=rank, neighbors=neighbors,
                                               threads=threads, seed=seed, build_hierarchy=False,
                                               preconditioner_backend=preconditioner_backend)
        self.model, self.points = model, self.native.points
        self.noise_variance = noise_variance
        self.rhs_block_size = rhs_block_size
        self.preconditioner_gpu = GPUAFNPreconditioner(self.native.preconditioner)
        preconditioner_seconds = perf_counter()-start
        start = perf_counter()
        radius = cutoff/model.kappa
        n = len(self.points)
        if graph_backend == 'gpu':
            coordinates, self.permutation, ptr, columns, counts = _gpu_graph(self.points, radius)
        elif graph_backend == 'cpu':
            tree = cKDTree(self.points)
            counts = tree.query_ball_point(self.points, radius, workers=threads, return_length=True)
            indptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
            _check_capacity(int(indptr[-1]), n)
            columns = cp.empty(int(indptr[-1]), dtype=cp.int32)
            # Bounded CPU neighbor lists avoid an O(nnz) Python-object peak.
            for first in range(0, n, block_size):
                last = min(first+block_size, n)
                rows = tree.query_ball_point(self.points[first:last], radius, workers=threads, return_sorted=True)
                flat = np.concatenate(rows).astype(np.int32)
                if len(flat) != indptr[last]-indptr[first]:
                    raise RuntimeError('neighbor count changed during CSR construction')
                columns[int(indptr[first]):int(indptr[last])] = cp.asarray(flat)
            ptr = cp.asarray(indptr, dtype=cp.int32)
            coordinates = cp.asarray(self.points)
            self.permutation = cp.arange(n)
        else:
            raise ValueError('graph_backend must be gpu or cpu')
        self.inverse_permutation = cp.argsort(self.permutation)
        nnz = len(columns)
        estimated_bytes = nnz*12+(n+1)*4
        values = cp.empty(nnz, dtype=cp.float64)
        kernel = cp.RawKernel(_FILL, 'fill_sparse')
        kernel((n,), (256,), (coordinates, ptr, columns, values,
                              model.kappa, model.amplitude, noise_variance, n))
        self.matrix = csr_matrix((values, columns, ptr), shape=(n, n))
        self.matrix.has_canonical_format = True
        cp.cuda.get_current_stream().synchronize()
        self.device_matrix_bytes = estimated_bytes+self.preconditioner_gpu.nbytes
        self.last_solve = None
        self.diagnostics = dict(cutoff=cutoff, radius=radius, nnz=nnz,
                                graph_backend=graph_backend,
                                mean_neighbors=float(counts.mean()), max_neighbors=int(counts.max()),
                                matrix_bytes=estimated_bytes, preconditioner_seconds=preconditioner_seconds,
                                graph_seconds=perf_counter()-start,
                                approximation='hard distance truncation; full-kernel audit required; PSD not guaranteed')

    def matmul(self, rhs):
        import cupy as cp
        b = cp.asarray(rhs, dtype=cp.float64)
        if b.ndim not in [1, 2] or b.shape[0] != len(self.points):
            raise ValueError('RHS must have shape (n,) or (n,k)')
        return (self.matrix@b[self.permutation])[self.inverse_permutation]

    def project(self, *args, **kwargs):
        result = super().project(*args, **kwargs)
        self.last_solve['bound_scope'] = 'diagnostic only; truncation error and positive definiteness require separate control'
        return result
