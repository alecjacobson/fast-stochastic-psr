"""Bounded-memory GPU radius-kernel control with the same AFN/cache contract."""
from time import perf_counter

import numpy as np

from .hierarchical import HierarchicalMaternSystem
from .hierarchical_gpu import GPUAFNPreconditioner, GPUHierarchicalSystem
from .nonperiodic_sparse import _RADIUS

_FILL = r'''
extern "C" __global__ void fill_local(
    const double* x, const long long* cells, const int* offsets,
    int ny, int nz, int nx, double radius2, int first, int last,
    const int* ptr, int* columns, double* values,
    double kappa, double amplitude, double noise) {
    int i=first+blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=last) return;
    long long cell=cells[i];
    int iz=cell%nz,iy=(cell/nz)%ny,ix=cell/(ny*nz);
    double a=x[3*i],b=x[3*i+1],c=x[3*i+2];
    int position=ptr[i-first];
    for(int u=max(0,ix-1);u<=min(nx-1,ix+1);u++)
    for(int v=max(0,iy-1);v<=min(ny-1,iy+1);v++)
    for(int w=max(0,iz-1);w<=min(nz-1,iz+1);w++) {
        int bin=(u*ny+v)*nz+w;
        for(int j=offsets[bin];j<offsets[bin+1];j++) {
            double dx=a-x[3*j],dy=b-x[3*j+1],dz=c-x[3*j+2];
            double d=dx*dx+dy*dy+dz*dz;
            if(d<=radius2) {
                double z=sqrt(d)*kappa;
                columns[position]=j;
                values[position]=amplitude*(1+z)*exp(-z)+(i==j?noise:0.);
                position++;
            }
        }
    }
}
'''


class GPUStreamedLocalSystem(GPUHierarchicalSystem):
    def __init__(self, model, points, noise_variance, cutoff=11., rank=8, neighbors=24,
                 threads=4, seed=12, budget_bytes=256*2**20,
                 preconditioner_backend='native', rhs_block_size=None):
        import cupy as cp
        if not np.isfinite(cutoff) or cutoff <= 0 or budget_bytes < 12:
            raise ValueError('positive cutoff and interaction budget required')
        start = perf_counter()
        self.native = HierarchicalMaternSystem(model, points, noise_variance,
             preconditioner_rank=rank, neighbors=neighbors, threads=threads, seed=seed,
             build_hierarchy=False, preconditioner_backend=preconditioner_backend)
        self.model, self.points = model, self.native.points
        self.noise_variance, self.rhs_block_size = noise_variance, rhs_block_size
        self.preconditioner_gpu = GPUAFNPreconditioner(self.native.preconditioner)
        preconditioner_seconds = perf_counter()-start
        start = perf_counter()
        self.radius = cutoff/model.kappa
        x = cp.asarray(self.points)
        bins = cp.floor((x-cp.min(x, axis=0))/self.radius).astype(cp.int64)
        self.nx, self.ny, self.nz = map(int, cp.asnumpy(bins.max(axis=0)+1))
        cells = self.nx*self.ny*self.nz
        if cells > max(1000000, 64*len(x)) or len(x) >= 2**31:
            raise ValueError('uniform-grid or point-index capacity exceeded')
        ids = (bins[:, 0]*self.ny+bins[:, 1])*self.nz+bins[:, 2]
        self.permutation = cp.argsort(ids)
        self.inverse_permutation = cp.argsort(self.permutation)
        self.coordinates, self.ids = x[self.permutation], ids[self.permutation]
        self.offsets = cp.r_[cp.zeros(1, dtype=cp.int32), cp.cumsum(cp.bincount(self.ids, minlength=cells), dtype=cp.int32)]
        counts = cp.empty(len(x), dtype=cp.int32)
        empty = cp.empty(0, dtype=cp.int32)
        count = cp.RawKernel(_RADIUS, 'radius_graph')
        count(((len(x)+127)//128,), (128,), (self.coordinates,self.ids,self.offsets,
            self.ny,self.nz,self.nx,self.radius**2,len(x),counts,empty,empty,0))
        host_counts = cp.asnumpy(counts)
        prefix = np.r_[0, np.cumsum(host_counts, dtype=np.int64)]
        cap = min(int(budget_bytes)//12, 2**31-1)
        self.chunks, first = [], 0
        while first < len(x):
            last = min(len(x), int(np.searchsorted(prefix, prefix[first]+cap, side='right')-1))
            if last <= first:
                raise MemoryError('one radius row exceeds the streaming budget')
            ptr = cp.asarray(prefix[first:last+1]-prefix[first], dtype=cp.int32)
            self.chunks.append((first, last, int(prefix[last]-prefix[first]), ptr))
            first = last
        max_nnz = max(c[2] for c in self.chunks)
        self.values, self.columns = cp.empty(max_nnz), cp.empty(max_nnz, dtype=cp.int32)
        self.kernel = cp.RawKernel(_FILL, 'fill_local'); self.kernel.compile()
        self.interaction_workspace_bytes = max_nnz*12
        self.device_matrix_bytes = sum(a.nbytes for a in [self.coordinates,self.ids,self.offsets,
              self.permutation,self.inverse_permutation])+sum(c[3].nbytes for c in self.chunks)+self.preconditioner_gpu.nbytes
        self.last_solve = None
        cp.cuda.get_current_stream().synchronize()
        self.diagnostics = dict(cutoff=cutoff, radius=self.radius, nnz=int(prefix[-1]),
            graph_backend='streamed-gpu', mean_neighbors=float(host_counts.mean()), max_neighbors=int(host_counts.max()),
            cached_graph_bytes=int(prefix[-1])*12+(len(x)+1)*4, chunks=len(self.chunks),
            workspace_bytes=self.interaction_workspace_bytes, preconditioner_seconds=preconditioner_seconds,
            graph_seconds=perf_counter()-start, approximation='hard distance truncation; full-kernel audit required; PSD not guaranteed')

    def matmul(self, rhs):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
        b = cp.asarray(rhs, dtype=cp.float64)
        if b.ndim not in [1,2] or b.shape[0] != len(self.points):
            raise ValueError('RHS must have shape (n,) or (n,k)')
        ordered = b[self.permutation]
        result = cp.empty_like(ordered)
        for first,last,nnz,ptr in self.chunks:
            values, columns = self.values[:nnz], self.columns[:nnz]
            self.kernel(((last-first+127)//128,), (128,), (self.coordinates,self.ids,self.offsets,
                self.ny,self.nz,self.nx,self.radius**2,first,last,ptr,columns,values,
                self.model.kappa,self.model.amplitude,self.noise_variance))
            matrix = csr_matrix((values,columns,ptr), shape=(last-first,len(self.points)))
            matrix.has_canonical_format = True
            result[first:last] = matrix@ordered
        return result[self.inverse_permutation]

    def project(self, *args, **kwargs):
        result = super().project(*args, **kwargs)
        self.last_solve['bound_scope'] = 'diagnostic only; truncation error and positive definiteness require separate control'
        return result
