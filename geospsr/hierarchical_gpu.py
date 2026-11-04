"""GPU H² actions with cached interactions and nested sparse transfers.

Exports the pinned CPU hierarchy and small/sparse AFN factors, never the dense
N-by-N matrix. Construction and CPU-to-GPU transfer costs belong to preparation.
"""
import numpy as np
from scipy.sparse import csr_matrix

from .hierarchical import batched_pcg


_FILL = r'''
extern "C" __global__ void fill(
    const double* xyz, const long long* ptr, const long long* tiles,
    const long long* meta, int nb, double kappa, double* values, int* cols) {
    __shared__ long long row0, col0, nr, nc, offset, tile0;
    if (threadIdx.x == 0) {
        int lo=0, hi=nb;
        while(lo+1<hi) {int mid=(lo+hi)/2; if(tiles[mid] <= blockIdx.x) lo=mid; else hi=mid;}
        row0=meta[5*lo]; col0=meta[5*lo+1]; nr=meta[5*lo+2];
        nc=meta[5*lo+3]; offset=meta[5*lo+4]; tile0=tiles[lo];
    }
    __syncthreads();
    long long t=((long long)blockIdx.x-tile0)*blockDim.x+threadIdx.x;
    if(t>=nr*nc) return;
    long long i=row0+t/nc, j=col0+t%nc;
    double d=0;
    for(int k=0;k<3;k++) {double v=xyz[3*i+k]-xyz[3*j+k];d+=v*v;}
    double z=sqrt(d)*kappa;
    long long address=ptr[i]+offset+t%nc;
    values[address]=(1+z)*exp(-z);
    cols[address]=(int)j;
}
'''


class GPUAFNPreconditioner:
    def __init__(self, native):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr
        from scipy.linalg import cho_solve
        snapshot = native.snapshot()
        factor = snapshot['L11_transpose'].T
        self.coarse_inverse = cp.asarray(cho_solve((factor, True), np.eye(len(factor))))
        self.coupling = cp.asarray(snapshot['K12_transpose'].T)
        permutation = snapshot['permutation'].ravel()
        self.permutation = cp.asarray(permutation)
        self.inverse_permutation = cp.asarray(np.argsort(permutation))
        n2 = len(permutation)-len(factor)
        self.fsai = gpu_csr((cp.asarray(snapshot['G_data'].ravel()),
                             cp.asarray(snapshot['G_columns'].ravel()),
                             cp.asarray(snapshot['G_indptr'].ravel())), shape=(n2, n2))
        self.transpose = self.fsai.T.tocsr()
        self.nbytes = self.coarse_inverse.nbytes+self.coupling.nbytes
        self.nbytes += sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in [self.fsai, self.transpose])

    def __call__(self, rhs):
        import cupy as cp
        b = cp.asarray(rhs, dtype=cp.float64)
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        permuted = b[self.permutation]
        rank = len(self.coarse_inverse)
        b1, b2 = permuted[:rank], permuted[rank:]
        t1 = self.coarse_inverse @ b1
        t2 = b2-self.coupling.T @ t1
        c2 = self.transpose @ (self.fsai @ t2)
        c1 = self.coarse_inverse @ (b1-self.coupling @ c2)
        result = cp.concatenate((c1, c2))[self.inverse_permutation]
        return result[:, 0] if vector else result


class GPUHierarchicalSystem:
    def __init__(self, native_system):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr
        self.native = native_system
        self.model = native_system.model
        self.points = native_system.points
        self.noise_variance = native_system.noise_variance
        snapshot = native_system.operator.snapshot()
        permutation = snapshot['permutation'].ravel()
        self.permutation = cp.asarray(permutation)
        self.inverse_permutation = cp.asarray(np.argsort(permutation))
        x = self.points[permutation]
        nodes = snapshot['nodes']
        levels = snapshot['levels'].ravel()
        clusters = snapshot['clusters']
        min_level = snapshot['min_far_level']
        ranks = np.array([node['basis_transpose'].shape[1] for node in nodes])
        offsets = len(x)+np.r_[0, np.cumsum(ranks)]
        self.augmented_size = int(offsets[-1])
        self.ranks = ranks
        self.max_rank = int(ranks.max(initial=0))
        coordinates = np.concatenate([x]+[x[node['skeleton'].ravel()] for node in nodes], axis=0)
        if len(coordinates) != self.augmented_size:
            raise ValueError('skeleton ranks do not match the exported basis sizes')
        # One sparse transfer per level, for the upward and downward passes.
        transfers = []
        for level in range(int(levels.max()), min_level-1, -1):
            indptr = np.zeros(self.augmented_size+1, dtype=np.int64)
            values, columns = [], []
            for i in np.flatnonzero(levels == level):
                if ranks[i] == 0:
                    continue
                node = nodes[i]
                children = node['children'].ravel()
                if len(children):
                    source = np.concatenate([np.arange(offsets[j], offsets[j+1]) for j in children])
                else:
                    source = np.arange(clusters[i, 0], clusters[i, 1]+1)
                basis = node['basis_transpose'].T
                if basis.shape != (ranks[i], len(source)):
                    raise ValueError('invalid exported nested basis')
                indptr[offsets[i]+1:offsets[i+1]+1] = len(source)
                values.append((int(offsets[i]), basis.ravel()))
                columns.append((int(offsets[i]), np.tile(source, ranks[i])))
            if not values:
                continue
            indptr = np.cumsum(indptr)
            data = np.concatenate([v for _, v in sorted(values)])
            cols = np.concatenate([v for _, v in sorted(columns)]).astype(np.int32)
            transfer = gpu_csr(csr_matrix((data, cols, indptr),
                                         shape=(self.augmented_size, self.augmented_size)))
            transfers.append((transfer, transfer.T.tocsr()))
        self.transfers = transfers
        # Interaction blocks target either leaf points or node skeletons.
        groups = {}

        def add(row, nr, col, nc):
            if nr and nc:
                groups.setdefault((int(row), int(nr)), []).append((int(col), int(nc)))

        for i, node in enumerate(nodes):
            ni = int(clusters[i, 1]-clusters[i, 0]+1)
            for j in node['near'].ravel():
                add(clusters[i, 0], ni, clusters[j, 0], clusters[j, 1]-clusters[j, 0]+1)
            for j in node['far'].ravel():
                if levels[i] == levels[j]:
                    add(offsets[i], ranks[i], offsets[j], ranks[j])
                elif levels[i] > levels[j]:
                    add(offsets[i], ranks[i], clusters[j, 0], clusters[j, 1]-clusters[j, 0]+1)
                else:
                    add(clusters[i, 0], ni, offsets[j], ranks[j])
        indptr = np.zeros(self.augmented_size+1, dtype=np.int64)
        metadata = []
        for (row, nr), sources in sorted(groups.items()):
            sources = sorted(sources)
            # Disjoint source blocks make every row sorted and duplicate-free.
            # Establish this before allocating GPU entries: otherwise CuPy's
            # first matmul may convert billions of entries to COO and sort them.
            end = -1
            for col, nc in sources:
                if col < end:
                    raise ValueError('overlapping source blocks in the H2 interaction graph')
                end = col+nc
            width = sum(nc for _, nc in sources)
            if np.any(indptr[row+1:row+nr+1]):
                raise ValueError('overlapping target groups in the H2 interaction graph')
            indptr[row+1:row+nr+1] = width
            offset = 0
            for col, nc in sources:
                metadata.append((row, col, nr, nc, offset))
                offset += nc
        indptr = np.cumsum(indptr)
        self.interaction_nnz = int(indptr[-1])
        if self.augmented_size >= 2**31 or self.interaction_nnz >= 2**31:
            raise ValueError('this GPU prototype requires 32-bit CSR indexing')
        metadata = np.asarray(metadata, dtype=np.int64)
        tiles = np.r_[0, np.cumsum((metadata[:, 2]*metadata[:, 3]+255)//256)]
        values = cp.empty(self.interaction_nnz, dtype=cp.float64)
        columns = cp.empty(self.interaction_nnz, dtype=cp.int32)
        kernel = cp.RawKernel(_FILL, 'fill')
        kernel((int(tiles[-1]),), (256,),
               (cp.asarray(coordinates), cp.asarray(indptr), cp.asarray(tiles),
                cp.asarray(metadata), len(metadata), self.model.kappa, values, columns))
        self.interactions = gpu_csr((values, columns, cp.asarray(indptr, dtype=cp.int32)),
                                    shape=(self.augmented_size, self.augmented_size))
        self.interactions.has_canonical_format = True
        self.preconditioner_gpu = GPUAFNPreconditioner(native_system.preconditioner)
        matrices = [self.interactions]+[t for pair in self.transfers for t in pair]
        self.device_matrix_bytes = sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in matrices)
        self.device_matrix_bytes += self.preconditioner_gpu.nbytes
        self.last_solve = None
        cp.cuda.get_current_stream().synchronize()

    def matmul(self, rhs):
        import cupy as cp
        b = cp.asarray(rhs, dtype=cp.float64)
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        if b.ndim != 2 or b.shape[0] != len(self.points):
            raise ValueError('RHS must have shape (n,) or (n,k)')
        if not b.shape[1]:
            return cp.empty_like(b)
        work = cp.zeros((self.augmented_size, b.shape[1]), dtype=cp.float64)
        work[:len(b)] = b[self.permutation]
        for transfer, _ in self.transfers:
            work += transfer @ work
        result = self.interactions @ work
        for _, transpose in reversed(self.transfers):
            result += transpose @ result
        output = self.model.amplitude*result[:len(b)][self.inverse_permutation] + self.noise_variance*b
        return output[:, 0] if vector else output

    def precondition(self, rhs):
        return self.preconditioner_gpu(rhs)

    def solve(self, rhs, tolerance=1e-6, max_iterations=500):
        import cupy as cp
        x, info = batched_pcg(self.matmul, self.precondition, rhs, tolerance=tolerance,
                             max_iterations=max_iterations, xp=cp)
        self.last_solve = info
        if not info['converged']:
            raise RuntimeError(f'GPU PCG did not converge: {info}')
        return cp.asnumpy(x)

    def project(self, basis, vectors, variance_iterations=5, mean_tolerance=1e-6):
        """Use residual-corrected quadratic forms for approximate variance solves.

        For Z≈A^-1 U and R=U-AZ, G=UᵀZ+ZᵀR differs from UᵀA^-1U
        by -RᵀA^-1R. The mean is solved independently. This identity is with
        respect to the represented H2 operator; its kernel approximation error
        must still pass an independent full-kernel accuracy check.
        """
        import cupy as cp
        u = cp.asarray(basis)
        alpha, mean_info = batched_pcg(self.matmul, self.precondition, vectors,
                                       tolerance=mean_tolerance, xp=cp)
        if not mean_info['converged']:
            raise RuntimeError('mean conditioning did not converge')
        z, variance_info, residual = batched_pcg(
            self.matmul, self.precondition, u, tolerance=1e-10,
            max_iterations=variance_iterations, xp=cp, return_residual=True)
        gram = u.T@z+z.T@residual
        gram = .5*(gram+gram.T)
        self.last_solve = dict(mean=mean_info, variance=variance_info,
                              variance_residual_corrected=True,
                              noise_scaled_residual_gram_trace=float(cp.sum(residual*residual).item()/self.noise_variance),
                              bound_scope='diagnostic only; H2 error and its smallest eigenvalue need separate control')
        return cp.asnumpy(gram), cp.asnumpy(u.T@alpha)
