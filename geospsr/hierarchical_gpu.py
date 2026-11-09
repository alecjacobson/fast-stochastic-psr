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
    def __init__(self, native, dtype='float64'):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr
        from scipy.linalg import cho_solve
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('preconditioner dtype must be float32 or float64')
        snapshot = native.snapshot()
        factor = snapshot['L11_transpose'].T
        self.coarse_inverse = cp.asarray(cho_solve((factor, True), np.eye(len(factor))), dtype=self.dtype)
        self.coupling = cp.asarray(snapshot['K12_transpose'].T, dtype=self.dtype)
        permutation = snapshot['permutation'].ravel()
        self.permutation = cp.asarray(permutation)
        self.inverse_permutation = cp.asarray(np.argsort(permutation))
        n2 = len(permutation)-len(factor)
        self.fsai = gpu_csr((cp.asarray(snapshot['G_data'].ravel(), dtype=self.dtype),
                             cp.asarray(snapshot['G_columns'].ravel()),
                             cp.asarray(snapshot['G_indptr'].ravel())), shape=(n2, n2))
        self.transpose = self.fsai.T.tocsr()
        self.nbytes = self.coarse_inverse.nbytes+self.coupling.nbytes
        self.nbytes += sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in [self.fsai, self.transpose])

    def __call__(self, rhs):
        import cupy as cp
        b = cp.asarray(rhs, dtype=self.dtype)
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
        result = result.astype(cp.float64, copy=False)
        return result[:, 0] if vector else result


class GPUHierarchicalSystem:
    """GPU hierarchy with optional float32 actions and float64 refinement.

    ``dtype='float32'`` retains a streamed float64 hierarchy action for solve
    refinement and corrected variance. This controls arithmetic error within
    the hierarchy; its approximation to the full kernel needs separate checks.
    """
    rhs_block_size = None
    mixed_actions = False

    def __init__(self, native_system, interaction_budget_bytes=None, rhs_block_size=None,
                 release_native=False, dtype='float64'):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as gpu_csr
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('hierarchy dtype must be float32 or float64')
        self.mixed_actions = self.dtype == np.float32
        self.native = native_system
        self.model = native_system.model
        self.points = native_system.points
        self.noise_variance = native_system.noise_variance
        self.rhs_block_size = rhs_block_size
        snapshot = native_system.operator.snapshot()
        if release_native:
            # snapshot owns its arrays; the GPU route no longer needs CPU actions.
            native_system.operator = None
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
        self.precise_transfers = transfers
        self.transfers = ([(a.astype(self.dtype), b.astype(self.dtype)) for a, b in transfers]
                          if self.dtype == np.float32 else transfers)
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
        if self.augmented_size >= 2**31 or (interaction_budget_bytes is None and self.interaction_nnz >= 2**31):
            raise ValueError('this GPU prototype requires 32-bit CSR indexing')
        metadata = np.asarray(metadata, dtype=np.int64)
        tiles = np.r_[0, np.cumsum((metadata[:, 2]*metadata[:, 3]+255)//256)]
        if interaction_budget_bytes is None:
            values = cp.empty(self.interaction_nnz, dtype=self.dtype)
            columns = cp.empty(self.interaction_nnz, dtype=cp.int32)
            typ = 'float' if self.dtype == np.float32 else 'double'
            kernel = cp.RawKernel(_FILL.replace('double* values', typ+'* values'), 'fill')
            kernel((int(tiles[-1]),), (256,),
                   (cp.asarray(coordinates), cp.asarray(indptr), cp.asarray(tiles),
                    cp.asarray(metadata), len(metadata), self.model.kappa, values, columns))
            self.interactions = gpu_csr((values, columns, cp.asarray(indptr, dtype=cp.int32)),
                                        shape=(self.augmented_size, self.augmented_size))
            self.interactions.has_canonical_format = True
            interaction_bytes = values.nbytes+columns.nbytes+self.interactions.indptr.nbytes
            self.interaction_workspace_bytes = 0
        else:
            from .hierarchical_interactions import StreamedInteractions
            self.interactions = StreamedInteractions(coordinates, indptr, metadata,
                                                     self.model.kappa, interaction_budget_bytes, dtype=self.dtype)
            interaction_bytes = self.interactions.nbytes
            self.interaction_workspace_bytes = self.interactions.workspace_bytes
        self.reference_matrix_bytes = self.reference_workspace_bytes = 0
        self.precise_interactions = self.interactions
        if self.dtype == np.float32:
            # An independent float64 action of the same exported hierarchy is
            # used for refinement and corrected variance. Never duplicate a
            # full cached interaction matrix just to verify reduced precision.
            from .hierarchical_interactions import StreamedInteractions
            self.precise_interactions = StreamedInteractions(coordinates, indptr, metadata,
                self.model.kappa, interaction_budget_bytes or 256*2**20)
            matrices = [m for pair in self.precise_transfers for m in pair]
            self.reference_matrix_bytes = self.precise_interactions.nbytes + sum(
                m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in matrices)
            self.reference_workspace_bytes = self.precise_interactions.workspace_bytes
        self.preconditioner_gpu = GPUAFNPreconditioner(native_system.preconditioner, dtype=self.dtype)
        if release_native:
            native_system.preconditioner = None
        matrices = [t for pair in self.transfers for t in pair]
        self.device_matrix_bytes = interaction_bytes+sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for m in matrices)
        self.device_matrix_bytes += self.preconditioner_gpu.nbytes
        self.device_matrix_bytes += self.reference_matrix_bytes
        self.last_solve = None
        cp.cuda.get_current_stream().synchronize()

    def matmul(self, rhs):
        return self._matmul(rhs, self.dtype, self.transfers, self.interactions)

    def accurate_matmul(self, rhs):
        """Float64 action of the represented hierarchy, not the full kernel.

        Hierarchy approximation error still needs a separate full-kernel check.
        """
        if not hasattr(self, 'precise_transfers'):
            # Existing float64 locality subclasses override matmul and share
            # the solve helpers without constructing a hierarchical snapshot.
            return self.matmul(rhs)
        return self._matmul(rhs, np.dtype('float64'), self.precise_transfers, self.precise_interactions)

    def _matmul(self, rhs, dtype, transfers, interactions):
        import cupy as cp
        b = cp.asarray(rhs, dtype=dtype)
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        if b.ndim != 2 or b.shape[0] != len(self.points):
            raise ValueError('RHS must have shape (n,) or (n,k)')
        if not b.shape[1]:
            return cp.empty(b.shape, dtype=cp.float64)
        work = cp.zeros((self.augmented_size, b.shape[1]), dtype=dtype)
        work[:len(b)] = b[self.permutation]
        for transfer, _ in transfers:
            work += transfer @ work
        result = interactions @ work
        for _, transpose in reversed(transfers):
            result += transpose @ result
        output = self.model.amplitude*result[:len(b)][self.inverse_permutation] + self.noise_variance*b
        output = output.astype(cp.float64, copy=False)
        return output[:, 0] if vector else output

    def precondition(self, rhs):
        return self.preconditioner_gpu(rhs)

    def _solve_block(self, rhs, tolerance, max_iterations):
        import cupy as cp
        if not 0 < tolerance < 1 or max_iterations < 1:
            raise ValueError('invalid PCG stopping parameters')
        if not self.mixed_actions:
            return batched_pcg(self.matmul, self.precondition, rhs, tolerance=tolerance,
                               max_iterations=max_iterations, xp=cp)
        b = cp.asarray(rhs, dtype=cp.float64)
        x, residual = cp.zeros_like(b), b.copy()
        norm = cp.linalg.norm(b, axis=0)
        stages = []; used = 0
        for _ in range(5):
            delta, info = batched_pcg(self.matmul, self.precondition, residual,
                tolerance=1e-3, max_iterations=min(max_iterations-used, 100), xp=cp)
            x += delta
            residual = b-self.accurate_matmul(x)
            relative = cp.linalg.norm(residual, axis=0)/cp.where(norm > 0, norm, 1.)
            stages.append(info)
            used += info['max_iterations']
            if bool(cp.all(relative <= tolerance)) or used >= max_iterations:
                break
        counts = np.sum([s['iterations'] for s in stages], axis=0)
        info = dict(converged=bool(cp.all(relative <= tolerance)),
            iterations=counts.tolist(), max_iterations=int(counts.max(initial=0)),
            relative_residuals=np.atleast_1d(cp.asnumpy(relative)).tolist(), tolerance=tolerance,
            refinement_stages=stages, residual_operator_dtype='float64', action_dtype='float32')
        return x, info

    def variance_solve(self, rhs, iterations):
        """Fixed work solve with a fresh float64 hierarchy residual."""
        import cupy as cp
        z, info, residual = batched_pcg(self.matmul, self.precondition, rhs,
            tolerance=1e-10, max_iterations=iterations, xp=cp, return_residual=True)
        if self.mixed_actions:
            b = cp.asarray(rhs, dtype=cp.float64)
            residual = b-self.accurate_matmul(z)
            norm = cp.linalg.norm(b, axis=0)
            relative = cp.linalg.norm(residual, axis=0)/cp.where(norm > 0, norm, 1.)
            info['low_precision_relative_residuals'] = info['relative_residuals']
            info['relative_residuals'] = np.atleast_1d(cp.asnumpy(relative)).tolist()
            info['converged'] = bool(cp.all(relative <= info['tolerance']))
            info['residual_operator_dtype'] = 'float64'
        return z, info, residual

    def solve(self, rhs, tolerance=1e-6, max_iterations=500):
        import cupy as cp
        if self.rhs_block_size and np.ndim(rhs) == 2 and rhs.shape[1] > self.rhs_block_size:
            outputs, infos = [], []
            for first in range(0, rhs.shape[1], self.rhs_block_size):
                x, info = self._solve_block(rhs[:, first:first+self.rhs_block_size], tolerance, max_iterations)
                outputs.append(cp.asnumpy(x)); infos.append(info)
            self.last_solve = dict(converged=all(i['converged'] for i in infos),
                iterations=sum([i['iterations'] for i in infos], []),
                relative_residuals=sum([i['relative_residuals'] for i in infos], []),
                max_iterations=max(i['max_iterations'] for i in infos), tolerance=tolerance)
            if self.mixed_actions:
                self.last_solve['blocks'] = infos
            if not self.last_solve['converged']:
                raise RuntimeError('blocked GPU PCG did not converge')
            return np.column_stack(outputs)
        x, info = self._solve_block(rhs, tolerance, max_iterations)
        self.last_solve = info
        if not info['converged']:
            raise RuntimeError(f'GPU PCG did not converge: {info}')
        return cp.asnumpy(x)

    def project(self, basis, vectors, variance_iterations=5, mean_tolerance=1e-6, progress=None):
        """Use residual-corrected quadratic forms for approximate variance solves.

        For Z≈A^-1 U and R=U-AZ, G=UᵀZ+ZᵀR differs from UᵀA^-1U
        by -RᵀA^-1R. The mean is solved independently. This identity is with
        respect to the represented H2 operator; its kernel approximation error
        must still pass an independent full-kernel accuracy check.
        An optional progress callback receives completed mean/variance stages.
        """
        import cupy as cp
        u = cp.asarray(basis, dtype=cp.float64)
        alpha, mean_info = self._solve_block(vectors, mean_tolerance, 500)
        if not mean_info['converged']:
            raise RuntimeError('mean conditioning did not converge')
        if progress:
            progress(dict(phase='mean complete', iterations=mean_info['iterations']))
        if self.rhs_block_size and u.shape[1] > self.rhs_block_size:
            z, residual = cp.empty_like(u), cp.empty_like(u)
            infos = []
            for first in range(0, u.shape[1], self.rhs_block_size):
                stop = min(first+self.rhs_block_size, u.shape[1])
                z[:, first:stop], info, residual[:, first:stop] = self.variance_solve(u[:, first:stop], variance_iterations)
                infos.append(info)
                if progress:
                    progress(dict(phase='variance block complete', first=first, stop=stop,
                                  iterations=info['max_iterations']))
            variance_info = dict(converged=all(i['converged'] for i in infos),
                                 iterations=sum([i['iterations'] for i in infos], []),
                                 relative_residuals=sum([i['relative_residuals'] for i in infos], []),
                                 max_iterations=max(i['max_iterations'] for i in infos), tolerance=1e-10,
                                 rhs_block_size=self.rhs_block_size)
        else:
            z, variance_info, residual = self.variance_solve(u, variance_iterations)
            if progress:
                progress(dict(phase='variance complete', iterations=variance_info['max_iterations']))
        gram = u.T@z+z.T@residual
        gram = .5*(gram+gram.T)
        self.last_solve = dict(mean=mean_info, variance=variance_info,
                              variance_residual_corrected=True,
                              noise_scaled_residual_gram_trace=float(cp.sum(residual*residual).item()/self.noise_variance),
                              bound_scope='diagnostic only; H2 error and its smallest eigenvalue need separate control')
        return cp.asnumpy(gram), cp.asnumpy(u.T@alpha)
