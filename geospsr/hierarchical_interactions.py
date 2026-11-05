"""Bounded-storage H² interaction actions, retaining every represented block."""
import numpy as np


_FILL = r'''
extern "C" __global__ void fill_streamed(
    const double* xyz, const int* ptr, const long long* tiles,
    const long long* meta, int nb, double kappa, double* values, int* cols,
    long long first) {
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
    long long address=ptr[i-first]+offset+t%nc;
    values[address]=(1+z)*exp(-z); cols[address]=(int)j;
}
'''


class StreamedInteractions:
    """Rebuild bounded CSR row tiles on each action; never truncate far blocks.

    The persistent hierarchy is unchanged. Trading repeated kernel evaluation
    for memory removes the global CSR entry-count and value-storage ceilings.
    """
    def __init__(self, coordinates, indptr, metadata, kappa, budget_bytes):
        import cupy as cp
        self.coordinates = cp.asarray(coordinates)
        self.kappa = float(kappa)
        self.size = len(indptr)-1
        if self.size >= 2**31 or budget_bytes < 12:
            raise ValueError('invalid streamed interaction dimensions or budget')
        self.kernel = cp.RawKernel(_FILL, 'fill_streamed')
        self.kernel.compile()
        self.chunks = []
        cap = min(int(budget_bytes)//12, 2**31-1)
        first = 0
        while first < self.size:
            last = min(self.size, int(np.searchsorted(indptr, indptr[first]+cap, side='right')-1))
            if last <= first:
                raise MemoryError('one interaction row exceeds the streaming budget')
            mask = (metadata[:, 0] < last) & (metadata[:, 0]+metadata[:, 2] > first)
            meta = metadata[mask].copy()
            end = np.minimum(meta[:, 0]+meta[:, 2], last)
            meta[:, 0] = np.maximum(meta[:, 0], first)
            meta[:, 2] = end-meta[:, 0]
            ptr = (indptr[first:last+1]-indptr[first]).astype(np.int32)
            tiles = np.r_[0, np.cumsum((meta[:, 2]*meta[:, 3]+255)//256)]
            if ptr[-1]:
                self.chunks.append((first, last, int(ptr[-1]), cp.asarray(ptr),
                                    cp.asarray(meta), cp.asarray(tiles), int(tiles[-1])))
            first = last
        self.nbytes = self.coordinates.nbytes+sum(sum(a.nbytes for a in c[3:6]) for c in self.chunks)
        max_nnz = max((c[2] for c in self.chunks), default=0)
        self.workspace_bytes = max_nnz*12
        self.values = cp.empty(max_nnz)
        self.columns = cp.empty(max_nnz, dtype=cp.int32)

    def __matmul__(self, work):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
        output = cp.zeros_like(work)
        for first, last, nnz, ptr, meta, tiles, blocks in self.chunks:
            values, columns = self.values[:nnz], self.columns[:nnz]
            self.kernel((blocks,), (256,), (self.coordinates, ptr, tiles, meta, len(meta),
                                           self.kappa, values, columns, first))
            matrix = csr_matrix((values, columns, ptr), shape=(last-first, self.size))
            matrix.has_canonical_format = True
            output[first:last] = matrix@work
            del matrix, values, columns
        return output
