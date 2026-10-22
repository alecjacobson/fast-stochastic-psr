"""Direct separable float64 reference for the finite core-plus-slabs model.

For overlap validation only: O(N M extent) work and N-by-M output. Frequencies
are summed independently of the shifted NUFFT plans or production coefficients.
"""
import cupy as cp
import numpy as np
from geospsr.spectral_blocks import axis_weights,one_high_axis_blocks

_source = r'''
extern "C" __global__ void block_rows(
    const float* q, const float* x, const double* rho, double* out,
    const int nq, const int nx, const int core, const int extent) {
    const long long id = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= (long long)nq * nx) return;
    const int i = id / nx, j = id % nx;
    double low[3], high[3];
    for (int d = 0; d < 3; ++d) {
        const double angle = (double)q[3*i+d] - (double)x[3*j+d];
        const double cosine = cos(angle);
        double previous = 1., current = cosine;
        low[d] = rho[0]; high[d] = 0.;
        for (int n = 1; n <= extent; ++n) {
            const double term = 2. * rho[n] * current;
            if (n <= core) low[d] += term; else high[d] += term;
            const double next = 2. * cosine * current - previous;
            previous = current; current = next;
        }
    }
    out[id] = low[0]*low[1]*low[2] + high[0]*low[1]*low[2]
            + low[0]*high[1]*low[2] + low[0]*low[1]*high[2];
}
'''
_rows = cp.RawKernel(_source,'block_rows')


def block_covariance_rows(query,points,core,extent,lengthscale=.01,variance=.1):
    one_high_axis_blocks(core,extent)
    q = cp.ascontiguousarray(cp.asarray(query,dtype='f'))
    x = cp.ascontiguousarray(cp.asarray(points,dtype='f'))
    rho = cp.asarray(axis_weights(np.arange(extent+1),lengthscale,variance))
    result = cp.empty((len(q),len(x)),dtype='d')
    _rows(((result.size+127)//128,),(128,),
          (q,x,rho,result,len(q),len(x),int(core),int(extent)))
    return result
