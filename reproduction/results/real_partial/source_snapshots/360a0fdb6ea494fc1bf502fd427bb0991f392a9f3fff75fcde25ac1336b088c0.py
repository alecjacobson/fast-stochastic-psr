"""Collapse a regional quadratic variance cache into scalar polynomials.

A degree-(p-1) cross interpolation produces degree at most 2(p-1) variance
correction in each coordinate. Resampling at 2p-1 Chebyshev nodes and a DCT
represents that polynomial, removing the O(p^6) online quadratic form.
"""
import numpy as np
from scipy.fft import dctn

from .nonperiodic import RegionalVarianceCache, _points


def _coefficients(values):
    p = values.shape[0]
    c = dctn(values, type=2)/p**3
    c[0] *= .5
    c[:, 0] *= .5
    c[:, :, 0] *= .5
    return c


_EVALUATE = r'''
#define P DEGREE
__device__ double evaluate(const double* c, double x, double y, double z) {
    double bx1=0,bx2=0,result=0;
    for(int i=P-1;i>=0;--i) {
        double by1=0,by2=0,cy=0;
        for(int j=P-1;j>=0;--j) {
            const double* row=c+(i*P+j)*P;
            double bz1=0,bz2=0;
            for(int k=P-1;k>=1;--k) {double t=2*z*bz1-bz2+row[k];bz2=bz1;bz1=t;}
            double cz=z*bz1-bz2+row[0];
            if(j==0) cy=y*by1-by2+cz;
            else {double t=2*y*by1-by2+cz;by2=by1;by1=t;}
        }
        if(i==0) result=x*bx1-bx2+cy;
        else {double t=2*x*bx1-bx2+cy;bx2=bx1;bx1=t;}
    }
    return result;
}
extern "C" __global__ void moments(const double* q, const double* box,
    const double* mean, const double* correction, double prior, long long n,
    double* out_mean, double* out_var) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    double x=(q[3*i]-box[0])/box[3],y=(q[3*i+1]-box[1])/box[4],z=(q[3*i+2]-box[2])/box[5];
    out_mean[i]=evaluate(mean,x,y,z);
    out_var[i]=prior-evaluate(correction,x,y,z);
}
'''


class PolynomialMomentField:
    def __init__(self, cache):
        self.center = cache.center.copy()
        self.halfwidth = cache.halfwidth.copy()
        self.prior_variance = cache.prior_variance
        p = cache.order
        self.order = 2*p-1
        nodes = RegionalVarianceCache.__new__(RegionalVarianceCache)
        nodes._initialize_box(self.center, self.halfwidth, self.order)
        phi = cache.basis(nodes.nodes)
        correction = np.sum((phi @ cache.correction)*phi, axis=1)
        self.correction = _coefficients(correction.reshape((self.order,)*3))
        self.mean = np.zeros_like(self.correction)
        self.mean[:p, :p, :p] = _coefficients(cache.mean_coefficients.reshape((p,)*3))
        self._gpu = None

    def predict(self, queries):
        x = (_points(queries)-self.center)/self.halfwidth
        if np.any(np.abs(x) > 1+1e-12):
            raise ValueError('queries must lie inside the field box')
        evaluate = np.polynomial.chebyshev.chebval3d
        return (evaluate(*x.T, self.mean),
                self.prior_variance-evaluate(*x.T, self.correction))

    def prepare_gpu(self):
        import cupy as cp
        kernel = cp.RawKernel(_EVALUATE.replace('DEGREE', str(self.order)), 'moments')
        kernel.compile()
        self._gpu = (kernel, cp.asarray(np.r_[self.center, self.halfwidth]),
                     cp.asarray(self.mean), cp.asarray(self.correction))
        cp.cuda.get_current_stream().synchronize()

    def predict_gpu(self, queries):
        import cupy as cp
        if self._gpu is None:
            self.prepare_gpu()
        q = cp.asarray(queries, dtype=cp.float64, order='C')
        if q.ndim != 2 or q.shape[1] != 3 or not cp.isfinite(q).all():
            raise ValueError('queries must be a finite (m,3) array')
        kernel, box, coefficients, correction = self._gpu
        if cp.any(cp.abs((q-box[:3])/box[3:]) > 1+1e-12):
            raise ValueError('queries must lie inside the field box')
        mean, variance = cp.empty(len(q)), cp.empty(len(q))
        if len(q):
            kernel(((len(q)+127)//128,), (128,),
                   (q, box, coefficients, correction, self.prior_variance, len(q), mean, variance))
        return mean, variance
