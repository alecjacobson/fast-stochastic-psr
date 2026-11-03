"""Fused float32 analytic matrix construction for a strong dense baseline."""
import cupy as cp
import numpy as np


_rows=cp.RawKernel(r'''
extern "C" __global__ void covariance32(const float* q,const float* x,float* out,
    int nq,int nx,float a,float c0,float c1,float scale) {
    long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=(long long)nq*nx)return;
    int i=id/nx,j=id%nx;float value=1.f;
    for(int d=0;d<3;++d) {
        float t=fmodf(fabsf(q[3*i+d]-x[3*j+d]),6.283185307179586f)/6.283185307179586f;
        value*=scale*((c0+c1*t)*expf(-a*t)+(c0+c1*(1.f-t))*expf(-a*(1.f-t)));
    }
    out[id]=value;
}
''','covariance32')


def analytic_covariance_rows32(query,points,lengthscale=.01,variance=.1):
    if not np.isfinite(lengthscale) or lengthscale<=0 or not np.isfinite(variance) or variance<0:
        raise ValueError('invalid kernel parameters')
    q,x=[cp.ascontiguousarray(cp.asarray(v,dtype='f')) for v in (query,points)]
    if any(v.ndim!=2 or v.shape[1]!=3 for v in (q,x)):raise ValueError('expected (N,3) coordinates')
    a=np.sqrt(3)/lengthscale
    c0=2*lengthscale+2*np.sqrt(3)*np.exp(-a)/(-np.expm1(-a))
    c1=2*np.sqrt(3)
    norm=c0+(c0+c1)*np.exp(-a)
    out=cp.empty((len(q),len(x)),dtype='f')
    _rows(((out.size+255)//256,),(256,),
          (q,x,out,len(q),len(x),np.float32(a),np.float32(c0),np.float32(c1),np.float32(variance/norm)))
    return out
