"""Float64 all-pairs mean evaluation without an N-by-query intermediate.

Visualization helper only: no truncation, interpolation or model change.
"""
import numpy as np
from geospsr.nonperiodic import _CROSS, _H

_SOURCE = r'''
extern "C" __global__ void fused_mean(const double* q, const double* x,
    const double* a, const double* c, const double* w, double* out,
    int n, int nc, double kappa, double amplitude) {
    int j=blockIdx.x, lane=threadIdx.x;
    double qx=q[3*j],qy=q[3*j+1],qz=q[3*j+2],sum=0;
    const double radial_coeff[] = {CROSS};
    const double scalar_coeff[] = {SCALAR};
    for(int i=lane;i<n;i+=blockDim.x) {
        double dx=qx-x[3*i],dy=qy-x[3*i+1],dz=qz-x[3*i+2];
        double z=kappa*sqrt(dx*dx+dy*dy+dz*dz),radial;
        if(z>=50) radial=8/(z*z*z); // Exponential subtraction rounds to 8 in float64.
        else if(z<.5) {
            radial=radial_coeff[CROSS_DEGREE];
            for(int k=CROSS_DEGREE-1;k>=0;k--) radial=radial*z+radial_coeff[k];
        } else radial=(8-exp(-z)*(8+8*z+4*z*z+z*z*z))/(z*z*z);
        sum+=amplitude*radial*(dx*a[3*i]+dy*a[3*i+1]+dz*a[3*i+2]);
    }
    for(int i=lane;i<nc;i+=blockDim.x) {
        double dx=qx-c[3*i],dy=qy-c[3*i+1],dz=qz-c[3*i+2];
        double z=kappa*sqrt(dx*dx+dy*dy+dz*dz),h;
        if(z>=50) h=8/z; // Same float64 result, not a spatial cutoff of the kernel.
        else if(z<.5) {
            h=scalar_coeff[SCALAR_DEGREE];
            for(int k=SCALAR_DEGREE-1;k>=0;k--) h=h*z+scalar_coeff[k];
        } else h=(8-exp(-z)*(8+5*z+z*z))/z;
        sum+=amplitude/(kappa*kappa)*h*w[i];
    }
    __shared__ double work[256];work[lane]=sum;__syncthreads();
    for(int size=128;size>0;size/=2) {
        if(lane<size) work[lane]+=work[lane+size];
        __syncthreads();
    }
    if(lane==0)out[j]=work[0];
}
'''


class FusedMean:
    def __init__(self, model, x, contacts, coefficients, weights):
        import cupy as cp
        self.model=model
        self.x,self.contacts,self.coefficients,self.weights=[cp.ascontiguousarray(cp.asarray(v,dtype=cp.float64))
            for v in [x,contacts,coefficients,weights]]
        source=_SOURCE.replace('CROSS_DEGREE',str(len(_CROSS)-1)).replace('SCALAR_DEGREE',str(len(_H)-1))
        source=source.replace('CROSS',','.join(format(v,'.17g') for v in _CROSS))
        source=source.replace('SCALAR',','.join(format(v,'.17g') for v in _H))
        self.kernel=cp.RawKernel(source,'fused_mean')

    def __call__(self, q, batch=8192):
        import cupy as cp
        result=np.empty(len(q))
        for first in range(0,len(q),batch):
            qq=cp.ascontiguousarray(cp.asarray(q[first:first+batch],dtype=cp.float64));out=cp.empty(len(qq),dtype=cp.float64)
            if len(qq):
                self.kernel((len(qq),),(256,),(qq,self.x,self.coefficients,self.contacts,self.weights,out,
                    len(self.x),len(self.contacts),self.model.kappa,self.model.amplitude))
            result[first:first+len(qq)]=cp.asnumpy(out)
        return result
