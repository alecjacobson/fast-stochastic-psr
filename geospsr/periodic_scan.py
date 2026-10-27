"""Bounded warp scans for the full analytic periodic range action.

Retains the serial planner and direct blocks. Each warp scans 32 z entries for
one feature, and sixteen warps contract all features within a block. Long groups
still traverse tiles serially; this is not yet a hierarchical inter-tile scan.
"""
import cupy as cp
from geospsr.periodic_range import PeriodicRangeKernel,_KERNEL,_direct

_SOURCE = _KERNEL.split('extern "C" __global__ void ranges')[0] + r'''
extern "C" __global__ void scan_ranges(const double* x, const int* ids,
    const unsigned char* labels, const long long* offsets, const double* anchors,
    const double* weights, double* out, int nrhs, double a, double om, double r,
    double c0, double c1, double scale) {
    int lane=threadIdx.x%32, feat=threadIdx.x/32;
    int rhs=blockIdx.x%nrhs, dir=(blockIdx.x/nrhs)%2, g=blockIdx.x/(2*nrhs);
    long long begin=offsets[g], end=offsets[g+1];
    const double* h=anchors+12*g+6*dir;
    double zmin=x[3*ids[begin]+2], zmax=x[3*ids[end-1]+2];
    double l0=0.,l1=0.,r0=0.,r1=0.;
    __shared__ double contributions[16][32];
    for(long long j=begin+lane; j<end; j+=32) {
        if(labels[j] != (1-dir)) continue;
        int id=ids[j]; const double* p=x+3*id;
        double w=weights[id*nrhs+rhs]*feature(p,h,a,c0,c1,feat,true);
        double dl=1+zmin-p[2], dr=1+p[2]-zmax;
        double vl=w*exp(-a*dl), vr=w*exp(-a*dr);
        l0+=vl; l1+=dl*vl; r0+=vr; r1+=dr*vr;
    }
    for(int step=16; step; step/=2) {
        l0+=__shfl_down_sync(0xffffffff,l0,step);
        l1+=__shfl_down_sync(0xffffffff,l1,step);
        r0+=__shfl_down_sync(0xffffffff,r0,step);
        r1+=__shfl_down_sync(0xffffffff,r1,step);
    }
    l0=__shfl_sync(0xffffffff,l0,0)/om;
    l1=(__shfl_sync(0xffffffff,l1,0)+r*l0)/om;
    r0=__shfl_sync(0xffffffff,r0,0)/om;
    r1=(__shfl_sync(0xffffffff,r1,0)+r*r0)/om;
    for(int pass=0; pass<2; ++pass) {
        double carry0=pass?r0:l0, carry1=pass?r1:l1, prev=pass?zmax:zmin;
        for(long long base=0; base<end-begin; base+=32) {
            long long k=base+lane; bool valid=k<end-begin;
            long long j=valid ? (pass?end-1-k:begin+k) : (pass?begin:end-1);
            int id=ids[j]; const double* p=x+3*id; double z=p[2];
            bool source=labels[j]==(1-dir);
            double s0=valid&&source ? weights[id*nrhs+rhs]*feature(p,h,a,c0,c1,feat,true) : 0.;
            double s1=0.;
            for(int step=1; step<32; step*=2) {
                double old0=__shfl_up_sync(0xffffffff,s0,step);
                double old1=__shfl_up_sync(0xffffffff,s1,step);
                double oldz=__shfl_up_sync(0xffffffff,z,step);
                if(lane>=step) {
                    double delta=fabs(z-oldz), decay=exp(-a*delta);
                    s0+=decay*old0; s1+=decay*(old1+delta*old0);
                }
            }
            double delta=fabs(z-prev), decay=exp(-a*delta);
            s0+=decay*carry0; s1+=decay*(carry1+delta*carry0);
            carry0=__shfl_sync(0xffffffff,s0,31);
            carry1=__shfl_sync(0xffffffff,s1,31);
            prev=__shfl_sync(0xffffffff,z,31);
            contributions[feat][lane]=valid&&!source ? feature(p,h,a,c0,c1,feat,false)*(s0+a*s1) : 0.;
            __syncthreads();
            if(feat==0 && valid && !source) {
                double v=0.;for(int f=0; f<16; ++f) v+=contributions[f][lane];
                atomicAdd(out+id*nrhs+rhs,scale*v);
            }
            __syncthreads();
        }
    }
}
'''
_scan=cp.RawKernel(_SOURCE,'scan_ranges')


class PeriodicScanKernel(PeriodicRangeKernel):
    """Same analytic action and geometry, with 32-entry warp scans."""
    def __call__(self,weights):
        w=cp.asarray(weights);vector=w.ndim==1
        if vector:w=w[:,None]
        if w.ndim!=2 or w.shape[0]!=self.n or not w.shape[1] or w.dtype.kind not in 'fiu' or not bool(cp.isfinite(w).all()):
            raise ValueError('weights must be finite real (N,) or (N,R)')
        w=cp.ascontiguousarray(w,dtype='d');nrhs=w.shape[1];out=w*self.noise_variance
        if self.record['far_groups']:
            _scan((self.record['far_groups']*2*nrhs,),(512,),
                  (self.x,*self.far,self.anchors,w,out,nrhs,self.a,self.om,self.r,self.c0,self.c1,self.scale))
        _direct((self.record['direct_groups']*nrhs,),(32,),
                (self.x,*self.direct,w,out,nrhs,self.a,self.c0,self.c1,self.norm,self.variance))
        return out[:,0] if vector else out
