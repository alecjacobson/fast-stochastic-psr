"""Separable float64 coefficient cache with warp-composed z transitions.

Stores 19 doubles per far entry, versus 46 in periodic_cached. Geometry and the
analytic model are unchanged. No intermediate expanded cache is constructed.
"""
import time
import cupy as cp
from geospsr.periodic_range import PeriodicRangeKernel,_direct
from geospsr.periodic_cached import _SCAN_SOURCE

_PREP=r'''
extern "C" __global__ void compact_features(const double* x,const int* ids,
    const unsigned char* labels,const int* groups,const double* anchors,
    double* coeff,long long count,double a,double c0,double c1) {
    long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=count*16) return;
    long long j=tid%count;int channel=tid/count,dir=channel/8,d=(channel%8)/4,c=channel%4;
    int branch=c>>1,moment=c&1,g=groups[j];const double* h=anchors+12*g+6*dir;
    double u=h[d]*x[3*ids[j]+d],lo=h[2+2*d],hi=h[3+2*d];bool source=labels[j]==(1-dir);
    double dist=source?(branch?u-lo:hi-u):(branch?1+lo-u:u-hi);
    coeff[tid]=exp(-a*dist)*(source?(moment?dist:1.):(moment?c1:c0+c1*dist));
}
extern "C" __global__ void compact_decays(const double* x,const int* ids,
    const int* groups,const long long* offsets,double* decays,long long count,double a) {
    long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(j>=count) return;
    int g=groups[j];long long begin=offsets[g],end=offsets[g+1];double z=x[3*ids[j]+2];
    decays[j]=j==begin?1.:exp(-a*(z-x[3*ids[j-1]+2]));
    decays[count+j]=exp(-a*(1+x[3*ids[begin]+2]-z));
    decays[2*count+j]=exp(-a*(1+z-x[3*ids[end-1]+2]));
}
'''
_features=cp.RawKernel(_PREP,'compact_features');_decays=cp.RawKernel(_PREP,'compact_decays')
_SOURCE=_SCAN_SOURCE.replace('void cached_scan_ranges(', 'void compact_scan_ranges(')
_SOURCE=_SOURCE.replace('long long begin=offsets[g], end=offsets[g+1];',
                        'long long count=offsets[gridDim.x/(2*nrhs)];\n    long long begin=offsets[g], end=offsets[g+1];')
_SOURCE=_SOURCE.replace('coeff[(2*j+dir)*16+feat]',
                        '(coeff[(dir*8+(feat&3))*count+j]*coeff[(dir*8+4+(feat>>2))*count+j])')
_SOURCE=_SOURCE.replace('decays[14*j+12]','decays[count+j]').replace('decays[14*j+13]','decays[2*count+j]')
_SOURCE=_SOURCE.replace('double s1=0.;','double s1=0.;\n            double transition=pass?(j+1<end?decays[j+1]:1.):decays[j];')
_SOURCE=_SOURCE.replace('double oldz=__shfl_up_sync', 'double oldp=__shfl_up_sync(0xffffffff,transition,step);\n                double oldz=__shfl_up_sync')
_SOURCE=_SOURCE.replace('decays[14*j+6*pass+level]','transition')
_SOURCE=_SOURCE.replace('s0+=decay*old0; s1+=decay*(old1+delta*old0);',
                        's0+=decay*old0; s1+=decay*(old1+delta*old0); transition*=oldp;')
_SOURCE=_SOURCE.replace('decays[14*j+6*pass+5]','transition')
_scan=cp.RawKernel(_SOURCE,'compact_scan_ranges')


class CompactPeriodicRangeKernel(PeriodicRangeKernel):
    def __init__(self,*args,max_cache_bytes=2**30,**kwargs):
        super().__init__(*args,**kwargs)
        n=self.record['far_entries'];size=n*19*8
        if size>max_cache_bytes:raise MemoryError('compact periodic cache exceeds byte cap')
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
        groups=cp.repeat(cp.arange(self.record['far_groups'],dtype='i'),cp.diff(self.far[2]))
        self.coeff=cp.empty((16,n),dtype='d');self.decays=cp.empty((3,n),dtype='d')
        if n:
            _features(((n*16+127)//128,),(128,),
                      (self.x,self.far[0],self.far[1],groups,self.anchors,self.coeff,n,self.a,self.c0,self.c1))
            _decays(((n+127)//128,),(128,),(self.x,self.far[0],groups,self.far[2],self.decays,n,self.a))
        cp.cuda.get_current_stream().synchronize()
        self.record.update(cache_setup_seconds=time.perf_counter()-start,cache_bytes=size,
                           cached_total_device_bytes=self.record['retained_device_bytes']+size)

    def __call__(self,weights):
        w=cp.asarray(weights);vector=w.ndim==1
        if vector:w=w[:,None]
        if w.ndim!=2 or w.shape[0]!=self.n or not w.shape[1] or w.dtype.kind not in 'fiu' or not bool(cp.isfinite(w).all()):
            raise ValueError('weights must be finite real (N,) or (N,R)')
        w=cp.ascontiguousarray(w,dtype='d');nrhs=w.shape[1];out=w*self.noise_variance
        if self.record['far_groups']:
            _scan((self.record['far_groups']*2*nrhs,),(512,),
                  (self.x,*self.far,self.anchors,self.coeff,self.decays,w,out,nrhs,
                   self.a,self.om,self.r,self.c0,self.c1,self.scale))
        _direct((self.record['direct_groups']*nrhs,),(32,),
                (self.x,*self.direct,w,out,nrhs,self.a,self.c0,self.c1,self.norm,self.variance))
        return out[:,0] if vector else out
