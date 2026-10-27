"""Geometry-coefficient cache for serial and warp-scan periodic actions.

This trades substantial retained storage for eliminating repeated double
exponentials. It is experimental; large-N memory feasibility is not established.
"""
import time
import cupy as cp
from geospsr.periodic_range import PeriodicRangeKernel,_KERNEL,_direct
from geospsr.periodic_scan import _SOURCE

_PREP = _KERNEL.split('extern "C" __global__ void ranges')[0]+r'''
extern "C" __global__ void cache_features(const double* x,const int* ids,
    const unsigned char* labels,const int* groups,const double* anchors,
    double* coeff,long long count,double a,double c0,double c1) {
    long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=count*32) return;
    long long j=tid/32;int dir=(tid%32)/16,f=tid%16,g=groups[j];
    coeff[tid]=feature(x+3*ids[j],anchors+12*g+6*dir,a,c0,c1,f,labels[j]==(1-dir));
}
extern "C" __global__ void cache_decays(const double* x,const int* ids,
    const int* groups,const long long* offsets,double* decays,long long count,double a) {
    long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(j>=count) return;
    int g=groups[j];long long begin=offsets[g],end=offsets[g+1];double z=x[3*ids[j]+2];
    for(int pass=0;pass<2;++pass) {
        long long rank=pass?end-1-j:j-begin;
        for(int b=0;b<6;++b) {
            long long prev= b==5 ? ((rank/32)*32 ? (rank/32)*32-1 : 0) : (rank>=(1<<b)?rank-(1<<b):0);
            long long k=pass?end-1-prev:begin+prev;
            decays[14*j+6*pass+b]=exp(-a*fabs(z-x[3*ids[k]+2]));
        }
    }
    decays[14*j+12]=exp(-a*(1+x[3*ids[begin]+2]-z));
    decays[14*j+13]=exp(-a*(1+z-x[3*ids[end-1]+2]));
}
'''
_features=cp.RawKernel(_PREP,'cache_features');_decays=cp.RawKernel(_PREP,'cache_decays')


def cached_source(source,scan):
    name='scan_ranges' if scan else 'ranges';channel='feat' if scan else 'lane'
    source=source.replace('void '+name+'(', 'void cached_'+name+'(')
    source=source.replace('const double* weights, double* out, int nrhs,',
                          'const double* coeff, const double* decays, const double* weights, double* out, int nrhs,')
    for role in ['true','false']:
        source=source.replace(f'feature(p,h,a,c0,c1,{channel},{role})',f'coeff[(2*j+dir)*16+{channel}]')
    source=source.replace('exp(-a*dl)','decays[14*j+12]').replace('exp(-a*dr)','decays[14*j+13]')
    if scan:
        source=source.replace('for(int step=1; step<32; step*=2)', 'for(int step=1,level=0; step<32; step*=2,++level)')
        source=source.replace('decay=exp(-a*delta);\n                    s0', 'decay=decays[14*j+6*pass+level];\n                    s0')
        source=source.replace('decay=exp(-a*delta);\n            s0', 'decay=decays[14*j+6*pass+5];\n            s0')
    else:
        source=source.replace('decay=exp(-a*delta);','decay=decays[14*j+6*pass];')
    return source

_SERIAL_SOURCE=cached_source(_KERNEL.split('extern "C" __global__ void direct')[0],False)
_SCAN_SOURCE=cached_source(_SOURCE,True)
_serial=cp.RawKernel(_SERIAL_SOURCE,'cached_ranges');_scan=cp.RawKernel(_SCAN_SOURCE,'cached_scan_ranges')


class CachedPeriodicRangeKernel(PeriodicRangeKernel):
    def __init__(self,*args,scan=False,max_cache_bytes=2**30,**kwargs):
        super().__init__(*args,**kwargs);self.scan=scan
        n=self.record['far_entries'];size=n*46*8
        if size>max_cache_bytes:raise MemoryError('periodic coefficient cache exceeds byte cap')
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
        groups=cp.repeat(cp.arange(self.record['far_groups'],dtype='i'),cp.diff(self.far[2]))
        self.coeff=cp.empty((n,2,16),dtype='d');self.decays=cp.empty((n,14),dtype='d')
        if n:
            _features(((n*32+127)//128,),(128,),
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
            kernel=_scan if self.scan else _serial
            kernel((self.record['far_groups']*2*nrhs,),(512 if self.scan else 32,),
                   (self.x,*self.far,self.anchors,self.coeff,self.decays,w,out,nrhs,
                    self.a,self.om,self.r,self.c0,self.c1,self.scale))
        _direct((self.record['direct_groups']*nrhs,),(32,),
                (self.x,*self.direct,w,out,nrhs,self.a,self.c0,self.c1,self.norm,self.variance))
        return out[:,0] if vector else out
