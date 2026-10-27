"""Cache exact analytic direct leaves for repeated periodic range actions."""
import time
import cupy as cp
from geospsr.periodic_hybrid import HybridPeriodicRangeKernel,_small,_large

_KERNEL=r'''
extern "C" __global__ void prepare_leaves(const double* x,const int* ids,
    const unsigned char* labels,const long long* offsets,const unsigned char* full,
    const long long* moff,double* matrix,double a,double c0,double c1,double norm,double variance) {
    int g=blockIdx.x;long long begin=offsets[g],len=offsets[g+1]-begin;
    for(long long k=threadIdx.x;k<len*len;k+=blockDim.x) {
        long long qi=begin+k/len,si=begin+k%len;double v=1.;
        if(!full[g]&&labels[qi]==labels[si])v=0.;
        else {
            int i=ids[qi],j=ids[si];
            for(int d=0;d<3;++d) {
                double t=fabs(x[3*i+d]-x[3*j+d]);
                v*=variance*((c0+c1*t)*exp(-a*t)+(c0+c1*(1-t))*exp(-a*(1-t)))/norm;
            }
        }
        matrix[moff[g]+k]=v;
    }
}
extern "C" __global__ void apply_leaves(const int* ids,const long long* offsets,
    const long long* moff,const double* matrix,const double* weights,double* out,int nrhs) {
    int g=blockIdx.x/nrhs,rhs=blockIdx.x%nrhs;
    long long begin=offsets[g],len=offsets[g+1]-begin;
    for(long long i=threadIdx.x;i<len;i+=blockDim.x) {
        double v=0.;
        for(long long j=0;j<len;++j)v+=matrix[moff[g]+i*len+j]*weights[ids[begin+j]*nrhs+rhs];
        atomicAdd(out+ids[begin+i]*nrhs+rhs,v);
    }
}
'''
_prepare=cp.RawKernel(_KERNEL,'prepare_leaves');_apply=cp.RawKernel(_KERNEL,'apply_leaves')


class LeafCachedRangeKernel(HybridPeriodicRangeKernel):
    def __init__(self,*args,max_cache_bytes=2**30,**kwargs):
        super().__init__(*args,max_cache_bytes=max_cache_bytes,**kwargs)
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
        sizes=cp.diff(self.direct[2]);self.matrix_offsets=cp.concatenate([cp.zeros(1,dtype='int64'),cp.cumsum(sizes*sizes)])
        elements=int(self.matrix_offsets[-1]);size=elements*8
        if self.record['cache_bytes']+size>max_cache_bytes:raise MemoryError('combined periodic/leaf cache exceeds byte cap')
        self.matrices=cp.empty(elements,dtype='d')
        _prepare((self.record['direct_groups'],),(128,),
                 (self.x,*self.direct,self.matrix_offsets,self.matrices,self.a,self.c0,self.c1,self.norm,self.variance))
        cp.cuda.get_current_stream().synchronize()
        self.record.update(leaf_cache_seconds=time.perf_counter()-start,leaf_cache_bytes=size,
                           matrix_offset_bytes=self.matrix_offsets.nbytes)
        self.record['cached_total_device_bytes']+=size+self.matrix_offsets.nbytes

    def __call__(self,weights):
        w=cp.asarray(weights);vector=w.ndim==1
        if vector:w=w[:,None]
        if w.ndim!=2 or w.shape[0]!=self.n or not w.shape[1] or w.dtype.kind not in 'fiu' or not bool(cp.isfinite(w).all()):
            raise ValueError('weights must be finite real (N,) or (N,R)')
        w=cp.ascontiguousarray(w,dtype='d');nrhs=w.shape[1];out=w*self.noise_variance
        for kernel,groups,threads in [(_small,self.small,32),(_large,self.large,512)]:
            if len(groups):
                kernel((len(groups)*2*nrhs,),(threads,),
                       (self.x,*self.far,self.anchors,self.coeff,self.decays,groups,self.record['far_groups'],w,out,nrhs,
                        self.a,self.om,self.r,self.c0,self.c1,self.scale))
        _apply((self.record['direct_groups']*nrhs,),(32,),
               (self.direct[0],self.direct[2],self.matrix_offsets,self.matrices,w,out,nrhs))
        return out[:,0] if vector else out
