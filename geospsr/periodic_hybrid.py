"""Serial small rectangles and warp scans for larger exact periodic sums."""
import time
import cupy as cp
from geospsr.native_range import NativeCompactRangeKernel
from geospsr.periodic_compact import _SOURCE
from geospsr.periodic_cached import _SERIAL_SOURCE
from geospsr.periodic_range import _direct


def selected(source,oldname,newname):
    source=source.replace('void '+oldname+'(', 'void '+newname+'(')
    source=source.replace('const double* weights, double* out, int nrhs,',
                          'const int* selected, int totalgroups, const double* weights, double* out, int nrhs,')
    source=source.replace('g=blockIdx.x/(2*nrhs);','g=selected[blockIdx.x/(2*nrhs)];')
    return source.replace('offsets[gridDim.x/(2*nrhs)]','offsets[totalgroups]')


_SMALL=selected(_SERIAL_SOURCE,'cached_ranges','small_ranges')
_SMALL=_SMALL.replace('long long begin=offsets[g], end=offsets[g+1];',
                       'long long count=offsets[totalgroups];\n    long long begin=offsets[g], end=offsets[g+1];')
_SMALL=_SMALL.replace('coeff[(2*j+dir)*16+lane]',
                      '(coeff[(dir*8+(lane&3))*count+j]*coeff[(dir*8+4+(lane>>2))*count+j])')
_SMALL=_SMALL.replace('decays[14*j+12]','decays[count+j]').replace('decays[14*j+13]','decays[2*count+j]')
_SMALL=_SMALL.replace('decays[14*j+6*pass]','(pass?(j+1<end?decays[j+1]:1.):decays[j])')
_LARGE=selected(_SOURCE,'compact_scan_ranges','large_ranges')
_small=cp.RawKernel(_SMALL,'small_ranges');_large=cp.RawKernel(_LARGE,'large_ranges')


class HybridPeriodicRangeKernel(NativeCompactRangeKernel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
        lengths=cp.diff(self.far[2]);self.small=cp.flatnonzero(lengths<=32).astype('i');self.large=cp.flatnonzero(lengths>32).astype('i')
        cp.cuda.get_current_stream().synchronize()
        self.record.update(scheduling_seconds=time.perf_counter()-start,small_group_threshold=32,
                           small_groups=len(self.small),large_groups=len(self.large),
                           scheduling_bytes=self.small.nbytes+self.large.nbytes)
        self.record['cached_total_device_bytes']+=self.record['scheduling_bytes']

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
        _direct((self.record['direct_groups']*nrhs,),(32,),
                (self.x,*self.direct,w,out,nrhs,self.a,self.c0,self.c1,self.norm,self.variance))
        return out[:,0] if vector else out
