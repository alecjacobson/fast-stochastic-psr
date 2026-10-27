"""Optional compiled CPU partitioner, preserving periodic_range's plan exactly."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
import cupy as cp
import numpy as np
from geospsr.periodic_range import PeriodicRangeKernel
from geospsr.periodic_compact import CompactPeriodicRangeKernel


def library():
    source=Path(__file__).parent/'native/range_plan.cpp'
    flags=['-O3','-std=c++17','-shared','-fPIC']
    digest=hashlib.sha256(source.read_bytes()+repr(flags).encode()).hexdigest()[:24]
    directory=Path(os.environ.get('GEOSPSR_RUNTIME',tempfile.gettempdir()))/'native-range';directory.mkdir(parents=True,exist_ok=True)
    target=directory/f'{digest}.so';cached=target.exists();start=time.perf_counter()
    if not cached:
        fd,name=tempfile.mkstemp(suffix='.so',dir=directory);os.close(fd)
        try:
            subprocess.run(['g++',*flags,str(source),'-o',name],check=True,capture_output=True,text=True)
            os.replace(name,target)
        finally:
            if os.path.exists(name):os.unlink(name)
    lib=ct.CDLL(str(target));lib.range_build.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int64,ct.c_void_p,ct.c_int];lib.range_build.restype=ct.c_void_p
    lib.range_free.argtypes=[ct.c_void_p];lib.range_free.restype=None
    lib.range_counts.argtypes=[ct.c_void_p,ct.c_void_p];lib.range_counts.restype=None
    lib.range_data.argtypes=[ct.c_void_p,ct.c_int];lib.range_data.restype=ct.c_void_p
    return lib,{'native_library_seconds':time.perf_counter()-start,'native_library_cached':cached,'native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}


class NativeRangeKernel(PeriodicRangeKernel):
    def __init__(self,points,lengthscale=.01,variance=.1,noise_variance=.02,leaf_size=32,max_entries=20_000_000):
        raw=cp.asnumpy(points) if isinstance(points,cp.ndarray) else np.asarray(points)
        if np.iscomplexobj(raw) or raw.ndim!=2 or raw.shape[1]!=3 or not len(raw) or not np.isfinite(raw).all():raise ValueError('points must be a nonempty finite real (N,3) array')
        if len(raw)>=2**31 or not np.isfinite([lengthscale,variance,noise_variance]).all() or lengthscale<=0 or variance<=0 or noise_variance<0:raise ValueError('invalid size or model parameters')
        if not isinstance(leaf_size,int) or leaf_size<2 or max_entries<1:raise ValueError('invalid leaf size or entry cap')
        x=np.ascontiguousarray(np.remainder(raw.astype('d')/(2*np.pi),1.));self.n=len(x);self.noise_variance=float(noise_variance)
        lib,record=library();start=time.perf_counter();error=ct.create_string_buffer(1024)
        handle=lib.range_build(x.ctypes.data,len(x),leaf_size,max_entries,error,len(error))
        if not handle:raise MemoryError(error.value.decode())
        try:
            counts=np.empty(5,dtype='int64');lib.range_counts(handle,counts.ctypes.data);nf,nd,nfe,nde,maxfar=map(int,counts)
            arrays=[]
            for field,(count,dtype) in enumerate([(nfe,'i'),(nfe,'u1'),(nf+1,'int64'),(nf*12,'d'),(nde,'i'),(nde,'u1'),(nd+1,'int64'),(nd,'u1')]):
                pointer=lib.range_data(handle,field)
                a=np.ctypeslib.as_array(ct.cast(pointer,ct.POINTER(np.ctypeslib.as_ctypes_type(np.dtype(dtype)))),shape=(count,)) if count else np.empty(0,dtype)
                arrays.append(a)
            record.update(n=len(x),leaf_size=leaf_size,far_groups=nf,direct_groups=nd,far_entries=nfe,direct_entries=nde,
                          max_far_group_entries=maxfar,cpu_plan_seconds=time.perf_counter()-start)
            cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
            self.x=cp.asarray(x);self.far=tuple(cp.asarray(a) for a in arrays[:3]);self.anchors=cp.asarray(arrays[3].reshape(-1,12));self.direct=tuple(cp.asarray(a) for a in arrays[4:])
            cp.cuda.get_current_stream().synchronize();record['device_transfer_seconds']=time.perf_counter()-start
        finally:lib.range_free(handle)
        record['retained_device_bytes']=sum(a.nbytes for a in (self.x,*self.far,*self.direct,self.anchors));self.record=record
        self.a=np.sqrt(3)/lengthscale;self.r=np.exp(-self.a);self.om=-np.expm1(-self.a)
        self.c0=2*lengthscale+2*np.sqrt(3)*self.r/self.om;self.c1=2*np.sqrt(3)
        self.norm=self.c0+(self.c0+self.c1)*self.r;self.variance=float(variance)
        z=1+2*(self.r/self.om+self.a*self.r/self.om**2);self.scale=variance**3/(self.norm**2*z)


class NativeCompactRangeKernel(CompactPeriodicRangeKernel,NativeRangeKernel):
    """Compact cache with native geometry construction; same GPU action."""
