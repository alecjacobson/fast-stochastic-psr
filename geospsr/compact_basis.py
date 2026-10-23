"""Decode a half-storage coarse basis in registers; accumulate at solve precision."""
from functools import lru_cache
import numpy as np
import cupy as cp


_SOURCE=r'''
#include <cuda_fp16.h>
#define REAL COMPUTE_TYPE
#define RB RHS_CHANNELS

__device__ inline void pair_values(const half* basis, long long offset,
                                  int rank, int column, REAL inverse_scale,
                                  REAL &a, REAL &b) {
    a=0; b=0;
    if (column>=rank) return;
    if ((rank & 1)==0) {
        const float2 v=__half22float2(*reinterpret_cast<const half2*>(basis+offset));
        a=REAL(v.x)*inverse_scale; b=REAL(v.y)*inverse_scale;
    } else {
        a=REAL(__half2float(basis[offset]))*inverse_scale;
        if (column+1<rank) b=REAL(__half2float(basis[offset+1]))*inverse_scale;
    }
}

extern "C" __global__ void project(const half* basis, const REAL* rhs,
    REAL* partial, long long n, int rank, int channels, int tiles, REAL inverse_scale) {
    const int lane=threadIdx.x & 31, warp=threadIdx.x >> 5;
    const int local=2*lane, column=64*blockIdx.x+local;
    const int first_channel=RB*blockIdx.z;
    REAL a[RB]={0},b[RB]={0};
    for (long long base=1024LL*blockIdx.y;base<n;base+=1024LL*tiles) {
        const long long end=(base+1024<n ? base+1024 : n);
        for (long long row=base+warp;row<end;row+=8) {
            REAL u,v;
            pair_values(basis,row*rank+column,rank,column,inverse_scale,u,v);
            #pragma unroll
            for (int k=0;k<RB;++k) {
                if (first_channel+k<channels) {
                    const REAL x=rhs[row*channels+first_channel+k];
                    a[k]+=u*x; b[k]+=v*x;
                }
            }
        }
    }
    __shared__ REAL cache[8][64][RB];
    #pragma unroll
    for (int k=0;k<RB;++k) {
        cache[warp][local][k]=a[k];cache[warp][local+1][k]=b[k];
    }
    __syncthreads();
    if (warp==0) {
        #pragma unroll
        for (int k=0;k<RB;++k) {
            REAL x=0,y=0;
            #pragma unroll
            for (int w=0;w<8;++w) {x+=cache[w][local][k];y+=cache[w][local+1][k];}
            const int c=first_channel+k;
            if (c<channels && column<rank)
                partial[(static_cast<long long>(blockIdx.y)*rank+column)*channels+c]=x;
            if (c<channels && column+1<rank)
                partial[(static_cast<long long>(blockIdx.y)*rank+column+1)*channels+c]=y;
        }
    }
}

extern "C" __global__ void correct(const half* basis, const REAL* projection,
    const REAL* rhs, REAL* result, long long n, int rank, int channels, REAL inverse_scale) {
    const int lane=threadIdx.x & 31, warp=threadIdx.x >> 5;
    const long long row=8LL*blockIdx.x+warp;
    if (row>=n) return; // Every lane in this warp takes the same branch.
    const int first_channel=RB*blockIdx.y;
    REAL accum[RB]={0};
    for (int column=2*lane;column<rank;column+=64) {
        REAL u,v;
        pair_values(basis,row*rank+column,rank,column,inverse_scale,u,v);
        #pragma unroll
        for (int k=0;k<RB;++k) {
            const int c=first_channel+k;
            if (c<channels) {
                accum[k]+=u*projection[static_cast<long long>(column)*channels+c];
                if (column+1<rank) accum[k]+=v*projection[static_cast<long long>(column+1)*channels+c];
            }
        }
    }
    #pragma unroll
    for (int k=0;k<RB;++k) {
        for (int delta=16;delta>0;delta>>=1)
            accum[k]+=__shfl_down_sync(0xffffffff,accum[k],delta);
        const int c=first_channel+k;
        if (lane==0 && c<channels) result[row*channels+c]=rhs[row*channels+c]-accum[k];
    }
}
'''


@lru_cache(maxsize=16)
def _kernels(dtype,channels):
    code=_SOURCE.replace('COMPUTE_TYPE','float' if dtype=='float32' else 'double').replace('RHS_CHANNELS',str(channels))
    module=cp.RawModule(code=code,options=('--std=c++11',))
    return module.get_function('project'),module.get_function('correct')


def fused_half_apply(basis,damping,rhs,scale):
    """Return rhs-U D U.T rhs with U=basis/scale, no full decoded basis.

    Projection scratch is at most 1024*rank*R solve-dtype values. Both solve
    precisions are supported. Channels are tiled by at most eight; source and
    rank tails are explicit. CUDA compilation/first launch is paid on first use.
    This function does not replace the preconditioner's construction/SPD guard.
    """
    if basis.dtype!=cp.float16 or basis.ndim!=2 or not basis.flags.c_contiguous or min(basis.shape)<1:
        raise ValueError('require a contiguous nonempty half basis')
    if rhs.dtype not in (cp.float32,cp.float64) or rhs.ndim!=2 or rhs.shape[0]!=basis.shape[0]:
        raise ValueError('require float32/float64 RHS (N,R)')
    if damping.dtype!=rhs.dtype or damping.shape!=(basis.shape[1],) or not damping.flags.c_contiguous:
        raise ValueError('require contiguous damping at solve precision')
    if not np.isfinite(scale) or scale<=0:raise ValueError('require finite positive scale')
    inverse=rhs.dtype.type(1./scale)
    if not np.isfinite(inverse):raise ValueError('inverse scale is not representable')
    rhs=cp.ascontiguousarray(rhs)
    n,rank=basis.shape;channels=rhs.shape[1]
    if not channels:return cp.empty_like(rhs)
    tiles=min(1024,(n+1023)//1024);rb=min(8,channels)
    project,correct=_kernels(rhs.dtype.name,rb)
    partial=cp.empty((tiles,rank,channels),dtype=rhs.dtype)
    project(((rank+63)//64,tiles,(channels+rb-1)//rb),(256,),
            (basis,rhs,partial,np.int64(n),np.int32(rank),np.int32(channels),np.int32(tiles),inverse))
    projection=partial.sum(axis=0,dtype=rhs.dtype)
    projection*=damping[:,None]
    result=cp.empty_like(rhs)
    correct(((n+7)//8,(channels+rb-1)//rb),(256,),
            (basis,projection,rhs,result,np.int64(n),np.int32(rank),np.int32(channels),inverse))
    return result
