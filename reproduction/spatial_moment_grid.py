"""Strong bounded-domain reference: cache direct moments, then interpolate.

Grid construction must use the reference's actual evaluator for timing claims.
Using a fast candidate only screens the interpolation density, not build time.
"""
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates, spline_filter
import numpy as np


class SpatialMomentGrid:
    def __init__(self, values, lower, upper, order=3, periodic=False):
        values=cp.asarray(values,dtype='d')
        if values.ndim!=4 or len(values)!=2 or len(set(values.shape[1:]))!=1 or values.shape[-1]<4:
            raise ValueError('expected (2,G,G,G) mean/variance grid, G>=4')
        self.lower=cp.asarray(lower,dtype='d');self.upper=cp.asarray(upper,dtype='d')
        if self.lower.shape!=(3,) or self.upper.shape!=(3,) or not bool(cp.all(self.upper>self.lower)):
            raise ValueError('invalid bounds')
        if order not in (1,3,5):raise ValueError('linear, cubic or quintic interpolation required')
        self.order,self.size=order,values.shape[-1]
        self.periodic=periodic
        self.mode='grid-wrap' if periodic else 'mirror'
        self.values=cp.stack([spline_filter(v,order=order,mode=self.mode) for v in values]) if order>1 else values
        self.cache_bytes=self.values.nbytes+self.lower.nbytes+self.upper.nbytes

    @classmethod
    def prepare(cls, evaluate, size, lower, upper, order=3, batch=16384, periodic=False):
        if size!=int(size) or size<4 or batch<1:raise ValueError('invalid size or batch')
        size=int(size);lo=cp.asarray(lower,dtype='d');hi=cp.asarray(upper,dtype='d')
        result=cp.empty((2,size**3),dtype='d')
        for start in range(0,size**3,batch):
            ids=cp.arange(start,min(size**3,start+batch),dtype='int64')
            coords=cp.stack([ids//(size*size),(ids//size)%size,ids%size],axis=1)
            points=(lo+(hi-lo)*coords/(size if periodic else size-1)).astype('f')
            mean,var=evaluate(points)
            result[0,start:start+len(ids)]=mean;result[1,start:start+len(ids)]=var
        return cls(result.reshape(2,size,size,size),lower,upper,order,periodic)

    def coordinates(self,points):
        q=cp.asarray(points,dtype='d')
        if q.ndim!=2 or q.shape[1]!=3 or not bool(cp.isfinite(q).all()):raise ValueError('invalid queries')
        if not bool(cp.all((q>=self.lower)&(q<=self.upper))):raise ValueError('query outside prepared domain')
        return cp.ascontiguousarray(((q-self.lower)/(self.upper-self.lower)*(self.size if self.periodic else self.size-1)).T)

    def evaluate(self, points):
        coords=self.coordinates(points)
        return tuple(map_coordinates(v,coords,order=self.order,mode=self.mode,prefilter=False) for v in self.values)

    def evaluate_channel(self,points,channel):
        return map_coordinates(self.values[channel],self.coordinates(points),order=self.order,mode=self.mode,prefilter=False)
