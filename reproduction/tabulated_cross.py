"""Batched GPU trilinear reference for the same finite Fourier cross kernel.

Uses the release's fifth-power nonuniform knot geometry, at selectable density.
Table refinement must be checked against direct Fourier queries before timing
claims. One thread produces all three components for an observation/query pair.
"""
import cupy as cp
import numpy as np


_kernel = cp.RawKernel(r'''
extern "C" __global__ void rows(const float* q, const float* x,
 const double* knots, const double* table, double* out,
 const int nq, const int nx, const int g) {
 long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i >= (long long)nq*nx) return;
 int lo[3]; double w[3];
 for(int d=0;d<3;++d){
   double a=(double)q[3*(i/nx)+d]-(double)x[3*(i%nx)+d];
   a-=6.2831853071795864769*floor((a+3.14159265358979323846)/6.2831853071795864769);
   int l=0,h=g-1;
   while(h-l>1){int m=(l+h)/2; if(knots[m]<=a)l=m;else h=m;}
   lo[d]=l;w[d]=(a-knots[l])/(knots[l+1]-knots[l]);
 }
 double v[3]={0.,0.,0.};
 for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int c=0;c<2;++c){
   double weight=(a?w[0]:1-w[0])*(b?w[1]:1-w[1])*(c?w[2]:1-w[2]);
   long long j=3*((long long)(lo[0]+a)*g*g+(lo[1]+b)*g+lo[2]+c);
   for(int d=0;d<3;++d)v[d]+=weight*table[j+d];
 }
 for(int d=0;d<3;++d)out[3*i+d]=v[d];
}''', 'rows')


class TabulatedCrossKernel:
    def __init__(self, operator, density=200):
        if density < 4 or density != int(density):
            raise ValueError('integer density >=4 required')
        self.density = int(density)
        self.knots = cp.linspace(-1,1,density,dtype='d')**5*np.pi
        grid = cp.stack(cp.meshgrid(self.knots,self.knots,self.knots,indexing='ij'),-1).reshape(-1,3)
        self.values = cp.ascontiguousarray(operator.cross_values(grid), dtype='d')

    def rows(self, query, points):
        q=cp.ascontiguousarray(cp.asarray(query,dtype='f'))
        x=cp.ascontiguousarray(cp.asarray(points,dtype='f'))
        result=cp.empty((len(q),len(x),3),dtype='d')
        count=len(q)*len(x)
        _kernel(((count+127)//128,),(128,),
                (q,x,self.knots,self.values,result,len(q),len(x),self.density))
        return result
