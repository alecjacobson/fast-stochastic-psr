"""Experimental full analytic periodic product-Matern GPU matrix action.

A dyadic x/y pair partition leaves 16 separable moment channels. Stable cyclic
z sweeps evaluate these channels without global growing exponential weights.
This is a serial-sweep research prototype, not a production solver backend.
"""
import time
import numpy as np
import cupy as cp


_KERNEL = r'''
__device__ double feature(const double* x, const double* h, double a,
                         double c0, double c1, int lane, bool source) {
    double value = 1.;
    for (int d=0; d<2; ++d) {
        int c = (lane >> (2*d)) & 3, branch=c>>1, moment=c&1;
        double u=h[d]*x[d], lo=h[2+2*d], hi=h[3+2*d];
        double dist=source ? (branch ? u-lo : hi-u) : (branch ? 1+lo-u : u-hi);
        // Ordering is non-strict; ties require no perturbation.
        value *= exp(-a*dist) * (source ? (moment ? dist : 1.)
                                  : (moment ? c1 : c0+c1*dist));
    }
    return value;
}
extern "C" __global__ void ranges(const double* x, const int* ids,
    const unsigned char* labels, const long long* offsets, const double* anchors,
    const double* weights, double* out, int nrhs, double a, double om, double r,
    double c0, double c1, double scale) {
    int lane=threadIdx.x;
    if(lane>=16) return;
    int rhs=blockIdx.x%nrhs, dir=(blockIdx.x/nrhs)%2, g=blockIdx.x/(2*nrhs);
    long long begin=offsets[g], end=offsets[g+1];
    const double* h=anchors+12*g+6*dir;
    double zmin=x[3*ids[begin]+2], zmax=x[3*ids[end-1]+2];
    double l0=0.,l1=0.,r0=0.,r1=0.;
    for(long long j=begin; j<end; ++j) {
        if(labels[j] != (1-dir)) continue;
        int id=ids[j]; const double* p=x+3*id;
        double w=weights[id*nrhs+rhs]*feature(p,h,a,c0,c1,lane,true);
        double dl=1+zmin-p[2], dr=1+p[2]-zmax;
        double vl=w*exp(-a*dl), vr=w*exp(-a*dr);
        l0+=vl; l1+=dl*vl; r0+=vr; r1+=dr*vr;
    }
    l0/=om; l1=(l1+r*l0)/om;
    r0/=om; r1=(r1+r*r0)/om;
    for(int pass=0; pass<2; ++pass) {
        double s0=pass?r0:l0, s1=pass?r1:l1, prev=pass?zmax:zmin;
        for(long long k=0; k<end-begin; ++k) {
            long long j=pass?end-1-k:begin+k; int id=ids[j]; const double* p=x+3*id;
            double delta=pass?prev-p[2]:p[2]-prev, decay=exp(-a*delta);
            s1=decay*(s1+delta*s0); s0*=decay; prev=p[2];
            if(labels[j] == (1-dir)) {
                s0+=weights[id*nrhs+rhs]*feature(p,h,a,c0,c1,lane,true);
            } else {
                double v=feature(p,h,a,c0,c1,lane,false)*(s0+a*s1);
                for(int step=8; step; step/=2) v+=__shfl_down_sync(0xffff,v,step,16);
                if(lane==0) atomicAdd(out+id*nrhs+rhs,scale*v);
            }
        }
    }
}
extern "C" __global__ void direct(const double* x, const int* ids,
    const unsigned char* labels, const long long* offsets, const unsigned char* full,
    const double* weights, double* out, int nrhs, double a, double c0, double c1,
    double norm, double variance) {
    int g=blockIdx.x/nrhs, rhs=blockIdx.x%nrhs;
    long long begin=offsets[g], end=offsets[g+1];
    for(long long k=begin+threadIdx.x; k<end; k+=blockDim.x) {
        int i=ids[k]; double result=0.;
        for(long long j=begin; j<end; ++j) {
            if(!full[g] && labels[k]==labels[j]) continue;
            int id=ids[j]; double v=1.;
            for(int d=0; d<3; ++d) {
                double t=fabs(x[3*i+d]-x[3*id+d]);
                v*=variance*((c0+c1*t)*exp(-a*t)+(c0+c1*(1-t))*exp(-a*(1-t)))/norm;
            }
            result+=v*weights[id*nrhs+rhs];
        }
        atomicAdd(out+i*nrhs+rhs,result);
    }
}
'''
_ranges = cp.RawKernel(_KERNEL, 'ranges')
_direct = cp.RawKernel(_KERNEL, 'direct')


class PeriodicRangeKernel:
    """Geometry-cached float64 action of K + noise_variance I.

    Coordinates are radians, period 2*pi. Variance is per axis, so diag(K)
    equals variance**3. Geometry construction is on CPU and is charged separately.
    The entry cap bounds retained lists, not peak Python planning overhead.
    """
    def __init__(self, points, lengthscale=.01, variance=.1, noise_variance=.02,
                 leaf_size=32, max_entries=20_000_000):
        start=time.perf_counter()
        raw=cp.asnumpy(points) if isinstance(points,cp.ndarray) else np.asarray(points)
        if np.iscomplexobj(raw) or raw.ndim!=2 or raw.shape[1]!=3 or not len(raw) or not np.isfinite(raw).all():
            raise ValueError('points must be a nonempty finite real (N,3) array')
        if len(raw)>=2**31 or not np.isfinite([lengthscale,variance,noise_variance]).all() or lengthscale<=0 or variance<=0 or noise_variance<0:
            raise ValueError('invalid size or model parameters')
        if not isinstance(leaf_size,int) or leaf_size<2 or max_entries<1:
            raise ValueError('invalid leaf size or entry cap')
        x=np.remainder(raw.astype('d')/(2*np.pi),1.)
        self.n=len(x); self.noise_variance=float(noise_variance)
        far=[]; direct=[]; anchors=[]; total=0
        def retain(container, ids, labels, full=False, signs=None):
            nonlocal total
            total+=len(ids)
            if total>max_entries: raise MemoryError('range plan entry cap exceeded')
            if signs is not None:
                h=[]
                for direction in range(2):
                    s=np.array(signs)*(1 if direction==0 else -1)
                    src=x[ids[labels==(1-direction)],:2]*s
                    bounds=[s[0],s[1],src[:,0].min(),src[:,0].max(),src[:,1].min(),src[:,1].max()]
                    h.extend(bounds)
                anchors.append(h)
                order=np.argsort(x[ids,2],kind='stable');ids=ids[order];labels=labels[order]
            container.append((ids.astype('i'),labels.astype('u1'),full))
        def ys(ids,labels):
            if not labels.any() or labels.all(): return
            if len(ids)<=leaf_size:
                retain(direct,ids,labels);return
            mid=len(ids)//2;lo,hi=ids[:mid],ids[mid:];ll,hl=labels[:mid],labels[mid:]
            for s,t,sy in [(lo[ll],hi[~hl],1),(hi[hl],lo[~ll],-1)]:
                if len(s) and len(t):
                    retain(far,np.concatenate([s,t]),np.r_[np.ones(len(s),bool),np.zeros(len(t),bool)],signs=(1,sy))
            ys(lo,ll);ys(hi,hl)
        def xs(ids):
            if len(ids)<=leaf_size:
                retain(direct,ids,np.ones(len(ids),bool),full=True);return
            mid=len(ids)//2
            labels=np.arange(len(ids))<mid
            order=np.argsort(x[ids,1],kind='stable')
            ys(ids[order],labels[order]);xs(ids[:mid]);xs(ids[mid:])
        xs(np.argsort(x[:,0],kind='stable'))
        def flatten(groups):
            offsets=np.r_[0,np.cumsum([len(g[0]) for g in groups])].astype('int64')
            ids=np.concatenate([g[0] for g in groups]) if groups else np.empty(0,'i')
            labels=np.concatenate([g[1] for g in groups]) if groups else np.empty(0,'u1')
            return ids,labels,offsets,np.array([g[2] for g in groups],dtype='u1')
        fh=flatten(far);dh=flatten(direct);ah=np.array(anchors,dtype='d').reshape(-1,12)
        self.record={'n':self.n,'leaf_size':leaf_size,'far_groups':len(far),'direct_groups':len(direct),
                     'far_entries':len(fh[0]),'direct_entries':len(dh[0]),
                     'cpu_plan_seconds':time.perf_counter()-start}
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
        self.x=cp.asarray(x);self.far=tuple(cp.asarray(a) for a in fh[:3]);self.direct=tuple(cp.asarray(a) for a in dh)
        self.anchors=cp.asarray(ah)
        cp.cuda.get_current_stream().synchronize()
        self.record['device_transfer_seconds']=time.perf_counter()-start
        self.record['retained_device_bytes']=sum(a.nbytes for a in (self.x,*self.far,*self.direct,self.anchors))
        self.record['max_far_group_entries']=max((len(g[0]) for g in far),default=0)
        self.a=np.sqrt(3)/lengthscale;self.r=np.exp(-self.a);self.om=-np.expm1(-self.a)
        self.c0=2*lengthscale+2*np.sqrt(3)*self.r/self.om;self.c1=2*np.sqrt(3)
        self.norm=self.c0+(self.c0+self.c1)*self.r;self.variance=float(variance)
        z=1+2*(self.r/self.om+self.a*self.r/self.om**2)
        self.scale=variance**3/(self.norm**2*z)

    def __call__(self, weights):
        w=cp.asarray(weights)
        vector=w.ndim==1
        if vector: w=w[:,None]
        if w.ndim!=2 or w.shape[0]!=self.n or not w.shape[1] or w.dtype.kind not in 'fiu' or not bool(cp.isfinite(w).all()):
            raise ValueError('weights must be finite real (N,) or (N,R)')
        w=cp.ascontiguousarray(w,dtype='d');nrhs=w.shape[1]
        out=w*self.noise_variance
        if self.record['far_groups']:
            _ranges((self.record['far_groups']*2*nrhs,),(32,),
                    (self.x,*self.far,self.anchors,w,out,nrhs,self.a,self.om,self.r,self.c0,self.c1,self.scale))
        _direct((self.record['direct_groups']*nrhs,),(32,),
                (self.x,*self.direct,w,out,nrhs,self.a,self.c0,self.c1,self.norm,self.variance))
        return out[:,0] if vector else out
