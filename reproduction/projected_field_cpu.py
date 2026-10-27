"""Independent CPU/FINUFFT controls for compressed Fourier posterior fields.

Dense training is intentionally limited to small overlap cases. These timings
must not be reported as GPU performance or an implementation for massive N.
"""
import time
import numpy as np
import finufft
from scipy.linalg import cho_factor,cho_solve
from reproduction.query_projection import prior_basis
from geospsr.projected_field import CompressedJointField


class CPUScalarFourier:
    def __init__(self,cutoff=50,lengthscale=.01,variance=.1,eps=1e-14,threads=4):
        self.cutoff=cutoff;self.ell=lengthscale;self.variance=variance;self.eps=eps;self.threads=threads
        self.a=np.sqrt(3)/lengthscale;r=np.exp(-self.a);om=-np.expm1(-self.a)
        self.normalization=1+2*r/om+2*self.a*r/om**2
        axis=np.arange(-cutoff,cutoff+1,dtype=float)
        rho1=variance*4*self.a**3/(self.a*self.a+(2*np.pi*axis)**2)**2/self.normalization
        self.shape=(len(axis),)*3
        self.freq=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
        rho=(rho1[:,None,None]*rho1[None,:,None]*rho1[None,None,:]).ravel()
        norm2=np.sum(self.freq*self.freq,axis=1)
        self.scalar=np.where(norm2>0,rho/np.where(norm2>0,norm2,1),0)
        self.cross=-1j*self.freq.T*self.scalar[None,:]

    def plan(self,kind,points,count):
        plan=finufft.Plan(kind,self.shape,n_trans=count,eps=self.eps,
                         isign=-1 if kind==1 else 1,dtype='complex128',nthreads=self.threads)
        plan.setpts(*[np.ascontiguousarray(np.asarray(points,dtype=float)[:,d]) for d in range(3)])
        return plan

    def source(self,plan,values):
        values=np.ascontiguousarray(values,dtype='complex128')
        count=len(values)
        return plan.execute(values[0] if count==1 else values).reshape(count,-1)

    def target(self,plan,coefficients):
        coefficients=np.ascontiguousarray(coefficients,dtype='complex128');count=len(coefficients)
        value=coefficients.reshape(count,*self.shape)
        return plan.execute(value[0] if count==1 else value).real.reshape(count,-1)

    def evaluate(self,points,coefficients,batch=4):
        values=[];plans={}
        for start in range(0,len(coefficients),batch):
            c=coefficients[start:start+batch];count=len(c)
            if count not in plans:plans[count]=self.plan(2,points,count)
            values.append(self.target(plans[count],c))
        return np.concatenate(values) if values else np.empty((0,len(points)))

    def covariance(self,query):
        q=np.asarray(query,dtype=float);diff=(q[:,None]-q[None,:]).reshape(-1,3)
        return self.evaluate(diff,self.scalar[None,:])[0].reshape(len(q),len(q))

    def analytic_training(self,points,noise):
        x=np.asarray(points,dtype=float);t=np.remainder(np.abs(x[:,None]-x[None,:])/(2*np.pi),1.)
        r=np.exp(-self.a);om=-np.expm1(-self.a)
        c0=1/om+self.a*r/om**2;c1=self.a/om
        per_axis=((c0+c1*t)*np.exp(-self.a*t)+(c0+c1*(1-t))*np.exp(-self.a*(1-t)))/self.normalization
        a=self.variance**3*np.prod(per_axis,axis=2);a.flat[::len(x)+1]+=noise
        return a


def prepare(points,queries,normals=None,rank=None,cutoff=50,batch=4,noise=.02,
            lengthscale=.01,variance=.1,progress=None):
    x=np.asarray(points,dtype=float);q=np.asarray(queries,dtype=float)
    if len(x)>2048:raise ValueError('CPU dense control limited to2048 observations')
    seconds={}
    def timed(name,fn):
        start=time.perf_counter();v=fn();seconds[name]=time.perf_counter()-start;return v
    op=timed('scalar_setup',lambda:CPUScalarFourier(cutoff,lengthscale,variance))
    a=timed('dense_training',lambda:op.analytic_training(x,noise))
    chol=timed('dense_factor',lambda:cho_factor(a,lower=True))
    k=timed('reference_prior',lambda:op.covariance(q));k=(k+k.T)*.5
    basis,tail,info=timed('basis',lambda:prior_basis(k,rank))
    rank=basis.shape[1];lam=np.asarray(info['prior_eigenvalues'][:rank])
    if rank and np.any(lam<=info['prior_roundoff_allowance']):
        raise ValueError('requested lift includes prior eigenvalues below the fixed roundoff allowance')
    # Prior-whitened coordinates: feature lift columns are orthonormal. This
    # avoids amplifying a small H-root error by inverse tiny eigenvalues.
    u=basis/np.sqrt(lam)
    l=np.empty((rank,len(op.scalar)),dtype='complex128');b=np.empty_like(l)
    weights=np.empty((3,rank,len(x)));rhs_all=np.empty_like(weights)
    def reduced():
        plans={}
        for start in range(0,rank,batch):
            end=min(start+batch,rank);count=end-start
            if count not in plans:
                plans[count]=(op.plan(1,q,count),op.plan(2,x,3*count),op.plan(1,x,3*count))
            forward,back,source=plans[count]
            modes=op.source(forward,u[:,start:end].T)
            rhs=op.target(back,(modes[:,None,:]*op.cross.conj()[None,:,:]).reshape(3*count,-1)).T
            w=cho_solve(chol,rhs)
            for d in range(3):weights[d,start:end]=w[:,d::3].T;rhs_all[d,start:end]=rhs[:,d::3].T
            transformed=op.source(source,w.T).reshape(count,3,-1)
            l[start:end]=np.sum(transformed*op.cross[None,:,:],axis=1)
            b[start:end]=modes*op.scalar[None,:]
            if progress:progress(end,rank)
    timed('reduced_conditioning_and_coefficients',reduced)
    def products():
        g=sum(w@a@w.T for w in weights)
        residual=np.stack([rhs-w@a for rhs,w in zip(rhs_all,weights)])
        upper=sum(r@r.T for r in residual)/noise
        return (g+g.T)*.5,residual,upper
    g,residual,solve_upper=timed('compressed_products',products)
    sampler=timed('compressed_sampler',lambda:CompressedJointField(op.scalar,l,b,g,
        frequency_chunk=16384,max_cache_bytes=40*2**30))
    def mean():
        if normals is None:return np.zeros(len(op.scalar),dtype='complex128'),0.
        alpha=cho_solve(chol,np.asarray(normals,dtype=float))
        modes=op.source(op.plan(1,x,3),alpha.T)
        coeff=np.sum(modes*op.cross,axis=0)
        offset=op.evaluate(x,coeff[None,:])[0].mean()
        return coeff,float(offset)
    mean_coeff,offset=timed('mean_fit_and_center',mean)
    seconds['prepare_total']=sum(seconds.values())
    return {'operator':op,'sampler':sampler,'mean_coefficients':mean_coeff,'offset':offset,
        'prior':k,'basis':basis,'eigenvalues':lam,'tail':tail,'weights':weights,
        'rhs':rhs_all,'residual':residual,'solve_upper':solve_upper,
        'cross_coefficients':l,'lift_coefficients':b,'projected_covariance':g,
        'record':{'seconds':seconds,'basis':info,'sampler':sampler.record,
                  'coordinates':'prior-whitened reference eigenfunctions',
                  'scope':'CPU dense/FINUFFT independent control; no GPU or large-N speedup'}}
