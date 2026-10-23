"""Spatial block Jacobi and independent PCG for an unchanged covariance action.

The preconditioner uses analytic periodic product-Matern principal blocks.
It is only an inverse approximation inside PCG, never a replacement covariance.
"""
import cupy as cp
import cupyx
import numpy as np


def pcg(action,rhs,preconditioner,rtol=1e-5,maxiter=500,x0=None):
    """Batched independent real PCG, with residuals in the original system."""
    if rtol<=0 or maxiter<1 or int(maxiter)!=maxiter:
        raise ValueError('require positive tolerance and iteration count')
    if cp.iscomplexobj(rhs) or (x0 is not None and cp.iscomplexobj(x0)):
        raise ValueError('PCG requires real arrays')
    b=cp.asarray(rhs,dtype=action.op.dtype)
    if b.ndim!=2: raise ValueError('expected RHS (N,R)')
    x=cp.zeros_like(b) if x0 is None else cp.array(x0,dtype=b.dtype,copy=True)
    if x.shape!=b.shape: raise ValueError('initial solution shape differs from RHS')
    r=b.copy() if x0 is None else b-action(x)
    bnorm2=(b*b).sum(0);threshold=rtol**2*bnorm2
    rr=(r*r).sum(0);active=rr>threshold
    z=preconditioner(r);rz=(r*z).sum(0)
    if not bool(cp.all((rz>0)|~active)): raise RuntimeError('nonpositive preconditioner curvature')
    p=z.copy();converged=not bool(cp.any(active));iteration=0
    for iteration in range(1,int(maxiter)+1):
        if converged: iteration-=1;break
        ap=action(p);denom=(p*ap).sum(0)
        if not bool(cp.all((denom>0)|~active)): raise RuntimeError('nonpositive operator curvature')
        step=cp.where(active,rz/cp.where(denom!=0,denom,1),0)
        x+=p*step;r-=ap*step
        rr=(r*r).sum(0);next_active=rr>threshold
        converged=not bool(cp.any(next_active))
        if converged: break
        z=preconditioner(r);next_rz=(r*z).sum(0)
        if not bool(cp.all((next_rz>0)|~next_active)): raise RuntimeError('nonpositive preconditioner curvature')
        beta=cp.where(active,next_rz/cp.where(rz!=0,rz,1),0)
        p=z+p*beta;rz=next_rz;active=next_active
    actual=cp.sqrt(((action(x)-b)**2).sum(0))/cp.where(bnorm2>0,cp.sqrt(bnorm2),1)
    return x,{'iterations':iteration,'recurrence_converged':converged,
              'relative_residual':cp.asnumpy(actual).tolist(),
              'converged':bool(cp.all(actual<=2*rtol)),'preconditioned':True}


def morton_order(points):
    """Periodic coordinates sorted by a deterministic 63-bit Morton key."""
    x=cp.asarray(points,dtype='d')
    q=cp.floor(cp.mod(x+np.pi,2*np.pi)/(2*np.pi)*(2**21)).astype(cp.uint64)
    key=cp.zeros(len(x),dtype=cp.uint64)
    for bit in range(21):
        for d in range(3): key|=((q[:,d]>>bit)&1)<<(3*bit+d)
    return cp.argsort(key)


def analytic_blocks(points,lengthscale,variance):
    """Double-precision analytic product covariance, points (blocks,size,3)."""
    x=cp.asarray(points,dtype='d');ell=float(lengthscale);rt3=np.sqrt(3.)
    first=2*ell+rt3/np.tanh(rt3/(2*ell));w0=rt3/2
    denominator=(1+np.exp(-rt3/ell))*(first-2*w0*np.tanh(w0/ell))
    result=cp.ones((*x.shape[:2],x.shape[1]),dtype='d')
    for d in range(3):
        distance=cp.mod(cp.abs(x[:,:,None,d]-x[:,None,:,d])/(2*np.pi),1)
        w=rt3*(distance-.5)
        value=(cp.exp((w-w0)/ell)+cp.exp((-w-w0)/ell))*(first-2*w*cp.tanh(w/ell))
        result*=variance*value/denominator
    return result


class SpatialBlockJacobi:
    """Fixed SPD inverse blocks with O(N * block_size) retained storage.

    Morton ordering groups nearby points without changing the solve operator.
    Block construction/inversion uses bounded batches in double precision;
    application uses the requested dtype. Heteroscedastic diagonal noise is
    supported, but sequential-prior semantics are outside this class.
    """
    def __init__(self,points,noise_variance,lengthscale=.01,variance=.1,
                 block_size=128,setup_batch=64,dtype='float32'):
        if (block_size!=int(block_size) or block_size<1 or setup_batch!=int(setup_batch)
                or setup_batch<1 or lengthscale<=0 or variance<=0):
            raise ValueError('invalid block size, setup batch or kernel parameters')
        self.dtype=np.dtype(dtype)
        if self.dtype not in (np.dtype('float32'),np.dtype('float64')): raise ValueError('invalid dtype')
        points=cp.asarray(points,dtype=self.dtype)
        if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not bool(cp.isfinite(points).all()):
            raise ValueError('require finite nonempty points (N,3)')
        self.n=len(points);self.size=int(block_size)
        noise=cp.broadcast_to(cp.asarray(noise_variance,dtype='d'),(self.n,))
        if not bool(cp.all(cp.isfinite(noise)&(noise>0))): raise ValueError('noise variances must be positive')
        self.order=morton_order(points);self.blocks=(self.n+self.size-1)//self.size
        self.inverse=cp.empty((self.blocks,self.size,self.size),dtype=self.dtype)
        ordered=cp.zeros((self.blocks*self.size,3),dtype=self.dtype);ordered[:self.n]=points[self.order]
        diagonal=cp.ones(self.blocks*self.size,dtype='d');diagonal[:self.n]=noise[self.order]
        ordered=ordered.reshape(self.blocks,self.size,3);diagonal=diagonal.reshape(self.blocks,self.size)
        ids=cp.arange(self.size)
        for start in range(0,self.blocks,int(setup_batch)):
            stop=min(start+int(setup_batch),self.blocks)
            matrix=analytic_blocks(ordered[start:stop],lengthscale,variance)
            matrix[:,ids,ids]+=diagonal[start:stop]
            if stop==self.blocks and self.n%self.size:
                valid=self.n%self.size
                matrix[-1,valid:,:]=0;matrix[-1,:,valid:]=0
                matrix[-1,ids[valid:],ids[valid:]]=1
            with cupyx.errstate(linalg='raise'):
                cp.linalg.cholesky(matrix)  # Reject loss of SPD; never add silent jitter.
                inv=cp.linalg.inv(matrix)
            self.inverse[start:stop]=((inv+inv.swapaxes(1,2))*.5).astype(self.dtype)
        self.storage_bytes=self.inverse.nbytes+self.order.nbytes

    def __call__(self,rhs):
        if cp.iscomplexobj(rhs): raise ValueError('require real RHS')
        rhs=cp.asarray(rhs,dtype=self.dtype)
        if rhs.ndim!=2 or len(rhs)!=self.n: raise ValueError('expected RHS (N,R)')
        padded=cp.zeros((self.blocks*self.size,rhs.shape[1]),dtype=self.dtype)
        padded[:self.n]=rhs[self.order]
        values=cp.matmul(self.inverse,padded.reshape(self.blocks,self.size,-1)).reshape(-1,rhs.shape[1])
        result=cp.empty_like(rhs);result[self.order]=values[:self.n]
        return result
