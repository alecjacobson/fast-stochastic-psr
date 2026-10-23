"""Landmark Nyström spectral preconditioning; the solve operator is unchanged."""
import cupy as cp
import cupyx
import numpy as np


def analytic_cross(points,landmarks,lengthscale=.01,variance=.1):
    """Periodic product-Matern covariance in double precision."""
    x=cp.asarray(points,dtype='d');y=cp.asarray(landmarks,dtype='d')
    ell=float(lengthscale);rt3=np.sqrt(3.);w0=rt3/2
    first=2*ell+rt3/np.tanh(w0/ell)
    denominator=(1+np.exp(-rt3/ell))*(first-2*w0*np.tanh(w0/ell))
    result=cp.ones((len(x),len(y)),dtype='d')
    for d in range(3):
        distance=cp.mod(cp.abs(x[:,None,d]-y[None,:,d])/(2*np.pi),1)
        w=rt3*(distance-.5)
        result*=variance*(cp.exp((w-w0)/ell)+cp.exp((-w-w0)/ell))*(first-2*w*cp.tanh(w/ell))/denominator
    return result


class LandmarkNystrom:
    """Apply I-U diag(1-(lambda_min+noise)/(lambda+noise)) U.T.

    U/lambda describe C W^dagger C.T, with C=K(X,landmarks). Only the
    preconditioner uses this analytic-kernel approximation. PCG must still
    apply the original full covariance. This is column sampling, not the
    Gaussian sketch used for the strongest randomized Nyström guarantees.

    Construction uses bounded row blocks and double precision. Rank-deficient
    landmark/feature Grams are truncated explicitly; no diagonal jitter is
    added. Retained basis dtype is independently selectable. An orthogonality
    check guards positive definiteness after storage rounding. Noise must be
    a positive scalar; heteroscedastic whitening is not implemented here.

    recompute_features=True builds the small Gram in one pass and recomputes
    rows for basis construction in a second pass, avoiding the double N-by-rank
    feature array. basis_dtype='float16' uses scaled storage, then decodes in
    bounded apply_row_batch blocks at the solve dtype (never float16 GEMM).
    fused_apply=True instead decodes half values in registers, with bounded
    projection scratch and the same effective stored basis in both passes.
    """
    def __init__(self,points,noise_variance,rank=512,lengthscale=.01,variance=.1,
                 seed=41,row_batch=8192,dtype='float32',eigen_rtol=1e-10,
                 basis_dtype=None,recompute_features=False,apply_row_batch=None,fused_apply=False):
        self.dtype=np.dtype(dtype)
        if self.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('invalid dtype')
        self.basis_dtype=self.dtype if basis_dtype is None else np.dtype(basis_dtype)
        if self.basis_dtype not in (np.dtype('float16'),np.dtype('float32'),np.dtype('float64')):
            raise ValueError('invalid basis dtype')
        if fused_apply and (self.basis_dtype!=np.dtype('float16') or apply_row_batch is not None):
            raise ValueError('fused application requires half storage without decoded row batches')
        self.fused_apply=bool(fused_apply)
        if apply_row_batch is not None and (apply_row_batch!=int(apply_row_batch) or apply_row_batch<1):
            raise ValueError('invalid application row batch')
        # Bound decompression instead of materializing a second N-by-rank basis.
        if self.basis_dtype!=self.dtype and apply_row_batch is None and not fused_apply:apply_row_batch=row_batch
        self.apply_row_batch=None if apply_row_batch is None else int(apply_row_batch)
        if (rank!=int(rank) or rank<1 or row_batch!=int(row_batch) or row_batch<1
                or not np.isfinite([lengthscale,variance,eigen_rtol]).all()
                or lengthscale<=0 or variance<=0 or not 0<eigen_rtol<1):
            raise ValueError('invalid rank, row batch or kernel parameters')
        noise=cp.asarray(noise_variance)
        if noise.ndim!=0 or not bool(cp.isfinite(noise)) or float(noise)<=0:raise ValueError('require positive scalar noise variance')
        self.noise=float(noise)
        x=cp.asarray(points,dtype=self.dtype)
        if x.ndim!=2 or x.shape[1]!=3 or not len(x) or not bool(cp.isfinite(x).all()):raise ValueError('require finite points (N,3)')
        if rank>len(x):raise ValueError('landmark rank exceeds observation count')
        self.n=len(x);rank=int(rank);row_batch=int(row_batch)
        self.landmark_ids=np.random.default_rng(seed).choice(self.n,rank,replace=False)
        landmarks=x[self.landmark_ids]
        w=analytic_cross(landmarks,landmarks,lengthscale,variance)
        with cupyx.errstate(linalg='raise'): values,vectors=cp.linalg.eigh((w+w.T)*.5)
        if not bool(cp.isfinite(values).all()) or float(values[-1])<=0:raise RuntimeError('invalid landmark Gram spectrum')
        if float(values[0]) < -eigen_rtol*float(values[-1]):raise RuntimeError('indefinite landmark Gram')
        keep=values>eigen_rtol*values[-1]
        whitening=vectors[:,keep]/cp.sqrt(values[keep])[None]
        landmark_rank=whitening.shape[1]
        features=None if recompute_features else cp.empty((self.n,landmark_rank),dtype='d')
        gram=cp.zeros((landmark_rank,landmark_rank),dtype='d')
        for start in range(0,self.n,row_batch):
            z=analytic_cross(x[start:start+row_batch],landmarks,lengthscale,variance)@whitening
            if features is not None:features[start:start+row_batch]=z
            gram+=z.T@z
        with cupyx.errstate(linalg='raise'): values,vectors=cp.linalg.eigh((gram+gram.T)*.5)
        if not bool(cp.isfinite(values).all()) or float(values[-1])<=0:raise RuntimeError('invalid feature Gram spectrum')
        if float(values[0]) < -eigen_rtol*float(values[-1]):raise RuntimeError('indefinite feature Gram')
        keep=values>eigen_rtol*values[-1]
        self.eigenvalues=values[keep]
        rotation=vectors[:,keep]/cp.sqrt(self.eigenvalues)[None]
        self.rank=len(self.eigenvalues)
        # A power-of-two scale reduces half-precision underflow without adding
        # rounding in the scale/unscale operation. Effective U is stored/scale.
        self.basis_scale=float(2**min(14,int(np.log2(self.n)//2))) if self.basis_dtype==np.dtype('float16') else 1.
        self.basis=cp.empty((self.n,self.rank),dtype=self.basis_dtype)
        for start in range(0,self.n,row_batch):
            z=(analytic_cross(x[start:start+row_batch],landmarks,lengthscale,variance)@whitening
               if features is None else features[start:start+row_batch])
            self.basis[start:start+row_batch]=(z@rotation)*self.basis_scale
        del features,z
        checked=cp.zeros((self.rank,self.rank),dtype='d')
        for start in range(0,self.n,row_batch):
            # Guard the basis actually used by multiplication, including any
            # storage-to-accumulation rounding (e.g. float64 storage/float32 solve).
            u=self.basis[start:start+row_batch].astype(self.dtype,copy=False).astype('d')/self.basis_scale
            checked+=u.T@u
        orthogonality_error=float(cp.linalg.norm(checked-cp.eye(self.rank)))
        self.damping=(1-(self.eigenvalues[0]+self.noise)/(self.eigenvalues+self.noise)).astype(self.dtype)
        spd_margin=1-float(self.damping.max())*(1+orthogonality_error)
        if not np.isfinite(spd_margin) or spd_margin<=0:
            raise RuntimeError('stored basis fails SPD guard; use higher precision or a better resolved coarse space')
        self.storage_bytes=self.basis.nbytes+self.damping.nbytes+self.eigenvalues.nbytes
        self.record={'requested_rank':rank,'landmark_rank':landmark_rank,'retained_rank':self.rank,
                     'seed':seed,'row_batch':row_batch,'eigen_rtol':eigen_rtol,
                     'eigenvalue_range':[float(self.eigenvalues[0]),float(self.eigenvalues[-1])],
                     'orthogonality_frobenius_error':orthogonality_error,'spd_margin_guard':spd_margin,
                     'basis_dtype':self.basis_dtype.name,'accumulation_dtype':self.dtype.name,
                     'basis_scale':self.basis_scale,'recompute_features':bool(recompute_features),
                     'apply_row_batch':self.apply_row_batch,'storage_bytes':self.storage_bytes,
                     'application_backend':'fused_half' if fused_apply else 'decoded',
                     'construction_feature_bytes':0 if recompute_features else self.n*landmark_rank*8,
                     'approximate_covariance_trace':float(self.eigenvalues.sum()),
                     'analytic_covariance_trace':self.n*variance**3}

    def __call__(self,rhs):
        if cp.iscomplexobj(rhs):raise ValueError('require real RHS')
        rhs=cp.asarray(rhs,dtype=self.dtype)
        if rhs.ndim!=2 or len(rhs)!=self.n:raise ValueError('expected RHS (N,R)')
        if self.fused_apply:
            from geospsr.compact_basis import fused_half_apply
            return fused_half_apply(self.basis,self.damping,rhs,self.basis_scale)
        if self.apply_row_batch is None:
            return rhs-self.basis@(self.damping[:,None]*(self.basis.T@rhs))
        projection=cp.zeros((self.rank,rhs.shape[1]),dtype=self.dtype)
        for start in range(0,self.n,self.apply_row_batch):
            stop=start+self.apply_row_batch
            u=self.basis[start:stop].astype(self.dtype,copy=False)
            if self.basis_scale!=1.:u/=self.basis_scale
            projection+=u.T@rhs[start:stop]
        projection*=self.damping[:,None]
        result=cp.empty_like(rhs)
        for start in range(0,self.n,self.apply_row_batch):
            stop=start+self.apply_row_batch
            u=self.basis[start:stop].astype(self.dtype,copy=False)
            if self.basis_scale!=1.:u/=self.basis_scale
            result[start:stop]=rhs[start:stop]-u@projection
        return result
