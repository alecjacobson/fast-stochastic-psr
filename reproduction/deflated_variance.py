"""Fixed-query covariance deflation; not a posterior field sampler.

For an independent fixed orthonormal Q and P=QQ.T, diagonal(B) equals
2*diagonal(PB)-diagonal(PBP)+diagonal((I-P)B(I-P)). Exact BQ and centered
Gaussian fields with covariance B make the resulting diagonal estimate
unbiased conditional on Q. Numerical covariance-action errors are separate.
"""
import numpy as np


class DeflatedCovarianceMoments:
    def __init__(self,basis,covariance_basis,xp=np):
        self.xp=xp
        if xp.iscomplexobj(basis) or xp.iscomplexobj(covariance_basis):raise ValueError('require real basis and covariance action')
        q=xp.array(basis,dtype='float64',copy=True);bq=xp.asarray(covariance_basis,dtype='float64')
        if q.ndim!=2 or not len(q) or q.shape[1]>len(q) or bq.shape!=q.shape:
            raise ValueError('expected matching (queries,rank) arrays with rank <= queries')
        if not bool(xp.all(xp.isfinite(q))) or not bool(xp.all(xp.isfinite(bq))):raise ValueError('nonfinite basis/action')
        error=float(xp.linalg.norm(q.T@q-xp.eye(q.shape[1])))
        if error>1e-8:raise ValueError('basis must be orthonormal at double precision')
        gram=q.T@bq
        asymmetry=float(xp.linalg.norm(gram-gram.T)/xp.maximum(xp.linalg.norm(gram),1e-300))
        # The antisymmetric part has identically zero quadratic form.
        gram=(gram+gram.T)*.5
        self.basis=q
        self.deterministic=2*xp.sum(q*bq,axis=1)-xp.sum((q@gram)*q,axis=1)
        self.sum_squares=xp.zeros(len(q),dtype='float64');self.count=0
        self.record={'queries':len(q),'rank':q.shape[1],'orthogonality_error':error,
                     'covariance_gram_asymmetry_relative':asymmetry,
                     'deterministic_negative_count':int((self.deterministic<0).sum()),
                     'retained_bytes':q.nbytes+self.deterministic.nbytes+self.sum_squares.nbytes}

    def update(self,values):
        xp=self.xp
        if xp.iscomplexobj(values):raise ValueError('require real centered fields')
        values=xp.asarray(values,dtype='float64')
        if values.ndim!=2 or not len(values) or values.shape[1]!=len(self.basis) or not bool(xp.all(xp.isfinite(values))):
            raise ValueError('expected finite nonempty (draw,queries) centered fields')
        residual=values-(values@self.basis)@self.basis.T if self.basis.shape[1] else values
        squares=xp.sum(residual*residual,axis=0)
        if not bool(xp.all(xp.isfinite(squares))):raise ValueError('second moment overflow')
        self.sum_squares+=squares;self.count+=len(values)

    def diagonal(self):
        if not self.count:raise ValueError('no evaluation draws')
        # No positivity clipping; the deterministic contribution may be negative.
        return self.deterministic+self.sum_squares/self.count


class FourierCorrectionCovariance:
    """Apply B=C A^-1 C.T at fixed queries using four NUFFT plans per block.

    solve(rhs) must apply the SAME three-component independent vector inverse
    used by the correction sampler. Approximate inverses need their own checks.
    This does not store a query-by-observation or query-by-query matrix.
    The scalar Fourier coefficients represent the real cross kernel C(q-x).
    """
    def __init__(self,points,queries,operator,solve,batch=4,coefficients=None):
        import cupy as cp
        if batch!=int(batch) or batch<1:raise ValueError('positive integer batch required')
        self.xp=cp;self.operator=operator;self.solve=solve;self.batch=int(batch)
        self.points=cp.array(points,dtype=operator.dtype,copy=True)
        self.queries=cp.array(queries,dtype=operator.dtype,copy=True)
        for value in [self.points,self.queries]:
            if value.ndim!=2 or value.shape[1]!=3 or not len(value) or not bool(cp.all(cp.isfinite(value))):
                raise ValueError('expected finite nonempty (count,3) positions')
        if coefficients is None:
            coefficients=operator.coefficients if hasattr(operator,'coefficients') else -1j*operator.multiplier
        self.coefficients=cp.array(coefficients,dtype=operator.cdtype,copy=True)
        if self.coefficients.shape!=(3,*operator.shape) or not bool(cp.all(cp.isfinite(self.coefficients))):
            raise ValueError('invalid cross-kernel Fourier coefficients')
        self.plans={}

    def _plans(self,count):
        if count not in self.plans:
            op=self.operator
            self.plans[count]=(op.plan(1,self.queries,count),op.plan(2,self.points,3*count),
                               op.plan(1,self.points,3*count),op.plan(2,self.queries,count))
        return self.plans[count]

    def __call__(self,values):
        cp=self.xp;op=self.operator
        if cp.iscomplexobj(values):raise ValueError('require real query vectors')
        values=cp.asarray(values,dtype=op.dtype)
        if values.ndim!=2 or len(values)!=len(self.queries) or not bool(cp.all(cp.isfinite(values))):
            raise ValueError('expected finite (queries,RHS) vectors')
        output=cp.empty_like(values)
        for start in range(0,values.shape[1],self.batch):
            block=values[:,start:start+self.batch];count=block.shape[1]
            qf,xb,xf,qb=self._plans(count)
            modes=qf.execute(cp.ascontiguousarray(block.T,dtype=op.cdtype))
            adjoint_modes=cp.ascontiguousarray((modes[:,None]*self.coefficients.conj()[None]).reshape(3*count,*op.shape))
            rhs=cp.ascontiguousarray(xb.execute(adjoint_modes).real.T)
            weights=self.solve(rhs)
            if weights.shape!=rhs.shape or weights.dtype!=op.dtype or not bool(cp.all(cp.isfinite(weights))):
                raise ValueError('solve changed shape/precision or returned nonfinite values')
            transformed=xf.execute(cp.ascontiguousarray(weights.T,dtype=op.cdtype)).reshape(count,3,*op.shape)
            coeff=cp.ascontiguousarray((transformed*self.coefficients[None]).sum(1))
            output[:,start:start+count]=qb.execute(coeff).real.T
        return output
