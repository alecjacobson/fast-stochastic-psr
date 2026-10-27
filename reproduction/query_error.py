"""Solve-error budgets and a quadratic residual correction in query space.

For X=C.T, R=X-AW and A>=noise_min*I:
S_corrected = Kff-sym(X.T W)-sym(W.T R)
            = S_exact+R.T A^-1 R.
Hence 0 <= S_corrected-S_exact <= R.T R/noise_min in PSD order.
The three independent vector components are summed throughout. These are
exact-arithmetic identities, not interval bounds for floating-point actions.
"""
import numpy as np


def residual_envelopes(cross_norms,residual_norms,noise_min):
    c=np.asarray(cross_norms,dtype='d');r=np.asarray(residual_norms,dtype='d')
    if c.ndim!=2 or c.shape[1]!=3 or not len(c) or r.shape!=c.shape or not np.isfinite(c).all() or not np.isfinite(r).all() or np.any(c<0) or np.any(r<0):
        raise ValueError('expected finite nonnegative (queries,3) norms')
    if not np.isfinite(noise_min) or noise_min<=0:raise ValueError('positive noise floor required')
    first=c@r.T/noise_min
    return (first+first.T)*.5,r@r.T/noise_min


def relative_covariance_budget(covariance,upper,xp=np):
    s=xp.asarray(covariance);b=xp.asarray(upper)
    if s.ndim!=2 or s.shape[0]!=s.shape[1] or b.shape!=s.shape or not bool(xp.isfinite(s).all() and xp.isfinite(b).all()):
        raise ValueError('expected matching finite square matrices')
    eig,vec=xp.linalg.eigh((s+s.T)*.5)
    threshold=100*np.finfo(float).eps*len(s)*float(xp.max(xp.abs(eig)))
    if float(eig.min())<=threshold:
        return {'informative':False,'reason':'covariance not numerically positive definite; use absolute/marginal bounds'}
    whitened=(vec.T@b@vec)/xp.sqrt(eig[:,None]*eig[None,:])
    eta=max(0.,float(xp.linalg.eigvalsh((whitened+whitened.T)*.5).max()))
    return {'informative':eta<1,'upper_relative_to_computed_covariance':eta,
            'upper_relative_to_exact_covariance':eta/(1-eta) if eta<1 else None}


class QueryResidualAudit:
    """Capture all-row residuals, correct the covariance, and retain error budgets.

    approximate(rhs) returns weights in observation-by-(query,component) order.
    Retaining weights/residuals uses O(3NM) storage; this optional correction is
    intended for small query panels. It does not inherit the streaming O(Nb)
    memory bound of the uncorrected covariance assembler. Clear vectors after
    validation with clear_vectors(); scalar norms and M*M budgets remain.
    """
    def __init__(self, action, approximate, noise_min):
        if not np.isfinite(noise_min) or noise_min<=0:raise ValueError('positive noise floor required')
        self.action,self.approximate,self.noise=action,approximate,float(noise_min)
        self.weights=[];self.residuals=[];self.cross_norms=[];self.residual_norms=[]
        self.record={'blocks':[],'qualification':'solve-error identity in exact arithmetic; excludes cross/prior NUFFT error, analytic-action numerical error, mean/centering error and roundoff/PSD projection'}
        self.corrected=False

    def __call__(self, rhs):
        import cupy as cp
        if self.corrected:raise RuntimeError('audit already consumed')
        b=cp.asarray(rhs,dtype='d')
        if b.ndim!=2 or not len(b) or not b.shape[1] or b.shape[1]%3 or not bool(cp.isfinite(b).all()):
            raise ValueError('expected finite (observations,3*queries) RHS')
        w=cp.asarray(self.approximate(b),dtype='d')
        if w.shape!=b.shape or not bool(cp.isfinite(w).all()):raise ValueError('invalid approximate weights')
        residual=b-self.action(w)
        if residual.shape!=b.shape or not bool(cp.isfinite(residual).all()):raise ValueError('invalid analytic residual')
        cn=cp.linalg.norm(b,axis=0);rn=cp.linalg.norm(residual,axis=0)
        self.cross_norms.append(cn.reshape(-1,3).get());self.residual_norms.append(rn.reshape(-1,3).get())
        self.record['blocks'].append({'relative_residual':(rn/cp.where(cn>0,cn,1)).get().tolist()})
        self._store_vectors(w,residual)
        return w

    def _store_vectors(self,weights,residual):
        self.weights.append(weights);self.residuals.append(residual)

    def _products(self, m):
        import cupy as cp
        self.w=cp.concatenate(self.weights,axis=1);self.weights.clear()
        self.r=cp.concatenate(self.residuals,axis=1);self.residuals.clear()
        n=len(self.w)
        cross=cp.zeros((m,m),dtype='d');gram=cp.zeros_like(cross)
        for d in range(3):
            wd=cp.ascontiguousarray(self.w.reshape(n,m,3)[:,:,d]);rd=cp.ascontiguousarray(self.r.reshape(n,m,3)[:,:,d])
            cross+=wd.T@rd;gram+=rd.T@rd
        self.record.update(vector_storage='gpu',vector_storage_bytes=self.w.nbytes+self.r.nbytes)
        return cross,gram

    def correct(self, covariance):
        import cupy as cp
        if self.corrected or not self.cross_norms:raise RuntimeError('audit is empty or already consumed')
        m=sum(len(v) for v in self.cross_norms)
        if covariance.shape!=(m,m):raise ValueError('query count does not match captured RHS')
        cross,gram=self._products(m)
        self.upper=(gram+gram.T)*(.5/self.noise)
        self.shift=(cross+cross.T)*.5
        result=(covariance+covariance.T)*.5-self.shift
        self.cross_norms=np.concatenate(self.cross_norms);self.residual_norms=np.concatenate(self.residual_norms)
        self.first_envelope,self.quadratic_envelope=residual_envelopes(self.cross_norms,self.residual_norms,self.noise)
        self.record.update(queries=m,noise_min=self.noise,cross_norms=self.cross_norms.tolist(),residual_norms=self.residual_norms.tolist(),
            first_order_max_entry_bound=float(self.first_envelope.max()),
            corrected_max_entry_bound=float(self.quadratic_envelope.max()),
            corrected_operator_error_upper=max(0.,float(cp.linalg.eigvalsh(self.upper).max())),
            corrected_max_marginal_error_upper=float(cp.diag(self.upper).max()),
            relative_budget=relative_covariance_budget(result,self.upper,xp=cp))
        self.corrected=True
        return result

    def clear_vectors(self):
        self.weights.clear();self.residuals.clear()
        self.w=self.r=None


class HostQueryResidualAudit(QueryResidualAudit):
    """Keep O(3NM) vectors on the host; form products in bounded GPU row chunks.

    This changes storage and accumulation order, not the correction or bound.
    Host allocation is capped explicitly. GPU product workspace is O(M*rows),
    plus O(M*M) matrices and the caller's analytic/NUFFT plans. No N*M GPU
    concatenation or repeated analytic action is needed. Host memory is NOT
    bounded independently of the problem size; this is not a low-rank method.
    """
    def __init__(self,action,approximate,noise_min,observations,queries,
                 row_chunk=8192,max_host_bytes=128*2**30):
        super().__init__(action,approximate,noise_min)
        for v in [observations,queries,row_chunk]:
            if v!=int(v) or v<1:raise ValueError('positive integer dimensions required')
        self.n,self.m,self.row_chunk=int(observations),int(queries),int(row_chunk)
        storage=2*3*self.n*self.m*8
        if storage>max_host_bytes:raise MemoryError(f'host vectors need {storage} bytes, cap {max_host_bytes}')
        self.host_w=np.empty((3,self.m,self.n),dtype='d');self.host_r=np.empty_like(self.host_w)
        self.stored=0
        self.record.update(vector_storage='host',vector_storage_bytes=storage,
            row_chunk=self.row_chunk,host_byte_cap=max_host_bytes,
            gpu_product_vector_bytes_upper=2*min(self.n,self.row_chunk)*self.m*8,
            storage_qualification='GPU bound covers two product operands only; excludes M*M matrices, GEMM workspace, capture blocks and caller kernel/NUFFT plans')

    def _store_vectors(self,weights,residual):
        import cupy as cp
        count=weights.shape[1]//3
        if len(weights)!=self.n or self.stored+count>self.m:raise ValueError('captured vectors exceed configured shape')
        if self.host_w is None:raise RuntimeError('host vectors have been cleared')
        end=self.stored+count
        for d in range(3):
            cp.ascontiguousarray(weights[:,d::3].T).get(out=self.host_w[d,self.stored:end])
            cp.ascontiguousarray(residual[:,d::3].T).get(out=self.host_r[d,self.stored:end])
        self.stored=end

    def _products(self,m):
        import cupy as cp
        if self.stored!=self.m or m!=self.m:raise ValueError('incomplete host query capture')
        cross=cp.zeros((m,m),dtype='d');gram=cp.zeros_like(cross)
        for d in range(3):
            for start in range(0,self.n,self.row_chunk):
                end=min(start+self.row_chunk,self.n)
                wd=cp.asarray(np.ascontiguousarray(self.host_w[d,:,start:end]))
                rd=cp.asarray(np.ascontiguousarray(self.host_r[d,:,start:end]))
                cross+=wd@rd.T;gram+=rd@rd.T
                # Release operands before allocating the next chunk, including
                # the last short chunk. CuPy may retain freed pool capacity.
                del wd,rd
        return cross,gram

    def clear_vectors(self):
        super().clear_vectors()
        self.host_w=self.host_r=None
