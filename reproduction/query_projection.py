"""Coherent query-basis approximation with separate solve/truncation bounds.

All PSD bounds are exact-arithmetic statements for a PSD joint GP prior.
Numerical prior/cross/action errors and root projection remain separate.
"""
import numpy as np


def assemble_projected(prior,basis,cross_action,weight_residual,tail,solve_upper,xp=np):
    """Assemble covariance and a PSD error upper bound from reduced solves.

    L=sum X_d.T W_d, H=sum W_d.T R_d, U=sum R_d.T R_d/noise_min.
    Tail must bound (I-QQ.T) Kff (I-QQ.T). Q has orthonormal columns.
    eta=(||U||F/||T||F)^(2/3) minimizes the Frobenius upper bound when
    tail and Q U Q.T have orthogonal support. Validity needs only eta>0.
    """
    k,q,l,h,t,u=[xp.asarray(v,dtype='float64') for v in [prior,basis,cross_action,weight_residual,tail,solve_upper]]
    if k.ndim!=2 or k.shape[0]!=k.shape[1] or q.ndim!=2 or q.shape[0]!=len(k):raise ValueError('invalid prior/basis dimensions')
    m,r=q.shape
    if r>m or l.shape!=(m,r) or h.shape!=(r,r) or u.shape!=(r,r) or t.shape!=k.shape:
        raise ValueError('inconsistent projected dimensions')
    if not all(bool(xp.isfinite(v).all()) for v in [k,q,l,h,t,u]):raise ValueError('nonfinite projected data')
    orth=float(xp.linalg.norm(q.T@q-xp.eye(r)))
    if orth>1e-8:raise ValueError('query basis must be orthonormal')
    g=q.T@l-h;g=(g+g.T)*.5
    covariance=k-l@q.T-q@l.T+q@g@q.T;covariance=(covariance+covariance.T)*.5
    a=float(xp.linalg.norm(t));b=float(xp.linalg.norm(u))
    if a==0:eta=None;upper=q@u@q.T
    elif b==0:eta=None;upper=t.copy()
    else:
        eta=(b/a)**(2/3)
        upper=(1+eta)*t+(1+1/eta)*(q@u@q.T)
    upper=(upper+upper.T)*.5
    bound=float(xp.linalg.norm(upper));norm=float(xp.linalg.norm(covariance))
    diag=xp.diag(covariance);ud=xp.diag(upper)
    informative=bool(xp.all(diag>ud))
    record={'rank':r,'queries':m,'orthogonality_error':orth,'eta':eta,
        'tail_frobenius':a,'reduced_solve_upper_frobenius':b,'covariance_error_frobenius_upper':bound,
        'relative_covariance_frobenius_bound':bound/(norm-bound) if norm>bound else None,
        'maximum_relative_marginal_bound':float(xp.max(ud/(diag-ud))) if informative else None,
        'qualification':'exact-arithmetic joint-prior bound; excludes prior/cross/action numerical errors, mean error and root projection'}
    return covariance,upper,record


def prior_basis(prior,rank=None,rtol=1e-7,xp=np):
    """Prior-only initial rank heuristic; the final output budget must pass.

    If rank is omitted, leave a prior Frobenius tail <=rtol/8 of the prior.
    This is not itself a posterior guarantee: assembly checks the residual
    and truncation budgets against the computed posterior afterwards.
    """
    automatic=rank is None;k=xp.asarray(prior,dtype='float64')
    if k.ndim!=2 or k.shape[0]!=k.shape[1] or not len(k) or not bool(xp.isfinite(k).all()) or not 0<rtol<1:
        raise ValueError('invalid prior or tolerance')
    eig,vec=xp.linalg.eigh((k+k.T)*.5)
    threshold=100*np.finfo(float).eps*len(k)*float(xp.max(xp.abs(eig)))
    if float(eig.min()) < -threshold:raise ValueError('prior indefinite beyond roundoff allowance')
    clipped=xp.maximum(eig,0)[::-1];vec=vec[:,::-1]
    if rank is None:
        tails=xp.concatenate([xp.sqrt(xp.cumsum(clipped[::-1]**2)[::-1]),xp.zeros(1)])
        norm=float(xp.linalg.norm(clipped))
        rank=int(xp.nonzero(tails<=rtol*norm/8)[0][0])
    if rank!=int(rank) or not 0<=rank<=len(k):raise ValueError('invalid rank')
    rank=int(rank);q=xp.ascontiguousarray(vec[:,:rank])
    tail=(vec[:,rank:]*clipped[rank:])@vec[:,rank:].T
    return q,tail,{'rank':rank,'selected_from_prior_only':True,'automatic':automatic,
        'negative_prior_eigenvalues':int((eig<0).sum()),'maximum_prior_tail_projection':max(0.,-float(eig.min())),
        'prior_roundoff_allowance':threshold,'prior_eigenvalues':clipped.tolist(),
        'qualification':'negative prior eigenvalues within roundoff allowance are clipped only for tail bound construction; output uses original symmetric prior'}


class ProjectedQueryPosterior:
    """Reduced basis solves; joint covariance and samples on the full panel."""
    def __init__(self,points,queries,operator,solve,analytic_action,mean_coefficients,offset,
                 noise_min=.02,rank=None,batch=4,rtol=1e-7,row_chunk=4096,progress=None):
        import cupy as cp
        from reproduction.query_posterior import principal_root
        from reproduction.query_error import HostQueryResidualAudit
        from reproduction.deflated_variance import FourierCorrectionCovariance
        from reproduction.analytic_posterior import timed
        if batch!=int(batch) or batch<1 or not np.isfinite(noise_min) or noise_min<=0:raise ValueError('positive batch and noise required')
        if operator.dtype!=np.dtype('float64') or operator.normalization!='analytic':raise ValueError('double analytic-normalized operator required')
        self.queries=cp.array(queries,dtype='d',copy=True)
        if self.queries.ndim!=2 or self.queries.shape[1]!=3 or not len(self.queries) or not bool(cp.isfinite(self.queries).all()):raise ValueError('invalid query panel')
        self.record={'seconds':{},'queries':len(self.queries),'batch':batch}
        self.mean,self.record['seconds']['mean']=timed(lambda:operator.evaluate(self.queries,mean_coefficients)[0]-offset)
        def prior():
            m=len(self.queries);norm2=(operator.ns**2).sum(0)
            rho=cp.where(norm2>0,operator.rho/cp.where(norm2>0,norm2,1),0)
            k=cp.empty((m,m),dtype='d')
            for start in range(0,m,64):
                diff=(self.queries[start:start+64,None]-self.queries[None]).reshape(-1,3)
                k[start:start+64]=operator.evaluate(diff,rho).reshape(-1,m)
            return (k+k.T)*.5
        self.prior,self.record['seconds']['prior']=timed(prior)
        (self.basis,self.tail,info),self.record['seconds']['basis']=timed(lambda:prior_basis(self.prior,rank,rtol,xp=cp))
        self.record['basis']=info;r=self.basis.shape[1];m=len(self.queries)
        def correction():
            if r==0:return cp.empty((m,0)),cp.empty((0,0)),cp.empty((0,0))
            self.audit=HostQueryResidualAudit(analytic_action,solve,noise_min,len(points),r,row_chunk=row_chunk)
            action=FourierCorrectionCovariance(points,self.queries,operator,self.audit,batch=batch)
            l=cp.empty((m,r),dtype='d')
            for start in range(0,r,batch):
                end=min(start+batch,r);l[:,start:end]=action(self.basis[:,start:end])
                if progress:progress(end,r)
            # H and U involve reduced coordinates, not point-query diagonals.
            cross,gram=self.audit._products(r)
            self.record['storage']=self.audit.record
            return l,(cross+cross.T)*.5,(gram+gram.T)*(.5/noise_min)
        (self.cross_action,self.weight_residual,self.solve_upper),self.record['seconds']['correction']=timed(correction)
        (covariance,self.upper,budget),self.record['seconds']['assembly']=timed(lambda:assemble_projected(
            self.prior,self.basis,self.cross_action,self.weight_residual,self.tail,self.solve_upper,xp=cp))
        self.record['budget']=budget
        self.record['budget_pass']=(budget['relative_covariance_frobenius_bound'] is not None and
            budget['relative_covariance_frobenius_bound']<=rtol and budget['maximum_relative_marginal_bound'] is not None and
            budget['maximum_relative_marginal_bound']<=rtol)
        (self.covariance,self.factor,psd),self.record['seconds']['root']=timed(lambda:principal_root(covariance))
        self.record['psd']=psd

    def clear_vectors(self):
        if hasattr(self,'audit'):self.audit.clear_vectors()

    def draw(self,count,seed=0,white=None):
        from reproduction.query_posterior import FixedQueryPosterior
        return FixedQueryPosterior.draw(self,count,seed,white)
