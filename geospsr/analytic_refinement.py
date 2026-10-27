"""Defect correction verified against an independent full analytic action.

The approximate solve may vary between calls; this is not fixed-preconditioner
CG. Convergence is judged against the original RHS using the analytic residual.
"""
import cupy as cp


def refine_analytic(action, approximate_solve, rhs, x0, rtol=1e-8, max_corrections=8):
    if rtol<=0 or not isinstance(max_corrections,int) or max_corrections<0:
        raise ValueError('need positive rtol and nonnegative integer correction limit')
    b=cp.asarray(rhs);x=cp.asarray(x0)
    if b.ndim!=2 or not b.shape[0] or not b.shape[1] or x.shape!=b.shape or cp.iscomplexobj(b) or cp.iscomplexobj(x):
        raise ValueError('rhs/x0 must be equally shaped real nonempty (N,R) arrays')
    if not bool(cp.isfinite(b).all() and cp.isfinite(x).all()):raise ValueError('nonfinite input')
    b=cp.asarray(b,dtype='d');x=cp.array(x,dtype='d',copy=True)
    norm=cp.linalg.norm(b,axis=0);denom=cp.where(norm>0,norm,1.)
    history=[];inners=[]
    for step in range(max_corrections+1):
        residual=b-action(x)
        relative=cp.linalg.norm(residual,axis=0)/denom
        if not bool(cp.isfinite(relative).all()):raise ArithmeticError('nonfinite analytic residual')
        history.append(cp.asnumpy(relative).tolist())
        if bool(cp.all(relative<=rtol)):
            return x,{'converged':True,'corrections':step,'analytic_actions':step+1,
                      'relative_residual':history[-1],'residual_history':history,'inner_solves':inners}
        if step==max_corrections:break
        delta,info=approximate_solve(residual)
        delta=cp.asarray(delta)
        if delta.shape!=x.shape or cp.iscomplexobj(delta) or not bool(cp.isfinite(delta).all()):raise ArithmeticError('invalid approximate correction')
        inners.append(info)
        x+=delta.astype('d')
    return x,{'converged':False,'corrections':max_corrections,'analytic_actions':max_corrections+1,
              'relative_residual':history[-1],'residual_history':history,'inner_solves':inners}
