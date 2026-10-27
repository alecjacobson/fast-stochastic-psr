"""Capped defect correction with an independently supplied verification action."""
import numpy as np


def refine_solution(solve_correction, verification_action, rhs, initial,
                    target_rtol=2e-5, max_corrections=2, xp=np):
    """Compute defects in float64 and retain weights at the initial precision.

    solve_correction(defect_at_storage_precision) returns (correction, metadata).
    The verification action must represent the SAME target system; this routine
    cannot establish that identity or certify the verifier's numerical error.
    Only the verification residual controls acceptance. target_rtol is the
    actual threshold, without an implicit tolerance multiplier. Zero RHS
    columns require zero residual. No mutation of the initial weights or RHS.
    """
    rhs=xp.asarray(rhs);weights=xp.array(initial,copy=True)
    if (rhs.ndim!=2 or rhs.shape!=weights.shape
            or rhs.dtype not in (np.dtype('float32'),np.dtype('float64'))
            or weights.dtype not in (np.dtype('float32'),np.dtype('float64'))):
        raise ValueError('expected matching real (N,R) RHS and weights')
    if not 0<target_rtol<1 or max_corrections<0 or max_corrections!=int(max_corrections):
        raise ValueError('invalid tolerance or correction count')
    rhs64=rhs.astype('float64');norm=xp.linalg.norm(rhs64,axis=0)
    history=[];subsolves=[]
    for iteration in range(max_corrections+1):
        applied=verification_action(weights.astype('float64'))
        if applied.shape!=rhs.shape or applied.dtype!=np.dtype('float64'):
            raise ValueError('verification must return matching float64 output')
        residual=rhs64-applied;residual_norm=xp.linalg.norm(residual,axis=0)
        if not bool(xp.all(xp.isfinite(residual_norm))):
            raise FloatingPointError('nonfinite verification residual')
        relative=residual_norm/xp.where(norm>0,norm,1)
        history.append(relative.tolist())
        passed=bool(xp.all(residual_norm<=target_rtol*norm))
        if passed or iteration==max_corrections:
            break
        correction,metadata=solve_correction(residual.astype(weights.dtype))
        if correction.shape!=weights.shape or correction.dtype!=weights.dtype:
            raise ValueError('correction changed storage shape or precision')
        weights=(weights.astype('float64')+correction.astype('float64')).astype(weights.dtype)
        subsolves.append(metadata)
    return weights,{'converged':passed,'corrections':len(subsolves),'relative_residual_history':history,
                    'target_rtol':target_rtol,'correction_solves':subsolves,
                    'storage_dtype':weights.dtype.name,'verification_dtype':'float64'}
