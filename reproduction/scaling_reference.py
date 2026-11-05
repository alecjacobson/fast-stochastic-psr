"""Offline analytic reference using mixed-precision iterative refinement.

The float32 production factor is only a preconditioner. Residuals use the full
float64 analytic kernel; corrected bilinear moments carry explicit residual
bounds using A >= noise I. This avoids a second, float64 N-by-N factor.
"""
import numpy as np


def refined_moments(solve, action, cross, normals, prior, noise, corrections=4):
    import cupy as cp
    c=cp.asarray(cross,dtype='d');y=cp.asarray(normals,dtype='d');n=len(y)
    rhs=cp.concatenate((y,c.transpose(1,0,2).reshape(n,-1)),axis=1)
    z=solve(rhs).astype('d');history=[]
    for step in range(corrections+1):
        residual=rhs-action(z)
        relative=cp.linalg.norm(residual,axis=0)/cp.maximum(cp.linalg.norm(rhs,axis=0),1e-300)
        history.append(float(relative.max()))
        if history[-1]<1e-10 or step==corrections:break
        z+=solve(residual)
    rz=cp.linalg.norm(residual,axis=0)
    alpha=z[:,:3];za=z[:,3:].reshape(n,-1,3)
    rc=residual[:,3:].reshape(n,-1,3)
    cc=c.transpose(1,0,2)
    mean=(cc*alpha[:,None,:]+za*residual[:,:3,None].transpose(0,2,1)).sum((0,2))
    variance=prior-(cc*za+za*rc).sum((0,2))
    vb=(rz[3:]**2).reshape(-1,3).sum(1)/noise
    mb=(rz[3:].reshape(-1,3)*rz[:3]).sum(1)/noise
    return cp.asnumpy(mean),cp.asnumpy(variance),dict(
        relative_residual_history=history,mean_error_bound=cp.asnumpy(mb).tolist(),
        variance_error_bound=cp.asnumpy(vb).tolist(),
        reference='full analytic float64 residuals; float32 factor preconditioner')
