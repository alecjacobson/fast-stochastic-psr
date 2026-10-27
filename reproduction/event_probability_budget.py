"""Finite-grid sign-event bounds from an ideal PSD covariance-error budget.

Assume S_approx - S_exact = E >= 0, E <= U, and a deterministic bound d
on the absolute mean error. Then an existence coupling is Y=X+delta+eta,
where eta is independent Gaussian with covariance E and |delta_i|<=d_i.
These formulas bound differences in event laws, not disagreements under an
arbitrary coupling. Floating-point evaluation is not an interval certificate;
the caller must separately justify the PSD/mean assumptions and numerical,
model and spatial-discretization budgets.
"""
import numpy as np
from scipy.special import log_ndtr


def sign_probability_budget(mean, variance, error_diagonal, mean_error=0., z_grid=None,
                            method='approximate_marginal'):
    """Bound each sign mismatch in the existence coupling.

    Let t=z sqrt(u) and a=t+d. A sign change implies |Y|<=a or |eta|>t.
    The known approximate marginal Y~N(mu,s) has density on [-a,a] at most
      exp(-max(|mu|-a,0)^2/(2s))/sqrt(2*pi*s).
    This does not require eta and Y to be independent, or s>u. More generally
    it applies to any Gaussian coupling whose centered difference has marginal
    variance <=u, including factor-difference budgets without PSD ordering.

    The older 'lower_exact_density' option instead uses |X|<=a, lower exact
    variance s-u>0 and unknown exact mean within d of mu. Its denominator is
    sqrt(2*pi*(s-u)) and distance is max(|mu|-t-2d,0). It is included to replay
    the initial derivation and requires the ideal PSD covariance-error premise.
    A grid minimum preserves the upper bound. Duplicate queries require no
    independence assumption. Float results do not have outward rounding.
    """
    mu,s,u,d=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in
                                 (mean,variance,error_diagonal,mean_error)])
    if not all(np.isfinite(v).all() for v in (mu,s,u,d)) or np.any(s<0) or np.any(u<0) or np.any(d<0):
        raise ValueError('finite means and nonnegative finite variances/error bounds required')
    z=np.geomspace(.5,30.,128) if z_grid is None else np.asarray(z_grid,dtype=float)
    if z.ndim!=1 or not len(z) or not np.isfinite(z).all() or np.any(z<=0):
        raise ValueError('threshold grid must contain positive finite values')
    if method not in ('approximate_marginal','lower_exact_density'):
        raise ValueError('unknown sign-budget method')
    shape=mu.shape;mu,s,u,d=[v.ravel() for v in (mu,s,u,d)]
    bound=np.ones(len(mu));chosen=np.full(len(mu),np.nan)
    # E_ii=0 means eta_i is identically zero, even if other coordinates vary.
    same=(u==0)&(d==0);bound[same]=0
    deterministic=(s==0)&(u==0)&(np.abs(mu)>d);bound[deterministic]=0
    valid=((s>0) if method=='approximate_marginal' else (s>u))&~same
    if np.any(valid):
        sv,uv,dv,mv=s[valid],u[valid],d[valid],mu[valid]
        t=z[:,None]*np.sqrt(uv)
        a=t+dv
        extra=dv if method=='lower_exact_density' else 0.
        density_variance=sv-uv if method=='lower_exact_density' else sv
        distance=np.maximum(np.abs(mv)-a-extra,0)
        # Work in log space so very small variance/tail inputs do not overflow.
        with np.errstate(divide='ignore',over='ignore',under='ignore'):
            log_density_mass=(np.log(2.)+np.log(a)-.5*(np.log(2*np.pi)+np.log(density_variance))
                              -.5*(distance/np.sqrt(sv))**2)
            log_tail=np.where(uv[None,:]>0,np.log(2.)+log_ndtr(-z[:,None]),-np.inf)
            log_bound=np.logaddexp(log_density_mass,log_tail)
            best=np.argmin(log_bound,axis=0)
            bound[valid]=np.exp(np.minimum(0,log_bound[best,np.arange(len(sv))]))
        chosen[valid]=z[best]
    return {'node_bound':bound.reshape(shape),'selected_z':chosen.reshape(shape)}


def prefix_probability_budget(node_bound):
    """Union bound for any sign-defined event on each prefix of the last axis."""
    b=np.asarray(node_bound,dtype=float)
    if b.ndim<1 or not np.isfinite(b).all() or np.any((b<0)|(b>1)):
        raise ValueError('node bounds must lie in [0,1]')
    return np.minimum(1,np.cumsum(b,axis=-1))


def scanning_score_intervals(probability, distance, probability_budget, mc_radius,
                             lower_threshold=.05, upper_threshold=.95):
    """Propagate law-error + simultaneous MC intervals to discrete ray scores.

    A score integrates 1{lower_threshold <= p <= upper_threshold} using
    trapezoidal weights on a fixed increasing distance grid. Coverage is
    inherited from the supplied simultaneous MC bound; it is not augmented
    by treating numerical budget values as independent random errors.
    """
    p=np.asarray(probability,dtype=float);x=np.asarray(distance,dtype=float)
    b=np.broadcast_to(np.asarray(probability_budget,dtype=float),p.shape)
    if (p.ndim!=2 or x.shape!=(p.shape[-1],) or len(x)<2 or not np.isfinite(x).all()
        or np.any(np.diff(x)<=0) or not np.isfinite(p).all() or np.any((p<0)|(p>1))
        or not np.isfinite(b).all() or np.any((b<0)|(b>1))
        or not np.isfinite(mc_radius) or not 0<=mc_radius<=1
        or not 0<=lower_threshold<upper_threshold<=1):
        raise ValueError('invalid probabilities, grid, thresholds or budgets')
    lo=np.maximum(0,p-b-mc_radius);hi=np.minimum(1,p+b+mc_radius)
    score=np.trapezoid(((p>=lower_threshold)&(p<=upper_threshold)).astype(float),x,axis=-1)
    lower=np.trapezoid(((lo>=lower_threshold)&(hi<=upper_threshold)).astype(float),x,axis=-1)
    upper=np.trapezoid(((lo<=upper_threshold)&(hi>=lower_threshold)).astype(float),x,axis=-1)
    winner=int(score.argmax())
    return {'probability_lower':lo,'probability_upper':hi,'score':score,
            'score_lower':lower,'score_upper':upper,'winner':winner,
            'winner_separated':bool(np.all(lower[winner]>np.delete(upper,winner)))}
