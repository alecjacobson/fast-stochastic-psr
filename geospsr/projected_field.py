"""Compressed joint sampling of finite Fourier posterior functions.

This prepares/draws from supplied reduced coupling coefficients. Computing
those coefficients with NUFFTs and validating their approximation to a target
posterior are separate responsibilities. The optional array backend is not a
performance claim. Mean and deterministic isovalue centering are external.
"""
import numpy as np


class CompressedJointField:
    """Draw centered scalar Fourier coefficients from a compressed joint GP.

    Flattened full-cube scalar variances s have shape (F,). cross_coefficients
    and lift_coefficients have shape (r,F), representing L(q)=C(q)W and B(q).
    projected_covariance is G=sum W_d.T A W_d, shape (r,r). Complex source
    coefficients use evaluation Re(sum coeff(n)*exp(+i*n.q)).

    With c=L/sqrt(s), H=G-Re(c c*) must be PSD. Independent complex white z
    (real and imaginary parts each variance one) and real white e give
      t=Re(z c*)+e sqrt(H).T, coefficients=z sqrt(s)-t B_coeff.
    This retains the joint coupling between prior fields and compressed data.
    The F-by-r caches and draw workspace are explicit; source conditioning,
    field evaluation, target error bounds and numerical validation are extra.
    """
    def __init__(self,scalar_variance,cross_coefficients,lift_coefficients,
                 projected_covariance,frequency_chunk=65536,max_cache_bytes=2**30,xp=np):
        self.xp=xp
        if (not np.isfinite(frequency_chunk) or frequency_chunk!=int(frequency_chunk) or frequency_chunk<1
            or not np.isfinite(max_cache_bytes) or max_cache_bytes<0):
            raise ValueError('positive frequency chunk and nonnegative cache cap required')
        self.chunk=int(frequency_chunk)
        s=xp.asarray(scalar_variance);l=xp.asarray(cross_coefficients)
        b=xp.asarray(lift_coefficients);g=xp.asarray(projected_covariance)
        if (s.ndim!=1 or not len(s) or l.ndim!=2 or b.shape!=l.shape
            or l.shape[1]!=len(s) or g.shape!=(l.shape[0],l.shape[0])):
            raise ValueError('expected s(F), cross/lift(r,F), G(r,r)')
        if (s.dtype!=np.dtype('float64') or g.dtype!=np.dtype('float64')
            or l.dtype!=np.dtype('complex128') or b.dtype!=np.dtype('complex128')):
            raise ValueError('double real/complex coefficients required')
        if not all(bool(xp.isfinite(v).all()) for v in (s,l,b,g)) or bool(xp.any(s<0)):
            raise ValueError('finite coefficients and nonnegative scalar variances required')
        rank,modes=l.shape;cache_bytes=2*l.nbytes+s.nbytes+4*g.nbytes
        if cache_bytes>max_cache_bytes:raise MemoryError(f'field cache needs {cache_bytes} bytes, cap {max_cache_bytes}')
        zero=s==0
        if bool(xp.any(l[:,zero]!=0)) or bool(xp.any(b[:,zero]!=0)):
            raise ValueError('cross/lift coefficients must vanish on zero-variance modes')
        scale=float(xp.linalg.norm(g));asym=float(xp.linalg.norm(g-g.T))/max(scale,1e-300)
        if asym>1e-10:raise ValueError('projected data covariance is materially asymmetric')
        self.rank,self.modes=rank,modes
        self.sigma=xp.sqrt(s)
        self.lift=xp.array(b,copy=True)
        self.projected_covariance=(g+g.T)*.5
        self.whitened=xp.empty_like(l)
        self.scalar_gram=xp.zeros_like(g)
        for start in range(0,modes,self.chunk):
            end=min(start+self.chunk,modes);sigma=self.sigma[start:end]
            self.whitened[:,start:end]=l[:,start:end]/xp.where(sigma>0,sigma,1)[None,:]
            block=self.whitened[:,start:end]
            self.scalar_gram+=(block@block.conj().T).real
        h=self.projected_covariance-self.scalar_gram;self.remainder=(h+h.T)*.5
        if rank:
            eig,vec=xp.linalg.eigh(self.remainder)
            component_scale=(float(xp.max(xp.abs(xp.linalg.eigvalsh(self.projected_covariance))))
                             +float(xp.max(xp.abs(xp.linalg.eigvalsh(self.scalar_gram)))))
            allowance=100*np.finfo(float).eps*rank*component_scale
            minimum=float(eig.min())
            if minimum < -allowance:
                raise ValueError(f'indefinite compressed remainder: {minimum}, roundoff allowance {allowance}')
            self.remainder_root=(vec*xp.sqrt(xp.maximum(eig,0)))@vec.T
            projection=float(xp.linalg.norm(self.remainder_root@self.remainder_root.T-self.remainder))
        else:
            minimum=allowance=projection=0.;self.remainder_root=xp.empty((0,0),dtype='float64')
        self.record={'rank':rank,'modes':modes,'frequency_chunk':self.chunk,
            'retained_cache_bytes':cache_bytes,'maximum_cache_bytes':max_cache_bytes,
            'projected_covariance_relative_asymmetry':asym,'remainder_minimum_eigenvalue':minimum,
            'remainder_roundoff_allowance':allowance,'remainder_root_covariance_frobenius_error':projection,
            'maximum_remainder_projection':max(0.,-minimum),
            'roundoff_qualification':'fixed heuristic allowance based on both subtracted matrices; excludes kernel/transform/action errors and is not an interval bound',
            'scope':'centered coherent functions from supplied reduced coupling; no target-posterior quality or speedup implied'}

    def draw(self,count,seed=0,white=None,extra_white=None,max_draw_bytes=2**30):
        """Return (count,F) coefficients; reuse these exact draws for new queries.

        Caller-supplied white arrays must be supplied together and independently
        distributed as specified above for probabilistic claims. They are never
        mutated. The workspace cap accounts for explicit arrays conservatively,
        not array-backend allocator pools or GEMM workspaces.
        """
        xp=self.xp
        if count!=int(count) or count<1:raise ValueError('positive integer draw count required')
        count=int(count)
        if not np.isfinite(max_draw_bytes) or max_draw_bytes<0:raise ValueError('finite nonnegative draw cap required')
        needed=3*count*self.modes*16+2*count*self.rank*8+self.rank*min(self.chunk,self.modes)*16
        if needed>max_draw_bytes:raise MemoryError(f'explicit draw workspace estimate {needed} exceeds cap {max_draw_bytes}')
        if (white is None)!=(extra_white is None):raise ValueError('supply both white arrays or neither')
        if white is None:
            rng=xp.random.RandomState(seed)
            white=rng.standard_normal((count,self.modes)).astype('complex128')
            white.imag=-rng.standard_normal((count,self.modes))
            extra_white=rng.standard_normal((count,self.rank))
        else:
            white=xp.asarray(white);extra_white=xp.asarray(extra_white)
            if (white.shape!=(count,self.modes) or extra_white.shape!=(count,self.rank)
                or white.dtype!=np.dtype('complex128') or extra_white.dtype!=np.dtype('float64')
                or not bool(xp.isfinite(white).all()) or not bool(xp.isfinite(extra_white).all())):
                raise ValueError('invalid white-array dimensions, precision or values')
        t=extra_white@self.remainder_root.T
        for start in range(0,self.modes,self.chunk):
            end=min(start+self.chunk,self.modes)
            t+=(white[:,start:end]@self.whitened[:,start:end].conj().T).real
        coefficients=white*self.sigma[None,:]
        for start in range(0,self.modes,self.chunk):
            end=min(start+self.chunk,self.modes)
            coefficients[:,start:end]-=t@self.lift[:,start:end]
        return coefficients
