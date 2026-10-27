"""Whitened correction-factor projection for marginal variance, not events."""
import numpy as np


class FactorDeflatedMoments:
    """diag(H H.T) = row_sum((H V)^2) + E[(H xi-H V V.T xi)^2].

    V must be orthonormal and chosen independently of evaluation xi. The same
    xi must generate each supplied H xi. Numerical factor errors are separate.
    """
    def __init__(self, basis, factor_basis, xp=np):
        self.xp = xp
        if xp.iscomplexobj(basis) or xp.iscomplexobj(factor_basis):
            raise ValueError('require real basis and factor action')
        v = xp.array(basis, dtype='float64', copy=True)
        u = xp.array(factor_basis, dtype='float64', copy=True)
        if v.ndim != 2 or u.ndim != 2 or not len(v) or not len(u) or v.shape[1] > len(v) or u.shape[1] != v.shape[1]:
            raise ValueError('expected (latent,rank) basis and (queries,rank) action')
        if not bool(xp.isfinite(v).all()) or not bool(xp.isfinite(u).all()):
            raise ValueError('nonfinite basis/action')
        error = float(xp.linalg.norm(v.T @ v - xp.eye(v.shape[1])))
        if error > 1e-8:
            raise ValueError('basis must be orthonormal at double precision')
        self.basis, self.factor_basis = v, u
        self.deterministic = xp.sum(u*u, axis=1)
        self.sum_squares = xp.zeros(len(u), dtype='float64')
        self.count = 0
        self.record = {'rank':v.shape[1], 'latent_size':len(v), 'queries':len(u),
                       'orthogonality_error':error,
                       'retained_bytes':v.nbytes+u.nbytes+self.deterministic.nbytes+self.sum_squares.nbytes}

    def update(self, values, standard_normal):
        xp = self.xp
        if xp.iscomplexobj(values) or xp.iscomplexobj(standard_normal):
            raise ValueError('require real paired fields and standard normals')
        values = xp.asarray(values, dtype='float64')
        xi = xp.asarray(standard_normal, dtype='float64')
        if values.ndim != 2 or not len(values) or values.shape[1] != len(self.factor_basis) or xi.shape != (len(self.basis),len(values)):
            raise ValueError('expected paired (draw,query) fields and (latent,draw) normals')
        if not bool(xp.isfinite(values).all()) or not bool(xp.isfinite(xi).all()):
            raise ValueError('nonfinite fields/normals')
        residual = values - (xi.T @ self.basis) @ self.factor_basis.T
        squares = xp.sum(residual*residual, axis=0)
        if not bool(xp.isfinite(squares).all()):
            raise ValueError('second moment overflow')
        self.sum_squares += squares
        self.count += len(values)

    def diagonal(self):
        if not self.count:
            raise ValueError('no evaluation draws')
        return self.deterministic + self.sum_squares/self.count


class FourierCorrectionFactor:
    """H=C L3^-T and H.T=L3^-1 C.T via NUFFT and one triangular solve.

    Latent rows use observation-major (observation,component) order. L3 is
    three independent copies of the supplied N by N Cholesky factor.
    """
    def __init__(self, points, queries, operator, cholesky, batch=4, coefficients=None):
        import cupy as cp
        if batch != int(batch) or batch < 1:
            raise ValueError('positive integer batch required')
        self.op, self.batch = operator, int(batch)
        self.points = cp.array(points, dtype=operator.dtype, copy=True)
        self.queries = cp.array(queries, dtype=operator.dtype, copy=True)
        for a in [self.points,self.queries]:
            if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not bool(cp.isfinite(a).all()):
                raise ValueError('expected finite nonempty (count,3) positions')
        self.chol = cp.asarray(cholesky)
        if self.chol.shape != (len(self.points),len(self.points)) or self.chol.dtype != operator.dtype:
            raise ValueError('Cholesky shape/precision mismatch')
        if coefficients is None:
            coefficients = operator.coefficients if hasattr(operator,'coefficients') else -1j*operator.multiplier
        self.coeff = cp.array(coefficients, dtype=operator.cdtype, copy=True)
        if self.coeff.shape != (3,*operator.shape) or not bool(cp.isfinite(self.coeff).all()):
            raise ValueError('invalid Fourier coefficients')
        self.plans = {}

    def prepare(self, count):
        if count not in self.plans:
            op = self.op
            self.plans[count] = (op.plan(1,self.queries,count),op.plan(2,self.points,3*count),
                                 op.plan(1,self.points,3*count),op.plan(2,self.queries,count))
        return self.plans[count]

    def _apply(self, values, adjoint):
        import cupy as cp
        import cupyx.scipy.linalg as cla
        n, m, op = len(self.points), len(self.queries), self.op
        rows, out_rows = (m,3*n) if adjoint else (3*n,m)
        if cp.iscomplexobj(values):
            raise ValueError('require real inputs')
        values = cp.asarray(values, dtype=op.dtype)
        if values.ndim != 2 or len(values) != rows or not bool(cp.isfinite(values).all()):
            raise ValueError('invalid input shape or nonfinite values')
        result = cp.empty((out_rows,values.shape[1]), dtype=op.dtype)
        for start in range(0,values.shape[1],self.batch):
            v = values[:,start:start+self.batch]
            p = v.shape[1]
            qf, xb, xf, qb = self.prepare(p)
            if adjoint:
                modes = qf.execute(cp.ascontiguousarray(v.T,dtype=op.cdtype))
                modes = cp.ascontiguousarray((modes[:,None]*self.coeff.conj()[None]).reshape(3*p,*op.shape))
                rhs = cp.ascontiguousarray(xb.execute(modes).real.T)
                z = cla.solve_triangular(self.chol,rhs,lower=True)
                result[:,start:start+p] = z.reshape(n,p,3).transpose(0,2,1).reshape(3*n,p)
            else:
                rhs = cp.ascontiguousarray(v.reshape(n,3,p).transpose(0,2,1).reshape(n,3*p))
                weights = cla.solve_triangular(self.chol.T,rhs,lower=False)
                modes = xf.execute(cp.ascontiguousarray(weights.T,dtype=op.cdtype)).reshape(p,3,*op.shape)
                coeff = cp.ascontiguousarray((modes*self.coeff[None]).sum(1))
                result[:,start:start+p] = qb.execute(coeff).real.T
        return result

    def forward(self, values):
        return self._apply(values, False)

    def adjoint(self, values):
        return self._apply(values, True)
