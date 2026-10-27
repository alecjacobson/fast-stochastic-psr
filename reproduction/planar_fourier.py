"""Exact finite-Fourier contraction for a small number of axis-aligned planes.

Only query geometry is specialized. No Fourier modes or query coordinates are
dropped. Roundoff/NUFFT error remains; off-plane inputs are never snapped.
"""
import cupy as cp
import cufinufft
import numpy as np


class PlanarFourierPlan:
    """Generic-plan compatible execute for explicit (batch,...) arrays.

    kind1 maps (batch,queries) to (batch,nx,ny,nz), with negative phase;
    kind2 maps the latter to (batch,queries), with positive phase. Plane
    grouping uses exact equality after the same precision cast as op.plan.
    Reject more than max_planes rather than approximating general geometry.
    """
    def __init__(self, kind, operator, points, n_trans, axis=1, max_planes=8, upsampfac=2.0):
        if kind not in (1,2) or axis not in (0,1,2):
            raise ValueError('require kind1/2 and axis0/1/2')
        if n_trans != int(n_trans) or n_trans < 1 or max_planes != int(max_planes) or max_planes < 1:
            raise ValueError('positive integer batch and plane limit required')
        self.kind,self.count,self.axis=kind,int(n_trans),axis
        self.shape=tuple(operator.shape);self.cdtype=operator.cdtype
        if len(self.shape)!=3 or any(n<1 for n in self.shape):
            raise ValueError('require three nonempty Fourier axes')
        self.points=cp.array(points,dtype=operator.dtype,copy=True)
        if self.points.ndim!=2 or self.points.shape[1]!=3 or not len(self.points) or not bool(cp.isfinite(self.points).all()):
            raise ValueError('expected finite nonempty (count,3) query points')
        heights=cp.unique(self.points[:,axis])
        if len(heights)>max_planes:
            raise ValueError('query set exceeds explicit plane limit')
        horizontal=[i for i in range(3) if i!=axis]
        self.permutation=[0]+[i+1 for i in horizontal]+[axis+1]
        self.inverse_permutation=np.argsort(self.permutation).tolist()
        self.plane_shape=tuple(self.shape[i] for i in horizontal)
        freq=cp.arange(-(self.shape[axis]//2),self.shape[axis]-self.shape[axis]//2,dtype=operator.dtype)
        self.groups=[]
        for height in heights:
            indices=cp.where(self.points[:,axis]==height)[0]
            coords=[cp.ascontiguousarray(self.points[indices,i]) for i in horizontal]
            plan=cufinufft.Plan(kind,self.plane_shape,n_trans=self.count,eps=operator.eps,
                               isign=-1 if kind==1 else 1,dtype=operator.cdtype.name,
                               upsampfac=upsampfac)
            plan.setpts(*coords)
            phase=cp.asarray(cp.exp(1j*freq*height),dtype=operator.cdtype)
            self.groups.append((indices,coords,phase,plan))

    def execute(self, values):
        values=cp.asarray(values,dtype=self.cdtype)
        expected=(self.count,len(self.points)) if self.kind==1 else (self.count,*self.shape)
        if values.shape!=expected:
            raise ValueError(f'expected explicit batched array {expected}')
        if self.kind==2:
            moved=cp.ascontiguousarray(values.transpose(self.permutation))
            output=cp.empty((self.count,len(self.points)),dtype=self.cdtype)
            for indices,coords,phase,plan in self.groups:
                modes=cp.ascontiguousarray(cp.tensordot(moved,phase,axes=([-1],[0])))
                output[:,indices]=plan.execute(modes).reshape(self.count,len(indices))
            return output
        moved=cp.zeros((self.count,*self.plane_shape,self.shape[self.axis]),dtype=self.cdtype)
        for indices,coords,phase,plan in self.groups:
            modes=plan.execute(cp.ascontiguousarray(values[:,indices])).reshape(self.count,*self.plane_shape)
            moved+=modes[...,None]*phase.conj()
        return cp.ascontiguousarray(moved.transpose(self.inverse_permutation))
