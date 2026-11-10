"""Input-resolution quality controls for dense partial scans (issue 124)."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from geospsr.nonperiodic import FreeSpacePoissonMatern
from geospsr.scalar_contacts import ScalarContactUpdate
from reproduction.partial_occupancy import load, farthest_contacts
from reproduction.partial_quality import occupancy, roots
from reproduction.stanford_partial_data import CACHE, digest

OUT = Path('reproduction/results/observed_fidelity')
OLD = Path('reproduction/results/quality_controls')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sources():
    folder = OUT/'sources'; folder.mkdir(exist_ok=True)
    result = {}
    for p in [__file__, 'reproduction/partial_quality.py', 'reproduction/partial_quality_large.py',
              'reproduction/partial_direct_gpu.py', 'geospsr/nonperiodic_dense_gpu.py',
              'geospsr/nonperiodic.py', 'geospsr/scalar_contacts.py']:
        h = digest(p); (folder/(h+'.py')).write_bytes(Path(p).read_bytes()); result[p] = h
    return result


def prepare():
    for i, group in enumerate(['validation', 'test_side']):
        path = OUT/(group+'-panel.npz')
        if path.exists(): raise FileExistsError(path)
        a = np.load(CACHE/(group+'.npy'))
        contacts = farthest_contacts(a[:, :3], 2048)
        spacing = cKDTree(a[:, :3]).query(a[:, :3], k=2)[0][:, 1]
        # Exclude every potential contact from input-fit probes.
        eligible = np.flatnonzero(cKDTree(contacts).query(a[:, :3])[0] > 1e-12)
        ids = np.random.default_rng(17124+i).choice(eligible, 512, replace=False)
        panel = dict(np.load(OLD/(group+'-panel.npz')))
        panel.update(contacts=contacts, fit_indices=ids, fit_points=a[ids, :3], fit_normals=a[ids, 3:],
                     fit_offsets=np.linspace(-.03, .03, 65), spacing=np.median(spacing))
        np.savez_compressed(path, **panel)
        write(path.with_suffix('.json'), dict(group=group, panel_sha256=digest(path),
            scan_sha256=digest(CACHE/(group+'.npy')), spacing_quantiles=np.quantile(spacing, [.1,.5,.9,.99]).tolist(),
            protocol_sha256=digest(OUT/'PROTOCOL.md')))


def fidelity(values, gradients, panel):
    """Closest interpolated root and oriented normal errors; no hidden clipping."""
    t = panel['fit_offsets']; values = np.asarray(values).reshape(-1, len(t))
    gradients = np.asarray(gradients)
    if not np.isfinite(values).all() or not np.isfinite(gradients).all():
        raise ValueError('nonfinite fidelity fields')
    denominator = np.diff(values, axis=1)
    fraction = np.divide(-values[:, :-1], denominator, out=np.zeros_like(denominator), where=denominator != 0)
    root = t[:-1]+fraction*np.diff(t)
    error = np.min(np.where(values[:, :-1]*values[:, 1:] <= 0, abs(root), np.inf), axis=1)
    missing = ~np.isfinite(error); error[missing] = max(abs(t))
    length = np.linalg.norm(gradients, axis=1)
    cosine = np.divide(np.sum(gradients*panel['fit_normals'], axis=1), length,
                       out=np.full(len(length), -1.), where=length > 0)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    spacing = float(panel['spacing'])
    result = dict(count=len(error), spacing=spacing, median=float(np.median(error)),
                  p95=float(np.quantile(error, .95)), missing_fraction=float(missing.mean()),
                  normal_median_degrees=float(np.median(angle)), normal_p95_degrees=float(np.quantile(angle, .95)))
    checks = dict(median=result['median'] <= spacing, p95=result['p95'] <= 2*spacing,
                  missing=result['missing_fraction'] <= .01, normal_median=result['normal_median_degrees'] <= 15,
                  normal_p95=result['normal_p95_degrees'] <= 40)
    result.update(checks=checks, passes=all(checks.values()))
    return result, error, angle


class FidelityFamily:
    """Independent full-kernel posterior family, with explicit scalar noise."""
    def __init__(self, model, x, y, contacts, noise, configurations):
        import cupy as cp
        from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
        from reproduction.partial_direct_gpu import cross_rhs
        self.model, self.x, self.contacts = model, cp.asarray(x), contacts
        self.configurations = configurations
        self.system = GPUDenseMaternSystem(model, x, noise, dtype='float64')
        self.alpha = cp.asarray(self.system.solve(y))
        b = cross_rhs(model, contacts, x); self.zc = cp.asarray(self.system.solve(cp.asnumpy(b)))
        width = len(contacts)
        m = cp.asnumpy(sum(b[:,d*width:(d+1)*width].T@self.alpha[:,d] for d in range(3)))
        red = sum(b[:,d*width:(d+1)*width].T@self.zc[:,d*width:(d+1)*width] for d in range(3))
        cov = model.scalar_covariance(contacts, contacts)-cp.asnumpy((red+red.T)*.5)
        self.updates = []; self.coefficients = []; self.weights = []
        for c, scalar_variance in configurations:
            update = ScalarContactUpdate(m[:c], cov[:c,:c], np.zeros(c), scalar_variance)
            self.updates.append(update); w = cp.asarray(update.weights)
            self.coefficients.append(self.alpha-cp.column_stack([self.zc[:,d*width:d*width+c]@w for d in range(3)]))
            self.weights.append(np.pad(update.weights, (0,width-c)))
        self.weights = np.column_stack(self.weights)

    def mean(self, q, batch=192):
        import cupy as cp
        from reproduction.partial_direct_gpu import cross_rhs
        out = np.empty((len(q),len(self.configurations)))
        coeff = [cp.column_stack([a[:,d] for a in self.coefficients]) for d in range(3)]
        for first in range(0,len(q),batch):
            qq=q[first:first+batch]; m=len(qq); b=cross_rhs(self.model,qq,self.x)
            out[first:first+m] = cp.asnumpy(sum(b[:,d*m:(d+1)*m].T@coeff[d] for d in range(3)))
            out[first:first+m] += self.model.scalar_covariance(qq,self.contacts)@self.weights
        return out

    def predict(self, q, batch=96):
        import cupy as cp
        from cupyx.scipy.linalg import solve_triangular
        from reproduction.partial_direct_gpu import cross_rhs
        means=np.empty((len(q),len(self.configurations))); variances=np.empty_like(means); nc=len(self.contacts)
        for first in range(0,len(q),batch):
            qq=q[first:first+batch]; m=len(qq); b=cross_rhs(self.model,qq,self.x)
            bm=cp.asnumpy(sum(b[:,d*m:(d+1)*m].T@self.alpha[:,d] for d in range(3)))
            w=solve_triangular(self.system.factor,b,lower=True,check_finite=False)
            bv=self.model.prior_variance-cp.asnumpy(cp.sum(w*w,axis=0).reshape(3,m).sum(axis=0))
            cross=self.model.scalar_covariance(qq,self.contacts)-cp.asnumpy(sum(b[:,d*m:(d+1)*m].T@self.zc[:,d*nc:(d+1)*nc] for d in range(3)))
            for k,(c,_) in enumerate(self.configurations):
                means[first:first+m,k],variances[first:first+m,k]=self.updates[k].apply(bm,bv,cross[:,:c])
        return means,variances


def screen(args):
    import cupy as cp
    from reproduction.gpu_audit import GpuProcessAudit
    path=OUT/f'screen-n{args.n}-ell{args.ell}-sigma{args.sigma}.json'
    if path.exists(): raise FileExistsError(path)
    panel=dict(np.load(OUT/'validation-panel.npz')); x,y,_,_=load('validation',args.n)
    model=FreeSpacePoissonMatern(args.ell,1/3)
    configurations=[(512,.0001)] if args.control else [(c,r*model.prior_variance) for c in [512,2048] for r in [1e-6,1e-4]]
    contacts=panel['contacts'][:max(c for c,_ in configurations)]
    source=sources();audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.5,cuda_device=cp.cuda.Device().id).start();start=perf_counter()
    record=dict(arguments=vars(args),source_sha256=source,panel_sha256=digest(OUT/'validation-panel.npz'),
                protocol_sha256=digest(OUT/'PROTOCOL.md'),backend='full float64 Cholesky')
    try:
        family=FidelityFamily(model,x,y,contacts,args.sigma**2,configurations)
        m,v=family.predict(panel['queries']);np.testing.assert_allclose(family.mean(panel['queries']),m,atol=1e-10,rtol=0)
        p=panel['fit_points'];normal=panel['fit_normals'];t=panel['fit_offsets']
        probes=p[:,None,:]+t[None,:,None]*normal[:,None,:]
        fit_mean=family.mean(probes.reshape(-1,3))
        delta=float(panel['spacing'])/8
        offsets=np.concatenate((np.eye(3),-np.eye(3)))*delta
        g=family.mean((p[:,None,:]+offsets[None,:,:]).reshape(-1,3)).reshape(len(p),6,-1)
        gradient=(g[:,:3]-g[:,3:])/(2*delta)
        reference_mean=family.mean(panel['probes'].reshape(-1,3))
        records=[];errors=[];angles=[]
        for k,(c,sv) in enumerate(configurations):
            fit,err,ang=fidelity(fit_mean[:,k],gradient[:,:,k],panel);errors.append(err);angles.append(ang)
            records.append(dict(contacts=c,scalar_variance=sv,scalar_variance_over_prior=sv/model.prior_variance,
                                fidelity=fit,occupancy=occupancy(m[:,k],v[:,k],panel),geometry=roots(reference_mean[:,k],panel)[0]))
        np.savez_compressed(path.with_suffix('.npz'),mean=m,variance=v,fit_mean=fit_mean,gradient=gradient,
            reference_mean=reference_mean,root_error=np.array(errors).T,normal_angle=np.array(angles).T,
            coefficients=cp.asnumpy(cp.stack(family.coefficients,axis=-1)),contact_weights=family.weights)
        record.update(status='completed',records=records,field_sha256=digest(path.with_suffix('.npz')))
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        record.update(seconds=perf_counter()-start,gpu_audit=audit.stop());write(path,record)
    print('SCREEN',args.ell,args.sigma,record['seconds'],[(r['contacts'],r['scalar_variance_over_prior'],r['fidelity']) for r in records],flush=True)


def select():
    rows=[]
    for path in sorted(OUT.glob('screen-n16384-*.json')):
        r=json.loads(path.read_text())
        if r['arguments']['control']:continue
        assert r['status']=='completed' and r['gpu_audit']['sampled_isolation_pass']
        assert digest(path.with_suffix('.npz'))==r['field_sha256']
        for k,sub in enumerate(r['records']):
            rows.append(dict(ell=r['arguments']['ell'],sigma=r['arguments']['sigma'],source=str(path),column=k,**sub))
    if len(rows)!=32:raise ValueError(f'incomplete screen {len(rows)}')
    passing=sorted([r for r in rows if r['fidelity']['passes']],key=lambda r:(r['fidelity']['p95'],r['fidelity']['normal_median_degrees']))
    path=OUT/'selection.json'
    if path.exists():raise FileExistsError(path)
    write(path,dict(protocol_sha256=digest(OUT/'PROTOCOL.md'),rows=rows,promotion_order=passing[:3]))
    print('PASSING',len(passing),'PROMOTION',[(r['ell'],r['sigma'],r['contacts'],r['scalar_variance_over_prior'],r['fidelity']) for r in passing[:3]],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','screen','select'])
    p.add_argument('--n',type=int,default=16384);p.add_argument('--ell',type=float);p.add_argument('--sigma',type=float)
    p.add_argument('--control',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='select':select()
    else:screen(a)
