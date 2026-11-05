"""Cross-model Eagle mean/variance comparison; no cross-model accuracy claim.

Periodic baselines receive the same regional cache as the free-space methods.
The NUFFT hybrid additionally builds its established global Fourier variance
field. All production work is timed; dense reference checks are offline.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import resource
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import RegionalVarianceCache, compress_cross_covariance
from geospsr.regional_field import PolynomialMomentField
from reproduction.benchmark_hierarchical_variance import query_batches, save_field
from reproduction.poissonrecon_data import load_cloud


def probe_points(a):
    rng = np.random.default_rng(731)
    q = np.array([1., 0, 0])+a.halfwidth*rng.uniform(-1, 1, (64, 3))
    corners = np.array(list(__import__('itertools').product([-1., 1.], repeat=3)))
    return np.vstack((q, [1., 0, 0]+a.halfwidth*corners, [1., 0, 0]))


def assess(mean, variance, reference_mean, reference_variance, prior):
    valid = bool(np.isfinite(mean).all() and np.isfinite(variance).all() and np.all(variance > 0))
    if not valid:
        return dict(passes=False, positive=False)
    p = ndtr((mean[-1]-mean)/np.sqrt(variance))
    rp = ndtr((reference_mean[-1]-reference_mean)/np.sqrt(reference_variance))
    pe = float(np.max(abs(p-rp)))
    ve = float(np.max(abs(variance-reference_variance))/prior)
    return dict(passes=bool(pe <= 1e-3 and ve <= 1e-4), positive=True,
                probability_max_error=pe, variance_max_error_over_prior=ve)


def cholesky_in_place(matrix):
    """Factor a symmetric GPU matrix without redundant N-by-N copies."""
    import cupy as cp
    from cupy.cuda import cusolver, cublas, device
    factor=matrix if matrix.flags.f_contiguous else matrix.T
    n=len(factor);handle=device.get_cusolver_handle();fill=cublas.CUBLAS_FILL_MODE_LOWER
    if factor.dtype==cp.float32:
        potrf,size=cusolver.spotrf,cusolver.spotrf_bufferSize
    else:
        potrf,size=cusolver.dpotrf,cusolver.dpotrf_bufferSize
    length=size(handle,fill,n,factor.data.ptr,n)
    work=cp.empty(length,dtype=factor.dtype);info=cp.empty(1,dtype=cp.int32)
    potrf(handle,fill,n,factor.data.ptr,n,work.data.ptr,length,info.data.ptr)
    error=int(info.item())
    if error:raise np.linalg.LinAlgError(f'Cholesky info={error}')
    return factor


def main(a):
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    x, y, metadata = load_cloud(a.point_cloud, a.n)
    # All models use the identical locations, normals, query batches and mean
    # level at the contact. This does not equate their covariance kernels.
    r = dict(arguments=vars(a), complete=False, stages={}, checkpoints=[],
             point_cloud_metadata=metadata, point_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
             normal_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             queries_total=a.queries*a.batches, cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
             host_limit=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
             numerical_gate=dict(probability_max_error=1e-3, variance_max_error_over_prior=1e-4),
             comparison_scope='Same workload, different instantiated models; no equal reconstruction/calibration claim')
    def save():
        out.write_text(json.dumps(r, indent=2)+'\n')
    cp = None
    def sync():
        if cp is not None:
            cp.cuda.get_current_stream().synchronize()
    def timed(name, fn):
        sync(); start = perf_counter(); value = fn(); sync()
        r['stages'][name] = perf_counter()-start
        print(name, r['stages'][name], flush=True); save()
        return value
    try:
        qcheck = probe_points(a)
        if a.method == 'original':
            import gpytoolbox as gpy
            fn = gpy.stochastic_poisson_surface_reconstruction
            r['gpytoolbox_version'] = importlib.metadata.version('gpytoolbox')
            r['original_source_sha256'] = hashlib.sha256(Path(inspect.getfile(fn)).read_bytes()).hexdigest()
            # Fixed full-scan bounds across N; changing N never shrinks the box.
            allx = np.load(a.point_cloud, mmap_mode='r')[:, :3]
            lo, hi = allx.min(0)-.1, allx.max(0)+.1
            gs = np.full(3, a.grid); h = (hi-lo)/(a.grid-1)
            r['grid'] = dict(shape=gs.tolist(), spacing=h.tolist(), corner=lo.tolist(),
                             kernel_sigma=float(1.5*min(h)), subspace_dim=a.subspace)
            if getattr(a, 'original_lengthscale', None) is not None:
                from reproduction.matched_lengthscale import matched_grid
                r['grid'] = matched_grid(lo, hi, a.original_lengthscale)
                r['grid']['subspace_dim'] = a.subspace
                gs = np.array(r['grid']['shape'])
                h = np.array(r['grid']['spacing'])
                lo = np.array(r['grid']['corner'])
            save()
            sigma, sigma_n = .05, .005
            controlled = getattr(a, 'original_normalization', 'published') == 'controlled'
            if controlled:
                from reproduction.density_protocol import original_parameters, replace_original_floor
                parameters = original_parameters(h, a.amplitude, a.amplitude*a.relative_noise)
                sigma, sigma_n = parameters['sigma'], parameters['sigma_n']
                r['original_parameter_control'] = parameters
            import cProfile, pstats
            profiler=cProfile.Profile();profiler.enable()
            mean, variance, _ = timed('original_full_fit', lambda: fn(
                x, y, gs=gs, h=h, corner=lo, output_variance=True,
                solve_subspace_dim=a.subspace, sigma_n=sigma_n, sigma=sigma))
            profiler.disable()
            if controlled:
                variance = timed('controlled_scalar_floor', lambda: replace_original_floor(variance, parameters))
            stats=pstats.Stats(profiler)
            np.savez_compressed(out.with_suffix('.npz'),mean=mean,variance=variance)
            r['original_grid_file']=str(out.with_suffix('.npz'))
            r['original_profile']=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3])
                for k,v in stats.stats.items() if v[3]>.01]
            def predict(q):
                w = gpy.fd_interpolate(q, gs=gs, h=h, corner=lo)
                return np.asarray(w@mean).ravel(), np.asarray(w@variance).ravel()
            level = float(predict(np.array([[1., 0, 0]]))[0][0])
            r['level'] = level
            qm, qv = predict(qcheck)
            r['probes'] = dict(query=qcheck.tolist(), mean=qm.tolist(), variance=qv.tolist(),
                               probability=ndtr((level-qm)/np.sqrt(qv)).tolist())
            r['accuracy_status'] = ('controlled prior/noise and scalar floor; original density-lumped solver; grid/model convergence not established'
                                    if controlled else 'unmodified original implementation; grid/model convergence not established')
            r['query_seconds'], r['probability_checksums'] = [], []
            for q in query_batches(a):
                start = perf_counter(); m, v = predict(q)
                p = ndtr((level-m)/np.sqrt(v)); scores = (4*p*(1-p)).reshape(-1,64).mean(1)
                int(np.argmax(scores)); checksum = float(p.sum())
                if not np.all(v > 0) or not np.isfinite(p).all():
                    raise ArithmeticError('invalid original marginal outputs')
                r['query_seconds'].append(perf_counter()-start)
                r['probability_checksums'].append(checksum)
            r['preparation_seconds'] = r['stages']['original_full_fit']
            r['preparation_seconds'] += r['stages'].get('controlled_scalar_floor', 0.)
            r['total_seconds'] = r['preparation_seconds']+sum(r['query_seconds'])
            r['production_host_peak_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            r['complete'] = True
            return

        import cupy as cp
        import cupyx.scipy.linalg as cla
        from cupyx.scipy.special import ndtr as gpu_ndtr
        from geospsr.spectral import FourierPoisson
        from geospsr.spectral_tail import analytic_covariance_rows64
        r['device'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        r['device_memory'] = cp.cuda.runtime.memGetInfo()[1]
        from reproduction.gpu_audit import GpuProcessAudit
        audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id, interval=.1, cuda_device=cp.cuda.Device().id).start()
        r['model'] = dict(training='analytic periodic product Matern32',
                          lengthscale=a.ell, physical_axis_lengthscale=2*np.pi*a.ell,
                          per_axis_variance=a.amplitude**(1/3), noise_variance=a.amplitude*a.relative_noise,
                          cross_and_scalar_cutoff=a.cutoff, normalization='analytic')
        amplitude_axis = a.amplitude**(1/3); noise = a.amplitude*a.relative_noise
        # Warm only the common marginal-query pipeline, as in the H² benchmark.
        dummy = RegionalVarianceCache.__new__(RegionalVarianceCache)
        dummy._initialize_box([1,0,0], a.halfwidth, a.order)
        dummy.prior_variance=1.; dummy.correction=np.zeros((a.order**3,)*2)
        dummy.mean_coefficients=np.zeros(a.order**3)
        field = PolynomialMomentField(dummy)
        def warm():
            m,v=field.predict_gpu(np.tile([1.,0,0],(a.queries,1)))
            p=gpu_ndtr(m/cp.sqrt(v)); float(p.sum()); int(cp.argmax((4*p*(1-p)).reshape(-1,64).mean(1)))
        timed('common_runtime_excluded', warm)
        del field, dummy; cp.get_default_memory_pool().free_all_blocks()
        begin=perf_counter()
        op=timed('cross_spectrum', lambda: FourierPoisson(a.cutoff, a.ell, amplitude_axis,
                                                        dtype='float64', eps=1e-11))
        norm=(op.ns**2).sum(0)
        prior=float(cp.sum(cp.where(norm>0,op.rho/cp.where(norm>0,norm,1),0)))
        class CrossModel:
            prior_variance=prior
            def cross_covariance(self, queries, points):
                # Bound type-2 output/coordinate workspaces, preserving every
                # finite Fourier mode. This is common to all regional controls.
                values=[]
                for start in range(0,len(queries),16):
                    delta=cp.asarray(queries[start:start+16])[:,None]-cp.asarray(points)[None]
                    values.append(cp.asnumpy(op.cross_values(delta.reshape(-1,3))).reshape(-1,len(points),3))
                return np.concatenate(values)
        model=CrossModel()
        cache=RegionalVarianceCache.__new__(RegionalVarianceCache)
        cache._initialize_box([1,0,0],a.halfwidth,a.order); cache.prior_variance=prior
        n=len(x); nodes=a.order**3
        if a.method!='nufft-hybrid':
            u,b,cross_error=timed('regional_cross_compression',lambda: compress_cross_covariance(
                model,x,cache.nodes,a.cross_tolerance))
            r['solve_rank']=u.shape[1]; r['cross_error']=cross_error
            rhs=np.column_stack((u,y)); rank=u.shape[1]
        def matrix():
            mat=analytic_covariance_rows64(x,x,a.ell,amplitude_axis)
            mat.flat[::n+1]+=noise
            return mat
        def factor():
            if a.method=='dense' and a.dense_dtype=='float32':
                from reproduction.dense_covariance import analytic_covariance_rows32
                mat=analytic_covariance_rows32(x,x,a.ell,amplitude_axis)
                mat.flat[::n+1]+=noise
            else:mat=matrix()
            return cholesky_in_place(mat)
        # The SDD reference factor is built after production timing below.
        chol=None
        if a.method in ('dense','nufft-hybrid'):
            if a.method=='dense' and a.dense_dtype=='float32':
                from reproduction.dense_covariance import analytic_covariance_rows32
                def production_matrix():
                    mat=analytic_covariance_rows32(x,x,a.ell,amplitude_axis)
                    mat.flat[::n+1]+=noise
                    return mat
            else:production_matrix=matrix
            chol=timed('matrix_construction',production_matrix)
            r['matrix_bytes']=chol.nbytes
            chol=timed('cholesky_factorization',lambda:cholesky_in_place(chol))
            def solve(b):
                return cla.solve_triangular(chol,cla.solve_triangular(chol,cp.asarray(b,dtype=chol.dtype),lower=True),lower=True,trans='T').astype('d')
        def finish(z):
            gram=.5*(u.T@z[:,:rank]+z[:,:rank].T@u)
            if a.method=='nufft-cg':
                residual=u-cp.asnumpy(action(cp.asarray(z[:,:rank])))
                extra=z[:,:rank].T@residual
                gram+=.5*(extra+extra.T)
            projected=u.T@z[:,rank:]
            cache.correction=sum(b[:,d*nodes:(d+1)*nodes].T@gram@b[:,d*nodes:(d+1)*nodes] for d in range(3))
            cache.mean_coefficients=sum(b[:,d*nodes:(d+1)*nodes].T@projected[:,d] for d in range(3))
            f=PolynomialMomentField(cache); f.prepare_gpu(); sync()
            return f
        def replay(f):
            level=float(f.predict([[1,0,0]])[0][0]); times=[]; sums=[]; minimum=[]
            for q in query_batches(a):
                sync(); t=perf_counter(); m,v=f.predict_gpu(q)
                p=gpu_ndtr((level-m)/cp.sqrt(v)); scores=(4*p*(1-p)).reshape(-1,64).mean(1)
                int(cp.argmax(scores)); checksum=float(cp.sum(p)); vm=float(cp.min(v)); sync()
                if vm<=0 or not bool(cp.isfinite(p).all()):
                    raise ArithmeticError('invalid marginal outputs')
                times.append(perf_counter()-t); sums.append(checksum); minimum.append(vm)
            return dict(level=level,query_seconds=times,probability_checksums=sums,
                        minimum_variance=min(minimum),field=save_field(f))
        if a.method == 'dense':
            z=timed('solve',lambda:cp.asnumpy(solve(rhs)))
            field=timed('field',lambda:finish(z))
        elif a.method == 'nufft-cg':
            from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
            from geospsr.packed_spectral import PackedFourierKernel
            from geospsr.training_spectrum import FourierCovariance
            from reproduction.analytic_posterior import _AnalyticCG
            action=timed('analytic_action_setup',lambda:LeafCachedRangeKernel(
                x.astype('f'),a.ell,amplitude_axis,noise,leaf_size=128,max_cache_bytes=2**30,max_entries=100_000_000))
            spectrum=timed('training_spectrum',lambda:FourierCovariance(
                a.fit_cutoff,a.ell,amplitude_axis,eps=1e-6,dtype='float32'))
            from geospsr.preconditioning import SpatialBlockJacobi
            preconditioner=timed('analytic_preconditioner',lambda:SpatialBlockJacobi(
                x,noise,a.ell,amplitude_axis,block_size=a.preconditioner_block,dtype='float64')) if a.preconditioner_block else None
            exact=_AnalyticCG(action);z=np.empty_like(rhs);r['solve_blocks']=[]
            start=perf_counter()
            plans={};fourier_seconds=analytic_seconds=0.
            for first in range(0,rank,a.rhs_block_size):
                bb=cp.asarray(rhs[:,first:min(first+a.rhs_block_size,rank)]);count=bb.shape[1]
                if count not in plans:plans[count]=PackedFourierKernel(x,spectrum,noise,rhs_count=count)
                # A deliberately bounded warm start; its truncation and solve
                # error are removed by CG against the full analytic action.
                sync();part=perf_counter()
                initial,finite_info=plans[count].solve(bb,rtol=1e-3,maxiter=a.warm_iterations)
                sync();fourier_seconds+=perf_counter()-part;part=perf_counter()
                value,info=exact.solve(bb,rtol=a.variance_rtol,maxiter=2000,x0=initial,preconditioner=preconditioner)
                sync();analytic_seconds+=perf_counter()-part
                z[:,first:first+count]=cp.asnumpy(value)
                r['solve_blocks'].append(dict(first=first,finite=finite_info,analytic=info))
                save()
                if not info['converged']:raise RuntimeError('full analytic correction failed')
                if perf_counter()-start>a.budget and first+count<rank:
                    r['status']='censored: conditioning budget exhausted before all regional RHS'
                    r['censored_preparation_seconds']=perf_counter()-begin
                    return
            sync();part=perf_counter()
            initial,finite_info=PackedFourierKernel(x,spectrum,noise,rhs_count=3).solve(y,rtol=1e-3,maxiter=a.warm_iterations)
            sync();fourier_seconds+=perf_counter()-part;part=perf_counter()
            value,info=exact.solve(y,rtol=1e-6,maxiter=2000,x0=initial,preconditioner=preconditioner)
            sync();analytic_seconds+=perf_counter()-part
            r['solve_breakdown']=dict(fourier_seconds=fourier_seconds,analytic_seconds=analytic_seconds)
            if not info['converged']:raise RuntimeError('analytic mean correction failed')
            z[:,rank:]=cp.asnumpy(value);r['mean_solve']=dict(finite=finite_info,analytic=info)
            r['stages']['solve']=perf_counter()-start
            field=timed('field',lambda:finish(z))
        elif a.method == 'nufft-hybrid':
            from reproduction.variance_series import prepare_variance_series, FourierSeriesEvaluator
            alpha=timed('mean_solve',lambda:solve(y))
            coeff=timed('mean_aggregate',lambda:op.aggregate(x,alpha))
            vc,profile=timed('global_variance_series',lambda:prepare_variance_series(
                x,chol,op,batch=8,method='gradient',working_dtype=a.moment_dtype,
                source_dtype='float32',paired_sources=True,inverse_batch=512))
            r['variance_series']=profile
            # Once built, restrict the global field to the same local query
            # cache. This is stronger here than repeatedly invoking NUFFT.
            def restrict():
                dummy=RegionalVarianceCache.__new__(RegionalVarianceCache)
                dummy._initialize_box([1,0,0],a.halfwidth,2*a.order-1)
                from geospsr.regional_field import _coefficients
                f=PolynomialMomentField.__new__(PolynomialMomentField)
                f.center=dummy.center;f.halfwidth=dummy.halfwidth;f.order=dummy.order;f.prior_variance=prior;f._gpu=None
                m=FourierSeriesEvaluator(coeff).evaluate(dummy.nodes)[0].get()
                v=FourierSeriesEvaluator(vc).evaluate(dummy.nodes)[0].get()
                f.mean=_coefficients(m.reshape((f.order,)*3));f.correction=_coefficients((prior-v).reshape((f.order,)*3))
                f.prepare_gpu();sync();return f
            field=timed('field',restrict)
        elif a.method == 'sdd':
            import jax.numpy as jnp
            from geospsr.kernels import Matern32Kernel, ProductKernel
            from geospsr.iterative import StochasticDualDescent
            k=Matern32Kernel(a.ell,1.);kernel=ProductKernel(k,k,k)
            def initialize():
                s=StochasticDualDescent(jnp.asarray(x,dtype=jnp.float32),jnp.asarray(rhs,dtype=jnp.float32),kernel,
                    noise_variance=a.relative_noise,batch=min(a.batch,n),lr=a.eta*min(a.batch,n)/n,
                    momentum=.9,polyak=.01,seed=a.seed,sampling='rejection')
                s.compile();return s
            solver=timed('sdd_compile',initialize)
            base=perf_counter()-begin
            # Reference excluded and recorded explicitly. Checkpoints assess
            # own-model output accuracy, not residual alone.
            r['sdd_pre_reference_gpu_audit']=audit.stop()
            chol=timed('reference_cholesky_excluded',factor)
            def solve(b):
                return cla.solve_triangular(chol,cla.solve_triangular(chol,cp.asarray(b),lower=True),lower=True,trans='T')
            cref=timed('reference_cross_excluded',lambda:model.cross_covariance(qcheck,x))
            def direct_reference():
                c=cp.asarray(cref);alpha=solve(y)
                rr=cp.ascontiguousarray(c.transpose(1,0,2).reshape(n,-1));w=cla.solve_triangular(chol,rr,lower=True)
                return cp.asnumpy(cp.einsum('qnd,nd->q',c,alpha)),prior-cp.asnumpy((w*w).sum(0).reshape(-1,3).sum(1))
            mr,vr=timed('reference_moments_excluded',direct_reference)
            del chol, cref
            cp.get_default_memory_pool().free_all_blocks()
            audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id, interval=.1, cuda_device=cp.cuda.Device().id).start()
            elapsed=0.; selected=False
            for target in [256,512,1024,2048,4096,8192,16384,32768,65536,131072]:
                sync(); t=perf_counter()
                while solver.iterations<target:
                    value=solver.advance(min(256,target-solver.iterations))
                    if elapsed+perf_counter()-t>=a.budget:break
                z=np.asarray(value,dtype=float)/a.amplitude;sync();elapsed+=perf_counter()-t
                t=perf_counter();field=finish(z);finalize=perf_counter()-t
                qm,qv=field.predict(qcheck); quality=assess(qm,qv,mr,vr,prior)
                row=dict(iterations=solver.iterations,solve_seconds=elapsed,
                         preparation_seconds=base+elapsed+finalize,quality=quality)
                r['checkpoints'].append(row);save();print('checkpoint',row,flush=True)
                if not np.isfinite(z).all():
                    r['diverged']=True
                    break
                if quality['passes']:
                    r['preparation_seconds']=row['preparation_seconds'];selected=True;break
                if elapsed>=a.budget:break
            r['quality']=quality
            r['stages']['solve']=elapsed;r['stages']['field']=finalize
            r['seconds_per_iteration']=elapsed/solver.iterations
            if not selected:
                r['status']='censored: no passing observed checkpoint'
                r['censored_preparation_seconds']=r['checkpoints'][-1]['preparation_seconds']
                return
        if a.method!='sdd':r['preparation_seconds']=perf_counter()-begin
        r.update(replay(field))
        r['total_seconds']=r['preparation_seconds']+sum(r['query_seconds'])
        r['production_host_peak_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        r['gpu_process_audit']=audit.stop()
        save()
        # Independent (uncompressed) regional probes against double Cholesky.
        if a.method!='sdd':
            if a.method=='nufft-cg':
                del plans,action,exact,spectrum
                cp.get_default_memory_pool().free_all_blocks()
                chol=timed('reference_cholesky_excluded',lambda:cholesky_in_place(matrix()))
                def solve(b):
                    return cla.solve_triangular(chol,cla.solve_triangular(chol,cp.asarray(b),lower=True),lower=True,trans='T')
            if a.method=='dense' and a.dense_dtype=='float32' and a.reference=='double':
                del chol;cp.get_default_memory_pool().free_all_blocks()
                chol=timed('reference_cholesky_excluded',lambda:cholesky_in_place(matrix()))
            c=timed('reference_cross_excluded',lambda:cp.asarray(model.cross_covariance(qcheck,x)))
            def direct():
                alpha=solve(y);rr=cp.ascontiguousarray(c.transpose(1,0,2).reshape(n,-1))
                w=cla.solve_triangular(chol,rr,lower=True)
                return cp.asnumpy(cp.einsum('qnd,nd->q',c,alpha)),prior-cp.asnumpy((w*w).sum(0).reshape(-1,3).sum(1))
            if a.method=='dense' and a.reference=='refine':
                from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
                from reproduction.scaling_reference import refined_moments
                action=timed('reference_action_excluded',lambda:LeafCachedRangeKernel(
                    x.astype('f'),a.ell,amplitude_axis,noise,leaf_size=128,max_cache_bytes=2**30,max_entries=100_000_000))
                mr,vr,details=timed('reference_moments_excluded',lambda:refined_moments(solve,action,c,y,prior,noise))
                # Independently verify the fast analytic action on direct rows.
                check=cp.asarray(np.random.default_rng(13).normal(size=(n,3)))
                ids=np.linspace(0,n-1,32,dtype=int)
                exact=analytic_covariance_rows64(x[ids],x,a.ell,amplitude_axis)@check+noise*check[ids]
                details['action_max_absolute_error']=float(cp.max(abs(action(check)[ids]-exact)))
                if details['action_max_absolute_error']>1e-9:raise ArithmeticError('reference action validation failed')
                r['reference_details']=details
            else:mr,vr=timed('reference_moments_excluded',direct)
            qm,qv=field.predict(qcheck);r['quality']=assess(qm,qv,mr,vr,prior)
            if 'reference_details' in r:
                # Keep reference uncertainty far below the output gates.
                reference_ok=(max(details['variance_error_bound'])/prior<1e-8 and
                              max(details['mean_error_bound'])/np.sqrt(min(vr))<1e-6)
                r['quality']['reference_bounds_pass']=bool(reference_ok)
                r['quality']['passes'] &= bool(reference_ok)
        r['probes']=dict(query=qcheck.tolist(),mean=qm.tolist(),variance=qv.tolist(),
                         reference_mean=mr.tolist(),reference_variance=vr.tolist(),prior=prior)
        r['complete']=True
        if 'gpu_process_audit' not in r:r['gpu_process_audit']=audit.stop()
    except Exception as exc:
        r['failure']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if 'audit' in locals() and 'gpu_process_audit' not in r:
            r['gpu_process_audit']=audit.stop()
        r['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if cp is not None:r['retained_gpu_pool_bytes']=cp.get_default_memory_pool().total_bytes()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['original','dense','sdd','nufft-hybrid','nufft-cg'],required=True)
    p.add_argument('--n',type=int,required=True)
    p.add_argument('--point-cloud',default='/home/horde/.cache/fast-stochastic-psr/poissonrecon/eagle.npy')
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=12)
    p.add_argument('--queries',type=int,default=65536);p.add_argument('--batches',type=int,default=16)
    p.add_argument('--halfwidth',type=float,default=.005)
    p.add_argument('--order',type=int,default=5);p.add_argument('--cross-tolerance',type=float,default=1e-8)
    p.add_argument('--ell',type=float,default=.03/(2*np.pi))
    p.add_argument('--amplitude',type=float,default=.0025);p.add_argument('--relative-noise',type=float,default=.01)
    p.add_argument('--cutoff',type=int,default=50)
    p.add_argument('--reference',choices=['double','refine'],default='refine')
    p.add_argument('--dense-dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--moment-dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--fit-cutoff',type=int,default=50)
    p.add_argument('--warm-iterations',type=int,default=20)
    p.add_argument('--variance-rtol',type=float,default=1e-3)
    p.add_argument('--preconditioner-block',type=int,default=128)
    p.add_argument('--rhs-block-size',type=int,default=32)
    p.add_argument('--batch',type=int,default=512);p.add_argument('--eta',type=float,default=1.)
    p.add_argument('--budget',type=float,default=120.)
    p.add_argument('--grid',type=int,default=16);p.add_argument('--subspace',type=int,default=0)
    p.add_argument('--original-lengthscale',type=float,
                   help='Override --grid using isotropic half-correlation matching to a physical Matern32 lengthscale')
    p.add_argument('--original-normalization', choices=['published','controlled'], default='published')
    a=p.parse_args()
    if a.queries%64 or a.batches<1:p.error('positive batches and queries divisible by 64 required')
    main(a)
