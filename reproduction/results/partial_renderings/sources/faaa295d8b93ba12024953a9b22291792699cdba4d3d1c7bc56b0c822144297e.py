"""Compute actual posterior fields for the PR122 scientific illustrations.

Extra spatial queries are visualization work, never part of the 128-query timer.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from reproduction.partial_mixed_h2 import task, probability
from reproduction.stanford_partial_data import CACHE, digest

OUT = Path('reproduction/results/partial_renderings')
N = 70652
HEIGHTS = [.30, -.25]
RESOLUTION = 72


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def inputs():
    x, y, contacts, panel, model, noise, truth, fit, contract = task(N)
    reference = dict(np.load(CACHE/'reference.npz'))
    v = reference['vertices']
    axes = [np.linspace(v[:, d].min()-.10, v[:, d].max()+.10, RESOLUTION) for d in [0, 2]]
    xx, zz = np.meshgrid(*axes)
    slices = np.array([np.column_stack((xx.ravel(), np.full(xx.size, h), zz.ravel())) for h in HEIGHTS])
    return x, y, contacts, panel, model, noise, truth, fit, contract, reference, axes, slices


def mean_coefficients(posterior):
    """Fold the 32 scalar-contact update into three vector-data coefficients."""
    import cupy as cp
    nc = len(posterior.contacts)
    w = cp.asarray(posterior.update.weights)
    coefficients = posterior.alpha-cp.column_stack([
        posterior.zc[:, d*nc:(d+1)*nc]@w for d in range(3)])
    return coefficients, posterior.update.weights


def mean_field(posterior, queries, coefficients, batch_size=192):
    import cupy as cp
    from reproduction.partial_direct_gpu import cross_rhs
    a, w = coefficients
    result = np.empty(len(queries))
    for first in range(0, len(queries), batch_size):
        q = queries[first:first+batch_size]; m = len(q)
        b = cross_rhs(posterior.model, q, posterior.x)
        result[first:first+m] = cp.asnumpy(sum(b[:, d*m:(d+1)*m].T@a[:, d] for d in range(3)))
        result[first:first+m] += posterior.model.scalar_covariance(q, posterior.contacts)@w
    return result


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, contacts, panel, model, noise, truth, fit, contract, ref, axes, slices = inputs()
    np.savez_compressed(OUT/'scene.npz', points=x, normals=y, contacts=contacts,
        vertices=ref['vertices'], faces=ref['faces'], benchmark_queries=panel['queries'],
        axis_x=axes[0], axis_z=axes[1], heights=HEIGHTS, slice_queries=slices)
    write_json(OUT/'contract.json', dict(training=contract, n=N, resolution=RESOLUTION,
        heights=HEIGHTS, plane_axes=['x', 'z'], plane_normal_axis='y',
        display_scope='New 72x72 slice fields and a sampled mean isosurface, not the timed 128-query task.',
        independent_check='8x8 deterministic grid points per slice plus all 128 benchmark queries; no uniform field certificate.',
        reference_mesh_scope='Same-acquisition reconstructed Stanford mesh; context and registration reference, not independent physical truth.',
        scene_sha256=digest(OUT/'scene.npz'), mesh_sha256=digest(CACHE/'reference.npz'),
        source_sha256=digest(__file__)))


def h2(iterations):
    import cupy as cp
    from geospsr.hierarchical import HierarchicalMaternSystem
    from geospsr.hierarchical_gpu import GPUHierarchicalSystem
    from reproduction.partial_direct_gpu import DirectContactQueries
    from reproduction.partial_competitor_common import gates
    from reproduction.gpu_audit import GpuProcessAudit
    if (OUT/'h2-fields.npz').exists(): raise FileExistsError('H2 illustration fields already exist')
    x, y, contacts, panel, model, noise, truth, fit, contract, ref, axes, slices = inputs()
    queries = np.concatenate([panel['queries'], *slices])
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id, interval=.5, cuda_device=cp.cuda.Device().id).start()
    start = perf_counter()
    native = HierarchicalMaternSystem(model, x, noise, tolerance=1e-6, leaf_size=40,
        preconditioner_rank=100, neighbors=80, threads=4, seed=9101, preconditioner_backend='prefix-kd')
    system = GPUHierarchicalSystem(native, dtype='float32', rhs_block_size=72, release_native=True)
    posterior = DirectContactQueries(model, x, y, contacts, system, iterations=iterations, solve_tolerance=1e-9)
    coefficients = mean_coefficients(posterior)
    mean, variance = np.empty(len(queries)), np.empty(len(queries))
    for first in range(0, len(queries), 384):
        stop = min(first+384, len(queries))
        mean[first:stop], variance[first:stop] = posterior.predict(queries[first:stop], batch_size=96)
        probability(mean[first:stop], variance[first:stop])
        print('SLICE', stop, len(queries), 'elapsed', perf_counter()-start, flush=True)
        np.savez_compressed(OUT/'h2-progress.npz', queries=queries[:stop], mean=mean[:stop], variance=variance[:stop])
    benchmark = gates(mean[:128], variance[:128], truth, model, fit)
    folded_error = float(np.max(abs(mean_field(posterior, queries[:128], coefficients)-mean[:128])))
    if not benchmark['passes'] or folded_error > 1e-10: raise ValueError((benchmark, folded_error))
    np.savez_compressed(OUT/'h2-fields.npz', queries=queries, mean=mean, variance=variance)
    record = dict(n=N, iterations=iterations, tolerance=1e-6, slice_queries=len(queries)-128,
        benchmark_errors=benchmark, folded_mean_max_error=folded_error,
        slice_seconds=perf_counter()-start, source_sha256=digest(__file__),
        field_sha256=digest(OUT/'h2-fields.npz'), contract_sha256=digest(OUT/'contract.json'))
    write_json(OUT/'h2-fields.json', record)
    # Visualization grid only: the GP itself remains free-space and mesh-free.
    v = ref['vertices']; volume_axes = [np.linspace(v[:, d].min()-.10, v[:, d].max()+.10, r)
                                     for d, r in enumerate([80, 96, 80])]
    q = np.stack(np.meshgrid(*volume_axes, indexing='ij'), axis=-1).reshape(-1, 3)
    vm = np.empty(len(q)); begin = perf_counter()
    for first in range(0, len(q), 12288):
        vm[first:first+12288] = mean_field(posterior, q[first:first+12288], coefficients)
        print('MEAN VOLUME', min(first+12288,len(q)),len(q),'elapsed',perf_counter()-begin,flush=True)
    np.savez_compressed(OUT/'mean-volume.npz', mean=vm.reshape(80,96,80), axis_x=volume_axes[0], axis_y=volume_axes[1], axis_z=volume_axes[2])
    record.update(mean_volume_seconds=perf_counter()-begin, mean_volume_queries=len(q),
        mean_volume_sha256=digest(OUT/'mean-volume.npz'), gpu_audit=audit.stop())
    write_json(OUT/'h2-fields.json', record)
    (OUT/'h2-progress.npz').unlink()


def check():
    import cupy as cp
    from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem, full_kernel_matmul
    from reproduction.partial_direct_gpu import cross_rhs
    from reproduction.partial_competitor_common import conditioned_moments, gates
    from reproduction.gpu_audit import GpuProcessAudit
    if (OUT/'reference-check.npz').exists(): raise FileExistsError('Reference checks already exist')
    x, y, contacts, panel, model, noise, truth, fit, contract, ref, axes, slices = inputs()
    h = dict(np.load(OUT/'h2-fields.npz'))
    ii = np.linspace(0, RESOLUTION-1, 8, dtype=int)
    grid_ids = (ii[:,None]*RESOLUTION+ii[None,:]).ravel()
    ids = np.r_[np.arange(128), 128+grid_ids, 128+RESOLUTION**2+grid_ids]
    q = h['queries'][ids]
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id, interval=.5, cuda_device=cp.cuda.Device().id).start()
    start = perf_counter(); system = GPUDenseMaternSystem(model, x, noise, dtype='float32')
    means, variances, histories = [], [], []
    for first in range(0, len(q), 128):
        qq = q[first:first+128]; b = cp.column_stack((cross_rhs(model,np.concatenate((contacts,qq)),x), cp.asarray(y)))
        z = cp.asarray(system.solve(cp.asnumpy(b))); norms = cp.linalg.norm(b, axis=0)
        previous = None; history = []
        for stage in range(6):
            residual = b-cp.asarray(full_kernel_matmul(model,x,noise,cp.asnumpy(z)))
            m,v = conditioned_moments(model,contacts,qq,b,z,cp,residual=residual)
            probability(m,v)
            relative = float(cp.max(cp.linalg.norm(residual,axis=0)/norms).item())
            change = gates(m,v,previous,model,fit) if previous is not None else None
            history.append(dict(stage=stage,relative_residual=relative,change=change))
            print('CHECK',first,stage,relative,flush=True)
            if relative <= 1e-9 and change and change['cdf'] <= 1e-6 and change['calibrated_cdf'] <= 1e-6: break
            previous = dict(mean=m.copy(),variance=v.copy()); z += cp.asarray(system.solve(cp.asnumpy(residual)))
        else: raise RuntimeError('independent full-kernel reference did not converge')
        means.extend(m); variances.extend(v); histories.append(history)
    m,v = np.array(means),np.array(variances)
    errors = gates(h['mean'][ids],h['variance'][ids],dict(mean=m,variance=v),model,fit)
    old_errors = gates(m[:128],v[:128],truth,model,fit)
    np.savez_compressed(OUT/'reference-check.npz', indices=ids, queries=q, mean=m, variance=v)
    write_json(OUT/'reference-check.json', dict(errors=errors, historical_reference_agreement=old_errors,
        count=len(ids), histories=histories, seconds=perf_counter()-start, gpu_audit=audit.stop(),
        field_sha256=digest(OUT/'reference-check.npz'), source_sha256=digest(__file__),
        scope='128 benchmark points plus 64 points on each slice; not a uniform bound on the rendered fields.'))
    print('ERRORS',errors,flush=True)
    if not errors['passes']: raise RuntimeError('new illustration check fails fixed gates')


def mesh():
    from skimage.measure import marching_cubes
    data = np.load(OUT/'mean-volume.npz'); axes = [data['axis_'+a] for a in 'xyz']; field = data['mean']
    vertices, faces, _, _ = marching_cubes(field, level=0., gradient_direction='ascent',
                                         spacing=[a[1]-a[0] for a in axes])
    vertices += np.array([a[0] for a in axes])
    edges = np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _, counts = np.unique(edges,axis=0,return_counts=True)
    np.savez_compressed(OUT/'mean-mesh.npz',vertices=vertices,faces=faces)
    write_json(OUT/'mean-mesh.json',dict(vertices=len(vertices),faces=len(faces),boundary_edges=int((counts==1).sum()),
        minimum=float(field.min()),maximum=float(field.max()),display_grid=list(field.shape),
        scope='Zero level set of the posterior mean on the declared display box; no surface repair or box capping.',
        field_sha256=digest(OUT/'mean-volume.npz'),mesh_sha256=digest(OUT/'mean-mesh.npz'),
        gradient_direction='ascent (negative inside)',source_sha256=digest(__file__)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','h2','check','mesh']);p.add_argument('--iterations',type=int,default=10)
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='h2':h2(a.iterations)
    elif a.action=='check':check()
    else:mesh()
