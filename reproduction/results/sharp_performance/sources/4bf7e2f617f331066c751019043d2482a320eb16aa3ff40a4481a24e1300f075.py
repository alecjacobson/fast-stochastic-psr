"""Direct GPU marginal queries with scalar contacts and no regional interpolation.

This correctness/performance control evaluates a finite query panel. It does not
claim the million-query regional-cache workload used in earlier experiments.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern, _CROSS
from geospsr.scalar_contacts import ScalarContactUpdate
from reproduction.partial_occupancy import ROOT, load, stratified
from reproduction.stanford_partial_data import digest


_KERNEL = r'''
extern "C" __global__ void cross_rhs(const double* q, const double* x,
    double* out, int m, int n, double kappa, double amplitude) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=(long long)n*3*m) return;
    int i=t/(3*m), d=(t/m)%3, j=t%m;
    double norm=0;
    for(int k=0;k<3;k++){double v=q[3*j+k]-x[3*i+k];norm+=v*v;}
    double z=kappa*sqrt(norm), radial;
    if(z<.5) {
        const double a[] = {COEFFICIENTS};
        radial=a[DEGREE];
        for(int k=DEGREE-1;k>=0;k--) radial=radial*z+a[k];
    } else radial=(8-exp(-z)*(8+8*z+4*z*z+z*z*z))/(z*z*z);
    out[t]=amplitude*radial*(q[3*j+d]-x[3*i+d]);
}
'''


def cross_rhs(model, queries, points):
    import cupy as cp
    q, x = cp.asarray(queries, dtype=cp.float64), cp.asarray(points, dtype=cp.float64)
    source = _KERNEL.replace('COEFFICIENTS', ','.join(format(v, '.17g') for v in _CROSS)).replace('DEGREE', str(len(_CROSS)-1))
    kernel = cp.RawKernel(source, 'cross_rhs')
    out = cp.empty((len(x), 3*len(q)), dtype=cp.float64)
    if out.size:
        kernel(((out.size+255)//256,), (256,), (q, x, out, len(q), len(x), model.kappa, model.amplitude))
    return out


class DirectContactQueries:
    def __init__(self, model, points, vectors, contacts, system, dense=False, iterations=32,
                 solve_tolerance=2.5e-11):
        import cupy as cp
        self.model, self.x, self.contacts, self.system = model, cp.asarray(points), contacts, system
        self.dense, self.iterations = dense, iterations
        solve = system.solve if dense else lambda b: system.solve(b, tolerance=solve_tolerance)
        self.alpha = cp.asarray(solve(vectors))
        self.mean_solve_info = deepcopy(getattr(system, 'last_solve', None))
        cc = cross_rhs(model, contacts, self.x)
        self.zc = cp.asarray(solve(cp.asnumpy(cc)))
        self.contact_solve_info = deepcopy(getattr(system, 'last_solve', None))
        count = len(contacts)
        mean = sum(cc[:, d*count:(d+1)*count].T@self.alpha[:, d] for d in range(3))
        correction = sum(cc[:, d*count:(d+1)*count].T@self.zc[:, d*count:(d+1)*count] for d in range(3))
        covariance = model.scalar_covariance(contacts, contacts)-cp.asnumpy((correction+correction.T)*.5)
        self.update = ScalarContactUpdate(cp.asnumpy(mean), covariance, np.zeros(count), .01**2)
        self.solve_iterations = []

    def predict(self, queries, batch_size=24, compute_variance=True):
        import cupy as cp
        from cupyx.scipy.linalg import solve_triangular
        from geospsr.hierarchical import batched_pcg
        mean, variance = np.empty(len(queries)), np.empty(len(queries))
        nc = len(self.contacts)
        for first in range(0, len(queries), batch_size):
            q = queries[first:first+batch_size]; m = len(q)
            c = cross_rhs(self.model, q, self.x)
            base_mean = sum(c[:, d*m:(d+1)*m].T@self.alpha[:, d] for d in range(3))
            contact_cross = self.model.scalar_covariance(q, self.contacts)-cp.asnumpy(sum(
                c[:, d*m:(d+1)*m].T@self.zc[:, d*nc:(d+1)*nc] for d in range(3)))
            if compute_variance:
                if self.dense:
                    w = solve_triangular(self.system.factor, c, lower=True, check_finite=False)
                    correction = cp.sum(w*w, axis=0)
                else:
                    if hasattr(self.system, 'variance_solve'):
                        z, info, residual = self.system.variance_solve(c, self.iterations)
                    else:
                        z, info, residual = batched_pcg(self.system.matmul, self.system.precondition,
                            c, tolerance=1e-10, max_iterations=self.iterations, xp=cp, return_residual=True)
                    correction = cp.sum(c*z+z*residual, axis=0)
                    self.solve_iterations.append(info['max_iterations'])
                    # Do not retain the previous batch's two N-by-RHS arrays
                    # while allocating the next solve and its workspace.
                    del z, residual
                base_var = self.model.prior_variance-cp.asnumpy(correction.reshape(3, m).sum(axis=0))
                mm, vv = self.update.apply(cp.asnumpy(base_mean), base_var, contact_cross)
            else:
                # Same full posterior mean. Only its variance is replaced by the
                # scalar-contact-only prior control; no normal-query RHS solves.
                from reproduction.evaluate_partial_controls import contact_only_variance
                mm = cp.asnumpy(base_mean)+contact_cross@self.update.weights
                vv = contact_only_variance(self.model, self.contacts, q)
            mean[first:first+m], variance[first:first+m] = mm, vv
        return mean, variance


def run(args):
    import cupy as cp
    from reproduction.gpu_audit import GpuProcessAudit
    x, vectors, contacts, panel = load(args.group, args.n)
    selection = json.loads((ROOT/'selected.json').read_text())
    model = FreeSpacePoissonMatern(selection['ell'], 1/3); noise = selection['sigma']**2
    rank, neighbors = getattr(args, 'rank', 50), getattr(args, 'neighbors', 40)
    # Shared runtime/kernel initialization precedes all method timings.
    cross_rhs(model, contacts[:2], x[:2]); cp.cuda.Device().synchronize()
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id, interval=.1, cuda_device=cp.cuda.Device().id).start()
    begin = perf_counter()
    if args.method == 'dense':
        from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
        system = GPUDenseMaternSystem(model, x, noise, dtype='float64')
    elif args.method == 'h2':
        from geospsr.hierarchical import HierarchicalMaternSystem
        from geospsr.hierarchical_gpu import GPUHierarchicalSystem
        native = HierarchicalMaternSystem(model, x, noise, tolerance=getattr(args, 'h2_tolerance', 1e-8),
            leaf_size=40, preconditioner_rank=rank, neighbors=neighbors, threads=4,
            seed=9101, preconditioner_backend='prefix-kd')
        system = GPUHierarchicalSystem(native, interaction_budget_bytes=256*2**20,
                                       rhs_block_size=72, release_native=True)
    elif args.method == 'dense-cg':
        import geospsr.nonperiodic_dense_cg as dense_cg
        original = dense_cg.HierarchicalMaternSystem
        def prefix(*a, **kw):
            kw['preconditioner_backend'] = 'prefix-kd'
            return original(*a, **kw)
        dense_cg.HierarchicalMaternSystem = prefix
        try:
            system = dense_cg.GPUDenseCGSystem(model, x, noise, dtype='float64',
                rank=rank, neighbors=neighbors, threads=4, seed=9101)
        finally:
            dense_cg.HierarchicalMaternSystem = original
        system.rhs_block_size = 72
    else:
        from geospsr.nonperiodic_sparse import GPUSparseMaternSystem
        system = GPUSparseMaternSystem(model, x, noise, cutoff=args.cutoff,
            rank=rank, neighbors=neighbors, threads=4, seed=9101, graph_backend='gpu',
            preconditioner_backend='prefix-kd', rhs_block_size=72)
    cp.cuda.Device().synchronize(); setup = perf_counter()-begin
    print('setup', setup, flush=True)
    begin = perf_counter()
    posterior = DirectContactQueries(model, x, vectors, contacts, system,
        dense=args.method == 'dense', iterations=args.iterations,
        solve_tolerance=getattr(args, 'solve_tolerance', 2.5e-11))
    cp.cuda.Device().synchronize(); contact_seconds = perf_counter()-begin
    begin = perf_counter()
    mean, variance = posterior.predict(panel['queries'], args.query_batch)
    probability_checksum = float(ndtr(-mean/np.sqrt(variance)).sum())
    cp.cuda.Device().synchronize(); query_seconds = perf_counter()-begin
    gpu_audit = audit.stop()
    begin = perf_counter()
    shortcut_mean, shortcut_variance = posterior.predict(panel['queries'], args.query_batch, compute_variance=False)
    shortcut_probability_checksum = float(ndtr(-shortcut_mean/np.sqrt(shortcut_variance)).sum())
    cp.cuda.Device().synchronize(); shortcut_query_seconds = perf_counter()-begin
    np.testing.assert_allclose(mean, shortcut_mean, atol=1e-12, rtol=0)
    record = dict(arguments=vars(args), queries=len(mean), setup_seconds=setup,
        contact_and_mean_seconds=contact_seconds, query_seconds=query_seconds,
        total_seconds=setup+contact_seconds+query_seconds,
        shortcut_query_seconds=shortcut_query_seconds,
        shortcut_total_seconds=setup+contact_seconds+shortcut_query_seconds,
        quality=stratified(mean, variance, panel), shortcut_quality=stratified(mean, shortcut_variance, panel),
        production_gpu_audit=gpu_audit, variance_iteration_max=max(posterior.solve_iterations, default=0),
        mean_solve_info=posterior.mean_solve_info, contact_solve_info=posterior.contact_solve_info,
        stored_backend_bytes=getattr(system, 'device_matrix_bytes', getattr(system, 'factor_bytes', None)),
        interaction_entries=getattr(system, 'interaction_nnz', None),
        locality_diagnostics=getattr(system, 'diagnostics', None),
        matrix_cached=getattr(system, 'matrix_cached', None),
        cdf_timed=True, probability_checksum=probability_checksum,
        shortcut_probability_checksum=shortcut_probability_checksum,
        scope='Direct finite query panel; no polynomial interpolation or million-query claim',
        source_sha256={p: digest(p) for p in [__file__, 'geospsr/scalar_contacts.py',
            'geospsr/hierarchical_gpu.py', 'geospsr/nonperiodic.py', 'geospsr/nonperiodic_dense_cg.py',
            'geospsr/nonperiodic_sparse.py', 'reproduction/evaluate_partial_controls.py']})
    exact = ROOT/'exact'/f'{args.group}-{args.n}.npz'
    if exact.exists():
        truth = np.load(exact)
        errors = dict(mean=float(np.max(abs(mean-truth['mean']))),
            variance_over_prior=float(np.max(abs(variance-truth['variance']))/model.prior_variance),
            cdf=float(np.max(abs(ndtr(-mean/np.sqrt(variance))-ndtr(-truth['mean']/np.sqrt(truth['variance']))))))
        record['exact_errors'] = errors
        record['passes_exact_panel'] = errors['variance_over_prior'] <= 1e-4 and errors['cdf'] <= 1e-3
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), mean=mean, variance=variance, shortcut_variance=shortcut_variance)
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k in ['total_seconds','shortcut_total_seconds','exact_errors','passes_exact_panel']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['dense','h2','locality','dense-cg'], required=True)
    p.add_argument('--group', default='test_side'); p.add_argument('--n', type=int, default=2048)
    p.add_argument('--iterations', type=int, default=32); p.add_argument('--cutoff', type=float, default=12.)
    p.add_argument('--query-batch', type=int, default=24); p.add_argument('--output', required=True)
    run(p.parse_args())
