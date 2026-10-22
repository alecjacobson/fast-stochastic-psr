"""Publish the covariance-tail diagnostic, including its retained failure."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / 'results/spectral_tail'


def covariance_kl_bound(covariance, h):
    """KL upper bound from a PSD covariance increment; means held identical."""
    l = np.linalg.cholesky(covariance)
    z = np.linalg.solve(l, h)
    z = np.linalg.solve(l, z.T).T
    eigenvalues = np.linalg.eigvalsh((z + z.T) / 2)
    if eigenvalues.min() < -1e-10 * max(float(np.abs(eigenvalues).max()), 1e-30):
        raise ArithmeticError('non-PSD covariance increment')
    e = np.maximum(eigenvalues, 0)
    terms = np.where(e < 1e-5, e*e/2 - e**3/3 + e**4/4, e-np.log1p(e))
    return float(.5 * terms.sum()), float(e.max())


def main():
    names = ['A', 'B', '1000000', 'B-short-low-noise']
    labels = ['A: 691', 'B: 17,419', '1,000,000', 'B: short / low noise']
    rows = []
    for name, label in zip(names, labels):
        record = json.loads((ROOT / name / 'results.json').read_text())
        if not record.get('complete', False):
            raise RuntimeError(f'incomplete audit: {name}')
        arrays = np.load(ROOT / name / 'fields.npz')
        v = arrays['finite_covariance']
        scale = np.linalg.norm(v)
        seeds = record['args']['seeds']
        trace = np.concatenate([arrays[f'{s}_trace_samples'] for s in seeds]) / scale
        diag = np.concatenate([arrays[f'{s}_diagonal_samples'] for s in seeds]) / np.diag(v)
        var_mean, var_se = diag.mean(0), diag.std(0, ddof=1)/np.sqrt(len(diag))
        row = {'name': name, 'label': label, 'N': record['N'], 'frequencies_total': len(trace),
               'trace_bound_estimate': float(trace.mean()),
               'trace_bound_standard_error': float(trace.std(ddof=1)/np.sqrt(len(trace))),
               'max_variance_bound_estimate': float(var_mean.max()),
               'max_variance_bound_marginal_normal_95_upper': float((var_mean+1.96*var_se).max()),
               'exact': record.get('exact'), 'seconds': record['seconds']}
        h = sum(arrays[f'{s}_bound_estimate'] for s in seeds) / len(seeds)
        kl, max_eigenvalue = covariance_kl_bound(v, h)
        row['covariance_only_event_probability_bound_estimate'] = min(1., float(np.sqrt(kl/2)))
        row['max_whitened_tail_eigenvalue_estimate'] = max_eigenvalue
        row['finite_covariance_condition_number'] = float(np.linalg.cond(v))
        if record['dense_reference']:
            full = arrays['analytic_covariance']
            exact_kl, _ = covariance_kl_bound(v, arrays['exact_bound'])
            actual_kl = .5 * (np.trace(np.linalg.solve(v, full)) - len(v)
                              + np.linalg.slogdet(v)[1] - np.linalg.slogdet(full)[1])
            assert -1e-12 < actual_kl <= exact_kl + 1e-12
            row['exact_covariance_only_kl'] = float(actual_kl)
            row['exact_covariance_only_kl_upper_bound'] = exact_kl
        rows.append(row)
    strict = np.load(ROOT / '1000000/fields.npz')
    prod = np.load(ROOT / '1000000-production/fields.npz')
    np.testing.assert_array_equal(strict['query'], prod['query'])
    v = strict['finite_covariance']
    difference = prod['finite_covariance'] - v
    precision = {'covariance_relative_frobenius_difference': float(np.linalg.norm(difference)/np.linalg.norm(v)),
                 'max_variance_relative_difference': float(np.max(np.abs(np.diag(difference))/np.diag(v))),
                 'bound_estimate_relative_frobenius_difference_by_seed': {
                     str(s): float(np.linalg.norm(prod[f'{s}_bound_estimate']-strict[f'{s}_bound_estimate'])/
                                   np.linalg.norm(strict[f'{s}_bound_estimate'])) for s in [71, 72, 73]},
                 'qualification': 'production eps1e-6 / CG rtol1e-5 versus requested eps1e-7 (clamped to float32 epsilon) / rtol1e-6; same inputs and frequencies; no exact large-N transform/solve certificate'}
    double = np.load(ROOT / '1000000-double-control/fields.npz')
    dc_record = json.loads((ROOT / '1000000-double-control/results.json').read_text())
    subset = dc_record['args']['probe_indices']
    np.testing.assert_array_equal(double['query'], strict['query'][subset])
    dv = double['finite_covariance']
    precision['double_control'] = {'probe_indices': subset, 'seconds': dc_record['seconds'],
                                   'scope': 'float64 eps1e-9 / CG rtol1e-9 on three of the same million-point probes; training oversampling1.25'}
    for name, data in [('stricter_float32', strict), ('production_float32', prod)]:
        difference = data['finite_covariance'][np.ix_(subset, subset)] - dv
        precision['double_control'][name] = {
            'covariance_relative_frobenius_difference': float(np.linalg.norm(difference)/np.linalg.norm(dv)),
            'max_variance_relative_difference': float(np.max(np.abs(np.diag(difference))/np.diag(dv)))}
    (ROOT / 'summary.json').write_text(json.dumps({'rows': rows, 'precision': precision}, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout='constrained')
    indices = np.arange(len(rows))
    axes[0].errorbar(indices, [r['trace_bound_estimate']*100 for r in rows],
                     yerr=[1.96*r['trace_bound_standard_error']*100 for r in rows],
                     fmt='o', capsize=4, color='#d17a22', label='Estimated trace bound ± 1.96 SE')
    exact_ids = [i for i,r in enumerate(rows) if r['exact']]
    axes[0].scatter(exact_ids, [rows[i]['exact']['covariance_gap_relative_frobenius']*100 for i in exact_ids],
                    marker='s', color='#175b83', label='Actual covariance change (dense)')
    axes[0].axhline(.02, color='0.5', ls='--', lw=1, label='Existing covariance target (0.02%)')
    axes[0].set_ylabel('Relative covariance Frobenius error / bound (%)')
    axes[1].scatter(indices, [r['max_variance_bound_estimate']*100 for r in rows], color='#d17a22', label='Estimated maximum diagonal bound')
    axes[1].scatter(exact_ids, [rows[i]['exact']['max_variance_gap_fraction']*100 for i in exact_ids],
                    marker='s', color='#175b83', label='Actual largest variance change (dense)')
    axes[1].set_ylabel('Largest checked variance change / bound (%)')
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xticks(indices, labels, rotation=15, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Training-tail covariance diagnostic: 16 fixed probes, cutoff 200')
    fig.savefig(ROOT / 'covariance-tail.png', dpi=170)
    plt.close(fig)
    table = []
    for r in rows:
        actual = f"{r['exact']['covariance_gap_relative_frobenius']*100:.6f}" if r['exact'] else 'not computed'
        table.append(f"| {r['label']} | {actual} | {100*r['trace_bound_estimate']:.6f} ± {196*r['trace_bound_standard_error']:.6f} | {100*r['max_variance_bound_estimate']:.6f} |")
    million = rows[2]
    text = f'''# Posterior covariance error from the omitted training spectrum

The million-point cutoff-200 pilot now has a **covariance diagnostic**, in
addition to sampled analytic solve residuals. At 16 fixed slice probes, the
estimated trace bound is **{100*million['trace_bound_estimate']:.4f}%** of the finite posterior
covariance's Frobenius norm. The largest checked variance bound is about
**{100*million['max_variance_bound_estimate']:.4f}%**. These estimates are comfortably below the
existing 0.02% covariance target on this panel. They do **not** establish
complete-application matching quality or a rigorous large-N certificate.

![Covariance tail diagnostic and retained stress failure](covariance-tail.png)

## Dense overlap and million-point estimates

All runs keep scalar/cross cutoff 50 and training cutoff 200. Default parameters
are length scale 0.01, per-axis variance 0.1 and noise variance 0.02. The stress
case changes length scale to 0.005 and noise variance to 0.0004. A/B use the
released-camera scorpion data; the million-point scan uses the previously
published area-uniform visible surface cloud. Input/query hashes are recorded.

Each panel has 16 queries, half selected among the 256 smallest absolute
default-parameter slice means and half from the remaining positions. Selection
seed 993 precedes tail sampling. Three independent seeds (71–73) each draw
16,384 frequencies from the **infinite** omitted spectrum. The table pools
49,152 frequencies; it does not count posterior samples.

| Input | Actual covariance change % | Estimated trace bound % ± 1.96 SE | Largest estimated variance bound % |
| --- | ---: | ---: | ---: |
{chr(10).join(table)}

Covariance change uses relative Frobenius norm, with the finite posterior as
denominator. The trace bound is a conservative upper bound in exact arithmetic;
its **estimate and empirical interval are not themselves a certified bound**.
Diagonal bounds divide by each query's finite variance. Off-diagonal covariance
entries need not increase. See the [derivation, sampler and limitations](METHOD.md).

At A/B, float64 dense analytic and independently summed finite covariance
matrices verify the PSD bracket directly. The bound is tight in these overlap
cases; the independent frequency estimates agree with the dense calculation
within ordinary sampling variability. The stress case remains a clear failure,
consistent with the earlier training-bandwidth audit.

For a connection to joint probabilities, whitening the estimated H by the
finite covariance gives a largest eigenvalue of
**{million['max_whitened_tail_eigenvalue_estimate']:.6g}** at the million-point panel.
The corresponding Gaussian KL/Pinsker calculation estimates a
**{100*million['covariance_only_event_probability_bound_estimate']:.5f} percentage-point** bound on
any event using these 16 values **if the two means are held identical**.
This isolates covariance error; it is not a bound on actual application
probabilities, which also depend on mean/centering and many more query points.
It inherits the empirical tail estimator's uncertainty. See [the derivation](METHOD.md).

## Numerical control and cost

The million-point primary audit uses CG rtol 1e-6 and requests NUFFT eps 1e-7
(cuFINUFFT clamps this to float32 machine epsilon). A second run uses production
rtol 1e-5 / eps 1e-6 on identical probes and tail frequencies. Their finite
covariance matrices differ by **{100*precision['covariance_relative_frobenius_difference']:.7f}%**;
the maximum diagonal difference is **{100*precision['max_variance_relative_difference']:.7f}%**.
This controls observed sensitivity; it is not an exact-solve certificate.

A float64 check (eps 1e-9, CG rtol 1e-9) covers three of those same probes,
indices 0, 8 and 15. Compared with that control, the stricter float32 panel
differs by **{100*precision['double_control']['stricter_float32']['covariance_relative_frobenius_difference']:.7f}%**
in covariance Frobenius norm and at most
**{100*precision['double_control']['stricter_float32']['max_variance_relative_difference']:.7f}%**
on its three variances. Production float32 differs by
**{100*precision['double_control']['production_float32']['covariance_relative_frobenius_difference']:.7f}%**
and **{100*precision['double_control']['production_float32']['max_variance_relative_difference']:.7f}%**
respectively. These numerical differences remain separate from the tail bound.
On these three probes, production float32 is closer to float64 than the
stricter-tolerance float32 run; tighter tolerances do not guarantee monotonically
better float32 results.
The float64 check does not cover all 16 probes. An initial float64 configuration
with training oversampling 2 failed during cuFFT plan creation; its log is
retained under `1000000-double-oversampling2-failure`. The completed control
uses training oversampling 1.25.

The stricter million-point run spends **{million['seconds']['finite_cross_solves']:.2f} s**
solving the 48 cross-covariance RHS (16 probes × three vector components).
Each 16,384-frequency tail estimate then takes about
**{np.median([million['seconds'][f'tail_features_{s}'] for s in [71,72,73]]):.2f} s**.
This is validation cost on an NVIDIA L40, not an application speedup.
The feature diagnostic streams source/frequency blocks and never forms an
N-by-N matrix. Its work is O(N × frequencies × probes).

## What remains open

The theorem establishes `0 <= V_full - V_finite <= H`. The experiment estimates
H for a limited panel and checks feasible dense cases. It does not validate
the posterior mean or centering, complete ray/collision joint distributions,
sampling accuracy, hitbox geometry or sequential scans. Empirical intervals
are not simultaneous certificates. The original eight-draw timing pilot still
has insufficient samples to demonstrate probability convergence.

Simply adding the omitted prior to a finite solve produces covariance
`V_finite + H`, **not** the full posterior covariance. The derivation explains
why prior and inverse corrections must be considered together. No new
posterior sampling or FMM backend is claimed here.

## Reproduce

```bash
./reproduction/run_env.sh -m reproduction.audit_spectral_tail --dataset A
./reproduction/run_env.sh -m reproduction.audit_spectral_tail --dataset B
./reproduction/run_env.sh -m reproduction.audit_spectral_tail --dataset 1000000 --eps 1e-7
./reproduction/run_env.sh -m reproduction.audit_spectral_tail --dataset B --lengthscale .005 --noise .0004 --output reproduction/results/spectral_tail/B-short-low-noise
./reproduction/run_env.sh -m reproduction.audit_spectral_tail --dataset 1000000 --rtol 1e-5 --eps 1e-6 --output reproduction/results/spectral_tail/1000000-production
./reproduction/run_env.sh -m reproduction.audit_spectral_tail --dataset 1000000 --precision float64 --rtol 1e-9 --eps 1e-9 --probe-indices 0 8 15 --output reproduction/results/spectral_tail/1000000-double-control
./reproduction/run_env.sh -m reproduction.summarize_spectral_tail
```

The existing matched inputs, training-bandwidth fields and million-point cloud
are prerequisites; their generation is documented in the root reproduction
guide. Each directory includes results JSON and compact NPZ arrays with queries,
covariance matrices, frequencies, diagonal/trace samples and standard errors.
Full source weights and dense training matrices are deliberately not archived.
'''
    (ROOT / 'README.md').write_text(text)
    print(json.dumps(precision, indent=2))


if __name__ == '__main__':
    main()
