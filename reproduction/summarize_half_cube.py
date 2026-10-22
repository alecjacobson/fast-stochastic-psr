"""Publish every repetition and the predeclared eligible conditioning medians."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduction/results/half_cube'


def main():
    summary={'scope':'same-model conditioning speed and numerical checks; not complete-application or upstream SDD speedup','methods':{}}
    rows=[]
    for name,label in [('full150-six','Full cube150'),('half150-six','Folded cube150')]:
        d=json.loads((OUT/name/'results.json').read_text())
        if not d['complete'] or not d['quality']['fields']['pass']:
            raise RuntimeError(name+' missing numerical validation')
        events=d['quality']['events']
        if not events['compatibility_pass'] or not events['uncertainty_coverage_pass'] or d['analytic_holdout']['max_upper_endpoint']>=.001:
            raise RuntimeError(name+' application/residual screen failed')
        if d['quality']['comparison_kind']!='same model numerical comparison':
            raise RuntimeError('requires strict same-model comparison')
        trials=[]
        for i in range(d['args']['repeats']):
            key=f'solve_{i}';audit=d['timing_process_audits'][key]
            trials.append({'repeat':i,'seconds':d['seconds'][key],**audit})
        eligible=[r for r in trials if r['sampled_isolation_pass']]
        if len(eligible)<3:
            raise RuntimeError(name+' needs at least three eligible repetitions')
        median=float(np.median([r['seconds'] for r in eligible]))
        peak=max(r['sampled_process_memory_peak_bytes'] for r in eligible)
        summary['methods'][name]={'repetitions':trials,'eligible_count':len(eligible),
                                  'eligible_median_seconds':median,'sampled_process_memory_peak_bytes':peak,
                                  'quality':d['quality'],'analytic_holdout':d['analytic_holdout']}
        rows.append(f'| {label} | {len(eligible)} / {len(trials)} | {median:.2f} s | {peak/2**30:.2f} GiB |')
    full,half=[summary['methods'][n] for n in ['full150-six','half150-six']]
    summary['speedup']=full['eligible_median_seconds']/half['eligible_median_seconds']
    large=json.loads((OUT/'half150-128/results.json').read_text())
    if not large['complete'] or not large['quality']['fields']['pass']:
        raise RuntimeError('128-draw numerical validation missing')
    if not large['quality']['events']['compatibility_pass'] or not large['quality']['events']['uncertainty_coverage_pass'] or large['analytic_holdout']['max_upper_endpoint']>=.001:
        raise RuntimeError('128-draw application/residual screen failed')
    summary['larger_sample_check']={'quality':large['quality'],'analytic_holdout':large['analytic_holdout'],
                                   'seconds':large['median_solve_seconds'],
                                   'interval_audit':large['timing_process_audits']['solve_0']}
    large_eligible=large['timing_process_audits']['solve_0']['sampled_isolation_pass']
    timing=(f"The single 128-draw solve takes **{large['median_solve_seconds']:.2f} s** with sampled isolation."
            if large_eligible else f"The raw 128-draw solve takes {large['median_solve_seconds']:.2f} s; its timing is **excluded** because its process-audit interval did not pass.")
    smoke=json.loads((OUT/'production-draws/results.json').read_text())
    if not smoke['complete'] or not smoke['paired_prior_gate_pass'] or not smoke['residual_gate_pass']:
        raise RuntimeError('production sampler validation missing')
    data=np.load(OUT/'production-draws/fields.npz')
    panel=np.load(ROOT/'reproduction/results/spectral_tail/1000000/fields.npz')
    old=np.load(ROOT/'reproduction/results/training_bandwidth/scaling/fields_1000000.npz')
    ids=panel['query_ids']
    if not np.array_equal(data['query'],old['query'][ids].astype('f')):
        raise RuntimeError('mean-reference queries differ')
    reference=old['200_mean'][ids]
    summary['production_draws']={'paired_prior_rhs_relative_error':smoke['paired_prior_rhs_relative_error'],
        'paired_scalar_coefficients_relative_error':smoke['paired_scalar_coefficients_relative_error'],
        'maximum_independent_finite_residual':max(smoke['finite_residual_per_component']),
        'mean_relative_error_vs_full_cube200':float(np.linalg.norm(data['mean']-reference)/np.linalg.norm(reference)),
        'reference_mean_sha256':hashlib.sha256(reference.tobytes()).hexdigest()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    f=half['quality']['fields'];lf=large['quality']['fields'];sd=summary['production_draws']
    report=f'''# Full-cube symmetry: a measured conditioning gain

At one million observations, a positive frequency half plus a 2D zero-axis
plane reduces the eligible median for eight conditional draws from
**{full['eligible_median_seconds']:.2f} s to {half['eligible_median_seconds']:.2f} s ({summary['speedup']:.2f}×)**.
Every frequency of the finite training cube is retained. This is an incremental
conditioning improvement over our existing cube150 solver, not a new complete
application or upstream SDD ratio.

## Matched workload and repeat selection

The comparison keeps the cutoff200 cube prior and mean, scalar/cross cutoff50,
centered cutoff150 inverse, eight stored random RHS, source ordering, offset,
and ray/collision query geometry fixed. Only the inverse's numerical
representation changes. The underlying wider-prior/narrower-inverse hybrid
is the same as in the [earlier report](../centered_hybrid/README.md); its
full-analytic and application-convergence limitations remain.

| Implementation | Eligible / total repetitions | Eligible median | Sampled process-memory peak |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

The [protocol](PROTOCOL.md) declares six repetitions per implementation,
at least three eligible repetitions, and the median of **all** eligible
measurements. No selection by speed occurs. Each solve interval must have
preceding/following GPU-process observations, no foreign PID/error in that
bracket, and no sampling gap longer than two intervals. Whole-run audits are
also retained; a job outside a solve need not invalidate an independently
bracketed solve. Periodic sampling can miss short jobs and is not proof of
exclusive GPU use. No other project was stopped.

Memory is the largest observed own-process usage in eligible windows, sampled
at approximately one-second intervals and including bracketing observations.
It may miss short-lived allocations and is **not an exact allocator peak**.
Source transfer, plan creation, prior generation and later queries are outside
the solve timer and are recorded separately. The three-repetition preliminary
runs had whole-run contention and do not supply the reported ratio.

[Full records](full150-six/results.json) · [Folded records](half150-six/results.json) ·
[All repetitions and summary](summary.json).

## Numerical and larger-sample checks

This is a **same-model** comparison: the numerical gates stay at field error /
fluctuation norm <0.05% and empirical variance/covariance errors <0.01%.
The folded eight-draw covariance error is {100*f['empirical_covariance_relative_l2']:.6f}%.
Ray/collision outputs and selected view agree in the recorded paired check.
Analytic residual holdouts are separate from these same-model gates and retain
the existing 0.1% target. They still concern the underlying finite-prior hybrid.

The 128-draw comparison against the prior full cube150 run also passes: field
error / fluctuation norm is {100*lf['error_over_fluctuation_l2']:.6f}%, empirical
covariance error is {100*lf['empirical_covariance_relative_l2']:.6f}%, and the
maximum held-out analytic residual upper endpoint is
{100*large['analytic_holdout']['max_upper_endpoint']:.5f}%.
{timing}
This is one confirmation, not repeated 128-draw performance evidence. Both
the original and folded implementations use finite common priors; this check
does not establish absolute Monte Carlo convergence or equivalence to the
full analytic posterior. [128-draw records](half150-128/results.json).

## Coherent priors and a usable posterior class

`HalfCubePosterior` implements fitting, coherent finite-cube draws, optional
centered narrower solves, and the existing prepared-query cache. Small tests
compare joint vector/scalar and posterior moments with independent explicit
Fourier sums, including all three possible folding axes, partial batches and
cached evaluation of non-Hermitian coefficient arrays. Action/CG tests also
cover odd/even widths, periodic seams, zero modes and float32/64 arithmetic.
See [complete test output](tests.log).

A production cutoff200 check generates four coherent draws at one million
observations. Rebuilding its first canonical prior on a complete 3D grid gives
RHS discrepancy {100*sd['paired_prior_rhs_relative_error']:.6f}% and scalar
coefficient discrepancy {100*sd['paired_scalar_coefficients_relative_error']:.6f}%.
Independent direct finite-covariance rows give maximum residual
{100*sd['maximum_independent_finite_residual']:.6f}% against the 0.002% gate.
Its mean differs from the saved full cube200 mean by
{100*sd['mean_relative_error_vs_full_cube200']:.6f}% on the same 16 probes.
Four draws and 512 validation rows are an execution/numerical check, not
posterior probability convergence. [Production records](production-draws/results.json).

For real weights, negative folded frequencies are the conjugates of the
positive half. Matrix actions add twice the positive half's real part plus
the zero-axis plane. Prior coefficients instead use amplitude `sqrt(2*rho)`
on the positive half and `sqrt(rho)` on the plane. The scalar low modes share
those same Gaussian draws. Negative-half scalar coefficients may be zero;
taking the real Fourier sum supplies the conjugate contribution. Seeded
realizations need not match a separately generated full-cube sampler.

```python
from reproduction.half_posterior import HalfCubePosterior

model = HalfCubePosterior(points, normals, training_cutoff=200)
model.draw(128, batch=1, solve_cutoff=150)  # existing centered hybrid law
model.prepare_query_cache()
mean = model.mean(queries)
samples = model.samples(queries)
```

Omit `solve_cutoff` for the consistently coupled finite-cube posterior.
Changing the inverse band is a separate modeling approximation; folding alone
does not change it. Dense/compiled-SDD references still win some smaller
workloads. There is no claim that this representation restores omitted analytic
frequencies or that one million points are intractable for the authors' SDD.
[Issue #26](https://github.com/alecjacobson/fast-stochastic-psr/issues/26) tracks
remaining full-workload coverage; [#27](https://github.com/alecjacobson/fast-stochastic-psr/issues/27)
proposes packing real RHS in full complex transforms as another speed trial.

## Reproduce

Use the existing common-RHS scratch dataset and prior reference artifacts.
Raw measurements here use NVIDIA L40, float32, CG rtol1e-5 and NUFFT eps1e-6.

```bash
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 8 --batch 1 --repeats 6 --centered-fluctuations --solve-cutoff 150 --audit-analytic --stop-rows 2048 --holdout-rows 4096 --holdout-seed 2711 --scratch /opt/geospsr-runtime/common-rhs-128 --reference reproduction/results/centered_hybrid/million-150 --output reproduction/results/half_cube/full150-six
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 8 --batch 1 --repeats 6 --centered-fluctuations --solve-cutoff 150 --half-cube --audit-analytic --stop-rows 2048 --holdout-rows 4096 --holdout-seed 2711 --scratch /opt/geospsr-runtime/common-rhs-128 --reference reproduction/results/half_cube/full150-six --output reproduction/results/half_cube/half150-six
./reproduction/run_env.sh -m reproduction.check_half_draws --output reproduction/results/half_cube/production-draws
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 128 --batch 1 --repeats 1 --centered-fluctuations --solve-cutoff 150 --half-cube --audit-analytic --stop-rows 2048 --holdout-rows 4096 --holdout-seed 2711 --scratch /opt/geospsr-runtime/common-rhs-128 --reference reproduction/results/centered_hybrid/million-150-128 --output reproduction/results/half_cube/half150-128
./reproduction/run_env.sh -m pytest tests -q
./reproduction/run_env.sh -m reproduction.summarize_half_cube
```
'''
    (OUT/'README.md').write_text(report)
    print(json.dumps({'speedup': summary['speedup'], 'eligible_repetitions': {k: v['eligible_count'] for k, v in summary['methods'].items()}, '128_draw_timing_eligible': large_eligible}, indent=2))


if __name__=='__main__':
    main()
