"""Publish preconditioner setup/solve costs, quality and retained regressions."""
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'results/preconditioning'


def main():
    names=['baseline8','block64','block128','block256','B-baseline4','B-block1284']
    runs={name:json.loads((OUT/name/'results.json').read_text()) for name in names}
    assert all(r['complete'] for r in runs.values())
    rows=[];quality_rows=[];summary={'scope':'fixed finite-cube conditioning only','runs':{}}
    for name,r in runs.items():
        args=r['args'];small=name.startswith('B-')
        assert r['N']==(17419 if small else 1_000_000)
        assert args['samples']==(4 if small else 8) and args['batch']==2 and args['repeats']==3
        assert args['pack_rhs'] and args['solve_cutoff']==200 and args['rtol']==1e-5
        assert not args['centered_fluctuations'] and not args['initial_mean']
        with np.load(OUT/name/'fields.npz') as fields:
            assert fields['fields'].shape==(args['samples'],257)
        iterations=[i['iterations'] for repeat in r['iterations'] for i in repeat]
        assert all(i['converged'] for repeat in r['iterations'] for i in repeat)
        setup=r['seconds'].get('preconditioner_setup',0.)
        solve=float(np.median([r['seconds'][f'solve_{i}'] for i in range(3)]))
        item={'setup_seconds':setup,'median_solve_seconds':solve,'setup_plus_solve_seconds':setup+solve,
              'iteration_range':[min(iterations),max(iterations)],'sampled_isolation_pass':r['gpu_process_audit']['sampled_isolation_pass'],
              'preconditioner_storage_bytes':r.get('preconditioner_storage_bytes',0)}
        if 'quality' in r:
            q=r['quality'];assert q['comparison_kind']=='same model numerical comparison'
            ref=runs['B-baseline4' if name.startswith('B-') else 'baseline8']
            assert r['source_manifest']['hashes']==ref['source_manifest']['hashes']
            assert r['geometry_sha256']==ref['geometry_sha256']
            assert r['args']['solve_cutoff']==ref['args']['solve_cutoff']==200
            assert not r['args']['centered_fluctuations']
            item['quality_pass']=bool(q['fields']['pass'] and q['events']['compatibility_pass']
                                     and q['events']['uncertainty_coverage_pass'] and r['independent_finite_residual']['pass'])
            item['quality']=q;item['independent_finite_residual']=r['independent_finite_residual']
            quality_rows.append(f'| {name} | {100*q["fields"]["error_over_fluctuation_l2"]:.6f}% | '
                                f'{100*q["fields"]["empirical_covariance_relative_l2"]:.6f}% | '
                                f'{100*r["independent_finite_residual"]["maximum"]:.6f}% | {item["quality_pass"]} |')
        summary['runs'][name]=item
        rows.append(f'| {name} | {setup:.3f} | {solve:.3f} | {setup+solve:.3f} | '
                    f'{min(iterations)}–{max(iterations)} | {item["preconditioner_storage_bytes"]/2**20:.1f} | '
                    f'{item["sampled_isolation_pass"]} |')
    eligible=all(r['sampled_isolation_pass'] for r in summary['runs'].values())
    summary['timing_ratio_eligible']=eligible
    if eligible:
        summary['setup_plus_solve_ratios']={name:summary['runs']['B-baseline4' if name.startswith('B-') else 'baseline8']['median_solve_seconds']/v['setup_plus_solve_seconds']
                                            for name,v in summary['runs'].items() if 'quality_pass' in v and v['quality_pass']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    pilot_names=['block64','block128','block256']
    no_iteration_gain=all(summary['runs'][n]['iteration_range'][0]>=summary['runs']['baseline8']['iteration_range'][1] for n in pilot_names)
    headline=('**The tested spatial blocks do not reduce million-point iteration counts.** They do not establish a conditioning speedup; the default solver remains unchanged.' if no_iteration_gain else 'The table below retains every candidate and its setup, iteration and accuracy tradeoffs; any speed claim must satisfy the timing audit.')
    followup=('No candidate reduced million-point iterations, so this pilot does not justify a 128-draw timing expansion. Next experiments should test a shared coarse space or deflation rather than assume larger ensembles will remove per-solve overhead.' if no_iteration_gain else 'Any selected candidate still needs larger-sample and coherent-producer validation before an application speed claim.')
    result=('All whole-run audits pass; eligible conditioning ratios appear in summary.json.' if eligible else
            '**No timing ratio is claimed:** the declared whole-run isolation requirement fails. Timings remain diagnostics, and no repetition is dropped.')
    report=f'''# Reusable spatial preconditioning for posterior draws

{headline}

This experiment adds optional spatial block Jacobi to packed full-cube200 CG.
It compares the **same covariance operator, right-hand sides, prior coefficients
and query geometry**. The analytic block matrices are used only as an inverse
preconditioner; they do not replace the finite training covariance or change
the posterior prior. The mean is held fixed. All three repetitions and every
candidate in the [declared protocol](PROTOCOL.md) are retained.

## Setup and conditioning

NVIDIA L40, float32, NUFFT epsilon1e-6, CG tolerance1e-5, noise variance .02,
length scale .01, per-axis variance .1. Million-point runs condition eight common
draws in batches of two; B runs condition four draws at 17,419 observations.
The three solve repetitions reuse each preconditioner, while the setup-plus-solve
column charges construction once for each map's first ensemble. Source transfers,
NUFFT plan setup, mean fitting, prior generation and downstream queries are
outside these conditioning totals.

| Run | Preconditioner setup (s) | Median solves (s) | Setup + solves (s) | Iterations/batch | Retained preconditioner (MiB) | Whole-run sampled isolation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

{result}

{followup}

On B, block128 reduces iterations from 11 to 9; recorded setup-plus-solve time
is 1.831 s versus 1.976 s without preconditioning. This modest small-input
improvement does not carry over to the million-point case and does not establish
that NUFFT is preferable to the existing dense/SDD alternatives on B.

Periodic process samples cannot prove exclusive device access. Other jobs were
not stopped. Retained preconditioner storage includes inverse blocks and the
permutation; it excludes construction scratch, NUFFT plans, residuals and source
arrays. Raw process observations and stage timings are in each run's results.
Do not compare these full-cube200 conditioning numbers with earlier centered150
hybrid timings as if they used the same inverse model.

## Numerical evidence

| Candidate | Field error / fluctuations | Empirical covariance discrepancy | Independent finite-row residual | All numerical/event gates |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(quality_rows)}

The unchanged same-model gates require field/fluctuation error below .05%,
empirical variance and covariance discrepancies below .01%, and the existing
paired ray/collision/next-view gates. Independent double-precision cosine sums
check 32 finite-covariance rows at seed20261023; their residual threshold is
.002%. These sampled rows supplement the full NUFFT true residual, but do not
certify the full analytic kernel. Raw quality records retain all failures.

Small independent tests compare analytic preconditioner blocks with long
Fourier sums, and PCG solutions with dense finite-kernel solves. They cover
periodic seam coordinates, heteroscedastic noise, partial blocks, zero RHS,
initial-solution preservation and rejection of nonpositive preconditioners.
[Full test output](tests.log).
The production experiment uses homoscedastic noise; this does not validate
ContactFusion's observation model or sequential prior.

## Implementation and limits

`SpatialBlockJacobi` uses a deterministic Morton ordering of periodic point
coordinates and contiguous groups. It builds analytic principal blocks in
bounded batches, adds observation variances, checks Cholesky factorization and
stores symmetrized inverses. Construction uses double precision, application
uses the requested solve dtype. No silent jitter is added. Padding in the last
block is decoupled from actual observations. The grouping can split neighbors
at block or periodic seams; it is a heuristic, not a nearest-neighbor guarantee.

`solve(..., preconditioner=...)` opts into independent PCG recurrences for full,
folded or packed real systems. The existing unpreconditioned path is unchanged.
PCG checks positive operator/preconditioner curvature and reports the residual
against the original covariance action and original RHS norm. A converged
recurrence alone is insufficient. Explicit inverse storage costs O(Nb), and
local blocks need not resolve long-range modes. More elaborate overlapping or
coarse-space preconditioners must include their additional setup/application
cost and preserve the same posterior model.

These are solver experiments on eight/four shared draws, not independent
probability convergence studies or complete applications. The
[collision discretization failures](../query_refinement/README.md), absolute
Monte Carlo accuracy and full analytic-tail/scalar-band convergence remain
open. B's established dense/SDD alternatives are faster than its NUFFT fits;
this comparison does not establish NUFFT as the right small-input method.

## Reproduction

Use the [common-RHS preparation](../nufft_batches/README.md), with scratch outside
the git checkout. Run the baseline first, then each block size in a fresh process:

```bash
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \\
  --samples 8 --batch 2 --repeats 3 --pack-rhs \\
  --scratch /opt/geospsr-runtime/common-rhs-128 \\
  --output /opt/geospsr-runtime/preconditioning/baseline8
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \\
  --samples 8 --batch 2 --repeats 3 --pack-rhs --preconditioner-block 128 \\
  --finite-rows 32 --scratch /opt/geospsr-runtime/common-rhs-128 \\
  --reference /opt/geospsr-runtime/preconditioning/baseline8 \\
  --output /opt/geospsr-runtime/preconditioning/block128
```

Repeat the candidate command for 64 and 256, keeping every output. For B use
`--dataset B --samples 4 --scratch /opt/geospsr-runtime/common-rhs-smoke`, with
its own baseline and block128 reference comparison. Compact fields and raw
results are published here. The summary generator reads those published files:
`./reproduction/run_env.sh -m reproduction.summarize_preconditioning`.
[Issue #30](https://github.com/alecjacobson/fast-stochastic-psr/issues/30) tracks
shared preconditioning; [#6](https://github.com/alecjacobson/fast-stochastic-psr/issues/6)
tracks broader conditioning/FMM approaches.
'''
    (OUT/'README.md').write_text(report)
    print(json.dumps({name:{k:v for k,v in r.items() if k not in ['quality','independent_finite_residual']} for name,r in summary['runs'].items()}))


if __name__=='__main__':main()
