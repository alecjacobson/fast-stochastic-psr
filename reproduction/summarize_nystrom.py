"""Report every declared landmark preconditioner pilot, including failures."""
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'results/nystrom_preconditioning'


def main():
    names=['baseline8','rank128','rank512','rank1024']
    raw={name:json.loads((OUT/name/'results.json').read_text()) for name in names}
    baseline=raw['baseline8'];assert baseline['complete']
    records={};rows=[];checks=[]
    for name,r in raw.items():
        a=r['args'];assert a['samples']==8 and a['batch']==2 and a['repeats']==3
        assert a['solve_cutoff']==200 and a['pack_rhs'] and a['rtol']==1e-5
        assert not a['centered_fluctuations'] and not a['initial_mean']
        assert r['source_manifest']['hashes']==baseline['source_manifest']['hashes']
        assert r['geometry_sha256']==baseline['geometry_sha256']
        record={'complete':r['complete'],'sampled_isolation_pass':r['gpu_process_audit']['sampled_isolation_pass']}
        if not r['complete']:
            record['failure']=r.get('failure','incomplete')
            records[name]=record;rows.append(f'| {name} | failed | — | — | — | — | — |');continue
        assert r['N']==1_000_000
        assert all(v['converged'] for repeat in r['iterations'] for v in repeat)
        with np.load(OUT/name/'fields.npz') as fields:assert fields['fields'].shape==(8,257)
        iterations=[v['iterations'] for repeat in r['iterations'] for v in repeat]
        solve=float(np.median([r['seconds'][f'solve_{i}'] for i in range(3)]))
        setup=r['seconds'].get('preconditioner_setup',0.)
        record.update(setup_seconds=setup,median_solve_seconds=solve,setup_plus_solve_seconds=setup+solve,
                      iteration_range=[min(iterations),max(iterations)],storage_bytes=r.get('preconditioner_storage_bytes',0),
                      finite_residual=r['independent_finite_residual'])
        record['quality_pass']=r['independent_finite_residual']['pass']
        if name!='baseline8':
            q=r['quality'];assert q['comparison_kind']=='same model numerical comparison'
            record['quality_pass']=bool(record['quality_pass'] and q['fields']['pass'] and q['events']['compatibility_pass'] and q['events']['uncertainty_coverage_pass'])
            record['quality']=q;record['preconditioner']=r['preconditioner']
            m=r['preconditioner']
            checks.append(f'| {name} | {m["retained_rank"]} | {m["orthogonality_frobenius_error"]:.2e} | '
                          f'{100*q["fields"]["error_over_fluctuation_l2"]:.6f}% | '
                          f'{100*q["fields"]["empirical_covariance_relative_l2"]:.6f}% | '
                          f'{100*r["independent_finite_residual"]["maximum"]:.6f}% | {record["quality_pass"]} |')
        records[name]=record
        rows.append(f'| {name} | {setup:.3f} | {solve:.3f} | {setup+solve:.3f} | {min(iterations)}–{max(iterations)} | '
                    f'{record["storage_bytes"]/2**30:.3f} | {record["sampled_isolation_pass"]} |')
    eligible=all(r['complete'] and r.get('quality_pass') and r['sampled_isolation_pass'] for r in records.values())
    summary={'scope':'same finite-model conditional solve pilot; no complete application or upstream ratio',
             'runs':records,'timing_ratio_eligible':eligible}
    if eligible:
        summary['setup_plus_solve_speedups']={n:records['baseline8']['median_solve_seconds']/r['setup_plus_solve_seconds'] for n,r in records.items() if n!='baseline8'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    timing=('All four whole-run sampled-isolation and numerical checks pass; conditioning ratios including setup are in [summary.json](summary.json).' if eligible else
            '**No timing ratio is claimed:** at least one declared whole-run isolation or quality requirement fails. All timings are retained as diagnostics; no favorable repetitions are selected.')
    largest=records['rank1024'];middle=records['rank512'];base=records['baseline8']
    headline='The retained sweep includes a failed numerical or execution outcome; inspect the table before interpreting performance.'
    tradeoff=''
    if largest['complete'] and largest.get('quality_pass'):
        headline=(f'**Rank 1,024 reduces million-point CG iterations from {base["iteration_range"][0]} to '
                  f'{largest["iteration_range"][0]}** in this pilot, with numerical and paired-event gates passing. '
                  f'It retains {largest["storage_bytes"]/2**30:.3f} GiB of basis data and costs '
                  f'{largest["setup_seconds"]:.3f} s to construct. Larger sample-count and coherent-producer validation remain open.')
    if largest['complete'] and middle['complete']:
        tradeoff=(f'Rank 512 recorded setup-plus-eight-draw cost is {middle["setup_plus_solve_seconds"]:.3f} s '
                  f'versus {largest["setup_plus_solve_seconds"]:.3f} s for rank 1,024. '
                  'Compare repeated solve cost, setup and storage separately; value for larger ensembles needs direct measurement. '
                  'These are diagnostics, not timing-ratio claims.')
    (OUT/'README.md').write_text(f'''# Shared landmark Nyström preconditioning

{headline}

Follow-up: [128-draw and coherent-streaming validation](../nystrom_validation/README.md)
now passes the numerical gates, including 120 pilot-held-out draws. It also
documents a legacy source-hash correction and a low-noise precision failure.
Qualified timing and complete application convergence remain open.

This pilot tests a global coarse correction after
[disjoint spatial blocks](../preconditioning/README.md) failed to reduce
million-point iteration counts. It preserves the packed full-cube200 covariance
action, original right-hand sides, prior coefficients, scalar cutoff50, solver
tolerance and query geometry. Only the inverse preconditioner changes.
The [protocol](PROTOCOL.md) fixes ranks 128, 512 and 1024 before running the sweep.

## Setup, conditioning and memory

NVIDIA L40; one million observations; eight common conditional draws in batches
of two; three solve repeats per configuration, with one shared setup. Noise
variance .02, length scale .01, per-axis variance .1, NUFFT epsilon1e-6 and
CG tolerance1e-5. Setup-plus-solve charges construction once for the first
ensemble on a map. Source transfer, transform plans, prior generation, mean
fitting, scalar aggregation and queries are outside these conditioning totals.

| Run | Setup (s) | Median solves (s) | Setup + solves (s) | Iterations/batch | Retained preconditioner (GiB) | Whole-run sampled isolation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

{timing}
Process sampling can miss brief overlaps and does not prove exclusive GPU use.
Other jobs were not stopped. Retained basis storage excludes construction
scratch, source arrays, CG vectors and NUFFT workspace. Raw results also retain
stage memory and sampled process observations. A larger rank can reduce
iterations while increasing time per iteration or setup cost; iteration count
alone is not an application speedup.

{tradeoff}
[Issue #31](https://github.com/alecjacobson/fast-stochastic-psr/issues/31)
tracks compressing, streaming or avoiding the observation-by-rank basis before
scaling beyond one million inputs.

## Numerical controls

| Candidate | Retained rank | Basis orthogonality error | Field error / fluctuations | Covariance discrepancy | Independent finite-row residual | Numerical/event gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(checks)}

Independent double-precision cosine sums check 32 covariance rows at seed20261023,
with a .002% residual gate, in addition to full NUFFT true residuals. Same-model
field error must be below .05% of the fluctuation norm; empirical variance and
covariance discrepancies below .01%. Paired ray/collision/next-view gates are
unchanged. These are numerical controls on eight common draws. They do not
establish absolute probability accuracy, continuous geometry, full analytic
training-tail or scalar-band convergence, or ContactFusion update semantics.
The [known collision sampling failure](../query_refinement/README.md) remains.

Tests compare covariance columns with independent long Fourier sums, the
preconditioner with a dense pseudoinverse/eigendecomposition, and conditioned
weights with an independent dense finite solve. Duplicate landmarks, periodic
seam coordinates, partial construction batches, both precisions, zero RHS and
invalid noise are covered. [Full test output](tests.log).

## Construction and application

Let C = K(X,L), W = K(L,L), and C W† Cᵀ = U diag(λ) Uᵀ. We apply
P⁻¹ = I − U diag(1 − (λ_min + σ)/(λ + σ)) Uᵀ, where σ is noise variance.
This spectral scaling follows Eq. 5.3 of
[Frangella, Tropp and Udell (2023)](https://tropp.caltech.edu/papers/FTU23-Randomized-Nystrom-SIMAX.pdf).
Our landmarks are uniform sampled columns, not a Gaussian sketch; the paper
distinguishes these constructions in §2.2.3. Its strongest Gaussian-sketch
guarantees do not automatically apply to this implementation.

The analytic product-Matern kernel constructs only P; PCG continues to solve
the original finite-cube system. Replacing the solve with the low-rank model
would change the posterior and is not performed here. Homoscedastic noise is
required by this implementation; heteroscedastic whitening remains separate.

Construction uses double precision in row blocks of 8192, NumPy landmark seed 41,
and relative eigenvalue cutoff 1e-10 in the two small Gram decompositions.
Discarded eigenmodes affect only the preconditioner. No silent jitter is added.
Setup also holds an N-by-rank double-precision feature array while constructing
the float32 basis; row batching bounds pairwise temporaries, not that array.
This pilot uses one landmark seed, and landmark sets at different ranks are not
required to be nested. It does not establish a seed-robust optimal rank.
The stored float32 basis is rechecked in double precision: the implementation
requires 1 − max(damping)·(1 + ‖UᵀU − I‖F) > 0. This is a numerical SPD guard,
not an interval-arithmetic proof. Setup includes the check. Eigenvalue ranges,
retained ranks, landmark hashes and approximate/analytic covariance traces are
recorded; trace capture is not posterior-error certification.

## Reproduction

Use the existing [common-RHS data preparation](../nufft_batches/README.md).
Run the following baseline, then each rank in a fresh process:

```bash
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \\
  --samples 8 --batch 2 --repeats 3 --pack-rhs --finite-rows 32 \\
  --scratch /opt/geospsr-runtime/common-rhs-128 \\
  --output /opt/geospsr-runtime/nystrom-preconditioning/baseline8
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \\
  --samples 8 --batch 2 --repeats 3 --pack-rhs --finite-rows 32 \\
  --preconditioner-rank 512 --scratch /opt/geospsr-runtime/common-rhs-128 \\
  --reference /opt/geospsr-runtime/nystrom-preconditioning/baseline8 \\
  --output /opt/geospsr-runtime/nystrom-preconditioning/rank512
```

Repeat for ranks 128 and 1024, preserving all results. After copying compact
results/fields into this directory, regenerate this report with
`./reproduction/run_env.sh -m reproduction.summarize_nystrom`.
[Issue #30](https://github.com/alecjacobson/fast-stochastic-psr/issues/30) tracks
larger-sample validation, shared coarse spaces and setup/amortization costs.
''')
    print(json.dumps({n:{k:v for k,v in r.items() if k not in ['quality','finite_residual','preconditioner']} for n,r in records.items()}))


if __name__=='__main__':main()
