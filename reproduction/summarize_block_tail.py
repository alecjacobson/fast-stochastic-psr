"""Summarize stratified tail diagnostics and retained centered-block pilots."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reproduction/results/block_tail'


def main():
    summary = {'scope':'fixed query covariance diagnostics and eight-draw centered-inverse screening; no complete-application speed claim','cases':{}}
    rows = []
    for name,label in [('B-default','B, default'),('B-stress','B, stress'),('million-100-200','1,000,000, default')]:
        record = json.loads((OUT/name/'results.json').read_text())
        data = np.load(OUT/name/'fields.npz')
        if not record['complete']:
            raise RuntimeError(name+' incomplete')
        seeds = record['args']['seeds']
        finite = data['finite_covariance'];scale=np.linalg.norm(finite)
        estimate = sum(data[f'{s}_bound_estimate'] for s in seeds)/len(seeds)
        traces,variances,diagonals,diag_variances = {},[],[],[]
        for region in ['corners','outside']:
            trace = np.concatenate([data[f'{s}_{region}_trace_samples'] for s in seeds])
            diagonal = np.concatenate([data[f'{s}_{region}_diagonal_samples'] for s in seeds])
            traces[region]=float(trace.mean()/scale)
            variances.append(float(trace.var(ddof=1)/len(trace)/scale**2))
            diagonals.append(diagonal.mean(0));diag_variances.append(diagonal.var(0,ddof=1)/len(diagonal))
        se = np.sqrt(sum(variances));trace=sum(traces.values())
        row = {'core':record['args']['core'],'extent':record['args']['extent'],
               'trace_estimate':trace,'trace_standard_error':float(se),'trace_by_region':traces,
               'corner_fraction_of_trace':traces['corners']/trace,
               'estimated_bound_frobenius_ratio':float(np.linalg.norm(estimate)/scale),
               'maximum_variance_estimate_fraction':float(np.max(sum(diagonals)/np.diag(finite))),
               'maximum_variance_marginal_normal_95_upper_fraction':float(np.max((sum(diagonals)+1.96*np.sqrt(sum(diag_variances)))/np.diag(finite))),
               'sampled_isolation_pass':record['gpu_process_audit']['sampled_isolation_pass']}
        if 'exact' in record:
            row['actual_covariance_gap']=record['exact']['covariance_gap_relative_frobenius']
            row['exact_bound_frobenius_ratio']=record['exact']['bound_relative_frobenius']
            row['maximum_stratum_diagonal_error_standard_errors']=max(
                r['strata'][k]['maximum_diagonal_error_standard_errors'] for r in record['seeds'] for k in ['corners','outside'])
        summary['cases'][name]=row
        actual='not computed' if 'actual_covariance_gap' not in row else f"{100*row['actual_covariance_gap']:.6f}"
        rows.append(f"| {label} | {row['core']} / {row['extent']} | {actual} | {100*row['estimated_bound_frobenius_ratio']:.6f} | {100*trace:.6f} ± {196*se:.6f} | {100*row['corner_fraction_of_trace']:.1f}% |")
    if not json.loads((OUT/'million-double/results.json').read_text())['complete']:
        raise RuntimeError('double control incomplete')
    small = np.load(OUT/'million-double/fields.npz')
    full = np.load(OUT/'million-100-200/fields.npz')
    ids = small['probe_indices'];ref=small['finite_covariance']
    approximate = full['finite_covariance'][np.ix_(ids,ids)]
    summary['numerical_control']={'probe_indices':ids.tolist(),
        'covariance_relative_frobenius':float(np.linalg.norm(approximate-ref)/np.linalg.norm(ref)),
        'maximum_variance_relative_difference':float(np.max(np.abs(np.diag(approximate-ref))/np.diag(ref)))}
    screen=json.loads((OUT/'corner-screen.json').read_text())
    summary['corner_screen']=screen
    screen_rows=[]
    for candidate in screen['candidates']:
        if candidate['edge'] in [200,250,300]:
            screen_rows.append(f"| {candidate['edge']} | {candidate['added_conjugate_paired_stored_modes']:,} | {100*candidate['captured_corner_trace_fraction']:.1f}% | {100*candidate['captured_total_trace_fraction']:.1f}% |")
    pilot_rows=[];summary['centered_pilots']={}
    for name,core in [('centered-million-50',50),('centered-million-100',100)]:
        d=json.loads((OUT/name/'results.json').read_text())
        if not d['complete']:
            raise RuntimeError(name+' incomplete')
        f=d['quality']['fields'];h=d['analytic_holdout'];iso=d['gpu_process_audit']['sampled_isolation_pass']
        row={'core':core,'seconds':d['median_solve_seconds'],'timing_eligible':iso,
             'field_gates_pass':f['pass'],'empirical_covariance_relative_l2':f['empirical_covariance_relative_l2'],
             'analytic_residual_max_upper_endpoint':h['max_upper_endpoint'],
             'event_compatibility_pass':d['quality']['events']['compatibility_pass'],
             'uncertainty_coverage_pass':d['quality']['events']['uncertainty_coverage_pass']}
        summary['centered_pilots'][name]=row
        timing=f"{row['seconds']:.2f} s"+(' (single run)' if iso else ' (excluded: contention)')
        pilot_rows.append(f"| {core} / 200 | {100*row['empirical_covariance_relative_l2']:.4f}% | {100*row['analytic_residual_max_upper_endpoint']:.4f}% | {'Pass' if f['pass'] else '**Fail**'} | {timing} |")
    control=summary['centered_pilots']['centered-million-100']
    if not control['field_gates_pass'] or not control['timing_eligible'] or control['seconds']<=38.14:
        raise RuntimeError('pilot100 outcome changed; revise the report conclusions before publishing')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    numerical=summary['numerical_control'];stress=summary['cases']['B-stress'];large=summary['cases']['million-100-200']
    report=f'''# All omitted block frequencies: covariance diagnostics and conditioning pilots

The positive-block backend now has an error diagnostic that includes **both
inside-cube corners and the infinite outer tail**. Dense controls validate the
two contributions independently. At one million observations, the fixed-panel
trace-bound estimate is **{100*large['trace_estimate']:.5f}%** of the finite posterior
covariance Frobenius norm. This is an empirical covariance diagnostic, not a
full application accuracy certificate or a speedup.

## Covariance overlap and massive-input panel

All runs retain scalar/cross cutoff 50 and analytic normalization. Default
parameters are length scale .01, noise variance .02; stress uses .005 and .0004.
Per-axis kernel variance is .1. Inputs and the previously selected 16 queries
are fixed; hashes are in each result. Each of three seeds (71–73) draws 16,384
frequencies **per region**, for 49,152 corner and 49,152 outer frequencies after
pooling. These are validation frequencies, not posterior sample counts.

| Input | Core / extent | Actual covariance gap % | Estimated H norm % | Estimated trace bound % ± 1.96 SE | Corners' share of trace |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The exact theorem is `0 <= V_full - V_block <= H_corner + H_outer`, a PSD
matrix inequality. Consequently `||gap||_F <= ||H||_F <= trace(H)`. The table
normalizes these quantities by `||V_block||_F`. Estimated H norms and traces
are not certified bounds. The reported interval applies only to the trace;
it is an empirical normal approximation, not rigorous or simultaneous coverage.
For the stress case, the trace is more conservative than the matrix norm:
the actual gap and exact H norm pass the earlier 0.02% covariance gate even
though the trace exceeds it. Neither that gate nor the table certifies every
joint direction or complete application probabilities.

At stress 150/800, corners contribute **{100*stress['corner_fraction_of_trace']:.1f}%**
of the trace bound. An outside-cube-only estimator would miss most of it.
[Issue #25](https://github.com/alecjacobson/fast-stochastic-psr/issues/25) tracks
whether adding selected two-high-axis blocks can improve the accuracy/cost
tradeoff. More transforms may erase any storage benefit.

[Default B records](B-default/results.json) · [Stress B records](B-stress/results.json) ·
[Million-point records](million-100-200/results.json) · [Method](../spectral_tail/METHOD.md).
Raw NPZ files retain covariance matrices, frequency draws and per-frequency
diagonal/trace samples. Small tests also verify the corner distribution by
explicit enumeration, stratum masses and independent feature quadratics.

## Numerical control and limitations

Dense controls use float64 direct separable covariance rows and Cholesky; no
extent-cubed reference spectrum is allocated. At large N the production run
uses float32, NUFFT eps 1e-6 and CG rtol 1e-5. A separate float64 run at eps/rtol
1e-9 checks probes {numerical['probe_indices']}. Its finite covariance differs
by **{100*numerical['covariance_relative_frobenius']:.6f}%** in relative Frobenius
norm, with maximum variance difference
**{100*numerical['maximum_variance_relative_difference']:.6f}%**.
This covers three probes, not all sixteen or a global numerical-error bound.
The feature diagnostic uses float64 phases/outer accumulation and float32
feature products in both runs. [Control records](million-double/results.json).

The million-point maximum pooled variance-bound estimate is
{100*large['maximum_variance_estimate_fraction']:.5f}%; taking the maximum of
marginal normal interval endpoints gives
{100*large['maximum_variance_marginal_normal_95_upper_fraction']:.5f}% and does
**not** provide simultaneous 95% coverage. Mean/centering, sample convergence,
full slices, ray/collision geometry, hitboxes and sequential decisions remain
separate requirements. This code does not add the omitted prior or solve the
full analytic posterior. The goal remains open.

## CPU screen for a positive corner refinement

Using the saved stress-panel frequencies and **unchanged old inverse weights**,
we screen disjoint two-high-axis boxes. Each added pair of high coordinates
extends from core150 to the selected edge; the third coordinate remains in
the core. Triple-high frequencies stay omitted. The partition is checked by
independent small-grid enumeration; it has no GPU backend yet.

| Edge | Added conjugate-paired stored modes | Captured corner trace contribution | Captured total trace contribution |
| --- | ---: | ---: | ---: |
{chr(10).join(screen_rows)}

These are contributions to the **existing diagnostic**, not predicted new
posterior errors. A new inverse changes its weights; the screening panel also
cannot serve as independent validation. Each nonempty refinement adds six
positive transform pairs, so repeated source spreading may outweigh the mode
count benefit. [Screening record](corner-screen.json); implementation and
matched-quality timing remain under issue #25.

## Centered block-inverse screening

A separate experiment keeps the cutoff200 cube prior, observed normals,
fitted mean, centering, eight stored random RHS and application geometry fixed.
Only the inverse applied to zero-mean fluctuations changes. This is a hybrid
law, different from the consistently finite block model audited above. Its
quality is assessed against the existing centered cube200 shared-draw control;
the tail diagnostic alone does not validate it.

| Inverse core / extent | Empirical covariance error | Analytic residual upper endpoint | Field gates | Eight-draw solve time |
| --- | ---: | ---: | --- | --- |
{chr(10).join(pilot_rows)}

The existing covariance gate is 0.02%; it is unchanged. Analytic residuals use
4,096 independent held-out rows, excluding the 2,048 stopping rows; their
reported upper endpoints are empirical marginal intervals with a 0.1% gate.
An identical selected view or sampled event pattern cannot override failed
covariance checks. Eight draws do not establish event-probability convergence.
These are single-run screens, not repeated performance measurements. Timings
with foreign GPU activity are retained and excluded; no other job was stopped.
No new speedup is claimed. The earlier clean three-repetition cube150 result
is 38.14 s for the same eight RHS, with its own quality evidence in the
[centered-hybrid report](../centered_hybrid/README.md).
The block100 pilot passes the field and residual gates but takes {control['seconds']:.2f} s in
its isolated single run, so it offers no speed advantage in this screen.
Reduced spectral storage did not translate into a faster conditional solve.
[Issue #26](https://github.com/alecjacobson/fast-stochastic-psr/issues/26)
proposes a different next trial: exploit conjugacy within the complete cube,
preserving the finite model while reducing the main FFT grid.

The [dense B block50 hybrid](centered-B-50/results.json) passes the fixed-panel
moment gates, but this alone does not predict million-point finite-ensemble
agreement. The CLI exposes `--block-core` only for centered fluctuations and
requires `50 <= core <= solve-cutoff <= 200`, preserving the held prior.
All pilot results, including failures, are retained.
The complete regression-suite output is archived in [tests.log](tests.log).

## Reproduce

Prerequisites are the matched B data, saved spectral-tail panels, million-point
cloud and common RHS scratch data described in the earlier reproduction reports.
Outputs may live under /opt scratch first to avoid filling the checkout disk.

```bash
./reproduction/run_env.sh -m reproduction.audit_block_tail --dataset B --output reproduction/results/block_tail/B-default
./reproduction/run_env.sh -m reproduction.audit_block_tail --dataset B --core 150 --extent 800 --lengthscale .005 --noise .0004 --output reproduction/results/block_tail/B-stress
./reproduction/run_env.sh -m reproduction.audit_block_tail --dataset 1000000 --output reproduction/results/block_tail/million-100-200
./reproduction/run_env.sh -m reproduction.audit_block_tail --dataset 1000000 --precision float64 --rtol 1e-9 --eps 1e-9 --probe-indices 0 8 15 --seeds 71 --output reproduction/results/block_tail/million-double
./reproduction/run_env.sh -m reproduction.audit_centered_hybrid --case B --cutoffs 200 --block-core 50 --output reproduction/results/block_tail/centered-B-50
./reproduction/run_env.sh -m reproduction.screen_corner_refinement --input reproduction/results/block_tail/B-stress --output reproduction/results/block_tail/corner-screen.json
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 8 --batch 1 --repeats 1 --centered-fluctuations --solve-cutoff 200 --block-core 50 --audit-analytic --stop-rows 2048 --holdout-rows 4096 --holdout-seed 2711 --scratch /opt/geospsr-runtime/common-rhs-128 --reference reproduction/results/centered_hybrid/million-200 --output reproduction/results/block_tail/centered-million-50
# Repeat the last command with --block-core 100 and output centered-million-100.
./reproduction/run_env.sh -m reproduction.summarize_block_tail
```
'''
    (OUT/'README.md').write_text(report)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
