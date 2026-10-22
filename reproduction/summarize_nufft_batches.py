"""Report fixed-RHS batching, rejected settings, and 128-draw confirmation."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / 'results/nufft_batches'


def read(name):
    data = json.loads((ROOT / name / 'results.json').read_text())
    if not data['complete']:
        raise RuntimeError(f'incomplete experiment: {name}')
    return data


def main():
    names = ['batch1','batch2','batch4','mean-start','mean-start-rtol1e-4']
    labels = ['Batch 1','Batch 2','Batch 4','Mean start','Mean start / loose tolerance']
    runs = [read(n) for n in names]
    prepare = read('prepare')
    pilots = [read('sdd128'),read('sdd128-rows2048')]
    reference, large = read('sdd128-validated'),read('nufft128')
    if any(r['N'] != 1000000 for r in [*runs,*pilots,reference,large]):
        raise RuntimeError('this report expects the million-point experiment')
    if any(r['args']['samples'] != 128 for r in [*pilots,reference,large]):
        raise RuntimeError('large confirmations must contain 128 draws')
    ref_fields = np.load(ROOT / 'sdd128-validated/fields.npz')
    ref_a = ref_fields['fields']
    ref_cov,ref_var = np.cov(ref_a[:,:32].T),ref_a.var(0,ddof=1)
    relative_error = lambda error,scale:float(np.linalg.norm(error)/np.linalg.norm(scale))
    pilot_quality = {}
    for name in ['sdd128','sdd128-rows2048']:
        data = np.load(ROOT / name / 'fields.npz')
        b = data['fields']
        q = {'field_error_over_fluctuations':relative_error(b-ref_a,ref_a-ref_a.mean(0)),
             'variance_relative_error':relative_error(b.var(0,ddof=1)-ref_var,ref_var),
             'covariance_relative_error':relative_error(np.cov(b[:,:32].T)-ref_cov,ref_cov),
             'ray_events_identical':bool(np.array_equal(data['rays'],ref_fields['rays'])),
             'collision_events_identical':bool(np.array_equal(data['collision'],ref_fields['collision']))}
        q['same_kernel_field_gates_pass'] = (q['field_error_over_fluctuations'] < .0005 and
                                            q['variance_relative_error'] < .0001 and
                                            q['covariance_relative_error'] < .0001)
        pilot_quality[name] = q
    (ROOT / 'sdd-pilot-comparison.json').write_text(json.dumps(
        {'reference':'sdd128-validated','comparisons':pilot_quality},indent=2)+'\n')
    geometry = np.load(ROOT / 'prepare/geometry.npz')
    rays,collision = ref_fields['rays'],ref_fields['collision']
    score = lambda t:np.trapezoid((t>=.05)&(t<=.95),geometry['distance'],axis=1)
    scores128 = score(rays.mean(0))
    prefixes = []
    for count in [8,16,32,64,128]:
        t,u = rays[:count].mean(0),collision[:count].mean(0)
        best = int(score(t).argmax())
        prefixes.append({'draws':count,
                         'max_ray_probability_delta_vs_128':float(np.max(np.abs(t-rays.mean(0)))),
                         'max_collision_probability_delta_vs_128':float(np.max(np.abs(u-collision.mean(0)))),
                         'best_view':best,'regret_under_128_draw_scores':float(scores128.max()-scores128[best])})
    (ROOT / 'sample-prefix-diagnostic.json').write_text(json.dumps({
        'qualification':'Nested prefixes of one ensemble; 128 draws are not ground truth. This is not an independent convergence study.',
        'prefixes':prefixes},indent=2)+'\n')
    prefix_table = [f"| {q['draws']} | {100*q['max_ray_probability_delta_vs_128']:.2f} | {100*q['max_collision_probability_delta_vs_128']:.2f} | {q['best_view']} |" for q in prefixes[:-1]]
    if not reference['sdd_converged'] or reference['analytic_holdout']['max_upper_endpoint'] >= .001:
        raise RuntimeError('confirmation reference failed its independent residual gate')
    if Path(large['args']['reference']).name != 'sdd128-validated':
        raise RuntimeError('NUFFT must compare against the validated SDD confirmation')
    if large['holdout_indices_sha256'] != reference['holdout_indices_sha256']:
        raise RuntimeError('confirmation holdout rows differ')
    nu_time, ref_time = large['median_solve_seconds'],reference['seconds']['sdd_updates']
    ratio = ref_time/nu_time
    timing_valid = (not reference.get('timing_excluded_reason')
                    and reference.get('gpu_process_audit',{}).get('sampled_isolation_pass',False)
                    and large.get('gpu_process_audit',{}).get('sampled_isolation_pass',False))
    status = (f'NUFFT is {ratio:.2f}× faster' if ratio >= 1 else f'NUFFT takes {1/ratio:.2f}× as long') if timing_valid else (
        'No new 128-draw speed ratio is established: GPU isolation failed or was not established')
    large_pass = (large['quality']['fields']['pass'] and large['quality']['events']['compatibility_pass']
                  and large['quality']['events']['uncertainty_coverage_pass']
                  and large['analytic_holdout']['max_upper_endpoint'] < .001)
    if not large_pass:
        status += '; quality checks fail, so this is not a matched-quality speedup'
    summary = {'eight_draw_runs':{n:r['median_solve_seconds'] for n,r in zip(names,runs)},
               '128_draw_sdd_row_batch':reference['args']['sdd_row_batch'],
               '128_draw_sdd_updates_seconds':ref_time,'128_draw_nufft_solve_seconds':nu_time,
               '128_draw_timing_valid':timing_valid,
               '128_draw_solve_ratio_sdd_over_nufft':ratio if timing_valid else None,
               '128_draw_quality_pass':large_pass,
               '128_draw_quality':large['quality']}
    (ROOT / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table = []
    for label,r in zip(labels,runs):
        q = r.get('quality')
        passed = 'reference' if q is None else ('pass' if q['fields']['pass'] and q['events']['compatibility_pass']
                                               and q['events']['uncertainty_coverage_pass'] else 'fail')
        cov = '—' if q is None else f"{100*q['fields']['empirical_covariance_relative_l2']:.6f}"
        times = [r['seconds'][f'solve_{i}'] for i in range(r['args']['repeats'])]
        memory = max(r['device_used_bytes_at_stage'].values())/2**30
        table.append(f'| {label} | {np.median(times):.2f} [{min(times):.2f}, {max(times):.2f}] | {memory:.2f} | {cov} | {passed} |')
    long_table = []
    for label,r in [('SDD 512-row pilot',pilots[0]),('SDD 2048-row pilot',pilots[1]),
                    ('SDD stricter confirmation',reference),('NUFFT confirmation',large)]:
        seconds = r['seconds'].get('sdd_updates',r.get('median_solve_seconds'))
        upper = r['analytic_holdout']['max_upper_endpoint']
        nrows = r['args'].get('holdout_rows',1024)
        passed = (r.get('sdd_converged',True) and upper < .001)
        long_table.append(f'| {label} | {seconds:.2f} | {nrows:,} | {100*upper:.5f} | {"pass" if passed else "fail"} |')
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].bar(range(5),[r['median_solve_seconds'] for r in runs],color=['#246b8e']*4+['#bd624c'])
    axes[0].set_xticks(range(5),['1','2','4','1 + mean','1 + mean\nrtol 1e-4'],fontsize=9)
    axes[0].set_xlabel('NUFFT draw batch / starting solution')
    axes[0].set_title('8 draws: three timing repetitions')
    axes[0].text(4,runs[-1]['median_solve_seconds']+2,'quality fail',ha='center',fontsize=8,color='#9d3e2c')
    f = large['quality']['fields']
    relative = [f['error_over_fluctuation_l2']/.002,f['empirical_variance_relative_l2']/.0002,
                f['empirical_covariance_relative_l2']/.0002,large['analytic_holdout']['max_upper_endpoint']/.001]
    axes[1].bar(range(4),relative,color=['#246b8e' if v < 1 else '#bd624c' for v in relative])
    axes[1].axhline(1,color='#9d3e2c',ls='--')
    axes[1].set_xticks(range(4),['Fields','Variance','Covariance','Residual'],fontsize=9)
    axes[1].set_title('128 draws: numerical confirmation')
    axes[0].set_ylabel('Conditional solve seconds')
    axes[1].set_ylabel('Error / acceptance threshold (must be <1)')
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(ROOT / 'conditioning.png',dpi=170)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,4.5),layout='constrained')
    for label,r in [('512-row pilot',pilots[0]),('2048-row pilot',pilots[1]),
                    ('512-row confirmation / larger panels',reference)]:
        history = r['sdd_history']
        line, = ax.semilogy([h['iterations']*r['sdd_parameters']['row_batch']/r['N'] for h in history],
                           [100*h['analytic_residual']['max_upper_endpoint'] for h in history],
                           label=label)
        ax.plot(history[-1]['iterations']*r['sdd_parameters']['row_batch']/r['N'],100*r['analytic_holdout']['max_upper_endpoint'],
                marker='x',ms=8,mew=2,color=line.get_color())
    ax.axhline(.1,color='#9d3e2c',ls='--',label='Holdout gate (0.1%)')
    ax.axhline(.05,color='#888',ls=':',label='Stopping threshold (0.05%)')
    ax.set_xlabel('Sampled training rows / observation (iterations × row batch / N)')
    ax.set_ylabel('Largest marginal bootstrap upper endpoint, %')
    ax.set_title('Stopping panels (lines) and independent holdouts (crosses)')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(ROOT / 'convergence.png',dpi=170)
    plt.close(fig)
    f = large['quality']['fields']
    a = reference['args']
    text = f'''# Conditioning at practical posterior sample counts

At one million observations and **128 shared draws**, numerical validation
**{'passes' if large_pass else 'fails'}**. **{status}**. The run extends
numerical validation to the larger workload; it does not extrapolate the earlier
eight-draw 4.39× ratio.
SDD shares analytic kernel rows across all 384 scalar RHS. Both methods start
from the same already-available finite posterior mean weights.

These are conditioning comparisons with a common **truncated** vector prior,
not full-analytic posterior sampling or complete application speedups. The
quality bridge and probability/geometry convergence remain open.

![Eight-draw tuning and 128-draw conditioning](conditioning.png)

## What changed

`FourierKernel.solve` accepts an optional starting solution. Its tolerance
remains relative to the **original RHS**, not the smaller initial residual.
`SpectralPosterior.draw(initial_mean=True)` uses the fitted mean weights;
the default remains unchanged. Compiled SDD also accepts an initial solution,
initializing both its iterate and Polyak average while keeping zero velocity.
Tests compare that update to solving the explicitly shifted residual equation.

The staged prior generator preserves the existing one-draw sampler's random
sequence. All variants replay the same RHS, scalar priors and offset; changing
batch size cannot silently change the samples. Source and geometry hashes,
seed 42 and all individual timings are recorded. A/B dense sampler improvements
from the earlier report are separate.

The first staging script omitted saving mean weights. They were regenerated
from the identical dataset/model; that scratch repair is recorded in
`prepare/mean-alpha-regeneration.json` and is not counted as another production
mean-fit stage. Every initialized solver records the same mean-weight hash.
The current preparation script saves those weights directly.

All **35 tests pass**, including independent dense solver checks, preservation
of staged random draws and hybrid covariance/mean identities. A CLI smoke check
also rejects the failed-holdout reference before transferring its RHS.

## Eight-draw sweep

All runs use the first eight draws of a prepared 128-draw ensemble, vector
cutoff 200, scalar/cross cutoff 50, length scale .01, per-axis variance .1 and
observation-noise variance .02. GPU: NVIDIA L40. Plan creation, source aggregation
and validation are outside solve times, with separate records. Ranges span
three repetitions; they are not confidence intervals.

| Variant | Solve median [min, max], s | Largest stage device usage, GiB | Paired covariance error % | Quality |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(table)}

Memory values are synchronized observations of total device usage, **not
guaranteed peaks or process-only allocations**.
Batches of two and four do not improve this workload. Mean initialization at
rtol 1e-5 provides a modest improvement; it preserves the checked ray/collision
events and view decision. The looser rtol 1e-4 trial is retained as a failure:
it passes sampled analytic residual and event checks but fails variance and
covariance checks. It is not selected for the larger run.

Same-kernel gates: field error/fluctuation norm <0.05%, empirical
variance/covariance errors <0.01%, and the existing local ray/collision/view
gates. These assess paired numerical differences, not absolute MC accuracy.

## 128-draw confirmation and the failed stopping pilot

The complete ensemble was generated before either solve. Eight draws tuned
NUFFT batching; the remaining 120 extend that numerical check. SDD uses eta 5,
momentum .9, Polyak .01 and seed 0. Both pilots used 512 stopping rows, a
maximum per-draw bootstrap upper endpoint <0.05%, and 1,024 independently
selected holdout rows. **The 512-row-batch pilot passed stopping but failed its
holdout: 0.118% against the 0.1% gate.** Its 813.92 s result is retained and
cannot serve as a matching-quality reference. Comparing the 2048-row pilot
against it is only diagnostic.

The final SDD confirmation uses **{a['sdd_row_batch']:,} rows per update**,
**{a['stop_rows']:,} stopping rows** and the unchanged stopping threshold of
**{100*a['stop_tolerance']:.3f}%**. Both final solvers are checked on the same
**{a['holdout_rows']:,} fresh holdout rows**, seed {a['holdout_seed']}, excluding
stopping rows. This is an empirical check, not a full-system or simultaneous
probability bound. The configurations tested here do not establish globally
optimal tuning. The stricter validation was chosen after the initial failure;
that adaptive history is part of the report.

| Solver | Conditional solves / updates, s | Holdout rows | Largest holdout upper endpoint % | Residual gate |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(long_table)}

Against the larger-panel SDD reference, the earlier 512-row pilot also fails
the same-kernel covariance gate: **{100*pilot_quality['sdd128']['covariance_relative_error']:.6f}%**
versus the **0.01%** limit. The 2048-row pilot passes, at
**{100*pilot_quality['sdd128-rows2048']['covariance_relative_error']:.6f}%**.
Both pilots preserve every checked ray/collision event. Event agreement alone
would therefore have accepted the failed reference.

![Stopping-panel convergence and independent holdouts](convergence.png)

**Timing caveat:** other GPU compute processes overlapped both final
confirmations. SDD's recorded **{ref_time:.2f} s** and NUFFT's **{nu_time:.2f} s**
are retained for provenance and cannot support a speed ratio. NUFFT's
`gpu_process_audit` records **{len(large['gpu_process_audit']['foreign_process_samples'])}** samples
with a foreign process. Earlier pilots were not continuously monitored.
The new monitor samples GPU process ownership once per second, records foreign
PIDs/errors and rejects contaminated timing comparisons. It can miss short-lived
work and is not proof of exclusive access. Numerical comparisons remain useful.

These large runs are **single confirmations**, not repeated timing distributions.
NUFFT uses one draw per batch, mean initialization and rtol 1e-5. Cross-method
field gates remain error/fluctuation norm <0.2%, empirical variance/covariance
errors <0.02%, plus the existing local event/view checks. Measured field error:
**{100*f['error_over_fluctuation_l2']:.6f}%**; variance error:
**{100*f['empirical_variance_relative_l2']:.6f}%**; covariance error:
**{100*f['empirical_covariance_relative_l2']:.6f}%**. Field gates:
**{f['pass']}**; local event gates:
**{large['quality']['events']['compatibility_pass']}**.

Common mean fitting costs **{prepare['seconds']['mean_fit']:.2f} s**; GPU prior/RHS
generation for 128 draws costs **{sum(prepare['prior_generation_seconds']):.2f} s**,
excluding transfers and scratch-file writes. Query reductions, scalar aggregation,
plans/JIT and validation have separate timings in raw JSON. The headline ratio
would exclude those stages and would not be fit-through-application time.
`benchmark_work_wall_seconds`, where present, includes benchmark I/O and adaptive
validation; it is also not a production application timing.

## Statistical scope

SDD solves the full analytic training covariance, but the **shared prior is
finite**. That reference does not itself generate full-model posterior samples.
The [covariance bracket](../spectral_tail/METHOD.md) now covers this
shared-prior/full-inverse case too. Limited-panel evidence does not prove full
ray/collision or hitbox accuracy.
The derivation also adds separate mean-error and unequal-prior/solve-band bounds,
checked with noncommuting dense matrices. Their usefulness for large application
panels remains to be measured; no speedup is attributed to those identities.

128 draws still have worst-case marginal binomial standard error of about
**4.4 percentage points**. Passing paired numerical gates does not establish
absolute probability convergence. Queries use direct cutoff-50 Fourier fields;
this is not the released 50³ cross-kernel interpolant. Checks cover 257 probes,
a 32-query covariance panel, 24 rays × 256 steps, 17 uncertain spheres × 256
surface points and associated view scores.

A prefix diagnostic illustrates the sampling limitation. On the same geometry,
the first eight draws differ from the 128-draw ensemble by up to
**{100*prefixes[0]['max_ray_probability_delta_vs_128']:.2f} percentage points** for rays and
**{100*prefixes[0]['max_collision_probability_delta_vs_128']:.2f} points** for collisions.
The selected views are listed below. **128 draws are not ground truth**, and these nested prefixes
of one ensemble are not an independent convergence study. Differences need not
decrease monotonically as more samples are added.

| Prefix draws | Maximum ray difference vs 128, pp | Maximum collision difference vs 128, pp | Selected view |
| ---: | ---: | ---: | ---: |
{chr(10).join(prefix_table)}

## Reproduce

The prior arrays and RHS use several GB, so keep scratch storage outside the
repository. The generated million-point visible-surface cloud is required.
Run all GPU jobs serially for comparable timings.

```bash
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches prepare --samples 128 --scratch /opt/geospsr-runtime/common-rhs-128 --output reproduction/results/nufft_batches/prepare
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 8 --batch 1 --repeats 3 --scratch /opt/geospsr-runtime/common-rhs-128 --output reproduction/results/nufft_batches/batch1
# Repeat with --batch 2 and 4; --reference .../batch1 and distinct outputs.
# Mean-start trials add --initial-mean, optionally --rtol 1e-4 --audit-analytic.
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches sdd --samples 128 --repeats 1 --initial-mean --sdd-row-batch {a['sdd_row_batch']} --stop-rows {a['stop_rows']} --stop-tolerance {a['stop_tolerance']} --holdout-rows {a['holdout_rows']} --holdout-seed {a['holdout_seed']} --retain-weights --scratch /opt/geospsr-runtime/common-rhs-128 --output reproduction/results/nufft_batches/sdd128-validated
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 128 --batch 1 --repeats 1 --initial-mean --audit-analytic --stop-rows {a['stop_rows']} --holdout-rows {a['holdout_rows']} --holdout-seed {a['holdout_seed']} --retain-weights --scratch /opt/geospsr-runtime/common-rhs-128 --reference reproduction/results/nufft_batches/sdd128-validated --output reproduction/results/nufft_batches/nufft128
./reproduction/run_env.sh -m reproduction.summarize_nufft_batches
```

Raw pilot commands are recorded in their JSON files. The final harness rejects
a reference with failed analytic holdout before running a candidate. Earlier
pilots predate that guard and the disjoint holdout selector. Compact field/event
arrays and raw JSON/logs are published; multi-GB RHS, priors and weights remain
regenerable scratch data.
'''
    (ROOT / 'README.md').write_text(text)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
