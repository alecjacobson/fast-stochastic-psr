"""Generate the report from saved runs; no GPU work or timing extrapolation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / 'results/centered_hybrid'


def read(name):
    data = json.loads((ROOT / name / 'results.json').read_text())
    if not data['complete']:
        raise ValueError(f'incomplete run: {name}')
    return data


def passed(data):
    q = data['quality']
    return (q['fields']['pass'] and q['events']['compatibility_pass']
            and q['events']['uncertainty_coverage_pass']
            and data['analytic_holdout']['max_upper_endpoint'] < .001
            and data.get('sdd_converged', True))


def isolated(data):
    return data.get('gpu_process_audit', {}).get('sampled_isolation_pass', False)


def main():
    controls = [read(f'million-{c}') for c in [200, 150, 100]]
    large, sdd = read('million-150-128'), read('sdd128')
    for data in controls[1:] + [large, sdd]:
        if (data['source_manifest']['hashes'] != controls[0]['source_manifest']['hashes']
                or data['geometry_sha256'] != controls[0]['geometry_sha256']):
            raise ValueError('source/RHS/query provenance differs')
    if sdd['holdout_indices_sha256'] != large['holdout_indices_sha256']:
        raise ValueError('128-draw analytic holdout differs')
    if sdd['initial_mean_sha256'] != large['held_mean_sha256']:
        raise ValueError('128-draw mean weights differ')
    if Path(sdd['args']['reference']).name != 'million-150-128':
        raise ValueError('expected direct paired SDD comparison')
    eight_valid = all(passed(d) and isolated(d) for d in controls[:2])
    large_valid = all(passed(d) and isolated(d) for d in [large, sdd])
    eight_ratio = controls[0]['median_solve_seconds'] / controls[1]['median_solve_seconds']
    large_ratio = sdd['seconds']['sdd_updates'] / large['median_solve_seconds']
    summary = {
        'eight_draw_150_vs_200_ratio': eight_ratio if eight_valid else None,
        '128_draw_sdd_over_hybrid_solve_ratio': large_ratio if large_valid else None,
        '128_draw_timing_and_quality_pass': large_valid,
        'scope': 'conditional solve only; common finite prior; not full-model or complete-application speedup',
        'eight_draw_quality': {str(d['args']['solve_cutoff']): passed(d) for d in controls},
        '128_draw_sdd_comparison_quality': sdd['quality'],
    }
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    table = []
    for data in controls:
        times = [data['seconds'][f'solve_{i}'] for i in range(data['args']['repeats'])]
        q = data['quality']['fields']
        table.append(f"| {data['args']['solve_cutoff']} | {np.median(times):.2f} "
                     f"[{min(times):.2f}, {max(times):.2f}] | "
                     f"{100*q['empirical_covariance_relative_l2']:.5f} | "
                     f"{100*data['analytic_holdout']['max_upper_endpoint']:.5f} | "
                     f"{'pass' if passed(data) else 'FAIL'} |")
    dense_table = []
    for name in ['dense-A', 'dense-B', 'dense-B-stress']:
        data = read(name)
        for row in data['rows']:
            e = row['errors_vs_analytic']
            dense_table.append(f"| {name} | {row['solve_cutoff']} | {100*e['mean_relative_l2']:.5f} | "
                               f"{100*e['covariance_relative_l2']:.6f} | {100*e['probability_rms']:.6f} | "
                               f"{'pass' if row['panel_gates_pass'] else 'FAIL'} |")
    # Both timing stages and validation overhead are reported. The monitored
    # script wall clock also includes artifact hashing/writing and is not an
    # end-to-end reconstruction latency.
    timing_table = []
    for label, data, key in [('Hybrid 150', large, 'solve_0'), ('Analytic SDD', sdd, 'sdd_updates')]:
        stages = data['seconds']
        if not isolated(data):
            label += ' (timing excluded)'
        timing_table.append(f"| {label} | {stages[key]:.2f} | "
                            f"{stages.get('sdd_stopping_checks', 0):.2f} | "
                            f"{stages['analytic_holdout']:.2f} | "
                            f"{stages['scalar_aggregation']:.2f} | "
                            f"{stages['probe_fields']+stages['rays']+stages['collision']:.2f} | "
                            f"{data['benchmark_work_wall_seconds']:.2f} |")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    axes[0].bar([0, 1, 2], [d['median_solve_seconds'] for d in controls],
                color=['#287496', '#287496', '#b45440'])
    axes[0].set_xticks([0, 1, 2], ['200', '150', '100\nquality fail'])
    axes[0].set_xlabel('Fluctuation inverse cutoff (prior/mean fixed at 200)')
    axes[0].set_ylabel('Conditional solve seconds')
    axes[0].set_title('One million observations, eight draws')
    f = sdd['quality']['fields']
    values = [f['error_over_fluctuation_l2']/.002,
              f['empirical_variance_relative_l2']/.0002,
              f['empirical_covariance_relative_l2']/.0002,
              large['analytic_holdout']['max_upper_endpoint']/.001,
              sdd['analytic_holdout']['max_upper_endpoint']/.001]
    axes[1].bar(range(len(values)), values,
                color=['#287496' if v < 1 else '#b45440' for v in values])
    axes[1].axhline(1, color='#b45440', ls='--')
    axes[1].set_xticks(range(5), ['Fields', 'Variance', 'Covariance', 'Hybrid\nresidual', 'SDD\nresidual'], fontsize=8)
    axes[1].set_ylabel('Error / acceptance threshold')
    axes[1].set_title('128 draws: direct SDD / hybrid comparison')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(ROOT / 'conditioning.png', dpi=170)
    plt.close(fig)
    eight_statement = f'**{eight_ratio:.2f}× faster**' if eight_valid else '**no accepted speed ratio**'
    large_statement = (f'It is **{large_ratio:.2f}× faster than the compiled analytic SDD update stage**.'
                       if large_valid else '**No accepted 128-draw speed ratio against SDD is established.**')
    f = sdd['quality']['fields']
    e = sdd['quality']['events']
    report = f'''# Wider prior with a cheaper centered fluctuation solve

At one million observations, a cutoff-150 inverse applied only to centered
fluctuations is {eight_statement} than the cutoff-200 centered control over
eight common draws (three repetitions each). The 128-draw confirmation takes
**{large['median_solve_seconds']:.2f} s**. {large_statement}
These are conditional-solve comparisons, **not full application speedups**.
The common vector prior and held mean remain at cutoff 200; scalar/cross cutoff
is 50. This experimental hybrid is not an exact posterior at either bandwidth.
The production sampler's default remains unchanged.

![Solve bandwidth and direct 128-draw quality checks](conditioning.png)

## Numerical controls and retained failures

[Declared protocol](PROTOCOL.md). The hybrid solves the centered prior-plus-noise
RHS from zero, then adds the held mean weights. Its CG tolerance is relative to
that centered RHS. A cutoff-200 centered control checks the effect of this
normalization against the existing original-RHS mean-start solver. It passes
the stricter same-model field gates; narrower inverses use the declared
cross-model gates. Independent analytic holdouts still test **total weights
against original RHS**, with the unchanged 0.1% gate.

All three eight-draw configurations have sampled GPU isolation. Times are
medians and ranges, not confidence intervals. Covariance errors compare to the
cutoff-200 centered control except the 200 control itself, which compares to
the earlier mean-start solver.

| Inverse cutoff | Solve median [min, max], s | Paired covariance error % | Analytic holdout upper % | All gates |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(table)}

Cutoff 100 is rejected: it fails covariance and analytic-residual gates despite
matching the tested events. The short-scale/low-noise dense case below also
fails. Neither speed is promoted as an accepted gain.

## 128 shared draws against a strong SDD reference

SDD shares each analytic kernel row across all 384 scalar RHS, starts from the
same held mean, uses row batch 2,048, and stops on 2,048 fixed analytic rows at
a maximum marginal bootstrap upper endpoint below 0.05%. A fresh, disjoint
4,096-row panel (seed 2711) audits both solvers. SDD validation rows are now
regenerated in blocks of 64 instead of retaining an 8 GB stopping matrix.
This memory option preserves the update; a B/4-draw smoke check with a partial
257-row validation block stops at the same 1,024 iterations and preserves events.

The direct paired comparison {'passes' if passed(sdd) else 'FAILS'}: field error
divided by fluctuation norm **{100*f['error_over_fluctuation_l2']:.6f}%**,
variance error **{100*f['empirical_variance_relative_l2']:.6f}%**, covariance
error **{100*f['empirical_covariance_relative_l2']:.6f}%**. Maximum ray/collision
probability differences are **{100*e['rays']['max_probability_difference']:.5f} /
{100*e['collision']['max_probability_difference']:.5f} percentage points**;
selected views are {e['next_view']['reference_best']} and {e['next_view']['candidate_best']}.
The JSON records the hybrid as the comparison reference and SDD as candidate;
denominators follow that direction. The hybrid also passes separately against
the previously saved wider NUFFT128 output, whose contaminated timing is not
used in any ratio here.

| 128-draw method | Solve/updates s | Stopping checks s | Independent holdout s | Scalar aggregation s | Queries/events s | Script work wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(timing_table)}

Plans/setup and RHS transfer are separately recorded in each JSON. Script wall
time additionally includes retained-weight transfers, hashing, result writes
and validation; it is not a complete application latency. Common input/prior
preparation and mean fitting are outside these runs. The earlier preparation
record has 7.39 s of prior GPU generation and 7.68 s of mean fitting, excluding
staging/IO. The 128-draw comparison has one timing per method. GPU-process audit
status is hybrid **{isolated(large)}**, SDD **{isolated(sdd)}**; periodic sampling
can miss short-lived work. No other project's process was stopped.
The SDD audit observed foreign GPU processes in
**{len(sdd['gpu_process_audit']['foreign_process_samples'])} samples**.
If this count is nonzero or monitoring failed, its raw timing is excluded from
speed ratios even when numerical gates pass; checking only the process list
before and after a run would miss such overlap.

SDD uses the full analytic inverse but the same **truncated vector prior**;
it is not a full-analytic posterior sampler. Shared finite draws isolate the
numerical comparison; 128 draws are not probability ground truth. Full mean,
joint-event, sample-count and geometry convergence remain open. SDD is feasible
at this size, so this does not establish upstream intractability.

## Exact dense moments on overlap sizes

Float64 Cholesky calculations compare the law directly on the fixed 16-query
panel from the spectral-tail audit, without Monte Carlo sampling error. The
default A/B cases pass; B with length scale .005 and noise variance .0004 fails.
The latter retains the erroneous wider mean: a narrower fluctuation inverse
cannot repair it. All means use their own data-derived deterministic offset.
The held hybrid mean is identical across inverse cutoffs.

| Input | Inverse cutoff | Mean error % | Covariance error % | Inside-probability RMS pp | Panel gates |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(dense_table)}

For training matrices B (narrow) and W (wide), cross matrix C, A=B⁻¹C and
Z=W⁻¹C, the covariance excess over the wider posterior is
`R = (A-Z)ᵀ W (A-Z)`, summed over vector components. This stable PSD expression
also gives the covariance of the paired draw difference. That difference is
independent of the wider posterior fluctuation under the shared joint Gaussian
prior. True-law covariance error is second order, while paired field error and
finite-ensemble covariance cross terms can be larger. This explains why the
dense panel and paired finite-draw errors differ; it does not waive either gate.
[Derivation](../spectral_tail/METHOD.md) and independent dense algebra/sampler
tests accompany the implementation.
[Full test output](tests.log) records validation of this milestone.

## Reproduce

Use the pinned environment from the main reproduction guide. Stage the common
RHS as in [the earlier report](../nufft_batches/README.md); large arrays live
outside the repository. Reuse its saved mean and prior. The saved source,
geometry, mean and holdout hashes make the pairing explicit.

```bash
./reproduction/run_env.sh -m reproduction.audit_centered_hybrid --case A --output reproduction/results/centered_hybrid/dense-A
./reproduction/run_env.sh -m reproduction.audit_centered_hybrid --case B --output reproduction/results/centered_hybrid/dense-B
./reproduction/run_env.sh -m reproduction.audit_centered_hybrid --case B --lengthscale .005 --noise .0004 --output reproduction/results/centered_hybrid/dense-B-stress
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 128 --batch 1 --repeats 1 --centered-fluctuations --solve-cutoff 150 --audit-analytic --stop-rows 2048 --holdout-rows 4096 --holdout-seed 2711 --retain-weights --scratch /opt/geospsr-runtime/common-rhs-128 --reference reproduction/results/nufft_batches/nufft128 --output reproduction/results/centered_hybrid/million-150-128
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches sdd --samples 128 --repeats 1 --initial-mean --sdd-row-batch 2048 --stop-rows 2048 --stop-tolerance .0005 --stream-stop-rows --holdout-rows 4096 --holdout-seed 2711 --retain-weights --reference reproduction/results/centered_hybrid/million-150-128 --scratch /opt/geospsr-runtime/common-rhs-128 --output reproduction/results/centered_hybrid/sdd128
./reproduction/run_env.sh -m reproduction.summarize_centered_hybrid
```

For eight draws use `--samples 8 --repeats 3`, inverse cutoffs 200/150/100 and
the reference paths recorded in their JSON. Scratch paths are configurable.
The separate [ContactFusion assessment](../../CONTACTFUSION.md) defines a
practical replay target; none of these scorpion runs measures ContactFusion.
'''
    (ROOT / 'README.md').write_text(report)


if __name__ == '__main__':
    main()
