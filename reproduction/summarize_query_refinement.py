"""Publish all declared nested-query replays, including failed accuracy gates."""
import json
from pathlib import Path
import numpy as np
from reproduction.coefficient_archive import digest
from reproduction.nested_queries import discrepancy_bound

OUT=Path(__file__).resolve().parent/'results/query_refinement'


def main():
    source=json.loads((OUT/'generate256/results.json').read_text())
    manifest=json.loads((OUT/'archive_manifest.json').read_text())
    runs=[json.loads((OUT/f'replay{i}/results.json').read_text()) for i in range(1,4)]
    arrays=[]
    for i in range(1,4):
        with np.load(OUT/f'replay{i}/counts.npz') as data:
            arrays.append({k:data[k] for k in data.files})
    assert source['complete'] and source['finite_residual_gate_pass'] and source['cache_gate_pass']
    assert manifest['complete'] and manifest['count']==manifest['written']==source['sample_count']==256
    assert manifest['metadata']['source_hashes']==source['hashes']
    assert manifest['metadata']['model']==source['model']
    assert manifest['metadata']['offset']==source['offset']
    assert manifest['metadata']['N']==1_000_000
    for run,data in zip(runs,arrays):
        assert run['complete'] and run['field_gate_pass'] and run['backend_event_gate_pass']
        assert all(run['source_event_replay_exact'].values())
        assert run['archive_manifest_sha256']==digest(OUT/'archive_manifest.json')
        assert run['source_metadata']==manifest['metadata']
        assert run['samples']==256 and run['args']['levels']==4
        assert data.keys()==arrays[0].keys()
        for k in data: np.testing.assert_array_equal(data[k],arrays[0][k])
    first=runs[0];data=arrays[0];count=first['samples'];rows=[]
    for index,entry in enumerate(first['refinement']):
        k=int(data['per_draw_refinement_disagreement'][:,index].sum())
        bound=discrepancy_bound(k,count)
        assert k==entry['disagreeing_draws'] and bound['upper']==entry['upper']
        assert all(run['refinement'][index]==entry for run in runs)
        rows.append(f'| {entry["kind"]} | {entry["coarse_size"]} → {entry["fine_size"]} | '
                    f'{100*entry["maximum_empirical_probability_change"]:.4f} pp | {k}/{count} | '
                    f'{100*entry["upper"]:.3f} pp | {"pass" if entry["pass_2pp"] else "fail"} |')
    with np.load(OUT/'generate256/checkpoint_256.npz') as original:
        for kind,name in [('rays','rays'),('sphere','collision')]:
            for level,prefix in [(0,''),(1,'fine_')]:
                np.testing.assert_array_equal(data[f'cached_{kind}_{level}_counts'],original[prefix+name+'_successes'])
        np.testing.assert_array_equal(data['distance_0'],original['distance'])
    eligible=all(r['gpu_process_audit']['sampled_isolation_pass'] for r in runs)
    medians={k:float(np.median([r['seconds'][k] for r in runs])) for k in first['seconds']}
    summary={'samples':count,'identical_replay_arrays':True,'refinement':first['refinement'],
             'next_view':first['next_view'],'maximum_cached_error_over_fluctuations':max(r['maximum_cached_error_over_fluctuations'] for r in runs),
             'maximum_backend_probability_difference':max(r['maximum_backend_probability_difference'] for r in runs),
             'absolute_probability_radius':source['checkpoints']['256']['simultaneous_probability_radius'],
             'timing_ratio_eligible':eligible,'median_diagnostic_seconds':medians,
             'complete_application_quality_pass':False}
    if eligible:
        summary['prepared_direct_over_cached_query_ratio']=medians['direct_query_with_packing']/medians['cached_query']
        summary['direct_over_cached_setup_plus_query_ratio']=float(np.median([
            r['seconds']['direct_query_with_packing']/(r['seconds']['cached_grid_setup']+r['seconds']['cached_query']) for r in runs]))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    timing_rows=[]
    for i,r in enumerate(runs,1):
        s=r['seconds']
        timing_rows.append(f'| {i} | {s["cached_grid_setup"]:.3f} | {s["cached_query"]:.3f} | '
                           f'{s["direct_query_with_packing"]:.3f} | {s["archive_block_io_hash"]:.3f} | '
                           f'{s["work_wall"]:.3f} | {r["gpu_process_audit"]["sampled_isolation_pass"]} |')
    view_rows=[f'| {r["steps"]} | {r["selected"]} | {max(r["scores"]):.8f} | {r["empirical_regret_vs_finest"]:.8f} |'
               for r in first['next_view']]
    timing_status=('All three audits pass; prepared-query ratios are in summary.json.' if eligible else
                   'Replay 2 encountered another GPU process. The declared all-three isolation rule fails, so **no timing ratio is reported**. No run was discarded or repeated to improve timing.')
    report=f'''# Nested application geometry on a million-point posterior

**Collision discretization is a material unresolved error.** On the same 256
coherent draws, changing sphere-surface probes from 512 to 1,024 points changes
one object's collision estimate by **21.875 percentage points**. The earlier
256 → 512 agreement did not establish convergence. Rays show no further sampled
event changes after 511 positions in this test, but that is not a continuous-ray
or full-model certificate. No new matching-quality application speedup is claimed.

![Nested collision and ray refinement](refinement.png)

Left: all 17 sphere-surface collision proxies under increasing point counts.
Right: the ray and original distance position with the largest 256 → 511 change,
followed across the same fixed coordinates on every grid. All estimates use the
same 256 draws; the finest resolution is not ground truth. These are diagnostic
plots, not the paper's unreleased application scenes.

## Paired finite-geometry evidence

The [declared protocol](PROTOCOL.md) fixes four nested refinements for each task
and a single final statistical look. Each draw contributes one Bernoulli trial:
did **any** event change in that comparison? Its probability upper-bounds every
individual event-probability difference at the common coordinates. We use the
one-sided exact Clopper–Pearson bound at alpha = .05/8, giving simultaneous 95%
coverage over these eight comparisons via Bonferroni.
[SciPy's exact interval documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
Spatial query events need not be independent; the posterior draws must be.
The bounds apply to this fixed numerical finite-band sampler and evaluator.

| Task | Refinement | Largest empirical change | Draws with any change | Simultaneous upper bound | ≤2 pp |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

The any-event bound is conservative when different objects change in different
draws. Even the 2,048 → 4,096 sphere comparison fails the declared 2 pp bound:
one changed draw gives a 2.770 pp upper bound. Zero changes among 256 draws give
1.963 pp. Neither outcome bounds errors shared by both resolutions. Sphere
points test the surface of the probe, not solid-volume collision or continuous
intersection. Adaptive surface/ray refinement and held-out validation remain
necessary; see [#29](https://github.com/alecjacobson/fast-stochastic-psr/issues/29).

Next-view scores integrate the floating-point indicator of transmittance lying
between .05 and .95. The zero-based selected ray is 14 at all five resolutions:

| Ray positions | Selected ray | Selected score | Empirical regret at finest grid |
| ---: | ---: | ---: | ---: |
{chr(10).join(view_rows)}

This is empirical agreement at a fixed sample count, not confidence that the
action is optimal. The separate generation protocol's simultaneous absolute
Monte Carlo radius is **{100*summary['absolute_probability_radius']:.2f} pp** at
256 draws, above the 2 pp target. That bound covers the producer's original and
first refined query sets and four declared checkpoints; it is not an absolute
accuracy bound for all later replay coordinates. Full analytic training-tail
and scalar/cross-band convergence also remain open.

## Numerical agreement and replay integrity

The mean, prior and conditioning use the same cube200 vector model, scalar/cross
cutoff50, length scale .01, variance .1 and noise variance .02. This run does not
use the narrower centered150 inverse. Packed solves and folded coherent priors
produce 256 fresh draws, seed51, in two-draw blocks. Independent finite-covariance
checks on 32 rows have maximum relative residual
**{100*source['maximum_independent_finite_residual']:.6f}%**, below 0.002%.
These sampled rows do not certify the full analytic system.

All three replays load the identical sealed, hashed coefficient archive. They
reproduce the producer's coarse/fine ray and sphere counts exactly, and all
saved arrays agree across the three replays. Cached field error is at most
**{100*summary['maximum_cached_error_over_fluctuations']:.6f}%** of the fluctuation
norm, below 0.05%; maximum cached/direct event-probability difference is
**{100*summary['maximum_backend_probability_difference']:.6f} pp**, below 2 pp.
The statistical refinement table uses cached events; numerical backend error
is a separate check and is not added into a continuous-geometry guarantee.

`PackedDirectEvaluator` is a stronger ordinary type-2 reference: it reuses plans,
projects each coefficient field onto its real component, independently scales
the fields, and packs pairs into one complex transform. Each call still performs
the ordinary NUFFT FFT. Tests compare against explicit Fourier sums, including
non-Hermitian inputs, zero channels, partial batches and 24 orders of magnitude
between channels. [All 93 tests pass](tests.log).

## Timing scope

Each replay evaluates the finest ray and sphere sets together, once per block;
all coarser levels are exact subsets/prefixes. Both backends therefore query
167,576 points per draw. Repeating separate evaluations for every coarser grid
would artificially disadvantage the ordinary type-2 baseline. Cached queries
include their point handling; the direct path reuses its fixed-point plans.
Direct-query timing includes coefficient projection, scaling and packing.

| Replay | Cached grid setup (s) | Cached queries (s) | Packed direct queries (s) | Archive read/hash (s) | Full replay work (s) | Sampled isolation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(timing_rows)}

{timing_status}
The recorded direct times are lower than cached query times in this fixed-query,
two-field workload, even before charging cached grid setup. This does not
invalidate earlier measurements with different workloads, but it means their
large ratios cannot be transferred to this stronger reference without rerunning.
Prepared-query, setup-plus-query and full application costs must remain separate.

Full replay work additionally includes plan setup, mean evaluation, file checks,
transfers, numerical controls, reductions and output writing. The generation
took {source['seconds']['instrumented_work_wall']:.2f} s including fitting,
conditioning, validation and archive export. Its audit also saw contention;
this is an execution diagnostic, not an application performance comparison.
Sample preparation is paid once per fitted map, before these replay timers.

## Archive and reproduction

The [manifest](archive_manifest.json) records complete ordered sample coverage,
model/seed/batch/offset, source and geometry hashes, and SHA-256 for every block,
mean and geometry file. The reader rejects incomplete archives, missing coverage
or failed hashes. A sealed archive supports bounded-memory query replay; it is
not a sampler checkpoint/resume mechanism. The approximately 2 GiB coefficient
archive stays outside git; compact raw counts, checkpoints and timings are
published here. Regeneration may change floating-point coefficients across GPU
environments; each replay must validate its own producer and source counts.

Starting from the existing common-RHS million-point dataset (see
[its preparation report](../nufft_batches/README.md)):

```bash
./reproduction/run_env.sh -m reproduction.stream_applications \\
  --samples 256 --checkpoints 8 32 128 256 \\
  --coefficient-dir /opt/geospsr-runtime/query-refinement/coefficients256 \\
  --output /opt/geospsr-runtime/query-refinement/generate256
./reproduction/run_env.sh -m reproduction.replay_query_refinement \\
  --archive /opt/geospsr-runtime/query-refinement/coefficients256 \\
  --source-results /opt/geospsr-runtime/query-refinement/generate256 \\
  --output /opt/geospsr-runtime/query-refinement/replay1
# Repeat in fresh processes for replay2 and replay3; keep every result.
./reproduction/run_env.sh -m reproduction.summarize_query_refinement
```

The summary command reads the compact published artifacts in this directory.
When replacing them with a new experiment, copy all three replay directories,
generation results/checkpoints and the matching manifest first. Keep the
coefficient directory empty before generation; never mix old and new draws.
'''
    (OUT/'README.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    sphere=np.stack([data[f'cached_sphere_{i}_counts']/count for i in range(5)])
    for j in range(sphere.shape[1]):
        axes[0].plot(first['sphere_points'],sphere[:,j],'-o',ms=3,alpha=.8)
    axes[0].set(xscale='log',xlabel='Points per sphere (nested)',ylabel='Collision probability',ylim=(-.03,1.03),title='Sphere-surface probes')
    axes[0].set_xticks(first['sphere_points'],first['sphere_points'])
    delta=data['rays_0_refinement_counts'];ray,pos=np.unravel_index(np.argmax(delta),delta.shape)
    y=[data[f'cached_rays_{i}_counts'][ray,pos*2**i]/count for i in range(5)]
    axes[1].plot(first['ray_steps'],y,'-o',color='#355e93')
    axes[1].set(xscale='log',xlabel='Positions per ray (nested)',ylabel='Transmittance probability',title=f'Ray {ray}, fixed distance {data["distance_0"][pos]:.4f}')
    axes[1].set_xticks(first['ray_steps'],first['ray_steps'])
    for ax in axes:
        ax.minorticks_off();ax.grid(alpha=.2)
    fig.savefig(OUT/'refinement.png',dpi=180);plt.close(fig)
    print(json.dumps({'identical_replays':True,'timing_ratio_eligible':eligible,'absolute_probability_radius':summary['absolute_probability_radius']}))


if __name__=='__main__': main()
