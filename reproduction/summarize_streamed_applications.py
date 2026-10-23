"""Publish bounded-storage execution, numerical gates and unresolved accuracy."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduction/results/streamed_applications'


def main():
    control=json.loads((OUT/'million-control8-corrected/results.json').read_text())
    data=json.loads((OUT/'million128/results.json').read_text())
    for d in [control,data]:
        assert d['complete'] and d['finite_residual_gate_pass'] and d['cache_gate_pass']
        assert not d['retained_ensemble_coefficients']
    cq=control['retained_control_quality']
    assert cq['fields']['pass'] and cq['events']['compatibility_pass']
    assert cq['events']['quality_gate_version']==3
    final=np.load(OUT/'million128/checkpoint_128.npz')
    reference=data['checkpoints']['128']['next_view']
    rows=[];prefix=[]
    for count in [8,32,128]:
        state=data['checkpoints'][str(count)]
        arrays=np.load(OUT/f'million128/checkpoint_{count}.npz')
        ray_difference=float(np.abs(arrays['rays_probability']-final['rays_probability']).max())
        collision_difference=float(np.abs(arrays['collision_probability']-final['collision_probability']).max())
        selected=state['next_view']['selected']
        regret=float(max(reference['scores'])-reference['scores'][selected])
        prefix.append({'count':count,'max_ray_probability_difference_vs_128':ray_difference,
                       'max_collision_probability_difference_vs_128':collision_difference,
                       'selected':selected,'empirical_regret_vs_128':regret})
        rows.append(f'| {count} | {100*state["simultaneous_probability_radius"]:.2f} pp | {100*state["ray_refinement_max_change"]:.3f} pp | {100*state["collision_refinement_max_change"]:.3f} pp | {selected} |')
    stages={key:sum(b[key] for b in data['batches']) for key in
            ['kernel_setup_seconds','prior_seconds','solve_seconds','aggregation_seconds']}
    live=[b['cupy_live_bytes_at_yield'] for b in data['batches']]
    process=[o['own_memory_bytes'] for o in data['gpu_process_audit']['observations'] if o['own_memory_bytes'] is not None]
    summary={'scope':data['scope'],'prefix_diagnostics':prefix,'sample_generation_stage_sums':stages,
             'maximum_batch_coefficient_bytes':max(b['coefficient_bytes'] for b in data['batches']),
             'cupy_live_bytes_at_yields_min_max':[min(live),max(live)],
             'sampled_process_memory_max_bytes':max(process),'checkpoints':data['checkpoints'],
             'maximum_independent_finite_residual':data['maximum_independent_finite_residual'],
             'maximum_cached_error_over_fluctuations':data['maximum_cached_error_over_fluctuations'],
             'retained_control_quality':cq,'complete_application_quality_pass':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    seconds=data['seconds'];isolation=data['gpu_process_audit']['sampled_isolation_pass']
    timings='passed' if isolation else 'did not pass'
    report=f'''# Streaming coherent application statistics

At one million observations, **128 coherent draws now flow through cached
field, ray, collision and next-view queries without retaining the sample
coefficient ensemble**. The largest yielded coefficient block is
{summary['maximum_batch_coefficient_bytes']/2**20:.2f} MiB (two draws), independent
of the total requested sample count. This removes the coefficient-list plus
concatenation limit from this streaming path; the original `draw()` still
retains its ensemble. This is an execution and convergence diagnostic, not a
new matching-quality application speedup.

## Numerical checks

The [eight-draw control](million-control8-corrected/results.json) compares
retained and streamed samples on the same fitted model and seed/batch schedule.
Its covariance discrepancy is {100*cq['fields']['empirical_covariance_relative_l2']:.6f}%;
sampled ray/collision events and the selected view agree. Fields and covariance
pass the unchanged strict same-model gates. The initial pre-score-correction
pilot is retained in [legacy-control8](legacy-control8/results.json).

All 128 streamed draws pass independently summed finite-covariance rows:
maximum relative residual **{100*data['maximum_independent_finite_residual']:.6f}%**
against 0.002%. Cached fields differ from ordinary type-2 evaluation by at most
**{100*data['maximum_cached_error_over_fluctuations']:.6f}%** of the fluctuation norm,
against 0.05%. These 32 validation rows are a numerical sample, not a full-system
or analytic-model certificate. The fit and prior use a consistently coupled
cube200 vector model and cutoff50 scalar/cross model; no narrower hybrid inverse
is used. Full analytic-tail and scalar-band convergence remain separate work.

## Probability and geometry convergence remain open

| Draws | Simultaneous MC radius | Ray refinement max change | Sphere refinement max change | Selected view |
| ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The [declared protocol](PROTOCOL.md) applies a 95% Hoeffding union bound across
all reported event locations and all three checkpoints. It requires independent
draws but does not assume independent spatial queries. The target is a radius
of at most **two percentage points**; the 128-draw run **does not pass**.
This conservative bound is not an estimate of actual error. The next-view
score bounds likewise do not certify a stable optimal action. Uncertainty
coverage includes {data['checkpoints']['128']['uncertain_collision_count']} nontrivial sphere-probe probabilities.

Refinement inserts ray midpoints (256 → 511 positions) and adds sphere-surface
points (256 → 512), preserving all old coordinates exactly. The table reports
paired empirical changes, not a bound on continuous geometry error. These are
sphere-surface collision proxies, not full solid collision tests. The
[prefix diagnostics](summary.json) also compare 8 and 32 draws with the final
128-draw estimate; the latter includes those prefixes and is not independent
ground truth. No complete matching-quality claim follows from this run.

Next-view scores now integrate a floating-point uncertainty-band indicator.
The inherited Boolean trapezoid undercounted consecutive true intervals. The
[historical audit](../view_score_audit/README.md) preserves source records and
checks the consequences; new diagnostics use gate version 3. Sequential scan
trajectories without saved transmittance still require rerunning.

## Execution and memory accounting

The single instrumented run takes {seconds['instrumented_work_wall']:.2f} s,
including {seconds['fit']:.2f} s fitting and {seconds['stream_wall']:.2f} s in the
streaming loop. The loop includes independent validation, refined geometry,
transfers, online statistics and checkpoint writes. Sample-generation stage sums
are prior {stages['prior_seconds']:.2f} s, conditional solve {stages['solve_seconds']:.2f} s,
and scalar aggregation {stages['aggregation_seconds']:.2f} s. Other stages and all
batch records are in [results.json](million128/results.json).
The whole-run process audit **{timings}** sampled isolation. This is one run,
not repeated optimized application timing. Do not compare this wall time directly
with the earlier fixed-RHS centered150 conditioning-only measurements.

CuPy live allocations observed at yields range from {min(live)/2**30:.2f} to
{max(live)/2**30:.2f} GiB; the largest sampled own-process usage is
{max(process)/2**30:.2f} GiB. These are different measurements: CuPy live bytes
omit native-library/context allocations, while process observations include
reserved pools but can miss transient peaks. Neither is an exact device peak.
The tests check release of previous coefficient arrays and bounded live storage
over 128 small-problem draws; large-input records show every yielded batch.

## API and reproduction

```python
from contextlib import closing
from reproduction.half_posterior import HalfCubePosterior
from geospsr.cached_spectral import CachedFourierEvaluator

model = HalfCubePosterior(points, normals, training_cutoff=200)
with closing(model.draw_batches(128, seed=51, batch=2, pack_rhs=True)) as batches:
    for block in batches:
        cache = CachedFourierEvaluator(block['coefficients'])
        fields = cache.evaluate(queries) - model.offset
        # Reduce these coherent draws into event counts or moments here.
        del fields, cache, block
```

Keep batching and seed fixed when comparing realizations. Use `closing` for
early exits or consumer exceptions. Only one stream may be active per model;
streaming clears the previously retained sample ensemble/cache, and mean
queries remain available. Tests cover partial batches, packed and folded solves,
optional centered inverses, cleanup and online moments/counts against retained
draws and independent dense controls. [Test output](tests.log).

The benchmark can optionally export coefficient blocks to CPU/disk using
`--coefficient-dir`; this enables later loading for moving queries, but is not
a checkpoint/resume protocol. Do not treat an incomplete run as a full ensemble.

```bash
./reproduction/run_env.sh -m reproduction.stream_applications --samples 8 --checkpoints 8 --retained-control --output reproduction/results/streamed_applications/million-control8-corrected
./reproduction/run_env.sh -m reproduction.stream_applications --output reproduction/results/streamed_applications/million128
./reproduction/run_env.sh -m pytest tests -q
./reproduction/run_env.sh -m reproduction.summarize_streamed_applications
```

The source/geometry scratch dataset is the existing common-RHS million-point
dataset; fresh coherent samples are generated rather than replaying stored RHS.
[Issue #20](https://github.com/alecjacobson/fast-stochastic-psr/issues/20) tracks
memory/scaling, and [#3](https://github.com/alecjacobson/fast-stochastic-psr/issues/3)
tracks complete application accuracy and timing.
'''
    (OUT/'README.md').write_text(report)
    print(json.dumps({k:v for k,v in summary.items() if k in ['maximum_batch_coefficient_bytes','cupy_live_bytes_at_yields_min_max','sampled_process_memory_max_bytes','complete_application_quality_pass']}))


if __name__=='__main__': main()
