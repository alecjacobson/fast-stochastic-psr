"""Summarize every eligible repetition of the declared packed-RHS trial."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduction/results/packed_rhs'


def validated(name):
    data=json.loads((OUT/name/'results.json').read_text())
    assert data['complete'],name
    quality=data['quality']
    assert quality['comparison_kind']=='same model numerical comparison',name
    assert quality['fields']['pass'],name
    assert quality['events']['compatibility_pass'] and quality['events']['uncertainty_coverage_pass'],name
    assert data['analytic_holdout']['max_upper_endpoint']<.001,name
    return data


def main():
    report={'scope':'same finite-model conditioning; not a complete-application or upstream speedup','methods':{}}
    table=[]
    for name,label in [('folded150-six','Folded, batch 1'),('packed150-six','Packed, batch 1'),('packed150-batch2-six','Packed, batch 2')]:
        data=validated(name)
        trials=[]
        paths=[name]
        if (OUT/(name+'-extra')/'results.json').exists():
            paths.append(name+'-extra')
        for path in paths:
            d=validated(path)
            for i in range(d['args']['repeats']):
                key=f'solve_{i}'
                trials.append({'run':path,'repeat':i,'seconds':d['seconds'][key],**d['timing_process_audits'][key]})
        eligible=[r for r in trials if r['sampled_isolation_pass']]
        assert len(eligible)>=3,(name,'insufficient eligible repetitions')
        median=float(np.median([r['seconds'] for r in eligible]))
        memory=max(r['sampled_process_memory_peak_bytes'] for r in eligible)
        report['methods'][name]={'trials':trials,'eligible_count':len(eligible),
            'median_seconds':median,'sampled_process_memory_peak_bytes':memory,
            'quality':data['quality'],'analytic_residual_upper':data['analytic_holdout']['max_upper_endpoint']}
        table.append(f'| {label} | {len(eligible)}/{len(trials)} | {median:.2f} s | {memory/2**30:.2f} GiB |')
    methods=report['methods'];best=min(['packed150-six','packed150-batch2-six'],key=lambda n:methods[n]['median_seconds'])
    report['best_packed_configuration']=best
    report['conditioning_speedup']=methods['folded150-six']['median_seconds']/methods[best]['median_seconds']
    large=validated('packed150-128')
    report['128_draw_check']={'quality':large['quality'],'solve_seconds':large['median_solve_seconds'],
        'timing_audit':large['timing_process_audits']['solve_0'],
        'analytic_residual_upper':large['analytic_holdout']['max_upper_endpoint']}
    smoke=json.loads((OUT/'production-draws/results.json').read_text())
    assert smoke['complete'] and smoke['paired_prior_gate_pass'] and smoke['residual_gate_pass']
    new=np.load(OUT/'production-draws/fields.npz')
    old=np.load(ROOT/'reproduction/results/half_cube/production-draws/fields.npz')
    np.testing.assert_array_equal(new['query'],old['query'])
    np.testing.assert_array_equal(new['validation_ids'],old['validation_ids'])
    field_error=float(np.linalg.norm(new['samples']-old['samples'])/np.linalg.norm(old['samples']-old['mean']))
    assert field_error<5e-4
    report['production_draws']={'field_error_over_fluctuations':field_error,
        'maximum_independent_finite_residual':max(smoke['finite_residual_per_component']),
        'paired_prior_rhs_relative_error':smoke['paired_prior_rhs_relative_error']}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    f=large['quality']['fields'];audit=large['timing_process_audits']['solve_0']
    timing='passes sampled isolation' if audit['sampled_isolation_pass'] else 'is excluded for process-audit failure'
    text=f'''# Paired real RHS: measured conditioning comparison

At one million observations, pairing scaled real RHS in full-cube complex
NUFFTs reduces the eligible eight-draw median from
**{methods['folded150-six']['median_seconds']:.2f} s to {methods[best]['median_seconds']:.2f} s
({report['conditioning_speedup']:.2f}× over our folded solver)**.
Every finite frequency, the noise model and the solver tolerance are retained.
This is an incremental conditioning result, not a complete-application,
upstream SDD or ContactFusion speedup.

## Repetitions and scope

| Method | Eligible / total | Median | Sampled process-memory peak |
| --- | ---: | ---: | ---: |
{chr(10).join(table)}

The [predeclared protocol](PROTOCOL.md) uses all eligible repetitions, with
at least three per configuration. Eligibility requires bracketing process
observations and excludes foreign GPU processes, errors and excessive gaps.
Every raw repetition, including exclusions, is in [summary.json](summary.json).
The initial [pilot](pilot150/results.json) is excluded from these medians.
Periodic process sampling cannot establish exclusive use. Memory is sampled
own-process usage, not an exact allocator peak. Other projects were not stopped.

The timer covers conditional solves, CG control and true residual checks.
Plans, source transfer, prior generation and application queries are outside it.
Each run records those separately. The comparison holds cutoff200 prior/mean,
centered cutoff150 inverse, scalar cutoff50, common RHS and query geometry fixed.
Measurements use NVIDIA L40, float32, CG rtol1e-5 and NUFFT eps1e-6.
The [hybrid-model limitations](../centered_hybrid/README.md) still apply.

## Larger sample count and coherent integration

At 128 shared draws, field error / fluctuation norm is
**{100*f['error_over_fluctuation_l2']:.6f}%** and empirical covariance discrepancy is
**{100*f['empirical_covariance_relative_l2']:.6f}%**. Recorded ray/collision events
and selected view agree. The analytic residual holdout upper endpoint is
{100*large['analytic_holdout']['max_upper_endpoint']:.5f}% against the 0.1% gate.
The raw solve takes {large['median_solve_seconds']:.2f} s and {timing}.
This single run is a numerical confirmation, not repeated 128-draw timing
evidence or a test of absolute Monte Carlo convergence.
[128-draw records](packed150-128/results.json).

`HalfCubePosterior.draw(..., pack_rhs=True)` keeps the coherent folded prior
and mean while applying the full-cube packed inverse. Small tests compare
independent dense finite-covariance solves, retained common RHS/prior arrays,
partial batches, centered narrower solves and cached query fields. An additional
million-point cutoff200 four-draw check differs from saved folded results by
{100*field_error:.6f}% of the fluctuation norm and has maximum independently
summed covariance-row residual {100*report['production_draws']['maximum_independent_finite_residual']:.6f}%.
Its prior agrees with an independent full-grid evaluation. These are execution
and numerical checks, not preparation timing ratios or probability convergence.
[Production records](production-draws/results.json) · [Tests](tests.log).

## Why scaling matters

For a real even kernel, `K(a + i*b) = K(a) + i*K(b)`. Each pair therefore needs
one full complex forward/backward transform. Before packing, each channel is
divided by its own maximum absolute input; output scales are restored separately.
This prevents a large channel's roundoff from overwhelming its small partner.
Zero channels are restored exactly. Tests include 24 orders of magnitude of
channel variation, odd/even counts, periodic seams, zero modes and float32/64.
This trick requires the complete symmetric spectrum; it cannot be applied
directly to the folded operator's real-part action. Prior generation stays
folded and is not naively packed from arbitrary complex Gaussian coefficients.

```python
from reproduction.half_posterior import HalfCubePosterior
model = HalfCubePosterior(points, normals, training_cutoff=200)
model.draw(128, batch=2, solve_cutoff=150, pack_rhs=True)
model.prepare_query_cache()
samples = model.samples(queries)
```

Omit `solve_cutoff` for the consistently coupled finite-cube model. The optional
narrower inverse changes the model; packing alone does not. Full application,
analytic-tail, sample and geometry convergence remain open. One million points
are still feasible for the authors' compiled SDD implementation.

The sampler still retains scalar coefficient blocks before concatenation.
At cutoff50, 4,096 complex64 draws occupy 31.44 GiB; blocks plus the concatenated
array alone require 62.89 GiB. This is an allocation calculation, not a measured
OOM. [Issue #20](https://github.com/alecjacobson/fast-stochastic-psr/issues/20)
tracks streaming query/event accumulation for larger sample-convergence studies.

## Reproduce

Use the existing common-RHS dataset under `/opt/geospsr-runtime/common-rhs-128`.
Run `reproduction.benchmark_nufft_batches solve` with the full arguments in each
`results.json`; the new flag is `--pack-rhs`, mutually exclusive with `--half-cube`.
The three six-repetition configurations are folded batch1, packed batch1 and
packed batch2, all eight draws with centered cutoff150 and analytic holdouts.
The 128-draw confirmation compares with the saved folded128 output.

```bash
./reproduction/run_env.sh -m reproduction.check_half_draws --pack-rhs --output reproduction/results/packed_rhs/production-draws
./reproduction/run_env.sh -m pytest tests -q
./reproduction/run_env.sh -m reproduction.summarize_packed_rhs
```

[Issue #27](https://github.com/alecjacobson/fast-stochastic-psr/issues/27) tracks
the implementation and remaining application coverage.
'''
    (OUT/'README.md').write_text(text)
    print(json.dumps({'best':best,'speedup':report['conditioning_speedup'],'128_draw_timing_eligible':audit['sampled_isolation_pass']}))


if __name__=='__main__':
    main()
