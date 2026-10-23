"""Keep fused-application, conditioning and setup scopes separate."""
import json
import re
from pathlib import Path
import numpy as np
from reproduction.summarize_nystrom_storage import verify_saved_fields,checked_quality

OUT=Path(__file__).resolve().parent/'results/nystrom_fused'


def load(name):return json.loads((OUT/name/'results.json').read_text())


def main():
    result={'scope':'same-model preconditioner/conditional-solve acceleration; no upstream or complete-application ratio'}
    micro_rows=[];micro={}
    for size in [1,3]:
        runs=[load(f'apply{size}m-{i}') for i in [1,2,3]]
        assert all(d['complete'] and d['N']==size*1_000_000 for d in runs)
        assert len({d['points_sha256'] for d in runs})==1
        assert len({d['kernel_source_sha256'] for d in runs})==1
        assert all(v['end_state_matches'] for d in runs for v in d['pool_live_memory'].values())
        quality=all(v['pass'] and v['zero_channel_exact'] for d in runs for v in d['quality'].values())
        eligible=quality and all(d['gpu_process_audit']['sampled_isolation_pass'] for d in runs)
        medians={k:float(np.median([d['medians'][k] for d in runs])) for k in ['decoded','fused']}
        extra={k:max(d['pool_live_memory'][f'{k}_{i}']['peak_live_bytes']-d['pool_live_memory'][f'{k}_{i}']['initial_live_bytes']
                     for d in runs for i in range(7)) for k in ['decoded','fused']}
        item={'quality_pass':quality,'timing_ratio_eligible':eligible,'median_seconds':medians,
              'incremental_pool_peak_bytes':extra,'storage_bytes':runs[0]['preconditioner']['storage_bytes'],
              'maximum_error_over_input':{k:max(max(d['quality'][k]['error_over_input']) for d in runs) for k in ['decoded','fused']}}
        if eligible:item['action_speedup']=medians['decoded']/medians['fused']
        micro[str(size*1_000_000)]=item
        ratio=f'{item["action_speedup"]:.2f}×' if eligible else 'ineligible'
        micro_rows.append(f'| {size:,} million | {1000*medians["decoded"]:.3f} | {1000*medians["fused"]:.3f} | {ratio} | '
                          f'{extra["decoded"]/2**20:.2f} / {extra["fused"]/2**20:.2f} | {quality} |')
    result['action_profiles']=micro
    source=json.loads((OUT.parent/'large_inputs/scorpion_3000000/manifest.json').read_text())
    assert load('apply3m-1')['points_sha256']==source['arrays']['points']['payload_sha256']
    assert source['distinct_observed_positions']==3_000_000
    names=['plain8','monolithic32','decoded16','fused16'];raw={name:load(name) for name in names}
    baseline=raw['plain8'];conditioning={};rows=[];quality_rows=[]
    for name,d in raw.items():
        a=d['args'];assert d['complete'] and d['N']==1_000_000
        assert a['samples']==8 and a['batch']==2 and a['repeats']==3 and a['pack_rhs']
        assert a['solve_cutoff']==200 and a['rtol']==1e-5 and a['profile_pool']
        assert d['source_integrity']==baseline['source_integrity'] and d['geometry_sha256']==baseline['geometry_sha256']
        assert all(v['end_state_matches'] for v in d['pool_live_memory'].values())
        good=d['independent_finite_residual']['pass'];q=d.get('quality')
        if q:
            good=good and checked_quality(q)
            verify_saved_fields(OUT/'plain8/fields.npz',OUT/name/'fields.npz',q)
        iterations=[v['iterations'] for repeat in d['iterations'] for v in repeat]
        assert all(v['converged'] for repeat in d['iterations'] for v in repeat)
        setup=d['seconds'].get('preconditioner_setup',0.)
        median=float(np.median([d['seconds'][f'solve_{i}'] for i in range(3)]))
        item={'quality_pass':good,'sampled_isolation_pass':d['gpu_process_audit']['sampled_isolation_pass'],
              'setup_seconds':setup,'median_solve_seconds':median,'setup_plus_median_seconds':setup+median,
              'setup_plus_first_solve_seconds':setup+d['seconds']['solve_0'],
              'storage_bytes':d.get('preconditioner_storage_bytes',0),
              'iteration_range':[min(iterations),max(iterations)],
              'finite_residual':d['independent_finite_residual']['maximum']}
        conditioning[name]=item
        rows.append(f'| {name} | {item["storage_bytes"]/2**30:.3f} | {setup:.3f} | {median:.3f} | '
                    f'{setup+median:.3f} | {item["setup_plus_first_solve_seconds"]:.3f} | {min(iterations)}–{max(iterations)} |')
        field='—' if q is None else f'{100*q["fields"]["error_over_fluctuation_l2"]:.6f}%'
        cov='—' if q is None else f'{100*q["fields"]["empirical_covariance_relative_l2"]:.6f}%'
        quality_rows.append(f'| {name} | {field} | {cov} | {100*item["finite_residual"]:.6f}% | '
                            f'{good if q else "finite residual only"} |')
    eligible=all(v['quality_pass'] and v['sampled_isolation_pass'] for v in conditioning.values())
    result['conditioning']={'runs':conditioning,'timing_ratio_eligible':eligible}
    timing='No conditional-solve timing ratio is eligible under the declared all-four audit/quality policy.'
    if eligible:
        fused=conditioning['fused16']
        ratios={name:v['median_solve_seconds']/fused['median_solve_seconds'] for name,v in conditioning.items() if name!='fused16'}
        result['conditioning']['warm_speedups_to_fused']=ratios
        result['conditioning']['setup_plus_median_ratios_to_fused']={name:v['setup_plus_median_seconds']/fused['setup_plus_median_seconds'] for name,v in conditioning.items() if name!='fused16'}
        timing=(f'All four whole-run sampled-isolation and numerical checks pass. With the basis prepared, fused conditioning '
                f'is **{ratios["decoded16"]:.2f}× faster than decoded half storage**, **{ratios["monolithic32"]:.2f}× '
                f'faster than the stronger float32 monolithic preconditioner**, and **{ratios["plain8"]:.2f}× '
                'faster than plain packed CG**. These are within-implementation conditional-solve ratios. '
                'Process sampling can miss brief overlaps; the full-solve pilot uses one fresh process per configuration '
                'with three solve repetitions, not three independent process repetitions.')
    confirmations=[]
    for name in ['fused128','coherent8']:
        if not (OUT/name/'results.json').exists():continue
        d=load(name);info={'complete':d['complete']}
        if not d['complete']:
            info['failure']=d.get('failure','incomplete');confirmations.append(f'**{name} failed:** `{info["failure"]}`.')
        elif name=='fused128':
            q=d['heldout_quality'];assert q['start']==8 and q['count']==120
            assert d['preconditioner']['application_backend']=='fused_half' and d['args']['rtol']==1e-5
            assert d['source_integrity']==baseline['source_integrity'] and d['geometry_sha256']==baseline['geometry_sha256']
            reference=OUT.parent/'nufft_batches/nufft128/fields.npz'
            verify_saved_fields(reference,OUT/name/'fields.npz',d['quality'])
            verify_saved_fields(reference,OUT/name/'fields.npz',q,start=8)
            info.update(full=d['quality'],heldout=q,finite=d['independent_finite_residual'])
            info['quality_pass']=checked_quality(d['quality']) and checked_quality(q) and d['independent_finite_residual']['pass']
            confirmations.append(f'All 128 common draws: numerical gates **{info["quality_pass"]}**. '
                f'On the 120 pilot-held-out draws, field/fluctuation error is {100*q["fields"]["error_over_fluctuation_l2"]:.6f}% '
                f'and empirical covariance discrepancy {100*q["fields"]["empirical_covariance_relative_l2"]:.6f}%. '
                f'Maximum ray/collision differences are {100*q["events"]["rays"]["max_probability_difference"]:.4f}/'
                f'{100*q["events"]["collision"]["max_probability_difference"]:.4f}pp; selected view '
                f'{q["events"]["next_view"]["reference_best"]} → {q["events"]["next_view"]["candidate_best"]}. '
                f'Independent finite residual is {100*d["independent_finite_residual"]["maximum"]:.6f}%. '
                f'Recorded setup/conditioning times {d["seconds"]["preconditioner_setup"]:.3f}/'
                f'{d["seconds"]["solve_0"]:.3f}s are diagnostics, not a ratio against the archived reference. '
                '[Raw confirmation](fused128/results.json). Geometry is reused; only additional draws are held out.')
        else:
            q=d['retained_control_quality']
            assert d['conditioning_preconditioner']['application_backend']=='fused_half'
            assert d['sample_count']==8 and not d['retained_ensemble_coefficients']
            info.update(quality=q,prior_rhs_exact=d['retained_control_prior_rhs_exact'])
            info['quality_pass']=checked_quality(q) and info['prior_rhs_exact'] and d['finite_residual_gate_pass'] and d['cache_gate_pass']
            confirmations.append(f'Eight-draw coherent streaming control: numerical gates **{info["quality_pass"]}**, '
                f'bitwise prior/RHS equality **{info["prior_rhs_exact"]}**, covariance discrepancy '
                f'{100*q["fields"]["empirical_covariance_relative_l2"]:.6f}%, maximum ray/collision difference '
                f'{100*q["events"]["rays"]["max_probability_difference"]:.4f}/'
                f'{100*q["events"]["collision"]["max_probability_difference"]:.4f}pp. '
                'The unpreconditioned retained control shares the fitted mean, seed 51 and batch schedule. '
                '[Raw control](coherent8/results.json).')
        result[name]=info
    startup='A fresh-cache compilation diagnostic is not yet recorded.'
    if (OUT/'cold-start.json').exists():
        cold=json.loads((OUT/'cold-start.json').read_text());assert cold['complete'] and cold['cache_initially_empty']
        result['cold_start']=cold
        startup=(f'A separate empty-cache 257-row/17-column/six-RHS diagnostic measures compile plus first fused use: '
                 f'{cold["seconds"]["float32_cold"]:.3f}s (float32), {cold["seconds"]["float64_cold"]:.3f}s (float64). '
                 'It excludes process/CUDA initialization and is not a throughput comparison. '
                 '[Raw startup record](cold-start.json).')
    tests=(OUT/'tests.log').read_text() if (OUT/'tests.log').exists() else ''
    count=re.findall(r'(\d+) passed',tests);result['tests_passed']=int(count[-1]) if count else None
    test_text=f'All **{count[-1]} tests pass** ([log](tests.log)).' if count else 'Full-suite result is not yet recorded.'
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'README.md').write_text(f'''# Fused compact-basis application and full conditioning

Fusing decode with multiplication removes the float32 basis temporaries while
preserving the stored half basis, solve precision and full covariance operator.
The [protocol](PROTOCOL.md) separates component timing from full conditioning
and includes the stronger float32 monolithic baseline.

## What carries through to conditional solves

{timing}

NVIDIA L40, one million observations, eight common draws, packed batch 2,
full cube 200/scalar 50, noise .02, CG tolerance 1e-5 and NUFFT epsilon 1e-6.
Every preconditioner uses rank 1,024 and landmark seed 41. Decoded/fused half
storage uses two-pass feature construction; the float32 baseline retains the
faster full double feature cache. Thus storage, setup and repeated-solve costs
are explicit alternatives, not interchangeable timing scopes.

| Configuration | Retained preconditioner (GiB) | Setup (s) | Median solves (s) | Setup + median (s) | Setup + first solve (s) | Iterations/batch |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

**At eight draws, fused setup plus conditioning is still slower than plain
packed CG and the float32 preconditioner.** The faster warmed solve does not
erase its setup cost for this small ensemble. More draws can amortize setup,
but a measured larger-ensemble paired timing comparison is still needed; do
not extrapolate a headline ratio from per-draw arithmetic.

These timers exclude source/prior generation, fitting, transform plans,
scalar aggregation, query evaluation and rendering. They are not complete
application timings or a comparison against upstream SDD. Every raw record
retains all three repetitions, stage timings, allocation events and process
observations. Pool profiling overhead is enabled for every configuration.

## Numerical checks and confirmation

| Configuration | Field error / fluctuations | Empirical covariance discrepancy | Independent finite-row residual | Numerical/event gates |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(quality_rows)}

The source arrays and geometry are identical and validated. Same-model gates
remain .05% field/fluctuation error, .01% empirical variance/covariance error,
the current paired ray/collision/next-view gates and .002% independent finite
residual on 32 rows. Saved fields/events are rechecked when generating this
report. Numerical agreement does not certify absolute sampling accuracy,
continuous geometry or the omitted analytic training spectrum.

{chr(10).join(chr(10)+s+chr(10) for s in confirmations)}

## Component timing at one and three million observations

Each size has three fresh-process profiles, each with seven warmed repetitions
per method in alternating order. Both methods apply the same stored half basis
and damping to six RHS. The RHS include mixed signs/scales, an exact zero
channel and a coarse-space direction that exposes cancellation. A blocked
float64 evaluation provides the arithmetic reference; error/input norm must be
below 5e-6 in every channel, with exact zeros preserved.

| Observations | Decoded application (ms) | Fused application (ms) | Eligible action ratio | Incremental pool peak, decoded / fused (MiB) | Arithmetic gates |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(micro_rows)}

The action ratios are much larger than the full-solve improvement above.
NUFFT covariance applications still occupy most of each CG iteration. The
three-million-point input has verified distinct observed positions
([provenance](../large_inputs/README.md)). Its action profile is **not** a
three-million-point posterior, application or upstream resource-limit result.
Retained half-basis storage is about 1.907/5.722 GiB at these two sizes.

Incremental pool peaks are measured above each call's starting live allocations
and include its new output and projection workspace. They exclude native
allocations, reserved pools, contexts and other threads; they are not exact
device-memory peaks. Allocation/free counts reconcile with the pool at every
measured stage. Source loading, basis construction, double-reference evaluation
and first use are recorded separately from warmed medians.

The numerical tests can warm CuPy's on-disk compilation cache before profiles
and full solves. Process first-use timings therefore do not claim cold NVRTC
compilation. {startup}

## Implementation and use

Projection reads half values into registers, accumulates at float32 or float64
solve precision and reduces fixed partial tiles. Correction uses the same
effective basis and warp reductions. Half2 loads are used only when row
alignment permits them; odd ranks take scalar half loads. Pointer arithmetic
uses 64-bit source offsets. No RHS is rounded to half, and no tensor-core
precision substitution or fast-math option is enabled.

Projection scratch is bounded by 1024 × rank × RHS_count solve-dtype values
(24 MiB for rank 1,024/six float32 RHS), plus the output and small projection.
It does not grow with observation count after the source-tile cap. The stored
basis still grows as N × rank. The existing post-storage numerical SPD guard,
true original-system residual checks and posterior gates remain required.

```python
model.prepare_conditioning_preconditioner(
    rank=1024, basis_dtype='float16', recompute_features=True,
    fused_apply=True)
# Then consume and close draw_batches(..., preconditioned=True, pack_rhs=True).
```

Fused application is opt-in and incompatible with decoded `apply_row_batch`.
The cache key includes the backend. Default preparation/draw behavior is
unchanged. Tests compare against independent double NumPy actions, long Fourier
covariances and dense finite solves. They exercise both solve precisions,
partial rank/source/channel tiles, a source count beyond the tile cap, strided
RHS, zero channels, scale variation, seams, duplicate points and low-noise
streaming. {test_text}

## Reproduction

```bash
./reproduction/run_env.sh -m reproduction.benchmark_compact_apply \\
  --points /opt/geospsr-runtime/common-rhs-128/points.npy \\
  --output /opt/geospsr-runtime/nystrom-fused/apply1m-1
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \\
  --samples 8 --batch 2 --repeats 3 --pack-rhs --finite-rows 32 --profile-pool \\
  --scratch /opt/geospsr-runtime/common-rhs-128 \\
  --output /opt/geospsr-runtime/nystrom-fused/plain8
```

Run three application profiles per size with separate outputs; three-million
points are at `/opt/geospsr-runtime/large-cloud-3000000/points.npy` after the
documented CPU preparation. For conditioning, use separate outputs and
`--reference` pointing to plain8. Add `--preconditioner-rank 1024` for the
float32 monolithic case; additionally `--basis-dtype float16 --recompute-features`
and `--preconditioner-apply-rows 32768` for decoded half. Replace that last flag
with `--fused-preconditioner` for fused half. Preserve the complete four-case
sweep. Existing legacy sources require the documented
[regeneration audit](../nystrom_validation/README.md#source-integrity-correction).
Regenerate this report with `reproduction.summarize_nystrom_fused`.
''')
    print(json.dumps({'action':micro,'conditioning':result['conditioning'],'tests_passed':result['tests_passed']}))


if __name__=='__main__':main()
