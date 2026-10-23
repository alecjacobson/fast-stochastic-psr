"""Report the declared storage sweep, including costs and failed outcomes."""
import json
import re
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'results/nystrom_storage'


def load(name):
    return json.loads((OUT/name/'results.json').read_text())


def checked_quality(q):
    return bool(q['fields']['pass'] and q['events']['compatibility_pass']
                and q['events']['uncertainty_coverage_pass'] and q['events']['quality_gate_version']==3)


def verify_saved_fields(reference,candidate,quality,start=0):
    with np.load(reference) as left,np.load(candidate) as right:
        a,b=left['fields'][start:],right['fields'][start:]
        assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        relative=lambda x,y:float(np.linalg.norm(x)/max(np.linalg.norm(y),1e-30))
        values={'error_over_fluctuation_l2':relative(b-a,a-a.mean(0)),
                'empirical_variance_relative_l2':relative(b.var(0,ddof=1)-a.var(0,ddof=1),a.var(0,ddof=1)),
                'empirical_covariance_relative_l2':relative(np.cov(b[:,:32].T)-np.cov(a[:,:32].T),np.cov(a[:,:32].T))}
        for key,value in values.items():assert np.isclose(value,quality['fields'][key],rtol=1e-6,atol=1e-12)
        for key in ['rays','collision']:
            delta=(right[key][start:]-left[key][start:]).mean(0)
            assert np.isclose(np.abs(delta).max(),quality['events'][key]['max_probability_difference'])


def main():
    names=['baseline32','recompute32','blocked32','compact16']
    runs={name:load(name) for name in names};baseline=runs['baseline32']
    rows=[];quality_rows=[];summary={}
    for name,d in runs.items():
        a=d['args']
        assert a['samples']==8 and a['batch']==2 and a['repeats']==3 and a['preconditioner_rank']==1024
        assert a['solve_cutoff']==200 and a['rtol']==1e-5 and a['pack_rhs'] and a['profile_pool']
        assert d['source_integrity']==baseline['source_integrity'] and d['geometry_sha256']==baseline['geometry_sha256']
        info={'complete':d['complete'],'sampled_isolation_pass':d['gpu_process_audit']['sampled_isolation_pass']}
        if not d['complete']:
            info['failure']=d.get('failure','incomplete');summary[name]=info
            rows.append(f'| {name} | failed | — | — | — | — | — | — |');continue
        pre=d['preconditioner'];pool=d['pool_live_memory']
        assert all(v['end_state_matches'] for v in pool.values())
        assert pre['landmark_ids_sha256']==baseline['preconditioner']['landmark_ids_sha256']
        assert pre['spd_margin_guard']>0
        iterations=[r['iterations'] for repeat in d['iterations'] for r in repeat]
        assert all(r['converged'] for repeat in d['iterations'] for r in repeat)
        good=d['independent_finite_residual']['pass']
        q=d.get('quality')
        if q is not None:
            good=good and checked_quality(q)
            verify_saved_fields(OUT/'baseline32/fields.npz',OUT/name/'fields.npz',q)
        info.update(quality_pass=good,storage_bytes=pre['storage_bytes'],
                    setup_pool_peak_bytes=pool['preconditioner_setup']['peak_live_bytes'],
                    solve_pool_peak_bytes=max(v['peak_live_bytes'] for k,v in pool.items() if k.startswith('solve_')),
                    setup_seconds=d['seconds']['preconditioner_setup'],
                    median_solve_seconds=float(np.median([d['seconds'][f'solve_{i}'] for i in range(3)])),
                    iteration_range=[min(iterations),max(iterations)],
                    orthogonality_error=pre['orthogonality_frobenius_error'],spd_margin=pre['spd_margin_guard'])
        info['setup_plus_solve_seconds']=info['setup_seconds']+info['median_solve_seconds']
        summary[name]=info
        rows.append(f'| {name} | {info["storage_bytes"]/2**30:.3f} | {info["setup_pool_peak_bytes"]/2**30:.3f} | '
                    f'{info["solve_pool_peak_bytes"]/2**30:.3f} | {info["setup_seconds"]:.3f} | '
                    f'{info["median_solve_seconds"]:.3f} | {info["setup_plus_solve_seconds"]:.3f} | {info["sampled_isolation_pass"]} |')
        field='—' if q is None else f'{100*q["fields"]["error_over_fluctuation_l2"]:.6f}%'
        cov='—' if q is None else f'{100*q["fields"]["empirical_covariance_relative_l2"]:.6f}%'
        quality_rows.append(f'| {name} | {min(iterations)}–{max(iterations)} | {field} | {cov} | '
                            f'{100*d["independent_finite_residual"]["maximum"]:.6f}% | {good if q is not None else "finite residual only"} |')
    eligible=all(v['complete'] and v.get('quality_pass') and v['sampled_isolation_pass'] for v in summary.values())
    result={'scope':'unchanged finite-model conditional-solve storage pilot; not full application validation',
            'runs':summary,'timing_ratio_eligible':eligible}
    timing='**No timing ratio is claimed:** at least one declared whole-run isolation or quality gate fails. Timings are diagnostics; no favorable repetition is selected.'
    if eligible:
        result['setup_plus_solve_ratios']={k:summary['baseline32']['setup_plus_solve_seconds']/v['setup_plus_solve_seconds'] for k,v in summary.items() if k!='baseline32'}
        timing='All declared whole-run sampled-isolation and numerical gates pass. Setup-plus-solve ratios are in [summary.json](summary.json); these compare storage variants of our solver, not upstream applications.'
    confirmation=[]
    for name in ['compact128','coherent8']:
        if not (OUT/name/'results.json').exists():continue
        d=load(name);info={'complete':d['complete']}
        if d['complete']:
            pre=d['preconditioner'] if name=='compact128' else d['conditioning_preconditioner']
            assert d['N']==1_000_000 and pre['basis_dtype']=='float16' and pre['accumulation_dtype']=='float32'
            assert pre['requested_rank']==1024 and pre['seed']==41 and pre['apply_row_batch']==32768 and pre['recompute_features']
        if not d['complete']:
            info['failure']=d.get('failure','incomplete');confirmation.append(f'**{name} failed:** `{info["failure"]}`.')
        elif name=='compact128':
            q=d['heldout_quality'];info.update(full=d['quality'],heldout=q,finite=d['independent_finite_residual'])
            assert q['start']==8 and q['count']==120
            assert d['source_integrity']==baseline['source_integrity'] and d['geometry_sha256']==baseline['geometry_sha256']
            assert d['args']['samples']==128 and d['args']['solve_cutoff']==200 and d['args']['rtol']==1e-5
            reference=OUT.parent/'nufft_batches/nufft128/fields.npz'
            verify_saved_fields(reference,OUT/name/'fields.npz',d['quality'])
            verify_saved_fields(reference,OUT/name/'fields.npz',q,start=8)
            info['quality_pass']=checked_quality(d['quality']) and checked_quality(q) and d['independent_finite_residual']['pass']
            confirmation.append(f'All 128 common draws: numerical gates **{info["quality_pass"]}**. '
                f'On the 120 pilot-held-out draws, field/fluctuation error is '
                f'{100*q["fields"]["error_over_fluctuation_l2"]:.6f}% and empirical covariance discrepancy '
                f'{100*q["fields"]["empirical_covariance_relative_l2"]:.6f}%. '
                f'Maximum ray/collision differences are {100*q["events"]["rays"]["max_probability_difference"]:.4f}/'
                f'{100*q["events"]["collision"]["max_probability_difference"]:.4f}pp; '
                f'selected view {q["events"]["next_view"]["reference_best"]} → {q["events"]["next_view"]["candidate_best"]}. '
                f'Independent finite residual is {100*d["independent_finite_residual"]["maximum"]:.6f}%. '
                f'Recorded setup/conditioning cost is {d["seconds"]["preconditioner_setup"]:.3f}/'
                f'{d["seconds"]["solve_0"]:.3f}s, retained as a diagnostic without a ratio. '
                'The reference is the archived same-model nufft128, not a contemporary timing baseline. '
                'Geometry/observations are reused; only the additional draws are held out. '
                '[Raw confirmation](compact128/results.json).')
        else:
            q=d['retained_control_quality'];info.update(quality=q,prior_rhs_exact=d['retained_control_prior_rhs_exact'])
            assert d['sample_count']==8 and not d['retained_ensemble_coefficients']
            assert all(d['hashes'][key]==baseline['source_integrity']['actual_array_sha256'][key] for key in ['points','normals'])
            info['quality_pass']=checked_quality(q) and info['prior_rhs_exact'] and d['finite_residual_gate_pass'] and d['cache_gate_pass']
            confirmation.append(f'Fresh eight-draw coherent streaming control: numerical gates **{info["quality_pass"]}**, '
                f'prior/RHS bitwise equality **{info["prior_rhs_exact"]}**. Empirical covariance discrepancy '
                f'{100*q["fields"]["empirical_covariance_relative_l2"]:.6f}%; maximum ray/collision differences '
                f'{100*q["events"]["rays"]["max_probability_difference"]:.4f}/'
                f'{100*q["events"]["collision"]["max_probability_difference"]:.4f}pp. '
                'Retained unpreconditioned samples and streamed compact-preconditioned samples share seed 51 and the fitted mean. '
                'The validation control deliberately retains reference RHS/priors. '
                '[Raw control](coherent8/results.json).')
        result[name]=info
    tests=(OUT/'tests.log').read_text() if (OUT/'tests.log').exists() else ''
    matches=re.findall(r'(\d+) passed',tests)
    test_text=f'All **{matches[-1]} tests pass** ([log](tests.log)).' if matches else 'Full-suite result is not yet recorded.'
    result['tests_passed']=int(matches[-1]) if matches else None
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    headline=''
    if summary['compact16'].get('quality_pass'):
        headline=(f'**At one million observations, compact storage lowers retained preconditioner '
                  f'memory from {summary["baseline32"]["storage_bytes"]/2**30:.3f} to '
                  f'{summary["compact16"]["storage_bytes"]/2**30:.3f} GiB**, with unchanged numerical/event '
                  f'gates passing. Recomputed features plus compact storage lower the measured '
                  f'CuPy setup peak from {summary["baseline32"]["setup_pool_peak_bytes"]/2**30:.2f} to '
                  f'{summary["compact16"]["setup_pool_peak_bytes"]/2**30:.2f} GiB. '
                  'This is a storage result; it does not establish a conditioning or application speedup.')
    (OUT/'README.md').write_text(f'''# Compact coarse-basis storage

{headline}

This [declared pilot](PROTOCOL.md) reduces the memory cost of the reusable
Nyström preconditioner while preserving the full-cube200 covariance action.
It tests two independent choices: recomputing feature rows during setup and
storing a scaled basis in float16. Projection and correction still accumulate
at the solve dtype (float32 in this million-point experiment).

## Memory and conditioning cost

NVIDIA L40, one million observations, rank 1,024 / seed 41, eight common seed 42
draws, packed batch 2 and three solve repetitions. Noise variance .02, scalar
cutoff 50, NUFFT epsilon 1e-6, CG tolerance 1e-5. Setup is paid once per map.

| Configuration | Retained basis etc. (GiB) | Setup pool peak (GiB) | Solve pool peak (GiB) | Setup (s) | Median solves (s) | Setup + solves (s) | Whole-run isolation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

- `baseline32`: existing double feature cache, float32 basis, monolithic apply.
- `recompute32`: two-pass feature construction, float32 basis, monolithic apply.
- `blocked32`: same construction/storage, row-blocked apply as an overhead control.
- `compact16`: two-pass construction, scaled float16 basis, row-blocked float32 apply.

Two-pass construction avoids the 8,192,000,000-byte double feature array.
It computes the small feature Gram in the first pass, then regenerates rows
to form the basis after the small eigendecomposition. It performs more kernel
work; it does not approximate the feature Gram further. Construction rows are
8192; application rows are 32768 for both blocked variants.

Pool peaks count rounded live CuPy allocation/free events in every timed stage,
with end-state reconciliation against the pool. They include the existing pool
allocations at each stage's start, and solve peaks include common RHS/weights.
They **exclude native NUFFT plans, CUDA contexts, reserved pool memory and other
allocators/threads**. All timings include the instrumentation overhead. Raw
records also include sampled process memory, which is not an exact device peak.
No memory figure here proves a complete larger-N application fits on the GPU.

{timing}

These totals cover preconditioner setup and conditioning eight pre-generated
RHS; source loading, prior generation, fitting, plans, scalar aggregation and
queries are separate. They are not preparation-through-application timings.
Defaults remain unchanged; compact storage is opt-in.

## Matching-quality checks

| Configuration | Iterations/batch | Field error / fluctuations | Covariance discrepancy | Independent finite residual | Numerical/event gates |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(quality_rows)}

Candidates use the fresh baseline32 fields and the exact same validated
source/RHS/prior/geometry. Field/fluctuation tolerance is .05%; variance and
covariance tolerances are .01%; the paired ray/collision/next-view gates are
unchanged. Independent double cosine sums check 32 rows at a .002% residual
gate. These numerical controls do not certify the full analytic posterior,
absolute Monte Carlo accuracy or continuous-geometry convergence.

{chr(10).join(chr(10)+s+chr(10) for s in confirmation)}

## Rounding, SPD and lifetime

The stored compact array is round_half(U × scale), where scale is a bounded
power of two near sqrt(N); effective U is decoded by dividing by that scale.
This reduces underflow for small basis entries without a learned correction.
Both passes decode bounded row blocks, so there is no additional full
float32 observation-by-rank copy. RHS and accumulation are never cast to half.
The original operator and posterior prior are unchanged.

After storage conversion, the existing double-precision orthogonality/SPD
guard is applied to the decoded basis. It rejects a nonpositive conservative
margin; compression does not bypass this check. A numerical SPD guard is not
an interval-arithmetic proof, and smaller noise or another geometry can fail.
True original-system residuals and posterior field/covariance gates remain
necessary. Small tests include seam coordinates, duplicate points, independent
long Fourier covariance and dense finite solves, partial blocks, cache reuse,
early stream close and low-noise coherent samples. {test_text}

The retained basis still grows as N × rank. At ten million points, float16
storage alone is about 19.07 GiB; construction and transforms need more. This
is a memory reduction, not an N-independent representation or evidence that
upstream SDD is intractable. [Issue #31](https://github.com/alecjacobson/fast-stochastic-psr/issues/31)
remains open for further storage/application tradeoffs and larger-N evidence.
[Issue #33](https://github.com/alecjacobson/fast-stochastic-psr/issues/33) tracks
fusing decoding with projection/correction to avoid the current float32
temporary writes and repeated conversion launches. No fused backend is
implemented or timed by this report.

## Use and reproduction

```python
model.prepare_conditioning_preconditioner(
    rank=1024, basis_dtype='float16', recompute_features=True,
    apply_row_batch=32768)
# Then consume and close model.draw_batches(..., preconditioned=True).
```

The cache key includes storage/construction/application options. Changes to
observations, kernel or noise still require a fresh fitted map. For the full
four-configuration benchmark, start with:

```bash
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \\
  --samples 8 --batch 2 --repeats 3 --pack-rhs --finite-rows 32 \\
  --preconditioner-rank 1024 --profile-pool \\
  --scratch /opt/geospsr-runtime/common-rhs-128 \\
  --output /opt/geospsr-runtime/nystrom-storage/baseline32
```

Repeat in fresh processes with `--reference` set to that baseline and separate
output directories. Add `--recompute-features` for recompute32; also
`--preconditioner-apply-rows 32768` for blocked32; also `--basis-dtype float16`
for compact16. Use the same compact options with 128 samples and one repetition and
`--quality-holdout-start 8 --reference reproduction/results/nufft_batches/nufft128`
for the numerical confirmation. The coherent control uses
`reproduction.stream_applications --samples 8 --checkpoints 8 --retained-control`
with the compact options and rank. Existing legacy caches require the
[source regeneration audit](../nystrom_validation/README.md#source-integrity-correction).
Regenerate this report using `reproduction.summarize_nystrom_storage`.
''')
    print(json.dumps({'timing_ratio_eligible':eligible,'runs':summary,'tests_passed':result['tests_passed']}))


if __name__=='__main__':main()
