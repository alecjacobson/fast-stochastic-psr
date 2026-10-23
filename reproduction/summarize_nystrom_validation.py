"""Summarize the fixed-rank confirmation, retaining precision failures."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'results/nystrom_validation'


def read(name):
    return json.loads((OUT/name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    common=read('common128/results.json');stream=read('coherent8/results.json')
    provenance=read('provenance.json');audit=read('storage-audit.json')
    assert common['complete'] and stream['complete']
    assert common['N']==stream['N']==1_000_000
    assert common['args']['preconditioner_rank']==stream['args']['preconditioner_rank']==1024
    assert common['args']['samples']==128 and common['args']['quality_holdout_start']==8
    assert common['heldout_quality']['count']==120
    assert common['independent_finite_residual']['pass']
    assert stream['sample_count']==8 and stream['retained_control_prior_rhs_exact']
    assert not stream['retained_ensemble_coefficients']
    assert stream['finite_residual_gate_pass'] and stream['cache_gate_pass']
    assert all(r['converged'] and r['preconditioned'] for repeat in common['iterations'] for r in repeat)
    assert all(r['converged'] and r['preconditioned'] for r in stream['batches'])
    reference=OUT.parent/'nufft_batches/nufft128'
    assert sha(reference/'results.json')==provenance['reference_record_sha256']
    assert sha(reference/'fields.npz')==provenance['reference_fields_sha256']
    assert sha(OUT/'storage-audit.json')==provenance['storage_audit_sha256']
    assert common['geometry_sha256']==provenance['geometry_sha256']
    assert audit['source_manifest_sha256']==hashlib.sha256(
        (json.dumps(common['source_manifest'],indent=2)+'\n').encode()).hexdigest()
    for audit_name,validation_name,count in [('storage-audit.json','provenance.json',128),
                                           ('storage-audit-B.json','source-integrity-B.json',4)]:
        a=read(audit_name);v=read(validation_name)
        assert a['complete'] and len(a['draws'])==count
        assert all(d['rhs_exact_after_storage_cast'] and d['prior_exact'] for d in a['draws'])
        assert a['stored_rhs_sha256']==v['actual_array_sha256']['all_rhs']
        assert a['prior_sha256']==v['actual_array_sha256']['all_prior']
        assert sha(OUT/audit_name)==v['storage_audit_sha256']
    fresh=read('roundtrip/results.json');fresh_check=read('roundtrip-validation.json')
    assert fresh['complete'] and fresh['array_hash_schema']==fresh_check['array_hash_schema']==2
    assert fresh['hashes']==fresh_check['actual_array_sha256'] and not fresh_check['legacy_cast_audit']
    with np.load(reference/'fields.npz') as a,np.load(OUT/'common128/fields.npz') as b:
        assert a['fields'].shape==b['fields'].shape==(128,257)
        for start,quality in [(0,common['quality']),(8,common['heldout_quality'])]:
            x=a['fields'][start:];y=b['fields'][start:]
            covariance=np.linalg.norm(np.cov(y[:,:32].T)-np.cov(x[:,:32].T))/np.linalg.norm(np.cov(x[:,:32].T))
            assert np.isclose(covariance,quality['fields']['empirical_covariance_relative_l2'],rtol=1e-8)
            for key in ['rays','collision']:
                delta=(b[key][start:]-a[key][start:]).mean(0)
                assert np.isclose(np.abs(delta).max(),quality['events'][key]['max_probability_difference'])
    checks={'All 128 common draws':common['quality'],'120 pilot-held-out draws':common['heldout_quality'],
            '8 coherent streamed draws':stream['retained_control_quality']}
    rows=[]
    for label,q in checks.items():
        f,e=q['fields'],q['events']
        assert f['pass'] and e['compatibility_pass'] and e['uncertainty_coverage_pass']
        assert e['quality_gate_version']==3 and not e['application_validation_complete']
        rows.append(f'| {label} | {100*f["error_over_fluctuation_l2"]:.6f}% | '
                    f'{100*f["empirical_covariance_relative_l2"]:.6f}% | '
                    f'{100*e["rays"]["max_probability_difference"]:.4f} | '
                    f'{100*e["collision"]["max_probability_difference"]:.4f} | '
                    f'{e["next_view"]["reference_best"]} → {e["next_view"]["candidate_best"]} |')
    precision=read('precision/results.json');precision_rows=[]
    assert len(precision['cases'])==12
    for case in precision['cases']:
        if case['complete']:
            outcome='pass' if case['fields']['pass'] else 'quality failure'
            error=f'{100*case["fields"]["empirical_covariance_relative_l2"]:.6f}%'
        else:outcome='execution/residual failure';error='—'
        precision_rows.append(f'| {case["eps"]:g} | {case["rtol"]:g} | '
                              f'{"packed / zero" if case["packed"] else "folded / mean"} | {error} | {outcome} |')
    assert '106 passed' in (OUT/'tests.log').read_text()
    summary={'scope':'fixed finite-model numerical validation; no new timing ratio or full application certification',
             'quality':checks,'common128_finite_residual':common['independent_finite_residual']['maximum'],
             'stream_finite_residual':stream['maximum_independent_finite_residual'],
             'stream_prior_rhs_exact':stream['retained_control_prior_rhs_exact'],
             'legacy128_regeneration_exact':True,'legacyB4_regeneration_exact':True,
             'fresh_serialized_hash_schema':2,'precision_cases':precision['cases'],
             'timing_ratio_eligible':False,'tests_passed':106}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'README.md').write_text(f'''# Nyström confirmation and reusable streaming

**The fixed rank-1,024 preconditioner passes 128-draw numerical validation at
one million observations**, including 120 draws held out from the eight-draw
pilot. It also passes a separate coherent streaming control. There is no new
speedup ratio: the GPU was shared, and the archived reference is not a current
packed-solver timing baseline.

The [protocol](PROTOCOL.md) fixes rank, seed, tolerances and acceptance before
the confirmation. Geometry and observations are reused; only the additional
draws are held out. The full-cube200 training action, scalar cutoff50, noise
variance .02 and solver tolerance1e-5 remain fixed. The preconditioner changes
the iterative solve, not the target finite posterior. All 64 two-draw batches
take 38 iterations, consistent with the pilot's reduction from 57 to 38.

## Numerical and downstream controls

| Comparison | Field error / fluctuations | Empirical covariance discrepancy | Max ray difference (pp) | Max collision difference (pp) | Selected view |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

The common128 reference is [nufft128](../nufft_batches/nufft128/results.json),
whose implementation at commit8fa5204 uses full cube200, noise .02 and a fitted
mean starting guess. That starting guess does not change the converged model.
Both sides use the same saved RHS, priors and query geometry. The current
numeric 5%–95% ray score is recomputed from raw fields/events; historical
gate-version2 scores are not reused. The maximum collision difference is one
event among 128 draws (or among 120 held-out draws), within the unchanged 2pp
paired numerical gate. Passing that gate does not establish absolute accuracy.

Independent double-precision finite cosine sums check 32 training rows:
maximum residual {100*common['independent_finite_residual']['maximum']:.6f}%
for common128 and {100*stream['maximum_independent_finite_residual']:.6f}%
for coherent8, against the unchanged .002% gate. Field/fluctuation error must
be below .05%; empirical variance and covariance discrepancies below .01%.

The coherent control uses fresh seed51 draws from the folded prior generator,
not the seed42 common-RHS archive. It compares retained unpreconditioned
packed solves with streamed preconditioned packed solves on the same fitted
mean. Prior coefficients and RHS match **bit for bit**. Cached query error is
{100*stream['maximum_cached_error_over_fluctuations']:.6f}% of fluctuations.
The validation control deliberately retains reference RHS/prior arrays;
ordinary streaming retains only the current sample batch, online statistics
and the shared preconditioner. No full ensemble coefficients remain in the
model during streaming.

Eight samples have a simultaneous probability radius of
{100*stream['checkpoints']['8']['simultaneous_probability_radius']:.2f}pp.
These controls do not resolve Monte Carlo, geometric discretization, scalar
bandwidth or full analytic training-tail convergence. The
[collision-refinement failure](../query_refinement/README.md) remains.
They do not establish ContactFusion's sequential update semantics or speed.

## Reusable preparation and memory

```python
from contextlib import closing
from reproduction.half_posterior import HalfCubePosterior

model = HalfCubePosterior(points, normals)
model.prepare_conditioning_preconditioner(rank=1024, seed=41)
with closing(model.draw_batches(128, seed=51, batch=2,
                               pack_rhs=True, preconditioned=True)) as batches:
    for block in batches:
        values = model.op.evaluate(query_points, block['coefficients']) - model.offset
        # Reduce statistics here; discard values before advancing the stream.
model.clear_conditioning_preconditioner()
```

Preparation is explicit and reused across draw calls on the fixed fitted map.
Default draws and the fitted mean remain unpreconditioned. Closing a partially
consumed generator releases transient plans; preparing/clearing the basis
during an active stream is rejected. Changing observations, kernel or noise
requires a new fitted model. Combining this option with a narrower hybrid
inverse is rejected until separately validated.

Rank1,024 retains 4,096,012,288 bytes (3.815 GiB) at one million observations.
Construction additionally holds a double-precision observation-by-rank array.
Unused construction buffers are released before later NUFFT plans. Streaming
samples does not eliminate this basis cost; [issue #31](https://github.com/alecjacobson/fast-stochastic-psr/issues/31)
tracks larger-scale storage alternatives.

Diagnostic timings: common128 setup {common['seconds']['preconditioner_setup']:.3f}s,
conditioning {common['seconds']['solve_0']:.3f}s; coherent8 setup
{stream['seconds']['preconditioner_setup']:.3f}s and streaming
{stream['seconds']['stream_wall']:.3f}s. Its instrumented total
{stream['seconds']['instrumented_work_wall']:.3f}s also includes fitting,
the retained unpreconditioned control and validation. Neither whole-run
isolation audit passes. Do not divide these by historical timings or interpret
this validation-only total as a production application benchmark.

## Retained low-noise precision failure

A new 23-observation float32 stress case (length scale .12, noise variance
.0001, training cube5, scalar cube2, five draws) exposes covariance discrepancies
above .01% at the default solver tolerance1e-5. The field gate alone passes.
The [initial failure](targeted-tests.log), [unsuccessful tighter-tolerance
attempt](targeted-corrected.log) and all 12 subsequent cases are retained.

| Requested NUFFT epsilon | CG tolerance | Solve / initial guess | Covariance discrepancy | Outcome |
| --- | --- | --- | ---: | --- |
{chr(10).join(precision_rows)}

At epsilon1e-6, tolerance3e-6 passes both representations with the same data
and unchanged gates. That is the stress test's selected setting; the million
point runs retain tolerance1e-5. Tolerance1e-6 hits the float32 true-residual
floor, and requested epsilon1e-7 is clamped to float32 machine precision by
cuFINUFFT. Tightening requested tolerances is not a general accuracy guarantee.
The sweep is exploratory, not a held-out confirmation of an optimal tolerance.
[Raw sweep](precision/results.json), [warnings](precision.log).

## Source integrity correction

The legacy common-RHS preparation hashed float64 intermediate RHS before
casting to its float32 memmap. Read-only regeneration reproduced **all 128
million-point RHS/prior pairs and all four B pairs exactly**, and recovered
both hashes. This corrects provenance bookkeeping; it does not change solver
inputs or rerun historical timings. Old result records remain untouched.

| Million-point payload | SHA-256 |
| --- | --- |
| Legacy pre-cast float64 RHS | `{audit['raw_rhs_sha256']}` |
| Serialized float32 RHS | `{audit['stored_rhs_sha256']}` |
| Scalar prior coefficients | `{audit['prior_sha256']}` |

[128-draw audit](storage-audit.json), [B audit](storage-audit-B.json) and
[reference/source provenance](provenance.json) retain the mapping. Common128
ran before the new loader; its source integrity was checked separately afterward.
New preparation uses schema2 hashes for the exact serialized RHS and prior
arrays. A fresh two-draw B preparation passes readback validation without any
legacy exception: [preparation](roundtrip/results.json),
[validation](roundtrip-validation.json). Existing caches require a completed,
matching regeneration audit; missing, stale or failed audits are rejected.

## Reproduction

For an existing legacy cache, first run this read-only regeneration (all saved
draws are checked, regardless of the later benchmark's sample count):

```bash
./reproduction/run_env.sh -m reproduction.audit_common_rhs_storage \\
  --scratch /opt/geospsr-runtime/common-rhs-128 \\
  --output /opt/geospsr-runtime/common-rhs-128/storage_audit.json
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \\
  --samples 128 --batch 2 --repeats 1 --pack-rhs --preconditioner-rank 1024 \\
  --finite-rows 32 --quality-holdout-start 8 \\
  --scratch /opt/geospsr-runtime/common-rhs-128 \\
  --reference reproduction/results/nufft_batches/nufft128 \\
  --output /opt/geospsr-runtime/nystrom-validation/common128
./reproduction/run_env.sh -m reproduction.stream_applications \\
  --samples 8 --checkpoints 8 --retained-control --preconditioner-rank 1024 \\
  --output /opt/geospsr-runtime/nystrom-validation/coherent8
./reproduction/run_env.sh -m reproduction.check_preconditioned_precision \\
  --output /opt/geospsr-runtime/nystrom-validation/precision
```

Fresh schema2 sources need no legacy audit. Compare fresh sources against
references generated from those sources: old and new manifest hash dictionaries
are intentionally not silently treated as identical. The compact saved fields
and [summary](summary.json) are in this directory. Regenerate this report with
`./reproduction/run_env.sh -m reproduction.summarize_nystrom_validation`.
All **106 tests pass** ([full log](tests.log)), including dense finite solves,
exact prior/RHS checks, partial batches, early stream closure, cache reuse,
hashing after dtype conversion and corrupted source rejection.
''')
    print('Validated 128/120 common draws, coherent8, storage audits and all precision outcomes.')


if __name__=='__main__':main()
