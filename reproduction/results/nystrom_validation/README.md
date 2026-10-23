# Nyström confirmation and reusable streaming

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
| All 128 common draws | 0.003232% | 0.001998% | 0.0000 | 0.7812 | 14 → 14 |
| 120 pilot-held-out draws | 0.003227% | 0.002149% | 0.0000 | 0.8333 | 14 → 14 |
| 8 coherent streamed draws | 0.003550% | 0.002211% | 0.0000 | 0.0000 | 4 → 4 |

The common128 reference is [nufft128](../nufft_batches/nufft128/results.json),
whose implementation at commit8fa5204 uses full cube200, noise .02 and a fitted
mean starting guess. That starting guess does not change the converged model.
Both sides use the same saved RHS, priors and query geometry. The current
numeric 5%–95% ray score is recomputed from raw fields/events; historical
gate-version2 scores are not reused. The maximum collision difference is one
event among 128 draws (or among 120 held-out draws), within the unchanged 2pp
paired numerical gate. Passing that gate does not establish absolute accuracy.

Independent double-precision finite cosine sums check 32 training rows:
maximum residual 0.001836%
for common128 and 0.001812%
for coherent8, against the unchanged .002% gate. Field/fluctuation error must
be below .05%; empirical variance and covariance discrepancies below .01%.

The coherent control uses fresh seed51 draws from the folded prior generator,
not the seed42 common-RHS archive. It compares retained unpreconditioned
packed solves with streamed preconditioned packed solves on the same fitted
mean. Prior coefficients and RHS match **bit for bit**. Cached query error is
0.000908% of fluctuations.
The validation control deliberately retains reference RHS/prior arrays;
ordinary streaming retains only the current sample batch, online statistics
and the shared preconditioner. No full ensemble coefficients remain in the
model during streaming.

Eight samples have a simultaneous probability radius of
92.87pp.
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

Diagnostic timings: common128 setup 15.972s,
conditioning 369.469s; coherent8 setup
8.978s and streaming
26.757s. Its instrumented total
83.833s also includes fitting,
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
| 1e-06 | 1e-05 | packed / zero | 0.014036% | quality failure |
| 1e-06 | 1e-05 | folded / mean | 0.017547% | quality failure |
| 1e-06 | 3e-06 | packed / zero | 0.003071% | pass |
| 1e-06 | 3e-06 | folded / mean | 0.002719% | pass |
| 1e-06 | 1e-06 | packed / zero | — | execution/residual failure |
| 1e-06 | 1e-06 | folded / mean | — | execution/residual failure |
| 1e-07 | 1e-05 | packed / zero | 0.004690% | pass |
| 1e-07 | 1e-05 | folded / mean | 0.011462% | quality failure |
| 1e-07 | 3e-06 | packed / zero | — | execution/residual failure |
| 1e-07 | 3e-06 | folded / mean | 0.007827% | pass |
| 1e-07 | 1e-06 | packed / zero | — | execution/residual failure |
| 1e-07 | 1e-06 | folded / mean | — | execution/residual failure |

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
| Legacy pre-cast float64 RHS | `8c4dc738cac5acc53b5821dd55084d74ade34746a48660e0f4786f7e165844ff` |
| Serialized float32 RHS | `503a3ec816628540eab8ec3c1e5c4b424509bb7eb8f4db2646604b77498b7896` |
| Scalar prior coefficients | `48a654bf597d56d6044c368fffebb5165503845f1f5ba1da4032ef0602f6db0e` |

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
./reproduction/run_env.sh -m reproduction.audit_common_rhs_storage \
  --scratch /opt/geospsr-runtime/common-rhs-128 \
  --output /opt/geospsr-runtime/common-rhs-128/storage_audit.json
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \
  --samples 128 --batch 2 --repeats 1 --pack-rhs --preconditioner-rank 1024 \
  --finite-rows 32 --quality-holdout-start 8 \
  --scratch /opt/geospsr-runtime/common-rhs-128 \
  --reference reproduction/results/nufft_batches/nufft128 \
  --output /opt/geospsr-runtime/nystrom-validation/common128
./reproduction/run_env.sh -m reproduction.stream_applications \
  --samples 8 --checkpoints 8 --retained-control --preconditioner-rank 1024 \
  --output /opt/geospsr-runtime/nystrom-validation/coherent8
./reproduction/run_env.sh -m reproduction.check_preconditioned_precision \
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
