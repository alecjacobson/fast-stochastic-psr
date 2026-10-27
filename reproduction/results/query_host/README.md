# Host-backed covariance correction on application panels

`HostQueryResidualAudit` moves the residual-correction vectors to host memory
and computes their products in GPU row chunks. It preserves the same joint
query covariance and residual Gram bound. It is an exact change of storage
and accumulation order, **not** a low-rank approximation or independent sampling
of query batches.

Seventeen targeted tests pass, including GPU/host products with partial query
blocks and row chunks, duplicate queries, tiny/zero components, and explicit
allocation-cap/incomplete-capture failures. Existing strict posterior tests
remain unchanged.

## Memory scope

The host arrays have shape `(3, queries, observations)` and retain all double
weights and analytic residuals. Their size is `48*N*M` bytes, explicitly capped
at 128 GiB by default. GPU operands for product assembly need at most
`16*min(N,row_chunk)*M` bytes; the experiments use `row_chunk=4096`.
This bound excludes the query matrices, GEMM workspace, capture blocks, and
the caller's analytic/NUFFT plans. Temporary NumPy row copies and pageable
host/device transfers are included in the measured preparation.

For the 17,419-observation, 1,024-query B panel, retained vectors occupy
856.16 MB on the host and the two GPU product operands occupy at most
67.11 MB. At a million observations the same panel needs **49.152 GB host
memory**, not 49.152 GB of GPU vector storage. The full million-point panel
has not yet been run. This approach bounds GPU workspace; it does not bound
host storage independently of N and M, and CuPy's pool may retain freed GPU
capacity. Peak process measurements are sampled separately.

## Application control

The panel is exactly the previous B application panel: rays 0, 8 and 17 with
256 points each, and 256 samples on sphere 8. All 1,024 points are in one
coherent Gaussian draw. The covariance target uses full analytic training
and scalar/cross cutoff 50. The mean is solved to analytic tolerance 1e-10.
Query solves use a finite spectrum, one analytic defect-correction step,
and the quadratic covariance correction. No dense training matrix is formed
by production; the reference covariance comes from the retained independent
dense Cholesky study.

At finite training cutoff 100, host-backed full preparation takes **222.89 s**,
including setup, mean, all query solves, host transfers/products, covariance
budget and root. The product/budget stage itself costs **0.411 s**. Covariance
relative L2 error is **2.27e-12**, maximum relative marginal-variance error
**2.56e-11**, and mean error divided by prior standard deviation **7.94e-10**.
The original 1e-7 output gates all pass; this is not a comparison against the
looser 0.01% solve-only budget of the small-panel study.

Across **32,768 paired draws**, there are **zero ray or collision event
disagreements** against the dense reference. Full-field relative L2 difference
is 2.15e-9. Sphere-8 collision probability is 0.53262329; the finite-grid
95% simultaneous Monte Carlo radius is 1.256 percentage points across 769
events. Propagated score intervals separate ray 0 as the best of the three
tested views. These sampling bounds exclude numerical/model bias and spatial
discretization. They do not establish continuous collision accuracy or full
sequential scanning quality.

The relative covariance budget is explicitly **noninformative** for this
nearly singular panel. Absolute/marginal residual budgets and independent
dense controls remain available. Roundoff-only PSD projection removes a
largest negative eigenvalue of 5.09e-18, below the fixed 8.52e-15 allowance;
no jitter or enlarged projection threshold is used.

An otherwise identical GPU-resident run takes **222.24 s**. The two output
covariances differ by 1.44e-16 relative L2, and both preserve every paired
event. Sampled GPU process peaks are **7.27 GB host-backed / 8.07 GB GPU-resident**;
host process peak RSS is **1.68 / 0.88 GB** respectively. Thus the storage
change moves memory to the host with little timing effect at B. The tiny
residual Gram matrices from the separate conditioning runs differ by 1.82e-5
relative (maximum absolute difference 1.54e-21); that comparison includes
conditioning roundoff, not just product accumulation order.

## Conditioning effort selected from marginal budgets

Cutoff 50 with one fixed defect step takes 160.85 s and preserves all paired
events, but **fails** the unchanged marginal-variance gate: maximum relative
error is 2.43e-7, above 1e-7. The failure is retained. Its solve-error bounds
identify 22 queries in six four-query blocks that need more conditioning.

`DiagonalBudgetSolve` uses the current full analytic residuals during each
solve. For query RHS b, weights w, residual r and scalar prior variance v,
the corrected marginal is `s=v-sum_d(w_d.T b_d+w_d.T r_d)` and its solve-error
upper bound is `u=sum_d||r_d||²/noise_min`. If `s>u`, `u/(s-u)` bounds the
relative marginal solve error in exact arithmetic. Whole four-query blocks
are refined until this ratio meets 1e-7, with an explicit correction limit.
The dense reference and the diagnostic failing-query IDs are not inputs.

The adaptive cutoff-50 run takes **177.37 s**. Of 256 blocks, 250 use one
defect step and six use two. Its maximum stopping bound is 9.94e-8. Independent
dense controls find covariance L2 error **9.79e-9** and maximum marginal error
**8.03e-8**, passing the original 1e-7 gates. All **32,768 paired ray/collision
draws again have zero event disagreements**; full-field difference is 3.89e-7.
Same-vector host/GPU products agree to 1.30e-15 relative or better.

This is **1.26× faster than cutoff-100 host preparation**, and **1.50× faster
than our earlier strict matrix-free query pipeline** (266.23 s), at the same
numerical output gates. The strong dense query reference still wins at B.
Only marginal solve error controls the local stopping decision; the full
covariance and events are checked separately. After assembly, the PSD bound
also gives `||error||_F <= trace(U)`: dividing by
`||S_corrected||_F-trace(U)` yields a solve-only relative Frobenius bound of
3.95e-8. These bounds exclude numerical/model error and are not interval
certificates. The global relative operator budget remains noninformative
near singularity. Adaptive stopping has not yet been tested at massive N.

## Million-point storage overlap

The host variant also completes the previous **32-query million-point control**
at cutoff 100, without a defect step for the query solves. Setup through root
takes **167.28 s**, versus 166.28 s in the earlier GPU-resident pilot. Covariance
difference is 3.19e-11 relative L2, with maximum marginal difference 1.71e-10.
This comparison uses the earlier approximate covariance as a storage control,
not an independent exact million-point posterior.

To isolate storage arithmetic, the new run compares chunked products against
unchunked GPU products of the **same retained vectors**. Shift and residual-Gram
relative errors are 7.04e-16 and 6.65e-16; this validation costs 0.295 s outside
preparation. The host vectors occupy 1.536 GB and the two GPU chunk operands
at most 2.10 MB. Sampled whole-process GPU peak is 35.07 GB, versus 34.56 GB
in the earlier run: no reduction in the overall peak is established at this
small panel, where mean fitting/planning also contributes. Host RSS peaks at
3.59 GB. A million-point application panel remains unmeasured.

Cached dense query covariance remains faster at B (the previous preparation
was 5.407 s). This prototype enables larger panels in GPU memory; it has not
established an upstream or complete-application speedup. The timings are
single-pass pilots, excluding process startup, input loading, independent
validation, artifact writes and final probability/score summaries.

## Next steps and reproduction

[#50](https://github.com/alecjacobson/fast-stochastic-psr/issues/50) tracks panel
storage and massive applications. [#51](https://github.com/alecjacobson/fast-stochastic-psr/issues/51)
tracks reducing solve count through a prior query basis with a separate
residual/truncation budget. For this panel, the omitted prior Frobenius norm
divided by the dense posterior norm is 1.20e-7 after 448 directions and
6.12e-10 after 512. These are overlap diagnostics, not a production rank rule
or a measured acceleration.

```
./reproduction/run_env.sh -m pytest -q tests/test_query_error.py tests/test_query_posterior.py tests/test_analytic_refinement.py
./reproduction/run_env.sh -m reproduction.benchmark_query_host --case B --panel applications --storage host --steps 1
./reproduction/run_env.sh -m reproduction.benchmark_query_host --case B --panel applications --storage gpu --steps 1
./reproduction/run_env.sh -m reproduction.benchmark_query_host --case B --panel applications --storage host --steps 1 --cutoff 50
./reproduction/run_env.sh -m reproduction.benchmark_query_host --case B --panel applications --storage host --steps 1 --cutoff 50 --adaptive-marginal
./reproduction/run_env.sh -m reproduction.benchmark_query_host --case million --panel moments --storage host --steps 0
```

Benchmarks refuse to overwrite existing results. Preserve those directories
elsewhere before regenerating. Run GPU commands serially. CPU replay uses
`python -m reproduction.validate_query_host DIRECTORY...`; it verifies retained
inputs, covariance and all event arrays, not every original analytic action
or all full field values. Source hashes for the first cutoff-100 B runs refer
to commit `158346f`; the fixed cutoff-50 run uses `e99d80c`, and the million
overlap uses `10bed28`. The adaptive source is included in this publication.
This history is preserved rather than rewriting earlier measurement records.

The B controls and event archives are public release assets to avoid adding
hundreds of megabytes to every Git clone. Fetch and verify them with:

```
./reproduction/run_env.sh -m reproduction.fetch_query_host
./reproduction/run_env.sh -m reproduction.validate_query_host reproduction/results/query_host/B_applications_host_s1 reproduction/results/query_host/B_applications_gpu_s1 reproduction/results/query_host/B_applications_host_s1_c50 reproduction/results/query_host/B_applications_host_s1_c50_adaptive reproduction/results/query_host/million_moments_host_s0
```

[Release manifest](release_manifest.json) records exact file hashes and URLs.
The small million-point controls remain in Git. All five artifact replays
pass; the fixed-step cutoff-50 quality failure remains explicitly false.
