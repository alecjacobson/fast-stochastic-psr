# Compact coarse-basis storage

**At one million observations, compact storage lowers retained preconditioner memory from 3.815 to 1.907 GiB**, with unchanged numerical/event gates passing. Recomputed features plus compact storage lower the measured CuPy setup peak from 12.04 to 2.81 GiB. This is a storage result; it does not establish a conditioning or application speedup.

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
| baseline32 | 3.815 | 12.042 | 6.055 | 8.799 | 22.279 | 31.077 | True |
| recompute32 | 3.815 | 4.717 | 6.055 | 11.878 | 22.346 | 34.224 | True |
| blocked32 | 3.815 | 4.717 | 6.055 | 11.886 | 31.155 | 43.040 | False |
| compact16 | 1.907 | 2.810 | 4.148 | 11.891 | 27.274 | 39.165 | False |

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

**No timing ratio is claimed:** at least one declared whole-run isolation or quality gate fails. Timings are diagnostics; no favorable repetition is selected.

These totals cover preconditioner setup and conditioning eight pre-generated
RHS; source loading, prior generation, fitting, plans, scalar aggregation and
queries are separate. They are not preparation-through-application timings.
Defaults remain unchanged; compact storage is opt-in.

## Matching-quality checks

| Configuration | Iterations/batch | Field error / fluctuations | Covariance discrepancy | Independent finite residual | Numerical/event gates |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline32 | 38–38 | — | — | 0.001583% | finite residual only |
| recompute32 | 38–38 | 0.000862% | 0.000702% | 0.001218% | True |
| blocked32 | 38–38 | 0.001422% | 0.000943% | 0.001173% | True |
| compact16 | 38–38 | 0.001763% | 0.000856% | 0.001249% | True |

Candidates use the fresh baseline32 fields and the exact same validated
source/RHS/prior/geometry. Field/fluctuation tolerance is .05%; variance and
covariance tolerances are .01%; the paired ray/collision/next-view gates are
unchanged. Independent double cosine sums check 32 rows at a .002% residual
gate. These numerical controls do not certify the full analytic posterior,
absolute Monte Carlo accuracy or continuous-geometry convergence.


All 128 common draws: numerical gates **True**. On the 120 pilot-held-out draws, field/fluctuation error is 0.003348% and empirical covariance discrepancy 0.002236%. Maximum ray/collision differences are 0.0000/0.8333pp; selected view 14 → 14. Independent finite residual is 0.001908%. Recorded setup/conditioning cost is 11.876/447.215s, retained as a diagnostic without a ratio. The reference is the archived same-model nufft128, not a contemporary timing baseline. Geometry/observations are reused; only the additional draws are held out. [Raw confirmation](compact128/results.json).


Fresh eight-draw coherent streaming control: numerical gates **True**, prior/RHS bitwise equality **True**. Empirical covariance discrepancy 0.002188%; maximum ray/collision differences 0.0000/0.0000pp. Retained unpreconditioned samples and streamed compact-preconditioned samples share seed 51 and the fitted mean. The validation control deliberately retains reference RHS/priors. [Raw control](coherent8/results.json).


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
early stream close and low-noise coherent samples. All **116 tests pass** ([log](tests.log)).

The retained basis still grows as N × rank. At ten million points, float16
storage alone is about 19.07 GiB; construction and transforms need more. This
is a memory reduction, not an N-independent representation or evidence that
upstream SDD is intractable. [Issue #31](https://github.com/alecjacobson/fast-stochastic-psr/issues/31)
remains open for further storage/application tradeoffs and larger-N evidence.
[Issue #33](https://github.com/alecjacobson/fast-stochastic-psr/issues/33) tracks
fusing decoding with projection/correction to avoid the current float32
temporary writes and repeated conversion launches. No fused backend is
implemented or timed by this report. The later [fused study](../nystrom_fused/README.md)
implements it and reports component and full-conditioning costs separately.

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
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \
  --samples 8 --batch 2 --repeats 3 --pack-rhs --finite-rows 32 \
  --preconditioner-rank 1024 --profile-pool \
  --scratch /opt/geospsr-runtime/common-rhs-128 \
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
