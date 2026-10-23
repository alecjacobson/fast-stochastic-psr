# Fused compact-basis application and full conditioning

Fusing decode with multiplication removes the float32 basis temporaries while
preserving the stored half basis, solve precision and full covariance operator.
The [protocol](PROTOCOL.md) separates component timing from full conditioning
and includes the stronger float32 monolithic baseline.

## What carries through to conditional solves

All four whole-run sampled-isolation and numerical checks pass. With the basis prepared, fused conditioning is **1.27× faster than decoded half storage**, **1.04× faster than the stronger float32 monolithic preconditioner**, and **1.42× faster than plain packed CG**. These are within-implementation conditional-solve ratios. Process sampling can miss brief overlaps; the full-solve pilot uses one fresh process per configuration with three solve repetitions, not three independent process repetitions.

NVIDIA L40, one million observations, eight common draws, packed batch 2,
full cube 200/scalar 50, noise .02, CG tolerance 1e-5 and NUFFT epsilon 1e-6.
Every preconditioner uses rank 1,024 and landmark seed 41. Decoded/fused half
storage uses two-pass feature construction; the float32 baseline retains the
faster full double feature cache. Thus storage, setup and repeated-solve costs
are explicit alternatives, not interchangeable timing scopes.

| Configuration | Retained preconditioner (GiB) | Setup (s) | Median solves (s) | Setup + median (s) | Setup + first solve (s) | Iterations/batch |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| plain8 | 0.000 | 0.000 | 30.536 | 30.536 | 30.554 | 57–57 |
| monolithic32 | 3.815 | 8.795 | 22.260 | 31.055 | 31.098 | 38–38 |
| decoded16 | 1.907 | 11.877 | 27.218 | 39.096 | 39.129 | 38–38 |
| fused16 | 1.907 | 11.889 | 21.437 | 33.326 | 33.350 | 38–38 |

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
| plain8 | — | — | 0.001070% | finite residual only |
| monolithic32 | 0.003241% | 0.001819% | 0.001235% | True |
| decoded16 | 0.003422% | 0.002280% | 0.001557% | True |
| fused16 | 0.003181% | 0.002093% | 0.001222% | True |

The source arrays and geometry are identical and validated. Same-model gates
remain .05% field/fluctuation error, .01% empirical variance/covariance error,
the current paired ray/collision/next-view gates and .002% independent finite
residual on 32 rows. Saved fields/events are rechecked when generating this
report. Numerical agreement does not certify absolute sampling accuracy,
continuous geometry or the omitted analytic training spectrum.


All 128 common draws: numerical gates **True**. On the 120 pilot-held-out draws, field/fluctuation error is 0.003346% and empirical covariance discrepancy 0.002192%. Maximum ray/collision differences are 0.0000/0.8333pp; selected view 14 → 14. Independent finite residual is 0.001738%. Recorded setup/conditioning times 11.873/346.938s are diagnostics, not a ratio against the archived reference. [Raw confirmation](fused128/results.json). Geometry is reused; only additional draws are held out.


Eight-draw coherent streaming control: numerical gates **True**, bitwise prior/RHS equality **True**, covariance discrepancy 0.002205%, maximum ray/collision difference 0.0000/0.0000pp. The unpreconditioned retained control shares the fitted mean, seed 51 and batch schedule. [Raw control](coherent8/results.json).


## Component timing at one and three million observations

Each size has three fresh-process profiles, each with seven warmed repetitions
per method in alternating order. Both methods apply the same stored half basis
and damping to six RHS. The RHS include mixed signs/scales, an exact zero
channel and a coarse-space direction that exposes cancellation. A blocked
float64 evaluation provides the arithmetic reference; error/input norm must be
below 5e-6 in every channel, with exact zeros preserved.

| Observations | Decoded application (ms) | Fused application (ms) | Eligible action ratio | Incremental pool peak, decoded / fused (MiB) | Arithmetic gates |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 million | 42.339 | 5.206 | 8.13× | 278.91 / 45.81 | True |
| 3 million | 127.546 | 15.229 | 8.38× | 324.69 / 92.69 | True |

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
compilation. A separate empty-cache 257-row/17-column/six-RHS diagnostic measures compile plus first fused use: 0.374s (float32), 0.338s (float64). It excludes process/CUDA initialization and is not a throughput comparison. [Raw startup record](cold-start.json).

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
streaming. All **128 tests pass** ([log](tests.log)).

## Reproduction

```bash
./reproduction/run_env.sh -m reproduction.benchmark_compact_apply \
  --points /opt/geospsr-runtime/common-rhs-128/points.npy \
  --output /opt/geospsr-runtime/nystrom-fused/apply1m-1
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \
  --samples 8 --batch 2 --repeats 3 --pack-rhs --finite-rows 32 --profile-pool \
  --scratch /opt/geospsr-runtime/common-rhs-128 \
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
