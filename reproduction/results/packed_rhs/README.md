# Paired real RHS: measured conditioning comparison

At one million observations, pairing scaled real RHS in full-cube complex
NUFFTs reduces the eligible eight-draw median from
**31.39 s to 19.60 s
(1.60× over our folded solver)**.
Every finite frequency, the noise model and the solver tolerance are retained.
This is an incremental conditioning result, not a complete-application,
upstream SDD or ContactFusion speedup.

## Repetitions and scope

| Method | Eligible / total | Median | Sampled process-memory peak |
| --- | ---: | ---: | ---: |
| Folded, batch 1 | 4/6 | 31.39 s | 2.88 GiB |
| Packed, batch 1 | 5/12 | 25.96 s | 3.47 GiB |
| Packed, batch 2 | 3/6 | 19.60 s | 4.86 GiB |

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
**0.000486%** and empirical covariance discrepancy is
**0.000591%**. Recorded ray/collision events
and selected view agree. The analytic residual holdout upper endpoint is
0.07896% against the 0.1% gate.
The raw solve takes 315.59 s and passes sampled isolation.
This single run is a numerical confirmation, not repeated 128-draw timing
evidence or a test of absolute Monte Carlo convergence.
[128-draw records](packed150-128/results.json).

`HalfCubePosterior.draw(..., pack_rhs=True)` keeps the coherent folded prior
and mean while applying the full-cube packed inverse. Small tests compare
independent dense finite-covariance solves, retained common RHS/prior arrays,
partial batches, centered narrower solves and cached query fields. An additional
million-point cutoff200 four-draw check differs from saved folded results by
0.030012% of the fluctuation norm and has maximum independently
summed covariance-row residual 0.000991%.
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
