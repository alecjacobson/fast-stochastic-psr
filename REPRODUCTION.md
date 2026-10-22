# Reproduction details and earlier experiments

This document retains the detailed methodology and historical measurements.
The [main README](README.md) is the current flagship status summary; the
[benchmark contract](reproduction/BENCHMARKS.md) defines acceptance criteria.

## Posterior covariance error from the omitted spectrum

The [covariance-tail report](reproduction/results/spectral_tail/README.md)
adds a million-point covariance diagnostic to the earlier analytic-residual
checks. A PSD bound isolates omitted vector-training modes while preserving
the cutoff-50 scalar/cross model. Dense A/B calculations validate the bound
and its infinite-tail frequency estimator. A 16-query million-point panel
supports small covariance error at default parameters; the short-scale,
low-noise B case remains a failure. Production/stricter solver comparisons
separate numerical sensitivity from spectral truncation.

The report includes commands, fixed query/frequency seeds, raw arrays, the
derivation and limitations. These are covariance diagnostics, not complete
application accuracy or speedup measurements. Simply adding a full prior to
a truncated inverse does not recover the analytic posterior.

## Batched conditional solves and optimized exact sampling

The [conditional-solve report](reproduction/results/conditional_solves/README.md)
compares shared-RHS dense, compiled SDD and cutoff-200 NUFFT conditioning. At B,
the dense solve wins; at one million points, eight conditional solves favor
NUFFT by 4.39× over shared-row SDD. This is not the 27.2× mean-fit ratio, and the
large-input pilot's common prior omits the analytic high-frequency tail.

`Posterior.draw(..., rng_backend='gpu', joint_solve=True)` enables GPU Gaussian
generation and a shared triangular solve. `Posterior(..., upsampfac=1.25)` reduces
source-transform workspace with validated numerical error. CPU RNG remains the
default for historical realizations. The separable finite-vector covariance
construction is now the default; `covariance_backend='nufft'` retains the earlier
construction. Independent moment tests and common-randomness checks cover the
new paths; CPU and GPU seeds do not produce the same draws.

```bash
./reproduction/run_env.sh reproduction/benchmark_sampler_setup.py
./reproduction/run_env.sh reproduction/benchmark_conditional_solves.py --dataset B --samples 128
./reproduction/run_env.sh reproduction/benchmark_conditional_solves.py --dataset 1000000 --samples 8 --chunk 512
./reproduction/run_env.sh reproduction/applications.py --cache-queries --fast-sampling --output reproduction/results/conditional_solves/applications-optimized
```

The full small-scorpion computation through hitbox export takes 11.69 s with the
earlier CPU pipeline and 1.77 s with fast preparation, excluding startup/plotting.
This is a same-method improvement under the corrected posterior law, not a
comparison against the untouched Holalkere stochastic API. The report contains
the paired GPU-control output check, timing scopes and remaining accuracy work.

## Converged iterative conditioning baseline

The [SDD comparison](reproduction/results/iterative_baseline/README.md) includes
parameter tuning, full analytic checks on A/B, independent holdout analytic rows
at 100k/1m observations, and separate first/warm fit timings. The compiled update
loop is verified against the untouched upstream solver, with its noise argument
mapped to the same variance. Existing GP `sdd_params` can enable it with
`compiled_loop=True`; noise in the local API remains a variance.

At one million points, both methods pass the sampled residual gate; first fits
take 205.7 s for compiled SDD and 7.58 s for cutoff-200 NUFFT. SDD is feasible at
this size and wins the warm A/B mean fits. This does not establish a complete
application speedup or full posterior equivalence. See the report for commands,
configuration failures, timing scope and remaining batched-sampling comparisons.

## Independent vector training bandwidth

The [training-bandwidth report](reproduction/results/training_bandwidth/README.md)
compares direct finite-covariance references and NUFFT-CG with full analytic
conditioning on matched A/B inputs, then tests distinct 100,000/million-point
clouds using independent sampled analytic rows. It includes a coherent sampler
whose vector prior covers the larger training spectrum while sharing scalar
low-frequency modes. No dense omitted-spectrum factor is needed.

```bash
./reproduction/run_env.sh reproduction/training_bandwidth.py --case A
./reproduction/run_env.sh reproduction/training_bandwidth.py --case B
./reproduction/run_env.sh reproduction/scale_training_bandwidth.py
./reproduction/run_env.sh reproduction/scale_spectral_sampling.py
./reproduction/run_env.sh reproduction/summarize_training_bandwidth.py
```

The report retains a short-scale/low-noise failure, distinguishes observed-normal
and conditional-sampling residuals, and documents the remaining application and
upstream-iterative comparisons. These are conditioning/sampling measurements,
not complete-application speedups or a proof of full-model equivalence at large N.

## Paired application benchmarks

```bash
./reproduction/run_env.sh reproduction/paired_applications.py --case A
./reproduction/run_env.sh reproduction/paired_applications.py --case B
```

These compare shared corrected posterior draws with a cached reference using
the released interpolation kernel. They are not the untouched stochastic API.
Use `--rebuild-plans` to reproduce the earlier per-batch plan construction;
use `--output PATH` to retain a separate result set. See the
[paired application report](reproduction/results/paired_applications/README.md)
for quality gates, preparation/query scope and known coverage gaps.

Cached interpolation-only queries retain every Fourier mode. The newer
[cached-query report](reproduction/results/cached_queries/README.md) measures
moving queries, uncertain collision probes, paired moments and reference-grid
convergence. Run `paired_applications.py --cached-grids --uncertain-collisions`
with a separate `--output` directory to reproduce those comparisons.

The reusable `Posterior.prepare_query_cache()` prepares mean/sample caches;
`mean()` and `samples()` then support arbitrary new query positions. Redrawing
samples invalidates their cache. Caches add memory and preparation costs and
are optional. To use them in the application demonstration without overwriting
earlier results:

```bash
./reproduction/run_env.sh reproduction/applications.py --cache-queries --output reproduction/results/cached_queries/applications-cached
```

GPU reproduction and acceleration experiments for **GeoSPSR**, with nonuniform
FFTs (NUFFT), matrix-free Gaussian-process solves, and an initial assessment of
fast multipole methods (FMM).

This is a research extension of [sholalkere/GeoSPSR](https://github.com/sholalkere/GeoSPSR),
the implementation of *Stochastic Poisson Surface Reconstruction with One Solve
using Geometric Gaussian Processes* by Sidhanth Holalkere, David S. Bindel,
Silvia Sellán, and Alexander Terenin (ICML 2025).
[Paper](https://arxiv.org/abs/2503.19136) · [Original project](https://geospsr.github.io/)

![A: original GeoSPSR and NUFFT on the same released 691-point scorpion scan](assets/scorpion-matched-A.png)

**(A) Reproduce the released scorpion first.** The exact camera setup, scan density,
noise seed, kernel parameters, and analytic training covariance come from
[the authors' example](https://github.com/sholalkere/GeoSPSR/blob/2352f1ef98aab20e0303611c1fa2b0948acd2a05/experiments/example.py).
Both methods use the **same 691 observations and the same two slices**. The
original fields are checked against the untouched released API. Our NUFFT
representation of its interpolated kernel agrees to **0.123% relative mean
error**, **0.0022% relative variance error**, and **0.048 percentage points RMS
inside-probability error**. Variances are deterministic, with no Monte Carlo noise.
The lower slice, `y = −0.75`, is supplied by the example; the upper slice,
`y = −0.35`, is inferred from Figure 1's two-lobed pattern, since its height is
not supplied. This is a reproduction of the released setup, not a claim that the
paper's exact Figure 1 observations or second slice were recovered.

![B: original GeoSPSR and NUFFT on an identical denser 17,419-point scorpion scan](assets/scorpion-matched-B.png)

**(B) Increase density; compare both methods again.** The same two cameras now
produce **17,419 observations—25× more points and 635× more covariance entries**.
Both methods still use the full analytic training covariance and deterministic
variance. On an **NVIDIA L40**, the sum of first-call reconstruction stages drops
from **28.10 s to 2.96 s (9.5×)**, including kernel setup, fitting, centering, mean,
and variance, excluding scan generation and rendering. With fitting and kernel
setup cached, **mean queries take 22.77 ms versus 392.05 ms (17.2×)**, including
NUFFT source aggregation, for both 100 × 100 slices. Mean and variance agree to
**0.0305%** and **0.0074%** relative L2; inside-probability error is **0.041 percentage
points RMS**, **1.665 points maximum**. Exact variance remains a bottleneck:
the stronger cached reference takes **1.17 s**, versus **2.27 s** for our NUFFT
path. The 17.2× ratio applies to the mean, not all warm queries together.
[Measurements A](reproduction/results/matched/A/comparison.json) ·
[Measurements B](reproduction/results/matched/B/comparison.json) ·
[Methods and error maps](reproduction/results/matched/README.md) ·
[PDF A](assets/scorpion-matched-A.pdf) · [PDF B](assets/scorpion-matched-B.pdf)

## What is here

- A headless GPU reproduction of the released scorpion example, retaining an
  untouched upstream checkout for comparisons.
- Joint posterior collision queries, ray transmittance, next-view scoring,
  sequential scanning, and uncertainty-aware hitbox meshes.
- cuFINUFFT source aggregation and query evaluation, including cached Fourier
  coefficients for means and coherent posterior samples.
- Matrix-free, Fourier-truncated covariance actions and batched conjugate-gradient
  solves, exercised up to **1,048,576 observations**.
- Corrections to covariance conditioning, noise handling, and example compatibility,
  covered by **ten numerical GPU tests**.
- Matched original/NUFFT scorpion figures with deterministic variance and an
  explicit Fourier representation of the released interpolation kernel.
- An exploratory million-observation finite-Fourier GP posterior with coherent
  conditional draws; this uses a different training model from the release.
- Far-field block-rank experiments to assess a future FMM implementation.

The paper's complete application scenes, seeds, and timing configurations are
not included in the upstream release. Our applications reconstruct its task
**types on the released scorpion**, rather than exactly reproducing the original
dragon/cat/falcon figures. The paper used an RTX A6000; these measurements use an
L40. The historical SPSR baseline has not been rerun.

## Measured results

Environment: NVIDIA L40 with 48 GB, driver 570.158.01, Python 3.11.15,
JAX 0.6.2, Flax 0.10.6, and cuFINUFFT 2.5.1. GPU operations are synchronized before
timing ends. Fast-sum benchmarks retain first-call timings and three warm
repetitions; quoted warm times are medians. GPU clocks were not locked.
See [environment metadata](reproduction/results/environment.json).

### Matched scorpion reproduction

The new figures match **observations, normals, query positions, noise convention,
training covariance, and kernel interpolation**. Input hashes are stored with
both runs. The original `compute_mean` and `compute_variance` functions are
imported from the untouched upstream checkout and run on every displayed query
in both cases. A reference implementation that reuses one Cholesky factor
agrees with those APIs to better than `5 × 10⁻⁷` relative error. This stronger
reference supplies the cached-query comparison, so repeated fitting is not
counted as a NUFFT advantage.

| Agreement with original code | A: 691 points | B: 17,419 points |
| --- | ---: | ---: |
| Mean relative L2 error | 0.123% | 0.0305% |
| Variance relative L2 error | 0.0022% | 0.0074% |
| Inside-probability RMS error, percentage points | 0.048 | 0.041 |
| Inside-probability maximum error, percentage points | 0.621 | 1.665 |
| Original cached mean | 13.43 ms | 392.05 ms |
| NUFFT mean, aggregation included | 20.77 ms | 22.77 ms |
| Original cached variance | 19.06 ms | 1.17 s |
| NUFFT deterministic variance | 983.50 ms | 2.27 s |

At the small size the cached reference is faster. At the larger size NUFFT
accelerates the mean, while cached reference algebra still wins for variance.
The full first-call workflow benefits from fast kernel setup and shared
factorization: **23.37 → 1.56 s** for A and **28.10 → 2.96 s** for B. These are sums
of synchronized first-call stages, including compilation where applicable,
not process-launch timings. Warm mean figures are medians of three runs;
variance figures are one repeated full pass after its first invocation.
Both paths stream cross-kernel/variance rows in batches of 1,024.

The released implementation interpolates the cross kernel on a 50³ nonuniform
grid. Evaluating its underlying Fourier series directly changes that numerical
kernel. For the matched figures, we analytically integrate the interpolant's
periodic piecewise-linear basis to obtain Fourier coefficients, truncate at
**201³ modes**, then use NUFFT. This is a separate cutoff from the original
**101³-mode Poisson kernel**. The remaining discrepancy is explicitly measured
above. See [the implementation](reproduction/interpolated_nufft.py).

The original example adds `sigma = 0.02` to the covariance diagonal; we preserve
that convention, along with coordinate and normal perturbations of standard
deviation 0.02. The statistical query is `Φ(−μ/√variance)`, using the released
example's mean-centering convention and outward normals. Variance uses
`prior − ||L⁻¹ k_vf||²`, with **no stochastic estimator or spatial smoothing**.
Color limits are shared between methods for each field/slice, and explicitly
labeled; probability always uses `[0, 1]`. Denser observations produce sharper
probabilities in **both** methods.

The previous [million-point image](assets/scorpion-million.png) changed the scan
sampling, slice height, training covariance, variance estimator, and color
normalization simultaneously. It did not establish equivalence to the released
teaser. Its [measurements](reproduction/results/teaser/million.json) remain as an
exploratory finite-GP experiment. The new B uses 17,419 points so the full
original model can actually be run and checked. A single dense float32 training
matrix at one million points would require 4 TB; matched full-model results at
that size remain future work.

### Kernel queries

| Experiment | Measurement |
| --- | ---: |
| Pairwise interpolation, N = 1,048,576, M = 4,096 | 7.838 s |
| NUFFT source aggregation, same N | 29.12 ms |
| NUFFT query, same M | 4.90 ms |
| Sum of aggregation and query medians | **34.02 ms** |
| Pairwise / NUFFT ratio | **230.4×** |
| Full-cutoff direct Fourier validation, relative L2 error | **2.8 × 10⁻⁶** |

The large-N benchmark uses synthetic sphere-supported sources and random vector
weights. Both query methods use the same finite Fourier kernel, but the baseline
approximates its values through a 50³ interpolation grid. Removing that
interpolation changes the large-N result by about 2.4% in relative L2; the much
smaller direct-Fourier validation error measures NUFFT accuracy separately.
Aggregation includes plan creation and source transfers; queries include plan
creation and query transfers. Training is excluded.

![Measured input and output scaling](reproduction/results/scaling.png)

### Matrix-free training

Three normal-vector right-hand sides, one million sphere points, length scale
0.01, noise variance 0.02, and relative CG tolerance 10⁻⁵:

| Fourier cutoff | Modes | Setup | Solve | Iterations | Analytic-kernel residual on 64 sampled rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 101³ | 0.028 s | **0.701 s** | 23 | 1.40% |
| 100 | 201³ | 0.194 s | **1.670 s** | 24 | 0.0883% |

CG solves the **truncated covariance**, not the full analytic Matérn covariance.
Its finite-system residual is below 10⁻⁵ in these runs. The last column measures
the separate truncation discrepancy; a 64-row spot check does not bound the full
solution error. These are single solve measurements after warming transform
plans, not multi-run median timings. The relatively large regularizer makes
these particular systems easy. The separate million-point finite-GP experiment supplies another
surface-supported case, but harder noise regimes remain untested.

### Upstream scorpion baseline

691 observed points, 10,000 slice queries, 101³ Fourier modes, and a 50³
nonuniform interpolation grid:

| Operation | Time |
| --- | ---: |
| Build interpolation grid, first call | 13.66 s |
| Mean query, repeated API invocation | 1.09 s |
| Marginal variance, first call | 26.53 s |
| Four posterior samples, first call | 31.05 s |

First-call measurements include compilation. Repeated upstream API calls can
also rebuild closures and recompile. These are not directly comparable to a
cached-query-only timing.

![Reconstructed scorpion applications: posterior fields, ray transmittance, next-view scores, and collision probabilities](reproduction/results/applications.png)

Raw measurements: [baseline](reproduction/results/upstream_baseline.json) ·
[fast sums](reproduction/results/benchmark.json) ·
[training](reproduction/results/training.json) ·
[applications](reproduction/results/applications.json) ·
[sequential scanning](reproduction/results/sequential_scan.json).
The [full HTML report](reproduction/results/report.html) includes additional
figures and caveats; open it locally after cloning, since GitHub displays HTML
as source.

## Reproduce

Use Linux, an NVIDIA CUDA-capable GPU, Git, and
[uv](https://github.com/astral-sh/uv). The measured configuration has 48 GB of GPU
memory; a smaller-memory configuration has not been validated. Choose a writable
runtime directory with at least roughly 7 GB free for dependencies and caches.
Use a full clone so the historical upstream commit is available.

```bash
git clone https://github.com/alecjacobson/fast-stochastic-psr.git
cd fast-stochastic-psr

# Override this if another filesystem has more free space.
export GEOSPSR_RUNTIME="$HOME/.cache/fast-stochastic-psr-runtime"
./reproduction/setup.sh
./reproduction/run_env.sh -m pytest -q tests

# Run GPU experiments sequentially to avoid timing interference.
# The baseline must run first: it generates the scan used by later scripts.
./reproduction/run_env.sh reproduction/baseline.py --upstream
./reproduction/run_env.sh reproduction/applications.py
./reproduction/run_env.sh reproduction/sequential_scan.py
./reproduction/run_env.sh reproduction/benchmark.py
./reproduction/run_env.sh reproduction/training.py
./reproduction/run_env.sh reproduction/fmm_assessment.py
./reproduction/run_env.sh reproduction/solver_audit.py
./reproduction/run_env.sh reproduction/capture_environment.py
./reproduction/run_env.sh reproduction/make_report.py
```

To regenerate the matched A/B figures, run the following jobs sequentially.
The reference commands execute the untouched released API and the stronger
cached-factorization baseline, and verify agreement. Allow a few minutes on the
measured GPU. Rendering uses a separate Blender environment (roughly 2 GB extra).

```bash
./reproduction/run_env.sh reproduction/matched_teaser.py --case A --backend original --literal-api
./reproduction/run_env.sh reproduction/matched_teaser.py --case A --backend compatible
./reproduction/run_env.sh reproduction/matched_teaser.py --case B --density 1000 --backend original --literal-api
./reproduction/run_env.sh reproduction/matched_teaser.py --case B --density 1000 --backend compatible
./reproduction/render_setup.sh
"$GEOSPSR_RUNTIME/render-venv/bin/python" reproduction/render_matched_cloud.py --case A
"$GEOSPSR_RUNTIME/render-venv/bin/python" reproduction/render_matched_cloud.py --case B
./reproduction/run_env.sh reproduction/compare_matched_teaser.py
./reproduction/run_env.sh reproduction/validate_matched_teaser.py
```

Inputs, fields, and cloud renders are checked in, so rerunning only
`compare_matched_teaser.py` regenerates the PNG/PDF/SVG figures without recomputing
statistics. The optional `--backend nufft` evaluates the original finite Fourier
kernel directly, bypassing its interpolant; those diagnostic fields are also
retained, but are not the matched-kernel panels.

The earlier million-point experiment is still reproducible with
`million_teaser.py`, `benchmark_million_teaser.py`, `render_million_cloud.py`, and
`compose_million_teaser.py`, in that order, using the respective compute/render
environments. It is separate from the matched original-model comparison.

Setup installs the [locked environment](reproduction/requirements-lock.txt) and
creates a detached worktree at upstream commit
[`2352f1ef98aab20e0303611c1fa2b0948acd2a05`](https://github.com/sholalkere/GeoSPSR/commit/2352f1ef98aab20e0303611c1fa2b0948acd2a05).
The lock is important: the unpinned latest Flax rejected the upstream kernel
container during our initial build.

The wrapper selects CUDA, disables JAX GPU preallocation, requests highest JAX
matrix-multiplication precision, and uses headless plotting. Keep
`GEOSPSR_RUNTIME` set for subsequent commands. On the original experiment machine,
the existing runtime is `/opt/geospsr-runtime`.

Results are written to `reproduction/results/`, including JSON timings, PNG
figures, OBJ hitboxes, and the HTML report. Generated NumPy archives are ignored
by Git and recreated by the scripts, except for the compact teaser fields and matched-comparison inputs/outputs.
Running the experiments overwrites the corresponding checked-in measurements,
so preserve a copy if you want to compare machines.

## How the fast sums work

For solved vector weights α at nonuniform source positions x, compute

```text
A(n) = Σ_j α_j exp(−i n·x_j)                  type-1 NUFFT
b(n) = −i ρ(n) [n·A(n)] / |n|²,  b(0) = 0   Poisson multiplier
f(q) = Re Σ_n b(n) exp(i n·q)                 type-2 NUFFT
```

Here ρ is the product-Matérn spectral density and n ranges over a finite integer
frequency grid. This evaluates the cross-covariance action without forming the
query-by-source matrix. Cached coefficients make additional queries independent
of the source count. Posterior samples add their prior coefficients before
querying.

The vector-field covariance has the finite representation
`K_L = Re Φ diag(ρ) Φ*`, permitting matrix-free CG with the same transforms.
**Spectral cutoff, NUFFT tolerance, and solve tolerance control different errors.**
Increasing the cutoff grows a three-dimensional grid; this motivates spatial
fast summation and hybrid approaches at shorter length scales.

## Statistical corrections and limitations

The upstream numerical implementation is preserved for baseline runs. The
working implementation makes these corrections:

1. Posterior covariance **subtracts** the conditioning update.
2. SDD and Cholesky consistently interpret `sigma` as **observation noise
   variance**, rather than disagreeing about whether to square it.
3. Pathwise conditioning draws observation noise. Pass `noise_key` independently
   of the key used for the prior samples.

The primary application sampler also accounts for the omitted vector-field
spectrum at observed points, using the full finite cross spectrum and a dense
residual covariance. Its empirical moments are tested. **This sampler still
uses dense training matrices**. The earlier million-point experiment instead uses the
explicitly finite-spectrum model throughout; its matrix-free conditional sampler
is implemented in `reproduction/million_teaser.py`.
The sequential-scanning experiment retains the paper-style smaller prior
spectrum and labels that approximation explicitly.

Other limits matter when interpreting the figures:

- Collision probabilities use discrete proxy-surface points, not certified
  continuous intersection tests. Primary application probabilities use 128
  coherent samples, giving a worst-case Monte Carlo standard error of about
  4.4 percentage points before model and discretization error.
- The most conservative hitbox reaches the query-domain boundary and has
  disconnected components; it is not a complete closed safety cage.
- Sequential scanning is one fixed-seed comparison. Its error metric is affected
  by meshing and unequal point counts; it does not establish consistent
  superiority over random views.
- The upstream SDD example does not converge sufficiently at its supplied
  learning rate in this scan. The [solver audit](reproduction/results/solver_audit.json)
  separates convergence from the noise-convention correction.
- The matched figures retain dense analytic training matrices. Scalable full-model
  conditioning and fast exact variance at million-point scale require further
  work. The earlier million-point experiment uses Monte Carlo variance and a
  truncated training covariance.

## FMM direction

[FMM assessment results](reproduction/results/fmm_assessment.json) measure
far-field block ranks at controlled error. They show compression potential, but
**there is no FMM backend or measured FMM speedup in this repository yet**.

Next experiments are a periodic kernel-independent FMM for the full analytic
covariance, spatial preconditioning, and controlled near-field/spectral
splitting. A standard Laplace or Coulomb FMM is not a drop-in replacement for
the product-Matérn kernel. Comparisons need matched posterior accuracy, periodic
boundary handling, setup/rebuild costs, and real surface-supported scans.

[Issue #18](https://github.com/alecjacobson/fast-stochastic-psr/issues/18) also
tracks exact sorted-range sums, motivated by
[Langrené, Warin and Gruet's product-Matérn decomposition](https://arxiv.org/abs/2508.01864).
A scalar periodic decomposition has been checked numerically; a stable GPU
implementation and measured conditioning/application gains remain unimplemented.

## Code map

| Path | Purpose |
| --- | --- |
| `geospsr/gp.py`, `kernels.py`, `sdd.py` | Upstream GP machinery with the documented corrections |
| `geospsr/spectral.py` | cuFINUFFT cross sums, prior coefficients, and matrix-free CG |
| `reproduction/baseline.py` | Headless upstream scorpion experiment |
| `reproduction/posterior.py` | Reusable dense-fit posterior with cached Fourier coefficients |
| `reproduction/applications.py`, `sequential_scan.py` | Application reconstructions |
| `reproduction/matched_teaser.py`, `interpolated_nufft.py` | Matched analytic-GP fields and Fourier representation of the released interpolant |
| `reproduction/compare_matched_teaser.py`, `validate_matched_teaser.py` | Paired figures, numerical agreement, and API checks |
| `reproduction/million_teaser.py`, `benchmark_million_teaser.py` | Exploratory finite-GP million-point experiment |
| `reproduction/render_million_cloud.py`, `compose_million_teaser.py` | Reproducible Figure-1-style rendering |
| `reproduction/benchmark.py`, `training.py` | Query and solve scaling experiments |
| `reproduction/fmm_assessment.py` | Far-field compression diagnostics |
| `tests/` | Covariance, sampling, solver, and NUFFT numerical checks |

## Attribution and license

The original GeoSPSR source, assets, and Git history are retained. The inherited
[MIT license](LICENSE) and copyright notice remain in place. Please credit the
original authors and paper when using the underlying method:

```bibtex
@misc{geospsr,
  title={Stochastic Poisson Surface Reconstruction with One Solve using Geometric Gaussian Processes},
  author={Sidhanth Holalkere and David S. Bindel and Silvia Sellán and Alexander Terenin},
  year={2025},
  eprint={2503.19136},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2503.19136}
}
```
