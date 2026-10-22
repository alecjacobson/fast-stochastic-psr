# Fast Stochastic Poisson Surface Reconstruction

GPU acceleration of [GeoSPSR](https://github.com/sholalkere/GeoSPSR), the implementation
of **Stochastic Poisson Surface Reconstruction with One Solve using Geometric
Gaussian Processes**, Holalkere, Bindel, Sellán and Terenin, ICML 2025.
[Paper](https://arxiv.org/abs/2503.19136) · [Project](https://geospsr.github.io/)

**Goal:** accelerate every paper application at matching quality, then demonstrate
those gains on massive scans beyond practical upstream execution.
[Tracking issue #1](https://github.com/alecjacobson/fast-stochastic-psr/issues/1) ·
[Benchmark protocol and coverage](reproduction/BENCHMARKS.md) ·
[Reproduction instructions and technical details](REPRODUCTION.md)

## Flagship comparisons

All times below were measured on an NVIDIA L40 (48 GB). Speedups have explicit
scopes; kernel-action timings are not complete-application speedups.

| Comparison | Quality evidence | Performance / status |
| --- | --- | --- |
| **A: released scorpion, 691 observations** | Same data and analytic training covariance; original APIs checked; mean error 0.123%, variance error 0.0022% | Reproduced. Cached original is faster at this size. |
| **B: denser scorpion, 17,419 observations** | Mean error 0.0305%, variance error 0.0074%; inside-probability RMS error 0.041 percentage points | Cached mean **17.2× faster**; deterministic variance **slower**. First computation stages **9.5× faster**. |
| **Collision, rays, next-view queries** | Paired coherent samples, uncertain collision probes, moving queries and two seeds; numerical gates pass | At B: prepared ray queries **11.1–11.2× faster**, collisions **10.5× faster**. Setup excluded; [report and remaining validation](reproduction/results/cached_queries/README.md). |
| **Full application demo (A)** | Same corrected posterior law; paired GPU control preserves events and all three hitboxes' connectivity | Fit through hitbox export **11.69 → 1.77 s (6.62×)** versus our earlier pipeline. [Report](reproduction/results/conditional_solves/README.md). Geometry/variance convergence remains open: [#4](https://github.com/alecjacobson/fast-stochastic-psr/issues/4). |
| **Sequential scanning** | Single-seed demonstration; multi-seed quality comparison pending | Training/update costs and decision stability open: [#7](https://github.com/alecjacobson/fast-stochastic-psr/issues/7). |
| **ContactFusion robotics replay** | [Assessment and reproduction plan](reproduction/CONTACTFUSION.md); author project page marks code coming soon | No measured speedup. Sequential prior, modality noise and action-quality validation tracked in [#23](https://github.com/alecjacobson/fast-stochastic-psr/issues/23). |
| **Million-point mean fitting** | Independent analytic residuals; [16-probe covariance-tail diagnostic](reproduction/results/spectral_tail/README.md) provides separate limited covariance evidence | [First fit **27.2× faster**](reproduction/results/iterative_baseline/README.md). Single confirmation; full application quality remains open. |
| **Million-point posterior conditioning** | Folded/full-cube comparison at 128 shared draws: covariance error 0.000596%, identical sampled events and selected view | [Folded conditioning **1.22× faster than our cube150 solver**](reproduction/results/half_cube/README.md): 38.16 → 31.38 s for eight draws, using eligible repetitions. 128-draw timing excluded for contention; full-model/application convergence remains open. |

Application query ratios use 128 shared **corrected** posterior draws and a strong
reference that also caches prior evaluation. Those runs used the earlier CPU
sampler (~13 s shared preparation). Optional GPU preparation now makes 128
corrected B draws **6.95× faster** within this implementation ([report](reproduction/results/conditional_solves/README.md)).
Additional compatibility-cache setup amortizes after about 17 ray batches in B.
These are repeated-query gains, not complete reconstruction or scanning speedups.
Absolute sampling accuracy and geometric discretization still need convergence studies.

### A — reproduce the released example

![Released scorpion: original GeoSPSR and matched NUFFT](assets/scorpion-matched-A.png)

Both methods receive the **same 691 observations, normals and slice queries**,
using the released camera setup, noise seed, full analytic covariance and
cross-kernel interpolation. Variances are deterministic. Mean/variance errors
are relative L2; inside-probability RMS error is **0.048 percentage points**.
The lower slice `y = −0.75` is released; the upper `y = −0.35` is visually inferred.
The exact Figure 1 point set and second slice were not published.

### B — increase density and compare again

![Denser scorpion: original GeoSPSR and matched NUFFT](assets/scorpion-matched-B.png)

At **17,419 observations and 20,000 queries**, cached mean evaluation falls from
**392.05 ms to 22.77 ms (17.2×)**, including NUFFT source aggregation. Deterministic
variance takes **1.17 s for the cached reference and 2.27 s for NUFFT**, so the
combined warm mean/variance workload is currently slower. The sum of first
computation stages (kernel setup, fitting, centering, mean and variance) falls
from **28.10 s to 2.96 s (9.5×)**; this excludes process startup, scan generation,
rendering and probability postprocessing. Probability error is **0.041 percentage
points RMS**, **1.665 points maximum**.

[Methods, timing scope and error maps](reproduction/results/matched/README.md) ·
[Measurements A](reproduction/results/matched/A/comparison.json) ·
[Measurements B](reproduction/results/matched/B/comparison.json)

## Approach and current limits

NUFFT aggregates observations into Fourier coefficients and evaluates means or
coherent posterior samples at arbitrary queries. We retain a compatibility mode
for the released interpolated kernel and investigate direct Fourier evaluation
at controlled application error ([#5](https://github.com/alecjacobson/fast-stochastic-psr/issues/5)).

Exact variance remains a bottleneck. Matrix-free conditioning now reaches a
million observations with independently checked training-bandwidth convergence,
but the full application quality bridge remains open. The authors' iterative
method also fits this million-point input; this scale is **not intractable for SDD**.
Compiled SDD wins the warm A/B fits in this comparison, motivating
[method selection by workload (#15)](https://github.com/alecjacobson/fast-stochastic-psr/issues/15).
The [training/sampling audit](reproduction/results/training_bandwidth/README.md)
includes coherent million-point draws and retained failures. Short-scale, low-noise
cases fail the earlier cube approximation. A new
[spectral-block GPU implementation](reproduction/results/spectral_blocks/README.md)
passes the fixed 16-query stress panel with a wider spectrum.
[Omitted-spectrum diagnostics](reproduction/results/block_tail/README.md) now
cover corners and the infinite tail, including a million-point panel; full
application validation remains open. A centered block-inverse pilot passed
quality checks but was slower than the existing cube solve.
[Local/FMM correction (#14)](https://github.com/alecjacobson/fast-stochastic-psr/issues/14)
remains another avenue.
FMM has no implemented backend or measured speedup. A dense matrix's memory
requirement alone does not establish that the authors' iterative method is intractable.

[Length-scale validation](reproduction/results/normalization/README.md) corrects
an inherited Fourier normalization mismatch and preserves an explicit released
coefficient mode. The .01/.005 flagship settings are unchanged; at length scale
1 the correction changes slice probabilities by up to 6.02 percentage points.

The full paper application scenes and timing configurations were not released.
Our application scenarios use the released scorpion and do not claim to recreate
the unpublished dragon/cat/falcon scenes. Earlier kernel-only benchmarks and the
exploratory million-point teaser remain documented in [REPRODUCTION.md](REPRODUCTION.md).

## Run

```bash
git clone https://github.com/alecjacobson/fast-stochastic-psr.git
cd fast-stochastic-psr
export GEOSPSR_RUNTIME="$HOME/.cache/fast-stochastic-psr-runtime"
./reproduction/setup.sh
./reproduction/run_env.sh -m pytest -q tests
```

See [REPRODUCTION.md](REPRODUCTION.md#reproduce) for GPU requirements, pinned
upstream checkout, commands for every experiment, rendering and raw outputs.

## Attribution

Original source, assets and history are retained under the [MIT license](LICENSE).
Please credit Holalkere et al. and their [paper](https://arxiv.org/abs/2503.19136)
when using the underlying method. [BibTeX](REPRODUCTION.md#attribution-and-license).
