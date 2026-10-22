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
| **Collision, rays, next-view queries** | Paired A/B evaluations with 128 shared coherent draws; local probability and uncertain-collision checks expose remaining gaps | [Initial benchmark report](reproduction/results/paired_applications/README.md). No validated complete-application speedup yet: [#3](https://github.com/alecjacobson/fast-stochastic-psr/issues/3), [#10](https://github.com/alecjacobson/fast-stochastic-psr/issues/10). |
| **Uncertainty-aware hitboxes** | Demonstration implemented; variance and meshing accuracy need controlled comparison | Variance acceleration open: [#4](https://github.com/alecjacobson/fast-stochastic-psr/issues/4). |
| **Sequential scanning** | Single-seed demonstration; multi-seed quality comparison pending | Training/update costs and decision stability open: [#7](https://github.com/alecjacobson/fast-stochastic-psr/issues/7). |
| **Million-point applications** | Exploratory finite-Fourier GP only; full-model equivalence unverified | Full-model conditioning [#6](https://github.com/alecjacobson/fast-stochastic-psr/issues/6) and massive-scale validation [#8](https://github.com/alecjacobson/fast-stochastic-psr/issues/8) open. |

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

Dense analytic conditioning and exact variance remain bottlenecks. Matrix-free
Fourier experiments reach a million observations, but truncate the training
covariance. FMM currently has compression diagnostics only, with no implemented
backend or measured FMM speedup. A dense matrix's memory requirement alone does
not establish that the authors' iterative method is intractable.

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
