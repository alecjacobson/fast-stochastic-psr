# Full analytic periodic range sums: correct first prototype, no speedup yet

The first 3D GPU range-sum implementation matches the **untruncated analytic
periodic product-Matérn covariance** on the released A691 and denser B17419
scorpion inputs. The worst covariance-only relative action error across three
signed RHS and four executions is **6.94e-15**. This removes Fourier truncation
from the matrix action; it does not yet accelerate conditioning or applications.

On the NVIDIA L40, this serial-sweep implementation is slower than the cached
analytic dense action on both inputs. B retains **7.90 MB** of device geometry
versus **2.43 GB** for a dense double matrix, but takes **90.17 ms** per action
versus **16.41 ms** for dense. These are retained object sizes, not process peak
memory; the range planner also allocates temporary CPU lists.

| Input / method | Setup (s) | First action (s) | Median of next 3 actions (ms) | Full analytic quality gate |
| --- | ---: | ---: | ---: | --- |
| A691 dense double | 0.02044 | 0.04434 | 0.0700 | Independent reference |
| A691 range double | 0.01389 | 0.11718 | 2.3226 | Pass |
| A691 packed cube200 NUFFT | 2.11511 | 0.05793 | 51.7750 | Fail |
| B17419 dense double | 0.21740 | 0.05384 | 16.4090 | Independent reference |
| B17419 range double | 0.92799 | 0.19362 | 90.1702 | Pass |
| B17419 packed cube200 NUFFT | 2.24538 | 0.05908 | 54.6333 | Fail |

Setup includes each method's geometry/kernel construction and transfer. Shared
input transfer is recorded separately. First actions include lazy initialization
and compilation where triggered, under the existing disk cache; these are not
fresh-install startup measurements. The range class compiles lazy raw kernels;
CuPy's cache state can differ between runs. No first-action result is discarded.
All arrays are resident for subsequent actions, which include output allocation
and the range input-validation synchronization. Each method acts on the same
three deterministic real RHS, seed 4501, using length scale .01, per-axis variance
.1 and noise variance .02. Dense comparison materializes the full GPU matrix,
then multiplies it; its construction cost is not charged to each reuse.

The predeclared 1e-7 relative L2 gate applies separately to every RHS, both with
and without the diagonal noise contribution. Cube200 errors in the covariance
alone are up to **0.3042% (A)** and **0.1293% (B)**; errors with noise included
are up to 0.01557% and 0.02156%. This strict full-kernel gate is distinct from
existing application-specific finite-model gates. These failures do not invalidate
those separate studies, and the NUFFT timings are **not a matching-quality
baseline here**. No range-versus-NUFFT speedup ratio is claimed.

Both runs pass sampled process-isolation checks. Samples cannot rule out
short-lived interference. Raw first and subsequent timings, all quality checks,
source hashes, input hashes, plan counts and audit observations are retained in
[A/results.json](A/results.json) and [B/results.json](B/results.json).
The saved `actions.npz` files hold the reference, weights and last output from
each method. [PROTOCOL.md](PROTOCOL.md) was declared before implementation and
measurements. There were no failed numerical tests or reruns to select timings.

## Derivation and numerical stability

This is our periodic GPU adaptation motivated by
[Langrené, Warin and Gruet's exact product-Matérn decomposition](https://arxiv.org/abs/2508.01864).
Their weighted-CDF method motivates exact fast sums; these centered periodic
features and GPU implementation are ours, not a reproduction of their code.

Normalize coordinates modulo 2π into [0,1). Set a = √3/ℓ, r = exp(−a),
u = 1−r (computed with `expm1`), c₀ = 2ℓ + 2√3 r/u, c₁ = 2√3,
and H = c₀ + (c₀+c₁)r. For t = |q−x|, the unit-variance circle kernel is

```text
k(t) = [(c₀+c₁t) exp(−at) + (c₀+c₁(1−t)) exp(−a(1−t))] / H.
```

A dyadic partition first separates x ranks. Each cross-block is partitioned by
y ranks. Off-diagonal y blocks are ordered in both x and y; mixed diagonal
blocks are split further, stopping at 32 entries. Direct leaf blocks handle
all remaining pairs and self terms. Both transpose directions are evaluated.
Stable rank sorting and non-strict inequalities retain repeated coordinates.
The tests explicitly enumerate the pair coverage independently of kernel values.

For one ordered axis, orient by s = ±1 so every query follows every source.
Let L and U bound the oriented source coordinates. The two branches use

```text
source distance v₀ = U−sx,      target distance d₀ = sq−U,
source distance v₁ = sx−L,      target distance d₁ = 1+L−sq.
```

All distances are nonnegative. For each branch there are two source features
`exp(−av)` and `v exp(−av)`, with target features `(c₀+c₁d) exp(−ad)` and
`c₁ exp(−ad)`. Their dot product recovers the two terms of Hk. Across x/y
this gives 16 moments. No positive global `exp(a*x)` weights are formed.

In z, use the periodization of `(1+a|z|) exp(−a|z|)`, divided by
`Z = 1 + 2[r/u + a*r/u²]`. Forward and backward states store the zeroth and
first distance-weighted moments. Moving distance Δ updates the state by

```text
S₁ ← exp(−aΔ) (S₁ + Δ S₀),
S₀ ← exp(−aΔ) S₀.
```

The old S₀ is used in the first line. A source adds its x/y-weighted RHS to S₀;
a target reads `S₀+aS₁`, contracted with its x/y features. For a full previous
period's contributions T₀,T₁ at the initial query bound, the closed periodic
state is `S₀=T₀/u`, `S₁=(T₁+r*S₀)/u`. Both sweep directions include the
infinite periodic tail through this closed recurrence. Sources and targets in
each rectangle have disjoint IDs; ties in z are assigned consistently by sweep
order, without duplication of their zero-distance term. Direct blocks include
diagonal terms. Final normalization is variance³/(H²Z).

The 13 GPU tests compare against an independent hyperbolic analytic formula
at length scales .001, .01, .3 and 3; they cover random, tied/coincident and
seam-crossing coordinates, unequal signed RHS, symmetry, positive quadratic
forms, pair coverage, and a separate finite periodic image sum at ℓ=.2.
No spatial threshold or Fourier cutoff is used by the range backend.
Floating-point underflow and accumulation roundoff still apply.

## Next optimization

The action uses O(N log² N) retained pair-partition entries in the worst case,
with constant 16-channel work per entry and RHS. The current CPU planner sorts
individual z blocks independently and is not optimized. B has 927,805 far
entries, 183,199 direct entries and 17,602 far groups; CPU planning takes
0.921 s. One warp handles each rectangle/direction/RHS, serially traversing its
z ordering. The longest B rectangle has 10,388 entries. This exposes a long
serial dependency, but a hardware profile is still needed to quantify it
against exponentials, atomic accumulation and launch overhead.

The next candidate is a hierarchical segmented scan of the same decaying
transfer states. It must retain periodic closure, ties and the full analytic
accuracy gate, with bounded scratch storage. Retain this implementation as the
independent algorithmic control. [Parallel-scan follow-up #43](https://github.com/alecjacobson/fast-stochastic-psr/issues/43)
tracks this optimization. CG, low-noise conditioning, SDD comparisons,
coherent priors and full application timings remain open under
[#18](https://github.com/alecjacobson/fast-stochastic-psr/issues/18).
A correct exact action alone does not solve the posterior-sampling problem.

## Reproduce

```bash
./reproduction/run_env.sh -m pytest -q tests/test_periodic_range.py
./reproduction/run_env.sh -m reproduction.benchmark_periodic_range --case A
./reproduction/run_env.sh -m reproduction.benchmark_periodic_range --case B
./reproduction/run_env.sh -m reproduction.validate_periodic_range
```
