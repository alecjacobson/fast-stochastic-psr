# Analytic verification and refinement of NUFFT fits

This experiment combines the fast finite-cube NUFFT solver with the full analytic
periodic range action. All A/B and short-scale/low-noise fits pass after either
analytic CG refinement or defect correction, but both combinations are **slower
than analytic CG from zero at those sizes**. Those losses are retained.

The million-point comparison is reported below after independent residual,
weight and mean-field validation. This is a mean-fitting study; it does not
validate covariance, coherent priors, full applications or an upstream SDD
speedup. The scalar/cross representation remains finite cutoff50.

## Method and stopping rules

Let A be the full analytic training covariance plus noise, and let B be its
finite cube200 approximation with the same noise. First solve approximately
with B using packed float32 NUFFT CG (eps1e-6, upsamp1.25, rtol1e-5, maxiter500).
Compare three routes to the analytic solution:

- Analytic CG from zero, double precision, rtol5e-9/maxiter1000.
- Analytic CG from the NUFFT initial solution, with the same stopping rule.
- Defect correction: compute `r = b - A*x`, solve approximately `B*delta = r`,
  and accumulate `x += delta` in double. Recompute the analytic residual after
  each correction, stopping at relative residual1e-8 or eight corrections.

The correction RHS changes, and inner solves use their own relative tolerance.
This is separately verified defect correction, not ordinary PCG with a fixed
SPD preconditioner. Convergence is not guaranteed for arbitrary length scales
and noise. The finite solve's reported residual is not an analytic residual.
The outer history and all inner solver reports are retained, including failures.

The quality gates are relative residual<=1e-8 per normal component, weight
relative L2<=1e-7, and centered mean relative L2<=1e-7 at all20,000 slice queries.
At A/B the reference is an independent double dense Cholesky solve. At one
million, it is a separately converged analytic CG solve, with independent direct
checks at1,024 rows. All-row range residual and sampled independent direct
residual are distinguished. These strict gates are not the same as earlier,
looser application-output tolerances.

Two unit tests verify correction convergence against an independent solve,
original-RHS normalization, zero/unequal channels, unchanged input weights,
exact initial solutions, divergence reporting and invalid/nonfinite corrections.
The prior range/NUFFT controls remain unchanged. The experimental implementation
is [analytic_refinement.py](../../../geospsr/analytic_refinement.py).

## A/B controls and retained timing losses

Same observations, normals and queries across every method. Standard model:
ell=.01, per-axis variance=.1, noise variance=.02. Stress B uses ell=.005 and
noise=.0004. Four runs each: first retained separately and median of the next
three below. Combined solve times include the initial finite fit; setup is
charged separately. All times are seconds.

| Case | Analytic CG from zero | Finite NUFFT only | NUFFT + analytic CG | NUFFT + defect correction |
| --- | ---: | ---: | ---: | ---: |
| A691 | .00426 | .25901 | .26169 | .77855 |
| B17419 | .18374 | .65790 | .77000 | 1.94830 |
| B short/low-noise | .56047 | 1.86179 | 2.32100 | 15.36654 |

Finite-only fits **fail the analytic quality gates** in all three cases; their
timings are diagnostic, not matching-quality baselines. Their analytic residuals
are approximately1.53e-4,1.40e-4 and1.71e-2. Mean errors are1.48e-4,7.55e-5 and
6.46e-3. These failures do not invalidate separate earlier studies with different
model/output tolerances.

Warm analytic CG needs3,9,43 iterations after the finite fit, compared with
6,17,54 from zero. Defect correction needs2,2,7 outer corrections. Final analytic
residuals are at most4.23e-9,4.14e-9,4.79e-9 for warm CG and3.86e-12,3.99e-11,
1.80e-9 for defect correction. Full weight and centered-mean gates pass. Final
means differ from dense by at most2.00e-9 for warm CG and2.36e-11 for defect
correction.

Analytic setup costs .0363,.0565,.0543s; finite spectrum/plans add2.1878,1.5668,
1.5840s. Thus setup strengthens the timing loss at A/B. Dense factorization and
triangular reuse costs are also retained in [results.json](results.json).
The native library was already compiled; a separately measured .98s cold build
is excluded. Common initialization can be reused by later methods; these are
not independently cold-process comparisons. No run or failed finite control is
discarded. Last-run weights and all slice means are in the three `*_fields.npz`
files; first and all subsequent numerical audits/timings are in the JSON record.

## Million-point pilot

Use the same first million points **and normals** from the archived three-million
scorpion cloud for every method. Point hashes match the published range control;
this is not assumed to be the earlier million-point SDD benchmark's input.
The standard ell=.01/noise=.02 model is tested; no large-scale short/low-noise
claim follows from it. The native leaf128 analytic action retains its existing
200M-entry/32GiB-cache cap, and the run requires40GiB initially free GPU memory.

| Method | First solve (s) | Second solve (s) | Iterations/corrections | Full analytic quality |
| --- | ---: | ---: | --- | --- |
| Analytic CG from zero |120.677|120.672|93 /93 CG iterations|Pass|
| Finite cube200 only |5.084|5.078|57 /57 CG iterations|Fail|
| NUFFT + analytic CG |75.684|76.963|53 /54 analytic iterations after NUFFT|Pass|
| NUFFT + defect correction |24.664|24.655|3 /3 corrections after NUFFT|Pass|

Combined solve times include the initial finite fit. Defect correction reduces
its worst all-observation analytic residual through approximately
`4.28e-4 → 7.61e-6 → 1.77e-7 → 4.53e-9`. All-row residuals are evaluated by the
analytic range backend; **1,024 separate direct analytic row sums include every
source** and independently pass. Final weight errors versus analytic CG are
at most1.26e-8, and the centered mean error over20,000 queries is9.59e-10.
Finite-only weights differ by about0.108%, although their mean error is only
1.84e-5. This distinction matters: a small mean error is not evidence of matching
conditioning weights or posterior uncertainty.

Analytic setup costs10.056s; finite spectrum/plans add2.191s. First setup+solve
therefore changes **130.733 → 36.912s (3.54×)** versus analytic CG from zero.
Repeated solves improve about **4.89×**. This is a comparison against **our full
analytic range-CG implementation**, not Holalkere's SDD or a full application.
The final mean aggregation/centering/query stages add .295s plain and .287s
defect correction, excluding common scalar setup of .031s. Independent direct
audits are validation-only and separately recorded. Analytic residual evaluations
performed by the solvers themselves are included in their measured costs.

The reference at this size is independently converged analytic CG, **not dense
Cholesky**. It passes all-row range residual<=4.62e-9 and sampled direct residual
<=4.74e-9. Defect correction passes sampled residual<=4.49e-9. These establish
mean-fit numerical agreement within the specified model, not covariance, coherent
prior, geometry convergence or downstream decision quality.

[Million pilot records](million/results.json) retain both repetitions, inner
solver histories, all setup/mean/audit times and sampled process-isolation data.
The native library was cached; this is not cold-install latency. No timing or
failed finite result was discarded.

## Runtime reset and confirmation

After the pilot completed, a runtime reset removed the ephemeral Python
environment and the pilot's full-weight `.npy` archives. The repository worktree
retained the complete JSON records, all inputs needed to repeat the experiment,
means, sampled weights and sampled independent residuals. We **cannot replay
those original full-weight hashes** and do not claim otherwise. Their surviving
mean/sample controls are replayable.

The locked environment was restored, and all12 targeted solver/leaf-kernel tests
passed. A separate one-pass confirmation of all four methods uses the surviving
identical input arrays and unchanged settings. It has its own timing record and
full-weight hashes; it does not overwrite or purport to recreate the lost pilot
weights. Confirmation full weights will be published as durable release assets.

The [confirmation record](confirmation/results.json) passes every analytic gate
for plain, warm-CG and defect-correction fits; finite-only fails again. Its solve
times are120.677s plain,5.065s finite-only,76.973s warm-CG, and24.580s defect
correction. Analytic setup is9.684s and finite setup1.852s, giving
**130.361 → 36.116s (3.61×)** for setup+first fit. Defect mean error is9.59e-10,
weight error<=1.26e-8 and sampled independent residual<=4.49e-9. The confirmation
follows preflight tests, so initialized code/native-library cache is available;
it is not a cold-install benchmark. Both pilot and confirmation process audits
pass, as does the A/B audit. Periodic process sampling cannot rule out short-lived
interference.

[Full confirmation weights](https://github.com/alecjacobson/fast-stochastic-psr/releases/tag/analytic-refinement-v1)
are release assets, with file sizes, SHA-256 file hashes and array hashes in the
[manifest](confirmation/release_manifest.json). This preserves auditability
without relying on ephemeral runtime storage. Inputs, means, sampled residuals
and sampled weights remain in the repository's control archives.

Replay also caught a precision mismatch in the finite-only sampled audit: the
original benchmark multiplied that method's float32 weights by the noise in
float32, whereas the initial replay promoted the term to double. The original
expression is now replayed exactly, with a separate fully double audit of every
method. The finite-only sampled residual still fails (up to4.41e-4); the largest
diagonal-rounding difference is9.06e-8. All accepted methods use double weights
and have zero discrepancy from this cause. No tolerance was loosened. The failed
replay and resolution are retained in [validation_history.json](validation_history.json),
with fully double results in [validation_diagnostics.json](validation_diagnostics.json).

Final replay verifies12 A/B/stress fits, the original million pilot's four
retained mean/sample controls, and four **complete confirmation** fits. GPU
checks regenerate full-row direct residuals at A/B, all-row range residuals at
one million, and1,024 independent direct residual rows for every confirmation
method. The lost original full-weight archives remain a documented limitation.

## Remaining work and reproduction

[#45](https://github.com/alecjacobson/fast-stochastic-psr/issues/45) remains open
for strong same-input SDD comparisons, reuse across posterior RHS and application
validation. The measured benefit is at one million observations under the stated
strict analytic mean-fit gates; A/B and stress B are slower than analytic CG from
zero. It is not a general speedup over NUFFT itself.

[#46](https://github.com/alecjacobson/fast-stochastic-psr/issues/46) tracks the next
required prior-law step: matrix-free coherent draws of analytic omitted covariance
plus observation noise, replacing the dense residual Cholesky already used in
small-N exact sampling. Corrected mean/conditional solves alone do not establish
matching posterior variance, ray/collision probabilities or next-view decisions.
No practical ContactFusion or other application speedup is claimed here.

[PROTOCOL.md](PROTOCOL.md) records the pilot, scale follow-up and runtime-loss
confirmation before their measurements. The original large pilot command needs
the archived3M input; the confirmation uses surviving repository input controls.
The locked environment can be restored with `reproduction/setup.sh`.

```bash
./reproduction/run_env.sh -m pytest -q tests/test_analytic_refinement.py
./reproduction/run_env.sh -m reproduction.benchmark_analytic_refinement
./reproduction/run_env.sh -m reproduction.benchmark_million_refinement
./reproduction/run_env.sh -m reproduction.confirm_million_refinement
# Or download complete confirmation weights instead of rerunning the fit:
gh release download analytic-refinement-v1 -R alecjacobson/fast-stochastic-psr \
  --dir /opt/geospsr-runtime/analytic-refinement/confirmation
./reproduction/run_env.sh -m reproduction.validate_analytic_refinement --gpu-audits \
  --confirmation-archive /opt/geospsr-runtime/analytic-refinement/confirmation
```
