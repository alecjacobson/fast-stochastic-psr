# Real-scan variance workload

The inputs are the oriented Eagle (796,825 points) and Torso (3,488,432 points)
from the [PoissonRecon sample data](https://github.com/mkazhdan/PoissonRecon).
`data/*.json` records upstream URLs, checksums, normalization and contact selection.
Downloaded data stay outside Git; `reproduction.fetch_poissonrecon` reproduces them.

Preserve positions and normal directions, normalize normal lengths, scale the
longest bounding-box side to two, and use a rigid frame at a geometrically chosen
contact. Nested subsets use a fixed random permutation. These decisions precede
solver timings. No timing-dependent contact selection, point jitter or extra noise
is applied. Torso's supplied normals have strongly varying magnitudes; normalizing
them is an explicit modeling choice. The envelope mesh is not used as truth or a
constraint. These experiments measure numerical performance, not calibrated
uncertainty or reconstruction quality.

The shared model is the nonperiodic independent-component Matérn-3/2 vector GP
and its free-space Poisson scalar projection. Lengthscale is 0.03, amplitude is
0.0025 and observation noise variance is 0.000025. The query box has halfwidth
0.005 around (1,0,0). A deterministic mean level at that known contact defines
marginal inside probabilities; it does not condition the scalar field on a new
contact observation. The workload contains 16 batches of 65,536 query samples
(1,048,576 total), grouped into rays and ranked by mean Bernoulli ambiguity.

The scaling regime increases sampling density on a fixed object, with fixed
physical lengthscale, noise and query region. At a fixed physical cutoff, the
number of radius neighbors grows with N, so the locality graph can approach
O(N²) entries. H² compresses separated interactions instead of discarding them;
bounded hierarchy ranks and iteration counts would give near-linear conditioning
work at fixed regional rank. Dense Cholesky costs O(N³) preparation plus O(N² r)
for r compressed variance right-hand sides. SDD costs O(N b r) per iteration for
batch size b; this study does not establish its iteration count at convergence.
Ranks, interpolation order and iterations can grow to maintain output accuracy,
so the measured curves—not those conditional complexity estimates—decide wins.

Every solver gets the same regional cross compression and polynomial mean/variance
cache. Reported `total_seconds` includes hierarchy/factor/preconditioner setup,
GPU representation, cross compression, mean and variance preparation, polynomial
collapse, and all probability queries. Common runtime/JIT query initialization,
input preparation and offline numerical verification are reported separately.
A factor or hierarchy is not supplied for free. No online cost is charged only
to a baseline when all methods can reuse a regional cache.

The experimental H² backend is an algebraic FMM-style hierarchy. Its optional
streamed interaction action preserves every represented near/far block while
bounding the CSR workspace. It recomputes values during each action; this extra
work is timed. Exact predecessor k-nearest neighbors replace the native AFN
coarse-octree search bottleneck. The optional `psr_higp_kd_cext` module preserves
the original native module and all NUFFT code paths. Preconditioning factors a
small coarse block and local sparse systems, never the full observation matrix.

The radius baseline has the same AFN and regional cache, but drops covariance
entries outside a declared radius. Its streamed variant bounds storage by
regenerating radius rows each action. Hard truncation is not guaranteed positive
definite; failed curvature or accuracy is recorded, not repaired by increasing
noise. Cached sparse is preferable where it fits and must be included there.
The cached allocation guard is a configurable fraction of available memory;
a guard refusal is distinct from an actual allocation failure. Streamed radius
fill uses a warp per row and ballot compaction. It preserves the scalar fill's
columns and values exactly; the retained chunk microbenchmark is not a solve
speedup. Both radius variants retain float64 values. A mixed-precision locality
control remains an open comparison (issue 87).

SDD uses the compiled original update with batch 128, momentum 0.9 and Polyak
weight 0.01, the same compressed right-hand sides, and float32 arithmetic. Its
optional rejection sampler draws uniform subsets without constructing a full-N
permutation at each iteration. This changes the random stream, not the target
sampling distribution. The original permutation sampler remains the default.
SDD budgets count measured solver time; compression and compilation are reported
separately, and would be charged in any passing complete-workload time. Failed
checkpoints are censored observations, not measured completion times. Optional
full-kernel mean correction is considered only after standard variance passes;
no expensive full-variance correction is forced on these SDD controls.

The audit uses direct cross kernels, full float64 all-pairs covariance actions,
and residual bounds based on the observation noise. H² may initialize the
reference but does not certify itself. For R=C−AZ and S=Y−AW, the corrected mean
CᵀW+ZᵀS has error RᵀA⁻¹S. The corrected variance has one-sided uncertainty bounded
by ||R||²/noise. Both bounds propagate through probability endpoints, including
the contact-level uncertainty. Reference refinement stops on output uncertainty,
not a needlessly stringent linear residual alone. The complete kernel action
is still quadratic verification work and is excluded from production timings.

Acceptance: positive variance, max variance error / prior ≤ 1e-4, and max marginal
probability error ≤ 1e-3, including reference uncertainty. Finite independent
probes comprise random queries, every corner, and the contact. These are sampled
checks, not a uniform-in-space theorem. Full-workload comparisons supplement them
when two validated fields are available. Original order-four pilots fail on some
real scans, including dense Cholesky; those records remain as negative evidence.
Do not count unaudited or failed production timings as matched-accuracy wins.
The final independent checks use 17 locations: eight random queries, eight box
corners and the contact. Two individually passing fields can still disagree
between probes: the full-Eagle order-six/12-step and order-seven/8-step fields
differ by 0.00149 over the complete workload. That comparison changes both
interpolation order and iteration count and is marked failed. The final
order-seven result has better sampled error margin; neither a sparse comparison
sharing its interpolation nor these finite probes proves uniform exact-GP error.

Scaling figures select explicit passing records in `selection.json`. They are
single-run exploratory measurements. Accuracy settings and cached/streamed
storage policies may change with N and are recorded per row; fitted slopes are
not complexity proofs. Logical interaction counts include regenerated entries.
The memory panel shows the retained CuPy pool at workload completion, including
reusable workspace; it is not a formally sampled peak or total context memory.
Raw stored-representation sizes separately exclude workspaces and solver vectors.

Hardware: RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb (23.625 GiB visible),
four-core CPU quota, 16 GiB host limit. Raw files record source and data hashes,
build metadata, ranks, iterations, storage and CPU/GPU memory peaks. A dense
factor exceeding capacity is a storage comparison, never an extrapolated runtime.
This is a shared-model algorithm comparison, not pointwise equivalence to periodic
NUFFT or the original Sellán–Jacobson discretization.
