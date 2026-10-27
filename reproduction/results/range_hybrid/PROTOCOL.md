# Range action scheduling profile and hybrid pilot

Before measurements, profile native compact action at B and1M using CUDA event
stage times, first and3 later calls. Separate far groups by size:1–32,33–128,
129–512,513–8192,>8192 entries. The partitioned profiling kernel must use the
same coefficient cache and preserve global group indexing; verify its summed
output equals the unpartitioned action. Timing is not hardware-counter profiling.
Retain memory/setup limits and all failures; no application claims.

If small-group scheduling is material, test a compact serial sweep for groups
<=32 and the existing warp scan for larger groups, with original direct leaves.
This fixed threshold is declared before its timings. No pruning/precision change;
full analytic A/B gate1e-7 per RHS including covariance alone, periodic/tied and
long/short-scale tests. Compare fresh native compact control and hybrid setup,
first and next3 actions. Only after credible evidence declare a scale/fit follow-up.

## Cached-leaf follow-up

The fixed32 hybrid passes and lowers B24.70ms to18.92ms, with negligible
additional storage; A is slightly slower. Million direct leaves alone cost371ms.
Before new measurements, cache analytic direct-leaf matrices in float64 and
compare leaf sizes32 and128, both using the same hybrid threshold32. This moves
work between exact pair partitions, without changing the kernel. Charge all
matrix construction and retained bytes; preserve both settings and failures.
Test explicit pair coverage and independent analytic actions, including ties,
seams and short/long scales. A/B first+3 actions and full covariance gate1e-7.
Do not infer scale or fitting gains before separately declaring those experiments.

## Leaf128 fit and million-point follow-up

Both cached-leaf pilots pass; at B leaf32 is14.6ms, leaf128 is10.0ms, versus
compact24.7ms. Fix leaf128 for this follow-up based on the retained two-setting
pilot. Repeat the A/B/stress-B full analytic fit protocol: same normals, dense
Cholesky, CG rtol5e-9/maxiter1000, independent residual<=1e-8 and per-RHS weights/
all20k centered mean<=1e-7. First+3 solves and setup separately; no SDD claim.

Compare fresh native compact32 and cached-leaf128 actions sequentially at1M,
using the saved seed4501 RHS and1,024 seed4601 reference rows, all sources in the
independent reference. First+3 calls, setup/storage and sampled isolation; cap
200M plan entries and32GiB combined cache, initially require40GiB free. Release
one model before constructing the next. No full-row/posterior/application claim.
Keep every result; no timing-based retuning or repeating for cleaner isolation.
