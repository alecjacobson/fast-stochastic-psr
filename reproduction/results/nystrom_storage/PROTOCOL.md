# Compact Nyström storage pilot

Fixed before the million-point sweep: rank 1024, landmark seed 41, construction
row blocks 8192, application blocks 32768 where selected. Preserve the original
full-cube200 action, scalar50 prior, noise variance .02, CG tolerance1e-5,
NUFFT epsilon1e-6 and common seed42 draws. No spectrum or gate changes.

Run four configurations in fresh processes, in this declared order:

1. Existing cached double features, float32 basis, monolithic application.
2. Two-pass recomputed features, float32 basis, monolithic application.
3. Two-pass recomputed features, float32 basis, row-blocked application.
4. Two-pass recomputed features, scaled float16 basis, row-blocked application
   with float32 projection/correction. Never use half-precision accumulation.

Each uses eight common draws, packed batch2, three solve repetitions, with
one setup per process. Compare configurations2–4 against the fresh baseline1.
Keep strict same-model field/fluctuation .05%, variance/covariance .01%, current
paired event gates, and independent32-row finite residual .002%. Check SPD
after rounding. Preserve every failed construction, solve and quality outcome.

Record retained storage, explicit full-feature temporary bytes, event-counted
CuPy pool live peaks and end-state reconciliation, sampled process memory,
setup and all solves. Pool peaks exclude native NUFFT/context allocations;
process sampling is not an exact device-memory peak. Pool instrumentation is
enabled for every configuration and its overhead is included in all timings.
Report setup-plus-median-solve, solve-only, and storage separately. A timing
ratio requires all four whole-run sampled-isolation and quality gates to pass;
do not select favorable repetitions or rerun to recover a ratio.

If compressed storage passes, confirm the fixed setting on all128 common
draws, reporting the 120 pilot-held-out draws separately against the archived
same-model reference. This is a numerical confirmation, not a historical
timing comparison. Then validate the compact basis in the coherent streaming
producer. Large-N application claims still require MC, geometry and spectral
convergence, and a memory-aware iterative reference.

Post-pilot code review extended the numerical SPD guard to include a possible
float64-storage-to-float32-accumulation cast. The float16 and float32 pilot
bases have the same effective values under that check. The 128-draw and
streaming confirmations use the refined guard; two additional precision cases
are included in the final test suite. No pilot is rerun or gate relaxed.
