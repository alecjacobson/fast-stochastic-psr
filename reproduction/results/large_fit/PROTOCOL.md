# Three-million-point full mean fitting

Use the restored three-million-point scorpion source with all three payload
hashes matching its published manifest. All observations enter each solve;
use the released A 20,000 slice queries, noise variance .02, length scale .01
and per-axis variance .1. Scalar/cross cutoff stays 50.

Run three fresh processes sequentially: conjugacy-folded NUFFT-CG with complete
finite training cubes at cutoffs 200 and 250, then compiled SDD with full analytic
covariance. NUFFT uses rtol 1e-5 and epsilon 1e-6, no coarse preconditioner.
SDD reuses the strongest selected million-point configuration: batch 512, eta 5,
lr=5*512/N, seed 0, momentum .9 and Polyak .01. This is a transfer of that setting,
not a three-million-point hyperparameter search or the untouched Python loop.
Its compiled update algebra has existing upstream-equivalence tests.

Give SDD 600 seconds of update time, excluding compilation and validation;
check every 128 iterations, allowing one-chunk overshoot. Cap at 1,000,000
iterations. Stop on the existing analytic relative-residual upper-endpoint
gate .001, using 128 fixed seeded stopping rows. Divergence and budget outcomes
must remain recorded; they are not proofs of upstream infeasibility.

Check each final solution on the SAME 1,024 analytic rows, drawn independently
and excluding all stopping rows. Compute rows and products in float64, stream
16 rows at a time, and require empirical bootstrap 95% upper residual <.001.
This is sampled validation, not a full-system norm bound or covariance test.
Report cutoff-200 versus cutoff-250 and SDD mean differences on all 20,000
queries, requiring relative L2 <=.002 for any paired-mean quality claim.

Record loading/hash verification, fitting through centering and mean evaluation,
SDD stopping checks, independent validation, full instrumented work wall and GPU
process/memory samples. SDD stopping machinery is part of its fit-through-mean
cost; independent final validation is reported separately. All timings include
first-use setup. Do not compare to historical-machine times. A timing ratio
requires all three runs to pass quality, both paired mean gates and whole-run
sampled GPU isolation. Otherwise retain costs without a speedup ratio.

This advances mean fitting beyond the earlier three-million-point action-only
profile. It does not certify scalar-bandwidth, covariance, posterior events,
application quality or an intractability boundary for the original method.

First-use scope includes model plans, solver compilation where needed, and
aggregation/centering; process imports and CUDA initialization are excluded.
Earlier tests may have warmed on-disk compiler caches. Process first use is not
a claim of cold compilation. Full source arrays and saved final weights live
outside Git; published payload hashes and field/residual panels allow auditing.
