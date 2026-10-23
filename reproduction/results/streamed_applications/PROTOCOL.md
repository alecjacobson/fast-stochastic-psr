# Streaming application execution and numerical checks

Keep the million-point source order and saved application geometry from the
common-RHS experiment. Fit a fresh consistently coupled finite cube200 model
with scalar cutoff50, noise variance .02, analytic normalization, NUFFT eps1e-6
and CG rtol1e-5. Use folded coherent priors and packed conditioning, batch2,
seed51. This study does not use the narrower centered inverse.

First compare eight retained and streamed draws on the same fitted model,
including cached fields and ray/collision events. Retain the strict same-model
field/covariance gates and event gates. Then execute 128 streamed draws with
predeclared checkpoints 8,32,128. At every batch check 32 independent finite
covariance rows, seed3187, against the existing 0.002% residual gate. Compare
cached query fields against ordinary type-2 evaluation with field error divided
by fluctuation norm <0.05%. Do not change gates after observing failures.

Accumulate fields, a 32-query covariance, binary events and nested two-resolution
geometry differences online. Preserve every coarse ray point and sphere point
in the finer discretization. Report a Hoeffding union bound across every event
and all declared checkpoints. A radius over two percentage points fails the
new absolute Monte Carlo target; paired numerical agreement cannot replace it.
Two resolutions alone do not certify continuous geometry or full-model error.
Numeric trapezoidal next-view scores replace the inherited Boolean integration
bug; historical scores are audited separately in issue28.

Record fit, prior generation, conditioning, aggregation, cache creation, queries,
reductions, validation and complete instrumented wall time. Include allocator
and process-memory observations with their sampling/coverage limitations.
This is an execution/convergence diagnostic, not an optimized end-to-end speed
comparison. No timing ratio or complete matching-quality claim is promoted.
The initial eight-draw run predates the score correction; archive it separately
and rerun that control with numeric integration.
