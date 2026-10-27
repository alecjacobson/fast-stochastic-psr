# Follow-up to the retained pathwise residual failures

The initial polynomial pilot remains unchanged. Its large-N pathwise controls
fail the native float32 recomputed-residual criterion despite recurrence
termination. Before adopting a residual-correction step, distinguish error in
the solution from error in evaluating that residual.

Recreate the million-prefix and three-million cases in fresh processes with
two draws at seed509, using the same finite training200/scalar50 kernel and
noise variance .02. Keep each generated RHS fixed through all checks/corrections
and save its hash. These are fresh coupled runs; no bitwise agreement with the
earlier pilot is assumed because those RHS were not archived.

Solve with the original float32 action, epsilon1e-6, rtol1e-5. Record its native
final residual and evaluate an independent float64 action of the SAME finite
cube at epsilon1e-10 on the saved float32 weights. Require each double residual
<=2e-5, matching the existing acceptance threshold. The double check is stronger
numerical evidence, not exact arithmetic or an analytic-kernel/truncation check.

Only if the double residual fails, restart native CG on the same full RHS using
the preceding weights as x0; allow at most two such corrections. Check double
residual after each and retain every attempt. Do not stop on a favorable native
float32 residual while the independent double check fails. Report operator
setup, all solve/correction and verification costs, source/RHS hashes, and GPU
memory/ownership samples. No polynomial or application speedup follows from
this diagnostic. Original failed flags and timings remain in their own records.

The first attempt to construct a three-RHS double action failed during backward
plan creation with CUFFT_INTERNAL_ERROR. Preserve that failure. Retry with one
RHS per double transform, checking the three columns serially to reduce plan
workspace. Precision, cutoff, epsilon and residual thresholds are unchanged;
the exact cause of the cuFFT plan failure has not been established.

## Second follow-up: defects from the independent action

The capped native-restart experiment leaves one three-million-point draw above
the double-precision threshold after two corrections, despite small native
residuals. Preserve this failure too. On the same archived RHS, perform fresh
native solves followed by at most two defect corrections. Compute each defect
using the float64 action, solve that defect approximately with native CG at
rtol .1, accumulate in float64 and then round weights back to float32 storage.
Recheck those stored weights with the independent action at the unchanged
2e-5 threshold. Keep all checks and costs, including the initially failed native
solve. This follow-up is separately labeled and does not alter pilot gates.
