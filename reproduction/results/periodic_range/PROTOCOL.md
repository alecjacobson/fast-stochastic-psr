# Stable periodic product-Matern range-sum prototype

Declared before implementation measurements. Work under #18/#6/#14. The target
is the full analytic periodic product of three circle Matern-3/2 kernels, not a
truncated Fourier cube. This prototype must first match independent analytic
matrix actions; no FMM or application speedup is presumed.

Normalize each coordinate modulo2pi into[0,1). Recursively split point ranks in
x, partition each cross-block again in y, and cover every pair once with ordered
x/y rectangles or small direct blocks. Each ordered x/y factor has four bounded
source/target features (two periodic directions times constant/linear moments),
so16 moment channels remain. A periodic causal exponential/linear sweep in z
sums each rectangle. Use nonpositive exponents relative to local source bounds
and closed periodic recurrence states; avoid global exp(a*x) weights. Evaluate
both transpose directions, and retain direct leaf blocks including diagonals.
This is our periodic GPU adaptation, not the cited authors' implementation.

Validate float64 actions against independent dense periodic formulas on random,
seam, repeated-coordinate, coincident and short/long length-scale inputs. Test
symmetric action and positive quadratic forms. Use multiple signed RHS and
independent periodic image-sum controls. Only then measure the released A691
and denser B17419 data with three same deterministic RHS columns. Charge CPU
partitioning, device transfer, first compilation/execution and retained storage
separately. Measure three subsequent actions with all times retained. Compare
GPU dense analytic action including its construction separately, and the existing
packed NUFFT cube200 action, reporting both accuracy and timings. NUFFT action
error is measured against the analytic target; a failed accuracy control is
not a matched-quality speed baseline. Full dense B may require substantial GPU
memory; serialize jobs and never interrupt foreign work.

Action relative L2 gate1e-7 for the double range backend. Use all A/B query rows
in its dense analytic reference, streamed if needed. Do not extrapolate matrix-
vector speed to CG/posterior/application gains. A serial sweep per rectangle may
limit GPU parallelism; report this bottleneck if measured. New scales, CG and
application validation follow only if this concrete control is credible. Keep
all failed tests and measurements, and avoid unplanned repetitions for good
isolation. No automatic goal completion follows from this prototype.
