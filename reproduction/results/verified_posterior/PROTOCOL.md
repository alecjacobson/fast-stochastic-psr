# Verified coherent posterior conditioning

Declared before application runs. Test released A (691 points) and the published
three-million-point cloud, eight coherent draws each, seed 701, batch one.
Training cube 200, scalar/cross cube 50, analytic normalization, lengthscale .01,
per-axis variance .1 and observation-noise variance .02. Production uses float32,
NUFFT eps 1e-6, CG rtol 1e-5; independently checked actual residual <=2e-5,
up to two corrections at native CG rtol .1. Serial float64 verification uses
eps 1e-10, oversampling 2 and the identical finite cube.

Use the production streaming API and retain each joint scalar prior and
normal-plus-prior-plus-noise RHS. Compare its output with conditioning those
**identical** arrays using dense finite-cube Cholesky for A, or float64 stored
weights refined to actual residual <=2e-7 (cap four corrections) for three
million. Reference scalar aggregation and queries use float64, eps 1e-10.
Reference mean is separately tightened and centered at the observations.
Shared prior arrays isolate conditioning/aggregation error: this does not prove
the float32 prior sampler's covariance law. The large reference shares the
verification operator and is a stronger numerical control, not an exact oracle.

Queries: 256 uniform points, 256 observation-near points (seed 702); 24 rays
with 256 steps, 17 spheres with 256 surface points each. Calibrate the spheres
around a deterministic reference-mean crossing before drawing; no sample-dependent
geometry selection. Retain query coordinates and paired output fields.
Use existing field_quality gates: fluctuation-scaled L2 <5e-4, empirical variance
and 32-query covariance relative L2 <1e-4. Existing v3 event gates: ray RMS
probability difference <=1pp, maximum ray and collision difference <=2pp, and
view-score regret <=two ray steps. Require at least one uncertain collision.
Report all outcomes, including failures. Eight samples test numerical coupling,
not absolute Monte Carlo probability accuracy or continuous collision geometry.

Time synchronized fit, prior generation, verified solves, coefficient aggregation
and queries separately. Reference/geometry/export costs stay separate. Audit
GPU processes. No upstream/application speedup is claimed from this experiment;
contended timings remain diagnostic. Save large RHS/weights/coefficients in the
runtime archive, hashes and small query fields in the public repository.
