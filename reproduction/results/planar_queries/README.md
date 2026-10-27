# Planar Fourier query controls

`PlanarFourierPlan` uses an exact finite Fourier identity to contract one axis,
with 2D NUFFTs on explicit axis-aligned planes. No modes,
query points or plane heights are removed. Actual arithmetic and NUFFT errors
are checked separately.

The small tests and full A/B fixed-setting variance confirmations pass. See the
[combined report](../packed_factor_sources/README.md) for setup-inclusive timings,
all failures and the separate massive aggregation control.

The original large type1 spectrum comparison failed its 1e-5 threshold against
native 3D float32. Direct sums showed that reference was itself less accurate
on the sampled high modes. [Precision follow-up](PRECISION_FOLLOWUP.md) documents
the change to an independent full double reference and planar oversampling 2.
The revised planar type1/type2 errors are 9.0e-6/2.18e-6. Generic type1 remains a
failed unweighted-spectrum control, so no query-only speed ratio is claimed.
This failure does not substitute for the separate kernel-weighted posterior gates.

[Instrumented forward profile](profile.json) · [Initial failure](initial-micro-failure/results.json) ·
[Selected-mode diagnosis](precision_diagnosis.json) · [Full-query check](microbenchmark.json).

Reproduce the profile and query check with `reproduction.profile_factor_forward`
and `reproduction.benchmark_planar_queries`. The profile needs the archived B
factor basis identified by its hash. The initial implementation is in commit
a7ff655; all later confirmation code and source manifests accompany the report.
