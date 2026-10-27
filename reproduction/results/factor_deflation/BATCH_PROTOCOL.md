# Wider factor-action batches at fixed selected settings

Declared after original A calibration completed, while original B calibration
was still running and before either original confirmation completed. A selects
rank256/count128 and its first-estimate cost is 1.81 s versus deterministic
1.66 s. The factor object already supports arbitrary RHS batches; test batch16
instead of batch4 to reduce transform/triangular-solve overhead. This changes
execution granularity, not rank, sample count, Gaussian law or precision.

After the original experiment finishes, run one new confirmation process per
case at exactly the rank/count selected by that case's original calibration.
Use the same independent confirmation sketch/evaluation seed pairs as the
original confirmation to compare the numerical effect of batching. Field
sampling stays at batch8, preserving its Gaussian stream. No ranks/counts are
reselected for the batch16 implementation. Compute fresh deterministic medians
(three evaluations) and all existing control/quality/isolation checks. Include
batch16 plan setup and all preparation. If quality, memory or isolation fails,
retain it with no rescue repetition and no ratio. Compare costs to each run's
fresh deterministic baseline; old batch4 costs remain descriptive, not a paired
fresh remeasurement. This is a separately declared implementation variant,
not selection of faster repetitions from the original confirmation.

Only launch after all earlier GPU jobs finish. Record source manifests for
both versions. Limit the candidate to batch16; testing other widths is separate
future work. Dense-factor and fixed-query scope remains unchanged.
