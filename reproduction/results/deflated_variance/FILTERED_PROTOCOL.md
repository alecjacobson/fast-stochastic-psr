# One covariance filtering step

Declared after the unfiltered pilot executions completed.
A selected rank256/count128 and passed quality, but its first-estimate cost
exceeds deterministic variance. Test a stronger independently chosen subspace:
Q0=orth(pilot fields), Q1=orth(B Q0), then compute B Q1 for the same diagonal
identity. Every action and both QR costs are charged. The evaluation draws remain
independent; filtering uses no evaluation data or reference variance.

Use exactly one filtering step, ranks32/64/128/256, looks32/128/512/2048 and the
same three calibration seed pairs as the original protocol. Keep the original
pilot and its independent confirmation. This is a separately predeclared
algorithm variant, not an extra repetition chosen for favorable timing or
isolation. All inputs, direct analytic model, contrast/probability/positivity
and double-control precision gates remain unchanged. Use the same candidate
selection rule and independent confirmation pairs from PROTOCOL.md, reporting
all candidates and failures. No timing ratios from calibration. Compare only
new confirmation costs against their freshly measured deterministic baseline.
