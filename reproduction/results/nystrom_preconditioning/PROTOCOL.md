# Landmark Nyström preconditioner pilot

Compare packed full-cube200 unpreconditioned CG with landmark ranks128,512,1024.
One million existing observations, eight fixed conditional draws, batch2,
rtol1e-5, noise variance.02, length scale.01, axis variance.1, scalar cutoff50.
Landmarks: uniform without replacement, NumPy seed41; identical source data.
Analytic covariance columns construct only the preconditioner. The unchanged
full finite-cube action and original prior determine the posterior solves.
No centered hybrid, narrowed spectrum, changed RHS or looser stopping tolerance.

Use the spectral scaling of Frangella/Tropp/Udell rather than replacing the
posterior with a low-rank solve. Column sampling is a different sketch from
their Gaussian-sketch guarantees. Double-precision construction in row blocks,
float32 retained basis; eigenvalue truncation relative1e-10 in both small Grams,
explicit orthogonality/SPD guard. Report retained ranks and eigenvalue range.

Fresh process per configuration, three solve repetitions reusing setup. Keep
all outcomes. Report setup, solve medians, setup plus solves, retained memory
and process memory separately. Timing ratios require all four whole-run
sampled-isolation audits to pass, with no favorable-run replacement.
Independent32-row finite residual gate .002%; same-model paired field .05%,
variance/covariance .01%, existing ray/collision/next-view gates unchanged.
These numerical gates do not certify full analytic, absolute MC or continuous
geometry accuracy. A significant iteration/total-cost improvement can motivate
larger held-out-RHS and coherent-producer validation, declared separately.
Otherwise retain negative results and do not expand sample counts merely to
hide repeated application overhead. Do not stop other GPU jobs.
