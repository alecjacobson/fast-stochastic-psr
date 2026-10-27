# Stronger feasible setup baseline

Declared while the strict plain run is in progress, before strict comparison
results. The first protocols use bounded two-pass float32 construction because
its 24.6GB feature temporary cannot coexist with the retained double verifier
and 12.3GB float32 basis. That cache need not stay alive during coarse setup.

After the strict plain/float32/fused16 runs finish, test a fourth fresh process:
float32 monolithic Nyström with **one-pass retained features**, releasing only
the verification-plan cache before construction and rebuilding it lazily in the
first conditional solve. Keep native kernel, fitted mean, rank1024/seed41,
2e-6 verification threshold, all eight RHS/prior/queries and every numerical
gate identical. No relaxed precision, kernel, rank or accuracy. Include cache
release in setup and verifier reconstruction in first conditioning cost. The
estimated peak may fit within 48GB; record any allocation failure rather than
silently changing the experiment. Preserve the bounded baseline either way.

Extend timing eligibility to include this stronger setup baseline: all four
strict runs must complete and pass numerical, pairwise, shared-hash and whole-run
sampled-isolation gates before any strict ratio is reported. Report fused warm
conditioning and first-use setup-inclusive costs against plain and BOTH float32
construction policies; do not choose a weaker comparator. This extension tests
an identified memory-lifetime optimization, not a rerun for favorable timing.
It does not change or replace any original/follow-up measurement.
