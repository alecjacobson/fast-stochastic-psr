# Numeric next-view score audit

NumPy trapezoidal integration of a Boolean mask undercounts consecutive true
intervals: the mask `[True, True]` on `[0, 1]` integrates to 0.5, whereas its
numeric indicator integrates to 1.0. The next-view diagnostic now converts
the 5%–95% uncertainty mask to floating point before integration. New paired
diagnostics record quality-gate version 3. This changes score units and can
change selection/regret; it does not change fields or ray/collision probabilities.

This audit recomputes **53 archived artifacts**, including all arrays
in each, without overwriting source records. **0 selections change**
and **0 two-ray-step regret gates change classification**.
**0 artifacts lack a verified/documented distance grid** and are
listed explicitly. Full scores, source hashes, distance arrays and provenance
are in [results.json](results.json). Application-demo grids inferred from the
fixed producing script are distinguished from stored/hashed geometry.

The sequential scan saved decisions but no per-step transmittance, so this
audit cannot reconstruct its adaptive trajectory. It requires a rerun; prior
scan-quality claims must not be treated as revalidated. That work remains in
[issue #7](https://github.com/alecjacobson/fast-stochastic-psr/issues/7).

Run `./reproduction/run_env.sh -m reproduction.audit_view_scores` to reproduce
the audit. Conditioning distance grids were recovered from original scratch
geometry only after checking its saved hash. This report archives those arrays
and [their hash index](geometry_index.json), so later runs can use them when
scratch geometry is unavailable. [Issue #28](https://github.com/alecjacobson/fast-stochastic-psr/issues/28)
tracks the correction and remaining coverage. This diagnostic is a proxy
acquisition score, not a validation of physical scanning performance.
