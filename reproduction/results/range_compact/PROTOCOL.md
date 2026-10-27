# Compact periodic coefficients

Declared before measurements under #44/#43. Profile cached far/direct kernels
with CUDA events on A/B: first plus three later calls, no hardware counters.
Test a float64 separable x/y cache (16 values per entry versus32) and three z
values per entry (adjacent decay and two periodic initial weights, versus14).
Warp prefix products construct stride and tile-entry decays from adjacent
transitions; no growing exponentials, division by tiny values or term pruning.
The unchanged pair planner and independent analytic controls remain references.

Compare fresh expanded-cache and compact actions with same seed4501 three signed
RHS on A/B, leaf32. Charge setup, first action, next3 actions, retained bytes
and sampled isolation; preserve all failures. Analytic gate1e-7 relative per RHS
with and without diagonal noise. Test seams, ties, long/short scales, multiple
tiles, signed/unequal/zero RHS. Any scale/fit experiments require a new declared
protocol after this pilot. No application speedup from actions alone.

## Native planner follow-up

The compact pilot passes but is slower: B24.7ms versus expanded21.9ms, while
retained bytes drop349MB to149MB. Before new measurements, implement the same
stable-rank partition in C++ with bounded retained entries. Verify every plan
array against Python on random/tied/seam inputs and all A/B arrays in benchmark.
Charge compilation/loading, plan CPU time and device transfer separately.
Compilation cache must be explicit; no claim of cold-process timings.

After matching plans/actions, measure native compact full fits using the prior
A/B/stress-B protocol (rtol5e-9, independent residual1e-8, weights/20k centered
mean1e-7, first plus3 subsequent solves). Keep same dense Cholesky reference.
Also test a fixed100,000-point prefix of the archived3M scorpion cloud, seed4501
three signed RHS, with 1,024 fixed random query rows (seed4601) evaluated against
all100k sources by the independent analytic kernel. This sampled action check
is weaker than full-row A/B validation; no posterior/application or massive
speedup from it. Retain first+3 actions, storage, process RSS high-water and
sampled GPU isolation. Use max_entries20M and max_cache_bytes4GiB; stop on caps.

## One-million action follow-up

The100k run passes every sampled analytic row with worst covariance error below
7e-15, retained1.20GB and warm action162ms. Before larger measurements, run the
same fixed-prefix/seed protocol at1M, using1,024 fixed query rows and all1M
sources in the independent reference. Cap retained plan entries at200M and
cache allocation at32GiB; require at least40GiB free GPU memory before starting.
Retain cap failures, setup/first/next3 timings, process RSS high-water, device
storage and sampled isolation. Reference rows are streamed in blocks of64.
No full-row/posterior/CG/application accuracy or speedup claim from this run.
The100k CPU setup overlapped a separate CPU-only cold-library compilation;
retain that run and this qualification rather than rerunning for a better time.
