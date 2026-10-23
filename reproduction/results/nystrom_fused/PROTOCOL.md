# Fused half-basis application

The implementation decodes the same stored half basis in registers, with
float32/float64 RHS and accumulation, without a full decoded basis. Projection
uses64 basis columns,1024 source rows per tile, at most1024 source tiles,256
threads and at most8 RHS channels per kernel tile. Correction uses one warp
per source row. Rank/source/RHS tails and odd-row half alignment are explicit.
No operator, prior, tolerance, spectrum or quality gate changes.

First profile the action on the existing million-point source with the same
rank1024/seed41 compact basis used by decoded and fused paths. Compare both to
a blocked float64 evaluation of exactly that stored basis and damping. Six
RHS include mixed scales/signs, a zero channel and a coarse-space direction to
expose cancellation. Application error/input norm must be below5e-6 per RHS,
with exact zeros preserved. This checks application arithmetic, not posterior
quality. Record setup, reference, first use/compilation and seven alternating-
order repetitions; pool live events and whole-run process audit are retained.

If the profile warrants full experiments, run three fresh-process profiles
at one million and three million distinct observations, retaining the initial
profile as the first million-point repetition. No extra repetitions to recover
a favorable timing ratio. A ratio at either N requires all three whole-run
sampled-isolation and numerical checks at that N. Compilation/cache state and
first use are reported separately; median warmed action time is not setup or
application time.

Then compare full conditional solves in fresh processes: unpreconditioned
packed CG as a context baseline, float32 monolithic
Nyström, decoded float16, and fused float16. Fix rank1024/seed41, full cube200,
scalar50, noise.02, rtol1e-5, eps1e-6, eight seed42 common draws, packed batch2,
three solve repeats,32 independent finite residual rows, and pool profiling.
Decoded/fused cases use two-pass construction; decoded row blocks32768. Use
the fresh unpreconditioned fields as reference. Preserve unchanged field/fluctuation
.05%, variance/covariance.01%, paired-event and finite-row.002% gates. Require
all four whole-run audits/quality gates for any timing ratio; include setup,
first solve and median solves separately. No upstream/application ratio follows.

If fused posterior gates pass, confirm the fixed setting on128 common draws,
including the120 pilot-held-out draws against archived nufft128. This is a
numerical check, not a historical timing comparison. Also run the coherent8
retained-versus-streamed control with exact prior/RHS matching. Larger-N action
profiles alone do not establish a larger-N posterior/application speedup.

The unpreconditioned context baseline was added before any full conditioning
run, after the first micro profile, to measure the combined coarse-space and
fusion change against plain packed CG. Use its fields as the common reference
for all three preconditioners. Timing comparisons must include the strongest
existing float32 monolithic preconditioner, not only decoded half storage.

The numerical tests may warm CuPy's disk compilation cache before timing.
Record that limitation for first-use timers. A separate fresh-cache, small-input
compile/first-use diagnostic may measure the fixed fused-kernel startup cost;
it is not an extra timing repetition and cannot rescue a failed isolation gate.
