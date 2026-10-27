# Three-million-point mean-zero slice refinement

Follow up the frozen `large_fit` solutions without refitting or selecting a
new stopping iteration. Check all saved weight hashes and the published source
hash. Replay the original mean queries with maximum absolute error <=1e-6.
Use the saved all-observation centering constants, scalar/cross cutoff 50,
float32 cuFINUFFT epsilon 1e-6, and the original two heights y=-.35 and -.75.
The higher training cutoff 250 is an internal reference, not an exact full
analytic solve. This does not validate scalar bandwidth.

Evaluate each mean directly on nested 257, 513, 1025 and 2049 square grids
covering [-pi/2, pi/2] in x and z. Extract level-zero polylines with contourpy
serial marching squares and keep every connected curve, including open ones.
Save all contour vertices and grid diagnostics; keep the larger mean grids
outside Git. No contour simplification or component filtering.

Measure symmetric Hausdorff distance between the extracted polylines using
samples spaced at most 1e-4 in model units. Report lower/upper bounds using
each polyline's sample-cover radius, rather than claiming that nearest sampled
vertices give exact curve distance. Unit tests must cover known distances,
different tessellations, omitted components, and open endpoints.

Report consecutive-grid distances for each method, both method-to-cutoff-250
distances on every grid, component counts, and open components. Declare only a
numerical refinement check: final consecutive-grid upper distance <=1e-4 times
domain width, with unchanged component and open-component counts. This is not
an application quality threshold or a continuous zero-set/topology proof.
Application tolerances need a task-specific interpretation of model units.

These new grids test discretization on the same two displayed planes; they are
not held-out slice geometry, a 3D surface check, posterior uncertainty, or
probability/event validation. Record query costs and GPU contention for audit,
but do not derive a solver speedup from this saved-fit diagnostic.
