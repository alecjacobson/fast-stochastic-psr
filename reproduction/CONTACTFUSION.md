# ContactFusion as a practical downstream benchmark

Status on 2026-10-22: **assessment and benchmark design; no ContactFusion speedup
has been measured**. Our scorpion timings do not establish a robotics result.
[Tracking issue #23](https://github.com/alecjacobson/fast-stochastic-psr/issues/23).

## Sources and release status

[ContactFusion, arXiv v2](https://arxiv.org/pdf/2503.16592v2), revised 17 July
2026, reports real-robot reconstruction on five sequences using a KUKA IIWA,
RealSense D435 and ATI force/torque sensor (§V-C). The real experiments use
prescribed sensing motions; active reconstruction is evaluated in simulation.
Table I reports 9,900 ms per SPSMap measurement update and an average F-score
of 0.649. These are the authors' measurements, not our baseline timings.

Its implementation uses a 5 mm grid, marching cubes, and interpolated variance
at mesh vertices (§V-A). Vision and contact normals have different noise
parameters, written as 0.1 and 0.05. Equation 4 carries the previous scalar
reconstruction into the next update as a prior. It does not specify transport
of the full previous posterior covariance. Action selection projects mesh
uncertainty onto camera rays or vertical probe candidates (§IV-D).

The [author's project repository](https://github.com/kamiradi/contact_fusion/tree/6b66b3ce9d8d548b0cba2fe1c5e32ccc6f3aec6c)
contains figures, photographs, videos and the project website. Its
[code link says “Code(coming soon)”](https://github.com/kamiradi/contact_fusion/blob/6b66b3ce9d8d548b0cba2fe1c5e32ccc6f3aec6c/docs/index.html#L302).
At this revision we found no reconstruction implementation or replayable sensor
sequences. A public project page is not a code/data release. Do not infer
sensor measurements from rendered figures. No authors have been contacted.

Two related public repositories provide useful leads, but not a verified
ContactFusion replay:

| Repository revision | Available material | Missing for this benchmark |
| --- | --- | --- |
| [dsvr, 6e7a63a](https://github.com/kamiradi/dsvr/tree/6e7a63a6dd5db6fff41d1fe8aed7705f2c4ad23e) | Dataset schemas and visualization scripts for RGB-D, robot/peg/hole transforms, force/torque streams, reconstructions and next-view scores | Its README points to local `inverse_graphics/data/real_data/.../dataset.npz` paths; no recordings or SPSMap update implementation in the inspected tree |
| [assembly_description, e6fe416](https://github.com/kamiradi/assembly_description/tree/e6fe4160412cd70c1c8e5c7fdf1a258daa563699) | Rectangular/arch peg and hole meshes, robot assembly configurations | Exact correspondence to the paper's experiments, sensor sequences and reconstruction code |

These repositories could make a later adapter/proxy more faithful. Their current
GitHub metadata does not identify a license; no assets or source were vendored
into this repository. A schema or CAD mesh alone is not real sensor data.

## What can transfer from this repository

NUFFT kernel actions, fitting and cached coherent samples address two distinct
costs: map preparation and repeated uncertainty queries. At fixed bandwidth,
preparing coefficients removes the observation count from later query sums.
A compatible uniform mean grid can use an FFT; irregular mesh vertices can use
a NUFFT. Mesh extraction, sample preparation and any interpolation needed for
compatibility still count toward the first result after **every** update.

These are proposed adaptations. Our existing measurements show gains for
specified scorpion workloads, including repeated queries and million-point
conditioning. They do not identify ContactFusion's cost breakdown. Dense or
sparse methods can win at smaller observation counts, so include them in the
baseline sweep. The 9.9-second paper figure cannot be divided by our L40 timings
to claim a speedup: hardware, workload, model and timing scope differ.

Our current spectral sampler uses scalar observation noise, a zero-mean vector
prior and a fitted constant scalar offset. Scalar noise can vary between
single-modality updates; jointly conditioning mixed modalities would require
per-observation noise. It does not implement ContactFusion's
sequential scalar-prior interface. A nonconstant scalar prior needs a specified
mapping to the vector mean and consistent centering/boundary conventions.
Appending all historical normals is a different update unless equivalence is
proved. Equation 4 suggests the current observation batch, rather than the
entire accumulated scan, may determine the conditioning size; this needs code
verification. Our million-observation advantage cannot be assumed to apply to
each small contact update. Retaining a posterior covariance versus restarting
a covariance prior also defines different inference. Spatially ambiguous contact hypotheses must
follow the reference's treatment; treating many hypotheses from one wrench as
independent measurements could create artificial confidence.

## Reproduction stages and acceptance

1. **Recover the reference.** Pin code, data and license when released. Identify
   input counts per update, normal construction, world-to-domain transform,
   scalar prior, covariance reset/transport, centering, boundaries and whether
   the reported noise parameters are variances or standard deviations. Preserve
   camera/contact timing and grouping. Record missing details rather than
   guessing them silently.
2. **Replay a fixed sensing sequence.** Start with the original SPSMap update
   and its 5 mm grid. Measure preprocessing, fitting, uncertainty preparation,
   mean-grid evaluation, marching cubes, mesh-variance queries and action
   scoring separately, plus total first-result wall time and memory. Repeated
   query throughput is a separate measurement. Use the same GPU/CPU resources
   for comparative runs and retain the original implementation result.
3. **Validate inference changes.** First compare acceleration under identical
   mathematical semantics. If GeoSPSR changes the kernel, boundaries or update,
   treat it as a new model and compare reconstruction quality and uncertainty
   calibration in addition to numerical error. On tractable panels compare
   deterministic means, variances and joint covariance. Keep full covariance
   checks for correlated-event queries; marginal agreement alone is insufficient.
4. **Validate task quality.** Report precision/recall/F-score at 1 and 2 mm,
   hole geometry/pose errors, and uncertainty behavior over each update. Compare
   action scores, best-action regret and near ties across independent sample
   seeds. Then test complete active trajectories in simulation, since a changed
   action changes subsequent data. A stable action on one map is insufficient.
   Use original recorded sequences for physical-data claims. No insertion
   success claim follows from reconstruction F-score alone.
5. **Resolve sample and geometry accuracy.** Sweep sample count, Fourier band,
   solver tolerance, grid spacing and action discretization independently. Set
   task-level noninferiority tolerances before timing selection, after reference
   repeatability and recording precision are known. Preserve failed settings.
6. **Scale separately.** Increase genuine distinct observations and query/action
   workloads independently. Compare compiled iterative as well as dense/sparse
   references under explicit budgets. Repeating points or inflating contact
   hypotheses does not establish useful massive-scale reconstruction.

For independent Gaussian samples, pointwise unbiased sample variance has
relative standard error `sqrt(2/(S-1))`: about 10% at 200 draws and 1% at 20,000.
These pointwise estimates do not guarantee action rankings or simultaneous
accuracy over a mesh. Antithetic partners are dependent and do not count as
independent draws for this formula. Common randomness helps *paired numerical*
comparisons but does not eliminate uncertainty in the estimated task quantity.
ContactFusion's stated selection rule uses marginal vertex uncertainty, so
coherent samples are an option, not a prerequisite. Include deterministic
marginal evaluation at the candidate vertices, especially for small inputs.
Coherent samples become necessary for our correlated-event formulation of rays
and collisions, which is a different task from projecting vertex uncertainty.

Until recordings are released, a constructed peg/hole sequence can test the
adapter and numerical semantics. It must be labeled a synthetic proxy, with no
claim to reproduce the real-data table or the authors' exact implementation.
