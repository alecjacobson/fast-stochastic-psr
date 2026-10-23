# Larger distinct inputs

A three-million-observation scorpion input is prepared for the next scaling
comparison. All **3,000,000 observed positions are distinct** under an exact
float32 row comparison; no smaller point cloud is tiled or duplicated.
[Manifest and payload hashes](scorpion_3000000/manifest.json),
[generation log](scorpion_3000000/generation.log).

This is synthetic input preparation, not a measured reconstruction or
application result. It does not establish uncertainty convergence, a speedup,
or an upstream resource limit at three million points.

The existing visible-cloud generator samples mesh triangles by area and tests
visibility from the two released cameras. It uses seed 2718 and the same .02
coordinate/normal perturbations as the million-point input. Changing the
requested count changes the random-number schedule; this cloud is not a
guaranteed nested extension of the saved million-point cloud. The manifest
pins the mesh and generator hashes, package versions and saved array hashes.
Saved noisy positions, normals and clean surface positions are float32 arrays
of shape `(3000000, 3)`, each 36 MB; they live outside Git.

```bash
CUDA_VISIBLE_DEVICES='' ./reproduction/run_env.sh \
  -m reproduction.prepare_large_cloud --points 3000000 \
  --output /opt/geospsr-runtime/large-cloud-3000000
```

Use a new output directory when reproducing: the preparer refuses to overwrite
an existing input. It checks finiteness, shape/dtype, distinct observed positions
and hashes after saving/reloading. Preparation runs on CPU; no posterior fit
or GPU job is launched by this command.
