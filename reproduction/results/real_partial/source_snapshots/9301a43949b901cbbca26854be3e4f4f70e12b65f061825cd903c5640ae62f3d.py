"""Real Stanford Armadillo range grids and a closed, reconstructed reference.

The mesh is a reference estimate from the same acquisition, not independent
physical ground truth. Raw positions are never snapped to that mesh. A single
global similarity registration supplies the coordinate convention for labels.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

import igl
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

CACHE = Path.home()/'.cache/fast-stochastic-psr/stanford-armadillo'
BASE = 'https://graphics.stanford.edu/pub/3Dscanrep/armadillo/'
SOURCES = {
    'Armadillo_scans.tar.gz': '0305894482fee00a19ab34f00c4fbff8040af99a9911b60380ad4d13bc78b5c8',
    'Armadillo.ply.gz': '8b9b56cc36e66d54429b1e1e75bd89e833645bfe0dc7c1afd1205877a7356a3f',
}
GROUPS = {'validation': [330, 0, 30], 'test_side': [60, 90, 120],
          'test_back': [150, 180, 210], 'test_other_side': [240, 270, 300]}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def fetch(cache=CACHE):
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCES.items():
        path = cache/name
        if not path.exists(): urllib.request.urlretrieve(BASE+name, path)
        if digest(path) != expected: raise ValueError('Stanford source checksum mismatch')
    if not (cache/'Armadillo_scans/README').exists():
        with tarfile.open(cache/'Armadillo_scans.tar.gz') as archive:
            for member in archive.getmembers():
                if (not (member.isfile() or member.isdir()) or Path(member.name).is_absolute()
                        or '..' in Path(member.name).parts):
                    raise ValueError('unsafe archive member')
            archive.extractall(cache, filter='data')
    if not (cache/'Armadillo.ply').exists():
        with gzip.open(cache/'Armadillo.ply.gz', 'rb') as src, (cache/'Armadillo.ply').open('wb') as out:
            shutil.copyfileobj(src, out)


def range_normals(path):
    """Area-weighted range-grid normals, oriented toward scanner +z.

    Reject triangles spanning depth discontinuities (>4 times grid increment).
    No reference normals, reconstruction labels, or synthetic noise enter input.
    """
    data = PlyData.read(path)
    info = dict(s.split(maxsplit=1) for s in data.obj_info)
    x = np.column_stack([data['vertex'][d] for d in 'xyz']).astype(float)
    rows, cols = int(info['num_rows']), int(info['num_cols'])
    entries = data['range_grid']['vertex_indices']
    indices = np.array([e[0] if len(e) == 1 else -1 for e in entries]).reshape(rows, cols)
    a, b, c, d = [z.ravel() for z in (indices[:-1, :-1], indices[1:, :-1],
                                      indices[:-1, 1:], indices[1:, 1:])]
    faces = np.concatenate((np.column_stack((a, b, c)), np.column_stack((b, d, c))))
    faces = faces[(faces >= 0).all(axis=1)]
    edges = x[faces[:, [1, 2, 0]]]-x[faces]
    faces = faces[np.max(np.linalg.norm(edges, axis=2), axis=1) <= 4*float(info['echo_lgincr'])]
    if not len(faces): raise ValueError('range grid has no connected surface triangles')
    cross = np.cross(x[faces[:, 1]]-x[faces[:, 0]], x[faces[:, 2]]-x[faces[:, 0]])
    if np.median(cross[:, 2]) < 0: cross *= -1
    normals = np.zeros_like(x)
    for col in range(3): np.add.at(normals, faces[:, col], cross)
    length = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(x).all(axis=1) & (length > 0)
    return x[valid], normals[valid]/length[valid, None], dict(raw_vertices=len(x), kept_vertices=int(valid.sum()))


def rigid_scans(cache):
    scans, stats = {}, {}
    for line in (cache/'Armadillo_scans/ArmadilloStand.conf').read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != 'bmesh': continue
        angle = int(Path(parts[1]).stem.rsplit('_', 1)[1])
        x, normal, stats[str(angle)] = range_normals(cache/'Armadillo_scans'/parts[1])
        translation = np.array(parts[2:5], float)
        rotation = Rotation.from_quat(np.array(parts[5:9], float)).inv()
        scans[angle] = (rotation.apply(x)+translation, rotation.apply(normal))
    return scans, stats


def register(scans, vertices):
    """One global transform, shared by all inputs. No per-test-view ICP."""
    x = np.concatenate([scans[a][0][::12] for a in sorted(scans)])
    tree = cKDTree(vertices)
    scale, rotation = 900., np.eye(3)
    translation = np.median(vertices, axis=0)-scale*np.median(x, axis=0)
    for _ in range(100):
        distances, indices = tree.query(scale*x@rotation.T+translation)
        keep = distances < np.quantile(distances, .8)
        a, b = x[keep], vertices[indices[keep]]
        ac, bc = a-a.mean(0), b-b.mean(0)
        u, s, vt = np.linalg.svd(ac.T@bc)
        diagonal = np.diag([1., 1., np.linalg.det(vt.T@u.T)])
        rotation = vt.T@diagonal@u.T
        scale = np.sum(s*np.diag(diagonal))/np.sum(ac*ac)
        translation = b.mean(0)-scale*a.mean(0)@rotation.T
    return float(scale), rotation, translation


def queries(vertices, faces, seed, count=6144):
    """Equal near-surface and volume counts, with exact reference mesh labels."""
    rng = np.random.default_rng(seed)
    area = igl.doublearea(vertices, faces)
    fi = rng.choice(len(faces), count//2, p=area/area.sum())
    bary = rng.dirichlet([1, 1, 1], len(fi))
    surface = np.sum(vertices[faces[fi]]*bary[:, :, None], axis=1)
    normal = igl.per_face_normals(vertices, faces, np.array([0., 0., 1.]))[fi]
    offsets = rng.choice([-1, 1], len(fi))*rng.uniform(.005, .15, len(fi))
    near = surface+normal*offsets[:, None]
    volume = rng.uniform(vertices.min(0)-.15, vertices.max(0)+.15, (count-len(near), 3))
    q = np.vstack((near, volume))
    signed, _, closest, _ = igl.signed_distance(q, vertices, faces, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
    winding = igl.fast_winding_number(vertices, faces, q)
    agreement = (signed < 0) == (np.abs(winding) > .5)
    keep = (np.abs(signed) > .005) & agreement
    return dict(queries=q[keep], inside=(signed[keep] < 0), distance=signed[keep],
                closest=closest[keep], near_surface=(np.arange(len(q)) < len(near))[keep]), dict(
                    seed=seed, proposed=count, retained=int(keep.sum()),
                    boundary_excluded=int((np.abs(signed) <= .005).sum()),
                    sign_disagreements=int((~agreement).sum()), boundary_margin=.005)


def prepare(cache=CACHE, output=Path('reproduction/results/real_partial')):
    fetch(cache); output.mkdir(parents=True, exist_ok=True)
    vertices, faces = igl.read_triangle_mesh(str(cache/'Armadillo.ply'))
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if not np.all(counts == 2): raise ValueError('reference has open or nonmanifold edges')
    if not igl.is_edge_manifold(faces): raise ValueError('nonmanifold reference')
    scans, stats = rigid_scans(cache)
    scale, rotation, translation = register(scans, vertices)
    origin = (vertices.min(0)+vertices.max(0))/2
    normalization = 2/np.ptp(vertices, axis=0).max()
    reference = (vertices-origin)*normalization
    np.savez(cache/'reference.npz', vertices=reference, faces=faces)
    transformed = {}
    diagnostics = {}
    for angle, (x, n) in scans.items():
        x = (scale*x@rotation.T+translation-origin)*normalization; n = n@rotation.T
        d, _, _, refnormal = igl.signed_distance(x[::20], reference, faces, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
        diagnostics[str(angle)] = dict(abs_distance_quantiles=np.quantile(abs(d), [.5, .9, .99]).tolist(),
                                      median_normal_alignment=float(np.median(np.sum(n[::20]*refnormal, axis=1))))
        transformed[angle] = np.column_stack((x, n))
    group_records = {}
    for index, (name, angles) in enumerate(GROUPS.items()):
        rows = np.concatenate([transformed[a] for a in angles])
        rows = rows[np.random.default_rng(7100+index).permutation(len(rows))]
        path = cache/(name+'.npy'); np.save(path, rows)
        panel, qstats = queries(reference, faces, 8100+index)
        distance_to_scan = cKDTree(rows[:, :3]).query(panel['closest'])[0]
        panel['observed'] = distance_to_scan < .025
        panel['occluded'] = distance_to_scan > .075
        np.savez_compressed(output/(name+'-queries.npz'), **panel)
        group_records[name] = dict(angles=angles, points=len(rows), sha256=digest(path),
                                   query_sha256=digest(output/(name+'-queries.npz')), queries=qstats,
                                   observed_queries=int(panel['observed'].sum()), occluded_queries=int(panel['occluded'].sum()))
    metadata = dict(source='Stanford Computer Graphics Laboratory, Cyberware 3030MS Armadillo range data',
        sources={BASE+k: v for k, v in SOURCES.items()}, reference_vertices=len(vertices), reference_faces=len(faces),
        all_edge_multiplicities_two=True, global_registration=dict(scale=scale, rotation=rotation.tolist(),
        translation=translation.tolist(), normalization=normalization, origin=origin.tolist()),
        input_filter='Range-grid triangle edges <= 4*echo_lgincr; vertices need a valid triangle; area-weighted normals face scanner +z.',
        transformation='Inverse scipy xyzw quaternion, then supplied translation; single 80%-trimmed similarity ICP using all 12 standing views; reference bbox longest side normalized to 2.',
        reference_scope='Complete reconstructed mesh from same acquisition, not independent physical truth. Reference used for global registration, normalization, query placement and labels, never to snap positions or supply normals.',
        limitation='No measured scanner likelihood or independent repeated acquisitions; inferred normals and same-scan scalar contacts have correlated errors under the independent-noise working model.',
        views=stats, alignment_diagnostics=diagnostics, groups=group_records,
        source_sha256=digest(__file__))
    (output/'dataset.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--output', type=Path, default=Path('reproduction/results/real_partial'))
    args = p.parse_args(); prepare(args.cache, args.output)
