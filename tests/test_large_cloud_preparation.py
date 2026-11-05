import json

import numpy as np
import pytest

from geospsr.nonperiodic import FreeSpacePoissonMatern, RegionalVarianceCache, compress_cross_covariance
from reproduction import poissonrecon_data as clouds


@pytest.mark.parametrize('block_size', [37, 256])
def test_streaming_qr_cross_compression_and_posterior(block_size):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(420, 3))*.15
    y = rng.normal(size=x.shape)
    model = FreeSpacePoissonMatern(.12, .7)
    nodes = rng.normal(size=(20, 3))*.01
    u, c, error = compress_cross_covariance(model, x, nodes, 1e-6, block_size)
    cross = model.cross_covariance(nodes, x).transpose(1, 2, 0).reshape(len(x), -1)
    np.testing.assert_allclose(u.T@u, np.eye(u.shape[1]), atol=3e-10)
    assert abs(np.linalg.norm(cross-u@c)/np.linalg.norm(cross)-error) < 2e-13
    a = model.vector_covariance(x, x)+.05*np.eye(len(x))
    solve = lambda rhs: np.linalg.solve(a, rhs)
    reference = RegionalVarianceCache.from_solver(model, x, y, solve, [0, 0, 0], .01, 3,
                                                  compression_tolerance=1e-6)
    streamed = RegionalVarianceCache.from_solver(model, x, y, solve, [0, 0, 0], .01, 3,
                                                 compression_tolerance=1e-6, cross_block_size=block_size)
    q = rng.uniform(-.01, .01, (50, 3))
    for actual, expected in zip(streamed.predict(q), reference.predict(q)):
        np.testing.assert_allclose(actual, expected, atol=5e-13, rtol=2e-10)


def test_cloud_preparation_preserves_geometry_normals_and_nested_prefixes(tmp_path, monkeypatch):
    rng = np.random.default_rng(14)
    x = rng.normal(size=(32, 3))
    v = rng.normal(size=(32, 3))
    rows = np.empty(32, dtype=[(k, '<f4') for k in ['x', 'y', 'z', 'nx', 'ny', 'nz']]+[('red', 'u1')])
    for k, value in zip(['x', 'y', 'z', 'nx', 'ny', 'nz'], np.column_stack((x, v)).T):
        rows[k] = value
    rows['red'] = 17
    source = tmp_path/'cloud.ply'
    header = 'ply\nformat binary_little_endian 1.0\nelement vertex 32\n'+''.join(f'property float {k}\n' for k in rows.dtype.names[:-1])+'property uchar red\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'
    source.write_bytes(header.encode()+rows.tobytes())
    monkeypatch.setitem(clouds.SOURCES, 'eagle', ('fixture', clouds.sha256(source)))
    output = tmp_path/'cloud.npy'
    clouds.prepare('eagle', source, output)
    xx, vv, meta = clouds.load_cloud(output, 32)
    subset = clouds.load_cloud(output, 11)
    np.testing.assert_array_equal(xx[:11], subset[0])
    np.testing.assert_array_equal(vv[:11], subset[1])
    np.testing.assert_allclose(np.linalg.norm(vv, axis=1), 1, atol=1e-14)
    order = np.random.default_rng(12).permutation(32)
    original = np.column_stack([rows[k] for k in ['x', 'y', 'z']]).astype(float)[order]
    np.testing.assert_allclose(np.linalg.norm(xx[:, None]-xx[None, :], axis=-1),
                               meta['coordinate_scale']*np.linalg.norm(original[:, None]-original[None, :], axis=-1), atol=1e-14)
    anchor = int(np.flatnonzero(order == meta['anchor_source_index'])[0])
    np.testing.assert_allclose(xx[anchor], [1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(vv[anchor], [1, 0, 0], atol=1e-15)
    with pytest.raises(ValueError, match='count'):
        clouds.load_cloud(output, 33)
    with pytest.raises(ValueError, match='checksum'):
        clouds.load_cloud(output, 11, 'wrong')
