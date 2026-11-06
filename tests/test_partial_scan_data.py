import numpy as np
from plyfile import PlyData, PlyElement

from reproduction.stanford_partial_data import range_normals


def test_range_grid_axes_orientation_and_discontinuities(tmp_path):
    # A nonsquare grid catches a transposed row/column interpretation.
    rows, cols = 3, 5
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    xyz = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(rows*cols))).astype(float)
    xyz[-1, 2] = 100  # Disconnected depth spike must not produce an observation.
    vertices = np.empty(len(xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    for i, name in enumerate('xyz'): vertices[name] = xyz[:, i]
    indices = np.empty(len(xyz), dtype=[('vertex_indices', 'O')])
    for i in range(len(xyz)): indices[i]['vertex_indices'] = np.array([i], dtype=np.int32)
    path = tmp_path/'grid.ply'
    PlyData([PlyElement.describe(vertices, 'vertex'), PlyElement.describe(indices, 'range_grid')],
            byte_order='>', obj_info=[f'num_rows {rows}', f'num_cols {cols}', 'echo_lgincr 1']).write(path)
    x, normal, info = range_normals(path)
    assert info['raw_vertices'] == 15 and info['kept_vertices'] == 14
    assert np.all(x[:, 2] == 0)
    np.testing.assert_allclose(normal, np.tile([0, 0, 1], (len(x), 1)))
