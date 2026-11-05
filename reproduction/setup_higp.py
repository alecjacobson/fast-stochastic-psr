"""Build an isolated, pinned HiGP extension with scalable AFN setup fixes.

Requires setuptools, NumPy, a C++ compiler and the LP64 OpenBLAS bundled by
SCS (or --blas /path/to/libopenblas.so). Does not modify the installed HiGP.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sysconfig

PIN = 'da44859a4dd8d6d12483c1161f54705dd8e33b9a'


def patch_adjacency(root):
    """Replace quadratic node-list capacity by exact-sized adjacency lists."""
    def original(path):
        return subprocess.check_output(['git', '-C', str(root), 'show', f'{PIN}:{path}'], text=True)
    path = 'cpp-src/h2mat/h2mat_typedef.h'
    source = original(path).replace('int *node_far;', 'int **node_far;').replace('int *node_near;', 'int **node_near;')
    (root/path).write_text(source)
    path = 'cpp-src/h2mat/h2mat_build.cpp'
    source = original(path)
    source = source.replace('int *node_near,', 'int **node_near,').replace('int *node_far,', 'int **node_far,')
    for kind in ['near', 'far']:
        import re
        source = re.sub(r'int \*node_'+kind+r'\s*= \(int \*\) malloc\(sizeof\(int\) \* n_node \* n_node\);',
                        f'int **node_{kind} = (int **) malloc(sizeof(int *) * n_node);', source)
        source = source.replace(f'node_{kind}[node0 * n_node + idx0]', f'node_{kind}[node0][idx0]')
        source = source.replace(f'node_{kind}[node1 * n_node + idx1]', f'node_{kind}[node1][idx1]')
    source = source.replace('node_near[i * n_node + idx]', 'node_near[i][idx]')
    marker = '    int n_near_pairs = h2m_2dbuf_ivec_get_size(near_pairs) / 2;'
    assert source.count(marker) == 1
    count = '''    for (int k=0; k<near_pairs->nrow; ++k) node_n_near[near_pairs->data_i[k]]++;
    for (int k=0; k<far_pairs->nrow; ++k) node_n_far[far_pairs->data_i[k]]++;
    for (int i=0; i<n_node; ++i) {
        node_near[i] = (int *)malloc(sizeof(int)*(node_n_near[i]+1));
        node_far[i] = (int *)malloc(sizeof(int)*node_n_far[i]);
        node_n_near[i] = node_n_far[i] = 0;
    }
'''
    source = source.replace(marker, count+marker)
    (root/path).write_text(source)
    path = 'cpp-src/h2mat/h2mat_matmul.cpp'
    source = original(path)
    import re
    for kind in ['near', 'far']:
        source = re.sub(r'int \*node_'+kind+r'\s*=', f'int **node_{kind} =', source)
        source = source.replace(f'node_{kind} + node0 * n_node', f'node_{kind}[node0]')
    (root/path).write_text(source)
    path = 'cpp-src/h2mat/h2mat_typedef.cpp'
    source = original(path).replace('    free(h2mat_->node_n_far);',
        '    for (int i=0; i<h2mat_->n_node; ++i) { free(h2mat_->node_far[i]); free(h2mat_->node_near[i]); }\n    free(h2mat_->node_n_far);')
    (root/path).write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/opt/geospsr-runtime/higp-psr')
    parser.add_argument('--blas')
    parser.add_argument('--prefix-knn', action='store_true', help='Build a separate exact predecessor-kd-tree AFN backend')
    args = parser.parse_args()
    module = 'psr_higp_kd_cext' if args.prefix_knn else 'psr_higp_cext'
    root = Path(args.root).resolve()
    if not root.exists():
        subprocess.run(['git', 'clone', 'https://github.com/huanghua1994/HiGP.git', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'checkout', '--detach', PIN], check=True)
    source = subprocess.check_output(['git', '-C', str(root), 'show', f'{PIN}:py-interface/higp_cext.c'], text=True)
    start = source.index('static PyObject* precond_setup(')
    end = source.index('static void precond_dealloc(', start)
    section = source[start:end]
    # Upstream initializes npt_s=-51 before parsing rank. Update it afterwards.
    section = section.replace('    if (seed >= 0)', '    npt_s = -rank - 1;\n\n    if (seed >= 0)', 1)
    for scalar in ['double', 'float']:
        marker = f'                    {scalar} param[4] = '
        assert section.count(marker) == 1
        prefix = ('                    octree_p spatial_tree = NULL;\n'
                  f'                    {scalar} leaf_size = 0;\n'
                  '                    octree_build(n, dim, val_type, dataset, 400, &leaf_size, &spatial_tree);\n')
        section = section.replace(marker, prefix + marker)
    old = '                        NULL, need_grad, &ap\n                    );'
    assert section.count(old) == 2
    section = section.replace(old, '                        spatial_tree, need_grad, &ap\n                    );\n                    octree_free(&spatial_tree);')
    if args.prefix_knn:
        section = section.replace('int need_grad = 1;', 'int need_grad = 0;')
    source = source[:start] + section + source[end:]
    # Stand-alone prediction does not use kernel hyperparameter derivatives.
    source = source.replace('NULL, kernel_type, 1, (void *) &h2_tol',
                            'NULL, kernel_type, 2, (void *) &h2_tol')
    marker = 'static PyObject* KrnlMatObject_matmul_grad(KrnlMatObject* self, PyObject* args, PyObject *kwds)\n{'
    assert marker in source
    source = source.replace(marker, marker+'''\n    if (self->_h2 && !((ss_h2mat_p)self->_mat)->dKdl_h2mat) {
        PyErr_SetString(PyExc_NotImplementedError, "This prediction-only H2 object has no derivative hierarchy");
        return NULL;
    }
''')
    if args.prefix_knn:
        # The exact predecessor tree builds itself; avoid the unused coarse octree.
        source = source.replace('octree_build(n, dim, val_type, dataset, 400, &leaf_size, &spatial_tree);', '')
    source = source.replace('PyInit_higp_cext(void)', f'PyInit_{module}(void)')
    snapshot = Path(__file__).resolve().parents[1]/'geospsr/native/higp_snapshot.inc'
    source = source.replace('#include "omp.h"', '#include "omp.h"\n'+snapshot.read_text())
    (root/f'py-interface/{module}.c').write_text(source)
    header = subprocess.check_output(['git', '-C', str(root), 'show', f'{PIN}:py-interface/higp_cext.h'], text=True)
    marker = 'static PyMethodDef KrnlMatObject_methods[] = \n{'
    assert marker in header
    header = header.replace(marker,
        'static PyObject* KrnlMatObject_snapshot(KrnlMatObject*, PyObject*);\n'+marker+
        '\n    {"snapshot", (PyCFunction)KrnlMatObject_snapshot, METH_NOARGS, "Copy the H2 hierarchy."},')
    marker = 'static PyMethodDef PrecondObject_methods[] = \n{'
    assert marker in header
    header = header.replace(marker,
        'static PyObject* PrecondObject_snapshot(PrecondObject*, PyObject*);\n'+marker+
        '\n    {"snapshot", (PyCFunction)PrecondObject_snapshot, METH_NOARGS, "Copy the AFN factors."},')
    (root/'py-interface/higp_cext.h').write_text(header)
    patch_adjacency(root)
    path = 'cpp-src/h2mat/ss_h2mat.cpp'
    source = subprocess.check_output(['git', '-C', str(root), 'show', f'{PIN}:{path}'], text=True)
    source = source.replace('if (algo == 1)', 'if (algo == 1 || algo == 2)')
    source = source.replace('        h2m_octree_proxy_points(octree, gkrnl', '        if (algo == 1) h2m_octree_proxy_points(octree, gkrnl')
    source = source.replace('        h2mat_build_with_proxy_points(octree, dKdl_pp', '        if (algo == 1) h2mat_build_with_proxy_points(octree, dKdl_pp')
    source = source.replace('            h2m_2dbuf_free(&dKdl_pp[i]);', '            if (dKdl_pp) h2m_2dbuf_free(&dKdl_pp[i]);')
    (root/path).write_text(source)
    path = 'cpp-src/solvers/fsai_precond.cpp'
    source = subprocess.check_output(['git', '-C', str(root), 'show', f'{PIN}:{path}'], text=True)
    if args.prefix_knn:
        helper = Path(__file__).resolve().parents[1]/'geospsr/native/prefix_knn.inc'
        marker = 'template<typename VT>\nstatic void fsai_exact_knn('
        source = source.replace(marker, helper.read_text()+'\n'+marker, 1)
        start = source.index(marker)
        begin = source.index('{', start)
        end = source.index('\n}\n', begin)
        source = source[:begin]+'''{
    ASSERT_PRINTF(pt_dim == 3, "PSR predecessor tree requires three dimensions\\n");
    psr_prefix_knn<VT>(fsai_npt, n, coord, ldc, nn_idx, nn_cnt);
'''+source[end:]
    if args.prefix_knn:
        # malloc(0) can return a non-null pointer; the upstream destructor then
        # wrongly traverses three unallocated derivative matrices.
        source = source.replace('dG  = (csr_mat_p *) malloc(sizeof(csr_mat_p) * n_grad);',
                                'if (n_grad) dG  = (csr_mat_p *) malloc(sizeof(csr_mat_p) * n_grad);')
        source = source.replace('dGT = (csr_mat_p *) malloc(sizeof(csr_mat_p) * n_grad);',
                                'if (n_grad) dGT = (csr_mat_p *) malloc(sizeof(csr_mat_p) * n_grad);')
    (root/path).write_text(source)
    site = Path(sysconfig.get_paths()['purelib'])
    if args.blas:
        blas = Path(args.blas).resolve()
    else:
        candidates = list(site.glob('scs.libs/libopenblas*.so'))
        if len(candidates) != 1:
            raise RuntimeError('Install scs or specify --blas with an LP64 OpenBLAS library')
        blas = candidates[0]
    from setuptools import Extension, setup
    import numpy
    sources = sorted(root.glob('cpp-src/**/*.cpp'))
    sources += [root/'cpp-src/utils.c', root/f'py-interface/{module}.c']
    extension = Extension(module, sources=[str(p) for p in sources],
                          include_dirs=[numpy.get_include()], language='c++',
                          extra_compile_args=['-O3', '-std=c++14', '-fopenmp', '-march=native',
                                              '-DUSE_OPENBLAS_LP64', '-Wno-unused-function', '-Wno-cpp'],
                          extra_objects=[str(blas)],
                          extra_link_args=['-fopenmp', '-lm', f'-Wl,-rpath,{blas.parent}'])
    os.chdir(root)
    setup(name='psr-higp-cext', version='0.1', ext_modules=[extension],
          script_args=['build_ext', '--parallel', '4', '--build-lib', str(site)])
    (site/('psr_higp_kd_build.json' if args.prefix_knn else 'psr_higp_build.json')).write_text(json.dumps(dict(
        upstream='https://github.com/huanghua1994/HiGP', commit=PIN, prefix_knn=args.prefix_knn,
        changes=['AFN rank selection after argument parsing',
                 ('exact predecessor kd-tree search; unused standalone AFN derivatives omitted' if args.prefix_knn
                  else 'octree neighbor search in AFN setup'),
                 'linear adjacency list storage', 'owned H2 and AFN snapshot export',
                 'prediction-only H2 setup omits unused derivative hierarchy'],
        blas=str(blas), source_root=str(root)), indent=2)+'\n')


if __name__ == '__main__':
    main()
