// Owned copies of a pinned HiGP H2 representation for independent GPU actions.
static PyObject* psr_array(const void* data, int type, npy_intp rows, npy_intp cols) {
    npy_intp dims[2] = {rows, cols};
    PyObject* out = PyArray_SimpleNew(2, dims, type);
    if (!out) return NULL;
    if (rows*cols) memcpy(PyArray_DATA((PyArrayObject*)out), data,
                          rows*cols*PyArray_ITEMSIZE((PyArrayObject*)out));
    return out;
}

static void psr_set(PyObject* dict, const char* key, PyObject* value) {
    PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
}

static PyObject* KrnlMatObject_snapshot(KrnlMatObject* self, PyObject*) {
    if (!self->_h2 || self->_dtype != FP64) {
        PyErr_SetString(PyExc_ValueError, "snapshot requires a float64 H2 matrix");
        return NULL;
    }
    h2mat_p h = ((ss_h2mat_p)self->_mat)->K_h2mat;
    octree_p t = h->octree;
    int n = t->n_node;
    PyObject* out = PyDict_New();
    psr_set(out, "permutation", psr_array(t->bwd_perm, NPY_INT, 1, t->npt));
    psr_set(out, "levels", psr_array(t->node_lvl, NPY_INT, 1, n));
    psr_set(out, "clusters", psr_array(t->pt_cluster, NPY_INT, n, 2));
    psr_set(out, "min_far_level", PyLong_FromLong(h->min_far_lvl));
    PyObject* nodes = PyList_New(n);
    for (int i=0; i<n; ++i) {
        PyObject* node = PyDict_New();
        psr_set(node, "children", psr_array(t->children+i*(1<<t->pt_dim), NPY_INT, 1, t->n_children[i]));
        h2m_2dbuf_p v = h->V_mats[i];
        h2m_2dbuf_p j = h->J_idxs[i];
        // V is column-major; return its transpose as a C-contiguous array.
        psr_set(node, "basis_transpose", psr_array(v->data, NPY_DOUBLE, v->ncol, v->nrow));
        psr_set(node, "skeleton", psr_array(j->data, NPY_INT, 1, j->nrow*j->ncol));
        psr_set(node, "far", psr_array(h->node_far[i], NPY_INT, 1, h->node_n_far[i]));
        psr_set(node, "near", psr_array(h->node_near[i], NPY_INT, 1, h->node_n_near[i]));
        PyList_SET_ITEM(nodes, i, node);
    }
    psr_set(out, "nodes", nodes);
    return out;
}

static PyObject* PrecondObject_snapshot(PrecondObject* self, PyObject*) {
    afn_precond_p p = (afn_precond_p)self->_prec;
    if (self->_dtype != FP64 || p->is_nys) {
        PyErr_SetString(PyExc_ValueError, "snapshot requires float64 AFN without Nystrom fallback");
        return NULL;
    }
    PyObject* out = PyDict_New();
    psr_set(out, "permutation", psr_array(p->perm, NPY_INT, 1, p->n));
    psr_set(out, "L11_transpose", psr_array(p->L11, NPY_DOUBLE, p->n1, p->n1));
    psr_set(out, "K12_transpose", psr_array(p->K12, NPY_DOUBLE, p->n2, p->n1));
    csr_mat_p g = p->fp->G;
    psr_set(out, "G_data", psr_array(g->val, NPY_DOUBLE, 1, g->nnz));
    psr_set(out, "G_columns", psr_array(g->colidx, NPY_INT, 1, g->nnz));
    psr_set(out, "G_indptr", psr_array(g->rowptr, NPY_INT, 1, g->nrow+1));
    return out;
}
