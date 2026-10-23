"""Check serialized common-RHS arrays, including audited legacy casts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from reproduction.coefficient_archive import digest


def array_digest(array):
    """C-order array bytes, bounded by roughly 4MiB or one outer slice."""
    step=max(1,4*1024**2//max(1,int(np.prod(array.shape[1:]))*array.dtype.itemsize))
    checksum=hashlib.sha256()
    for start in range(0,len(array),step):checksum.update(array[start:start+step].tobytes(order='C'))
    return checksum.hexdigest()


def validate_source(scratch,manifest):
    scratch=Path(scratch)
    arrays={key:np.load(scratch/name,mmap_mode='r',allow_pickle=False) for key,name in
            [('points','points.npy'),('normals','normals.npy'),('all_rhs','rhs.npy'),('all_prior','prior.npy')]}
    x,y,rhs,prior=[arrays[k] for k in ['points','normals','all_rhs','all_prior']]
    if (x.ndim!=2 or x.shape[1]!=3 or y.shape!=x.shape or rhs.shape!=(manifest['args']['samples'],len(x),3)
            or prior.ndim!=4 or len(set(prior.shape[1:]))!=1 or prior.shape[-1]%2!=1
            or any(v.dtype!=np.dtype('float32') for v in [x,y,rhs]) or prior.dtype!=np.dtype('complex64')):
        raise ValueError('invalid common-RHS array shapes or dtypes')
    actual={key:array_digest(value) for key,value in arrays.items()}
    if not manifest['complete'] or len(arrays['all_rhs'])!=manifest['args']['samples'] or len(arrays['all_prior'])!=len(arrays['all_rhs']):
        raise ValueError('incomplete common-RHS source')
    if any(actual[k]!=manifest['hashes'][k] for k in ['points','normals']):
        raise ValueError('source observation hash mismatch')
    if manifest.get('array_hash_schema')==2:
        if actual!=manifest['hashes']:raise ValueError('serialized source array hash mismatch')
        return {'actual_array_sha256':actual,'array_hash_schema':2,'legacy_cast_audit':False}
    # Legacy hashes may describe promoted float64 RHS before float32 storage.
    # Require a read-only regeneration audit, never silently rewrite history.
    path=scratch/'storage_audit.json'
    if not path.exists():raise ValueError('legacy source requires audit_common_rhs_storage output at scratch/storage_audit.json')
    audit=json.loads(path.read_text())
    if (not audit.get('complete') or audit['source_manifest_sha256']!=digest(scratch/'manifest.json')
            or audit['source_hashes']!={k:actual[k] for k in ['points','normals']}
            or audit['raw_rhs_sha256']!=manifest['hashes']['all_rhs']
            or audit['stored_rhs_sha256']!=actual['all_rhs'] or audit['prior_sha256']!=actual['all_prior']
            or len(audit['draws'])!=len(arrays['all_rhs'])
            or [d['index'] for d in audit['draws']]!=list(range(len(arrays['all_rhs'])))
            or not all(d['rhs_exact_after_storage_cast'] and d['prior_exact'] for d in audit['draws'])):
        raise ValueError('legacy storage audit does not match source arrays')
    return {'actual_array_sha256':actual,'array_hash_schema':1,'legacy_cast_audit':True,
            'storage_audit_sha256':digest(path),'raw_rhs_dtype':audit['raw_rhs_dtype'],'stored_rhs_dtype':audit['stored_rhs_dtype']}
