import hashlib
import json
import numpy as np
import pytest
from reproduction.common_rhs_integrity import array_digest,validate_source
from reproduction.coefficient_archive import digest


def data(tmp_path):
    raw=np.arange(24,dtype='d').reshape(2,4,3)+2**-30
    arrays={'points':np.zeros((4,3),dtype='f'),'normals':np.ones((4,3),dtype='f'),
            'all_rhs':raw.astype('f'),'all_prior':np.ones((2,3,3,3),dtype='F')}
    names={'points':'points.npy','normals':'normals.npy','all_rhs':'rhs.npy','all_prior':'prior.npy'}
    for k,a in arrays.items():np.save(tmp_path/names[k],a)
    return raw,arrays


def test_serialized_hash_detects_data_and_shape_changes(tmp_path):
    raw,arrays=data(tmp_path)
    hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
    assert hashes['all_rhs']!=hashlib.sha256(raw.tobytes()).hexdigest()
    manifest={'complete':True,'args':{'samples':2},'array_hash_schema':2,'hashes':hashes}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    checked=validate_source(tmp_path,manifest)
    assert checked['actual_array_sha256']==hashes and not checked['legacy_cast_audit']
    for k,a in arrays.items():assert array_digest(a)==hashes[k]
    np.save(tmp_path/'rhs.npy',arrays['all_rhs'].reshape(2,3,4))
    with pytest.raises(ValueError,match='shapes'):validate_source(tmp_path,manifest)
    np.save(tmp_path/'rhs.npy',arrays['all_rhs'])
    changed=arrays['all_prior'].copy();changed[0,0,0,0]+=1
    np.save(tmp_path/'prior.npy',changed)
    with pytest.raises(ValueError,match='hash mismatch'):validate_source(tmp_path,manifest)


def test_legacy_requires_matching_regeneration_audit(tmp_path):
    raw,arrays=data(tmp_path)
    hashes={k:array_digest(v) for k,v in arrays.items() if k!='all_prior'}
    hashes['all_rhs']=hashlib.sha256(raw.tobytes()).hexdigest()
    manifest={'complete':True,'args':{'samples':2},'hashes':hashes}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='requires audit'):validate_source(tmp_path,manifest)
    audit={'complete':True,'source_manifest_sha256':digest(tmp_path/'manifest.json'),
           'source_hashes':{k:hashes[k] for k in ['points','normals']},
           'raw_rhs_sha256':hashes['all_rhs'],'stored_rhs_sha256':array_digest(arrays['all_rhs']),
           'prior_sha256':array_digest(arrays['all_prior']),'raw_rhs_dtype':'<f8','stored_rhs_dtype':'<f4',
           'draws':[{'index':i,'rhs_exact_after_storage_cast':True,'prior_exact':True} for i in range(2)]}
    (tmp_path/'storage_audit.json').write_text(json.dumps(audit))
    assert validate_source(tmp_path,manifest)['legacy_cast_audit']
    audit['draws'][1]['prior_exact']=False
    (tmp_path/'storage_audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='audit does not match'):validate_source(tmp_path,manifest)
