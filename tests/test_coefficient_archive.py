import json
import numpy as np
import pytest
from reproduction.coefficient_archive import CoefficientArchiveWriter,CoefficientArchive


def test_archive_contiguous_partial_batch_and_validation(tmp_path):
    c=(np.arange(81).reshape(3,3,3,3)+1j).astype('complex64')
    writer=CoefficientArchiveWriter(tmp_path,3,{'offset':.3},c[:1],{'query':np.zeros((2,3))})
    with pytest.raises(ValueError,match='empty'):
        CoefficientArchiveWriter(tmp_path,3,{},c[:1],{})
    with pytest.raises(ValueError,match='incomplete'): CoefficientArchive(tmp_path)
    with pytest.raises(ValueError,match='contiguous'): writer.append(1,c[:1])
    writer.append(0,c[:2])
    with pytest.raises(ValueError,match='incomplete'): writer.finish({})
    writer.append(2,c[2:]);writer.finish({'numerical_control':True})
    reader=CoefficientArchive(tmp_path)
    np.testing.assert_array_equal(np.concatenate([b['coefficients'] for b in reader.blocks()]),c)
    np.testing.assert_array_equal(reader.mean(),c[:1])
    assert reader.geometry()['query'].shape==(2,3)
    with pytest.raises(ValueError,match='sealed'): writer.append(3,c[:1])
    with pytest.raises(ValueError,match='sealed'): writer.finish({})
    with pytest.raises(ValueError,match='unregistered'): reader.checked_path('../mean.npy')


def test_archive_detects_corruption_and_missing_coverage(tmp_path):
    c=np.ones((1,3,3,3),dtype='complex64')
    writer=CoefficientArchiveWriter(tmp_path,1,{},c,{})
    writer.append(0,c);writer.finish({})
    path=tmp_path/'00000000-00000001.npy'
    np.save(path,2*c)
    with pytest.raises(ValueError,match='hash mismatch'): list(CoefficientArchive(tmp_path).blocks())
    manifest=json.loads((tmp_path/'manifest.json').read_text());manifest['blocks'][0]['start']=1
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='coverage'): CoefficientArchive(tmp_path)
