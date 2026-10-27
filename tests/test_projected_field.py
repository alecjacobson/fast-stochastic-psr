import numpy as np
import pytest
from geospsr.projected_field import CompressedJointField


def fixture(rank=7,modes=41):
    rng=np.random.default_rng(5403)
    s=np.exp(rng.uniform(-3,1,modes));s[0]=0
    l=(rng.normal(size=(rank,modes))+1j*rng.normal(size=(rank,modes)))*np.sqrt(s)*.1
    b=(rng.normal(size=(rank,modes))+1j*rng.normal(size=(rank,modes)))*.05;b[:,0]=0
    c=l/np.where(s>0,np.sqrt(s),1)
    a=rng.normal(size=(rank,rank));h=a@a.T*.01
    g=(c@c.conj().T).real+h
    return s,l,b,g


@pytest.mark.parametrize('chunk',[1,13,100])
def test_covariance_and_coherent_new_query_evaluation(chunk):
    s,l,b,g=fixture();model=CompressedJointField(s,l,b,g,frequency_chunk=chunk)
    rng=np.random.default_rng(5404);freq=rng.integers(-4,5,(len(s),3));freq[0]=0
    query=rng.normal(size=(8,3));query[1]=query[0]
    phase=np.exp(1j*(query@freq.T))
    k=(phase*s)@phase.conj().T;k=k.real
    cross=(phase@l.T).real;lift=(phase@b.T).real
    target=k-cross@lift.T-lift@cross.T+lift@g@lift.T
    coeff=model.draw(32768,seed=5405,max_draw_bytes=100*2**20)
    np.testing.assert_array_equal(coeff[:,0],0)
    fields=(coeff@phase.T).real
    se=np.sqrt(np.diag(target)/len(fields))
    cse=np.sqrt((target**2+np.diag(target)[:,None]*np.diag(target)[None,:])/(len(fields)-1))
    assert np.max(np.abs(fields.mean(0))/se)<6
    assert np.max(np.abs(np.cov(fields.T)-target)/cse)<6
    # A later query evaluation uses the same coefficients, not another draw.
    split=np.concatenate([(coeff@phase[:3].T).real,(coeff@phase[3:].T).real],axis=1)
    np.testing.assert_allclose(split,fields,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(fields[:,0],fields[:,1])


def test_supplied_white_immutability_chunk_invariance_and_cache_ownership():
    s,l,b,g=fixture();rng=np.random.default_rng(5406)
    white=rng.normal(size=(17,len(s)))-1j*rng.normal(size=(17,len(s)))
    other=rng.normal(size=(17,len(g)));old=white.copy();old_other=other.copy()
    first=CompressedJointField(s,l,b,g,frequency_chunk=5)
    second=CompressedJointField(s,l,b,g,frequency_chunk=100)
    a=first.draw(17,white=white,extra_white=other);actual=second.draw(17,white=white,extra_white=other)
    np.testing.assert_allclose(a,actual,rtol=1e-12,atol=1e-12)
    s[:]=10;l[:]=10;b[:]=10;g[:]=10
    np.testing.assert_array_equal(a,first.draw(17,white=white,extra_white=other))
    np.testing.assert_array_equal(white,old);np.testing.assert_array_equal(other,old_other)


def test_zero_rank_and_degenerate_compressed_remainder():
    s,l,b,g=fixture(rank=0);model=CompressedJointField(s,l,b,g)
    white=np.ones((3,len(s)),dtype='complex128')*(1-2j)
    np.testing.assert_array_equal(model.draw(3,white=white,extra_white=np.empty((3,0))),white*np.sqrt(s))
    s,l,b,g=fixture();c=l/np.where(s>0,np.sqrt(s),1);gram=(c@c.conj().T).real
    model=CompressedJointField(s,l,b,gram)
    assert model.record['remainder_root_covariance_frobenius_error']<=model.record['remainder_roundoff_allowance']*len(g)
    with pytest.raises(ValueError):CompressedJointField(s,l,b,gram-.1*np.eye(len(g)))


def test_invalid_inputs_and_explicit_memory_caps():
    s,l,b,g=fixture()
    with pytest.raises(MemoryError):CompressedJointField(s,l,b,g,max_cache_bytes=1)
    with pytest.raises(ValueError):CompressedJointField(s.astype('f'),l,b,g)
    with pytest.raises(ValueError):CompressedJointField(s,l.astype('complex64'),b,g)
    with pytest.raises(ValueError):CompressedJointField(-s,l,b,g)
    with pytest.raises(ValueError):CompressedJointField(s,l[:,:-1],b,g)
    invalid=l.copy();invalid[:,0]=1
    with pytest.raises(ValueError):CompressedJointField(s,invalid,b,g)
    invalid=g.copy();invalid[0,1]+=1
    with pytest.raises(ValueError):CompressedJointField(s,l,b,invalid)
    model=CompressedJointField(s,l,b,g)
    with pytest.raises(MemoryError):model.draw(2,max_draw_bytes=1)
    with pytest.raises(ValueError):model.draw(0)
    with pytest.raises(ValueError):model.draw(2,white=np.ones((2,len(s))))
    with pytest.raises(ValueError):model.draw(2,white=np.ones((2,len(s))),extra_white=np.ones((2,len(g))))
