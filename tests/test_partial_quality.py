import numpy as np
import pytest
from scipy.linalg import block_diag

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_quality import DenseContactFamily, roots


def test_nested_contacts_against_augmented_gaussian():
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('GPU required')
    rng = np.random.default_rng(124)
    x, y = rng.normal(size=(48, 3)), rng.normal(size=(48, 3))
    contacts, q = rng.normal(size=(12, 3)), rng.normal(size=(11, 3))
    model = FreeSpacePoissonMatern(.48, 1/3)
    counts = [4, 8, 12]
    family = DenseContactFamily(model, x, y, contacts, .04, counts)
    mean, variance = family.predict(q, batch=5)
    a = model.vector_covariance(x, x)+.04*np.eye(len(x))
    for k, c in enumerate(counts):
        cc = model.cross_covariance(contacts[:c], x).transpose(0, 2, 1).reshape(c, -1)
        joint = np.block([[block_diag(a, a, a), cc.T],
                         [cc, model.scalar_covariance(contacts[:c], contacts[:c])+.0001*np.eye(c)]])
        cross = np.column_stack((model.cross_covariance(q, x).transpose(0, 2, 1).reshape(len(q), -1),
                                 model.scalar_covariance(q, contacts[:c])))
        observations = np.r_[y.T.ravel(), np.zeros(c)]
        truth_mean = cross@np.linalg.solve(joint, observations)
        truth_var = model.prior_variance-np.sum(cross*np.linalg.solve(joint, cross.T).T, axis=1)
        np.testing.assert_allclose(mean[:, k], truth_mean, atol=2e-13, rtol=0)
        np.testing.assert_allclose(variance[:, k], truth_var, atol=2e-13, rtol=0)
    np.testing.assert_allclose(family.mean(q, batch=7), mean, atol=2e-13, rtol=0)


def test_root_error_retains_censored_missing_roots():
    offsets = np.linspace(-.15, .15, 33)
    values = np.array([offsets-.023, offsets+.075, offsets*0+1])
    panel = dict(offsets=offsets, surface=np.zeros((3, 3)),
                 surface_observed=np.array([True, True, False]),
                 surface_occluded=np.array([False, False, True]))
    stats, errors, missing = roots(values, panel)
    np.testing.assert_allclose(errors, [.023, .075, .15], atol=1e-14)
    np.testing.assert_array_equal(missing, [False, False, True])
    assert stats['occluded']['missing_fraction'] == 1.
    with pytest.raises(ValueError, match='nonfinite'):
        roots(values*np.nan, panel)


def test_refined_full_kernel_against_float64_posterior():
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('GPU required')
    from reproduction.partial_quality_large import RefinedDenseSystem
    from reproduction.partial_direct_gpu import DirectContactQueries
    from geospsr.nonperiodic import DenseFreeSpacePosterior
    from geospsr.scalar_contacts import ContactConditionedDensePosterior
    rng = np.random.default_rng(932)
    x, y = rng.normal(size=(80, 3)), rng.normal(size=(80, 3))
    contacts, q = rng.normal(size=(12, 3)), rng.normal(size=(13, 3))
    model = FreeSpacePoissonMatern(.48, 1/3)
    system = RefinedDenseSystem(model, x, .04, block=8)
    posterior = DirectContactQueries(model, x, y, contacts, system)
    truth = ContactConditionedDensePosterior(DenseFreeSpacePosterior(model, x, y, .04), contacts, np.zeros(12), .0001)
    np.testing.assert_allclose(posterior.predict(q, batch_size=5), truth.predict(q), atol=1e-10, rtol=0)
    assert all(h[-1]['relative_residual'] <= 1e-10 for h in system.histories)


@pytest.mark.parametrize('method', ['plain', 'plain-assembled64', 'corrected', 'corrected-wide'])
def test_dense_quality_competitors_against_augmented_gaussian(method):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('GPU required')
    from reproduction.benchmark_partial_quality import dense_moments
    from geospsr.nonperiodic import DenseFreeSpacePosterior
    from geospsr.scalar_contacts import ContactConditionedDensePosterior
    rng = np.random.default_rng(382)
    x, y = rng.normal(size=(48, 3)), rng.normal(size=(48, 3))
    c, q = rng.normal(size=(12, 3)), rng.normal(size=(13, 3))
    model = FreeSpacePoissonMatern(.48, 1/3)
    m,v,_ = dense_moments(model,x,y,c,q,.25,method)
    tm,tv = ContactConditionedDensePosterior(DenseFreeSpacePosterior(model,x,y,.25),c,np.zeros(12),.0001).predict(q)
    np.testing.assert_allclose(m,tm,atol=1e-7 if method.startswith('plain') else 1e-12,rtol=0)
    np.testing.assert_allclose(v,tv,atol=1e-8 if method.startswith('plain') else 1e-12,rtol=0)


def test_joint_hierarchy_contacts_against_augmented_gaussian():
    pytest.importorskip('psr_higp_kd_cext')
    cp = pytest.importorskip('cupy')
    from geospsr.hierarchical import HierarchicalMaternSystem
    from geospsr.hierarchical_gpu import GPUHierarchicalSystem
    from geospsr.nonperiodic import DenseFreeSpacePosterior
    from geospsr.scalar_contacts import ContactConditionedDensePosterior
    from reproduction.benchmark_partial_quality import h2_joint_moments
    rng=np.random.default_rng(524)
    x,y=rng.normal(size=(160,3))*.3,rng.normal(size=(160,3))
    c,q=rng.normal(size=(28,3))*.3,rng.normal(size=(15,3))*.3
    model=FreeSpacePoissonMatern(.48,1/3)
    native=HierarchicalMaternSystem(model,x,.25,tolerance=1e-9,leaf_size=20,
        preconditioner_rank=30,neighbors=20,threads=4,seed=9101,preconditioner_backend='prefix-kd')
    system=GPUHierarchicalSystem(native,dtype='float32',rhs_block_size=72,release_native=True)
    mean,var,info=h2_joint_moments(system,model,x,y,c,q,20,block=72)
    expected=ContactConditionedDensePosterior(DenseFreeSpacePosterior(model,x,y,.25),c,np.zeros(len(c)),.0001).predict(q)
    np.testing.assert_allclose((mean,var),expected,atol=1e-8,rtol=0)
    assert len(info['cross_solves'])==2  # Exercises the final short solve block and component boundaries.
    assert all(i['residual_operator_dtype']=='float64' for i in info['cross_solves'])


def test_double_assembly_preserves_small_separations_with_float_factor():
    cp=pytest.importorskip('cupy')
    from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
    rng=np.random.default_rng(1234)
    x=1e7+rng.normal(size=(24,3))*.1
    rhs=rng.normal(size=(24,2));model=FreeSpacePoissonMatern(.3,1/3)
    system=GPUDenseMaternSystem(model,x,.25,dtype='float32',kernel_dtype='float64')
    z=system.solve(rhs)
    a=model.vector_covariance(x,x)+.25*np.eye(len(x))
    assert np.linalg.norm(a@z-rhs)/np.linalg.norm(rhs)<2e-6
    assert system.factor.dtype==np.float32


@pytest.mark.parametrize('cached', [True, False])
def test_dense_cg_shared_kernel_compatibility(cached):
    pytest.importorskip('psr_higp_kd_cext')
    cp=pytest.importorskip('cupy')
    import geospsr.nonperiodic_dense_cg as dense_cg
    rng=np.random.default_rng(391)
    x,b=rng.normal(size=(48,3)),rng.normal(size=(48,3))
    model=FreeSpacePoissonMatern(.48,1/3)
    system=dense_cg.GPUDenseCGSystem(model,x,.25,dtype='float64',rank=10,neighbors=8,
        threads=2,cache=cached,block_size=11,preconditioner_backend='prefix-kd')
    expected=(model.vector_covariance(x,x)+.25*np.eye(len(x)))@b
    np.testing.assert_allclose(cp.asnumpy(system.matmul(b)),expected,atol=1e-13,rtol=0)
