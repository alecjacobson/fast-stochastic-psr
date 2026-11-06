import numpy as np
from geospsr.nonperiodic import RegionalVarianceCache
from geospsr.regional_field import _coefficients
from reproduction.noise_selection import surface_points,surface_metrics,HALFWIDTH,validation_observations
from reproduction.benchmark_hierarchical_variance import data


def test_known_ellipsoid_roots_and_normals():
    cache=RegionalVarianceCache.__new__(RegionalVarianceCache);cache._initialize_box([1,0,0],HALFWIDTH,4)
    q=cache.nodes;values=q[:,0]**2+(q[:,1]/.8)**2+(q[:,2]/.6)**2-1
    coefficients=_coefficients(values.reshape((4,)*3))
    metrics=surface_metrics(coefficients,surface_points(123))
    assert metrics['missing_roots']==0
    assert metrics['surface_rms']<1e-10
    np.testing.assert_allclose(metrics['normal_alignment'],1.,atol=1e-12)


def test_noise_matches_unrenormalized_component_likelihood_and_nested_data():
    x,noisy,_=data(100000,44,'ellipsoid',normal_noise=.2)
    xc,clean,_=data(100000,44,'ellipsoid')
    np.testing.assert_array_equal(x,xc)
    np.testing.assert_allclose(np.std(noisy-clean,axis=0),.2,rtol=.01)
    xs,ys,_=data(128,44,'ellipsoid',normal_noise=.2)
    np.testing.assert_array_equal(xs,x[:128]);np.testing.assert_array_equal(ys,noisy[:128])
    replicas=validation_observations(clean[:128],.2,44,2)
    np.testing.assert_array_equal(replicas[0],noisy[:128])
    assert not np.array_equal(replicas[0],replicas[1])
    np.testing.assert_array_equal(replicas[-1],clean[:128])


def test_missing_roots_are_penalized_in_selection():
    coefficients=np.zeros((3,3,3));coefficients[0,1,0]=1
    result=surface_metrics(coefficients,surface_points(8))
    assert result['missing_roots']==result['count']
    np.testing.assert_allclose(result['penalized_surface_rms'],2*HALFWIDTH)


def test_shared_mean_solve_matches_independent_dense_posteriors():
    from geospsr.nonperiodic import FreeSpacePoissonMatern,DenseFreeSpacePosterior
    from reproduction.noise_selection import build_mean_fields,mean_values
    x,clean,_=data(31,12,'ellipsoid')
    observations=validation_observations(clean,.2,12,2)
    model=FreeSpacePoissonMatern(.48,1/3)
    matrix=model.vector_covariance(x,x)+.04*np.eye(len(x))
    fields,_=build_mean_fields(lambda rhs:np.linalg.solve(matrix,rhs),model,x,observations)
    points=surface_points(619,count=19)
    for coefficients,y in zip(fields,observations):
        reference=DenseFreeSpacePosterior(model,x,y,.04).predict(points)[0]
        np.testing.assert_allclose(mean_values(coefficients,points),reference,rtol=1e-10,atol=1e-11)


def test_numerical_geometry_check_respects_contact_anchor(tmp_path):
    import json
    from reproduction.run_noise_comparison import mean_geometry_check
    q=np.array([[1-HALFWIDTH,0,0],[1+HALFWIDTH,0,0],[1,0,0]])
    ref=tmp_path/'reference.json'
    ref.write_text(json.dumps(dict(queries=q.tolist(),mean=(17+q[:,0]-1).tolist(),mean_error_bound=[0,0,0])))
    coefficients=np.zeros((2,2,2));coefficients[1,0,0]=HALFWIDTH
    field=dict(center=[1,0,0],halfwidth=[HALFWIDTH]*3,order=2,prior_variance=1,
               mean=coefficients.tolist(),correction=np.zeros_like(coefficients).tolist())
    record=dict(field=field,independent_audit=dict(reference=str(ref)))
    cell=dict(validation_rms=1e-5)
    # A constant mean offset cancels at the contact, while a gradient error does not.
    assert mean_geometry_check(record,cell)['passes']
    coefficients[1,0,0]*=1.03;field['mean']=coefficients.tolist()
    assert not mean_geometry_check(record,cell)['passes']
