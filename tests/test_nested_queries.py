import numpy as np
import pytest
from reproduction.nested_queries import nested_rays,nested_spheres,discrepancy_bound


def test_nested_geometry_preserves_every_coarse_coordinate():
    rng=np.random.default_rng(1)
    unit=rng.normal(size=(16,3));unit/=np.linalg.norm(unit,axis=1)[:,None]
    objects=(np.array([[.1,.2,.3],[-.1,.3,-.7]])[:,None]+.12*unit).astype('f')
    spheres=nested_spheres(objects,3)
    ray=np.random.default_rng(9).normal(size=(3,5,3)).astype('f')
    rays,distance=nested_rays(ray,np.linspace(0,5,5,dtype='f'),3)
    for i in range(3):
        np.testing.assert_array_equal(spheres[i+1][:,:spheres[i].shape[1]],spheres[i])
        np.testing.assert_array_equal(rays[i+1][:,::2],rays[i])
        np.testing.assert_array_equal(distance[i+1][::2],distance[i])


def test_exact_paired_bound_uses_draws_not_spatial_locations():
    result=discrepancy_bound(0,256)
    assert result['upper']==pytest.approx(1-(.05/8)**(1/256),rel=1e-10)
    assert result['pass_2pp']
    assert not discrepancy_bound(1,256)['pass_2pp']
    assert discrepancy_bound(256,256)['upper']==1.
    with pytest.raises(ValueError): discrepancy_bound(257,256)
