"""Independent algebra check for unequal prior and conditioning bands."""
import numpy as np


def test_noncommuting_hybrid_covariance_and_omitted_tail_bound():
    rng = np.random.default_rng(953)
    for _ in range(10):
        factors = [rng.normal(size=(13,13)) for _ in range(3)]
        b = factors[0] @ factors[0].T + np.eye(13)
        d = .15 * factors[1] @ factors[1].T
        e = .02 * factors[2] @ factors[2].T
        c = rng.normal(size=(13,5))
        a, z = np.linalg.solve(b,c), np.linalg.solve(b+d,c)
        prior = np.eye(5)*(1+np.linalg.norm(c.T@a))
        vb = prior-c.T@a
        vp = prior-c.T@z
        vf = prior-c.T@np.linalg.solve(b+d+e,c)
        # Covariance of f_prior - C.T B^-1(vector_prior + noise).
        hybrid = prior-c.T@a-a.T@c+a.T@(b+d)@a
        np.testing.assert_allclose(hybrid,vb+a.T@d@a,atol=1e-12,rtol=1e-12)
        # Cholesky whitening avoids assuming a common eigenbasis for B,D.
        lower = np.linalg.cholesky(b)
        x = np.linalg.solve(lower,np.linalg.solve(lower,d).T).T
        q = np.linalg.solve(lower,c)
        second_order = q.T @ x @ x @ np.linalg.solve(np.eye(13)+x,q)
        np.testing.assert_allclose(hybrid-vp,second_order,atol=1e-12,rtol=1e-11)
        assert np.linalg.eigvalsh(second_order).min() >= -1e-12
        error = hybrid-vf
        assert np.linalg.eigvalsh(second_order-error).min() >= -1e-12
        assert np.linalg.eigvalsh(error-second_order+z.T@e@z).min() >= -1e-12


def test_mean_outer_product_tail_bound_with_centered_queries():
    rng = np.random.default_rng(712)
    for _ in range(10):
        f,g = rng.normal(size=(17,17)),rng.normal(size=(17,17))
        b,d = f@f.T+np.eye(17),.08*g@g.T
        y = rng.normal(size=(17,3))
        cross = rng.normal(size=(3,17,7))
        alpha = np.linalg.solve(b,y)
        alpha_full = np.linalg.solve(b+d,y)
        data_quadratic = np.sum(alpha*(d@alpha))
        for centered in [False,True]:
            # A fixed linear functional determines the deterministic offset.
            c = cross-cross[:,:,:2].mean(2,keepdims=True) if centered else cross
            h = np.zeros((7,7)); difference = np.zeros(7)
            for component in range(3):
                a = np.linalg.solve(b,c[component])
                h += a.T@d@a
                difference += c[component].T@(alpha[:,component]-alpha_full[:,component])
            remainder = data_quadratic*h-np.outer(difference,difference)
            assert np.linalg.eigvalsh(remainder).min() >= -1e-12*np.linalg.norm(remainder)
