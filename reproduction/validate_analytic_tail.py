"""Replay retained tail-root draws against independent CPU dense kernels."""
import json
from pathlib import Path
import numpy as np
from geospsr.inverse_root import periodic_matern_axis
from geospsr.spectral import spectral_axis
from reproduction.common_rhs_integrity import array_digest


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'reproduction/results/analytic_tail'
    record=json.loads((out/'results.json').read_text())
    assert record['complete'] and record['gpu_process_audit']['sampled_isolation_pass']
    x=np.load(root/'reproduction/results/matched/A/input.npz')['points']
    result={'passed':False,'scope':'retained GPU draw replay against independent CPU analytic and trigonometric kernels; no timing replay','cases':{}}
    for label,v in record['cases'].items():
        assert array_digest(x)==v['input_sha256']
        saved=np.load(out/f'{label}_controls.npz')
        distance=np.remainder(np.abs(x.astype('d')[:,None]-x.astype('d')[None]),2*np.pi)
        distance=np.minimum(distance,2*np.pi-distance)
        analytic=periodic_matern_axis(distance,v['lengthscale'],v['variance_per_axis']).prod(2)
        modes,rho=spectral_axis(v['cutoff'],v['lengthscale'],v['variance_per_axis'])
        finite=np.ones_like(analytic)
        for d in range(3):
            delta=x.astype('d')[:,None,d]-x.astype('d')[None,:,d]
            # Direct trigonometric sums, independent of GPU cosine recurrences.
            finite*=sum(weight*np.cos(mode*delta) for mode,weight in zip(modes,rho))
        target=analytic-finite+v['noise_variance']*np.eye(len(x))
        eig,vec=np.linalg.eigh(target)
        np.testing.assert_allclose(eig,saved['eigenvalues'],rtol=1e-10,atol=1e-13)
        expected=(vec*np.sqrt(eig))@(vec.T@saved['white_noise'])
        error=float(np.linalg.norm(saved['actual']-expected)/np.linalg.norm(expected))
        reference_error=float(np.linalg.norm(saved['reference']-expected)/np.linalg.norm(expected))
        lower,upper=v['bound']['lower'],v['bound']['upper']
        assert eig.min()>=lower*(1-1e-8) and eig.max()<=upper*(1+1e-10)
        mapped=(eig-(upper+lower)/2)/((upper-lower)/2)
        p=np.polynomial.chebyshev.chebval(mapped,saved['coefficients'])
        covariance_error=float(np.max(np.abs(p*p/eig-1)))
        assert error<=record['gates']['paired_root_relative_l2']
        assert reference_error<=1e-7
        assert covariance_error<=record['gates']['dense_polynomial_whitened_covariance_operator_error']
        result['cases'][label]={'retained_gpu_draw_relative_l2':error,
                               'retained_gpu_reference_relative_l2':reference_error,
                               'dense_polynomial_whitened_covariance_operator_error':covariance_error}
        print(label,result['cases'][label],flush=True)
    result['passed']=True
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Three scorpion tail-root cases replayed against independent CPU kernels.')


if __name__=='__main__':main()
