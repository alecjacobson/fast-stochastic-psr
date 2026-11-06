"""Strong inexpensive probability controls, fitted only on validation labels."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr

from geospsr.scalar_contacts import ScalarContactUpdate
from reproduction.partial_occupancy import ROOT, exact, load, stratified
from reproduction.stanford_partial_data import digest


def contact_only_variance(model, contacts, queries):
    update = ScalarContactUpdate(np.zeros(len(contacts)), model.scalar_covariance(contacts, contacts),
                                 np.zeros(len(contacts)), .01**2)
    return update.apply(np.zeros(len(queries)), np.full(len(queries), model.prior_variance),
                        model.scalar_covariance(queries, contacts))[1]


def fit_probit(mean, variance, inside):
    z = -mean/np.sqrt(variance)
    def loss(theta):
        value = np.exp(theta[0])*z+theta[1]
        return -np.where(inside, log_ndtr(value), log_ndtr(-value)).mean()
    fit = minimize(loss, [0., 0.], method='L-BFGS-B', bounds=[(-8, 8), (-8, 8)])
    if not fit.success: raise RuntimeError(fit.message)
    return dict(log_scale=float(fit.x[0]), bias=float(fit.x[1]), validation_log_loss=float(fit.fun))


def calibrated(mean, variance, fit, panel):
    z = -mean/np.sqrt(variance)
    return stratified(-(np.exp(fit['log_scale'])*z+fit['bias']), np.ones(len(mean)), panel)


def main():
    chosen = json.loads((ROOT/'selected.json').read_text())
    mean, variance, panel, model, _ = exact('validation', 768, chosen['ell'], chosen['sigma'], query_stride=2)
    contacts = load('validation', 768, 2)[2]
    contact_variance = contact_only_variance(model, contacts, panel['queries'])
    fits = dict(spatial=fit_probit(mean, variance, panel['inside']),
                mean_only=fit_probit(mean, np.ones(len(mean)), panel['inside']),
                contact_only=fit_probit(mean, contact_variance, panel['inside']))
    result = dict(fits=fits, fit_scope='two-parameter monotone probit calibration, validation group only at N=768',
                  source_sha256={p: digest(p) for p in [__file__, 'reproduction/partial_occupancy.py']}, tests=[])
    for path in sorted((ROOT/'exact').glob('*.json')):
        record = json.loads(path.read_text()); group, n = record['group'], record['n']
        _, _, contacts, panel = load(group, n)
        field = np.load(path.with_suffix('.npz')); m, v = field['mean'], field['variance']
        cv = contact_only_variance(model, contacts, panel['queries'])
        one = dict(group=group, n=n, contact_only_variance=stratified(m, cv, panel),
                   calibrated_spatial=calibrated(m, v, fits['spatial'], panel),
                   calibrated_mean_only=calibrated(m, np.ones(len(m)), fits['mean_only'], panel),
                   calibrated_contact_only=calibrated(m, cv, fits['contact_only'], panel),
                   contact_only_numerical_error=dict(
                       variance_over_prior=float(np.max(abs(cv-v))/model.prior_variance),
                       cdf=float(np.max(abs(ndtr(-m/np.sqrt(cv))-ndtr(-m/np.sqrt(v)))))),
                   boundary_sensitivity={})
        for margin in [.01, .02]:
            keep = abs(panel['distance']) > margin
            subpanel = {k:a[keep] for k,a in panel.items()}
            one['boundary_sensitivity'][str(margin)] = dict(
                posterior=stratified(m[keep], v[keep], subpanel),
                constant_same_mean=stratified(m[keep], chosen['best_constant_variance'], subpanel),
                contact_only_variance=stratified(m[keep], cv[keep], subpanel))
        result['tests'].append(one)
    (ROOT/'controls.json').write_text(json.dumps(result, indent=2)+'\n')
    for r in result['tests']:
        print(r['group'], r['n'], {k:round(r[k]['all']['brier'],5) for k in
            ['contact_only_variance','calibrated_spatial','calibrated_mean_only','calibrated_contact_only']}, flush=True)


if __name__ == '__main__': main()
