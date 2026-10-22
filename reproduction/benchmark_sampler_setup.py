"""Same-seed covariance-backend comparison for exact corrected posterior draws."""
import argparse
import json
import hashlib
from pathlib import Path
import cupy as cp
import numpy as np

from reproduction.matched_teaser import ROOT, measure
from reproduction.posterior import Posterior
from reproduction.paired_applications import (application_geometry, uncertain_objects,
                                             events, quality, field_quality)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=['A', 'B'], default='B')
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--profile', action='store_true', help='synchronize and record preparation stages')
    a = p.parse_args()
    out = ROOT / 'reproduction/results/conditional_solves' / ('setup-' + a.case)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / 'reproduction/results/matched' / a.case / 'input.npz')
    x, y = d['points'], d['normals']
    record = {'args': vars(a), 'N': len(x), 'seconds': {}, 'backends': {},
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), 'seed': 42,
              'hashes': {name: hashlib.sha256(v.tobytes()).hexdigest() for name,v in [('points',x),('normals',y)]},
              'scope': 'controlled preparation variants for the same corrected analytic-vector / finite-scalar posterior; draw timings exclude initial fitting; GPU and CPU realizations differ'}
    def timed(name, fn):
        v, s = measure(fn)
        record['seconds'][name] = s
        (out / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
        print(name, s, flush=True)
        return v
    model = timed('common_fit', lambda: Posterior(x, y))
    q = np.random.default_rng(975).uniform(-np.pi, np.pi, (257, 3)).astype('f')
    rays, objects, distance = application_geometry(x)
    norm2 = (model.op.ns ** 2).sum(0)
    std = float(cp.sqrt(cp.sum(cp.where(norm2 > 0, model.op.rho / cp.maximum(norm2, 1), 0))))
    objects = uncertain_objects(model.mean, objects, std)
    event_outputs, fields = {}, {}
    for backend in ['nufft', 'separable', 'gpu_stream', 'gpu_optimized']:
        if backend == 'gpu_optimized':
            model = timed('optimized_fit', lambda: Posterior(x, y, upsampfac=1.25))
        timings = []
        for repeat in range(a.repeats + 1):
            key = f'{backend}_draw_{repeat}'
            timed(key, lambda: model.draw(a.samples, seed=42, batch=4,
                  covariance_backend='nufft' if backend == 'nufft' else 'separable',
                  rng_backend='gpu' if backend.startswith('gpu') else 'cpu',
                  joint_solve=backend == 'gpu_optimized',profile=a.profile))
            if repeat:
                timings.append(record['seconds'][key])
        fields[backend] = model.samples(q).copy()
        event_outputs[backend] = {kind: cp.asnumpy(events(model.samples(pos.reshape(-1, 3)), pos.shape[:2], kind))
                                  for kind, pos in [('rays', rays), ('collision', objects)]}
        record['backends'][backend] = {'median_draw_seconds': float(np.median(timings)),
                                       'last_draw_profile': model.draw_profile}
    record['field_quality'] = field_quality(fields['nufft'], fields['separable'])
    record['application_quality'] = quality(event_outputs['nufft'], event_outputs['separable'], distance)
    record['speedup'] = record['backends']['nufft']['median_draw_seconds'] / record['backends']['separable']['median_draw_seconds']
    record['gpu_field_quality'] = field_quality(fields['gpu_stream'], fields['gpu_optimized'])
    record['gpu_application_quality'] = quality(event_outputs['gpu_stream'], event_outputs['gpu_optimized'], distance)
    record['optimized_speedup'] = record['backends']['nufft']['median_draw_seconds'] / record['backends']['gpu_optimized']['median_draw_seconds']
    record['rng_qualification'] = 'CPU and GPU seeds do not produce identical draws. Their posterior law is independently tested; paired checks compare only equal-RNG runs.'
    (out / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    np.savez_compressed(out / 'fields.npz', query=q, **{k: cp.asnumpy(v) for k, v in fields.items()})
    print('quality', record['field_quality'], record['application_quality'], flush=True)


if __name__ == '__main__':
    main()
