"""Screen edge blocks using saved frequencies and OLD inverse weights.

This ranks contributions to an existing H diagnostic. It does not recompute
the inverse and must not be interpreted as the new posterior error or runtime.
The input query panel is reused for selection; independent validation is needed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from geospsr.spectral_blocks import two_high_axis_refinement_blocks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--edges',type=int,nargs='+',default=[175,200,250,300,400,800])
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    record = json.loads((args.input/'results.json').read_text())
    data = np.load(args.input/'fields.npz')
    core,extent = record['args']['core'],record['args']['extent']
    seeds = record['args']['seeds']
    modes = np.concatenate([data[f'{s}_corners_frequencies'] for s in seeds])
    trace = np.concatenate([data[f'{s}_corners_trace_samples'] for s in seeds])
    outside = np.concatenate([data[f'{s}_outside_trace_samples'] for s in seeds])
    scale = np.linalg.norm(data['finite_covariance'])
    total = trace.mean()+outside.mean()
    output = {'input':str(args.input),'core':core,'extent':extent,'hashes':record['hashes'],
              'scope':'fixed OLD inverse weights, reused selection panel; no updated posterior error or measured performance',
              'baseline_trace_estimate_over_covariance_frobenius':float(total/scale),'candidates':[]}
    for edge in args.edges:
        boxes = two_high_axis_refinement_blocks(core,edge,extent)
        modes_count = sum(int(np.prod([hi-lo+1 for lo,hi in box])) for box in boxes)
        added = 12*(edge-core)**2*(2*core+1)
        captured = ((np.abs(modes)>core).sum(1)==2)&(np.max(np.abs(modes),axis=1)<=edge)
        contribution = trace*captured
        remaining = trace*(~captured)
        estimate = remaining.mean()+outside.mean()
        se = np.sqrt(remaining.var(ddof=1)/len(remaining)+outside.var(ddof=1)/len(outside))
        output['candidates'].append({'edge':edge,'added_retained_modes':added,'added_conjugate_paired_stored_modes':added//2,
            'total_retained_modes':modes_count,'added_positive_transform_pairs':6 if edge>core else 0,
            'captured_corner_trace_fraction':float(contribution.mean()/trace.mean()),
            'captured_total_trace_fraction':float(contribution.mean()/total),
            'remaining_fixed_weight_trace_over_covariance_frobenius':float(estimate/scale),
            'remaining_fixed_weight_trace_standard_error':float(se/scale)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
