"""Large real scans with a fixed, modest batch of new occupancy queries.

This separates training-size growth from a costly whole-volume cache and from
thousands of uncompressed variance RHS. All preparation remains timed. Query
selection is deterministic and label-independent; no accepted result is reused
as a claim about one million queries.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reproduction import partial_direct_gpu as direct
from reproduction.stanford_partial_data import digest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['dense', 'h2', 'locality'], required=True)
    p.add_argument('--group', default='test_side'); p.add_argument('--n', type=int, default=32768)
    p.add_argument('--iterations', type=int, default=16); p.add_argument('--cutoff', type=float, default=12.)
    p.add_argument('--query-batch', type=int, default=96); p.add_argument('--query-limit', type=int, default=128)
    p.add_argument('--output', required=True); args=p.parse_args()
    original=direct.load
    def subset(group,n):
        x,y,c,panel=original(group,n)
        if not 1 <= args.query_limit <= len(panel['queries']): raise ValueError('invalid query limit')
        indices=np.linspace(0,len(panel['queries'])-1,args.query_limit,dtype=int)
        return x,y,c,{k:v[indices] for k,v in panel.items()}
    direct.load=subset
    # Existing CPU exact records cover full panels and cannot be compared
    # implicitly to this subset. This harness is deliberately for larger N.
    if (direct.ROOT/'exact'/f'{args.group}-{args.n}.npz').exists():
        raise ValueError('use a larger N without a full-panel automatic CPU audit')
    direct.run(args)
    path=Path(args.output);record=json.loads(path.read_text())
    record['source_sha256'][__file__]=digest(__file__)
    record['query_selection']='Evenly spaced indices of fixed held-out panel, independent of labels; includes surface-band and volume queries.'
    path.write_text(json.dumps(record,indent=2)+'\n')
