"""Audit saved sharp-model timing fields and plot an explicit record selection."""
import json
from pathlib import Path

import numpy as np

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.benchmark_observed_fidelity import OUT,FROZEN
from reproduction.partial_occupancy import ROOT
from reproduction.partial_competitor_common import gates
from reproduction.stanford_partial_data import digest


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def truth_for(record):
    contract=record['contract'];cfg=contract['configuration'];ids=contract['query_indices']
    refpath=Path(record['reference_path']);ref=json.loads(refpath.read_text())
    assert digest(refpath)==record['reference_sha256']
    assert digest(refpath.with_suffix('.npz'))==ref['field_sha256']
    fields=np.load(refpath.with_suffix('.npz'))
    if contract['validation_panel'] and contract['group']=='validation':
        assert contract['n']==16384 and ref['status']=='completed'
        column=next(i for i,r in enumerate(ref['records']) if r['contacts']==cfg['contacts'] and r['scalar_variance']==cfg['scalar_variance'])
        return {k:fields[k][ids,column] for k in ['mean','variance']}
    if contract['validation_panel']:
        assert contract['group']=='test_side' and contract['n']==70652
        assert ref['status']=='completed' and ref['configuration']==cfg
        return {k:fields[k][ids] for k in ['mean','variance']}
    assert ref['accepted'] and ref['contract']==contract
    return dict(fields)


def replay():
    selected=json.loads((FROZEN/'selected.json').read_text())['configuration']
    model=FreeSpacePoissonMatern(selected['ell'],1/3)
    fit=json.loads((ROOT/'controls.json').read_text())['fits']['spatial']
    rows=[]
    for path in sorted(OUT.glob('*.json')):
        r=json.loads(path.read_text())
        if 'arguments' not in r:continue
        args=r['arguments'];contract=r['contract']
        assert contract['configuration']==selected
        assert contract['selected_sha256']==digest(FROZEN/'selected.json')
        assert contract['dataset_sha256']==digest(ROOT/'dataset.json')
        assert contract['panel_sha256']==digest(FROZEN/(contract['group']+'-panel.npz'))
        assert contract['query_sha256']==digest(ROOT/(contract['group']+'-queries.npz'))
        assert contract['protocol_sha256']==digest(OUT/'PROTOCOL.md')
        for source,sha in r['source_sha256'].items():assert digest(OUT/'sources'/(sha+'.py'))==sha,source
        if r.get('accepted'):assert r['gpu_audit']['sampled_isolation_pass']
        if 'field_sha256' in r:assert digest(path.with_suffix('.npz'))==r['field_sha256']
        if args['method']=='reference':
            assert r['accepted']
            if 'refinement_history' in r:assert all(h[-1]['relative_residual']<=1e-10 for h in r['refinement_history'])
            continue
        if 'field_sha256' in r:
            f=np.load(path.with_suffix('.npz'));errors=gates(f['mean'],f['variance'],truth_for(r),model,fit)
            expected=(r['selected']['variants'][r['selected']['selected_variant']]['errors'] if args['method']=='sdd' else r['errors'])
            assert errors==expected,(path,errors,expected)
            if r.get('accepted'):assert errors['passes']
        audit=r.get('gpu_audit',{});observations=audit.get('observations',[])
        gpu=[o['own_memory_bytes'] for o in observations if o.get('own_memory_bytes') is not None]
        host=[o['own_host_rss_bytes'] for o in observations if o.get('own_host_rss_bytes') is not None]
        row=dict(file=path.name,record_sha256=digest(path),method=args['method'],n=contract['n'],queries=contract['queries'],
                 group=contract['group'],validation_panel=contract['validation_panel'],accepted=r.get('accepted',False),
                 status=r['status'],seconds=r.get('total_seconds',r.get('selected',{}).get('total_seconds')),
                 gpu_peak_gib=max(gpu)/2**30 if gpu else None,host_peak_gib=max(host)/2**30 if host else None,
                 arguments=args,errors=r.get('errors',r.get('selected',{})),setup_seconds=r.get('setup_seconds'))
        rows.append(row)
    write(OUT/'summary.json',dict(rows=rows,source_sha256=digest(__file__),legacy_controls_sha256=digest(ROOT/'controls.json'),
        scope='Replay all retained trials, including failed/slower settings; numerical gates unchanged.'))
    print('Replayed',len(rows),'trials;',sum(r['accepted'] for r in rows),'accepted.')
    return rows


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selection=json.loads((OUT/'display-selection.json').read_text())
    byname={r['file']:r for r in rows};curves=[]
    defaults=dict(query_block=0,cross_storage='dual',normal_preconditioner_dtype=None,
                  leaf_size=40,coupled_preconditioner='prior',locality_storage='streamed')
    keys=['method','iterations','tolerance','leaf_size','rank','neighbors','cutoff','refinements',
          'coupled_preconditioner','locality_storage','cross_storage','normal_preconditioner_dtype','query_block','block']
    def settings(row):
        return {key:row['arguments'].get(key,defaults.get(key)) for key in keys}
    for curve in selection['curves']:
        if curve.get('audit'):
            check=byname[curve['audit']]
            assert check['accepted'] and check['queries']==512 and check['validation_panel'] and check['n']==70652
        points=[]
        for n,names in curve['points'].items():
            records=[byname[name] for name in names]
            assert all(r['accepted'] and r['n']==int(n) and r['queries']==128 and not r['validation_panel'] and r['group']=='test_side' for r in records)
            if curve.get('audit'):assert all(settings(r)==settings(check) for r in records)
            times=[r['seconds'] for r in records]
            points.append(dict(n=int(n),seconds=float(np.median(times)),minimum=min(times),maximum=max(times),
                gpu_peak_gib=max(r['gpu_peak_gib'] for r in records),host_peak_gib=max(r['host_peak_gib'] for r in records),records=names))
        points.sort(key=lambda r:r['n']);curve=dict(curve,measurements=points)
        if len(points)>1:curve['slope']=float(np.polyfit(np.log([r['n'] for r in points]),np.log([r['seconds'] for r in points]),1)[0])
        curves.append(curve)
    write(OUT/'comparison.json',dict(curves=curves,selection_sha256=digest(OUT/'display-selection.json'),source_sha256=digest(__file__)))
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.7),layout='constrained')
    for curve in curves:
        p=curve['measurements'];n=[r['n'] for r in p];time=np.array([r['seconds'] for r in p])
        label=curve['label']+(f' (slope {curve["slope"]:.2f})' if 'slope' in curve else '')
        axes[0].errorbar(n,time,yerr=[time-[r['minimum'] for r in p],[r['maximum'] for r in p]-time],
            marker=curve.get('marker','o'),color=curve['color'],capsize=3,label=label)
        axes[1].plot(n,[r['gpu_peak_gib'] for r in p],marker=curve.get('marker','o'),color=curve['color'],label=curve['label'])
    for axis in axes:
        axis.set_xscale('log');axis.set_yscale('log');axis.set_xlabel('Normal observations N');axis.grid(True,which='both',alpha=.18)
        axis.set_xticks([32768,49152,70652],['32,768','49,152','70,652']);axis.minorticks_off()
        axis.legend(frameon=False,fontsize=9)
    axes[0].set(ylabel='Fresh setup + contacts + mean/variance/CDF (s)',title='All selected outputs pass unchanged accuracy gates')
    axes[1].axhline(23.625,color='.5',ls='--',lw=1);axes[1].text(33000,24.5,'GPU allocation: 23.625 GiB',fontsize=9,color='.35')
    axes[1].set(ylabel='Sampled process GPU peak (GiB)',title='Counts cross storage and every preconditioner')
    fig.suptitle('Sharp partial-scan posterior: C = 2,048 contacts; Q = 128 marginal queries',fontsize=14)
    for extension in ['png','pdf']:fig.savefig(OUT/('performance.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    rows=replay()
    if (OUT/'display-selection.json').exists():plot(rows)
