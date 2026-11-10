"""Matched scientific renders of the observed-fidelity meshes."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from reproduction import render_partial_figures as renderer
from reproduction.render_quality_comparison import save_once

OUT=Path('reproduction/results/observed_fidelity')


def run(tag,view,resolution):
    folder=OUT/('render-'+tag);folder.mkdir(exist_ok=True)
    scene=dict(np.load('reproduction/results/partial_renderings/scene.npz'))
    if tag=='selected':scene['contacts']=np.load(OUT/'test_side-panel.npz')['contacts']
    if tag=='input':scene['contacts']=np.empty((0,3))  # Show the measured surface without contact markers obscuring detail.
    save_once(folder/'scene.npz',scene)
    if tag!='input':
        data=np.load(OUT/f'display-{tag}-{resolution}.npz')
        save_once(folder/'mean-mesh.npz',dict(vertices=data['vertices'],faces=data['faces']))
    renderer.OUT=folder.resolve()
    renderer.run(SimpleNamespace(kind='input' if tag=='input' else 'mean',view=view,planes=False,preview=False,device='GPU'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',choices=['control','selected','input'],required=True)
    p.add_argument('--view',choices=['observed','back'],required=True);p.add_argument('--resolution',type=int,default=240)
    a=p.parse_args();run(a.tag,a.view,a.resolution)
