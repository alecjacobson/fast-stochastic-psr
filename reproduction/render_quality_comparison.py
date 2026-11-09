"""Render retained quality-control meshes with the existing matched camera."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from reproduction import render_partial_figures as renderer

OUT=Path('reproduction/results/quality_controls')


def save_once(path,values):
    if path.exists():
        previous=np.load(path)
        assert set(previous.files)==set(values)
        for key in values:np.testing.assert_array_equal(previous[key],values[key])
    else:np.savez_compressed(path,**values)


def run(tag,view):
    folder=OUT/('render-'+tag);folder.mkdir(exist_ok=True)
    scene=dict(np.load('reproduction/results/partial_renderings/scene.npz'))
    if tag=='selected':scene['contacts']=np.load(OUT/'test_side-panel.npz')['contacts']
    save_once(folder/'scene.npz',scene)
    data=np.load(OUT/f'display-{tag}-80.npz')
    save_once(folder/'mean-mesh.npz',dict(vertices=data['vertices'],faces=data['faces']))
    renderer.OUT=folder.resolve()
    renderer.run(SimpleNamespace(kind='mean',view=view,planes=False,preview=False,device='GPU'))
    if tag=='selected':renderer.run(SimpleNamespace(kind='input',view=view,planes=False,preview=False,device='GPU'))
    (folder/'wrapper.json').write_text(json.dumps(dict(source=__file__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry=f'display-{tag}-80.npz',geometry_sha256=hashlib.sha256((OUT/f'display-{tag}-80.npz').read_bytes()).hexdigest(),
        scope='Actual posterior mean, same camera/material/display grid for old and selected settings; no mesh repair'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',choices=['old','selected'],required=True)
    p.add_argument('--view',choices=['observed','back'],required=True);a=p.parse_args();run(a.tag,a.view)
