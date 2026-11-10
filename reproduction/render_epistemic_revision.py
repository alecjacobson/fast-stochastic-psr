"""Render the unchanged geometry with the new uncertainty section located in 3D."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from reproduction import render_partial_figures as renderer
from reproduction.render_quality_comparison import save_once

OUT=Path('reproduction/results/epistemic_revision')


def main():
    folder=OUT/'render';folder.mkdir(exist_ok=True)
    section=np.load(OUT/'queries.npz')
    scene=dict(np.load('reproduction/results/partial_renderings/scene.npz'))
    scene.update(heights=np.array([section['plane_y']]),axis_x=section['axis_x'],axis_z=section['axis_z'],contacts=np.empty((0,3)))
    save_once(folder/'scene.npz',scene)
    mesh=np.load('reproduction/results/observed_fidelity/display-selected-240.npz')
    save_once(folder/'mean-mesh.npz',dict(vertices=mesh['vertices'],faces=mesh['faces']))
    renderer.OUT=folder.resolve()
    for kind in ['input','mean']:
        renderer.run(SimpleNamespace(kind=kind,view='back',planes=True,preview=False,device='GPU'))
    sources=[__file__,'reproduction/render_partial_figures.py','reproduction/render_quality_comparison.py']
    (folder/'wrapper.json').write_text(json.dumps(dict(
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        query_sha256=hashlib.sha256((OUT/'queries.npz').read_bytes()).hexdigest(),
        scope='Unchanged mean geometry, matched cameras, actual input points; plane locates sampled marginals, not a joint confidence surface.'),indent=2)+'\n')


if __name__=='__main__':main()
