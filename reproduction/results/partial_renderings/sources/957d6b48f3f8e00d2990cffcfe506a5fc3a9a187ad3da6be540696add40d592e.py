"""Cycles renders of actual Armadillo inputs and computed mean geometry.

Use reproduction/render_setup.sh's separate bpy environment. Quantitative
probability colors use emission and the Standard view transform.
"""
import argparse
import hashlib
import json
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

OUT = Path(__file__).resolve().parent/'results/partial_renderings'


def transform(v):
    return np.stack([v[...,0],-v[...,2],v[...,1]],axis=-1)


def material(name, color, emission=False, opacity=1.):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    nodes=mat.node_tree.nodes;nodes.clear();links=mat.node_tree.links
    out=nodes.new('ShaderNodeOutputMaterial')
    shader=nodes.new('ShaderNodeEmission' if emission else 'ShaderNodeBsdfPrincipled')
    if emission:
        # Matplotlib colors are sRGB; Blender emission inputs are scene-linear.
        rgb=np.asarray(color);rgb=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
        shader.inputs['Color'].default_value=(*rgb,1)
    else:
        shader.inputs['Base Color'].default_value=(*color,1)
        shader.inputs['Roughness'].default_value=.78
        shader.inputs['Specular IOR Level'].default_value=.22
    if opacity<1:
        transparent=nodes.new('ShaderNodeBsdfTransparent');mix=nodes.new('ShaderNodeMixShader');mix.inputs[0].default_value=opacity
        links.new(transparent.outputs[0],mix.inputs[1]);links.new(shader.outputs[0],mix.inputs[2]);shader=mix
    links.new(shader.outputs[0],out.inputs[0])
    return mat


def triangle_mesh(name, vertices, faces, mat):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(transform(vertices).tolist(),[],faces.tolist());mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj);mesh.materials.append(mat)
    for face in mesh.polygons:face.use_smooth=True
    return obj


def cloud(name, positions, radius, mat):
    xyz=transform(positions).astype(np.float32)
    mesh=bpy.data.meshes.new(name);mesh.vertices.add(len(xyz));mesh.vertices.foreach_set('co',xyz.ravel());mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj)
    mod=obj.modifiers.new('Measured points','NODES');group=bpy.data.node_groups.new(name,'GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry');mod.node_group=group
    nodes=group.nodes;links=group.links;src=nodes.new('NodeGroupInput');dst=nodes.new('NodeGroupOutput')
    pts=nodes.new('GeometryNodeMeshToPoints');pts.mode='VERTICES';pts.inputs['Radius'].default_value=radius
    sm=nodes.new('GeometryNodeSetMaterial');sm.inputs['Material'].default_value=mat
    links.new(src.outputs['Geometry'],pts.inputs['Mesh']);links.new(pts.outputs['Points'],sm.inputs['Geometry']);links.new(sm.outputs['Geometry'],dst.inputs['Geometry'])
    return obj


def curve(name, corners, mat, radius=.003):
    data=bpy.data.curves.new(name,'CURVE');data.dimensions='3D';data.bevel_depth=radius;data.bevel_resolution=2
    poly=data.splines.new('POLY');poly.points.add(len(corners)-1)
    for p,v in zip(poly.points,transform(corners)):p.co=(*v,1)
    obj=bpy.data.objects.new(name,data);bpy.context.scene.collection.objects.link(obj);data.materials.append(mat)
    return obj


def run(args):
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data=np.load(OUT/'scene.npz');bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene
    scene.render.engine='CYCLES';scene.cycles.samples=24 if args.preview else 128;scene.cycles.use_denoising=True
    scene.render.threads_mode='FIXED';scene.render.threads=2
    scene.render.resolution_x=720 if args.preview else 1500;scene.render.resolution_y=900 if args.preview else 1875
    scene.render.resolution_percentage=100;scene.render.film_transparent=True
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.world=bpy.data.worlds.new('White studio');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.45
    scene.cycles.device=args.device
    if args.device=='GPU':
        prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
        for dev in prefs.devices:dev.use=dev.type=='OPTIX'
    gold=material('Measured range points',(.25,.23,.15));red=material('Scalar contacts',(.72,.035,.025),emission=True)
    stone=material('Reference geometry',(.20,.23,.25));blue=material('Posterior mean surface',(.12,.20,.24))
    if args.kind=='reference':triangle_mesh('Complete reconstructed reference',data['vertices'],data['faces'],stone)
    elif args.kind=='mean':
        mesh=np.load(OUT/'mean-mesh.npz');triangle_mesh('Posterior mean zero level set',mesh['vertices'],mesh['faces'],blue)
    elif args.kind=='input':
        cloud('All 70652 actual range observations',data['points'],.0033,gold)
        cloud('32 scalar surface contacts',data['contacts'],.011,red)
    else:
        ghost=material('Reference context',(.65,.69,.72),opacity=.12)
        triangle_mesh('Reference context',data['vertices'],data['faces'],ghost)
        # These are exactly the timed marginal query positions, not surface samples.
        colors=np.load(OUT/'probe-colors.npz')[args.kind.split('-')[1]]
        for i,(q,c) in enumerate(zip(data['benchmark_queries'],colors)):
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=.018,location=transform(q))
            bpy.context.object.data.materials.append(material('Query%d'%i,tuple(c[:3]),emission=True))
    plane_corners=[]
    if args.planes:
        for i,height in enumerate(data['heights']):
            x,z=data['axis_x'],data['axis_z']
            corners=np.array([[x[0],height,z[0]],[x[-1],height,z[0]],[x[-1],height,z[-1]],[x[0],height,z[-1]]])
            color=(.16,.39,.58) if i==0 else (.70,.31,.10)
            obj=triangle_mesh('Cross-section '+str(i),corners,np.array([[0,1,2],[0,2,3]]),material('Section fill',color,emission=True,opacity=.12));obj.visible_shadow=False
            curve('Cross-section rim',np.vstack((corners,corners[0])),material('Section rim',color,emission=True),.0025)
            plane_corners.append(corners)
    def area(name,position,energy,size):
        light=bpy.data.lights.new(name,'AREA');light.energy=energy;light.shape='DISK';light.size=size
        obj=bpy.data.objects.new(name,light);scene.collection.objects.link(obj);obj.location=position
        obj.rotation_euler=(Vector((0,0,.1))-obj.location).to_track_quat('-Z','Y').to_euler()
    area('Key',(-3,-4,7),350,4);area('Fill',(4,2,4),160,4)
    camera=bpy.data.cameras.new('Matched orthographic camera');cam=bpy.data.objects.new('Camera',camera);scene.collection.objects.link(cam)
    cam.location=(4.8,-6.0,2.4) if args.view=='observed' else (-4.8,6.0,2.4)
    target=Vector((0,0,.04));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
    camera.type='ORTHO';camera.ortho_scale=2.72 if not args.planes else 3.5;scene.camera=cam
    bpy.context.view_layer.update()
    anchors=[]
    for corners in plane_corners:
        pp=[world_to_camera_view(scene,cam,Vector(c)) for c in transform(corners)]
        p=max(pp,key=lambda p:p.x);anchors.append([float(p.x),float(p.y)])
    name=args.kind+'-'+args.view+('-planes' if args.planes else '')+('-preview' if args.preview else '')
    scene.render.filepath=str(OUT/(name+'.png'));bpy.ops.render.render(write_still=True)
    (OUT/(name+'.json')).write_text(json.dumps(dict(kind=args.kind,view=args.view,planes=args.planes,
        preview=args.preview,device=args.device,blender=bpy.app.version_string,camera=list(cam.location),
        target=list(target),orthographic_scale=camera.ortho_scale,point_radius=.0033,contact_radius=.011,
        samples=scene.cycles.samples,plane_label_anchors=anchors,
        source_sha256=source_hash,scene_sha256=hashlib.sha256((OUT/'scene.npz').read_bytes()).hexdigest(),
        geometry_sha256=hashlib.sha256((OUT/('mean-mesh.npz' if args.kind=='mean' else 'scene.npz')).read_bytes()).hexdigest(),
        color_management='Standard / None; quantitative emission colors converted from sRGB to linear',
        reference_used=args.kind in ['reference','probes-h2','probes-dense']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['input','reference','mean','probes-h2','probes-dense'],required=True)
    p.add_argument('--view',choices=['observed','back'],default='observed');p.add_argument('--planes',action='store_true')
    p.add_argument('--device',choices=['GPU','CPU'],default='GPU');p.add_argument('--preview',action='store_true');run(p.parse_args())
