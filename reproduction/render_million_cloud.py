"""Blender/Cycles render of every observed point and the two statistical slices.

Run with the separate render environment (bpy 4.3 + NumPy 1.26), not JAX's env.
"""
import argparse,json,os
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true')
p.add_argument('--input',type=Path);p.add_argument('--radius',type=float,default=.0025);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/teaser'
if a.input:
    out=a.input.parent
    data=np.load(a.input);fields=data
else:
    data=np.load(out/'cloud_1000000.npz');fields=np.load(out/'fields.npz')
points=data['points'];heights=fields['heights'];extent=float(fields['axis'][-1])
# Original geometry is Y-up. Blender geometry is Z-up.
def transform(v):return np.stack([v[...,0],-v[...,2],v[...,1]],axis=-1)
xyz=transform(points).astype(np.float32)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene
scene.render.engine='CYCLES';scene.cycles.samples=32 if a.preview else 192
scene.cycles.use_denoising=True
scene.render.resolution_x=1000 if a.preview else 1800
scene.render.resolution_y=1000 if a.preview else 1800
scene.render.resolution_percentage=100
scene.render.film_transparent=True
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
scene.view_settings.view_transform='Standard';scene.view_settings.look='Medium High Contrast'
scene.world=bpy.data.worlds.new('White studio');scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.6
prefs=bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type='OPTIX';prefs.get_devices()
for device in prefs.devices:device.use=device.type=='OPTIX'
scene.cycles.device='GPU'

mat=bpy.data.materials.new('Warm stone point cloud');mat.use_nodes=True
bsdf=mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value=(.20,.21,.12,1)
bsdf.inputs['Roughness'].default_value=.82
bsdf.inputs['Specular IOR Level'].default_value=.15
mesh=bpy.data.meshes.new('Measured positions')
mesh.vertices.add(len(xyz));mesh.vertices.foreach_set('co',xyz.ravel());mesh.update()
obj=bpy.data.objects.new(f'Input — {len(points):,} observations',mesh);scene.collection.objects.link(obj)
mod=obj.modifiers.new('Actual measured points','NODES')
group=bpy.data.node_groups.new('Point cloud, no subsampling','GeometryNodeTree')
group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
mod.node_group=group
nodes=group.nodes;links=group.links
src=nodes.new('NodeGroupInput');dst=nodes.new('NodeGroupOutput')
pts=nodes.new('GeometryNodeMeshToPoints');pts.mode='VERTICES';pts.inputs['Radius'].default_value=a.radius
setmat=nodes.new('GeometryNodeSetMaterial');setmat.inputs['Material'].default_value=mat
links.new(src.outputs['Geometry'],pts.inputs['Mesh']);links.new(pts.outputs['Points'],setmat.inputs['Geometry']);links.new(setmat.outputs['Geometry'],dst.inputs['Geometry'])

plane_mat=bpy.data.materials.new('Translucent blue-gray slice');plane_mat.use_nodes=True
nodes=plane_mat.node_tree.nodes;nodes.clear();links=plane_mat.node_tree.links
output=nodes.new('ShaderNodeOutputMaterial');mix=nodes.new('ShaderNodeMixShader');mix.inputs[0].default_value=.22
transparent=nodes.new('ShaderNodeBsdfTransparent');diffuse=nodes.new('ShaderNodeEmission');diffuse.inputs['Color'].default_value=(.45,.58,.64,1)
links.new(transparent.outputs[0],mix.inputs[1]);links.new(diffuse.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],output.inputs[0])
edge_mat=bpy.data.materials.new('Slice rim');edge_mat.diffuse_color=(.19,.25,.27,1)
for i,h in enumerate(heights):
    corners=transform(np.array([[-extent,h,-extent],[extent,h,-extent],[extent,h,extent],[-extent,h,extent]]))
    m=bpy.data.meshes.new('Slice');m.from_pydata(corners.tolist(),[],[(0,1,2,3)]);m.materials.append(plane_mat)
    plane=bpy.data.objects.new('Upper slice' if i==0 else 'Lower slice',m);scene.collection.objects.link(plane);plane.visible_shadow=False
    curve=bpy.data.curves.new('Slice rim','CURVE');curve.dimensions='3D';curve.bevel_depth=.0015;curve.bevel_resolution=2
    line=curve.splines.new('POLY');line.points.add(4)
    for pt,c in zip(line.points,np.vstack([corners,corners[0]])):pt.co=(*c,1)
    edge=bpy.data.objects.new('Rim',curve);scene.collection.objects.link(edge);curve.materials.append(edge_mat)

def area(name,location,power,size):
    light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
    obj=bpy.data.objects.new(name,light);scene.collection.objects.link(obj);obj.location=location
    obj.rotation_euler=(Vector((0,0,0))-obj.location).to_track_quat('-Z','Y').to_euler()
area('Large softbox',(-3,-4,7),700,5)
area('Fill',(4,2,5),400,4)
camera=bpy.data.cameras.new('Orthographic scientific view');cam=bpy.data.objects.new('Camera',camera);scene.collection.objects.link(cam)
cam.location=(-10,-.8,1.3);target=Vector((0,0,.0));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
camera.type='ORTHO';camera.ortho_scale=4.1;camera.lens=50;scene.camera=cam
bpy.context.view_layer.update()
anchors=[]
for h in heights:
    corners=transform(np.array([[-extent,h,-extent],[extent,h,-extent],[extent,h,extent],[-extent,h,extent]]))
    projected=[world_to_camera_view(scene,cam,Vector(c)) for c in corners]
    anchor=max(projected,key=lambda p:p.x)
    anchors.append([float(anchor.x),float(anchor.y)])
(out/'cloud_projection.json').write_text(json.dumps({'anchors':anchors,'rendered_points':len(points),'camera':list(cam.location),'radius':a.radius,'blender':bpy.app.version_string},indent=2))
scene.render.filepath=str(out/('cloud_preview.png' if a.preview else 'cloud_render.png'))
bpy.ops.render.render(write_still=True)
print('Rendered',len(points),'observations to',scene.render.filepath)
