#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

def get_transform_from_pose(pose_text):
    if not pose_text:
        return np.eye(4)
    values = [float(x) for x in pose_text.split()]
    
    # x y z r p y
    trans = values[:3]
    rot_euler = values[3:] if len(values) >= 6 else [0, 0, 0]
    
    T = np.eye(4)
    T[:3, 3] = trans
    r = R.from_euler('xyz', rot_euler, degrees=False)
    T[:3, :3] = r.as_matrix()
    return T

def create_primitive(geometry_node, resolution=64):
    # Box
    box = geometry_node.find("box")
    if box is not None:
        size_node = box.find("size")
        if size_node is not None:
            size = [float(x) for x in size_node.text.split()]
            return trimesh.creation.box(extents=size)

    # Cylinder
    cylinder = geometry_node.find("cylinder")
    if cylinder is not None:
        r = float(cylinder.find("radius").text)
        l = float(cylinder.find("length").text)
        return trimesh.creation.cylinder(radius=r, height=l, sections=resolution)

    # Sphere
    sphere = geometry_node.find("sphere")
    if sphere is not None:
        r = float(sphere.find("radius").text)
        subdivisions = 3 if resolution < 32 else (4 if resolution < 64 else 5)
        return trimesh.creation.icosphere(radius=r, subdivisions=subdivisions)
        
    # Plane
    plane = geometry_node.find("plane")
    if plane is not None:
        size_node = plane.find("size")
        if size_node is not None:
            size_raw = [float(x) for x in size_node.text.split()] # Usually x y
            # Create a thin box for the plane
            size = [size_raw[0], size_raw[1], 0.01]
            return trimesh.creation.box(extents=size)

    return None

def main():
    parser = argparse.ArgumentParser(description="Convert SDF Primitives to Single PLY")
    parser.add_argument("input_sdf", help="Input SDF file")
    parser.add_argument("output_ply", help="Output PLY file")
    parser.add_argument("--resolution", type=int, default=64, help="Primitive resolution")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_sdf):
        print(f"Error: Input file not found: {args.input_sdf}")
        sys.exit(1)
        
    tree = ET.parse(args.input_sdf)
    root = tree.getroot()
    
    # Process Models
    models = root.findall(".//model")
    if root.tag == "model":
        models.append(root)
        
    scene_meshes = []
    
    print(f"Processing {len(models)} models...")
    
    for model in models:
        # Model Pose
        m_pose = model.find("pose")
        model_transform = get_transform_from_pose(m_pose.text if m_pose is not None else None)
        
        links = model.findall(".//link")
        for link in links:
            # Link Pose
            l_pose = link.find("pose")
            link_transform = get_transform_from_pose(l_pose.text if l_pose is not None else None)
            
            # Combine Poses
            current_transform = np.dot(model_transform, link_transform)
            
            # Get geometry from Collision (preferred) or Visual
            # User request said "Parses box/plane/cylinder/sphere", usually found in collision or visual
            # We'll check both, preferring visuals for mesh-like export
            sources = link.findall("visual")
            if not sources:
                 sources = link.findall("collision")
                 
            for source in sources:
                geom = source.find("geometry")
                if geom is None: continue
                
                # Check for Primitives
                mesh = create_primitive(geom, resolution=args.resolution)
                if mesh:
                    # Apply Pose of geometry relative to link
                    v_pose = source.find("pose")
                    if v_pose is not None:
                         v_transform = get_transform_from_pose(v_pose.text)
                         mesh.apply_transform(v_transform)
                         
                    # Apply global transform
                    mesh.apply_transform(current_transform)
                    scene_meshes.append(mesh)
    
    if not scene_meshes:
        print("No primitives found to export.")
        sys.exit(1)
        
    print(f"Merging {len(scene_meshes)} primitives...")
    final_mesh = trimesh.util.concatenate(scene_meshes)
    
    print(f"Exporting to {args.output_ply}...")
    final_mesh.export(args.output_ply)
    print("Done.")

if __name__ == "__main__":
    main()
