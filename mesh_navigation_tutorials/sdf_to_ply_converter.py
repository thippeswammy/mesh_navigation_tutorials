import trimesh
import numpy as np
import xml.etree.ElementTree as ET
import os
import sys
import argparse
def convert_sdf_to_ply(sdf_path, output_path):
    if not os.path.exists(sdf_path):
        print(f"Error: SDF file not found: {sdf_path}")
        return
    print(f"Parsing SDF: {sdf_path}")
    try:
        tree = ET.parse(sdf_path)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return
        
    root = tree.getroot()
    
    # Locate models: works for world files (<world><model>...) and model files (<model>...)
    models = root.findall(".//model")
    # If root itself is model
    if root.tag == "model":
        models.append(root)
        
    if not models:
        print("No models found in SDF.")
        return
    scene_meshes = []
    base_dir = os.path.dirname(os.path.abspath(sdf_path))
    for model in models:
        model_name = model.get("name")
        print(f"Processing model: {model_name}")
        
        # Get model pose
        m_pose_text = model.find("pose").text if model.find("pose") is not None else "0 0 0 0 0 0"
        m_vals = [float(x) for x in m_pose_text.split()]
        # Assuming RPY order for Euler
        model_m = trimesh.transformations.euler_matrix(m_vals[3], m_vals[4], m_vals[5]) 
        model_m[:3, 3] = m_vals[:3]
        for link in model.findall(".//link"):
            l_pose_text = link.find("pose").text if link.find("pose") is not None else "0 0 0 0 0 0"
            l_vals = [float(x) for x in l_pose_text.split()]
            link_m = trimesh.transformations.euler_matrix(l_vals[3], l_vals[4], l_vals[5])
            link_m[:3, 3] = l_vals[:3]
            
            combined_transform = np.dot(model_m, link_m)
            for visual in link.findall("visual"):
                geom = visual.find("geometry")
                if geom is None: continue
                
                mesh_elem = geom.find("mesh")
                if mesh_elem is not None:
                    uri = mesh_elem.find("uri").text
                    scale_text = mesh_elem.find("scale").text if mesh_elem.find("scale") is not None else "1 1 1"
                    scale = [float(x) for x in scale_text.split()]
                    
                    full_mesh_path = uri
                    # Handle model:// URI
                    if uri.startswith("model://"):
                         # This is complex without a full ROS package path resolver
                         # Simplistic approach: remove model:// and look in ../models?
                         # Or just warn user. 
                         # For floor_is_lava, it was "meshes/..." relative path
                         print(f"Warning: 'model://' URI not fully supported in standalone script: {uri}")
                         continue
                    
                    if not os.path.isabs(uri):
                        full_mesh_path = os.path.join(base_dir, uri)
                    
                    if not os.path.exists(full_mesh_path):
                        print(f"Mesh file not found: {full_mesh_path}")
                        continue
                        
                    print(f"Loading mesh: {full_mesh_path}")
                    try:
                        submesh = trimesh.load(full_mesh_path, force='mesh')
                        if isinstance(submesh, trimesh.Scene):
                            if len(submesh.geometry) == 0:
                                print(f"Warning: Empty scene in {full_mesh_path}")
                                continue
                            submesh = trimesh.util.concatenate([g for g in submesh.geometry.values()])
                        
                        if scale != [1,1,1]:
                            submesh.apply_scale(scale)
                            
                        submesh.apply_transform(combined_transform)
                        scene_meshes.append(submesh)
                    except Exception as e:
                        print(f"Failed to load mesh {full_mesh_path}: {e}")
                # Simple primitives support
                box = geom.find("box")
                if box is not None:
                    size = [float(x) for x in box.find("size").text.split()]
                    m = trimesh.creation.box(extents=size)
                    m.apply_transform(combined_transform)
                    scene_meshes.append(m)
                    
                cylinder = geom.find("cylinder")
                if cylinder is not None:
                    r = float(cylinder.find("radius").text)
                    l = float(cylinder.find("length").text)
                    m = trimesh.creation.cylinder(radius=r, height=l)
                    m.apply_transform(combined_transform)
                    scene_meshes.append(m)
                sphere = geom.find("sphere")
                if sphere is not None:
                    r = float(sphere.find("radius").text)
                    m = trimesh.creation.icosphere(radius=r)
                    m.apply_transform(combined_transform)
                    scene_meshes.append(m)
    if not scene_meshes:
        print("No meshes loaded or generated.")
        return
    print(f"Concatenating {len(scene_meshes)} meshes...")
    final_mesh = trimesh.util.concatenate(scene_meshes)
    
    print(f"Exporting to {output_path}...")
    final_mesh.export(output_path)
    print("Done.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SDF model/world to PLY/DAE mesh.")
    parser.add_argument("input_sdf", help="Path to input SDF or World file")
    parser.add_argument("output_path", help="Path to output mesh file (e.g. .ply or .dae)")
    
    args = parser.parse_args()
    convert_sdf_to_ply(args.input_sdf, args.output_path)
