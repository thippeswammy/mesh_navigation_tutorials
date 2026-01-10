import trimesh
import numpy as np
import xml.etree.ElementTree as ET
import os
import sys
def convert_sdf_to_ply(sdf_path, output_path, max_edge_length=None):
    print(f"Parsing SDF: {sdf_path}")
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    
    # Locate the models/geometry
    # If the input is a model.sdf, it has <model> under <sdf>
    # If it is world, it has <world><model>...
    
    models = root.findall(".//model")
    if not models:
        print("No models found in SDF.")
        return
    scene_meshes = []
    base_dir = os.path.dirname(sdf_path)
    for model in models:
        model_name = model.get("name")
        print(f"Processing model: {model_name}")
        
        # Get model pose
        m_pose_text = model.find("pose").text if model.find("pose") is not None else "0 0 0 0 0 0"
        m_vals = [float(x) for x in m_pose_text.split()]
        # trimesh euler is usually xyz, gazebo is locally defined. Assuming standard for now.
        model_m = trimesh.transformations.euler_matrix(m_vals[3], m_vals[4], m_vals[5]) 
        model_m[:3, 3] = m_vals[:3]
        for link in model.findall("link"):
            link_name = link.get("name")
            
            l_pose_text = link.find("pose").text if link.find("pose") is not None else "0 0 0 0 0 0"
            l_vals = [float(x) for x in l_pose_text.split()]
            link_m = trimesh.transformations.euler_matrix(l_vals[3], l_vals[4], l_vals[5])
            link_m[:3, 3] = l_vals[:3]
            
            combined_transform = np.dot(model_m, link_m)
            # Visuals or Collisions? Usually Visuals for PLY map, or Collision for navigation.
            # Let's use Visuals as they likely represent the "Mesh".
            # Actually for navigation we want Collision usually, but often they are the same or Collision is simpler.
            # floor_is_lava uses the same DAE for both.
            
            for visual in link.findall("visual"):
                geom = visual.find("geometry")
                if geom is None: continue
                
                mesh_elem = geom.find("mesh")
                if mesh_elem is not None:
                    uri = mesh_elem.find("uri").text
                    scale_text = mesh_elem.find("scale").text if mesh_elem.find("scale") is not None else "1 1 1"
                    scale = [float(x) for x in scale_text.split()]
                    
                    # Resolve URI
                    if "://" in uri:
                         # Handle model:// or file://
                         # Simplification: assume relative or model:// points to same dir if simple
                         # But floor_is_lava uses "meshes/floor_is_lava.dae" (relative)
                         pass
                    
                    # Construct full path
                    # content of uri in model.sdf is "meshes/floor_is_lava.dae"
                    full_mesh_path = os.path.join(base_dir, uri)
                    
                    if not os.path.exists(full_mesh_path):
                        print(f"Propagated path not found: {full_mesh_path}")
                        # Try finding it relative to current cwd?
                        # Or resolve 'model://'
                        continue
                        
                    print(f"Loading mesh: {full_mesh_path}")
                    try:
                        submesh = trimesh.load(full_mesh_path, force='mesh')
                        if isinstance(submesh, trimesh.Scene):
                            # if it loads as a scene, dump all geometries
                            submesh = trimesh.util.concatenate([g for g in submesh.geometry.values()])
                        
                        # Apply scale
                        if scale != [1,1,1]:
                            submesh.apply_scale(scale)
                            
                        # Apply pose
                        submesh.apply_transform(combined_transform)
                        scene_meshes.append(submesh)
                    except Exception as e:
                        print(f"Failed to load mesh {full_mesh_path}: {e}")
                # Handle primitives (box, etc) if needed?
                # floor_is_lava seems to be just one large specific mesh.
    if not scene_meshes:
        print("No meshes loaded.")
        return
    print(f"Concatenating {len(scene_meshes)} meshes...")
    final_mesh = trimesh.util.concatenate(scene_meshes)
    
    if max_edge_length is not None and max_edge_length > 0:
        print(f"Subdividing mesh to max edge length: {max_edge_length}")
        # remesh.subdivide_to_size returns vertices, faces
        new_v, new_f = trimesh.remesh.subdivide_to_size(final_mesh.vertices, final_mesh.faces, max_edge_length)
        final_mesh = trimesh.Trimesh(vertices=new_v, faces=new_f)
        print(f"Subdivided mesh: {len(final_mesh.vertices)} vertices, {len(final_mesh.faces)} faces")

    print(f"Exporting to {output_path}...")
    final_mesh.export(output_path)
    print("Done.")

if __name__ == "__main__":
    import argparse
    
    # Defaults
    default_input = "/media/thippe/SDV/Ubuntu/github_testing/mesh_navigation/src/mesh_navigation_tutorials/mesh_navigation_tutorials_sim/models/floor_is_lava/model.sdf"
    # Note: We will generate output in src, then you can copy to install
    default_output_dir = "/media/thippe/SDV/Ubuntu/github_testing/mesh_navigation/src/mesh_navigation_tutorials/mesh_navigation_tutorials_sim/models/floor_is_lava_1/meshes"
    
    parser = argparse.ArgumentParser(description="Convert SDF to Mesh with optional subdivision")
    parser.add_argument("--input", default=default_input, help="Input SDF file")
    parser.add_argument("--output_dir", default=default_output_dir, help="Directory to save output files")
    parser.add_argument("--max_edge", type=float, default=0.32, help="Max edge length for subdivision (meters). Default 0.32")
    parser.add_argument("--name", default="floor_is_lava_1", help="Base name for output files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Generate PLY (High-res for navigation)
    ply_path = os.path.join(args.output_dir, args.name + ".ply")
    convert_sdf_to_ply(args.input, ply_path, max_edge_length=args.max_edge)
    
    # Generate DAE (Low-res/Original for visuals - preserves textures)
    dae_path = os.path.join(args.output_dir, args.name + ".dae")
    convert_sdf_to_ply(args.input, dae_path, max_edge_length=0)

