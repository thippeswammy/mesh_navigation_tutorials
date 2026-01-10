#!/usr/bin/env python3
import os
import trimesh
import numpy as np
import xml.etree.ElementTree as ET
import subprocess
import shutil
import sys
import argparse
import struct
import h5py
import time
import uuid

# ROS 2 Launch Imports
try:
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
    from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from ament_index_python.packages import get_package_share_directory
    LAUNCH_SUPPORT = True
except ImportError:
    LAUNCH_SUPPORT = False

def inject_h5_attributes_to_ply(ply_path, h5_path, mesh_uuid=None):
    print(f"Injecting attributes from {h5_path} to {ply_path}...")
    if not os.path.exists(h5_path):
        print("Error: H5 file not found.")
        return

    try:
        mesh = trimesh.load(ply_path, force='mesh')
        
        with h5py.File(h5_path, 'r+') as f:
            # Inject UUID if provided
            if mesh_uuid:
                f.attrs['uuid'] = str(mesh_uuid)
                print(f"  Set H5 UUID: {mesh_uuid}")
            
            # Check for attributes
            attributes = ['roughness', 'height_diff', 'border']
            for attr in attributes:
                path = f"mesh/vertex_attributes/{attr}"
                if path in f:
                    data = f[path][:]
                    # Handle both scalar and vector attributes
                    if len(data.shape) > 1 and data.shape[1] == 1:
                        data = data.flatten()
                    
                    if len(data) == len(mesh.vertices):
                        print(f"  Injecting vertex attribute {attr}: {data.shape}")
                        mesh.vertex_attributes[attr] = data
                    elif len(data) == len(mesh.faces):
                        print(f"  Injecting face attribute {attr}: {data.shape}")
                        mesh.face_attributes[attr] = data
                    else:
                        print(f"  Warning: Attribute {attr} size {len(data)} does not match mesh ({len(mesh.vertices)} vertices). Resizing...")
                        # Map to closest possible or pad/truncate
                        if len(data) > len(mesh.vertices):
                            mesh.vertex_attributes[attr] = data[:len(mesh.vertices)]
                        else:
                            new_data = np.zeros(len(mesh.vertices), dtype=data.dtype)
                            new_data[:len(data)] = data
                            mesh.vertex_attributes[attr] = new_data
                else:
                    print(f"  Warning: Attribute {attr} not found in H5.")

            # Add quality attribute as found in original files (default to 1.0)
            if 'quality' not in mesh.vertex_attributes:
                mesh.vertex_attributes['quality'] = np.ones(len(mesh.vertices), dtype=np.float32)
                print("  Added quality attribute (1.0)")
            
            # Match original format: quality on BOTH vertex and face
            if 'quality' not in mesh.vertex_attributes:
                mesh.vertex_attributes['quality'] = np.ones(len(mesh.vertices), dtype=np.float32)
                print("  Added vertex quality attribute (1.0)")
            
            if 'quality' not in mesh.face_attributes:
                mesh.face_attributes['quality'] = np.ones(len(mesh.faces), dtype=np.float32)
                print("  Added face quality attribute (1.0)")
                
            # Add dummy texcoords placeholder if needed for specific header requirements
            # However, trimesh PLY exporter restricts attributes to 1 or 2 dimensions.
            # (N, 3, 2) for face texcoords is not supported directly as a PLY attribute.
            # if 'texcoord' not in mesh.face_attributes:
            #     try:
            #         mesh.face_attributes['texcoord'] = np.zeros((len(mesh.faces), 3, 2), dtype=np.float32)
            #         print("  Added dummy face texcoords placeholder")
            #     except:
            #         pass
            
            # Save back to PLY
            mesh.export(ply_path)
            print(f"  Attributes injected for Type/Format match to {os.path.basename(ply_path)}")
        
    except Exception as e:
        print(f"Failed to inject attributes: {e}")

import time

def get_transform_from_pose(pose_text):
    if not pose_text:
        return np.eye(4)
    vals = [float(x) for x in pose_text.split()]
    # Gazebo pose is x y z r p y
    # Trimesh euler_matrix expects (roll, pitch, yaw)
    if len(vals) == 6:
        x, y, z, r, p, yaw = vals
        # Create rotation matrix
        # Note: trimesh uses 'sxyz' static (extrinsic) by default which matches typical ROS/Gazebo
        mat = trimesh.transformations.euler_matrix(r, p, yaw)
        mat[:3, 3] = [x, y, z]
        return mat
    return np.eye(4)
    if not pose_text:
        return np.eye(4)
    vals = [float(x) for x in pose_text.split()]
    # Gazebo pose is x y z r p y
    # Trimesh euler_matrix expects (roll, pitch, yaw)
    if len(vals) == 6:
        x, y, z, r, p, yaw = vals
        # Create rotation matrix
        # Note: trimesh uses 'sxyz' static (extrinsic) by default which matches typical ROS/Gazebo
        mat = trimesh.transformations.euler_matrix(r, p, yaw)
        mat[:3, 3] = [x, y, z]
        return mat
    return np.eye(4)

def create_high_res_primitive(geometry_node, resolution=64):
    # Returns trimesh object or None
    box = geometry_node.find("box")
    if box is not None:
        size_node = box.find("size")
        if size_node is not None:
            size = [float(x) for x in size_node.text.split()]
            return trimesh.creation.box(extents=size)
        return None

    cylinder = geometry_node.find("cylinder")
    if cylinder is not None:
        r_node = cylinder.find("radius")
        l_node = cylinder.find("length")
        if r_node is not None and l_node is not None:
            r = float(r_node.text)
            l = float(l_node.text)
            # Use resolution for sections
            return trimesh.creation.cylinder(radius=r, height=l, sections=resolution)
        return None

    sphere = geometry_node.find("sphere")
    if sphere is not None:
        r_node = sphere.find("radius")
        if r_node is not None:
            r = float(r_node.text)
            # Map resolution to icosphere subdivisions
            subs = 3 if resolution < 32 else (4 if resolution < 64 else 5)
            return trimesh.creation.icosphere(radius=r, subdivisions=subs)
        return None
        
    plane = geometry_node.find("plane")
    if plane is not None:
        size_node = plane.find("size")
        if size_node is not None:
            size = [float(x) for x in size_node.text.split()] # x y
            m = trimesh.creation.box(extents=[size[0], size[1], 0.01])
            return m
        return None

    return None

def resolve_uri(uri, base_dir):
    """
    Robustly resolve model:// and file:// URIs.
    """
    if uri.startswith("file://"):
        path = uri.replace("file://", "")
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        return path
    
    elif uri.startswith("model://"):
        model_rel_path = uri.replace("model://", "")
        
        # 1. Check GAZEBO_MODEL_PATH
        env_paths = os.environ.get("GAZEBO_MODEL_PATH", "").split(":")
        for p in env_paths:
            candidate = os.path.join(p, model_rel_path)
            if os.path.exists(candidate):
                return candidate
                
        # 2. Check IGN_GAZEBO_RESOURCE_PATH
        env_paths = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "").split(":")
        for p in env_paths:
            candidate = os.path.join(p, model_rel_path)
            if os.path.exists(candidate):
                return candidate

        # 3. Check ~/.gazebo/models
        home = os.path.expanduser("~")
        candidate = os.path.join(home, ".gazebo", "models", model_rel_path)
        if os.path.exists(candidate):
            return candidate

        # 4. Check relative to base_dir (e.g. if models are local)
        # Often models are in ../models relative to a world file
        candidate = os.path.join(base_dir, "..", "models", model_rel_path)
        if os.path.exists(candidate):
            return candidate
            
        candidate = os.path.join(base_dir, model_rel_path)
        if os.path.exists(candidate):
            return candidate

        print(f"    [Warning] Could not resolve URI: {uri}")
        return None

    # Treat as relative path
    candidate = os.path.join(base_dir, uri)
    if os.path.exists(candidate):
        return candidate
        
    return uri 

def extract_meshes_from_sdf(sdf_path, base_dir, resolution=64):
    try:
        tree = ET.parse(sdf_path)
    except ET.ParseError as e:
        print(f"Error parsing SDF: {e}")
        return []

    root = tree.getroot()
    scene_meshes = []

    # Handle both world and model files
    models = root.findall(".//model")
    if root.tag == "model":
        models.append(root)

    print(f"Found {len(models)} model(s) in SDF.")

    for model in models:
        model_name = model.get("name")
        print(f"  Processing Model: {model_name}")
        
        # Model Pose
        m_pose = model.find("pose")
        model_transform = get_transform_from_pose(m_pose.text if m_pose is not None else None)

        # Links
        links = model.findall(".//link")
        
        for link in links:
            l_pose = link.find("pose")
            link_transform = get_transform_from_pose(l_pose.text if l_pose is not None else None)
            
            # Combine transforms
            current_transform = np.dot(model_transform, link_transform)

            # Find Visual Mesh URI (for optimization)
            visual_path = None
            visuals = link.findall("visual")
            for v in visuals:
                geom = v.find("geometry")
                if geom:
                    mesh = geom.find("mesh")
                    if mesh:
                        uri_elem = mesh.find("uri")
                        if uri_elem is not None:
                            resolved = resolve_uri(uri_elem.text, base_dir)
                            if resolved and os.path.exists(resolved):
                                visual_path = resolved
                                break # Take first valid visual
            
            # Prefer collision geometry for navigation mesh
            sources = link.findall("collision")
            if not sources:
                sources = link.findall("visual")

            for source in sources:
                geom = source.find("geometry")
                if geom is None: continue

                # Check for Mesh
                mesh_node = geom.find("mesh")
                if mesh_node is not None:
                    uri_elem = mesh_node.find("uri")
                    if uri_elem is None: continue
                    uri = uri_elem.text
                    
                    scale_elem = mesh_node.find("scale")
                    scale = [float(x) for x in scale_elem.text.split()] if scale_elem is not None else [1.0, 1.0, 1.0]

                    # Resolve URI
                    mesh_path = resolve_uri(uri, base_dir)

                    # Check if file exists and is not empty
                    if not os.path.exists(mesh_path) or os.path.getsize(mesh_path) == 0:
                        print(f"    Warning: Mesh file missing or empty: {mesh_path}")
                        continue
                    
                    try:
                        m = trimesh.load(mesh_path, force='mesh')
                        if isinstance(m, trimesh.Scene):
                            if len(m.geometry) > 0:
                                m = trimesh.util.concatenate([g for g in m.geometry.values()])
                            else:
                                continue
                        
                        m.apply_scale(scale)
                        # Pose of visual/collision
                        v_pose = source.find("pose")
                        if v_pose is not None:
                            v_transform = get_transform_from_pose(v_pose.text)
                            m.apply_transform(v_transform)
                        
                        m.apply_transform(current_transform)
                        
                        scene_meshes.append({
                            'mesh': m,
                            'source_path': mesh_path,
                            'visual_path': visual_path,
                            'type': 'mesh'
                        })
                    except Exception as e:
                        print(f"    Failed to load mesh {mesh_path}: {e}")

                    # Primitives
                    m = create_high_res_primitive(geom, resolution=resolution)
                    if m:
                        # Pose of geometry
                        v_pose = source.find("pose")
                        if v_pose is not None:
                             v_transform = get_transform_from_pose(v_pose.text)
                             m.apply_transform(v_transform)

                        m.apply_transform(current_transform)
                        scene_meshes.append({
                            'mesh': m,
                            'source_path': None,
                            'visual_path': visual_path,
                            'type': 'primitive'
                        })

    # Handle Included Models
    includes = root.findall(".//include")
    print(f"Found {len(includes)} included model(s).")
    for include in includes:
        uri_elem = include.find("uri")
        if uri_elem is None: continue
        uri = uri_elem.text
        
        # Include Pose
        i_pose = include.find("pose")
        include_transform = get_transform_from_pose(i_pose.text if i_pose is not None else None)

        print(f"  Processing Include: {uri}")
        
        # Resolve URI
        model_path = resolve_uri(uri, base_dir)
        if not model_path or not os.path.exists(model_path):
            print(f"    [Warning] Could not resolve included model: {uri}")
            continue

        # If directory, find sdf
        if os.path.isdir(model_path):
            # Check model.config
            config_path = os.path.join(model_path, "model.config")
            sdf_file = "model.sdf" # Default
            if os.path.exists(config_path):
                try:
                    c_tree = ET.parse(config_path)
                    c_root = c_tree.getroot()
                    sdf_elem = c_root.find("sdf")
                    if sdf_elem is not None:
                        sdf_file = sdf_elem.text
                except:
                    pass
            
            model_path = os.path.join(model_path, sdf_file)

            # Recursive call
            sub_mesh_data = extract_meshes_from_sdf(model_path, os.path.dirname(model_path), resolution=resolution)
            # Apply include transform
            for data in sub_mesh_data:
                data['mesh'].apply_transform(include_transform)
                scene_meshes.append(data)
        else:
            print(f"    [Warning] Model SDF not found at: {model_path}")
    
    return scene_meshes

def find_lvr2_tool():
    # Check PATH
    tool = shutil.which("lvr2_hdf5_mesh_tool")
    if tool:
        return tool
    
    # Check build dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    candidate = os.path.join(workspace_root, "build/lvr2/bin/lvr2_hdf5_mesh_tool")
    if os.path.exists(candidate):
        return candidate
        
    return None

def compare_meshes(mesh1, mesh2, label1="Generated", label2="Reference"):
    print(f"\n--- Mesh Comparison: {label1} vs {label2} ---")
    
    # 1. Vertex, Face, and Edge Counts
    v1, f1, e1 = len(mesh1.vertices), len(mesh1.faces), len(mesh1.edges)
    v2, f2, e2 = len(mesh2.vertices), len(mesh2.faces), len(mesh2.edges)
    
    print(f"  Vertices: {v1} vs {v2} (Diff: {v1-v2})")
    print(f"  Faces:    {f1} vs {f2} (Diff: {f1-f2})")
    print(f"  Edges:    {e1} vs {e2} (Diff: {e1-e2})")
    
    # 2. Bounding Boxes
    b1, b2 = mesh1.bounds, mesh2.bounds
    print(f"  Bounds 1: {b1.tolist()}")
    print(f"  Bounds 2: {b2.tolist()}")
    
    # Check if counts match exactly
    if v1 == v2 and f1 == f2 and e1 == e2:
        print("  [OK] Vertex, Face, and Edge counts match EXACTLY.")
    else:
        print("  [Note] Mesh topology differs (common if subdivided or remeshed).")

    # Check if bounds are similar (within 5cm tolerance)
    if np.allclose(b1, b2, atol=0.05):
        print("  [OK] Bounding boxes match within 5cm tolerance.")
    else:
        print("  [!] WARNING: Bounding boxes differ significantly!")
        print(f"      Diff: {np.abs(b1 - b2).tolist()}")

    # 3. Surface Area
    a1, a2 = mesh1.area, mesh2.area
    print(f"  Surface Area: {a1:.4f} vs {a2:.4f} (Diff: {abs(a1-a2):.4f})")
    if abs(a1 - a2) < 0.1 * max(a1, a2):
        print("  [OK] Surface areas match within 10% tolerance.")
    else:
        print("  [!] WARNING: Surface areas differ significantly!")

    # 4. Properties
    p1 = list(mesh1.vertex_attributes.keys())
    p2 = list(mesh2.vertex_attributes.keys())
    print(f"  Properties: {p1} vs {p2}")
    if set(p1) == set(p2):
        print("  [OK] Property keys match.")
    else:
        print(f"  [Note] Property keys differ. (Missing in Ref: {set(p1)-set(p2)}, Extra in Ref: {set(p2)-set(p1)})")

def compare_files_hash(file1, file2):
    import hashlib
    def get_hash(filename):
        with open(filename, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    h1 = get_hash(file1)
    h2 = get_hash(file2)
    
    if h1 == h2:
        print(f"  [OK] Files are identical (MD5: {h1})")
    else:
        print(f"  [!] Files differ (MD5: {h1} vs {h2})")

def main():
    parser = argparse.ArgumentParser(description="Automate Gazebo SDF to Mesh Navigation Pipeline")
    parser.add_argument("input_sdf", help="Path to input SDF/World file")
    parser.add_argument("world_name", help="Name of the new world/environment")
    parser.add_argument("--ref-ply", help="Reference PLY file for comparison")
    parser.add_argument("--ref-dae", help="Reference DAE file for comparison")
    parser.add_argument("--no-subdivide", action="store_true", help="Skip mesh subdivision (identity conversion)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate extraction and resolution, skip processing")
    parser.add_argument("--gen-h5", action="store_true", help="Generate .h5 map file using lvr2 tool (disabled by default)")
    parser.add_argument("--max-edge", type=float, default=0.36, help="Maximum edge length for subdivision (default 0.36m)")
    parser.add_argument("--target-density", type=float, help="Target vertex density per square meter (overrides --max-edge)")
    parser.add_argument("--primitive-resolution", type=int, default=64, help="Resolution for primitives (default 64)")
    parser.add_argument("--weld-threshold", type=float, default=0.01, help="Epsilon threshold for vertex welding (default 0.01m)")
    parser.add_argument("--force-upward", action="store_true", default=True, help="Force normals of near-horizontal faces to point upward (+Z)")
    parser.add_argument("--align-ground", action="store_true", help="Automatically align primary ground normal to +Z")
    parser.add_argument("--flatten-ground", action="store_true", help="Snap traversable ground vertices to Z=0")
    parser.add_argument("--flatten-threshold", type=float, default=0.1, help="Z-range for ground flattening (default 0.1m)")
    parser.add_argument("--shrink-faces", type=float, default=0.0, help="Optional face shrinking factor (0.0 to 1.0)")
    parser.add_argument("--no-build", action="store_true", help="Skip colcon build after generation")
    parser.add_argument("--no-dae", action="store_true", help="Skip DAE export (speeds up generation)")
    parser.add_argument("--filter-steep", type=float, default=0.5, help="Filter out faces with normal.z < threshold (default 0.5 = 60 deg)")
    parser.add_argument("--stitch-threshold", type=float, default=0.0, help="Aggressively stitch border edges within this distance (default 0.0 = off)")
    
    args = parser.parse_args()
    
    input_sdf = os.path.abspath(args.input_sdf)
    world_name = args.world_name
    
    # Paths relative to launch folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Try to find source workspace root
    # We expect the script to be in src/repo/pkg/launch/
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..")) # src/repo/
    
    tutorials_pkg = os.path.join(repo_root, "mesh_navigation_tutorials")
    sim_pkg = os.path.join(repo_root, "mesh_navigation_tutorials_sim")
    
    # 1.5 Check Current Working Directory (common for ros2 launch from workspace root)
    cwd = os.getcwd()
    if not os.path.exists(tutorials_pkg) or not os.path.exists(sim_pkg):
        cwd_tutorials = os.path.join(cwd, "src", "mesh_navigation_tutorials", "mesh_navigation_tutorials")
        cwd_sim = os.path.join(cwd, "src", "mesh_navigation_tutorials", "mesh_navigation_tutorials_sim")
        if os.path.exists(cwd_tutorials) and os.path.exists(cwd_sim):
            tutorials_pkg = cwd_tutorials
            sim_pkg = cwd_sim
            repo_root = os.path.join(cwd, "src", "mesh_navigation_tutorials")

    # Validation: If source folders don't exist, try more hops
    if not os.path.exists(tutorials_pkg) or not os.path.exists(sim_pkg):
        # Try to find 'src' sibling to repo_root or deeper
        candidate = os.path.abspath(os.path.join(script_dir, "../../../..")) # workspace root
        if os.path.exists(os.path.join(candidate, "src")):
            repo_root = os.path.join(candidate, "src", "mesh_navigation_tutorials")
            tutorials_pkg = os.path.join(repo_root, "mesh_navigation_tutorials")
            sim_pkg = os.path.join(repo_root, "mesh_navigation_tutorials_sim")
            
    # Fallback to share directory if still not found (as last resort)
    if not os.path.exists(tutorials_pkg) or not os.path.exists(sim_pkg):
        if LAUNCH_SUPPORT:
            try:
                # If we are in the install space, use share directories
                tutorials_pkg = get_package_share_directory("mesh_navigation_tutorials")
                sim_pkg = get_package_share_directory("mesh_navigation_tutorials_sim")
            except:
                pass
                
    if not os.path.exists(input_sdf):
        # Default path logic: check in sim/worlds if relative
        if not os.path.isabs(args.input_sdf):
            # Try source first
            sim_worlds_source = os.path.join(sim_pkg, "worlds", args.input_sdf)
            if os.path.exists(sim_worlds_source):
                input_sdf = sim_worlds_source
            elif LAUNCH_SUPPORT:
                # Try share directory
                try:
                    sim_pkg_share = get_package_share_directory("mesh_navigation_tutorials_sim")
                    sim_worlds_install = os.path.join(sim_pkg_share, "worlds", args.input_sdf)
                    if os.path.exists(sim_worlds_install):
                        input_sdf = sim_worlds_install
                except:
                    pass
            
    print(f"  Input SDF: {input_sdf}")
    
    maps_dir = os.path.join(tutorials_pkg, "maps")
    models_dir = os.path.join(sim_pkg, "models", world_name)
    worlds_dir = os.path.join(sim_pkg, "worlds")
    
    mesh_output_name = f"{world_name}.ply"
    h5_output_name = f"{world_name}.h5"
    
    ply_dest_path = os.path.join(maps_dir, mesh_output_name)
    dae_dest_path = os.path.join(maps_dir, f"{world_name}.dae")
    h5_dest_path = os.path.join(maps_dir, h5_output_name)
    stl_dest_path = os.path.join(maps_dir, f"{world_name}.stl")
    
    model_stl_path = os.path.join(models_dir, "meshes", f"{world_name}.stl")
    model_dae_path = os.path.join(models_dir, "meshes", f"{world_name}.dae")
    model_ply_path = os.path.join(models_dir, "meshes", f"{world_name}.ply")
    
    if os.path.exists(ply_dest_path):
        os.remove(ply_dest_path)
    if os.path.exists(h5_dest_path):
        os.remove(h5_dest_path)

    print("\n=== Stage 1: Extraction & Validation ===")
    mesh_data_list = extract_meshes_from_sdf(input_sdf, os.path.dirname(input_sdf), resolution=args.primitive_resolution)
    
    if not mesh_data_list:
        print("Error: No meshes extracted. Check SDF file content.")
        sys.exit(1)
        
    print(f"\nExtraction Summary (Resolution: {args.primitive_resolution}):")
    print(f"  Total Sub-meshes: {len(mesh_data_list)}")
    for i, d in enumerate(mesh_data_list):
        m_type = d['type']
        m_source = d['source_path'] if d['source_path'] else "Primitive"
        print(f"  [{i}] Type: {m_type:9} | Verts: {len(d['mesh'].vertices):6} | Source: {m_source}")

    if args.validate_only:
        print("\n[OK] Validation complete. Skipping mesh processing as requested.")
        sys.exit(0)

    print("\n=== Stage 2: Mesh Processing ===")
    
    # 1. Initial Merge with Welding
    print("Merging sub-meshes and welding vertices...")
    initial_merged_mesh = trimesh.util.concatenate([d['mesh'] for d in mesh_data_list])
    
    # Global Vertex Welding
    weld_thresh = args.weld_threshold
    print(f"  Welding vertices (distance threshold: {weld_thresh}m)...")
    
    # Multi-stage welding for robustness
    # 1. Distance-based grouping
    res = trimesh.grouping.group_distance(initial_merged_mesh.vertices, weld_thresh)
    if isinstance(res, tuple) and len(res) == 2:
        centers, groups = res
        merge_count = 0
        if len(groups) > 0:
            new_vertices = initial_merged_mesh.vertices.copy()
            for group in groups:
                if len(group) > 1:
                    avg_pos = np.mean(new_vertices[group], axis=0)
                    new_vertices[group] = avg_pos
                    merge_count += len(group) - 1
            initial_merged_mesh.vertices = new_vertices
            # 2. Stringent coordinate-based merge
            initial_merged_mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=6)
            print(f"  Distance-based welding merged {merge_count} vertices.")
    else:
        initial_merged_mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=3)
    
    # Optional: Filter out steep faces (walls and bottom)
    if args.filter_steep > 0:
        print(f"Filtering out steep/downward faces (normal.z < {args.filter_steep})...")
        keep_mask = initial_merged_mesh.face_normals[:, 2] > args.filter_steep
        initial_merged_mesh.update_faces(keep_mask)
        print(f"  Removed {np.sum(~keep_mask)} faces. Remaining: {len(initial_merged_mesh.faces)}")

    # 2. Topological Repair & Cleanup
    print("Performing topological repair and cleanup...")
    initial_merged_mesh.remove_unreferenced_vertices()
    initial_merged_mesh.remove_infinite_values()
    initial_merged_mesh.update_faces(initial_merged_mesh.unique_faces())
    initial_merged_mesh.update_faces(initial_merged_mesh.nondegenerate_faces())
    
    # Border Diagnostics
    unique_edges, counts = np.unique(initial_merged_mesh.edges_sorted, axis=0, return_counts=True)
    border_edges_count = np.sum(counts == 1)
    print(f"Initial geometry merged. Vertices: {len(initial_merged_mesh.vertices)}, Faces: {len(initial_merged_mesh.faces)}")
    print(f"  Detected {border_edges_count} border edges.")
    
    num_components = trimesh.graph.connected_components(initial_merged_mesh.edges)
    print(f"  Connected components: {len(num_components)}")
    
    # Fix winding and normals early (helps with manifold checks)
    # Skip for very large meshes to avoid hangs
    if len(initial_merged_mesh.faces) < 50000:
        trimesh.repair.fix_winding(initial_merged_mesh)
        trimesh.repair.fix_normals(initial_merged_mesh)
    
    # Fill holes
    if not initial_merged_mesh.is_watertight and len(initial_merged_mesh.faces) < 50000:
        print("  Mesh is not watertight. Attempting to fill holes...")
        trimesh.repair.fill_holes(initial_merged_mesh)
    
    # 2.5 Component Filtering (Remove "dust" / small islands)
    try:
        # Split into connected components
        components = initial_merged_mesh.split(only_watertight=False)
        print(f"  Split into {len(components)} connected components.")
        
        filtered_components = []
        for i, comp in enumerate(components):
            if len(comp.faces) >= 500:
                filtered_components.append(comp)
            else:
                print(f"  Removing small component {i} with {len(comp.faces)} faces.")
                
        if len(filtered_components) < len(components):
            if len(filtered_components) > 0:
                initial_merged_mesh = trimesh.util.concatenate(filtered_components)
                print(f"  Reassembled mesh with {len(filtered_components)} components. (Original: {len(components)})")
            else:
                print("  Warning: All components were small! Keeping largest.")
                components.sort(key=lambda m: m.area, reverse=True)
                initial_merged_mesh = components[0]
        else:
            print("  No small components found (all > 500 faces).")

    except Exception as e:
        print(f"  Component filtering failed: {e}")

    # 2.5a Aggressive Boundary Stitching (Sewing) - Safe Threshold
    if args.stitch_threshold > 0:
        stitch_thresh = args.stitch_threshold
        print(f"Applying aggressive boundary stitching (threshold: {stitch_thresh}m)...")
        # Identify border vertices
        unique_edges, counts = np.unique(initial_merged_mesh.edges_sorted, axis=0, return_counts=True)
        border_edges = unique_edges[counts == 1]
        border_verts = np.unique(border_edges)
        
        if len(border_verts) > 0:
            # Group border vertices by distance
            res = trimesh.grouping.group_distance(initial_merged_mesh.vertices[border_verts], stitch_thresh)
            if isinstance(res, tuple) and len(res) == 2:
                centers, groups = res
                stitch_count = 0
                new_vertices = initial_merged_mesh.vertices.copy()
                for group in groups:
                    if len(group) > 1:
                        # Actual indices in the mesh
                        actual_indices = border_verts[group]
                        avg_pos = np.mean(new_vertices[actual_indices], axis=0)
                        new_vertices[actual_indices] = avg_pos
                        stitch_count += len(group) - 1
                
                initial_merged_mesh.vertices = new_vertices
                # Use a smaller digits_vertex to be more forgiving, but follow up with cleanup
                initial_merged_mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=4)
                print(f"  Boundary stitching merged {stitch_count} border vertices.")
    
    # 2.7 Final Cleaning (Critical for HEM/LVR2 compatibility)
    print("Performing final Stage 2 cleanup for HEM compatibility...")
    initial_merged_mesh.remove_unreferenced_vertices()
    initial_merged_mesh.update_faces(initial_merged_mesh.nondegenerate_faces())
    # If the mesh became non-manifold, try to resolve basic issues
    initial_merged_mesh.update_faces(initial_merged_mesh.nondegenerate_faces())
    if args.align_ground:
        print("Aligning ground normal and centering Z=0...")
        # 3.1 Identify "Up" faces (ground candidates)
        # Weight by area to find a stable principal normal
        up_mask = initial_merged_mesh.face_normals[:, 2] > 0.5
        if np.any(up_mask):
            ground_normals = initial_merged_mesh.face_normals[up_mask]
            ground_areas = initial_merged_mesh.area_faces[up_mask]
            
            # Weighted average normal
            avg_normal = np.average(ground_normals, axis=0, weights=ground_areas)
            avg_normal /= np.linalg.norm(avg_normal)
            
            print(f"  Detected ground normal: {avg_normal}")
            
            # Compute rotation to align avg_normal with [0,0,1]
            z_axis = np.array([0, 0, 1])
            if not np.allclose(avg_normal, z_axis):
                # Using trimesh.geometry.align_vectors
                rotation_matrix = trimesh.geometry.align_vectors(avg_normal, z_axis)
                initial_merged_mesh.apply_transform(rotation_matrix)
                print("  Applied rotation to level ground.")
            
            # 3.2 Vertical Centering (Ground at Z=0)
            # Re-find ground after rotation for precise Z-offset
            upward_faces = np.where(initial_merged_mesh.face_normals[:, 2] > 0.9)[0]
            if len(upward_faces) > 0:
                # Shift ground level to Z=0
                ground_z = np.median(initial_merged_mesh.vertices[initial_merged_mesh.faces[upward_faces]].flatten()[2::3])
                print(f"  Detected ground Z-level: {ground_z:.4f}m. Shifting to Z=0.")
                initial_merged_mesh.vertices[:, 2] -= ground_z
        else:
            print("  Warning: No upward-facing faces found for alignment.")
    
    final_mesh = initial_merged_mesh
    
    # 4. Face Shrinking (Safety Buffer)
    if args.shrink_faces > 0:
        shrink = args.shrink_faces
        print(f"Applying face shrinking (factor: {shrink})...")
        # This will disconnect the mesh (non-manifold)
        # We need to create a new mesh where each face has unique vertices
        new_vertices = []
        new_faces = []
        
        for face in final_mesh.faces:
            pts = final_mesh.vertices[face]
            centroid = np.mean(pts, axis=0)
            # Shrink toward centroid
            shrunk_pts = centroid + (1.0 - shrink) * (pts - centroid)
            
            idx = len(new_vertices)
            new_vertices.extend(shrunk_pts)
            new_faces.append([idx, idx+1, idx+2])
            
        final_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        print(f"  Face shrinking complete. Mesh is now non-manifold (disconnected faces).")

    # 5. Adaptive Resampling (Subdivision)
    max_edge = args.max_edge
    
    if args.target_density:
        # L approx sqrt(2 / (sqrt(3) * D))
        # Where D is vertices per m^2
        max_edge = np.sqrt(2.0 / (np.sqrt(3.0) * args.target_density))
        print(f"Adaptive Resampling: Target Density {args.target_density} v/m^2 -> Max Edge {max_edge:.4f}m")

    # Apply subdivision to increase resolution for MeshNav
    if not args.no_subdivide and args.shrink_faces == 0:
        print(f"Subdividing mesh (max edge length: {max_edge:.4f}m)...")
        # Ensure we don't crash on very large subdivisions
        try:
            vertices, faces = trimesh.remesh.subdivide_to_size(final_mesh.vertices, final_mesh.faces, max_edge)
            # Re-create mesh with process=False to keep the exact subdivision results
            final_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            final_mesh.remove_unreferenced_vertices()
            print(f"Subdivision complete. Vertices: {len(final_mesh.vertices)}, Faces: {len(final_mesh.faces)}")
        except Exception as e:
            print(f"  Subdivision failed: {e}. Keeping original resolution.")
    elif args.shrink_faces > 0:
        print("Skipping subdivision due to face shrinking (requires manifold connectivity).")
    else:
        print("Identity conversion requested. Skipping subdivision.")

    # --- Adaptive Surface Flattening (Post-Subdivision) ---
    if args.flatten_ground:
        print(f"Flattening traversable ground (threshold: {args.flatten_threshold}m)...")
        # Identify vertices on near-horizontal faces that are close to Z=0
        # Re-calc horizontal faces after alignment/subdivision
        horiz_mask = np.abs(final_mesh.face_normals[:, 2]) > 0.99
        horiz_faces = np.where(horiz_mask)[0]
        
        # Vertices belonging to these faces
        ground_verts_indices = np.unique(final_mesh.faces[horiz_faces])
        
        # Filter by Z-threshold
        z_near_zero = np.abs(final_mesh.vertices[ground_verts_indices, 2]) < args.flatten_threshold
        snap_indices = ground_verts_indices[z_near_zero]
        
        final_mesh.vertices[snap_indices, 2] = 0.0
        print(f"  Snapping {len(snap_indices)} vertices to Z=0.")

    # 6. Final Cleanup & Normal Unification
    print("Finalizing mesh (ensuring manifoldness for LVR2)...")
    # HEM required cleaning
    final_mesh.remove_unreferenced_vertices()
    final_mesh.update_faces(final_mesh.nondegenerate_faces())
    
    if args.force_upward:
        print("  Forcing near-horizontal normals to point toward +Z...")
        # Get face normals
        face_normals = final_mesh.face_normals
        # Find faces where Z component is negative but it's mostly horizontal (or just any negative Z for ground)
        # In ROS/Gazebo, +Z is up.
        mask = face_normals[:, 2] < -0.5 # Faces pointing down
        if np.any(mask):
            print(f"    Flipping {np.sum(mask)} downward-pointing faces.")
            final_mesh.faces[mask] = np.fliplr(final_mesh.faces[mask])
            final_mesh.fix_normals()

    # Re-verify normals consistency
    if final_mesh.vertex_normals is None or len(final_mesh.vertex_normals) != len(final_mesh.vertices):
        final_mesh.vertex_normals = trimesh.geometry.weighted_vertex_normals(
            len(final_mesh.vertices),
            final_mesh.faces,
            final_mesh.face_normals,
            final_mesh.face_angles
        )

    print(f"Final Mesh Bounds: {final_mesh.bounds.tolist()}")
    # PLY is critical for MeshNav
    final_mesh.export(ply_dest_path)
    print(f"Saved PLY to: {ply_dest_path}")
    
    if not args.no_dae:
        final_mesh.export(dae_dest_path)
        print(f"Saved DAE to: {dae_dest_path}")
    
    final_mesh.export(stl_dest_path)
    print(f"Saved STL to: {stl_dest_path}")
    
    # Export to model meshes folder (use the processed final_mesh for consistency)
    os.makedirs(os.path.join(models_dir, "meshes"), exist_ok=True)
    final_mesh.export(model_ply_path)
    
    if not args.no_dae:
        final_mesh.export(model_dae_path)
    
    final_mesh.export(model_stl_path)
    
    source_copied = False
    # if len(mesh_data_list) == 1:
    #     item = mesh_data_list[0]
    #     copy_source = item.get('visual_path')
    #     if not copy_source and item['type'] == 'mesh':
    #         copy_source = item['source_path']
    #         
    #     if copy_source:
    #          print(f"optimization: Single source mesh detected ({copy_source}). Copying to preserve textures.")
    #          shutil.copy2(copy_source, dae_dest_path)
    #          source_copied = True
    
    # Export to model meshes folder (use the processed final_mesh for consistency)
    os.makedirs(os.path.join(models_dir, "meshes"), exist_ok=True)
    final_mesh.export(model_ply_path)
    final_mesh.export(model_dae_path)
    
    if not source_copied:
        # If the mesh has no visual information, assign a default color for DAE export
        if final_mesh.visual.kind is None:
            final_mesh.visual = trimesh.visual.ColorVisuals(mesh=final_mesh, vertex_colors=[180, 180, 180, 255])
        final_mesh.export(dae_dest_path)
        # Match original format: .dae for visualization and collision
        os.makedirs(os.path.join(models_dir, "meshes"), exist_ok=True)
        final_mesh.export(model_dae_path)
    
    print(f"Saved DAE to: {dae_dest_path}")
    
    if args.gen_h5:
        print("\n=== Stage 3: H5 Generation ===")
        lvr2_tool = find_lvr2_tool()
        if not lvr2_tool:
            print("[!] Error: lvr2_hdf5_mesh_tool not found in PATH or build/ directory.")
            print("    Please install or build the lvr2 package to generate .h5 maps.")
            sys.exit(1)
            
        print(f"Using tool: {lvr2_tool}")
        cmd = [lvr2_tool, "-i", ply_dest_path, "-o", h5_dest_path]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            print(f"Generated H5: {h5_dest_path}")
            
            # Generate a stable UUID based on the world name
            mesh_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, world_name)
            
            # Injecting back to PLY to match original 'data' expectations
            inject_h5_attributes_to_ply(ply_dest_path, h5_dest_path, mesh_uuid=mesh_uuid)
            # Also inject to the model copy
            inject_h5_attributes_to_ply(model_ply_path, h5_dest_path, mesh_uuid=mesh_uuid)
            
            # Fused attributes are now in the PLY.
            print(f"Match Original Format: Navigation attributes and UUID ({mesh_uuid}) mapped to PLY.")
            # os.remove(h5_dest_path) # Retain for now to check if MBF needs it
        except subprocess.CalledProcessError as e:
            print(f"H5 Generation failed: {e}")
            sys.exit(1)
    else:
        print("\n=== Stage 3: H5 Generation (SKIPPED) ===")
        print("Use --gen-h5 to enable HDF5 map generation.")

    # --- NEW: Comparison Step ---
    if args.ref_ply or args.ref_dae:
        print("\n=== Stage 3.5: Comparison with Reference ===")
        if args.ref_ply:
            if os.path.exists(args.ref_ply):
                ref_mesh = trimesh.load(args.ref_ply, force='mesh')
                compare_meshes(final_mesh, ref_mesh, label1=world_name, label2="Reference")
            else:
                print(f"  [Error] Reference PLY not found: {args.ref_ply}")
        
        if args.ref_dae:
            if os.path.exists(args.ref_dae):
                print(f"--- DAE Comparison: {world_name} vs Reference ---")
                compare_files_hash(dae_dest_path, args.ref_dae)
            else:
                print(f"  [Error] Reference DAE not found: {args.ref_dae}")

    print("\n=== Stage 4: Workspace Organization ===")
    
    # Create model directory and meshes folder
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "meshes"), exist_ok=True)

    # The files were initially exported to tutorials_pkg/maps (ply_dest_path, stl_dest_path, dae_dest_path)
    # Replicate original structure (already exported above)
    print(f"Organized meshes in model directory: {models_dir}/meshes/")

    # Write model.config
    with open(os.path.join(models_dir, "model.config"), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>{world_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Auto Generated</name>
  </author>
  <description>Auto generated from {input_sdf}</description>
</model>
""")

    # Write model.sdf (Exact match of floor_is_lava structure)
    with open(os.path.join(models_dir, "model.sdf"), "w") as f:
        f.write(f"""<?xml version="1.0" ?>
<sdf version="1.6">
<model name="{world_name}">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <mesh>
          <scale>1 1 1</scale>
          <uri>meshes/{world_name}.dae</uri>
        </mesh>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <mesh>
          <scale>1 1 1</scale>
          <uri>meshes/{world_name}.dae</uri>
        </mesh>
      </geometry>
    </visual>
  </link>
</model>
</sdf>
""")

    # Write World file
    world_dest_path = os.path.join(worlds_dir, f"{world_name}.sdf")
    with open(world_dest_path, "w") as f:
        f.write(f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="{world_name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="libignition-gazebo-physics-system.so" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="libignition-gazebo-sensors-system.so" name="ignition::gazebo::systems::Sensors">
        <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="libignition-gazebo-scene-broadcaster-system.so" name="ignition::gazebo::systems::SceneBroadcaster"></plugin>
    <plugin filename="libignition-gazebo-user-commands-system.so" name="gz::sim::systems::UserCommands"></plugin>
    
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <include>
      <uri>model://{world_name}</uri>
    </include>
    <model name="spawn_platform">
        <static>true</static>
        <pose>0 0 0 0 0 0</pose> 
    </model>
  </world>
</sdf>
""")
    
    print(f"Created World file: {world_dest_path}")

    # --- NEW: Launch File Generation ---
    print("\n=== Stage 4.5: Launch File Generation ===")
    launch_dir = os.path.join(tutorials_pkg, "launch")
    os.makedirs(launch_dir, exist_ok=True)
    launch_file_path = os.path.join(launch_dir, f"launch_{world_name}.py")
    
    with open(launch_file_path, "w") as f:
        f.write(f"""import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_mesh_navigation_tutorials = get_package_share_directory("mesh_navigation_tutorials")
    
    # Navigation and Visualization Arguments
    localization = LaunchConfiguration("localization", default="ground_truth")
    start_rviz = LaunchConfiguration("start_rviz", default="True")
    
    return LaunchDescription([
        DeclareLaunchArgument("localization", default_value="ground_truth"),
        DeclareLaunchArgument("start_rviz", default_value="True"),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_mesh_navigation_tutorials, "launch", "mesh_navigation_tutorials_launch.py")
            ),
            launch_arguments={{
                "world_name": "{world_name}",
                "map_name": "{world_name}",
                "localization": localization,
                "start_rviz": start_rviz
            }}.items()
        )
    ])
""")
    print(f"Created dedicated launch file: {launch_file_path}")

    print("\n=== Stage 5: Building ===")
    # Attempt to find the source workspace root by searching upwards for 'src'
    workspace_root = script_dir
    found_root = False
    for _ in range(6): # Search up to 6 levels
        if os.path.exists(os.path.join(workspace_root, "src")):
            found_root = True
            break
        workspace_root = os.path.dirname(workspace_root)
        if workspace_root == os.path.dirname(workspace_root): # Reached /
            break
            
    if not found_root:
        print("Note: Source workspace root not found (searched up 6 levels). Skipping Stage 5 (Building).")
        print("      This is expected if running from an install/share directory without a source overlay.")
        return

    if args.no_build:
        print("\n=== Stage 5: Building (SKIPPED) ===")
        return

    build_cmd = ["colcon", "build", "--packages-select", "mesh_navigation_tutorials_sim", "mesh_navigation_tutorials", "--allow-overriding", "mesh_navigation_tutorials", "mesh_navigation_tutorials_sim"]
    print(f"Running: {' '.join(build_cmd)}")
    
    try:
        subprocess.check_call(build_cmd, cwd=workspace_root)
        print("Build successful.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)
    
    print("\n=== Stage 6: Launching & Monitoring ===")
    launch_cmd = ["ros2", "launch", "mesh_navigation_tutorials", "mesh_navigation_tutorials_launch.py", f"world_name:={world_name}"]
    launch_cmd_str = f"source install/setup.bash && {' '.join(launch_cmd)}"
    print(f"Running launch command: {launch_cmd_str}")

    process = subprocess.Popen(['/bin/bash', '-c', launch_cmd_str], cwd=workspace_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    print("----------------------------------------------------------------")
    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip()) # Mirror output
                
                if "TF Transform Cache" in line and "timestamp" in line:
                    print("\n\033[91m[!] CRITICAL: TF TRANSFORM CACHE ERROR DETECTED [!]\033[0m")
                if "Start pose in collision" in line or "Costmap is not valid" in line:
                    print("\n\033[93m[!] WARNING: ROBOT START POSE IN COLLISION [!]\033[0m")
                    
    except KeyboardInterrupt:
        print("\nStopping launch...")
        process.terminate()
        
    print("Launch finished.")


def launch_setup(context, *args, **kwargs):
    input_sdf = LaunchConfiguration('input_sdf').perform(context)
    world_name = LaunchConfiguration('world_name').perform(context)
    gen_h5 = LaunchConfiguration('gen_h5').perform(context).lower() == 'true'
    max_edge = LaunchConfiguration('max_edge').perform(context)
    target_density = LaunchConfiguration('target_density').perform(context)
    primitive_resolution = LaunchConfiguration('primitive_resolution').perform(context)
    align_ground = LaunchConfiguration('align_ground').perform(context).lower() == 'true'
    flatten_ground = LaunchConfiguration('flatten_ground').perform(context).lower() == 'true'
    flatten_threshold = LaunchConfiguration('flatten_threshold').perform(context)
    shrink_faces = LaunchConfiguration('shrink_faces').perform(context)
    
    # Mock sys.argv for main()
    sys.argv = [sys.argv[0], input_sdf, world_name]
    if gen_h5:
        sys.argv.append("--gen-h5")
    if max_edge:
        sys.argv.extend(["--max-edge", max_edge])
    if target_density:
        sys.argv.extend(["--target-density", target_density])
    if primitive_resolution:
        sys.argv.extend(["--primitive-resolution", primitive_resolution])
    if align_ground:
        sys.argv.append("--align-ground")
    if flatten_ground:
        sys.argv.append("--flatten-ground")
    if flatten_threshold:
        sys.argv.extend(["--flatten-threshold", flatten_threshold])
    if shrink_faces:
        sys.argv.extend(["--shrink-faces", shrink_faces])
        
    print(f"\n[Launch] Initializing generation for: {world_name} (Max Edge: {max_edge}m, Resolution: {primitive_resolution})")
    main()
    
    # After generation, include the newly created launch file
    pkg_mesh_navigation_tutorials = get_package_share_directory("mesh_navigation_tutorials")
    launch_file = os.path.join(pkg_mesh_navigation_tutorials, "launch", f"launch_{world_name}.py")
    
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file),
            launch_arguments={
                "localization": LaunchConfiguration("localization"),
                "start_rviz": LaunchConfiguration("start_rviz")
            }.items()
        )
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("input_sdf", description="Path to input SDF/World file"),
        DeclareLaunchArgument("world_name", description="Name of the new world/environment"),
        DeclareLaunchArgument("gen_h5", default_value="false", description="Generate H5 map file"),
        DeclareLaunchArgument("max_edge", default_value="0.36", description="Maximum edge length for subdivision"),
        DeclareLaunchArgument("target_density", default_value="", description="Target vertex density per square meter"),
        DeclareLaunchArgument("primitive_resolution", default_value="64", description="Resolution for primitives"),
        DeclareLaunchArgument("align_ground", default_value="false", description="Align ground normal to +Z"),
        DeclareLaunchArgument("flatten_ground", default_value="false", description="Snap ground vertices to Z=0"),
        DeclareLaunchArgument("flatten_threshold", default_value="0.1", description="Z-range for ground flattening"),
        DeclareLaunchArgument("shrink_faces", default_value="0.0", description="Face shrinking factor"),
        DeclareLaunchArgument("localization", default_value="ground_truth"),
        DeclareLaunchArgument("start_rviz", default_value="True"),
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == "__main__":
    main()


