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
import gc
import hashlib

import collections

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

class Stopwatch:
    def __init__(self, name):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] START")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        print(f"[{self.name}] END - Duration: {duration:.4f}s")


def clean_mesh_iterative(mesh, iterations=3, verbose=True):
    """
    Perform iterative cleaning to repair non-manifold geometry and duplicated elements.
    Optimized to reduce redundant calculations.
    """
    if verbose: print(f"  Starting iterative cleaning (max {iterations} iterations)...")
    
    # 1. Remove Duplicates (expensive, do once efficiently)
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    else:
        mesh.update_faces(mesh.unique_faces())
    
    mesh.remove_unreferenced_vertices()

    for i in range(iterations):
        dirty = False
        n_faces_start = len(mesh.faces)
        
        # 2. Degenerate Faces
        try:
            # remove_degenerate_faces() is usually fast
            mesh.remove_degenerate_faces()
            if len(mesh.faces) < n_faces_start:
                 if verbose: print(f"    [Iter {i}] Removed {n_faces_start - len(mesh.faces)} degenerate faces.")
                 dirty = True
        except:
            pass

        mesh.remove_unreferenced_vertices()
        
        # 3. Non-Manifold Edges (Expensive check - only do if strictly needed or last iter)
        # Replacing slow is_volume check with explicit edge check only on final iterations
        if i == iterations - 1:
             try:
                 edges = mesh.edges_sorted
                 unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
                 non_manifold_edges = unique_edges[counts > 2]
                 
                 if len(non_manifold_edges) > 0:
                     if verbose: print(f"    [Iter {i}] Found {len(non_manifold_edges)} non-manifold edges. Stripping faces...")
                     # Find faces containing these edges and remove them
                     edge_hash = set([tuple(e) for e in non_manifold_edges])
                     face_edges = mesh.edges_sorted_face
                     mask = np.array([any(tuple(e) in edge_hash for e in fe) for fe in face_edges])
                     mesh.update_faces(~mask)
                     dirty = True
             except Exception as e:
                 if verbose: print(f"    Warning: Non-manifold edge repair failed: {e}")

        if not dirty:
            if verbose: print(f"    [Iter {i}] Mesh converged (no degenerate faces found).")
            break
            
    # Final normals fix - moved out of loop
    try:
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
    except Exception as e:
        print(f"    Warning: Normal repair failed: {e}")
        
    return mesh

def inject_h5_attributes_to_ply(ply_path, h5_path, mesh_uuid=None):
    print(f"Injecting attributes from {h5_path} to {ply_path}...")
    if not os.path.exists(h5_path):
        print("Error: H5 file not found.")
        return

    try:
        # CRITICAL: Load with process=False to prevent re-ordering or merging vertices.
        # This ensures the H5 arrays (generated from the PLY on disk) map 1:1 to this loaded mesh.
        mesh = trimesh.load(ply_path, force='mesh', process=False)
        
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
                    
                    # Store UUID in metadata if available
                    if mesh_uuid:
                         # Trimesh stores comments in metadata, which PLY writers often use
                         mesh.metadata['uuid'] = str(mesh_uuid)
                         # Also try to set it as a custom header if possible (trimesh support varies)
                         mesh.metadata['obj_info'] = {'uuid': str(mesh_uuid)}
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
            
            # Save back to PLY
            try:
                mesh.export(ply_path)
                print(f"  Attributes injected for Type/Format match to {os.path.basename(ply_path)}")
            except Exception as e:
                 print(f"  Error saving PLY: {e}")

    except Exception as e:
        print(f"Failed to inject attributes: {e}")
        import traceback
        traceback.print_exc()

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

def create_high_res_primitive(geometry_node, resolution=64):
    # Returns trimesh object or None
    box = geometry_node.find("box")
    if box is not None:
        size_node = box.find("size")
        if size_node is not None:
            size_str = size_node.text
            # print(f"DEBUG: Found box with size {size_str}")
            size = [float(x) for x in size_str.split()]
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

    polyline = geometry_node.find("polyline")
    if polyline is not None:
        points = []
        for point in polyline.findall("point"):
            coords = [float(x) for x in point.text.split()]
            if len(coords) >= 2:
                points.append(coords[:2])
        
        height_node = polyline.find("height")
        height = float(height_node.text) if height_node is not None else 1.0

        if len(points) >= 3:

             # Use shapely for polygon creation (trimesh.path.polygons is deprecated/removed)
             try:
                 from shapely.geometry import Polygon
                 poly = Polygon(points)
             except ImportError:
                 # Fallback if shapely is not available or for older trimesh versions compatibility if needed
                 # But standard ROS 2 desktop includes shapely
                 import shapely.geometry
                 poly = shapely.geometry.Polygon(points)

             # Extrude along Z axis to create a 3D shape from the 2D footprint
             m = trimesh.creation.extrude_polygon(poly, height)
             # trimesh extrusion goes from Z=0 to Z=height, which matches typical SDF definitions
             return m

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
        
        # 0. Check relative to base_dir (Priority -> localized models)
        # Check ../models (common structure: world_dir/../models/model_name)
        candidate = os.path.join(base_dir, "..", "models", model_rel_path)
        if os.path.exists(candidate):
            return candidate
            
        candidate = os.path.join(base_dir, model_rel_path)
        if os.path.exists(candidate):
            return candidate

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

        print(f"    [Warning] Could not resolve URI: {uri}")
        return None

    # Treat as relative path
    candidate = os.path.join(base_dir, uri)
    if os.path.exists(candidate):
        return candidate
        
    return uri 

def extract_meshes_from_sdf(sdf_path, base_dir, resolution=64, exclude_list=[]):
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
        
        # Check exclusion
        if any(ex in model_name for ex in exclude_list):
            print(f"  [Exclude] Skipping model: {model_name}")
            continue

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

                # Primitives (Unindented to run if no mesh tag found, or in addition)
                # Note: 'elif' would be better if mutually exclusive, but 'if' is safe as create_high_res_primitive checks tags.
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
    
    # Check common workspace locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search upwards for 'build' or 'install'
    current_dir = script_dir
    for _ in range(6):
        # Check build/lvr2/bin
        candidate_build = os.path.join(current_dir, "build/lvr2/bin/lvr2_hdf5_mesh_tool")
        if os.path.exists(candidate_build):
            return candidate_build
            
        # Check install/lvr2/lib/lvr2/bin (colcon default for some) or install/lvr2/bin
        candidate_install = os.path.join(current_dir, "install/lvr2/lib/lvr2/bin/lvr2_hdf5_mesh_tool")
        if os.path.exists(candidate_install):
            return candidate_install
            
        candidate_install_bin = os.path.join(current_dir, "install/lvr2/bin/lvr2_hdf5_mesh_tool")
        if os.path.exists(candidate_install_bin):
            return candidate_install_bin
            
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break
        current_dir = parent
        
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
    parser.add_argument("world_name", nargs='?', help="Name of the new world/environment (optional, defaults to SDF filename)")
    parser.add_argument("--maps-dir", help="Directory to save map files (PLY, H5). Default: auto-detect 'mesh_navigation_tutorials/maps'")
    parser.add_argument("--models-dir", help="Base directory to save model files. The script will create a subdirectory <world_name> here. Default: auto-detect 'mesh_navigation_tutorials_sim/models'")
    parser.add_argument("--ref-ply", help="Reference PLY file for comparison")
    parser.add_argument("--ref-dae", help="Reference DAE file for comparison")
    parser.add_argument("--no-subdivide", action="store_true", help="Skip mesh subdivision (identity conversion)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate extraction and resolution, skip processing")

    parser.add_argument("--max-edge", type=float, default=0.36, help="Maximum edge length for subdivision (default 0.20m to capture 0.3m roughness)")
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
    parser.add_argument("--filter-steep", type=float, default=0.0, help="Filter out faces with normal.z < threshold (default 0.0 = off)")
    parser.add_argument("--stitch-threshold", type=float, default=0.0, help="Aggressively stitch border edges within this distance (default 0.0 = off)")
    parser.add_argument("--single-layer", action="store_true", help="Optimize for Single Layer MeshNav (High density, clean topology, flattened ground).")
    parser.add_argument("--clean-iter", type=int, default=0, help="Number of iterative cleaning passes (default 0, auto-enabled by --single-layer)")
    parser.add_argument("--exclude", nargs='+', default=[], help="List of model names/substrings to exclude (e.g. 'wall obstacle')")
    
    args = parser.parse_args()
    
    input_sdf = os.path.abspath(args.input_sdf)
    
    # Auto-derive world_name if not provided
    if not args.world_name:
        base = os.path.basename(input_sdf)
        args.world_name = os.path.splitext(base)[0]
        print(f"Auto-derived world_name: {args.world_name}")

    world_name = args.world_name
    
    # Sanitize world_name to avoid double extensions or artifacts
    for ext in ['.sdf', '.world', '.ply', '.dae', '.stl']:
        if world_name.lower().endswith(ext):
            world_name = world_name[:-len(ext)]
            print(f"Sanitized world_name: {args.world_name} -> {world_name}")
            break

    # Directory Setup
    maps_dir = None
    models_root_dir = None
    tutorials_pkg = None
    sim_pkg = None
    
    # If explicit paths are provided, use them
    if args.maps_dir:
        maps_dir = os.path.abspath(args.maps_dir)
    if args.models_dir:
        models_root_dir = os.path.abspath(args.models_dir)

    # Auto-detection logic if paths are missing
    if not maps_dir or not models_root_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Strategies to find workspace root
        repo_root = os.path.abspath(os.path.join(script_dir, "..", "..")) # src/repo/
        tutorials_pkg = os.path.join(repo_root, "mesh_navigation_tutorials")
        sim_pkg = os.path.join(repo_root, "mesh_navigation_tutorials_sim")
        
        # Check CWD
        cwd = os.getcwd()
        if not os.path.exists(tutorials_pkg) or not os.path.exists(sim_pkg):
             cwd_tutorials = os.path.join(cwd, "src", "mesh_navigation_tutorials", "mesh_navigation_tutorials")
             cwd_sim = os.path.join(cwd, "src", "mesh_navigation_tutorials", "mesh_navigation_tutorials_sim")
             if os.path.exists(cwd_tutorials) and os.path.exists(cwd_sim):
                 tutorials_pkg = cwd_tutorials
                 sim_pkg = cwd_sim
                 repo_root = os.path.join(cwd, "src", "mesh_navigation_tutorials")
        
        # Try deeper search
        if not os.path.exists(tutorials_pkg) or not os.path.exists(sim_pkg):
             candidate = os.path.abspath(os.path.join(script_dir, "../../../.."))
             if os.path.exists(os.path.join(candidate, "src")):
                 repo_root = os.path.join(candidate, "src", "mesh_navigation_tutorials")
                 tutorials_pkg = os.path.join(repo_root, "mesh_navigation_tutorials")
                 sim_pkg = os.path.join(repo_root, "mesh_navigation_tutorials_sim")

        # Fallback to share directory (installed)
        if not os.path.exists(tutorials_pkg) or not os.path.exists(sim_pkg):
            if LAUNCH_SUPPORT:
                try:
                    tutorials_pkg = get_package_share_directory("mesh_navigation_tutorials")
                    sim_pkg = get_package_share_directory("mesh_navigation_tutorials_sim")
                except:
                    pass

        # Assign defaults if found
        if not maps_dir and os.path.exists(tutorials_pkg):
            maps_dir = os.path.join(tutorials_pkg, "maps")
            
        if not models_root_dir and os.path.exists(sim_pkg):
            models_root_dir = os.path.join(sim_pkg, "models")
            
    # Final Fallback: local output if nothing found
    if not maps_dir:
        maps_dir = os.path.join(os.getcwd(), "maps_output")
        print(f"[Info] Maps directory not found. Using local: {maps_dir}")
    if not models_root_dir:
        models_root_dir = os.path.join(os.getcwd(), "models_output")
        print(f"[Info] Models directory not found. Using local: {models_root_dir}")

    models_dir = os.path.join(models_root_dir, world_name)
    
    if sim_pkg:
        worlds_dir = os.path.join(sim_pkg, "worlds")
    else:
        # Assuming sibling directory structure if generic
        worlds_dir = os.path.abspath(os.path.join(models_root_dir, "..", "worlds"))
        
    # Ensure directories exist
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(worlds_dir, exist_ok=True)
    
    # Handle Input SDF resolution if relative
    if not os.path.exists(input_sdf) and not os.path.isabs(input_sdf):
         # Try common locations
         candidates = [
             os.path.join(os.getcwd(), input_sdf),
             os.path.join(sim_pkg, "worlds", input_sdf) if 'sim_pkg' in locals() else None
         ]
         for c in candidates:
             if c and os.path.exists(c):
                 input_sdf = c
                 break
                 
    print(f"  Input SDF: {input_sdf}")
    print(f"  Maps output:   {maps_dir}")
    print(f"  Models output: {models_dir}")
    
    mesh_output_name = f"{world_name}.ply"
    h5_output_name = f"{world_name}.h5"
    
    ply_dest_path = os.path.join(maps_dir, mesh_output_name)
    dae_dest_path = os.path.join(maps_dir, f"{world_name}.dae")
    # h5_dest_path = os.path.join(maps_dir, h5_output_name)
    h5_dest_path = h5_output_name
    stl_dest_path = os.path.join(maps_dir, f"{world_name}.stl")
    
    model_stl_path = os.path.join(models_dir, "meshes", f"{world_name}.stl")
    model_dae_path = os.path.join(models_dir, "meshes", f"{world_name}.dae")
    model_ply_path = os.path.join(models_dir, "meshes", f"{world_name}.ply")
    
    if os.path.exists(ply_dest_path):
        os.remove(ply_dest_path)
    if os.path.exists(h5_dest_path):
        os.remove(h5_dest_path)

    print("\n=== Stage 1: Extraction & Validation ===")
    with Stopwatch("Stage 1: Extraction"):
        mesh_data_list = extract_meshes_from_sdf(input_sdf, os.path.dirname(input_sdf), resolution=args.primitive_resolution, exclude_list=args.exclude)

    
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

    # --- Single Layer Optimizations ---
    if args.single_layer:
        print("\n=== Single Layer MeshNav Optimization Enabled ===")
        if not args.target_density:
            args.target_density = 100.0 # Reduced from 200.0 to prevent OOM on large maps
            print(f"  [Auto] Set target density to {args.target_density} v/m^2")
        
        args.align_ground = True
        print("  [Auto] Enabled Ground Alignment (Z-Up)")
        args.flatten_ground = True
        print("  [Auto] Enabled Ground Flattening")
        args.force_upward = True
        
        if args.clean_iter == 0:
            args.clean_iter = 3
            print("  [Auto] Enabled Iterative Cleaning (3 passes)")
            
        if args.filter_steep == 0.0:
            args.filter_steep = -0.5
            print("  [Auto] Enabled Bottom-Face Filtering (Keep Normal.Z > -0.5). Preserving Walls.")


    print("\n=== Stage 2: Mesh Processing ===")
    
    with Stopwatch("Stage 2: Processing"):
        # 1. Initial Merge with Welding
        print("Merging sub-meshes and welding vertices...")
        initial_merged_mesh = trimesh.util.concatenate([d['mesh'] for d in mesh_data_list])
        
        # Cleanup source list to free memory
        del mesh_data_list
        gc.collect() 
        
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
            with Stopwatch("Stringent Merge"):
                initial_merged_mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=6)
            print(f"  Distance-based welding merged {merge_count} vertices.")
    else:
        with Stopwatch("Vertex Merge"):
             initial_merged_mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=3)

    
    # Optional: Filter out steep faces (walls and bottom)
    # Optional: Filter out steep faces (walls and bottom)
    if args.filter_steep != 0.0:
        # Ensure normals are consistent before filtering so we don't skip faces due to stale normals
        _ = initial_merged_mesh.face_normals
        
        print(f"Filtering faces per threshold (normal.z > {args.filter_steep})...")
        keep_mask = initial_merged_mesh.face_normals[:, 2] > args.filter_steep
        initial_merged_mesh.update_faces(keep_mask)
        print(f"  Removed {np.sum(~keep_mask)} faces. Remaining: {len(initial_merged_mesh.faces)}")

    # 2. Topological Repair & Cleanup
    print("Performing topological repair and cleanup...")
    with Stopwatch("Iterative Cleaning"):
        if args.clean_iter > 0:
            initial_merged_mesh = clean_mesh_iterative(initial_merged_mesh, iterations=args.clean_iter)
        else:
            initial_merged_mesh.remove_unreferenced_vertices()
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
    
    # Fill holes - SKIP for single layer to avoid re-creating volume
    if not args.single_layer and args.filter_steep == 0 and not initial_merged_mesh.is_watertight and len(initial_merged_mesh.faces) < 50000:
        print("  Mesh is not watertight. Attempting to fill holes...")
        trimesh.repair.fill_holes(initial_merged_mesh)
    elif args.single_layer:
        print("  [Single Layer] Skipping hole filling to preserve open surface.")
    
    # 2.5 Component Filtering (Remove "dust" / small islands)
    try:
        # Split into connected components
        components = initial_merged_mesh.split(only_watertight=False)
        print(f"  Split into {len(components)} connected components.")
        
        filtered_components = []
        for i, comp in enumerate(components):
            if len(comp.faces) >= 1:
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
            print("  No small components found (all > 1 faces).")

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
    # Stage 2: Pre-Subdivision Sanitization
    print("Sanitizing geometry before subdivision...")
    final_mesh.process(validate=True)
    
    if not args.no_subdivide and args.shrink_faces == 0:
        print(f"Subdividing mesh (max edge length: {max_edge:.4f}m)...")
        
        # Smart Subdivision Check
        current_area = float(initial_merged_mesh.area)
        estimated_faces = current_area * args.target_density * 2.5 if args.target_density else (current_area / (max_edge**2)) * 4
        print(f"  Estimated Faces after subdivision: ~{int(estimated_faces)}")
        
        if estimated_faces > 80000: # Lowered to 800k for safety
             print(f"  [Warning] Estimated face count {int(estimated_faces / 1e3)}k exceeds limit (800k).")
             print(f"  [Auto] Adjusting max_edge to reduce load...")
             # Relax density to safe level
             # Est = Area * D * 2.5 -> D = Est / (Area * 2.5)
             safe_density = 80000 / (current_area * 2.5)
             new_max_edge = np.sqrt(2.0 / (np.sqrt(3.0) * safe_density))
             
             print(f"  Adjusted max_edge: {max_edge:.4f}m -> {new_max_edge:.4f}m (Effective Density: {safe_density:.1f})")
             max_edge = new_max_edge
        
        # Ensure we don't crash on very large subdivisions
        try:
            with Stopwatch("Subdivision"):
                vertices, faces = trimesh.remesh.subdivide_to_size(final_mesh.vertices, final_mesh.faces, max_edge)
                
            # Re-create mesh with process=False to keep the exact subdivision results
            final_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            final_mesh.remove_unreferenced_vertices()
            
            # Explicit GC
            gc.collect()
            
            print(f"Subdivision complete. Vertices: {len(final_mesh.vertices)}, Faces: {len(final_mesh.faces)}")
            
            # Stage 2: Post-Subdivision Normal Fix
            print("Ensuring consistent winding after subdivision...")
            final_mesh.fix_normals()
            
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

    def split_bowties(mesh):
        """
        Manually split bow-tie vertices (one vertex shared by two disjoint face loops).
        LVR2 HEM panics if these exist.
        """
        import collections
        
        # 1. Map vertices to adjacent faces
        v_to_f = collections.defaultdict(list)
        for f_idx, face in enumerate(mesh.faces):
            for v_idx in face:
                v_to_f[v_idx].append(f_idx)
        
        splits_needed = {} # {old_v_idx: [[face_ids1], [face_ids2], ...]}
        
        for v_idx, face_indices in v_to_f.items():
            if len(face_indices) < 2: continue
            
            # Find clusters of faces that share an edge AT this vertex
            clusters = []
            faces_to_process = set(face_indices)
            
            while faces_to_process:
                seed = faces_to_process.pop()
                cluster = {seed}
                queue = [seed]
                while queue:
                    f1 = queue.pop()
                    # Check other neighbors of f1 in the original set
                    for f2 in list(faces_to_process):
                        # Do f1 and f2 share an edge that includes v_idx?
                        shared_v = set(mesh.faces[f1]) & set(mesh.faces[f2])
                        if len(shared_v) >= 2 and v_idx in shared_v:
                            cluster.add(f2)
                            faces_to_process.remove(f2)
                            queue.append(f2)
                clusters.append(list(cluster))
            
            if len(clusters) > 1:
                splits_needed[v_idx] = clusters
        
        if not splits_needed:
            return mesh
            
        print(f"  [Repair] Splitting {len(splits_needed)} bow-tie vertices...")
        
        new_vertices = list(mesh.vertices)
        new_faces = mesh.faces.copy()
        
        for old_v, face_groups in splits_needed.items():
            # Keep the first group with the original vertex
            # Create new vertices for the rest
            for group in face_groups[1:]:
                new_v_idx = len(new_vertices)
                new_vertices.append(mesh.vertices[old_v])
                # Update face indices in this group
                for f_idx in group:
                    face = new_faces[f_idx]
                    new_faces[f_idx] = [new_v_idx if v == old_v else v for v in face]
        
        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)

    # Final cleanup and manifold enforcement
    final_mesh = split_bowties(final_mesh)
    final_mesh.process()

    # Strict Non-Manifold Edge Removal
    def strip_non_manifold(mesh):
        edges_unique_inverse = mesh.edges_unique_inverse # Mapping of edges to unique edge indices
        counts = np.bincount(edges_unique_inverse)
        non_manifold_unique_indices = np.where(counts > 2)[0]
        
        if len(non_manifold_unique_indices) > 0:
            print(f"  [Repair] Stripping faces due to {len(non_manifold_unique_indices)} non-manifold edges...")
            is_bad_unique_edge = np.isin(edges_unique_inverse, non_manifold_unique_indices)
            is_bad_face = is_bad_unique_edge.reshape((-1, 3)).any(axis=1)
            mesh.update_faces(~is_bad_face)
            mesh.process()
        return mesh

    final_mesh = strip_non_manifold(final_mesh)

    # Re-verify normals consistency and sanitize
    if final_mesh.vertex_normals is None or len(final_mesh.vertex_normals) != len(final_mesh.vertices):
        final_mesh.vertex_normals = trimesh.geometry.weighted_vertex_normals(
            len(final_mesh.vertices),
            final_mesh.faces,
            final_mesh.face_normals,
            final_mesh.face_angles
        )
    
    # Strict Normal Sanitization (Prevent Embree ray.valid() assertion failure)
    # Check for NaNs or Infs in vertex normals and replace with default [0, 0, 1]
    # Use a copy to avoid read-only assignment errors
    sanitized_normals = final_mesh.vertex_normals.copy()
    invalid_mask = ~np.all(np.isfinite(sanitized_normals), axis=1)
    if np.any(invalid_mask):
        n_invalid = np.sum(invalid_mask)
        print(f"  [Repair] Found {n_invalid} invalid (NaN/Inf) vertex normals. Replacing with [0, 0, 1].")
        sanitized_normals[invalid_mask] = [0.0, 0.0, 1.0]
    
    # Ensure all normals are normalized
    norms = np.linalg.norm(sanitized_normals, axis=1)
    zero_norms = (norms < 1e-6)
    if np.any(zero_norms):
        sanitized_normals[zero_norms] = [0.0, 0.0, 1.0]
        norms[zero_norms] = 1.0
    sanitized_normals /= norms.reshape((-1, 1))
    
    final_mesh.vertex_normals = sanitized_normals

    # Aggressive cleaning to ensure count matches LVR2 internal loading
    final_mesh.process()
    
    # Resolve LVR2 HEM Panic: Enforce manifoldness
    # hole filling is disabled as it often creates non-manifold geometry in complex meshes
    if not final_mesh.is_watertight:
        print("  [Note] Mesh is not watertight (has holes), which is acceptable for LVR2 if manifold.")
    
    print(f"Final Mesh Stats: Vertices={len(final_mesh.vertices)}, Faces={len(final_mesh.faces)}")
    print(f"Final Mesh Bounds: {final_mesh.bounds.tolist()}")
    
    # --- Export to Maps Directory (User Requirement: Only PLY) ---
    # Validate before export to ensure no stale attributes leak into H5 generation
    # 'include_attributes=False' ensures a clean PLY header
    final_mesh.export(ply_dest_path, include_attributes=False)
    print(f"Saved PLY to: {ply_dest_path}")
    
    # final_mesh.export(dae_dest_path)
    # final_mesh.export(stl_dest_path)
    
    # --- Export to Models Directory (User Requirement: PLY and DAE) ---
    os.makedirs(os.path.join(models_dir, "meshes"), exist_ok=True)
    
    final_mesh.export(model_ply_path)
    print(f"Saved Model PLY to: {model_ply_path}")
    
    if not args.no_dae:
        # If the mesh has no visual information, assign a default color for DAE export
        if final_mesh.visual.kind is None:
            final_mesh.visual = trimesh.visual.ColorVisuals(mesh=final_mesh, vertex_colors=[180, 180, 180, 255])
            
        final_mesh.export(model_dae_path)
        print(f"Saved Model DAE to: {model_dae_path}")
        
    # Removed STL export to models directory as per request
    # final_mesh.export(model_stl_path)
    
    print("\n=== Stage 3: H5 Generation ===")
    lvr2_tool = find_lvr2_tool()
    
    with Stopwatch("Stage 3: H5 Generation"):
        if not lvr2_tool:
            print("[!] Error: lvr2_hdf5_mesh_tool not found in PATH or build/ directory.")
            print("    Please install or build the lvr2 package to generate .h5 maps.")
            sys.exit(1)
            
        print(f"Using tool: {lvr2_tool}")
        
        # Atomic Generation: Remove existing H5 to prevent appending/mismatch
        if os.path.exists(h5_dest_path):
            try:
                os.remove(h5_dest_path)
                print(f"  Removed existing H5 file: {h5_dest_path}")
            except OSError as e:
                print(f"  Warning: Could not remove existing H5: {e}")

        cmd = [lvr2_tool, "-i", ply_dest_path, "-o", h5_dest_path]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            print(f"Generated H5: {h5_dest_path}")
            
            # Generate a stable UUID based on the content of the PLY file
            # This forces RViz / mesh_map to reload if the geometry changes.
            with open(ply_dest_path, "rb") as f_ply:
                ply_content = f_ply.read()
                content_hash = hashlib.sha1(ply_content).hexdigest()
                mesh_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, content_hash)
            
            print(f"Generated Content-Based UUID: {mesh_uuid} (hash: {content_hash[:8]}...)")
            
            # Injecting back to PLY to match original 'data' expectations
            inject_h5_attributes_to_ply(ply_dest_path, h5_dest_path, mesh_uuid=mesh_uuid)
            # Also inject to the model copy
            inject_h5_attributes_to_ply(model_ply_path, h5_dest_path, mesh_uuid=mesh_uuid)
            
            # Fused attributes are now in the PLY.
            print(f"Match Original Format: Navigation attributes and UUID ({mesh_uuid}) mapped to PLY.")
            
            # Explicit Verification
            print("Verifying H5 Normals...")
            with h5py.File(h5_dest_path, 'r') as f:
                # Check for normals in common locations (LVR2 structure can vary)
                if 'mesh/vertex_normals' in f:
                    v_norm_path = 'mesh/vertex_normals'
                elif 'mesh/vertex_attributes/vertex_normals' in f:
                    v_norm_path = 'mesh/vertex_attributes/vertex_normals'
                else:
                    v_norm_path = None
                    
                if 'mesh/face_normals' in f:
                    f_norm_path = 'mesh/face_normals'
                elif 'mesh/face_attributes/face_normals' in f:
                    f_norm_path = 'mesh/face_attributes/face_normals'
                else:
                    f_norm_path = None

                v_norm_shape = f[v_norm_path].shape if v_norm_path else (0,)
                f_norm_shape = f[f_norm_path].shape if f_norm_path else (0,)
                
                print(f"  H5 Vertex Normals: {v_norm_shape} vs Mesh: {len(final_mesh.vertices)}")
                print(f"  H5 Face Normals:   {f_norm_shape} vs Mesh: {len(final_mesh.faces)}")
                
                if v_norm_shape[0] != len(final_mesh.vertices):
                     print(f"  [!] CRITICAL ERROR: Vertex count mismatch! H5: {v_norm_shape[0]}, Mesh: {len(final_mesh.vertices)}")
                     print("  [!] Aborting attribute injection to prevent corruption.")
                     # Delete the invalid H5 to prevent usage
                     f.close()
                     os.remove(h5_dest_path)
                     sys.exit(1)
                     
                if f_norm_shape[0] != len(final_mesh.faces):
                     print(f"  [!] CRITICAL WARNING: Face count mismatch! H5: {f_norm_shape[0]}, Mesh: {len(final_mesh.faces)}")
                     # Often acceptable if vertices match (some faces might be degenerate/removed by lvr2 internally)
                     # But we should be wary.

        except subprocess.CalledProcessError as e:
            print(f"H5 Generation failed: {e}")
            sys.exit(1)

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
                compare_files_hash(model_dae_path, args.ref_dae)
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
    
    # Prevent overwriting input SDF
    if os.path.abspath(world_dest_path) == os.path.abspath(input_sdf):
        print(f"[Warning] Output world path matches input path. Renaming output to avoid overwrite.")
        world_dest_path = os.path.join(worlds_dir, f"{world_name}_generated.sdf")
        
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
    
    # --- NEW: Sync to Install Directory ---
    # This ensures that 'ros2 launch' immediately sees the updated map files.
    if LAUNCH_SUPPORT and tutorials_pkg:
        install_maps_dir = os.path.join(get_package_share_directory("mesh_navigation_tutorials"), "maps")
        os.makedirs(install_maps_dir, exist_ok=True)
        
        # Copy PLY and H5
        try:
            shutil.copy2(ply_dest_path, os.path.join(install_maps_dir, os.path.basename(ply_dest_path)))
            shutil.copy2(h5_dest_path, os.path.join(install_maps_dir, os.path.basename(h5_dest_path)))
            print(f"\n[Sync] Successfully updated install/ directory maps: {install_maps_dir}")
        except Exception as e:
            print(f"\n[Sync] Warning: Could not copy files to install/ directory: {e}")
    
    '''
    # --- NEW: Launch File Generation ---
    if tutorials_pkg:
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
    else:
        print("\n=== Stage 4.5: Launch File Generation (SKIPPED) ===")
        print("      Mesh Navigation Tutorials package not found.")

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
    if tutorials_pkg:
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
    else:
        print("Skipping launch (Mesh Navigation Tutorials package not found).")
        print("Generated files are available in the specified output directories.")
'''

def launch_setup(context, *args, **kwargs):
    input_sdf = LaunchConfiguration('input_sdf').perform(context)
    world_name = LaunchConfiguration('world_name').perform(context)

    max_edge = LaunchConfiguration('max_edge').perform(context)
    target_density = LaunchConfiguration('target_density').perform(context)
    primitive_resolution = LaunchConfiguration('primitive_resolution').perform(context)
    align_ground = LaunchConfiguration('align_ground').perform(context).lower() == 'true'
    flatten_ground = LaunchConfiguration('flatten_ground').perform(context).lower() == 'true'
    flatten_threshold = LaunchConfiguration('flatten_threshold').perform(context)
    shrink_faces = LaunchConfiguration('shrink_faces').perform(context)
    single_layer = LaunchConfiguration('single_layer').perform(context).lower() == 'true'
    clean_iter = LaunchConfiguration('clean_iter').perform(context)
    # Handle exclude list (passed as string space-separated)
    exclude_str = LaunchConfiguration('exclude').perform(context)
    
    # Mock sys.argv for main()
    sys.argv = [sys.argv[0], input_sdf, world_name]


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
    if single_layer:
        sys.argv.append("--single-layer")
    if clean_iter:
        sys.argv.extend(["--clean-iter", clean_iter])
    if exclude_str:
        sys.argv.append("--exclude")
        sys.argv.extend(exclude_str.split())
        
    print(f"\n[Launch] Initializing generation for: {world_name} (Max Edge: {max_edge}m, Resolution: {primitive_resolution})")
    main()
    
    # After generation, include the newly created launch file
    # pkg_mesh_navigation_tutorials = get_package_share_directory("mesh_navigation_tutorials")
    # launch_file = os.path.join(pkg_mesh_navigation_tutorials, "launch", f"launch_{world_name}.py")
    
    # return [
    #     IncludeLaunchDescription(
    #         PythonLaunchDescriptionSource(launch_file),
    #         launch_arguments={
    #             "localization": LaunchConfiguration("localization"),
    #             "start_rviz": LaunchConfiguration("start_rviz")
    #         }.items()
    #     )
    # ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("input_sdf", description="Path to input SDF/World file"),
        DeclareLaunchArgument("world_name", description="Name of the new world/environment"),

        DeclareLaunchArgument("max_edge", default_value="0.36", description="Maximum edge length for subdivision"),
        DeclareLaunchArgument("target_density", default_value="", description="Target vertex density per square meter"),
        DeclareLaunchArgument("primitive_resolution", default_value="64", description="Resolution for primitives"),
        DeclareLaunchArgument("align_ground", default_value="false", description="Align ground normal to +Z"),
        DeclareLaunchArgument("flatten_ground", default_value="false", description="Snap ground vertices to Z=0"),
        DeclareLaunchArgument("flatten_threshold", default_value="0.1", description="Z-range for ground flattening"),
        DeclareLaunchArgument("shrink_faces", default_value="0.0", description="Face shrinking factor"),
        DeclareLaunchArgument("single_layer", default_value="false", description="Optimize for Single Layer MeshNav"),
        DeclareLaunchArgument("clean_iter", default_value="0", description="Number of iterative cleaning passes"),
        DeclareLaunchArgument("exclude", default_value="", description="Model names to exclude (space separated)"),
        DeclareLaunchArgument("localization", default_value="ground_truth"),

        DeclareLaunchArgument("start_rviz", default_value="True"),
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == "__main__":
    main()


