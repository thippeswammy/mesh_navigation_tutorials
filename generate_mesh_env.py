#!/usr/bin/env python3
import os
import trimesh
import trimesh.remesh
import numpy as np
import xml.etree.ElementTree as ET
import subprocess
import shutil
import sys
import argparse
import struct
import h5py

def inject_h5_attributes_to_ply(ply_path, h5_path):
    print(f"Injecting attributes from {h5_path} to {ply_path}...")
    if not os.path.exists(h5_path):
        print("Error: H5 file not found.")
        return

    try:
        mesh = trimesh.load(ply_path, force='mesh')
        
        with h5py.File(h5_path, 'r') as f:
            # Check for attributes
            attributes = ['roughness', 'height_diff', 'border']
            for attr in attributes:
                path = f"mesh/vertex_attributes/{attr}"
                if path in f:
                    data = f[path][:]
                    # Ensure flat array for scalar attributes
                    if len(data.shape) > 1 and data.shape[1] == 1:
                        data = data.flatten()
                    
                    print(f"  Injecting {attr}: {data.shape}")
                    mesh.vertex_attributes[attr] = data
                else:
                    print(f"  Warning: Attribute {attr} not found in H5.")

        # Save back to PLY
        mesh.export(ply_path)
        print("  Attributes injected and PLY saved.")
        
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

def create_high_res_primitive(geometry_node):
    # Returns trimesh object or None
    box = geometry_node.find("box")
    if box is not None:
        size = [float(x) for x in box.find("size").text.split()]
        return trimesh.creation.box(extents=size)

    cylinder = geometry_node.find("cylinder")
    if cylinder is not None:
        r = float(cylinder.find("radius").text)
        l = float(cylinder.find("length").text)
        # High resolution cylinder
        return trimesh.creation.cylinder(radius=r, height=l, sections=64)

    sphere = geometry_node.find("sphere")
    if sphere is not None:
        r = float(sphere.find("radius").text)
        # High resolution sphere for organic organic shapes
        return trimesh.creation.icosphere(radius=r, subdivisions=5)
        
    plane = geometry_node.find("plane")
    if plane is not None:
        # Plane is typically infinite in Gazebo but we need a mesh.
        size = [float(x) for x in plane.find("size").text.split()] # x y
        # Create a thin box to represent the plane
        m = trimesh.creation.box(extents=[size[0], size[1], 0.01])
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

def extract_meshes_from_sdf(sdf_path, base_dir):
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

                    if mesh_path and os.path.exists(mesh_path):
                        try:
                            m = trimesh.load(mesh_path, force='mesh')
                            # Handle scene
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
                    else:
                        print(f"    Warning: Mesh file not found: {mesh_path} (URI: {uri})")

                else:
                    # Primitives
                    m = create_high_res_primitive(geom)
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

        if os.path.isfile(model_path):
            # Recursive call
            sub_mesh_data = extract_meshes_from_sdf(model_path, os.path.dirname(model_path))
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

def main():
    parser = argparse.ArgumentParser(description="Automate Gazebo SDF to Mesh Navigation Pipeline")
    parser.add_argument("input_sdf", help="Path to input SDF/World file")
    parser.add_argument("world_name", help="Name of the new world/environment")
    
    args = parser.parse_args()
    
    input_sdf = os.path.abspath(args.input_sdf)
    world_name = args.world_name
    
    if not os.path.exists(input_sdf):
        print(f"Error: Input file does not exist: {input_sdf}")
        sys.exit(1)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tutorials_pkg = os.path.join(script_dir, "mesh_navigation_tutorials")
    sim_pkg = os.path.join(script_dir, "mesh_navigation_tutorials_sim")
    
    maps_dir = os.path.join(tutorials_pkg, "maps")
    models_dir = os.path.join(sim_pkg, "models", world_name)
    worlds_dir = os.path.join(sim_pkg, "worlds")
    
    mesh_output_name = f"{world_name}.ply"
    h5_output_name = f"{world_name}.h5"
    
    ply_dest_path = os.path.join(maps_dir, mesh_output_name)
    h5_dest_path = os.path.join(maps_dir, h5_output_name)
    
    if os.path.exists(ply_dest_path):
        os.remove(ply_dest_path)
    if os.path.exists(h5_dest_path):
        os.remove(h5_dest_path)

    print("\n=== Step 1: Mesh Extraction ===")
    mesh_data_list = extract_meshes_from_sdf(input_sdf, os.path.dirname(input_sdf))
    
    if not mesh_data_list:
        print("Error: No meshes extracted. Check SDF file content.")
        sys.exit(1)
        
    final_mesh = trimesh.util.concatenate([d['mesh'] for d in mesh_data_list])
    print(f"Extracted {len(mesh_data_list)} sub-meshes. Vertices: {len(final_mesh.vertices)}, Faces: {len(final_mesh.faces)}")
    
    # Subdivision Step
    max_edge = 0.2
    print(f"Subdividing mesh (max edge length: {max_edge}m)...")
    vertices, faces = trimesh.remesh.subdivide_to_size(final_mesh.vertices, final_mesh.faces, max_edge)
    final_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print(f"Subdivision complete. Vertices: {len(final_mesh.vertices)}, Faces: {len(final_mesh.faces)}")

    # Ensure valid mesh
    print("Cleaning up mesh (processing, fixing normals)...")
    final_mesh.process() # Removes NaN, duplicate vertices, etc.
    final_mesh.fix_normals() # Ensure face and vertex normals are computed
    
    # Verify normals
    if final_mesh.vertex_normals is None or len(final_mesh.vertex_normals) != len(final_mesh.vertices):
        print("Warning: Vertex normals are missing or inconsistent! Recomputing...")
        final_mesh.vertex_normals = trimesh.geometry.weighted_vertex_normals(
            len(final_mesh.vertices),
            final_mesh.faces,
            final_mesh.face_normals,
            final_mesh.face_angles
        )

    print(f"Mesh Bounds: {final_mesh.bounds}")
    print(f"Vertex Normals Check: {len(final_mesh.vertex_normals)} normals for {len(final_mesh.vertices)} vertices.")
    
    # Export PLY (for Navigation)
    final_mesh.export(ply_dest_path)
    print(f"Saved PLY to: {ply_dest_path}")
    
    # Export STL (for Collision)
    stl_output_name = f"{world_name}.stl"
    stl_dest_path = os.path.join(maps_dir, stl_output_name)
    final_mesh.export(stl_dest_path)
    
    # Export DAE (for Visualization)
    # Optimization: If we have exactly one mesh from a source file, and it hasn't been heavily transformed 
    # (or we can assume identity for this simple case), copy it to preserve textures.
    dae_output_name = f"{world_name}.dae"
    dae_dest_path = os.path.join(maps_dir, dae_output_name)

    source_copied = False
    if len(mesh_data_list) == 1:
        item = mesh_data_list[0]
        # Prefer visual path if available, else source path (if it is a mesh)
        copy_source = item.get('visual_path')
        if not copy_source and item['type'] == 'mesh':
            copy_source = item['source_path']
            
        if copy_source:
             print(f"optimization: Single source mesh detected ({copy_source}). Copying to preserve textures.")
             shutil.copy2(copy_source, dae_dest_path)
             source_copied = True
    
    if not source_copied:
        final_mesh.export(dae_dest_path)
    
    print(f"Saved DAE to: {dae_dest_path}")
    
    print("\n=== Step 2: H5 Generation ===")
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
    except subprocess.CalledProcessError as e:
        print(f"H5 Generation failed: {e}")
        sys.exit(1)

    # Inject attributes back into PLY
    inject_h5_attributes_to_ply(ply_dest_path, h5_dest_path)

    print("\n=== Step 3: Workspace Organization ===")
    
    # Create model directory and copy meshes
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "meshes"), exist_ok=True)
    
    model_ply_path = os.path.join(models_dir, "meshes", mesh_output_name)
    shutil.copy2(ply_dest_path, model_ply_path)
    
    model_stl_path = os.path.join(models_dir, "meshes", f"{world_name}.stl")
    shutil.copy2(stl_dest_path, model_stl_path)
    
    model_dae_path = os.path.join(models_dir, "meshes", f"{world_name}.dae")
    shutil.copy2(dae_dest_path, model_dae_path)
    
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

    # Write model.sdf
    with open(os.path.join(models_dir, "model.sdf"), "w") as f:
        f.write(f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{world_name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>meshes/{world_name}.stl</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>meshes/{world_name}.dae</uri>
            <scale>1 1 1</scale>
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

    print("\n=== Step 4: Building ===")
    workspace_root = os.path.abspath(os.path.join(script_dir, "../../"))
    build_cmd = ["colcon", "build", "--packages-select", "mesh_navigation_tutorials_sim", "mesh_navigation_tutorials"]
    print(f"Running: {' '.join(build_cmd)}")
    
    try:
        subprocess.check_call(build_cmd, cwd=workspace_root)
        print("Build successful.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)
    
    print("\n=== Step 5: Launching & Monitoring ===")
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
                    print("    -> Reason: Synchronization mismatch between Gazebo and ROS.")
                    print("    -> Suggested Fix: Ensure 'use_sim_time' is set to True in all nodes.")
                    print("    -> Suggested Fix: Verify Gazebo-ROS bridge clock synchronization.")
                    
                if "Start pose in collision" in line or "Costmap is not valid" in line:
                    print("\n\033[93m[!] WARNING: ROBOT START POSE IN COLLISION [!]\033[0m")
                    print("    -> Reason: Robot spawned inside the mesh.")
                    print("    -> Suggested Fix: Adjust the spawn pose in the launch file or move the mesh.")
                    
    except KeyboardInterrupt:
        print("\nStopping launch...")
        process.terminate()
        
    print("Launch finished.")


if __name__ == "__main__":
    main()
