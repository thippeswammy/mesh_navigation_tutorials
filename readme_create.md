# Mesh Environment Generation Guide

This guide explains how to use `generate_mesh_env.py` to automate the conversion of Gazebo/Ignition SDF files into functional Mesh Navigation environments.

## Features
- **Mesh Extraction**: Automatically aggregates primitive shapes (Box, Sphere, Cylinder) and model meshes (`.stl`, `.dae`).
- **Adaptive Scaling**: Configurable subdivision to maintain consistent quad/triangle density across maps.
- **Automated Validation**: Built-in comparison stage to verify generated geometry against reference files.
- **Pipeline Automation**: Handles H5 generation, workspace organization, `colcon build`, and simulation launch.

## Usage

### Quick Start
You can run the tool as a standalone script or as a **ROS 2 launch file**:

**ROS 2 Launch (Recommended):**
```bash
ros2 launch mesh_navigation_tutorials generate_mesh_env.py \
  input_sdf:=floor_is_lava.sdf \
  world_name:=my_new_world
```

**CLI Script:**
```bash
python3 src/mesh_navigation_tutorials/mesh_navigation_tutorials/launch/generate_mesh_env.py \
  src/mesh_navigation_tutorials/mesh_navigation_tutorials_sim/worlds/floor_is_lava.sdf \
  my_new_world
```

### Core Arguments
- `input_sdf`: Path to the source SDF world or model file.
- `world_name`: Unique name for the generated environment.

### Optimization & Scaling Options
- `--gen-h5`: Generate the `.h5` navigation layers using `lvr2_hdf5_mesh_tool`. (Disabled by default).
- `--max-edge <float>`: Set the maximum edge length for subdivision. 

> [!NOTE]
> When using `ros2 launch`, parameters are passed as `key:=value` (e.g., `gen_h5:=true`).
  - **Default**: `0.36m` (Optimal for balanced collision and navigation).
- `--primitive-resolution <int>`: Set the mesh segments for spheres/cylinders (default: `64`).

### Validation & Comparison
- `--validate-only`: Runs the extraction and URI resolution stage, then exits. Useful for verifying that all assets can be found without full processing.
- `--ref-ply <path>`: Compare the generated PLY against a reference navigation mesh (checks counts, bounds, area).
- `--ref-dae <path>`: Compare visual file against a reference using MD5 hashing.

---

## Examples

### 1. High-Resolution Navigation Map (Default)
Useful for complex environments like `uneven_terrain`.
```bash
python3 generate_mesh_env.py src/.../uneven_terrain.sdf uneven_terrain_gen
```

### 2. Identity Conversion (Matching Reference)
Matches vertex, face, and edge counts exactly by skipping subdivision.
```bash
python3 generate_mesh_env.py src/.../floor_is_lava/model.sdf floor_is_lava_1 --no-subdivide --ref-ply src/.../maps/floor_is_lava.ply
```

### 3. Quick Resource Validation
Check if all model URIs resolve correctly without building.
```bash
python3 generate_mesh_env.py src/.../complex_world.sdf test_env --validate-only
```

### 4. Complete World Conversion
Full pipeline for a specific world file including build and launch. Example using the replicated `floor_is_lava_1` world:
```bash
python3 src/mesh_navigation_tutorials/generate_mesh_env.py \
  src/mesh_navigation_tutorials/mesh_navigation_tutorials_sim/worlds/floor_is_lava_1.sdf \
  floor_is_lava_1
```

> [!TIP]
> Run the above command from the **workspace root** (the directory containing the `src` folder).

---

## Generated File Structure

The script organizes output files into two main ROS 2 packages within the `src` directory.

### 1. Navigation Maps
Located in: `mesh_navigation_tutorials/maps/`
These files are the core output for the **Mesh Navigation** stack.
- `<world_name>.ply`: High-resolution navigation mesh.
- `<world_name>.h5`: HDF5 file containing navigation layers (only if `--gen-h5` is used).

### 2. Simulation & Visual Assets
Located in: `mesh_navigation_tutorials_sim/models/<world_name>/meshes/`
- `<world_name>.ply`: Local copy of the navigation mesh.
- `<world_name>.stl`: Clean collision geometry for Gazebo.
- `<world_name>.dae`: Visual representation for Gazebo and RViz.

### 3. Dedicated Launch Files
Located in: `mesh_navigation_tutorials/launch/`
- `launch_<world_name>.py`: Pre-configured launch file that starts the specific world and its corresponding map.

---

## Deployment Stages

1.  **Extraction & Validation**: Resolves URIs and merges all geometry into a single global mesh.
2.  **Mesh Processing**: Performs cleaning, normal fixing, and subdivision.
3.  **H5 Generation**: Triggers `lvr2_hdf5_mesh_tool` to create the navigation-ready H5 map.
4.  **Comparison (Optional)**: Validates bounds and topology against references.
5.  **Workspace Organization**: Creates the standard ROS 2 package structure for models and worlds.
6.  **Building & Launching**: Runs `colcon build` and launches the environment with robot monitoring.

---

## Requirements
- `trimesh`, `numpy`, `h5py`
- `lvr2` (Specifically `lvr2_hdf5_mesh_tool`)
- ROS 2 environment (Humble/Galactic) with `mesh_navigation` stack.
