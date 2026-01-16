# Mesh Generation Tool (`generate_mesh_env.py`)

This script automates the pipeline for converting a Gazebo SDF world file into a functional Mesh Navigation environment.

## Overview

The tool performs the following key steps:

1. **Extraction**: Parses the input SDF file to extract all geometric elements, including both explicit meshes (e.g., `.dae`, `.stl`) and Gazebo primitives (boxes, cylinders, spheres), retaining their poses and hierarchy.
2. **Processing**: Merges these elements into a single monolithic mesh. It applies geometry processing techniques such as vertex welding, topological repair (to ensure the mesh is manifold), and hole filling. It also supports adaptive subdivision (via `--max-edge`) to ensure the mesh has sufficient resolution for navigation layers.
3. **Generation**: Produces all necessary artifacts for the `mesh_navigation` stack:
    * **Map Files**: Generates a `.ply` file with injected navigation attributes (roughness, height difference) and a `.h5` attribute map (using `lvr2_hdf5_mesh_tool`).
    * **Simulation Files**: Creates a corresponding Gazebo model (`.dae` visual, `model.sdf`, `model.config`) and a new world file referencing this generated model.

## Usage

Run the script from the root of your workspace:

```bash
python3 src/mesh_navigation_tutorials/mesh_navigation_tutorials/launch/generate_mesh_env.py <path_to_sdf_file> [options]
```

### Example Workflow

#### Method 1: Using ROS 2 Launch (Recommended)

This is the standard way to run the tool in a ROS 2 environment.

```bash
ros2 launch mesh_navigation_tutorials generate_mesh_env.py input_sdf:=src/mesh_navigation_tutorials/mesh_navigation_tutorials_sim/worlds/uneven_terrain_big.sdf world_name:=uneven_terrain_big
```

#### Method 2: Using Python Direct Execution

Useful for debugging or if you want to bypass the launch system.

```bash
python3 src/mesh_navigation_tutorials/mesh_navigation_tutorials/launch/generate_mesh_env.py \
src/mesh_navigation_tutorials/mesh_navigation_tutorials_sim/worlds/uneven_terrain_big.sdf
```

#### 2. Build the Workspace

Rebuild the packages to ensure the new maps and models are installed correctly.

```bash
colcon build --packages-select mesh_navigation_tutorials_sim mesh_navigation_tutorials --allow-overriding mesh_navigation_tutorials mesh_navigation_tutorials_sim
```

#### 3. Launch the Simulation

Launch the tutorial to see the result.

```bash
ros2 launch mesh_navigation_tutorials mesh_navigation_tutorials_launch.py world_name:=uneven_terrain_big
```

> **Note:** Initializing the mesh environment in the simulation might take a few seconds. If the robot spawns in the air or the map isn't visible immediately, **wait for ~10 seconds** for the systems to synchronize and the mesh to load.

### Options

* `world_name`: (Optional) Name of the new world/environment. Defaults to the input SDF filename.
* `--max-edge`: Maximum edge length for subdivision (default: 0.36m). Lower values create denser meshes.

* `--primitive-resolution`: Resolution for generating meshes from primitives (default: 64).
* `--weld-threshold`: Distance threshold for welding vertices (default: 0.01m).
* `--force-upward`: Force normals of near-horizontal faces to point upward (+Z).
* `--align-ground`: Automatically align primary ground normal to +Z.
* `--flatten-ground`: Snap traversable ground vertices to Z=0.
* `--single-layer`: Optimize for Single Layer MeshNav (High density, clean topology, flattened ground, wall preservation).
* `--filter-steep`: Filter out faces with normal.z < threshold (default 0.0). set -0.5 to preserve walls.
* `--clean-iter`: Number of iterative cleaning passes (default 0, auto-enabled by --single-layer).
* `--stitch-threshold`: Aggressively stitch border edges within this distance (default 0.0).
* `--no-build`: Skip colcon build after generation.
* `--no-dae`: Skip DAE export (speeds up generation).
* `--exclude`: List of model names/substrings to exclude (e.g. 'wall obstacle').
