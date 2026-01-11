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

### Example

To generate a mesh environment for `uneven_terrain.sdf`:

```bash
python3 src/mesh_navigation_tutorials/mesh_navigation_tutorials/launch/generate_mesh_env.py src/mesh_navigation_tutorials/mesh_navigation_tutorials_sim/worlds/uneven_terrain.sdf
```

### Visual Example

![Uneven Terrain Mesh Generation](https://github.com/nature-robots/mesh_navigation/raw/master/doc/images/uneven_terrain_mesh.png)

### Options

* `world_name`: (Optional) Name of the new world/environment. Defaults to the input SDF filename.
* `--max-edge`: Maximum edge length for subdivision (default: 0.36m). Lower values create denser meshes.
* `--no-h5`: Skips generation of the `.h5` map file (useful if `lvr2_hdf5_mesh_tool` is not installed).
* `--primitive-resolution`: Resolution for generating meshes from primitives (default: 64).
* `--weld-threshold`: Distance threshold for welding vertices (default: 0.01m).
* `--force-upward`: Force normals of near-horizontal faces to point upward (+Z).
* `--align-ground`: Automatically align primary ground normal to +Z.
* `--flatten-ground`: Snap traversable ground vertices to Z=0.
