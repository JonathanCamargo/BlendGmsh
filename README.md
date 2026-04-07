# BlendGmsh

A Blender-to-Gmsh boundary condition workflow. Visually select faces on a CAD model in Blender, then auto-assign those selections as Gmsh physical groups -- producing a tagged `.msh` file ready for any FEM solver.

## How It Works

Two modes, auto-detected from the input:

- **BREP mode** (STEP import): physical groups are assigned by BREP surface tag *before* meshing. Lossless -- no coordinate matching needed.
- **Mesh mode** (STL/OBJ import): KDTree coordinate matching of Blender vertex selections *post-meshing*.

```
                          BREP mode (STEP)
STEP file ──> Blender addon ──> bc_groups.json ──> matching_library ──> tagged.msh
                (visual face       (surface tags)    (tag before mesh)   (physical
                 selection)                                               groups)

                          Mesh mode (STL/OBJ)
STL/OBJ ───> Blender addon ──> bc_groups.json ──> matching_library ──> tagged.msh
                (visual face       (vertex coords)   (KDTree match)     (physical
                 selection)                                              groups)
```

## Requirements

- Python 3.10+
- [Gmsh](https://gmsh.info/) Python API (`pip install gmsh`)
- [SciPy](https://scipy.org/) (`pip install scipy`)
- [jsonschema](https://python-jsonschema.readthedocs.io/) (`pip install jsonschema`)
- [PyVista](https://pyvista.org/) (`pip install pyvista`) -- for visualization
- [NumPy](https://numpy.org/) (`pip install numpy`)
- [Blender 4.2+](https://www.blender.org/) -- for the addon (not needed for CLI/library usage)

```bash
pip install gmsh scipy jsonschema pyvista numpy
```

## Quick Start

### CLI Usage

Full pipeline -- geometry file + BC groups to tagged mesh:

```bash
python -m matching_library run bc_groups.json model.step output.msh
```

Tag an existing mesh:

```bash
python -m matching_library tag bc_groups.json existing.msh output.msh
```

Inspect a mesh file:

```bash
python -m matching_library inspect output.msh
```

Cross-reference BC groups against STEP geometry:

```bash
python -m matching_library inspect-bc bc_groups.json model.step
```

Visualize BC groups on the mesh:

```bash
python -m matching_library visualize bc_groups.json output.msh --output preview.png
```

### Python API

```python
from matching_library import run_full_pipeline, tag_existing_mesh

# Full pipeline: geometry + JSON -> tagged .msh
report = run_full_pipeline("bc_groups.json", "model.step", "output.msh")

# Tag existing mesh: .msh + JSON -> re-tagged .msh
report = tag_existing_mesh("bc_groups.json", "input.msh", "output.msh")
```

Debug tools:

```python
from matching_library import inspect_msh, inspect_bc_groups, visualize_bc_groups

info = inspect_msh("output.msh")
info = inspect_bc_groups("bc_groups.json", "model.step")
visualize_bc_groups("bc_groups.json", "output.msh", output_png="preview.png")
```

### Visualization

```python
from visualization import visualize_tagged_mesh, load_group_point_cloud, plot_comparison
from visualization import read_tagged_msh, build_polydata_with_labels

# Render tagged mesh with per-group colors
visualize_tagged_mesh("output.msh", output_png="tagged.png")

# Side-by-side comparison: Blender selection vs matched facets
blender_pts = load_group_point_cloud("bc_groups.json", "inlet")
groups = read_tagged_msh("output.msh")
mesh = build_polydata_with_labels(groups, {"inlet": 1})
plot_comparison(blender_pts, mesh, "inlet", output_png="comparison.png")
```

## Blender Addon Installation

1. Open Blender 4.2+
2. Go to **Edit > Preferences > Add-ons**
3. Click **Install from Disk** and select the `blender_addon/` folder (or zip it first)
4. Enable **BlendGmsh** in the addon list

To package the addon with all dependencies:

```bash
# Linux / macOS
./export.sh

# Windows (PowerShell)
.\export.ps1
```

### Addon Panel

The addon adds a **FEM BC** panel in the 3D viewport sidebar (press `N`):

- **Import STEP** -- imports a STEP file via Gmsh, creates per-BREP-surface vertex groups
- **Import STL/OBJ** -- imports mesh files for mesh-mode workflows
- **Status** -- shows import state, face count, and active mode (BREP or Mesh)
- **Face Groups** -- create, name, and color-code boundary condition groups
- **Assign Faces** -- assign selected faces to the active group. In BREP mode, selecting any triangle on a surface assigns the entire BREP face.
- **Export JSON** -- write selections to a `bc_groups_v1` JSON file
- **Generate Mesh** -- run the full pipeline from within Blender (subprocess, non-blocking)

## JSON Schema

The `bc_groups_v1` schema (`schema/bc_groups_v1.json`) defines the exchange format between addon and library. The `mode` field (or auto-detection from group contents) selects the pipeline:

BREP mode (STEP import):

```json
{
  "schema_version": 1,
  "source": "blender_addon",
  "mode": "brep",
  "step_file": "model.step",
  "units": "meters",
  "groups": {
    "inlet": {
      "surface_tags": [3, 5]
    },
    "outlet": {
      "surface_tags": [7]
    }
  },
  "mesh_stats": {
    "total_vertices": 500,
    "total_faces": 250,
    "bounding_box": {"min": [0,0,0], "max": [1,1,1]}
  }
}
```

Mesh mode (STL/OBJ import):

```json
{
  "schema_version": 1,
  "source": "blender_addon",
  "mode": "mesh",
  "step_file": "model.stl",
  "units": "meters",
  "groups": {
    "inlet": {
      "vertices": [[x, y, z], ...],
      "face_vertex_indices": [[i, j, k], ...],
      "vertex_count": 100,
      "face_count": 50
    }
  },
  "mesh_stats": {
    "total_vertices": 500,
    "total_faces": 250,
    "bounding_box": {"min": [0,0,0], "max": [1,1,1]}
  }
}
```

## Project Structure

```
blendgmsh/
  matching_library/      # Core matching engine
    __init__.py           #   run_full_pipeline(), tag_existing_mesh(), debug re-exports
    __main__.py           #   CLI entry point (python -m matching_library)
    matcher.py            #   KDTree matching with majority voting
    tolerance.py          #   Auto-tolerance from mesh geometry
    mesher.py             #   Gmsh tet mesh generation + surface data extraction
    tagger.py             #   Physical group assignment + .msh writing
    coverage.py           #   CoverageReport with per-group statistics
    debug.py              #   inspect_msh(), inspect_bc_groups(), visualize_bc_groups()
  step_converter/         # STEP tessellation
    tessellate.py         #   tessellate_step() -- STEP to per-surface mesh data
    validate.py           #   check_fused_faces() -- island count validation
  blender_addon/          # Blender 4.2+ extension
    __init__.py           #   Addon registration + dependency bootstrap
    operators.py          #   ImportSTEP, ImportMesh, AddGroup, RemoveGroup,
                          #   AssignFaces, ExportJSON, GenerateMesh
    panels.py             #   FEMBC_PT_MainPanel (N-panel sidebar)
    properties.py         #   FEMBCGroupItem, FEMBCMainProps
    export.py             #   Pure-Python JSON export (no bpy dependency)
    vertex_groups.py      #   Island detection (BFS)
    mesh_import.py        #   STL/OBJ import with face-island vertex groups
    blender_manifest.toml #   Extension manifest
  visualization/          # PyVista validation
    visualizer.py         #   Tagged mesh rendering + side-by-side comparison
  schema/
    bc_groups_v1.json     # JSON Schema (Draft 2020-12) for group exchange
  examples/
    box.step              # Unit box test geometry (6 faces)
    cylinder.step         # Cylinder test geometry (3 faces)
    demo_visualization.py # End-to-end demo producing PNG visualizations
  tests/                  # 100 pytest tests
  export.sh               # Linux/macOS addon packaging
  export.ps1              # Windows addon packaging
```

## How Matching Works

### BREP mode (STEP files)

1. **Import**: Gmsh imports the STEP file and exposes BREP surface tags.
2. **Physical groups**: Each BC group's surface tags are assigned as Gmsh physical groups *before* meshing. No coordinate matching needed.
3. **Mesh**: Gmsh generates the tet mesh. Physical groups survive meshing because they're on geometry entities.
4. **`_untagged` group**: Surfaces not in any BC group get an `_untagged` physical group (required by `.msh` v4 to prevent silent element drops).

### Mesh mode (STL/OBJ files)

1. **Tessellation**: Gmsh imports the geometry and creates a tetrahedral mesh. Boundary surface facets are extracted per surface entity.
2. **KDTree lookup**: For each Blender vertex group, a KDTree is built from the group's world-space coordinates. Each Gmsh boundary facet's vertices are queried against all group KDTrees.
3. **Majority voting**: Each facet votes for the group with the most vertex matches within tolerance. This prevents edge contamination at shared face boundaries.
4. **Auto-tolerance**: Tolerance is computed as `avg_boundary_edge_length / 5.0` from the actual tet mesh -- adapts to part scale automatically with no manual tuning.
5. **Physical groups**: Matched facets are assigned to Gmsh physical groups. An `_untagged` group captures all unmatched boundary facets.

## Running Tests

```bash
python -m pytest tests/ -q
```

## Examples

Generate test STEP geometry and run the visualization demo:

```bash
python examples/generate_test_geometry.py
python examples/demo_visualization.py
```

This produces `output_tagged.png` (tagged mesh visualization) and `comparison_top.png` (side-by-side comparison).

## License

See LICENSE file for details.
