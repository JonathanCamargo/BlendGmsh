"""Visualization module: PyVista rendering of tagged Gmsh meshes.

Public API:
  - read_tagged_msh(msh_path) -> dict
  - build_polydata_with_labels(groups, group_name_to_id) -> pv.PolyData
  - plot_tagged_mesh(mesh, id_to_name, output_png) -> None
  - visualize_tagged_mesh(msh_path, output_png) -> None
  - load_group_point_cloud(bc_json_path, group_name) -> np.ndarray
  - plot_comparison(blender_points, matched_mesh, group_name, output_png) -> None
"""

from visualization.visualizer import (
    read_tagged_msh,
    build_polydata_with_labels,
    plot_tagged_mesh,
    visualize_tagged_mesh,
    load_group_point_cloud,
    plot_comparison,
)

__all__ = [
    "read_tagged_msh",
    "build_polydata_with_labels",
    "plot_tagged_mesh",
    "visualize_tagged_mesh",
    "load_group_point_cloud",
    "plot_comparison",
]
