"""
PyVista rendering of tagged Gmsh meshes.

Functions:
  - read_tagged_msh(msh_path) -> dict
  - build_polydata_with_labels(groups, group_name_to_id) -> pv.PolyData
  - plot_tagged_mesh(mesh, id_to_name, output_png) -> None
  - visualize_tagged_mesh(msh_path, output_png) -> None
"""

import json
import gmsh
import numpy as np
import pyvista as pv
import matplotlib
import matplotlib.colors


def read_tagged_msh(msh_path: str) -> dict:
    """
    Read a tagged .msh file and return per-group surface triangle data.

    Returns
    -------
    dict: {group_name: {'verts': np.ndarray(N,3), 'tris': np.ndarray(M,3)}}
        tris contains LOCAL indices into verts (0-based, not global node tags).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        gmsh.merge(msh_path)
        groups = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            if dim != 2:
                continue
            name = gmsh.model.getPhysicalName(dim, tag)
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)

            all_verts = []
            all_tris = []
            vert_offset = 0
            for ent in entities:
                node_tags, coords, _ = gmsh.model.mesh.getNodes(
                    2, int(ent), includeBoundary=True
                )
                verts = np.array(coords, dtype=np.float64).reshape(-1, 3)
                node_to_local = {
                    int(nt): i + vert_offset for i, nt in enumerate(node_tags)
                }
                _, _, elem_nodes = gmsh.model.mesh.getElements(2, int(ent))
                if elem_nodes:
                    raw_tris = elem_nodes[0].reshape(-1, 3).astype(np.int64)
                    local_tris = np.array(
                        [[node_to_local[int(n)] for n in tri] for tri in raw_tris],
                        dtype=np.int32,
                    )
                    all_tris.append(local_tris)
                all_verts.append(verts)
                vert_offset += len(verts)

            if all_verts:
                groups[name] = {
                    "verts": np.vstack(all_verts),
                    "tris": (
                        np.vstack(all_tris)
                        if all_tris
                        else np.empty((0, 3), dtype=np.int32)
                    ),
                }
        return groups
    finally:
        gmsh.finalize()


def build_polydata_with_labels(groups: dict, group_name_to_id: dict) -> "pv.PolyData":
    """
    Assemble all group triangle data into a single PolyData with integer cell labels.

    Parameters
    ----------
    groups : dict
        {group_name: {'verts': np.ndarray(N,3), 'tris': np.ndarray(M,3)}}
    group_name_to_id : dict
        {group_name: int}  -- integer label per group for colormap mapping

    Returns
    -------
    pv.PolyData with cell_data['group_id'] array (int32)
    """
    all_verts = []
    all_faces = []  # PyVista padded format: [3, i0, i1, i2, ...]
    all_labels = []
    offset = 0

    for name, data in groups.items():
        verts = data["verts"]
        tris = data["tris"].astype(np.int32) + offset
        # PyVista VTK face format: prepend count (3 for triangles) per face
        padded = np.hstack([np.full((len(tris), 1), 3, dtype=np.int32), tris])
        all_verts.append(verts)
        all_faces.append(padded.ravel())
        all_labels.extend([group_name_to_id[name]] * len(tris))
        offset += len(verts)

    mesh = pv.PolyData(np.vstack(all_verts), np.concatenate(all_faces))
    mesh.cell_data["group_id"] = np.array(all_labels, dtype=np.int32)
    return mesh


def plot_tagged_mesh(mesh, id_to_name: dict, output_png: str = None) -> None:
    """
    Render the tagged mesh with categorical coloring.

    _untagged gets warning orange [1.0, 0.5, 0.0, 1.0] per VIZ-02.
    All other groups receive tab10 colors.

    Parameters
    ----------
    mesh : pv.PolyData
        Tagged mesh with cell_data['group_id'] integer labels.
    id_to_name : dict
        {int: str}  -- maps group ID to group name.
    output_png : str, optional
        If provided, render off-screen and save to this path.
    """
    n_groups = len(id_to_name)
    tab10 = matplotlib.colormaps["tab10"].colors
    color_list = []
    tab_idx = 0
    for gid in sorted(id_to_name.keys()):
        name = id_to_name[gid]
        if name == "_untagged":
            color_list.append([1.0, 0.5, 0.0, 1.0])  # warning orange per VIZ-02
        else:
            color_list.append(list(tab10[tab_idx % 10]) + [1.0])
            tab_idx += 1

    annotations = {float(gid): name for gid, name in id_to_name.items()}

    pl = pv.Plotter(off_screen=(output_png is not None))
    pl.add_mesh(
        mesh,
        scalars="group_id",
        cmap=matplotlib.colors.ListedColormap(color_list),
        n_colors=n_groups,
        categories=True,
        annotations=annotations,
    )
    if output_png:
        pl.show(screenshot=output_png)
        print(f"Saved tagged mesh visualization: {output_png}")
    else:
        pl.show()


def visualize_tagged_mesh(msh_path: str, output_png: str = None) -> None:
    """
    High-level: read .msh, build PolyData, render with group colors.

    Convenience wrapper: read_tagged_msh -> build_polydata_with_labels -> plot_tagged_mesh.

    Parameters
    ----------
    msh_path : str
        Path to the tagged .msh file produced by run_full_pipeline.
    output_png : str, optional
        If provided, render off-screen and save PNG to this path.
    """
    groups = read_tagged_msh(msh_path)
    if not groups:
        raise ValueError(
            f"No surface physical groups found in {msh_path}. "
            "Ensure run_full_pipeline completed successfully."
        )
    group_names = sorted(groups.keys())
    name_to_id = {n: i for i, n in enumerate(group_names)}
    id_to_name = {v: k for k, v in name_to_id.items()}
    mesh = build_polydata_with_labels(groups, name_to_id)
    plot_tagged_mesh(mesh, id_to_name, output_png=output_png)


def load_group_point_cloud(bc_json_path: str, group_name: str) -> np.ndarray:
    """Extract vertex coordinates for a named group from bc_groups JSON as (N,3) array.

    Parameters
    ----------
    bc_json_path : str
        Path to the bc_groups JSON file.
    group_name : str
        Name of the group to extract.

    Returns
    -------
    np.ndarray of shape (N, 3) -- world-space vertex coordinates.

    Raises
    ------
    KeyError
        If group_name is not found in the JSON groups.
    """
    with open(bc_json_path) as f:
        data = json.load(f)
    if group_name not in data["groups"]:
        raise KeyError(
            f"Group '{group_name}' not found in {bc_json_path}. "
            f"Available: {list(data['groups'].keys())}"
        )
    return np.array(data["groups"][group_name]["vertices"], dtype=np.float64)


def plot_comparison(
    blender_points: np.ndarray,
    matched_mesh: "pv.PolyData",
    group_name: str,
    output_png: str = None,
) -> None:
    """Render side-by-side: Blender selection point cloud (left) vs matched mesh facets (right).

    Parameters
    ----------
    blender_points : np.ndarray
        (N, 3) world-space vertex coordinates from bc_groups JSON.
    matched_mesh : pv.PolyData
        PolyData of the matched surface triangles for this group.
    group_name : str
        Name of the group (used in subplot titles).
    output_png : str, optional
        If provided, render off-screen and save screenshot to this path.
    """
    pl = pv.Plotter(shape=(1, 2), off_screen=(output_png is not None))

    # Left: Blender selection point cloud
    pl.subplot(0, 0)
    pl.add_title(f"Blender selection: {group_name}")
    pl.add_points(
        blender_points,
        color="blue",
        point_size=5,
        render_points_as_spheres=True,
    )

    # Right: Matched mesh facets
    pl.subplot(0, 1)
    pl.add_title(f"Matched mesh: {group_name}")
    pl.add_mesh(matched_mesh, color="green", show_edges=True)

    pl.link_views()

    if output_png:
        pl.show(screenshot=output_png)
        print(f"Saved comparison plot for group '{group_name}': {output_png}")
    else:
        pl.show()
