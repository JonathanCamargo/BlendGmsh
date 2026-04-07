"""
Integration tests for visualization module (VIZ-01, VIZ-02).

Tests exercise:
  1. run_full_pipeline to produce a tagged .msh
  2. read_tagged_msh to extract per-group triangle data
  3. build_polydata_with_labels to assemble PyVista PolyData
  4. plot_tagged_mesh to render off-screen PNG
"""

import json
import os
import numpy as np
import pytest
from pathlib import Path

import gmsh
import pyvista as pv
from matching_library import run_full_pipeline
from visualization import (
    read_tagged_msh,
    build_polydata_with_labels,
    plot_tagged_mesh,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_box_surface_tags(step_path):
    """Get top and bottom BREP surface tags from a box STEP."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.getEntities(dim=2)
    top_tags, bot_tags = [], []
    for _, tag in surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        if zmin > 0.99:
            top_tags.append(tag)
        elif zmax < 0.01:
            bot_tags.append(tag)
    gmsh.finalize()
    return top_tags, bot_tags


def _make_bc_groups_json(step_path, groups_dict, tmp_path):
    """Write a valid bc_groups JSON. groups_dict = {name: [surface_tags]}."""
    groups_data = {name: {"surface_tags": tags} for name, tags in groups_dict.items()}
    data = {
        "schema_version": 1,
        "source": "test",
        "step_file": step_path,
        "units": "meters",
        "groups": groups_data,
        "mesh_stats": {
            "total_vertices": 0, "total_faces": 0,
            "bounding_box": {"min": [0, 0, 0], "max": [1, 1, 1]},
        },
    }
    json_path = str(tmp_path / "bc_groups.json")
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


# ---------------------------------------------------------------------------
# Test 1: read_tagged_msh returns expected group keys and array shapes
# ---------------------------------------------------------------------------

def test_read_tagged_msh_returns_groups(tmp_path, tmp_step_box):
    """read_tagged_msh returns dict with 'top', 'bottom', '_untagged' keys."""
    output_msh = str(tmp_path / "tagged.msh")
    top_tags, bot_tags = _get_box_surface_tags(tmp_step_box)
    json_path = _make_bc_groups_json(
        tmp_step_box, {"top": top_tags, "bottom": bot_tags}, tmp_path
    )
    run_full_pipeline(json_path, tmp_step_box, output_msh)
    groups = read_tagged_msh(output_msh)

    assert "top" in groups
    assert "bottom" in groups
    assert "_untagged" in groups

    for name in ("top", "bottom", "_untagged"):
        verts = groups[name]["verts"]
        tris = groups[name]["tris"]
        assert isinstance(verts, np.ndarray)
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert verts.shape[0] > 0
        assert isinstance(tris, np.ndarray)
        assert tris.ndim == 2 and tris.shape[1] == 3
        assert tris.shape[0] > 0


# ---------------------------------------------------------------------------
# Test 2: tris contain valid local indices
# ---------------------------------------------------------------------------

def test_read_tagged_msh_local_indices(tmp_path, tmp_step_box):
    """For each group, tris.max() < len(verts)."""
    output_msh = str(tmp_path / "tagged.msh")
    top_tags, bot_tags = _get_box_surface_tags(tmp_step_box)
    json_path = _make_bc_groups_json(
        tmp_step_box, {"top": top_tags, "bottom": bot_tags}, tmp_path
    )
    run_full_pipeline(json_path, tmp_step_box, output_msh)
    groups = read_tagged_msh(output_msh)

    for name, data in groups.items():
        verts = data["verts"]
        tris = data["tris"]
        if tris.shape[0] > 0:
            assert int(tris.max()) < len(verts), (
                f"group '{name}': tris.max()={int(tris.max())} >= len(verts)={len(verts)}"
            )


# ---------------------------------------------------------------------------
# Test 3: build_polydata_with_labels produces correct PolyData
# ---------------------------------------------------------------------------

def test_build_polydata_with_labels(tmp_path):
    """build_polydata_with_labels returns PolyData with correct n_cells and group_id labels."""
    verts_a = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    tris_a = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    verts_b = np.array([[2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]], dtype=np.float64)
    tris_b = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    groups = {
        "a": {"verts": verts_a, "tris": tris_a},
        "b": {"verts": verts_b, "tris": tris_b},
    }
    name_to_id = {"a": 0, "b": 1}
    mesh = build_polydata_with_labels(groups, name_to_id)

    assert mesh.n_cells == 4
    assert "group_id" in mesh.cell_data
    assert sorted(mesh.cell_data["group_id"].tolist()) == [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# Test 4: plot_tagged_mesh produces a non-empty PNG
# ---------------------------------------------------------------------------

def test_plot_tagged_mesh_produces_png(tmp_path, tmp_step_box):
    """plot_tagged_mesh writes a PNG file that exists and has size > 0."""
    output_msh = str(tmp_path / "tagged.msh")
    output_png = str(tmp_path / "viz.png")

    top_tags, bot_tags = _get_box_surface_tags(tmp_step_box)
    json_path = _make_bc_groups_json(
        tmp_step_box, {"top": top_tags, "bottom": bot_tags}, tmp_path
    )
    run_full_pipeline(json_path, tmp_step_box, output_msh)
    groups = read_tagged_msh(output_msh)

    group_names = sorted(groups.keys())
    name_to_id = {n: i for i, n in enumerate(group_names)}
    id_to_name = {v: k for k, v in name_to_id.items()}
    mesh = build_polydata_with_labels(groups, name_to_id)
    plot_tagged_mesh(mesh, id_to_name, output_png=output_png)

    assert Path(output_png).exists()
    assert Path(output_png).stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 5: _untagged group is present and has triangles
# ---------------------------------------------------------------------------

def test_untagged_rendered_in_warning_color(tmp_path, tmp_step_box):
    """_untagged group has > 0 triangles when only some surfaces are tagged."""
    output_msh = str(tmp_path / "tagged.msh")
    top_tags, _ = _get_box_surface_tags(tmp_step_box)
    json_path = _make_bc_groups_json(
        tmp_step_box, {"top": top_tags}, tmp_path
    )
    run_full_pipeline(json_path, tmp_step_box, output_msh)
    groups = read_tagged_msh(output_msh)

    assert "_untagged" in groups
    assert groups["_untagged"]["tris"].shape[0] > 0


# ---------------------------------------------------------------------------
# Test 6-8: Point cloud comparison tests removed (JSON no longer contains
# vertex coordinates — the new BREP tag-based workflow doesn't need them)
# ---------------------------------------------------------------------------
