"""
Integration tests for STEP-to-tet mesh surface extraction via Gmsh.

These tests require Gmsh and use the tmp_step_box fixture from conftest.py.
"""

import numpy as np
import pytest

from matching_library.mesher import step_to_surface_data, load_existing_mesh


def test_step_to_surface_data_box(tmp_step_box):
    """
    step_to_surface_data on a unit box STEP returns 6 surfaces, each with
    non-empty verts and tris arrays, and exactly 1 volume tag.
    """
    result = step_to_surface_data(tmp_step_box)

    assert "surface_tags" in result
    assert "surface_data" in result
    assert "volume_tags" in result
    assert "tolerance" in result

    # Unit box has 6 BREP surfaces
    assert len(result["surface_tags"]) == 6, (
        f"Expected 6 surfaces, got {len(result['surface_tags'])}"
    )

    # Each surface must have vertices and triangles
    for tag in result["surface_tags"]:
        data = result["surface_data"][tag]
        assert "verts" in data
        assert "tris" in data
        assert "node_tags" in data
        assert data["verts"].shape[0] > 0, f"Surface {tag} has no vertices"
        assert data["tris"].shape[0] > 0, f"Surface {tag} has no triangles"
        assert data["verts"].shape[1] == 3, f"Surface {tag} verts not (N,3)"
        assert data["tris"].shape[1] == 3, f"Surface {tag} tris not (M,3)"

    # Single solid -> 1 volume
    assert len(result["volume_tags"]) == 1, (
        f"Expected 1 volume, got {len(result['volume_tags'])}"
    )

    # Tolerance should be positive
    assert result["tolerance"] > 0.0


def test_surface_data_vertex_coordinates(tmp_step_box):
    """
    All vertex coordinates from a unit box STEP lie within [0, 1]^3.

    The box.step is a unit box; all boundary node coordinates must fall
    within bounds [0, 1] on all three axes.
    """
    result = step_to_surface_data(tmp_step_box)

    for tag in result["surface_tags"]:
        verts = result["surface_data"][tag]["verts"]
        assert verts.shape[0] > 0

        x_min, y_min, z_min = verts.min(axis=0)
        x_max, y_max, z_max = verts.max(axis=0)

        margin = 0.01  # small tolerance for floating point
        assert x_min >= -margin, f"Surface {tag}: x_min={x_min:.6f} < 0"
        assert y_min >= -margin, f"Surface {tag}: y_min={y_min:.6f} < 0"
        assert z_min >= -margin, f"Surface {tag}: z_min={z_min:.6f} < 0"
        assert x_max <= 1.0 + margin, f"Surface {tag}: x_max={x_max:.6f} > 1"
        assert y_max <= 1.0 + margin, f"Surface {tag}: y_max={y_max:.6f} > 1"
        assert z_max <= 1.0 + margin, f"Surface {tag}: z_max={z_max:.6f} > 1"


def test_load_existing_mesh(tmp_step_box, tmp_path):
    """
    load_existing_mesh on a .msh produced by step_to_surface_data returns
    the same number of surfaces as the original call.
    """
    import gmsh

    # Step 1: generate mesh from STEP and write a .msh file
    result_orig = step_to_surface_data(tmp_step_box)
    n_surfaces_orig = len(result_orig["surface_tags"])

    # Step 2: write the mesh to a .msh file using Gmsh
    msh_path = str(tmp_path / "box.msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.importShapes(tmp_step_box)
    gmsh.model.occ.synchronize()
    import math
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)
    mesh_size = max(min(diag / 50.0, 10.0), 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(msh_path)
    gmsh.finalize()

    # Step 3: load that .msh and verify surface count matches
    result_loaded = load_existing_mesh(msh_path)

    assert len(result_loaded["surface_tags"]) == n_surfaces_orig, (
        f"Expected {n_surfaces_orig} surfaces from loaded mesh, "
        f"got {len(result_loaded['surface_tags'])}"
    )

    # Each surface from loaded mesh should have verts and tris
    for tag in result_loaded["surface_tags"]:
        data = result_loaded["surface_data"][tag]
        assert data["verts"].shape[0] > 0
        assert data["tris"].shape[0] > 0
