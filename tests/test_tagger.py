"""
Integration tests for tag_and_write: physical group creation and .msh v4 output.

Tests operate within a manually managed Gmsh session (initialize -> mesh -> tag -> finalize)
because tag_and_write must run within the same session as meshing.
"""

import math
import os
import pytest
import gmsh
import numpy as np

from matching_library.mesher import _extract_surface_data
from matching_library.tagger import tag_and_write


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mesh_box(step_path: str) -> dict:
    """Initialize gmsh, import step_path, generate tet mesh, extract surface data.

    Returns the surface_data dict. Does NOT finalize -- caller manages lifecycle.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    mesh_size = max(min(diag / 50.0, 10.0), 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)
    gmsh.model.mesh.generate(3)

    return _extract_surface_data()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tag_and_write_creates_msh_file(tmp_path, tmp_step_box):
    """tag_and_write produces a non-empty .msh file on disk."""
    output_msh = str(tmp_path / "out.msh")

    result = _mesh_box(tmp_step_box)
    surface_tags = result["surface_tags"]
    volume_tags = result["volume_tags"]

    # Assign all surfaces to a single group "wall"
    match_results = {
        tag: {"groups": {"wall": 1}, "total_facets": 2, "unmatched_facets": 0}
        for tag in surface_tags
    }

    tag_and_write(match_results, volume_tags, output_msh)
    gmsh.finalize()

    assert os.path.exists(output_msh), "Output .msh file should exist"
    assert os.path.getsize(output_msh) > 0, "Output .msh file should be non-empty"


def test_msh_contains_named_physical_groups(tmp_path, tmp_step_box):
    """Named groups 'inlet' and 'outlet' appear in the output .msh physical groups."""
    output_msh = str(tmp_path / "out.msh")

    result = _mesh_box(tmp_step_box)
    surface_tags = result["surface_tags"]  # 6 for unit box
    volume_tags = result["volume_tags"]

    # Assign first surface to 'inlet', second to 'outlet', rest are None
    match_results = {}
    for i, tag in enumerate(surface_tags):
        if i == 0:
            groups = {"inlet": 10}
        elif i == 1:
            groups = {"outlet": 10}
        else:
            groups = {}
        match_results[tag] = {
            "groups": groups,
            "total_facets": 10,
            "unmatched_facets": 0,
        }

    tag_and_write(match_results, volume_tags, output_msh)
    gmsh.finalize()

    # Reload and check physical group names
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.merge(output_msh)

    physical_groups = gmsh.model.getPhysicalGroups()
    names = {gmsh.model.getPhysicalName(dim, tag) for dim, tag in physical_groups}
    gmsh.finalize()

    assert "inlet" in names, f"Expected 'inlet' in physical group names, got: {names}"
    assert "outlet" in names, f"Expected 'outlet' in physical group names, got: {names}"


def test_untagged_surfaces_get_untagged_group(tmp_path, tmp_step_box):
    """Surfaces with group=None are collected into '_untagged' physical group."""
    output_msh = str(tmp_path / "out.msh")

    result = _mesh_box(tmp_step_box)
    surface_tags = result["surface_tags"]  # 6 surfaces for unit box
    volume_tags = result["volume_tags"]

    # Only tag one surface; rest are unmatched
    match_results = {}
    for i, tag in enumerate(surface_tags):
        groups = {"tagged_surface": 5} if i == 0 else {}
        match_results[tag] = {
            "groups": groups,
            "total_facets": 5,
            "unmatched_facets": 0,
        }

    tag_and_write(match_results, volume_tags, output_msh)
    gmsh.finalize()

    # Reload and verify _untagged group exists
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.merge(output_msh)

    physical_groups = gmsh.model.getPhysicalGroups()
    names = {gmsh.model.getPhysicalName(dim, tag) for dim, tag in physical_groups}
    gmsh.finalize()

    assert "_untagged" in names, f"Expected '_untagged' in physical group names, got: {names}"
    assert "tagged_surface" in names, f"Expected 'tagged_surface' in physical group names, got: {names}"


def test_volume_gets_domain_group(tmp_path, tmp_step_box):
    """The volume is tagged as 'domain' physical group in the output .msh."""
    output_msh = str(tmp_path / "out.msh")

    result = _mesh_box(tmp_step_box)
    surface_tags = result["surface_tags"]
    volume_tags = result["volume_tags"]

    match_results = {
        tag: {"groups": {"wall": 1}, "total_facets": 2, "unmatched_facets": 0}
        for tag in surface_tags
    }

    tag_and_write(match_results, volume_tags, output_msh)
    gmsh.finalize()

    # Reload and verify 'domain' group with dim=3
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.merge(output_msh)

    physical_groups = gmsh.model.getPhysicalGroups()
    names_by_dim = {}
    for dim, ptag in physical_groups:
        names_by_dim.setdefault(dim, set()).add(gmsh.model.getPhysicalName(dim, ptag))
    gmsh.finalize()

    assert "domain" in names_by_dim.get(3, set()), (
        f"Expected 'domain' in dim=3 physical groups, got: {names_by_dim}"
    )


def test_all_boundary_triangles_preserved(tmp_path, tmp_step_box):
    """Total boundary triangle count is preserved after tagging (no silent element loss)."""
    output_msh = str(tmp_path / "out.msh")

    result = _mesh_box(tmp_step_box)
    surface_tags = result["surface_tags"]
    volume_tags = result["volume_tags"]

    # Count original boundary triangles
    original_tri_count = sum(
        len(result["surface_data"][tag]["tris"]) for tag in surface_tags
    )

    # Mix of tagged and untagged surfaces
    match_results = {}
    for i, tag in enumerate(surface_tags):
        groups = {"top": 5} if i == 0 else {}
        match_results[tag] = {
            "groups": groups,
            "total_facets": 5,
            "unmatched_facets": 0,
        }

    tag_and_write(match_results, volume_tags, output_msh)
    gmsh.finalize()

    # Reload and count boundary triangles
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.merge(output_msh)

    reloaded = _extract_surface_data()
    reloaded_tri_count = sum(
        len(reloaded["surface_data"][tag]["tris"])
        for tag in reloaded["surface_tags"]
    )
    gmsh.finalize()

    assert reloaded_tri_count == original_tri_count, (
        f"Boundary triangle count changed: {original_tri_count} -> {reloaded_tri_count}. "
        "Missing _untagged group causes silent element loss."
    )
