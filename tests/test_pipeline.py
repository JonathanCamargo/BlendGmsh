"""
End-to-end integration tests for run_full_pipeline and tag_existing_mesh.

The new pipeline assigns physical groups to BREP surfaces by tag before meshing,
so tests provide surface_tags directly (no KDTree matching needed).
"""

import json
import math
import os
import pytest
import gmsh

from matching_library import run_full_pipeline, tag_existing_mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_box_surface_tags(step_path):
    """Get BREP surface tags from a box STEP file, grouped by z-coordinate."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    top_tags = []
    bot_tags = []
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
    groups_data = {
        name: {"surface_tags": tags}
        for name, tags in groups_dict.items()
    }
    data = {
        "schema_version": 1,
        "source": "test",
        "step_file": step_path,
        "units": "meters",
        "groups": groups_data,
        "mesh_stats": {
            "total_vertices": 0,
            "total_faces": 0,
            "bounding_box": {"min": [0, 0, 0], "max": [1, 1, 1]},
        },
    }
    json_path = str(tmp_path / "bc_groups.json")
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_pipeline_box(tmp_path, tmp_step_box):
    """run_full_pipeline produces .msh with named physical groups."""
    output_msh = str(tmp_path / "tagged.msh")
    top_tags, bot_tags = _get_box_surface_tags(tmp_step_box)

    json_path = _make_bc_groups_json(
        tmp_step_box, {"top": top_tags, "bottom": bot_tags}, tmp_path
    )
    report = run_full_pipeline(json_path, tmp_step_box, output_msh)

    assert os.path.exists(output_msh)
    assert os.path.getsize(output_msh) > 0

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.merge(output_msh)
    names = {gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()}
    gmsh.finalize()

    assert "top" in names
    assert "bottom" in names


def test_full_pipeline_coverage_report(tmp_path, tmp_step_box):
    """run_full_pipeline returns a CoverageReport with group_stats."""
    output_msh = str(tmp_path / "tagged.msh")
    top_tags, bot_tags = _get_box_surface_tags(tmp_step_box)

    json_path = _make_bc_groups_json(
        tmp_step_box, {"top": top_tags, "bottom": bot_tags}, tmp_path
    )
    report = run_full_pipeline(json_path, tmp_step_box, output_msh)

    assert "top" in report.group_stats
    assert "bottom" in report.group_stats


def test_tagging_only_mode(tmp_path, tmp_step_box):
    """tag_existing_mesh re-tags an existing .msh with named physical groups."""
    raw_msh = str(tmp_path / "raw.msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.importShapes(tmp_step_box)
    gmsh.model.occ.synchronize()
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
    ms = max(min(diag / 50.0, 10.0), 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms * 2.0)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(raw_msh)
    gmsh.finalize()

    top_tags, bot_tags = _get_box_surface_tags(tmp_step_box)
    json_path = _make_bc_groups_json(
        raw_msh, {"top_face": top_tags, "bottom_face": bot_tags}, tmp_path
    )

    output_msh = str(tmp_path / "tagged.msh")
    report = tag_existing_mesh(json_path, raw_msh, output_msh)
    assert os.path.exists(output_msh)


def test_pipeline_validates_json_schema(tmp_path, tmp_step_box):
    """run_full_pipeline raises ValidationError for invalid JSON."""
    import jsonschema
    output_msh = str(tmp_path / "out.msh")
    invalid_data = {
        "source": "test",
        "step_file": tmp_step_box,
        "units": "meters",
        "groups": {},
        "mesh_stats": {
            "total_vertices": 0, "total_faces": 0,
            "bounding_box": {"min": [0, 0, 0], "max": [1, 1, 1]},
        },
    }
    json_path = str(tmp_path / "invalid.json")
    with open(json_path, "w") as f:
        json.dump(invalid_data, f)

    with pytest.raises(jsonschema.ValidationError):
        run_full_pipeline(json_path, tmp_step_box, output_msh)


def test_pipeline_invalid_tag_raises(tmp_path, tmp_step_box):
    """run_full_pipeline raises RuntimeError when a group references non-existent surface tags."""
    output_msh = str(tmp_path / "out.msh")
    json_path = _make_bc_groups_json(
        tmp_step_box, {"bad_group": [9999]}, tmp_path
    )
    with pytest.raises(RuntimeError):
        run_full_pipeline(json_path, tmp_step_box, output_msh)


# ---------------------------------------------------------------------------
# Mesh mode pipeline tests
# ---------------------------------------------------------------------------

def _make_mesh_mode_json(step_path, groups_dict, tmp_path):
    """Write a mesh-mode bc_groups JSON. groups_dict = {name: [[x,y,z], ...]}."""
    groups_data = {
        name: {
            "vertices": verts,
            "face_vertex_indices": [],
            "vertex_count": len(verts),
            "face_count": 0,
        }
        for name, verts in groups_dict.items()
    }
    data = {
        "schema_version": 1,
        "source": "test",
        "mode": "mesh",
        "step_file": step_path,
        "units": "meters",
        "groups": groups_data,
        "mesh_stats": {
            "total_vertices": 0,
            "total_faces": 0,
            "bounding_box": {"min": [0, 0, 0], "max": [1, 1, 1]},
        },
    }
    json_path = str(tmp_path / "bc_mesh_mode.json")
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


def _get_surface_vertices(step_path, z_filter):
    """Mesh the STEP and return actual boundary vertices for surfaces matching z_filter.

    This simulates what Blender would export: actual mesh node positions on a face.
    """
    import numpy as np
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
    ms = max(min(diag / 50.0, 10.0), 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms * 2.0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)
    gmsh.model.mesh.generate(3)

    verts = []
    for _, tag in gmsh.model.getEntities(dim=2):
        bb = gmsh.model.getBoundingBox(2, tag)
        if z_filter(bb[2], bb[5]):
            nodes, coords, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
            if len(coords) > 0:
                verts.extend(np.array(coords).reshape(-1, 3).tolist())
    gmsh.finalize()
    return verts


def test_mesh_mode_pipeline(tmp_path, tmp_step_box):
    """Mesh mode pipeline uses KDTree matching to tag boundary facets."""
    output_msh = str(tmp_path / "mesh_mode.msh")

    # Extract actual mesh vertices from the top face (z=1)
    top_verts = _get_surface_vertices(
        tmp_step_box, z_filter=lambda zmin, zmax: zmin > 0.99
    )
    assert len(top_verts) > 0, "No vertices found on top face"

    json_path = _make_mesh_mode_json(
        tmp_step_box, {"top_face": top_verts}, tmp_path
    )
    report = run_full_pipeline(json_path, tmp_step_box, output_msh)

    assert os.path.exists(output_msh)
    assert "top_face" in report.group_stats
    assert len(report.group_stats["top_face"]["surfaces"]) > 0


def test_mesh_mode_auto_detect(tmp_path, tmp_step_box):
    """Pipeline auto-detects mesh mode from JSON content (no explicit mode field)."""
    top_verts = _get_surface_vertices(
        tmp_step_box, z_filter=lambda zmin, zmax: zmin > 0.99
    )
    groups_data = {
        "top": {
            "vertices": top_verts,
            "face_vertex_indices": [],
            "vertex_count": len(top_verts),
            "face_count": 0,
        }
    }
    data = {
        "schema_version": 1,
        "source": "test",
        "step_file": tmp_step_box,
        "units": "meters",
        "groups": groups_data,
        "mesh_stats": {
            "total_vertices": 0, "total_faces": 0,
            "bounding_box": {"min": [0, 0, 0], "max": [1, 1, 1]},
        },
    }
    json_path = str(tmp_path / "auto_detect.json")
    with open(json_path, "w") as f:
        json.dump(data, f)

    output_msh = str(tmp_path / "auto.msh")
    report = run_full_pipeline(json_path, tmp_step_box, output_msh)
    assert os.path.exists(output_msh)
    assert "top" in report.group_stats
