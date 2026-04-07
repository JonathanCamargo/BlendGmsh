"""
CLI tests for `python -m matching_library` entry point.
"""

import json
import math
import os
import subprocess
import sys
import pytest
import gmsh


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


def _make_bc_groups_json(step_path, groups_dict, tmp_path, filename="bc_groups.json"):
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
    json_path = str(tmp_path / filename)
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


def _make_raw_msh(step_path, out_path):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
    ms = max(min(diag / 50.0, 10.0), 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms * 2.0)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(out_path)
    gmsh.finalize()


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "matching_library", *args],
        capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_help_lists_subcommands():
    result = _run_cli("--help")
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "run" in combined
    assert "tag" in combined


def test_run_help_lists_args():
    result = _run_cli("run", "--help")
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "json" in combined
    assert "step" in combined
    assert "output" in combined


def test_tag_help_lists_args():
    result = _run_cli("tag", "--help")
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "json" in combined
    assert "input_msh" in combined
    assert "output" in combined


def test_run_no_args_exits_nonzero():
    result = _run_cli("run")
    assert result.returncode != 0


def test_no_subcommand_exits_nonzero():
    result = _run_cli()
    assert result.returncode != 0


def test_run_integration(tmp_path, tmp_step_box):
    """CLI 'run' produces a tagged .msh file."""
    top_tags, bot_tags = _get_box_surface_tags(tmp_step_box)
    json_path = _make_bc_groups_json(
        tmp_step_box, {"top": top_tags, "bottom": bot_tags}, tmp_path
    )
    output_msh = str(tmp_path / "output.msh")
    result = _run_cli("run", json_path, tmp_step_box, output_msh)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert os.path.exists(output_msh)
    assert os.path.getsize(output_msh) > 0


def test_tag_integration(tmp_path, tmp_step_box):
    """CLI 'tag' re-tags an existing .msh file."""
    raw_msh = str(tmp_path / "raw.msh")
    _make_raw_msh(tmp_step_box, raw_msh)

    top_tags, bot_tags = _get_box_surface_tags(tmp_step_box)
    json_path = _make_bc_groups_json(
        raw_msh, {"top_face": top_tags, "bottom_face": bot_tags}, tmp_path
    )
    output_msh = str(tmp_path / "tagged.msh")
    result = _run_cli("tag", json_path, raw_msh, output_msh)
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert os.path.exists(output_msh)
    assert os.path.getsize(output_msh) > 0
