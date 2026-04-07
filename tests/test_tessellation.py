"""
Tests for CONV-01 and CONV-02: STEP tessellation with per-face identity preservation.

Covers:
  - BREP surface counts from tessellate_step return value
  - STL file contains one named solid section per BREP surface
  - Per-surface nodes are present in the Gmsh model
  - All facets are present (no dropped boundary-node triangles)
"""

import os
import re
import gmsh
from step_converter import tessellate_step


def _count_stl_solids(stl_path: str) -> int:
    """Count the number of 'solid ' opener lines in an ASCII STL file."""
    count = 0
    with open(stl_path, "r", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("solid ") and not stripped.startswith("solid\n"):
                count += 1
            elif stripped == "solid":
                # degenerate header — counts as one merged solid
                count += 1
    return count


def _count_stl_facets_per_solid(stl_path: str) -> dict[str, int]:
    """Return {solid_name: facet_count} for each solid section."""
    with open(stl_path, "r") as fh:
        content = fh.read()
    sections = re.split(r"(?=^solid )", content, flags=re.MULTILINE)
    result = {}
    for s in sections:
        s = s.strip()
        if not s:
            continue
        name = s.split("\n")[0].strip()
        result[name] = s.count("facet normal")
    return result


# ---------------------------------------------------------------------------
# CONV-01: face count correctness
# ---------------------------------------------------------------------------

def test_box_face_count(tmp_step_box, tmp_path):
    """Box has 6 BREP surfaces; tessellate_step must return n_surfaces=6."""
    out_stl = str(tmp_path / "box.stl")
    result = tessellate_step(tmp_step_box, out_stl)
    assert result["n_surfaces"] == 6, f"Expected 6 surfaces, got {result['n_surfaces']}"


def test_cylinder_face_count(tmp_step_cylinder, tmp_path):
    """Cylinder has 3 BREP surfaces; tessellate_step must return n_surfaces=3."""
    out_stl = str(tmp_path / "cylinder.stl")
    result = tessellate_step(tmp_step_cylinder, out_stl)
    assert result["n_surfaces"] == 3, f"Expected 3 surfaces, got {result['n_surfaces']}"


# ---------------------------------------------------------------------------
# CONV-02: per-surface identity in STL output
# ---------------------------------------------------------------------------

def test_stl_has_multiple_solids(tmp_step_box, tmp_path):
    """STL from box.step must contain exactly 6 'solid ' sections."""
    out_stl = str(tmp_path / "box_ms.stl")
    tessellate_step(tmp_step_box, out_stl)
    n_solids = _count_stl_solids(out_stl)
    assert n_solids == 6, (
        f"Expected 6 solid sections in STL, found {n_solids}. "
        "Likely StlOneSolidPerSurface produced a merged solid and fallback failed."
    )


def test_stl_solid_count_matches_brep(tmp_step_box, tmp_path):
    """Number of 'endsolid' markers in STL must equal n_surfaces from return value."""
    out_stl = str(tmp_path / "box_sc.stl")
    result = tessellate_step(tmp_step_box, out_stl)
    endsolid_count = 0
    with open(out_stl, "r", errors="replace") as fh:
        for line in fh:
            if line.strip().startswith("endsolid"):
                endsolid_count += 1
    assert endsolid_count == result["n_surfaces"], (
        f"endsolid count ({endsolid_count}) != n_surfaces ({result['n_surfaces']})"
    )


def test_per_surface_nodes(tmp_step_box, tmp_path):
    """Each surface tag must have at least one node in the mesh."""
    out_stl = str(tmp_path / "box_psn.stl")
    result = tessellate_step(tmp_step_box, out_stl)

    # Re-open the same STEP to inspect the mesh (tessellate_step calls gmsh.finalize)
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.importShapes(tmp_step_box)
    gmsh.model.occ.synchronize()
    import math
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    mesh_size = max(min(diag / 50.0, 10.0), 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)
    gmsh.model.mesh.generate(2)

    for tag in result["surface_tags"]:
        node_tags, coords, _ = gmsh.model.mesh.getNodes(dim=2, tag=tag)
        assert len(node_tags) > 0, f"Surface tag {tag} has no nodes"

    gmsh.finalize()


def test_no_empty_solid_sections(tmp_step_box, tmp_path):
    """Every solid section in the STL must have at least one facet (no dropped triangles)."""
    out_stl = str(tmp_path / "box_noempty.stl")
    result = tessellate_step(tmp_step_box, out_stl)
    facets_per_solid = _count_stl_facets_per_solid(out_stl)

    assert len(facets_per_solid) == result["n_surfaces"], (
        f"Expected {result['n_surfaces']} solids, found {len(facets_per_solid)}"
    )
    for name, count in facets_per_solid.items():
        assert count > 0, f"Solid section '{name}' has 0 facets — boundary nodes likely dropped"


# ---------------------------------------------------------------------------
# tri_counts: per-surface triangle counts
# ---------------------------------------------------------------------------

def test_tri_counts_returned(tmp_step_box, tmp_path):
    """tessellate_step must return tri_counts matching surface_tags length."""
    out_stl = str(tmp_path / "box_tc.stl")
    result = tessellate_step(tmp_step_box, out_stl)
    assert "tri_counts" in result, "tri_counts missing from tessellate_step result"
    assert len(result["tri_counts"]) == len(result["surface_tags"])
    assert all(c > 0 for c in result["tri_counts"]), "Every surface should have >0 triangles"


def test_tri_counts_match_stl_facets(tmp_step_box, tmp_path):
    """tri_counts must match the actual facet counts in the STL file."""
    out_stl = str(tmp_path / "box_tcm.stl")
    result = tessellate_step(tmp_step_box, out_stl)
    facets_per_solid = _count_stl_facets_per_solid(out_stl)

    for tag, expected_count in zip(result["surface_tags"], result["tri_counts"]):
        solid_name = f"solid face_{tag}"
        actual_count = facets_per_solid.get(solid_name, 0)
        assert actual_count == expected_count, (
            f"Surface {tag}: tri_counts says {expected_count}, STL has {actual_count}"
        )
