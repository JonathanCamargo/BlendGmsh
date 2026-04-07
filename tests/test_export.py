"""
Unit tests for blender_addon/export.py pure-Python helper.

All tests use mock objects (no bpy import) to simulate Blender mesh data.
This follows the established project pattern: pure-Python helpers tested
outside Blender.
"""

import json
import pathlib
import re

import jsonschema
import numpy as np
import pytest

from blender_addon.export import build_bc_groups_dict, generate_group_colors, detect_bc_mode
from step_converter import tessellate_step


SCHEMA_PATH = pathlib.Path(__file__).parent.parent / "schema" / "bc_groups_v1.json"


def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Mock Blender data structures
# ---------------------------------------------------------------------------

class MockVertex:
    """Mimics bpy.types.MeshVertex."""
    def __init__(self, index, co):
        self.index = index
        self.co = co


class MockPolygon:
    """Mimics bpy.types.MeshPolygon."""
    def __init__(self, vertices):
        self.vertices = vertices


class MockVertexGroup:
    """Mimics bpy.types.VertexGroup.

    assigned_indices: set of vertex indices that belong to this group.
    Calling weight() for an unassigned index raises RuntimeError (matches Blender API).
    """
    def __init__(self, name, assigned_indices):
        self.name = name
        self._assigned = set(assigned_indices)

    def weight(self, index):
        if index in self._assigned:
            return 1.0
        raise RuntimeError(f"Vertex {index} not in group {self.name}")


class MockMatrix:
    """Mimics mathutils.Matrix for affine transforms."""
    def __init__(self, translate=(0.0, 0.0, 0.0)):
        self._translate = translate

    def __matmul__(self, vec):
        return _TranslatedVector(
            [vec[0] + self._translate[0],
             vec[1] + self._translate[1],
             vec[2] + self._translate[2]]
        )

    def __eq__(self, other):
        if isinstance(other, MockMatrix):
            return self._translate == other._translate
        if isinstance(other, _IdentityMatrix):
            return self._translate == (0.0, 0.0, 0.0)
        return NotImplemented


class _TranslatedVector(list):
    pass


class _IdentityMatrix:
    pass


class _MockMeshData:
    def __init__(self, vertices, polygons):
        self.vertices = vertices
        self.polygons = polygons


class MockMeshObject:
    """Mimics a bpy mesh object with .data, .vertex_groups, .matrix_world."""
    def __init__(self, vertices, polygons, vertex_groups, translate=(0.0, 0.0, 0.0)):
        self.data = _MockMeshData(vertices, polygons)
        self.vertex_groups = vertex_groups
        self.matrix_world = MockMatrix(translate)


# ---------------------------------------------------------------------------
# Helper: build a mesh with face_N vertex groups (simulating STEP import)
# and a BC group that references some of those faces.
#
# 4 vertices, 2 triangles, 2 BREP surfaces (face_1, face_2)
# BC group "inlet" covers face_1 vertices
# ---------------------------------------------------------------------------

def make_mesh_with_surface_tags(translate=(0.0, 0.0, 0.0)):
    vertices = [
        MockVertex(0, [0.0, 0.0, 0.0]),
        MockVertex(1, [1.0, 0.0, 0.0]),
        MockVertex(2, [1.0, 1.0, 0.0]),
        MockVertex(3, [0.0, 1.0, 0.0]),
    ]
    polygons = [
        MockPolygon((0, 1, 2)),  # belongs to face_1
        MockPolygon((0, 2, 3)),  # belongs to face_2
    ]
    # Simulate STEP import: face_1 owns verts 0,1,2 and face_2 owns verts 0,2,3
    face_1_group = MockVertexGroup("face_1", {0, 1, 2})
    face_2_group = MockVertexGroup("face_2", {0, 2, 3})
    # BC group "inlet" assigned to same verts as face_1
    inlet_group = MockVertexGroup("inlet", {0, 1, 2})

    all_groups = [face_1_group, face_2_group, inlet_group]
    return MockMeshObject(vertices, polygons, all_groups, translate=translate)


def make_mesh_two_bc_groups():
    """Mesh with 2 BC groups mapping to different BREP surfaces."""
    vertices = [
        MockVertex(0, [0.0, 0.0, 0.0]),
        MockVertex(1, [1.0, 0.0, 0.0]),
        MockVertex(2, [1.0, 1.0, 0.0]),
        MockVertex(3, [0.0, 1.0, 0.0]),
    ]
    polygons = [
        MockPolygon((0, 1, 2)),
        MockPolygon((0, 2, 3)),
    ]
    face_1_group = MockVertexGroup("face_1", {0, 1, 2})
    face_2_group = MockVertexGroup("face_2", {0, 2, 3})
    inlet_group = MockVertexGroup("inlet", {0, 1, 2})
    outlet_group = MockVertexGroup("outlet", {0, 2, 3})

    all_groups = [face_1_group, face_2_group, inlet_group, outlet_group]
    return MockMeshObject(vertices, polygons, all_groups)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_bc_groups_dict_schema_valid():
    """build_bc_groups_dict() output passes jsonschema validation."""
    obj = make_mesh_with_surface_tags()
    schema = load_schema()
    # Only pass BC groups (not face_N groups) as the `groups` argument
    bc_groups = [g for g in obj.vertex_groups if not g.name.startswith("face_")]
    result = build_bc_groups_dict(
        obj=obj,
        groups=bc_groups,
        step_filepath="test.step",
        blender_version="4.2.0",
    )
    jsonschema.validate(instance=result, schema=schema)


def test_surface_tags_exported():
    """Groups in output contain surface_tags mapping to BREP face tags."""
    obj = make_mesh_with_surface_tags()
    bc_groups = [g for g in obj.vertex_groups if not g.name.startswith("face_")]
    result = build_bc_groups_dict(
        obj=obj,
        groups=bc_groups,
        step_filepath="test.step",
        blender_version="4.2.0",
    )
    assert "inlet" in result["groups"]
    assert "surface_tags" in result["groups"]["inlet"]
    assert 1 in result["groups"]["inlet"]["surface_tags"]


def test_two_bc_groups_separate_tags():
    """Two BC groups map to different surface tags."""
    obj = make_mesh_two_bc_groups()
    bc_groups = [g for g in obj.vertex_groups if not g.name.startswith("face_")]
    result = build_bc_groups_dict(
        obj=obj,
        groups=bc_groups,
        step_filepath="test.step",
        blender_version="4.2.0",
    )
    assert "inlet" in result["groups"]
    assert "outlet" in result["groups"]
    assert 1 in result["groups"]["inlet"]["surface_tags"]
    assert 2 in result["groups"]["outlet"]["surface_tags"]


def test_empty_group_excluded():
    """A group with no assigned vertices must not appear in output."""
    vertices = [
        MockVertex(0, [0.0, 0.0, 0.0]),
        MockVertex(1, [1.0, 0.0, 0.0]),
        MockVertex(2, [1.0, 1.0, 0.0]),
    ]
    polygons = [MockPolygon((0, 1, 2))]
    face_1_group = MockVertexGroup("face_1", {0, 1, 2})
    empty_group = MockVertexGroup("empty_bc", set())
    obj = MockMeshObject(vertices, polygons, [face_1_group, empty_group])

    result = build_bc_groups_dict(
        obj=obj,
        groups=[empty_group],
        step_filepath="test.step",
        blender_version="4.2.0",
    )
    assert "empty_bc" not in result["groups"]


def test_distinct_colors():
    """generate_group_colors(5) returns 5 RGB tuples that are all distinct."""
    colors = generate_group_colors(5)
    assert len(colors) == 5
    color_set = {tuple(c) for c in colors}
    assert len(color_set) == 5


# ---------------------------------------------------------------------------
# Mesh mode tests (no face_N vertex groups)
# ---------------------------------------------------------------------------

def make_mesh_no_brep():
    """Mesh with NO face_N vertex groups -- simulates STL/OBJ import."""
    vertices = [
        MockVertex(0, [0.0, 0.0, 0.0]),
        MockVertex(1, [1.0, 0.0, 0.0]),
        MockVertex(2, [1.0, 1.0, 0.0]),
        MockVertex(3, [0.0, 1.0, 0.0]),
    ]
    polygons = [
        MockPolygon((0, 1, 2)),
        MockPolygon((0, 2, 3)),
    ]
    inlet_group = MockVertexGroup("inlet", {0, 1, 2})
    outlet_group = MockVertexGroup("outlet", {0, 2, 3})
    return MockMeshObject(vertices, polygons, [inlet_group, outlet_group])


def test_detect_bc_mode_brep():
    """detect_bc_mode returns 'brep' when face_N groups exist."""
    obj = make_mesh_with_surface_tags()
    assert detect_bc_mode(obj) == "brep"


def test_detect_bc_mode_mesh():
    """detect_bc_mode returns 'mesh' when no face_N groups exist."""
    obj = make_mesh_no_brep()
    assert detect_bc_mode(obj) == "mesh"


def test_mesh_mode_exports_vertices():
    """Mesh mode export produces vertices and face_vertex_indices, not surface_tags."""
    obj = make_mesh_no_brep()
    result = build_bc_groups_dict(
        obj=obj,
        groups=obj.vertex_groups,
        step_filepath="test.stl",
        blender_version="4.2.0",
    )
    assert result["mode"] == "mesh"
    assert "inlet" in result["groups"]
    inlet = result["groups"]["inlet"]
    assert "vertices" in inlet
    assert "face_vertex_indices" in inlet
    assert "vertex_count" in inlet
    assert "face_count" in inlet
    assert "surface_tags" not in inlet


def test_mesh_mode_schema_valid():
    """Mesh mode export passes schema validation."""
    obj = make_mesh_no_brep()
    schema = load_schema()
    result = build_bc_groups_dict(
        obj=obj,
        groups=obj.vertex_groups,
        step_filepath="test.stl",
        blender_version="4.2.0",
    )
    jsonschema.validate(instance=result, schema=schema)


def test_mesh_mode_vertex_coords():
    """Mesh mode exports correct world-space coordinates."""
    obj = make_mesh_no_brep()
    result = build_bc_groups_dict(
        obj=obj,
        groups=obj.vertex_groups,
        step_filepath="test.stl",
        blender_version="4.2.0",
    )
    inlet = result["groups"]["inlet"]
    assert inlet["vertex_count"] == 3
    assert inlet["face_count"] == 1
    # Vertices 0, 1, 2 in world space (identity transform)
    assert [0.0, 0.0, 0.0] in inlet["vertices"]
    assert [1.0, 0.0, 0.0] in inlet["vertices"]
    assert [1.0, 1.0, 0.0] in inlet["vertices"]


def test_mesh_stats_present():
    """Output contains mesh_stats with bounding box."""
    obj = make_mesh_with_surface_tags()
    bc_groups = [g for g in obj.vertex_groups if not g.name.startswith("face_")]
    result = build_bc_groups_dict(
        obj=obj,
        groups=bc_groups,
        step_filepath="test.step",
        blender_version="4.2.0",
    )
    assert "mesh_stats" in result
    assert "bounding_box" in result["mesh_stats"]
    assert result["mesh_stats"]["total_vertices"] == 4
    assert result["mesh_stats"]["total_faces"] == 2


# ---------------------------------------------------------------------------
# Strict surface_tags tests (boundary vertex contamination)
# ---------------------------------------------------------------------------

def make_box_like_mesh():
    """Mesh mimicking 3 surfaces of a box with shared boundary vertices.

    Surface layout (6 polygons across 3 BREP surfaces):
      face_10: polygons 0,1 (verts 0-4, front face)
      face_20: polygons 2,3 (verts 2-6, right face, shares verts 2,3 with face_10)
      face_30: polygons 4,5 (verts 4-8, top face, shares verts 4 with face_10, vert 5 from face_20)
    """
    vertices = [MockVertex(i, [float(i), 0.0, 0.0]) for i in range(9)]
    polygons = [
        # face_10 polygons
        MockPolygon((0, 1, 2)),   # poly 0
        MockPolygon((0, 2, 3)),   # poly 1
        # face_20 polygons (shares verts 2, 3 with face_10)
        MockPolygon((2, 3, 4)),   # poly 2
        MockPolygon((3, 4, 5)),   # poly 3
        # face_30 polygons (shares vert 4 with face_20, vert 5 with face_20)
        MockPolygon((4, 5, 6)),   # poly 4
        MockPolygon((5, 6, 7)),   # poly 5
    ]
    # face_N groups: each surface's polygons' vertices
    face_10 = MockVertexGroup("face_10", {0, 1, 2, 3})
    face_20 = MockVertexGroup("face_20", {2, 3, 4, 5})
    face_30 = MockVertexGroup("face_30", {4, 5, 6, 7})
    # BC groups: "fixed" = front face (face_10), "force" = top face (face_30)
    fixed = MockVertexGroup("fixed", {0, 1, 2, 3})
    force = MockVertexGroup("force", {4, 5, 6, 7})

    all_groups = [face_10, face_20, face_30, fixed, force]
    return MockMeshObject(vertices, polygons, all_groups)


def test_bc_groups_exact_surface_tags():
    """BC groups must map to EXACT surface tags, not leak into neighbors via shared vertices."""
    obj = make_box_like_mesh()
    bc_groups = [g for g in obj.vertex_groups if not g.name.startswith("face_")]
    result = build_bc_groups_dict(
        obj=obj,
        groups=bc_groups,
        step_filepath="test.step",
        blender_version="4.2.0",
    )
    # "fixed" should map to ONLY face_10, not face_20
    assert result["groups"]["fixed"]["surface_tags"] == [10], (
        f"Expected [10], got {result['groups']['fixed']['surface_tags']}"
    )
    # "force" should map to ONLY face_30, not face_20
    assert result["groups"]["force"]["surface_tags"] == [30], (
        f"Expected [30], got {result['groups']['force']['surface_tags']}"
    )


def test_two_bc_groups_exact_tags():
    """Stricter version: inlet must map to EXACTLY [1], outlet to EXACTLY [2]."""
    obj = make_mesh_two_bc_groups()
    bc_groups = [g for g in obj.vertex_groups if not g.name.startswith("face_")]
    result = build_bc_groups_dict(
        obj=obj,
        groups=bc_groups,
        step_filepath="test.step",
        blender_version="4.2.0",
    )
    assert result["groups"]["inlet"]["surface_tags"] == [1], (
        f"Expected [1], got {result['groups']['inlet']['surface_tags']}"
    )
    assert result["groups"]["outlet"]["surface_tags"] == [2], (
        f"Expected [2], got {result['groups']['outlet']['surface_tags']}"
    )


# ---------------------------------------------------------------------------
# End-to-end: real STEP -> STL -> simulated Blender import -> export
# Uses centroid matching (assign_by_centroids) and deliberately shuffles
# polygon order to prove robustness against Blender STL importer reordering.
# ---------------------------------------------------------------------------

from blender_addon.vertex_groups import assign_by_centroids
import random


def _parse_stl_solids(stl_path: str):
    """Parse ASCII STL into per-solid triangle vertex coordinates.

    Returns list of (solid_name, list_of_triangles) where each triangle
    is a list of 3 (x, y, z) tuples.
    """
    with open(stl_path) as f:
        content = f.read()
    solids = []
    for match in re.finditer(
        r"solid (face_\d+)\n(.*?)endsolid face_\d+", content, re.DOTALL
    ):
        name = match.group(1)
        body = match.group(2)
        tris = []
        verts = re.findall(r"vertex\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)", body)
        for i in range(0, len(verts), 3):
            tri = [tuple(float(c) for c in v) for v in verts[i:i+3]]
            if len(tri) == 3:
                tris.append(tri)
        solids.append((name, tris))
    return solids


def _simulate_blender_import(solids, shuffle=False):
    """Simulate Blender STL import: merge coincident vertices, create polygons.

    Parameters
    ----------
    solids : list of (solid_name, list_of_triangles)
    shuffle : bool
        If True, randomly shuffle polygon order to simulate Blender reordering.

    Returns (vertices, polygons, surface_tag_per_polygon).
    """
    # First pass: collect all triangles with their ground truth tags
    raw_polys = []  # (tag, (v0_coords, v1_coords, v2_coords))
    for solid_name, tris in solids:
        tag = int(solid_name.split("_")[1])
        for tri_coords in tris:
            raw_polys.append((tag, tri_coords))

    if shuffle:
        random.Random(42).shuffle(raw_polys)

    # Second pass: merge vertices and build polygon list
    merged_verts = []
    coord_to_idx = {}
    polygons = []
    poly_surface_tags = []
    MERGE_PRECISION = 10

    for tag, tri_coords in raw_polys:
        poly_vert_indices = []
        for x, y, z in tri_coords:
            key = (round(x, MERGE_PRECISION), round(y, MERGE_PRECISION), round(z, MERGE_PRECISION))
            if key not in coord_to_idx:
                coord_to_idx[key] = len(merged_verts)
                merged_verts.append([x, y, z])
            poly_vert_indices.append(coord_to_idx[key])
        polygons.append(tuple(poly_vert_indices))
        poly_surface_tags.append(tag)

    return merged_verts, polygons, poly_surface_tags


def _build_mock_with_centroids(merged_verts, polygons, face_centroids,
                                bc_assignments):
    """Build a MockMeshObject with face_N vertex groups from centroid matching
    and BC groups from bc_assignments.

    bc_assignments: dict of {bc_name: surface_tag} -- each BC group gets
                    the vertices of the specified surface.
    """
    mock_verts = [MockVertex(i, co) for i, co in enumerate(merged_verts)]
    mock_polys = [MockPolygon(p) for p in polygons]

    # Use assign_by_centroids to determine face_N groups (same as real addon)
    tag_to_verts = assign_by_centroids(mock_verts, mock_polys, face_centroids)

    vg_list = []
    for tag in sorted(tag_to_verts.keys()):
        vg_list.append(MockVertexGroup(f"face_{tag}", tag_to_verts[tag]))

    # Add BC groups
    for bc_name, surface_tag in bc_assignments.items():
        vg_list.append(MockVertexGroup(bc_name, tag_to_verts.get(surface_tag, set())))

    return MockMeshObject(mock_verts, mock_polys, vg_list)


def test_roundtrip_box_centroid_matching(tmp_step_box, tmp_path):
    """Box STEP round-trip: centroid matching correctly maps all polygons
    even when polygon order is shuffled."""
    stl_path = str(tmp_path / "box_rt.stl")
    result = tessellate_step(tmp_step_box, stl_path)
    face_centroids = result["face_centroids"]

    solids = _parse_stl_solids(stl_path)
    # SHUFFLE polygons to simulate Blender reordering
    merged_verts, polygons, ground_truth = _simulate_blender_import(solids, shuffle=True)

    mock_verts = [MockVertex(i, co) for i, co in enumerate(merged_verts)]
    mock_polys = [MockPolygon(p) for p in polygons]

    tag_to_verts = assign_by_centroids(mock_verts, mock_polys, face_centroids)

    # Verify: every polygon's ground truth tag should own all its vertices
    errors = 0
    for pi, poly in enumerate(polygons):
        expected_tag = ground_truth[pi]
        expected_verts = tag_to_verts.get(expected_tag, set())
        if not all(vi in expected_verts for vi in poly):
            errors += 1

    assert errors == 0, (
        f"{errors}/{len(polygons)} polygons not matched correctly after shuffle"
    )


def test_roundtrip_bc_groups_shuffled(tmp_step_box, tmp_path):
    """BC group export produces correct exact tags even with shuffled polygons."""
    stl_path = str(tmp_path / "box_bc.stl")
    result = tessellate_step(tmp_step_box, stl_path)
    surface_tags = result["surface_tags"]
    face_centroids = result["face_centroids"]

    solids = _parse_stl_solids(stl_path)
    merged_verts, polygons, _ = _simulate_blender_import(solids, shuffle=True)

    tag_fixed = surface_tags[0]
    tag_force = surface_tags[3]

    obj = _build_mock_with_centroids(
        merged_verts, polygons, face_centroids,
        {"fixed": tag_fixed, "force": tag_force},
    )
    bc_groups = [g for g in obj.vertex_groups if not g.name.startswith("face_")]

    export = build_bc_groups_dict(
        obj=obj, groups=bc_groups,
        step_filepath="test.step", blender_version="4.2.0",
    )

    assert export["groups"]["fixed"]["surface_tags"] == [tag_fixed], (
        f"fixed: expected [{tag_fixed}], got {export['groups']['fixed']['surface_tags']}"
    )
    assert export["groups"]["force"]["surface_tags"] == [tag_force], (
        f"force: expected [{tag_force}], got {export['groups']['force']['surface_tags']}"
    )


def test_roundtrip_cylinder_shuffled(tmp_step_cylinder, tmp_path):
    """Cylinder round-trip with shuffled polygons and curved geometry."""
    stl_path = str(tmp_path / "cyl_rt.stl")
    result = tessellate_step(tmp_step_cylinder, stl_path)
    surface_tags = result["surface_tags"]
    tri_counts = result["tri_counts"]
    face_centroids = result["face_centroids"]

    solids = _parse_stl_solids(stl_path)
    merged_verts, polygons, _ = _simulate_blender_import(solids, shuffle=True)

    biggest_tag = surface_tags[tri_counts.index(max(tri_counts))]

    obj = _build_mock_with_centroids(
        merged_verts, polygons, face_centroids,
        {"wall": biggest_tag},
    )
    bc_groups = [g for g in obj.vertex_groups if not g.name.startswith("face_")]

    export = build_bc_groups_dict(
        obj=obj, groups=bc_groups,
        step_filepath="cyl.step", blender_version="4.2.0",
    )

    assert export["groups"]["wall"]["surface_tags"] == [biggest_tag], (
        f"wall: expected [{biggest_tag}], got {export['groups']['wall']['surface_tags']}"
    )
