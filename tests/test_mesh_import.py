"""
Unit tests for blender_addon/mesh_import.py STL and OBJ parsers.

All tests use inline fixture helpers -- no external files required.
Tests run outside Blender (no bpy dependency in mesh_import.py).
"""

import struct

import pytest

from blender_addon.mesh_import import parse_stl, parse_obj, detect_stl_format


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_binary_stl(path, triangles):
    """Write a valid binary STL file.

    triangles: list of 3-tuples of (x, y, z) float 3-tuples, one per triangle.
    """
    with open(path, 'wb') as f:
        f.write(b'\x00' * 80)  # 80-byte header
        f.write(struct.pack('<I', len(triangles)))  # triangle count
        for tri in triangles:
            f.write(struct.pack('<3f', 0.0, 0.0, 0.0))  # normal (zeros)
            for x, y, z in tri:
                f.write(struct.pack('<3f', x, y, z))
            f.write(b'\x00\x00')  # attribute byte count


def _write_ascii_stl(path, triangles):
    """Write a valid ASCII STL file.

    triangles: list of 3-tuples of (x, y, z) float 3-tuples.
    """
    lines = ['solid test\n']
    for tri in triangles:
        lines.append('facet normal 0 0 0\n')
        lines.append('  outer loop\n')
        for x, y, z in tri:
            lines.append(f'    vertex {x} {y} {z}\n')
        lines.append('  endloop\n')
        lines.append('endfacet\n')
    lines.append('endsolid test\n')
    with open(path, 'w') as f:
        f.writelines(lines)


def _write_obj(path, vertices, faces):
    """Write a valid OBJ file.

    vertices: list of (x, y, z) float tuples.
    faces: list of index tuples (0-based, will be written as 1-based).
    """
    with open(path, 'w') as f:
        for x, y, z in vertices:
            f.write(f'v {x} {y} {z}\n')
        for face in faces:
            f.write('f ' + ' '.join(str(i + 1) for i in face) + '\n')


# ---------------------------------------------------------------------------
# Unit cube test data (8 unique vertices, 12 triangles)
# ---------------------------------------------------------------------------

CUBE_VERTS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
CUBE_TRIS = [
    # bottom (z=0)
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    # top (z=1)
    ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0)),
    ((0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    # front (y=0)
    ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
    # back (y=1)
    ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)),
    ((0.0, 1.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
    # left (x=0)
    ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0)),
    ((0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
    # right (x=1)
    ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
    ((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 0.0)),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_stl_binary(tmp_path):
    path = tmp_path / "cube.stl"
    _write_binary_stl(str(path), CUBE_TRIS)
    vertices, faces = parse_stl(str(path))
    assert len(vertices) == 8, f"Expected 8 unique vertices, got {len(vertices)}"
    assert len(faces) == 12, f"Expected 12 faces, got {len(faces)}"
    # All vertices are 3-tuples of float
    for v in vertices:
        assert len(v) == 3
        assert all(isinstance(c, float) for c in v), f"Vertex coords not float: {v}"
    # All faces are 3-tuples of int
    for f in faces:
        assert len(f) == 3
        assert all(isinstance(i, int) for i in f), f"Face indices not int: {f}"


def test_parse_stl_ascii(tmp_path):
    path = tmp_path / "cube_ascii.stl"
    _write_ascii_stl(str(path), CUBE_TRIS)
    vertices, faces = parse_stl(str(path))
    assert len(vertices) == 8, f"Expected 8 unique vertices, got {len(vertices)}"
    assert len(faces) == 12, f"Expected 12 faces, got {len(faces)}"


def test_detect_stl_format(tmp_path):
    binary_path = tmp_path / "binary.stl"
    ascii_path = tmp_path / "ascii.stl"
    _write_binary_stl(str(binary_path), CUBE_TRIS)
    _write_ascii_stl(str(ascii_path), CUBE_TRIS)
    assert detect_stl_format(str(binary_path)) == 'binary'
    assert detect_stl_format(str(ascii_path)) == 'ascii'


def test_stl_vertex_deduplication(tmp_path):
    """Two triangles sharing an edge: 4 unique vertices, 6 vertex records."""
    # Two triangles sharing edge (1,0,0)-(0,1,0)
    triangles = [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # bottom-left
        ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),  # top-right
    ]
    path = tmp_path / "two_tris.stl"
    _write_binary_stl(str(path), triangles)
    vertices, faces = parse_stl(str(path))
    assert len(vertices) == 4, f"Expected 4 unique vertices, got {len(vertices)}"
    assert len(faces) == 2
    # All face indices must be valid references into the vertex list
    for face in faces:
        for idx in face:
            assert 0 <= idx < len(vertices), f"Face index {idx} out of range"


def test_parse_obj_vertex_only(tmp_path):
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
             (0.0, 0.0, 1.0)]
    # Three triangles
    faces_in = [(0, 1, 2), (0, 1, 3), (0, 2, 3)]
    path = tmp_path / "simple.obj"
    _write_obj(str(path), verts, faces_in)
    vertices, faces = parse_obj(str(path))
    assert len(vertices) == 4
    assert len(faces) == 3
    # Coordinates must match written values
    for i, (x, y, z) in enumerate(verts):
        assert vertices[i] == (x, y, z), f"Vertex {i} mismatch"


def test_parse_obj_vtn_format(tmp_path):
    """f v/vt/vn format: only vertex index should be extracted."""
    content = (
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "vt 0.0 0.0\n"
        "vt 1.0 0.0\n"
        "vt 0.0 1.0\n"
        "vn 0.0 0.0 1.0\n"
        "f 1/1/1 2/2/1 3/3/1\n"
    )
    path = tmp_path / "vtn.obj"
    path.write_text(content)
    vertices, faces = parse_obj(str(path))
    assert len(vertices) == 3
    assert len(faces) == 1
    assert faces[0] == (0, 1, 2)


def test_parse_obj_quads(tmp_path):
    """Quad face fan-triangulates to 2 triangles."""
    content = (
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 1.0 1.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "f 1 2 3 4\n"
    )
    path = tmp_path / "quad.obj"
    path.write_text(content)
    vertices, faces = parse_obj(str(path))
    assert len(faces) == 2, f"Expected 2 triangles from quad, got {len(faces)}"
    assert faces[0] == (0, 1, 2)
    assert faces[1] == (0, 2, 3)


def test_parse_obj_negative_indices(tmp_path):
    """Negative OBJ indices are relative to end of vertex list."""
    content = (
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "v 0.0 0.0 1.0\n"
        "f -4 -3 -2\n"
    )
    path = tmp_path / "neg_idx.obj"
    path.write_text(content)
    vertices, faces = parse_obj(str(path))
    # -4 -> index 0, -3 -> index 1, -2 -> index 2
    assert faces[0] == (0, 1, 2), f"Expected (0,1,2), got {faces[0]}"


def test_face_count_equals_triangle_count(tmp_path):
    """Mix of triangles and quads: face count equals total triangles after fan."""
    content = (
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 1.0 1.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "v 0.5 0.5 1.0\n"
        "f 1 2 3\n"       # triangle -> 1 face
        "f 1 2 3 4\n"     # quad -> 2 faces
        "f 1 2 5\n"       # triangle -> 1 face
    )
    path = tmp_path / "mix.obj"
    path.write_text(content)
    vertices, faces = parse_obj(str(path))
    # 1 + 2 + 1 = 4 triangles total
    assert len(faces) == 4, f"Expected 4 triangles, got {len(faces)}"


def test_stl_coordinates_preserved(tmp_path):
    """Exact coordinate values must survive binary STL round-trip."""
    known_tris = [
        ((1.5, 2.7, 3.14), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ]
    path = tmp_path / "coords.stl"
    _write_binary_stl(str(path), known_tris)
    vertices, faces = parse_stl(str(path))
    # Find the vertex with x=1.5 (float32 precision)
    found = [v for v in vertices if abs(v[0] - 1.5) < 1e-5]
    assert len(found) == 1, "Vertex (1.5, ...) not found"
    assert abs(found[0][1] - 2.7) < 1e-5
    assert abs(found[0][2] - 3.14) < 1e-5


def test_obj_comments_and_blank_lines(tmp_path):
    """Comments and blank lines must be ignored."""
    content = (
        "# This is a comment\n"
        "\n"
        "v 0.0 0.0 0.0\n"
        "\n"
        "# Another comment\n"
        "v 1.0 0.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "\n"
        "f 1 2 3\n"
        "\n"
    )
    path = tmp_path / "comments.obj"
    path.write_text(content)
    vertices, faces = parse_obj(str(path))
    assert len(vertices) == 3
    assert len(faces) == 1
    assert faces[0] == (0, 1, 2)


def test_mesh_import_no_bpy_import():
    """mesh_import.py must be importable outside Blender (no top-level bpy import)."""
    import sys
    import importlib
    # If we get here without ImportError, the module loaded successfully outside Blender
    import blender_addon.mesh_import as m
    # bpy should NOT be a name in the module
    assert 'bpy' not in dir(m), "mesh_import.py must not import bpy at module level"
    # Verify stdlib imports are present
    assert 'struct' in dir(m) or hasattr(m, 'parse_stl'), "Module should expose parse_stl"
