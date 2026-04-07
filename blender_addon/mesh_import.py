"""
STL and OBJ file parsers for BlendGmsh mesh import.

Pure Python (stdlib only) -- no bpy dependency so pytest can test without Blender.
Returns (vertices, faces) tuples compatible with mesh.from_pydata(vertices, [], faces).
"""

import os
import re
import struct


def detect_stl_format(filepath):
    """Return 'binary' or 'ascii' based on file size formula.

    Uses the unambiguous binary size formula: 84 + count * 50 bytes.
    Does NOT rely on 'solid' header bytes (many binary STLs start with 'solid').
    """
    size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        f.read(80)
        count_bytes = f.read(4)
    if len(count_bytes) < 4:
        return 'ascii'
    count = struct.unpack('<I', count_bytes)[0]
    return 'binary' if size == 84 + count * 50 else 'ascii'


def parse_stl(filepath):
    """Parse STL file (binary or ASCII). Returns (vertices, faces).

    vertices: list of (x, y, z) float tuples, deduplicated.
    faces: list of (i0, i1, i2) int tuples (0-based indices into vertices).
    """
    if detect_stl_format(filepath) == 'binary':
        return _parse_stl_binary(filepath)
    return _parse_stl_ascii(filepath)


def _parse_stl_binary(filepath):
    """Parse binary STL file."""
    coord_map = {}
    vertices = []
    faces = []
    with open(filepath, 'rb') as f:
        f.read(80)  # skip header
        count = struct.unpack('<I', f.read(4))[0]
        for _ in range(count):
            f.read(12)  # skip normal vector
            tri = []
            for _ in range(3):
                x, y, z = struct.unpack('<3f', f.read(12))
                key = (round(x, 6), round(y, 6), round(z, 6))
                if key not in coord_map:
                    coord_map[key] = len(vertices)
                    vertices.append((float(x), float(y), float(z)))
                tri.append(coord_map[key])
            f.read(2)  # skip attribute byte count
            faces.append(tuple(tri))
    return vertices, faces


def _parse_stl_ascii(filepath):
    """Parse ASCII STL file."""
    with open(filepath) as f:
        content = f.read()
    coord_map = {}
    vertices = []
    faces = []
    for match in re.finditer(
        r'facet normal.*?outer loop(.*?)endloop', content, re.DOTALL
    ):
        raw = re.findall(
            r'vertex\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)',
            match.group(1)
        )
        tri = []
        for sx, sy, sz in raw[:3]:
            x, y, z = float(sx), float(sy), float(sz)
            key = (round(x, 6), round(y, 6), round(z, 6))
            if key not in coord_map:
                coord_map[key] = len(vertices)
                vertices.append((x, y, z))
            tri.append(coord_map[key])
        if len(tri) == 3:
            faces.append(tuple(tri))
    return vertices, faces


def parse_obj(filepath):
    """Parse OBJ file. Returns (vertices, faces).

    Handles v/vt/vn notation, negative indices, quads (fan triangulation).
    vertices: list of (x, y, z) float tuples.
    faces: list of (i0, i1, i2) int tuples (0-based).
    """
    vertices = []
    faces = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('v '):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith('f '):
                parts = line.split()[1:]
                raw = []
                for part in parts:
                    idx = int(part.split('/')[0])
                    idx = (len(vertices) + idx) if idx < 0 else (idx - 1)
                    raw.append(idx)
                # Fan triangulation for quads and n-gons
                for i in range(1, len(raw) - 1):
                    faces.append((raw[0], raw[i], raw[i + 1]))
    return vertices, faces
