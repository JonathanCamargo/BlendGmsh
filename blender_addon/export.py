"""
Pure-Python JSON export helper for BlendGmsh addon.

This module does NOT import bpy at module level so it is testable
without Blender. The functions receive Blender mesh data as arguments
(duck-typed — works with real bpy objects or test mocks).

Exports:
    generate_group_colors(n)       -- generate n visually distinct RGB tuples
    detect_bc_mode(obj)            -- detect BREP vs mesh mode from vertex groups
    build_bc_groups_dict(...)      -- build a dict conforming to bc_groups_v1.json schema
"""
import colorsys
import re
from collections import defaultdict


def generate_group_colors(n):
    """Generate n visually distinct RGB color tuples using HSV spacing.

    Returns list of (r, g, b) tuples with values in [0, 1].
    Uses evenly-spaced hues at fixed saturation and value so all colors
    are distinct even for large n.
    """
    return [colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.9) for i in range(n)]


def _extract_surface_tag(vg_name: str):
    """Extract the integer surface tag from a 'face_N' vertex group name.

    Returns the integer tag, or None if the name doesn't match.
    """
    m = re.match(r"^face_(\d+)$", vg_name)
    return int(m.group(1)) if m else None


def detect_bc_mode(obj):
    """Detect whether the object has BREP surface tags (STEP import) or not.

    Returns "brep" if any vertex group matches face_N pattern, "mesh" otherwise.
    """
    for vg in obj.vertex_groups:
        if _extract_surface_tag(vg.name) is not None:
            return "brep"
    return "mesh"


def build_bc_groups_dict(obj, groups, step_filepath, blender_version, units="meters",
                         triangles=None):
    """Build a bc_groups_v1-conforming dict from a mesh object and face groups.

    Auto-detects BREP vs mesh mode from vertex groups:
    - BREP mode (face_N groups present): exports surface_tags per group
    - Mesh mode (no face_N groups): exports vertex coordinates and face topology

    Parameters
    ----------
    obj : mesh object
        Must expose:
          - obj.data.vertices   (iterable of objects with .index and .co)
          - obj.data.polygons   (iterable of objects with .vertices)
          - obj.vertex_groups   (iterable of objects with .name and .weight(index))
          - obj.matrix_world    (supports @ operator for vertex transform)
    groups : iterable
        Iterable of group-like objects (each has a .name attribute).
        These provide the BC group names; the actual vertex assignments are
        read from obj.vertex_groups by matching name.
    step_filepath : str
        Path or filename of the source geometry file.
    blender_version : str
        Blender version string e.g. "4.2.0".
    units : str
        World coordinate units, one of "meters", "millimeters", "inches".
    triangles : list of (int, int, int), optional
        Mesh mode only: triangulated faces (vertex index triples) to use instead
        of obj.data.polygons. Must match the triangulation written to the STL
        handed to the mesher so that face centroids line up.

    Returns
    -------
    dict
        Conforms to schema/bc_groups_v1.json.
    """
    mode = detect_bc_mode(obj)

    if mode == "brep":
        groups_out = _build_brep_groups(obj, groups)
    else:
        groups_out = _build_mesh_groups(obj, groups, triangles)

    mesh = obj.data
    matrix_world = obj.matrix_world

    # Compute overall mesh bounding box from ALL vertices (world-space)
    all_world_coords = [list(matrix_world @ v.co) for v in mesh.vertices]
    if all_world_coords:
        xs = [c[0] for c in all_world_coords]
        ys = [c[1] for c in all_world_coords]
        zs = [c[2] for c in all_world_coords]
        bb_min = [min(xs), min(ys), min(zs)]
        bb_max = [max(xs), max(ys), max(zs)]
    else:
        bb_min = [0.0, 0.0, 0.0]
        bb_max = [0.0, 0.0, 0.0]

    return {
        "schema_version": 1,
        "source": "blendgmsh",
        "blender_version": blender_version,
        "mode": mode,
        "step_file": step_filepath,
        "units": units,
        "groups": groups_out,
        "mesh_stats": {
            "total_vertices": len(mesh.vertices),
            "total_faces": len(mesh.polygons),
            "bounding_box": {
                "min": bb_min,
                "max": bb_max,
            },
        },
    }


def _build_brep_groups(obj, groups):
    """Build BREP mode groups: each BC group -> list of surface tags."""
    mesh = obj.data
    vg_by_name = {vg.name: vg for vg in obj.vertex_groups}

    # Build vertex -> set of face_N tags (a boundary vertex belongs to multiple)
    vert_to_tags = defaultdict(set)
    for vg in obj.vertex_groups:
        tag = _extract_surface_tag(vg.name)
        if tag is None:
            continue
        for v in mesh.vertices:
            try:
                w = vg.weight(v.index)
                if w > 0.0:
                    vert_to_tags[v.index].add(tag)
            except RuntimeError:
                pass

    # Build polygon -> surface tag by intersecting each polygon's vertex tag sets.
    poly_to_tag = {}
    for poly in mesh.polygons:
        tag_sets = [vert_to_tags.get(vi, set()) for vi in poly.vertices]
        if not tag_sets:
            continue
        common = set.intersection(*tag_sets)
        if len(common) == 1:
            poly_to_tag[poly] = common.pop()
        elif common:
            poly_to_tag[poly] = min(common)

    groups_out = {}
    for group_item in groups:
        group_name = group_item.name
        vg = vg_by_name.get(group_name)
        if vg is None:
            continue

        assigned_indices = set()
        for v in mesh.vertices:
            try:
                w = vg.weight(v.index)
                if w > 0.0:
                    assigned_indices.add(v.index)
            except RuntimeError:
                pass

        if not assigned_indices:
            continue

        surface_tags = set()
        for poly, tag in poly_to_tag.items():
            if all(vi in assigned_indices for vi in poly.vertices):
                surface_tags.add(tag)

        if not surface_tags:
            continue

        groups_out[group_name] = {
            "surface_tags": sorted(surface_tags),
        }

    return groups_out


def _build_mesh_groups(obj, groups, triangles=None):
    """Build mesh mode groups: each BC group -> vertex coords + face topology."""
    mesh = obj.data
    matrix_world = obj.matrix_world
    vg_by_name = {vg.name: vg for vg in obj.vertex_groups}

    groups_out = {}
    for group_item in groups:
        group_name = group_item.name
        vg = vg_by_name.get(group_name)
        if vg is None:
            continue

        assigned_indices = set()
        for v in mesh.vertices:
            try:
                w = vg.weight(v.index)
                if w > 0.0:
                    assigned_indices.add(v.index)
            except RuntimeError:
                pass

        if not assigned_indices:
            continue

        # Build ordered vertex list and index remapping
        assigned_sorted = sorted(assigned_indices)
        world_verts = [list(matrix_world @ mesh.vertices[i].co) for i in assigned_sorted]
        index_map = {orig: local for local, orig in enumerate(assigned_sorted)}

        # Find faces where ALL polygon vertices are in this group
        face_list = (triangles if triangles is not None
                     else [poly.vertices for poly in mesh.polygons])
        group_faces = []
        for face in face_list:
            if all(vi in assigned_indices for vi in face):
                local_face = [index_map[vi] for vi in face]
                group_faces.append(local_face)

        if not world_verts:
            continue

        groups_out[group_name] = {
            "vertices": world_verts,
            "face_vertex_indices": group_faces,
            "vertex_count": len(world_verts),
            "face_count": len(group_faces),
        }

    return groups_out


def write_binary_stl(path, vertices, triangles):
    """Write a binary STL from world-space vertex coords and index triples."""
    import struct
    with open(path, "wb") as f:
        f.write(b"BlendGmsh native object".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            v0, v1, v2 = vertices[a], vertices[b], vertices[c]
            ux, uy, uz = (v1[i] - v0[i] for i in range(3))
            wx, wy, wz = (v2[i] - v0[i] for i in range(3))
            n = (uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx)
            length = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5 or 1.0
            f.write(struct.pack("<3f", *(x / length for x in n)))
            for v in (v0, v1, v2):
                f.write(struct.pack("<3f", *v))
            f.write(b"\0\0")
