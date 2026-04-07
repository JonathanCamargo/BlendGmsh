"""
Vertex group assignment for the BlendGmsh addon.

Contains two layers:
  1. find_islands() -- pure Python BFS, no bpy dependency, fully testable.
  2. create_vertex_groups_per_island() -- Blender wrapper that assigns
     vertex groups using centroid matching (preferred) or BFS fallback.
  3. assign_by_centroids() -- pure Python centroid matching, fully testable.
"""
from __future__ import annotations

from collections import defaultdict


def find_islands(adjacency: dict[int, set[int]]) -> list[set[int]]:
    """
    BFS flood-fill to find connected components in a vertex adjacency graph.

    adjacency: {vertex_index: set of neighbor vertex indices}
    Returns: list of sets, each set is one island's vertex indices.
    Pure Python -- no bpy dependency.
    """
    visited: set[int] = set()
    islands: list[set[int]] = []

    for start in adjacency:
        if start in visited:
            continue
        # BFS from this unvisited vertex
        island: set[int] = set()
        queue: list[int] = [start]
        while queue:
            v = queue.pop()
            if v in island:
                continue
            island.add(v)
            visited.add(v)
            for neighbor in adjacency.get(v, set()):
                if neighbor not in island:
                    queue.append(neighbor)
        islands.append(island)

    return islands


def _centroid_key(cx, cy, cz, precision=4):
    """Round centroid coordinates to create a hashable lookup key.

    Precision=4 absorbs the float64->STL (:.6e format) roundtrip error
    while remaining unique for distinct mesh triangles.
    """
    return (round(cx, precision), round(cy, precision), round(cz, precision))


def assign_by_centroids(vertices, polygons, face_centroids):
    """Match polygons to BREP surfaces by centroid coordinates.

    Pure Python -- no bpy dependency, fully testable.

    Parameters
    ----------
    vertices : list-like
        Each element has a .co attribute (or is indexable) giving [x, y, z].
    polygons : list-like
        Each element has a .vertices attribute giving vertex indices.
    face_centroids : dict
        {surface_tag: [[cx, cy, cz], ...]} from tessellation.

    Returns
    -------
    dict[int, set[int]]
        {surface_tag: set of vertex indices} for vertex group creation.
    """
    # Build lookup: rounded centroid -> surface tag
    centroid_to_tag = {}
    for tag, centroids in face_centroids.items():
        for c in centroids:
            key = _centroid_key(c[0], c[1], c[2])
            centroid_to_tag[key] = tag

    # Match each polygon to its surface by computing its centroid
    tag_to_verts = defaultdict(set)
    for poly in polygons:
        poly_verts = list(poly.vertices)
        n = len(poly_verts)
        if n == 0:
            continue
        cx = sum(vertices[vi].co[0] for vi in poly_verts) / n
        cy = sum(vertices[vi].co[1] for vi in poly_verts) / n
        cz = sum(vertices[vi].co[2] for vi in poly_verts) / n

        key = _centroid_key(cx, cy, cz)
        tag = centroid_to_tag.get(key)
        if tag is not None:
            for vi in poly_verts:
                tag_to_verts[tag].add(vi)

    return dict(tag_to_verts)


def create_vertex_groups_per_island(obj, ordered_tags: list[int],
                                    tri_counts: list[int] = None,
                                    face_centroids: dict = None) -> int:
    """
    Assign one vertex group per BREP surface.

    Preferred path: centroid matching (order-independent, robust to
    Blender STL importer polygon reordering).

    Fallback: BFS island detection (legacy).

    obj: bpy.types.Object with mesh data
    ordered_tags: Gmsh surface tags in STL solid order.
    tri_counts: (unused, kept for API compat)
    face_centroids: {tag: [[cx,cy,cz], ...]} from tessellation.
    Returns: number of surface groups created.

    CRITICAL: Must be in Object Mode before calling vertex_group.add().
    """
    import bpy  # type: ignore  # available inside Blender runtime

    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = obj.data

    if face_centroids is not None:
        tag_to_verts = assign_by_centroids(mesh.vertices, mesh.polygons, face_centroids)
        groups_created = 0
        for tag in sorted(tag_to_verts.keys()):
            vg = obj.vertex_groups.new(name=f"face_{tag}")
            vg.add(list(tag_to_verts[tag]), 1.0, 'REPLACE')
            groups_created += 1
        return groups_created

    # Legacy fallback: BFS island detection
    import bmesh as _bmesh  # type: ignore
    bm = _bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    adjacency: dict[int, set[int]] = {v.index: set() for v in bm.verts}
    for edge in bm.edges:
        a = edge.verts[0].index
        b = edge.verts[1].index
        adjacency[a].add(b)
        adjacency[b].add(a)
    bm.free()
    islands = find_islands(adjacency)
    for i, vert_indices in enumerate(islands):
        tag = ordered_tags[i] if i < len(ordered_tags) else i
        vg = obj.vertex_groups.new(name=f"face_{tag}")
        vg.add(list(vert_indices), 1.0, 'REPLACE')
    return len(islands)
