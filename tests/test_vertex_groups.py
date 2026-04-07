"""
Tests for the pure-Python island detection logic in blender_addon/vertex_groups.py.
No Blender/bpy dependency -- only find_islands() is tested here.
"""
from blender_addon.vertex_groups import find_islands


def test_single_island():
    """A fully-connected triangle (3 vertices, all neighbours) should be 1 island."""
    adjacency = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1},
    }
    islands = find_islands(adjacency)
    assert len(islands) == 1
    assert islands[0] == {0, 1, 2}


def test_two_islands():
    """Two disconnected triangles (6 verts, no shared edges) should be 2 islands."""
    adjacency = {
        # Triangle 1: vertices 0,1,2
        0: {1, 2},
        1: {0, 2},
        2: {0, 1},
        # Triangle 2: vertices 3,4,5 -- no connection to triangle 1
        3: {4, 5},
        4: {3, 5},
        5: {3, 4},
    }
    islands = find_islands(adjacency)
    assert len(islands) == 2
    vertex_sets = [frozenset(isl) for isl in islands]
    assert frozenset({0, 1, 2}) in vertex_sets
    assert frozenset({3, 4, 5}) in vertex_sets


def test_island_count_matches():
    """Six disconnected triangles should yield exactly 6 islands."""
    adjacency: dict[int, set[int]] = {}
    for tri in range(6):
        a, b, c = tri * 3, tri * 3 + 1, tri * 3 + 2
        adjacency[a] = {b, c}
        adjacency[b] = {a, c}
        adjacency[c] = {a, b}
    islands = find_islands(adjacency)
    assert len(islands) == 6


def test_empty_graph():
    """An empty adjacency dict should return 0 islands."""
    islands = find_islands({})
    assert len(islands) == 0


def test_no_vertex_merging():
    """
    Two triangles sharing the same edge position but with separate vertex indices
    (as happens in an STL file with no vertex merging) should be 2 islands.

    In a raw STL each triangle has its own 3 vertices even if two triangles
    share an edge in world space.  Vertices are not merged by index, so the
    two triangles remain disconnected.
    """
    # Triangle 1: vertex indices 0, 1, 2 (e.g., occupies the shared edge at 0-1)
    # Triangle 2: vertex indices 3, 4, 5 (duplicates of 0, 1 positionally, but
    #             separate indices -- no edge in the adjacency graph)
    adjacency = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1},
        3: {4, 5},
        4: {3, 5},
        5: {3, 4},
    }
    islands = find_islands(adjacency)
    # Even though positions of vertices 0,1 and 3,4 might coincide in world
    # space, the graph has no edge between them -- they are separate islands.
    assert len(islands) == 2
