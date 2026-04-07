"""
Unit tests for KDTree matcher core.

All tests use synthetic surface_data (no Gmsh dependency) to ensure
full testability in isolation.
"""

import numpy as np
import pytest

from matching_library.matcher import match_groups_to_surfaces, match_surfaces_by_centroids


# ---------------------------------------------------------------------------
# Synthetic mesh helpers
# ---------------------------------------------------------------------------

def _make_plane_surface(z_val: float, n_pts: int = 5) -> dict:
    """
    Generate a uniform grid of vertices on z=z_val and triangulate it.

    Returns dict with keys:
      'verts': np.ndarray(N, 3)
      'node_tags': np.ndarray(N,) int64 -- sequential from 1
      'tris': np.ndarray(M, 3) int64 -- node_tag triples (NOT index triples)
    """
    xs = np.linspace(0.0, 1.0, n_pts)
    ys = np.linspace(0.0, 1.0, n_pts)
    verts = []
    for x in xs:
        for y in ys:
            verts.append([x, y, float(z_val)])
    verts = np.array(verts, dtype=np.float64)
    N = len(verts)
    node_tags = np.arange(1, N + 1, dtype=np.int64)

    # Triangulate the grid: for each cell (i,j) create 2 triangles
    tris = []
    for i in range(n_pts - 1):
        for j in range(n_pts - 1):
            # indices into verts array
            a = i * n_pts + j
            b = i * n_pts + (j + 1)
            c = (i + 1) * n_pts + j
            d = (i + 1) * n_pts + (j + 1)
            # Convert to node_tags (1-based)
            tris.append([node_tags[a], node_tags[b], node_tags[c]])
            tris.append([node_tags[b], node_tags[d], node_tags[c]])

    tris = np.array(tris, dtype=np.int64)
    return {"verts": verts, "node_tags": node_tags, "tris": tris}


def _make_plane_surface_xy(x_val: float, n_pts: int = 5, tag_offset: int = 0) -> dict:
    """
    Generate a uniform grid of vertices on x=x_val and triangulate it.

    Returns same structure as _make_plane_surface.
    tag_offset shifts node_tags to avoid collisions when combining surfaces.
    """
    ys = np.linspace(0.0, 1.0, n_pts)
    zs = np.linspace(0.0, 1.0, n_pts)
    verts = []
    for y in ys:
        for z in zs:
            verts.append([float(x_val), y, z])
    verts = np.array(verts, dtype=np.float64)
    N = len(verts)
    node_tags = np.arange(1 + tag_offset, N + 1 + tag_offset, dtype=np.int64)

    tris = []
    for i in range(n_pts - 1):
        for j in range(n_pts - 1):
            a = i * n_pts + j
            b = i * n_pts + (j + 1)
            c = (i + 1) * n_pts + j
            d = (i + 1) * n_pts + (j + 1)
            tris.append([node_tags[a], node_tags[b], node_tags[c]])
            tris.append([node_tags[b], node_tags[d], node_tags[c]])

    tris = np.array(tris, dtype=np.int64)
    return {"verts": verts, "node_tags": node_tags, "tris": tris}


def _make_plane_surface_yz(y_val: float, n_pts: int = 5, tag_offset: int = 0) -> dict:
    """
    Generate a uniform grid of vertices on y=y_val and triangulate it.
    """
    xs = np.linspace(0.0, 1.0, n_pts)
    zs = np.linspace(0.0, 1.0, n_pts)
    verts = []
    for x in xs:
        for z in zs:
            verts.append([x, float(y_val), z])
    verts = np.array(verts, dtype=np.float64)
    N = len(verts)
    node_tags = np.arange(1 + tag_offset, N + 1 + tag_offset, dtype=np.int64)

    tris = []
    for i in range(n_pts - 1):
        for j in range(n_pts - 1):
            a = i * n_pts + j
            b = i * n_pts + (j + 1)
            c = (i + 1) * n_pts + j
            d = (i + 1) * n_pts + (j + 1)
            tris.append([node_tags[a], node_tags[b], node_tags[c]])
            tris.append([node_tags[b], node_tags[d], node_tags[c]])

    tris = np.array(tris, dtype=np.int64)
    return {"verts": verts, "node_tags": node_tags, "tris": tris}


def _surf_centroids(surf):
    """Compute triangle centroids from a synthetic surface dict."""
    local_map = {int(nt): i for i, nt in enumerate(surf["node_tags"])}
    cents = []
    for tri in surf["tris"]:
        vs = [surf["verts"][local_map[int(n)]] for n in tri]
        cents.append(np.mean(vs, axis=0))
    return np.array(cents)


# ---------------------------------------------------------------------------
# Vertex mode tests
# ---------------------------------------------------------------------------

def test_match_single_group_box():
    """Single group on top face (z=1); matcher assigns that surface to the group."""
    top_surf = _make_plane_surface(z_val=1.0, n_pts=5)
    surface_data = {1: top_surf}

    groups = {
        "top": np.array([[0.1, 0.1, 1.0], [0.5, 0.5, 1.0], [0.9, 0.9, 1.0]])
    }
    tolerance = 0.5  # generous for synthetic mesh

    result = match_groups_to_surfaces(surface_data, groups, tolerance)

    assert 1 in result
    assert "top" in result[1]["groups"]
    assert result[1]["total_facets"] > 0


def test_match_two_groups_box():
    """Top (z=1) and bottom (z=0) groups; each surface correctly matched."""
    top_surf = _make_plane_surface(z_val=1.0, n_pts=5)
    bot_surf = _make_plane_surface(z_val=0.0, n_pts=5)

    # Side surfaces (should be unmatched - no group covers them)
    side1 = _make_plane_surface_xy(x_val=0.0, n_pts=5, tag_offset=100)
    side2 = _make_plane_surface_xy(x_val=1.0, n_pts=5, tag_offset=200)
    side3 = _make_plane_surface_yz(y_val=0.0, n_pts=5, tag_offset=300)
    side4 = _make_plane_surface_yz(y_val=1.0, n_pts=5, tag_offset=400)

    surface_data = {
        1: top_surf,
        2: bot_surf,
        3: side1,
        4: side2,
        5: side3,
        6: side4,
    }

    groups = {
        "top": np.array([[0.25, 0.25, 1.0], [0.5, 0.5, 1.0], [0.75, 0.75, 1.0]]),
        "bottom": np.array([[0.25, 0.25, 0.0], [0.5, 0.5, 0.0], [0.75, 0.75, 0.0]]),
    }
    tolerance = 0.3

    result = match_groups_to_surfaces(surface_data, groups, tolerance)

    assert "top" in result[1]["groups"], f"Expected 'top', got {result[1]['groups']}"
    assert "bottom" in result[2]["groups"], f"Expected 'bottom', got {result[2]['groups']}"
    # Side faces should be unmatched
    assert result[3]["groups"] == {}
    assert result[4]["groups"] == {}
    assert result[5]["groups"] == {}
    assert result[6]["groups"] == {}


def test_no_edge_contamination():
    """Adjacent faces sharing an edge; each surface gets >90% correct facet assignment."""
    # Top face z=1 (shares edge at z=1, y=0 with front face)
    top_surf = _make_plane_surface(z_val=1.0, n_pts=7)

    # Front face y=0 (shares edge at z=1, y=0 with top face)
    front_surf = _make_plane_surface_yz(y_val=0.0, n_pts=7, tag_offset=1000)

    surface_data = {1: top_surf, 2: front_surf}

    groups = {
        "top_group": np.array([
            [0.2, 0.2, 1.0], [0.5, 0.5, 1.0], [0.8, 0.8, 1.0],
            [0.3, 0.7, 1.0], [0.7, 0.3, 1.0],
        ]),
        "front_group": np.array([
            [0.2, 0.0, 0.5], [0.5, 0.0, 0.3], [0.8, 0.0, 0.7],
            [0.3, 0.0, 0.8], [0.7, 0.0, 0.2],
        ]),
    }
    tolerance = 0.4

    result = match_groups_to_surfaces(surface_data, groups, tolerance)

    # Top surface should be assigned to top_group
    assert "top_group" in result[1]["groups"], f"Expected top_group, got {result[1]['groups']}"
    # Front surface should be assigned to front_group
    assert "front_group" in result[2]["groups"], f"Expected front_group, got {result[2]['groups']}"


def test_unmatched_surface_returns_none():
    """Surface with no group vertices nearby returns empty groups."""
    # Surface is a side face (x=0.5 plane), no group covers it
    side_surf = _make_plane_surface_xy(x_val=0.5, n_pts=5)
    surface_data = {1: side_surf}

    # Groups only cover z=1 top face -- far from x=0.5 side
    groups = {
        "top": np.array([[0.1, 0.1, 1.0], [0.9, 0.9, 1.0]])
    }
    tolerance = 0.05  # tight tolerance -- no match possible

    result = match_groups_to_surfaces(surface_data, groups, tolerance)

    assert result[1]["groups"] == {}
    assert result[1]["total_facets"] > 0


def test_empty_groups_dict():
    """Empty groups dict -> all surfaces unmatched (empty groups)."""
    top_surf = _make_plane_surface(z_val=1.0, n_pts=5)
    bot_surf = _make_plane_surface(z_val=0.0, n_pts=5)
    surface_data = {1: top_surf, 2: bot_surf}

    groups = {}  # no groups
    tolerance = 0.5

    result = match_groups_to_surfaces(surface_data, groups, tolerance)

    assert result[1]["groups"] == {}
    assert result[2]["groups"] == {}
    assert result[1]["total_facets"] > 0
    assert result[2]["total_facets"] > 0


# ---------------------------------------------------------------------------
# Overlapping group tests
# ---------------------------------------------------------------------------

def test_overlapping_groups_vertex_mode():
    """When 'body' overlaps 'force', both groups claim the shared surface."""
    top_surf = _make_plane_surface(z_val=1.0, n_pts=5)
    bot_surf = _make_plane_surface(z_val=0.0, n_pts=5)
    surface_data = {1: top_surf, 2: bot_surf}

    # "body" covers BOTH surfaces (like selecting all faces)
    body_verts = np.vstack([top_surf["verts"], bot_surf["verts"]])
    # "force" only covers bottom surface
    force_verts = bot_surf["verts"].copy()

    groups = {
        "body": body_verts,
        "force": force_verts,
    }
    tolerance = 0.3

    result = match_groups_to_surfaces(surface_data, groups, tolerance)

    # Top: only body
    assert "body" in result[1]["groups"]
    assert "force" not in result[1]["groups"]
    # Bottom: BOTH body and force (overlapping)
    assert "body" in result[2]["groups"]
    assert "force" in result[2]["groups"]


def test_overlapping_groups_centroid_mode():
    """When 'body' overlaps 'force', both groups claim the shared surface."""
    top_surf = _make_plane_surface(z_val=1.0, n_pts=5)
    bot_surf = _make_plane_surface(z_val=0.0, n_pts=5)
    surface_data = {1: top_surf, 2: bot_surf}

    top_cents = _surf_centroids(top_surf)
    bot_cents = _surf_centroids(bot_surf)

    # "body" covers BOTH surfaces
    body_centroids = np.vstack([top_cents, bot_cents])
    # "force" only covers bottom
    force_centroids = bot_cents.copy()

    group_centroids = {
        "body": body_centroids,
        "force": force_centroids,
    }
    tolerance = 0.3

    result = match_surfaces_by_centroids(surface_data, group_centroids, tolerance)

    # Top: only body
    assert "body" in result[1]["groups"]
    assert "force" not in result[1]["groups"]
    # Bottom: BOTH body and force
    assert "body" in result[2]["groups"]
    assert "force" in result[2]["groups"]


def test_three_overlapping_groups_centroid_mode():
    """Three-way overlap: body > fixed > force. Shared surfaces belong to
    all covering groups."""
    s1 = _make_plane_surface(z_val=0.0, n_pts=5)
    s2 = _make_plane_surface(z_val=0.5, n_pts=5)
    s3 = _make_plane_surface(z_val=1.0, n_pts=5)
    surface_data = {1: s1, 2: s2, 3: s3}

    c1 = _surf_centroids(s1)
    c2 = _surf_centroids(s2)
    c3 = _surf_centroids(s3)

    # body covers all 3, fixed covers s2+s3, force covers only s3
    group_centroids = {
        "body": np.vstack([c1, c2, c3]),
        "fixed": np.vstack([c2, c3]),
        "force": c3.copy(),
    }
    tolerance = 0.3

    result = match_surfaces_by_centroids(surface_data, group_centroids, tolerance)

    # s1: body only
    assert result[1]["groups"].keys() == {"body"}
    # s2: body + fixed
    assert result[2]["groups"].keys() == {"body", "fixed"}
    # s3: all three
    assert result[3]["groups"].keys() == {"body", "fixed", "force"}
