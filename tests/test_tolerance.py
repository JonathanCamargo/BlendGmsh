"""
Unit tests for auto-tolerance computation from tet mesh boundary edge lengths.

Uses synthetic surface_data (no Gmsh dependency).
"""

import numpy as np
import pytest

from matching_library.tolerance import compute_tolerance


# ---------------------------------------------------------------------------
# Synthetic mesh helpers (duplicated from test_matcher to keep independence)
# ---------------------------------------------------------------------------

def _make_plane_surface(z_val: float, n_pts: int = 5, tag_offset: int = 0) -> dict:
    """Generate a grid of vertices on z=z_val with triangulation."""
    xs = np.linspace(0.0, 1.0, n_pts)
    ys = np.linspace(0.0, 1.0, n_pts)
    verts = []
    for x in xs:
        for y in ys:
            verts.append([x, y, float(z_val)])
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


def _make_unit_box_surface_data(n_pts: int = 5) -> dict:
    """Build synthetic surface_data for all 6 faces of a unit box [0,1]^3."""
    from tests.test_matcher import (
        _make_plane_surface_xy, _make_plane_surface_yz,
    )

    top = _make_plane_surface(z_val=1.0, n_pts=n_pts, tag_offset=0)
    bot = _make_plane_surface(z_val=0.0, n_pts=n_pts, tag_offset=100)
    x0 = _make_plane_surface_xy(x_val=0.0, n_pts=n_pts, tag_offset=200)
    x1 = _make_plane_surface_xy(x_val=1.0, n_pts=n_pts, tag_offset=300)
    y0 = _make_plane_surface_yz(y_val=0.0, n_pts=n_pts, tag_offset=400)
    y1 = _make_plane_surface_yz(y_val=1.0, n_pts=n_pts, tag_offset=500)

    return {1: top, 2: bot, 3: x0, 4: x1, 5: y0, 6: y1}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tolerance_unit_box():
    """
    compute_tolerance on synthetic unit box surface data gives ~0.25/5=0.05.

    For a 5-point grid (n_pts=5) on a unit face, edge lengths are 0.25 (along
    the grid) or ~0.35 (diagonal). Mean is around 0.3, so tolerance ~0.06.
    The requirement says range [0.02, 0.05] for a real tet mesh -- for the
    synthetic grid we just test it is in a reasonable range.

    The research-validated result is ~0.032 for the actual Gmsh tet mesh.
    For our synthetic 5-point grid, edge length is 1/(n_pts-1)=0.25, so
    tolerance = mean_edge/5 ~ 0.25/5 = 0.05. We verify it falls in [0.02, 0.10].
    """
    surface_data = _make_unit_box_surface_data(n_pts=5)
    tol = compute_tolerance(surface_data)

    assert isinstance(tol, float)
    assert 0.02 <= tol <= 0.10, (
        f"Expected tolerance in [0.02, 0.10], got {tol:.4f}"
    )


def test_tolerance_scaled_box():
    """
    10x scaled box gives ~10x larger tolerance.

    We scale all vertex coordinates by 10. Edge lengths scale by 10,
    so tolerance should scale by ~10.
    """
    surface_data = _make_unit_box_surface_data(n_pts=5)

    # Scale all coordinates by 10
    scaled_data = {
        tag: {
            "verts": d["verts"] * 10.0,
            "node_tags": d["node_tags"],
            "tris": d["tris"],
        }
        for tag, d in surface_data.items()
    }

    tol_unit = compute_tolerance(surface_data)
    tol_scaled = compute_tolerance(scaled_data)

    # Scaled tolerance should be approximately 10x larger (within 20% tolerance on the ratio)
    ratio = tol_scaled / tol_unit
    assert 8.0 <= ratio <= 12.0, (
        f"Expected ~10x scaling, got ratio={ratio:.2f} (tol_unit={tol_unit:.4f}, tol_scaled={tol_scaled:.4f})"
    )


def test_tolerance_empty_fallback():
    """Empty surface_data returns fallback value 1e-3."""
    tol = compute_tolerance({})
    assert tol == 1e-3, f"Expected 1e-3 fallback, got {tol}"
