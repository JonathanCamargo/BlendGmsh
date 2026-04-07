"""
Auto-tolerance computation from tet mesh boundary edge lengths.

Provides:
  compute_tolerance(surface_data) -> float

Returns avg_boundary_edge_length / 5.
Rationale: chord error on a well-tessellated surface is well under mesh_size/10.
Dividing by 5 (not 10) gives a generous margin while keeping it tight enough
to avoid false matches on adjacent surfaces.
"""

import numpy as np


def compute_tolerance(surface_data: dict) -> float:
    """
    Auto-compute matching tolerance from actual tet mesh boundary edge lengths.

    Parameters
    ----------
    surface_data:
        Dict mapping surface_tag -> {'verts': np.ndarray(N,3),
                                      'node_tags': np.ndarray(N,),
                                      'tris': np.ndarray(M,3)}

    Returns
    -------
    float
        avg_boundary_edge_length / 5.0
        Falls back to 1e-3 if no edge lengths found (empty mesh).
    """
    edge_lengths = []
    SAMPLE_SIZE = 200  # sample first N triangles per surface for speed

    for tag, data in surface_data.items():
        local_map = {int(nt): i for i, nt in enumerate(data["node_tags"])}
        for tri in data["tris"][:SAMPLE_SIZE]:
            try:
                v = [data["verts"][local_map[int(n)]] for n in tri]
                edge_lengths += [
                    np.linalg.norm(v[1] - v[0]),
                    np.linalg.norm(v[2] - v[1]),
                    np.linalg.norm(v[0] - v[2]),
                ]
            except KeyError:
                pass

    if not edge_lengths:
        # Fallback: should not happen on a valid mesh
        return 1e-3
    return float(np.mean(edge_lengths)) / 5.0
