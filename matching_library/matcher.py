"""
KDTree matcher: assigns Gmsh boundary surfaces to Blender face groups.

Provides:
  match_groups_to_surfaces(surface_data, groups, tolerance) -> dict
  match_surfaces_by_centroids(surface_data, group_centroids, tolerance) -> dict

Both matchers support **overlapping groups**: a single Gmsh surface can be
assigned to multiple BC groups simultaneously (e.g. body + force).  This is
valid in Gmsh -- a surface entity can belong to multiple physical groups.

Algorithm (vertex mode): For each Gmsh boundary triangle, each vertex votes
for ALL groups within tolerance.  Each group that receives at least
MIN_MATCH_FRACTION of facet votes claims the surface.

Algorithm (centroid mode): For each Gmsh boundary triangle, find all BC group
centroids within tolerance and vote for every matching group.  Any group with
a positive vote count claims the surface.
"""

from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree


MIN_MATCH_FRACTION = 0.5  # facets with < 50% votes for any group = UNMATCHED


def match_groups_to_surfaces(
    surface_data: dict,
    groups: dict,
    tolerance: float,
) -> dict:
    """
    Match each Gmsh surface to Blender face groups using KDTree voting.

    Supports overlapping groups: a surface can belong to multiple groups.

    Parameters
    ----------
    surface_data:
        Dict mapping surface_tag -> {'verts': np.ndarray(N,3),
                                      'node_tags': np.ndarray(N,),
                                      'tris': np.ndarray(M,3)}
        tris contains node_tag triples (NOT index triples).
    groups:
        Dict mapping group_name -> np.ndarray(K, 3) vertex coordinates.
    tolerance:
        Maximum distance (in mesh units) for a vertex to be considered
        "within" a group. Vertices beyond this distance don't vote.

    Returns
    -------
    dict
        {surface_tag: {
            'groups': {group_name: facet_count, ...},
            'total_facets': int,
            'unmatched_facets': int,
        }}
        A surface with an empty 'groups' dict matched no group.
    """
    _empty = lambda d: {"groups": {}, "total_facets": len(d["tris"]),
                         "unmatched_facets": len(d["tris"])}

    if not groups:
        return {tag: _empty(data) for tag, data in surface_data.items()}

    # Build one KDTree per group
    trees = {name: KDTree(verts) for name, verts in groups.items()}

    results = {}
    for tag, data in surface_data.items():
        local_map = {int(nt): i for i, nt in enumerate(data["node_tags"])}
        group_facet_votes = defaultdict(int)
        unmatched_facets = 0

        for tri in data["tris"]:
            # Collect ALL groups within tolerance for each vertex
            facet_groups = set()
            for nid in tri:
                if int(nid) not in local_map:
                    continue
                v = data["verts"][local_map[int(nid)]]
                for gname, tree in trees.items():
                    d, _ = tree.query(v)
                    if d <= tolerance:
                        facet_groups.add(gname)

            if facet_groups:
                for g in facet_groups:
                    group_facet_votes[g] += 1
            else:
                unmatched_facets += 1

        total_facets = len(data["tris"])
        # Apply MIN_MATCH_FRACTION threshold per group independently
        matched_groups = {
            g: count for g, count in group_facet_votes.items()
            if total_facets > 0 and count / total_facets >= MIN_MATCH_FRACTION
        }

        results[tag] = {
            "groups": matched_groups,
            "total_facets": total_facets,
            "unmatched_facets": unmatched_facets,
        }

    return results


def _compute_gmsh_centroids(data: dict) -> np.ndarray:
    """Compute triangle centroids for one Gmsh surface entity.

    Returns (M, 3) array of centroids, one per triangle.
    """
    local_map = {int(nt): i for i, nt in enumerate(data["node_tags"])}
    centroids = []
    for tri in data["tris"]:
        verts = [data["verts"][local_map[int(nid)]]
                 for nid in tri if int(nid) in local_map]
        if len(verts) == 3:
            centroids.append(np.mean(verts, axis=0))
    return np.array(centroids) if centroids else np.empty((0, 3))


def match_surfaces_by_centroids(
    surface_data: dict,
    group_centroids: dict,
    tolerance: float,
) -> dict:
    """
    Match Gmsh surfaces to BC groups by face centroid proximity.

    Supports overlapping groups: a surface can belong to multiple groups.
    Each triangle votes for ALL groups that have a centroid within tolerance.

    Parameters
    ----------
    surface_data:
        Dict mapping surface_tag -> {'verts', 'node_tags', 'tris'}
    group_centroids:
        Dict mapping group_name -> np.ndarray(K, 3) of face centroids.
    tolerance:
        Maximum centroid distance for a match.

    Returns
    -------
    dict
        Same format as match_groups_to_surfaces().
    """
    _empty = lambda d: {"groups": {}, "total_facets": len(d["tris"]),
                         "unmatched_facets": len(d["tris"])}

    if not group_centroids:
        return {tag: _empty(d) for tag, d in surface_data.items()}

    # Build single KDTree from all group centroids
    all_centroids = []
    centroid_labels = []
    for gname, cents in group_centroids.items():
        if len(cents) == 0:
            continue
        for c in cents:
            all_centroids.append(c)
            centroid_labels.append(gname)

    if not all_centroids:
        return {tag: _empty(d) for tag, d in surface_data.items()}

    centroid_labels = np.array(centroid_labels)
    tree = KDTree(np.array(all_centroids))

    results = {}
    for tag, data in surface_data.items():
        gmsh_cents = _compute_gmsh_centroids(data)
        total_facets = len(data["tris"])

        if len(gmsh_cents) == 0:
            results[tag] = _empty(data)
            continue

        # For each Gmsh centroid, find ALL BC centroids within tolerance
        ball_results = tree.query_ball_point(gmsh_cents, tolerance)

        group_votes = defaultdict(int)
        unmatched = 0
        for matches in ball_results:
            if not matches:
                unmatched += 1
                continue
            # Vote for every group that has a centroid within tolerance
            matched_groups = set(centroid_labels[i] for i in matches)
            for g in matched_groups:
                group_votes[g] += 1

        # Any positive centroid match assigns the surface — centroids are
        # unique per face (no shared-edge false positives).
        results[tag] = {
            "groups": dict(group_votes),
            "total_facets": total_facets,
            "unmatched_facets": unmatched,
        }

    return results
