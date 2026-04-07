"""
Physical group tagger: assign match results to Gmsh physical groups and write .msh v4.

Provides:
  tag_and_write(match_results, volume_tags, output_msh) -> None

IMPORTANT: This module operates within the caller's Gmsh session.
It does NOT call gmsh.initialize() or gmsh.finalize() -- the caller owns
the Gmsh lifecycle. Use from within a pipeline function that manages init/finalize.

Critical: Gmsh .msh v4 only writes elements belonging to a physical group.
Surfaces not in any group have their boundary triangles silently dropped.
Always add an '_untagged' group for unmatched surfaces.
"""

import gmsh


def tag_and_write(
    match_results: dict,
    volume_tags: list,
    output_msh: str,
) -> None:
    """
    Create named physical groups from match results and write .msh v4.

    Supports overlapping groups: a surface can belong to multiple physical
    groups simultaneously (e.g. body + force).

    Named groups go to user-specified names.
    Unmatched surfaces (empty groups) go to '_untagged' group (preserves mesh completeness).
    Volume tags get 'domain' group (required for FEM solvers).

    Parameters
    ----------
    match_results:
        Dict from matcher:
        {surface_tag: {'groups': {name: facet_count, ...},
                       'total_facets': int, 'unmatched_facets': int}}
    volume_tags:
        List of volume entity tags from the active Gmsh model.
    output_msh:
        Path for the output .msh file.
    """
    # Invert: build group_name -> list[surface_tag] and untagged list
    group_to_surfaces: dict = {}
    untagged_surfaces: list = []

    for surf_tag, result in match_results.items():
        groups = result["groups"]
        if groups:
            for grp in groups:
                group_to_surfaces.setdefault(grp, []).append(surf_tag)
        else:
            untagged_surfaces.append(surf_tag)

    # Create named physical groups for matched surfaces
    for gname, stags in group_to_surfaces.items():
        gmsh.model.addPhysicalGroup(2, stags, name=gname)

    # CRITICAL: _untagged group preserves all boundary elements in .msh v4 output.
    # Without this, Gmsh silently drops triangular elements for untagged surfaces.
    if untagged_surfaces:
        gmsh.model.addPhysicalGroup(2, untagged_surfaces, name="_untagged")

    # Volume group -- required for FEM solvers to identify the mesh domain
    if volume_tags:
        gmsh.model.addPhysicalGroup(3, volume_tags, name="domain")

    # Write .msh v4.1 format
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(output_msh)
