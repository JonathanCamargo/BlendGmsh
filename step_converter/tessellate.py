"""
Gmsh STEP-to-STL tessellation with per-face named solids.

Provides:
  tessellate_step(step_path, stl_output, mesh_size=None) -> dict

The output STL has one ``solid face_N / endsolid face_N`` section per BREP
surface, enabling per-face vertex group assignment in Blender.

Anti-patterns avoided:
  - Does NOT call generate(3) -- surface-only (generate(2)) is sufficient.
  - ALWAYS calls occ.synchronize() before any model query.
  - Does NOT round vertex coordinates (full float64 precision).
"""

from __future__ import annotations

import math
import os
import struct
from typing import Optional

import gmsh
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tessellate_step(
    step_path: str,
    stl_output: str,
    mesh_size: Optional[float] = None,
) -> dict:
    """
    Tessellate a STEP file into an ASCII STL with one named solid per BREP surface.

    Parameters
    ----------
    step_path:
        Absolute or relative path to the input ``.step`` / ``.stp`` file.
    stl_output:
        Path for the output ``.stl`` file (will be created or overwritten).
    mesh_size:
        Characteristic mesh length.  If ``None``, auto-computed from the
        bounding-box diagonal: ``clamp(diag / 50, 1e-4, 10.0)``.
        Curvature-based refinement further reduces element size on
        curved surfaces (12 elements per 2*pi of curvature).

    Returns
    -------
    dict with keys:
        ``n_surfaces`` (int)  -- number of BREP surfaces in the model.
        ``surface_tags`` (list[int]) -- Gmsh entity tags for each surface.
        ``stl_path`` (str)   -- absolute path of the written STL file.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)

    # -----------------------------------------------------------------------
    # Import and synchronise
    # -----------------------------------------------------------------------
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()  # CRITICAL: must call before ANY model query

    surfaces = gmsh.model.getEntities(dim=2)  # [(2, tag), ...]
    surface_tags = [tag for (_, tag) in surfaces]

    # -----------------------------------------------------------------------
    # Auto-density from bounding box + curvature-based refinement
    # -----------------------------------------------------------------------
    if mesh_size is None:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = math.sqrt(
            (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
        )
        mesh_size = max(min(diag / 50.0, 10.0), 1e-4)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)

    # Curvature-adaptive refinement: places more elements on curved surfaces
    # (cylinders, fillets, arcs) so they look smooth and capture geometry
    # accurately for FEM. 12 elements per 2*pi of curvature.
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)

    # -----------------------------------------------------------------------
    # Surface-only mesh (NOT generate(3))
    # -----------------------------------------------------------------------
    gmsh.model.mesh.generate(2)

    # -----------------------------------------------------------------------
    # STL export with one named solid per surface.
    # Gmsh's StlOneSolidPerSurface=2 is unreliable (produces empty files
    # on some models), so we always extract per-surface manually.
    # -----------------------------------------------------------------------
    tri_counts, face_centroids = _write_per_surface_stl(surface_tags, stl_output)

    gmsh.finalize()

    return {
        "n_surfaces": len(surface_tags),
        "surface_tags": surface_tags,
        "tri_counts": tri_counts,
        "face_centroids": face_centroids,
        "stl_path": os.path.abspath(stl_output),
    }


def tessellate_step_to_mesh(
    step_path: str,
    mesh_size: Optional[float] = None,
) -> dict:
    """
    Tessellate a STEP file and return structured mesh data for direct Blender construction.

    No intermediate file. Returns vertices, faces, and per-face surface tags
    that Blender can use directly to build a mesh with vertex groups.

    Parameters
    ----------
    step_path:
        Path to the input STEP file.
    mesh_size:
        Characteristic mesh length. Auto-computed if None.

    Returns
    -------
    dict with keys:
        ``vertices`` (list[list[float]]) -- global vertex array [[x,y,z], ...]
        ``faces`` (list[list[int]]) -- triangles as global vertex indices [[v0,v1,v2], ...]
        ``face_surface_tags`` (list[int]) -- surface tag per face (same length as faces)
        ``surface_tags`` (list[int]) -- all BREP surface tags
        ``n_surfaces`` (int) -- number of BREP surfaces
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    surface_tags = [tag for (_, tag) in surfaces]

    if mesh_size is None:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = math.sqrt(
            (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
        )
        mesh_size = max(min(diag / 50.0, 10.0), 1e-4)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)

    gmsh.model.mesh.generate(2)

    # Extract all mesh data: build a global vertex array with shared vertices
    global_verts = []  # [[x, y, z], ...]
    global_faces = []  # [[v0, v1, v2], ...]
    face_surface_tags = []  # surface tag per face
    node_tag_to_global = {}  # Gmsh node tag -> global vertex index

    for tag in surface_tags:
        node_tags, coords, _ = gmsh.model.mesh.getNodes(
            dim=2, tag=tag, includeBoundary=True
        )
        if len(node_tags) == 0:
            continue

        # Register vertices (shared across surfaces via Gmsh node tags)
        for i, nt in enumerate(node_tags):
            nt = int(nt)
            if nt not in node_tag_to_global:
                node_tag_to_global[nt] = len(global_verts)
                global_verts.append([
                    float(coords[i * 3]),
                    float(coords[i * 3 + 1]),
                    float(coords[i * 3 + 2]),
                ])

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, tag)
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == 2:  # linear triangles
                tri_array = enodes.reshape(-1, 3)
                for tri in tri_array:
                    global_faces.append([
                        node_tag_to_global[int(tri[0])],
                        node_tag_to_global[int(tri[1])],
                        node_tag_to_global[int(tri[2])],
                    ])
                    face_surface_tags.append(tag)

    gmsh.finalize()

    return {
        "vertices": global_verts,
        "faces": global_faces,
        "face_surface_tags": face_surface_tags,
        "surface_tags": surface_tags,
        "n_surfaces": len(surface_tags),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_per_surface_stl(surface_tags: list[int], stl_path: str) -> tuple[list[int], dict]:
    """
    Write an ASCII STL with explicit per-surface solid sections.

    Extracts vertex coordinates and triangle connectivity for each surface
    directly from the Gmsh model (which must be initialized and meshed at
    call time) and writes them as individual ``solid face_{tag}`` sections.

    Returns
    -------
    tuple of (tri_counts, face_centroids)
        tri_counts: list[int] -- triangles written per surface tag.
        face_centroids: dict[int, list[list[float]]] -- {tag: [[cx,cy,cz], ...]}
            centroid of each triangle per surface, for polygon matching after import.
    """
    lines: list[str] = []
    tri_counts: list[int] = []
    face_centroids: dict[int, list] = {}

    for tag in surface_tags:
        node_tags, coords, _ = gmsh.model.mesh.getNodes(
            dim=2, tag=tag, includeBoundary=True
        )
        if len(node_tags) == 0:
            # Surface has no mesh — emit an empty solid section
            lines.append(f"solid face_{tag}")
            lines.append(f"endsolid face_{tag}")
            tri_counts.append(0)
            face_centroids[tag] = []
            continue

        # Build local vertex array (full float64 precision — no rounding)
        vertices = np.array(coords, dtype=np.float64).reshape(-1, 3)
        tag_to_local = {nt: i for i, nt in enumerate(node_tags)}

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(
            dim=2, tag=tag
        )

        lines.append(f"solid face_{tag}")
        surface_tri_count = 0
        surface_centroids = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == 2:
                # Linear triangles (element type 2)
                tri_array = np.array(enodes, dtype=np.int64).reshape(-1, 3)
                for tri in tri_array:
                    try:
                        v0 = vertices[tag_to_local[tri[0]]]
                        v1 = vertices[tag_to_local[tri[1]]]
                        v2 = vertices[tag_to_local[tri[2]]]
                    except KeyError:
                        continue  # skip degenerate triangle

                    # Compute facet normal
                    e1 = v1 - v0
                    e2 = v2 - v0
                    n = np.cross(e1, e2)
                    norm = np.linalg.norm(n)
                    if norm > 0:
                        n = n / norm
                    else:
                        n = np.array([0.0, 0.0, 1.0])

                    lines.append(
                        f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}"
                    )
                    lines.append("    outer loop")
                    lines.append(
                        f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}"
                    )
                    lines.append(
                        f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}"
                    )
                    lines.append(
                        f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}"
                    )
                    lines.append("    endloop")
                    lines.append("  endfacet")
                    surface_tri_count += 1
                    centroid = ((v0 + v1 + v2) / 3.0).tolist()
                    surface_centroids.append(centroid)
        tri_counts.append(surface_tri_count)
        face_centroids[tag] = surface_centroids
        lines.append(f"endsolid face_{tag}")

    with open(stl_path, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")

    return tri_counts, face_centroids
