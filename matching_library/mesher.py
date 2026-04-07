"""
Gmsh tet mesher: STEP file to per-surface boundary data.

Provides:
  step_to_surface_data(step_path) -> dict
  load_existing_mesh(msh_path) -> dict

Both return the same structure:
  {
    'surface_tags': list[int],
    'surface_data': {tag: {'verts': np.ndarray, 'node_tags': np.ndarray, 'tris': np.ndarray}},
    'volume_tags': list[int],
    'tolerance': float,  -- auto-computed from boundary edge lengths
  }

Anti-patterns avoided:
  - Always calls occ.synchronize() before any model query.
  - Uses generate(3) (tet mesh), NOT generate(2) (surface tessellation).
  - Always calls gmsh.finalize() -- no leaked state between calls.
  - Uses gmsh.merge() (not importShapes) for loading existing .msh files.
"""

import math

import gmsh
import numpy as np

from matching_library.tolerance import compute_tolerance


def step_to_surface_data(step_path: str) -> dict:
    """
    Import a STEP file, generate a 3D tet mesh, and return per-surface boundary data.

    Parameters
    ----------
    step_path:
        Path to the input .step / .stp file.

    Returns
    -------
    dict with keys:
      'surface_tags': list[int]
      'surface_data': dict[int, {'verts', 'node_tags', 'tris'}]
      'volume_tags': list[int]
      'tolerance': float
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    _setup_and_mesh_step(step_path)
    result = _extract_surface_data()
    tol = compute_tolerance(result["surface_data"])
    result["tolerance"] = tol
    gmsh.finalize()
    return result


def load_existing_mesh(msh_path: str) -> dict:
    """
    Load an existing .msh file and return per-surface boundary data.

    Identical return type to step_to_surface_data() -- the matcher core
    does not know or care which mode was used.

    Parameters
    ----------
    msh_path:
        Path to an existing Gmsh .msh file.

    Returns
    -------
    dict with keys:
      'surface_tags': list[int]
      'surface_data': dict[int, {'verts', 'node_tags', 'tris'}]
      'volume_tags': list[int]
      'tolerance': float
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    _merge_mesh(msh_path)
    result = _extract_surface_data()
    tol = compute_tolerance(result["surface_data"])
    result["tolerance"] = tol
    gmsh.finalize()
    return result


def _setup_and_mesh_step(step_path: str) -> None:
    """
    Import a STEP file and generate a 3D tet mesh in the active Gmsh session.

    Does NOT call gmsh.initialize() or gmsh.finalize() -- operates within
    the caller's Gmsh session. Used by both step_to_surface_data() and
    pipeline functions that manage their own Gmsh lifecycle.

    Parameters
    ----------
    step_path:
        Path to the input .step / .stp file.
    """
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()  # CRITICAL: always before any model query

    # Auto mesh size from bounding box
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    mesh_size = max(min(diag / 50.0, 10.0), 1e-4)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)
    gmsh.model.mesh.generate(3)  # full tet mesh (NOT generate(2) -- that is for tessellation only)


def _merge_mesh(msh_path: str) -> None:
    """
    Load an existing .msh file in the active Gmsh session.

    Does NOT call gmsh.initialize() or gmsh.finalize(). Used by both
    load_existing_mesh() and pipeline functions that manage their own lifecycle.

    Parameters
    ----------
    msh_path:
        Path to an existing Gmsh .msh file.
    """
    gmsh.merge(msh_path)  # NOT importShapes -- merge() loads .msh files
    # No synchronize() needed after merge -- mesh is already finalized


def _extract_surface_data() -> dict:
    """
    Extract per-surface boundary mesh data from the active Gmsh model.

    Must be called while gmsh is initialized and meshed.

    Returns
    -------
    dict with keys:
      'surface_tags': list[int]
      'surface_data': dict[int, {'verts': np.ndarray(N,3),
                                  'node_tags': np.ndarray(N,),
                                  'tris': np.ndarray(M,3)}]
      'volume_tags': list[int]
    """
    surfaces = gmsh.model.getEntities(dim=2)
    volumes = gmsh.model.getEntities(dim=3)

    surface_data = {}
    for (dim, tag) in surfaces:
        # includeBoundary=True: include edge and corner nodes shared with other surfaces
        node_tags, coords, _ = gmsh.model.mesh.getNodes(
            dim=2, tag=tag, includeBoundary=True
        )
        verts = np.array(coords, dtype=np.float64).reshape(-1, 3)
        _, elem_ids, elem_nodes = gmsh.model.mesh.getElements(dim=2, tag=tag)
        tris = (
            elem_nodes[0].reshape(-1, 3).astype(np.int64)
            if elem_ids
            else np.array([], dtype=np.int64).reshape(0, 3)
        )
        surface_data[tag] = {
            "verts": verts,
            "node_tags": np.array(node_tags, dtype=np.int64),
            "tris": tris,
        }

    return {
        "surface_tags": [t for (_, t) in surfaces],
        "surface_data": surface_data,
        "volume_tags": [t for (_, t) in volumes],
    }
