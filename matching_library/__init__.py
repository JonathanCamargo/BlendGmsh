"""Matching library: boundary condition assignment for FEM meshes.

Supports two modes:
  - BREP mode: physical groups assigned by BREP surface tag before meshing
  - Mesh mode: KDTree coordinate matching of Blender vertex selections post-meshing

Public API:
  - run_full_pipeline(bc_groups_json, step_file, output_msh) -> CoverageReport
  - tag_existing_mesh(bc_groups_json, input_msh, output_msh) -> CoverageReport
"""

import json
import math
from pathlib import Path

import gmsh
import jsonschema
import numpy as np

from matching_library.coverage import CoverageReport, build_coverage_report
from matching_library.debug import inspect_msh, inspect_bc_groups, visualize_bc_groups
from matching_library.matcher import match_groups_to_surfaces, match_surfaces_by_centroids
from matching_library.mesher import _extract_surface_data, _merge_mesh
from matching_library.tagger import tag_and_write
from matching_library.tolerance import compute_tolerance


__all__ = [
    "run_full_pipeline",
    "tag_existing_mesh",
    "inspect_msh",
    "inspect_bc_groups",
    "visualize_bc_groups",
]


# Load the JSON schema once at import time
_SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "bc_groups_v1.json"
_SCHEMA: dict = json.loads(_SCHEMA_PATH.read_text())


def _configure_mesh_sizing(curvature: bool = True) -> None:
    """Set mesh sizing options from model bounding box.

    curvature=True adds curvature refinement (appropriate for CAD geometry,
    not for discrete STL/OBJ meshes).
    """
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diag = math.sqrt(
        (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
    )
    mesh_size = max(min(diag / 50.0, 10.0), 1e-4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)
    if curvature:
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)


def _assign_brep_physical_groups(group_tags: dict) -> set:
    """Validate surface tags and create physical groups for BREP mode.

    Creates named physical groups for each BC group, an '_untagged' group
    for remaining surfaces, and a 'domain' group for volumes.

    Returns the set of untagged surface tags.
    """
    surfaces = gmsh.model.getEntities(dim=2)
    model_surface_tags = {tag for (_, tag) in surfaces}

    for gname, tags in group_tags.items():
        missing = set(tags) - model_surface_tags
        if missing:
            raise RuntimeError(
                f"Group '{gname}' references surface tags {sorted(missing)} "
                f"not found in the model."
            )

    tagged_surfaces = set()
    for gname, tags in group_tags.items():
        gmsh.model.addPhysicalGroup(2, tags, name=gname)
        tagged_surfaces.update(tags)

    untagged = model_surface_tags - tagged_surfaces
    if untagged:
        gmsh.model.addPhysicalGroup(2, sorted(untagged), name="_untagged")

    volumes = gmsh.model.getEntities(dim=3)
    if volumes:
        vol_tags = [t for (_, t) in volumes]
        gmsh.model.addPhysicalGroup(3, vol_tags, name="domain")

    return untagged


def _load_and_validate(bc_json_path: str) -> dict:
    """Load and validate a bc_groups JSON file against the schema.

    Returns the full parsed JSON dict.
    """
    with open(bc_json_path, "r") as f:
        data = json.load(f)

    jsonschema.validate(instance=data, schema=_SCHEMA)
    return data


def _detect_mode(data: dict) -> str:
    """Detect mode from JSON data. Returns 'brep' or 'mesh'."""
    # Explicit mode field takes precedence
    if "mode" in data:
        return data["mode"]
    # Auto-detect from group contents
    for group in data["groups"].values():
        if "surface_tags" in group:
            return "brep"
        if "vertices" in group:
            return "mesh"
    return "brep"  # default for empty groups


def run_full_pipeline(bc_json_path: str, step_path: str, output_msh: str) -> CoverageReport:
    """
    Full pipeline: STEP file + bc_groups JSON -> tagged .msh v4 with physical groups.

    Detects mode from JSON:
    - BREP mode: assigns physical groups to BREP surfaces before meshing.
    - Mesh mode: meshes first, then matches Blender vertices to boundary
      facets via KDTree and assigns physical groups post-meshing.

    Parameters
    ----------
    bc_json_path:
        Path to bc_groups JSON file (validated against schema).
    step_path:
        Path to the STEP file to mesh.
    output_msh:
        Path for the output .msh file.

    Returns
    -------
    CoverageReport
        Per-group match/assignment statistics.
    """
    data = _load_and_validate(bc_json_path)
    mode = _detect_mode(data)

    if mode == "brep":
        return _run_brep_pipeline(data, step_path, output_msh)
    else:
        return _run_mesh_pipeline(data, step_path, output_msh)


def tag_existing_mesh(
    bc_json_path: str, input_msh: str, output_msh: str
) -> CoverageReport:
    """
    Tagging-only mode: existing .msh + bc_groups JSON -> re-tagged .msh v4.

    Detects mode from JSON:
    - BREP mode: assigns physical groups by surface tag directly.
    - Mesh mode: KDTree matches Blender vertex coordinates to mesh boundary facets.

    Parameters
    ----------
    bc_json_path:
        Path to bc_groups JSON file (validated against schema).
    input_msh:
        Path to an existing .msh file to re-tag.
    output_msh:
        Path for the output .msh file.

    Returns
    -------
    CoverageReport
        Per-group match/assignment statistics.
    """
    data = _load_and_validate(bc_json_path)
    mode = _detect_mode(data)

    if mode == "brep":
        return _tag_existing_brep(data, input_msh, output_msh)
    else:
        return _tag_existing_mesh_mode(data, input_msh, output_msh)


# ---------------------------------------------------------------------------
# BREP mode implementation
# ---------------------------------------------------------------------------

def _extract_brep_groups(data: dict) -> dict:
    """Extract {group_name: [surface_tags]} from BREP mode JSON."""
    return {
        name: group["surface_tags"]
        for name, group in data["groups"].items()
    }


def _run_brep_pipeline(data: dict, step_path: str, output_msh: str) -> CoverageReport:
    """BREP pipeline: assign physical groups before meshing."""
    group_tags = _extract_brep_groups(data)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()

        untagged = _assign_brep_physical_groups(group_tags)
        _configure_mesh_sizing()
        gmsh.model.mesh.generate(3)

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(output_msh)

        report = _build_brep_report(group_tags, untagged)
        report.print_report()

    finally:
        gmsh.finalize()

    return report


def _tag_existing_brep(data: dict, input_msh: str, output_msh: str) -> CoverageReport:
    """BREP tagging on an existing mesh."""
    group_tags = _extract_brep_groups(data)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        gmsh.merge(input_msh)

        untagged = _assign_brep_physical_groups(group_tags)

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(output_msh)

        report = _build_brep_report(group_tags, untagged)
        report.print_report()

    finally:
        gmsh.finalize()

    return report


def _build_brep_report(group_tags: dict, untagged: set) -> CoverageReport:
    """Build a CoverageReport from BREP tag assignment."""
    group_stats = {}
    for gname, tags in group_tags.items():
        group_stats[gname] = {
            "surfaces": tags,
            "matched_facets": len(tags),
            "total_facets": len(tags),
        }

    return CoverageReport(
        group_stats=group_stats,
        unmatched_surfaces=sorted(untagged),
        total_boundary_facets=sum(len(t) for t in group_tags.values()) + len(untagged),
    )


# ---------------------------------------------------------------------------
# Mesh mode implementation (KDTree matching)
# ---------------------------------------------------------------------------

def _extract_mesh_groups(data: dict) -> dict:
    """Extract {group_name: np.ndarray(K, 3)} vertex coords from mesh mode JSON."""
    return {
        name: np.array(group["vertices"], dtype=np.float64)
        for name, group in data["groups"].items()
    }


def _extract_group_centroids(data: dict) -> dict:
    """Extract {group_name: np.ndarray(K, 3)} face centroids from mesh mode JSON.

    Computes the centroid of each face from vertices + face_vertex_indices.
    Used by the STL/OBJ pipeline for centroid-based matching which avoids
    edge-vertex bleeding.
    """
    result = {}
    for name, group in data["groups"].items():
        verts = np.array(group["vertices"], dtype=np.float64)
        faces = group.get("face_vertex_indices", [])
        if not faces:
            result[name] = np.empty((0, 3))
            continue
        centroids = [np.mean(verts[face], axis=0) for face in faces]
        result[name] = np.array(centroids, dtype=np.float64)
    return result


def _run_mesh_pipeline(data: dict, step_path: str, output_msh: str) -> CoverageReport:
    """Mesh pipeline: KDTree match Blender selections, then mesh with physical groups.

    Dispatches to the appropriate sub-pipeline based on input file type:
    - STEP: mesh first, then KDTree match post-mesh (vertices are on CAD surfaces).
    - STL/OBJ: KDTree match against original surface mesh to find surface entity
      assignments, assign physical groups, THEN remesh at FEM resolution.
      Physical groups survive remeshing because they're on geometry entities.
    """
    groups = _extract_mesh_groups(data)
    ext = Path(step_path).suffix.lower()

    if ext in ('.stl', '.obj'):
        group_centroids = _extract_group_centroids(data)
        return _run_surface_mesh_pipeline(group_centroids, step_path, output_msh)
    else:
        return _run_step_mesh_pipeline(groups, step_path, output_msh)


def _run_step_mesh_pipeline(
    groups: dict, step_path: str, output_msh: str,
) -> CoverageReport:
    """STEP mesh-mode: import CAD, mesh, then KDTree match post-mesh."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()

        _configure_mesh_sizing()
        gmsh.model.mesh.generate(3)

        surface_result = _extract_surface_data()
        tol = compute_tolerance(surface_result["surface_data"])
        match_results = match_groups_to_surfaces(
            surface_result["surface_data"], groups, tol
        )

        report = build_coverage_report(match_results, list(groups.keys()))
        zeros = report.check_zero_coverage_groups(list(groups.keys()))
        if zeros:
            raise RuntimeError(
                f"Groups with zero matched surfaces: {zeros}. "
                "Check that group vertex coordinates lie on the geometry surfaces."
            )

        tag_and_write(match_results, surface_result["volume_tags"], output_msh)
        report.print_report()

    finally:
        gmsh.finalize()

    return report


def _run_surface_mesh_pipeline(
    group_centroids: dict, mesh_path: str, output_msh: str,
) -> CoverageReport:
    """STL/OBJ pipeline: import, match BCs on original STL, split entities, then tet-mesh.

    1. Merge STL/OBJ, classify into discrete surface entities.
    2. Match BC face centroids to original STL triangle centroids (pre-mesh).
    3. Split surface entities by group membership so each group gets its own
       entity — gives per-element precision in physical groups.
    4. Build volume from surface loop, generate tet mesh.
    5. Physical groups survive meshing because they're on geometry entities.
    """
    import sys
    from collections import defaultdict
    from scipy.spatial import KDTree

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        # --- Phase 1: import STL, classify ---
        gmsh.merge(mesh_path)
        gmsh.model.mesh.removeDuplicateNodes()
        angle = 40
        gmsh.model.mesh.classifySurfaces(
            angle * math.pi / 180,
            True,    # boundary
            False,   # forReparametrization — keep original STL triangulation
            180 * math.pi / 180,  # curveAngle
        )
        gmsh.model.mesh.createTopology()
        gmsh.model.geo.synchronize()

        # --- Phase 2: match BC centroids to original STL faces ---
        # No volume mesh exists yet, so we can freely split surface entities.
        all_node_tags, all_coords, _ = gmsh.model.mesh.getNodes()
        node_coord_map = {}
        coords_arr = np.array(all_coords).reshape(-1, 3)
        for i, nt in enumerate(all_node_tags):
            node_coord_map[int(nt)] = coords_arr[i]

        elem_info = {}  # elem_tag -> (surf_tag, [n0, n1, n2])
        boundary_centroids = []
        boundary_elem_tags = []

        for dim, tag in gmsh.model.getEntities(dim=2):
            elem_types, elem_tags_list, elem_node_list = (
                gmsh.model.mesh.getElements(dim, tag)
            )
            for etype, etags, enodes in zip(
                elem_types, elem_tags_list, elem_node_list
            ):
                if etype != 2:
                    continue
                nodes = enodes.reshape(-1, 3)
                for ei, tri_nodes in zip(etags, nodes):
                    ei = int(ei)
                    nds = [int(n) for n in tri_nodes]
                    elem_info[ei] = (tag, nds)
                    v0 = node_coord_map[nds[0]]
                    v1 = node_coord_map[nds[1]]
                    v2 = node_coord_map[nds[2]]
                    boundary_centroids.append((v0 + v1 + v2) / 3.0)
                    boundary_elem_tags.append(ei)

        boundary_centroids = np.array(boundary_centroids)
        n_boundary = len(boundary_centroids)

        surface_result = _extract_surface_data()
        tol = compute_tolerance(surface_result["surface_data"])

        print(
            f"[BlendGmsh] boundary_faces={n_boundary}, "
            f"tolerance={tol:.6e}",
            file=sys.stderr,
        )
        print(
            f"[BlendGmsh] BC centroids: "
            f"{{{', '.join(f'{k}: {len(v)}' for k, v in group_centroids.items())}}}",
            file=sys.stderr,
        )

        boundary_tree = KDTree(boundary_centroids)

        elem_to_groups = defaultdict(set)
        group_matched_elems = defaultdict(set)

        for gname, bc_cents in group_centroids.items():
            if len(bc_cents) == 0:
                continue
            dists, indices = boundary_tree.query(bc_cents)
            for dist, idx in zip(dists, indices):
                if dist <= tol:
                    etag = boundary_elem_tags[idx]
                    elem_to_groups[etag].add(gname)
                    group_matched_elems[gname].add(etag)

        # --- Phase 3: split surface entities by group combo ---
        # No volume mesh yet — clear() works on surface entities.
        next_entity_tag = max(
            t for _, t in gmsh.model.getEntities(dim=2)
        ) + 1

        combo_entities = defaultdict(list)

        for dim, orig_tag in gmsh.model.getEntities(dim=2):
            combo_elems = defaultdict(list)
            elem_types, elem_tags_list, _ = (
                gmsh.model.mesh.getElements(dim, orig_tag)
            )
            for etags in elem_tags_list:
                for etag in etags:
                    etag = int(etag)
                    combo = frozenset(elem_to_groups.get(etag, set()))
                    combo_elems[combo].append(
                        (etag, elem_info[etag][1])
                    )

            if len(combo_elems) <= 1:
                combo = next(iter(combo_elems))
                combo_entities[combo].append(orig_tag)
                continue

            # Multiple combos on same surface — split.
            # clear() removes nodes too; nodes on curves/points survive.
            # Only re-add nodes that were actually removed.
            gmsh.model.mesh.clear([(2, orig_tag)])

            surviving_tags, _, _ = gmsh.model.mesh.getNodes()
            surviving = set(int(t) for t in surviving_tags)

            first = True
            for combo, elems in combo_elems.items():
                if first:
                    target_tag = orig_tag
                    first = False
                else:
                    target_tag = next_entity_tag
                    next_entity_tag += 1
                    gmsh.model.addDiscreteEntity(2, target_tag)

                # Re-add only nodes that were removed by clear()
                subset_node_set = set()
                for _, nds in elems:
                    subset_node_set.update(nds)
                missing = sorted(subset_node_set - surviving)
                if missing:
                    missing_coords = []
                    for n in missing:
                        c = node_coord_map[n]
                        missing_coords.extend(c)
                    gmsh.model.mesh.addNodes(
                        2, target_tag, missing, missing_coords
                    )
                    surviving.update(missing)

                e_tags = [e[0] for e in elems]
                n_tags = []
                for e in elems:
                    n_tags.extend(e[1])
                gmsh.model.mesh.addElementsByType(
                    target_tag, 2, e_tags, n_tags
                )
                combo_entities[combo].append(target_tag)

        # --- Phase 4: create physical groups ---
        group_to_entity_tags = defaultdict(set)
        untagged_entity_tags = set()

        for combo, entity_tags in combo_entities.items():
            if combo:
                for gname in combo:
                    group_to_entity_tags[gname].update(entity_tags)
            else:
                untagged_entity_tags.update(entity_tags)

        for gname, etags in group_to_entity_tags.items():
            gmsh.model.addPhysicalGroup(2, sorted(etags), name=gname)

        if untagged_entity_tags:
            gmsh.model.addPhysicalGroup(
                2, sorted(untagged_entity_tags), name="_untagged"
            )

        # --- Phase 5: build volume, generate tet mesh ---
        volumes = gmsh.model.getEntities(dim=3)
        if not volumes:
            surfaces = gmsh.model.getEntities(dim=2)
            if surfaces:
                sl = gmsh.model.geo.addSurfaceLoop(
                    [t for (_, t) in surfaces]
                )
                gmsh.model.geo.addVolume([sl])
                gmsh.model.geo.synchronize()
                volumes = gmsh.model.getEntities(dim=3)

        if not volumes:
            raise RuntimeError(
                "Could not create volume from STL surface. "
                "Check that the mesh is a closed watertight shell."
            )

        vol_tags = [t for (_, t) in volumes]
        gmsh.model.addPhysicalGroup(3, vol_tags, name="domain")

        _configure_mesh_sizing(curvature=False)
        gmsh.model.mesh.generate(3)

        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(output_msh)

        report = _build_stl_coverage_report(
            group_centroids, group_matched_elems, n_boundary,
            group_to_entity_tags, untagged_entity_tags,
        )
        report.print_report()

    finally:
        gmsh.finalize()

    return report


def _build_stl_coverage_report(
    group_centroids: dict,
    group_matched_elems: dict,
    total_boundary_facets: int,
    group_to_entity_tags: dict,
    untagged_entity_tags: set,
) -> CoverageReport:
    """Build CoverageReport for the STL element-matching pipeline."""
    group_stats = {}
    for gname in group_centroids:
        matched = group_matched_elems.get(gname, set())
        group_stats[gname] = {
            "surfaces": sorted(group_to_entity_tags.get(gname, set())),
            "matched_facets": len(matched),
            "total_facets": len(group_centroids[gname]),
        }

    return CoverageReport(
        group_stats=group_stats,
        unmatched_surfaces=sorted(untagged_entity_tags),
        total_boundary_facets=total_boundary_facets,
    )


def _tag_existing_mesh_mode(data: dict, input_msh: str, output_msh: str) -> CoverageReport:
    """Mesh mode tagging on an existing mesh via KDTree matching."""
    groups = _extract_mesh_groups(data)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        _merge_mesh(input_msh)

        surface_result = _extract_surface_data()
        tol = compute_tolerance(surface_result["surface_data"])
        match_results = match_groups_to_surfaces(
            surface_result["surface_data"], groups, tol
        )

        report = build_coverage_report(match_results, list(groups.keys()))
        zeros = report.check_zero_coverage_groups(list(groups.keys()))
        if zeros:
            raise RuntimeError(
                f"Groups with zero matched surfaces: {zeros}. "
                "Check that group vertex coordinates lie on the mesh surfaces."
            )

        tag_and_write(match_results, surface_result["volume_tags"], output_msh)
        report.print_report()

    finally:
        gmsh.finalize()

    return report
