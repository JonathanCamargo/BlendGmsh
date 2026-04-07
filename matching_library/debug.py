"""Debug tools for inspecting meshes, BC groups, and STEP geometry.

Public API:
  - inspect_msh(msh_path) -> dict
  - inspect_bc_groups(bc_json_path, step_path) -> dict
  - visualize_bc_groups(bc_json_path, msh_path, output_png=None) -> None
"""

import json
import math

import gmsh
import numpy as np


def inspect_msh(msh_path: str) -> dict:
    """Inspect a .msh file: element counts, physical groups, per-surface stats.

    Returns a dict with keys:
        nodes: int
        elements: {type_name: count}
        physical_groups: [{dim, tag, name, n_entities, n_elements}]
        surfaces: [{tag, centroid, area, n_triangles}]
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        gmsh.merge(msh_path)

        # Nodes
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)

        # Elements by type
        elem_counts = {}
        for dim in range(4):
            for d, t in gmsh.model.getEntities(dim):
                etypes, etags, _ = gmsh.model.mesh.getElements(d, t)
                for etype, et in zip(etypes, etags):
                    name, _, _, _, _, _ = gmsh.model.mesh.getElementProperties(etype)
                    elem_counts[name] = elem_counts.get(name, 0) + len(et)

        # Physical groups
        phys_list = []
        for dim, pt in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, pt)
            ents = gmsh.model.getEntitiesForPhysicalGroup(dim, pt)
            n_elems = 0
            for e in ents:
                etypes, etags, _ = gmsh.model.mesh.getElements(dim, int(e))
                n_elems += sum(len(et) for et in etags)
            phys_list.append({
                "dim": dim, "tag": pt, "name": name,
                "n_entities": len(ents), "n_elements": n_elems,
            })

        # Per-surface entity stats
        surf_list = []
        for _, tag in gmsh.model.getEntities(dim=2):
            etypes, etags, enodes = gmsh.model.mesh.getElements(2, tag)
            n_tris = 0
            tri_nodes = None
            for etype, et, en in zip(etypes, etags, enodes):
                if etype == 2:  # Triangle 3
                    n_tris = len(et)
                    tri_nodes = en.reshape(-1, 3)

            node_tags_s, coords, _ = gmsh.model.mesh.getNodes(
                2, tag, includeBoundary=True
            )
            if len(coords) == 0:
                surf_list.append({
                    "tag": tag, "centroid": None, "area": 0.0,
                    "n_triangles": 0, "n_nodes": 0,
                })
                continue

            verts = np.array(coords).reshape(-1, 3)
            node_map = {int(nt): i for i, nt in enumerate(node_tags_s)}
            centroid = verts.mean(axis=0).tolist()

            # Compute area from triangles
            area = 0.0
            if tri_nodes is not None:
                for tri in tri_nodes:
                    idx = [node_map.get(int(n)) for n in tri]
                    if any(i is None for i in idx):
                        continue
                    v0, v1, v2 = verts[idx[0]], verts[idx[1]], verts[idx[2]]
                    area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

            surf_list.append({
                "tag": tag, "centroid": centroid, "area": float(area),
                "n_triangles": n_tris, "n_nodes": len(node_tags_s),
            })

    finally:
        gmsh.finalize()

    result = {
        "nodes": n_nodes,
        "elements": elem_counts,
        "physical_groups": phys_list,
        "surfaces": surf_list,
    }

    # Print summary
    print(f"=== {msh_path} ===")
    print(f"Nodes: {n_nodes}")
    for name, count in sorted(elem_counts.items()):
        print(f"  {name}: {count}")

    if phys_list:
        print(f"\nPhysical groups ({len(phys_list)}):")
        for pg in phys_list:
            flag = " *** EMPTY ***" if pg["n_elements"] == 0 else ""
            print(f"  dim={pg['dim']} \"{pg['name']}\": "
                  f"{pg['n_entities']} entities, {pg['n_elements']} elements{flag}")

    empty_surfs = [s for s in surf_list if s["n_triangles"] == 0 and s["n_nodes"] > 0]
    if empty_surfs:
        print(f"\nWARNING: {len(empty_surfs)} surfaces have nodes but no triangle elements")
        print("  (mesh was likely saved without SaveAll or surface physical groups)")

    print(f"\nSurfaces with triangles: "
          f"{sum(1 for s in surf_list if s['n_triangles'] > 0)}/{len(surf_list)}")

    return result


def inspect_bc_groups(bc_json_path: str, step_path: str) -> dict:
    """Cross-reference bc_groups.json surface tags against STEP geometry.

    For each BC group, looks up its surface tags in the STEP model and
    computes bounding box, centroid, and area from BREP geometry.

    Returns a dict: {group_name: {tags, surfaces: [{tag, centroid, bbox, area}], combined_centroid}}
    """
    with open(bc_json_path) as f:
        bc_data = json.load(f)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()

        model_tags = {t for _, t in gmsh.model.getEntities(dim=2)}

        result = {}
        for gname, gdata in bc_data["groups"].items():
            tags = gdata["surface_tags"]
            surfaces = []
            all_centroids = []
            all_areas = []

            for tag in tags:
                if tag not in model_tags:
                    surfaces.append({
                        "tag": tag, "centroid": None, "bbox": None,
                        "area": None, "error": "NOT FOUND in STEP",
                    })
                    continue

                bb = gmsh.model.getBoundingBox(2, tag)
                centroid = [
                    (bb[0] + bb[3]) / 2,
                    (bb[1] + bb[4]) / 2,
                    (bb[2] + bb[5]) / 2,
                ]
                size = [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]

                # Get area via mass properties
                try:
                    mass = gmsh.model.occ.getMass(2, tag)
                except (RuntimeError, ValueError):
                    mass = 0.0

                surfaces.append({
                    "tag": tag, "centroid": centroid,
                    "bbox": {"min": list(bb[:3]), "max": list(bb[3:]), "size": size},
                    "area": mass,
                })
                all_centroids.append(centroid)
                all_areas.append(mass)

            # Area-weighted combined centroid
            combined_centroid = None
            total_area = sum(all_areas)
            if all_centroids and total_area > 0:
                weighted = [0.0, 0.0, 0.0]
                for c, a in zip(all_centroids, all_areas):
                    for i in range(3):
                        weighted[i] += c[i] * a
                combined_centroid = [w / total_area for w in weighted]

            result[gname] = {
                "tags": tags,
                "surfaces": surfaces,
                "combined_centroid": combined_centroid,
                "total_area": total_area,
            }

    finally:
        gmsh.finalize()

    # Print summary
    print(f"=== BC groups: {bc_json_path} vs {step_path} ===")
    print(f"Model has {len(model_tags)} surfaces (tags 1-{max(model_tags)})\n")

    for gname, gdata in result.items():
        print(f"Group \"{gname}\" ({len(gdata['tags'])} surfaces, "
              f"total area={gdata['total_area']:.4f}):")
        if gdata["combined_centroid"]:
            c = gdata["combined_centroid"]
            print(f"  Combined centroid: ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})")
        for s in gdata["surfaces"]:
            if "error" in s:
                print(f"  surf {s['tag']}: {s['error']}")
            else:
                c = s["centroid"]
                sz = s["bbox"]["size"]
                print(f"  surf {s['tag']:3d}: centroid=({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})  "
                      f"size=({sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f})  area={s['area']:.4f}")
        print()

    return result


def visualize_bc_groups(
    bc_json_path: str, msh_path: str, output_png: str = None
) -> None:
    """Visualize BC group surfaces on the mesh, colored by group.

    Loads the mesh, maps surface entities to BC groups via the JSON,
    and renders with PyVista: each BC group gets a distinct color,
    _untagged surfaces are transparent gray wireframe for context.
    """
    import pyvista as pv

    with open(bc_json_path) as f:
        bc_data = json.load(f)

    # Build surface_tag -> group_name mapping
    tag_to_group = {}
    for gname, gdata in bc_data["groups"].items():
        for tag in gdata["surface_tags"]:
            tag_to_group[tag] = gname

    group_names = list(bc_data["groups"].keys())

    color_cycle = ["blue", "red", "green", "purple", "orange", "cyan", "magenta"]
    colors = {gname: color_cycle[i % len(color_cycle)]
              for i, gname in enumerate(group_names)}

    # Extract mesh data per group from the .msh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    try:
        gmsh.merge(msh_path)

        group_meshes = {gname: {"verts": [], "faces": [], "offset": 0}
                        for gname in group_names}
        untagged_verts = []
        untagged_faces = []
        untagged_offset = 0

        for _, tag in gmsh.model.getEntities(dim=2):
            gname = tag_to_group.get(tag)

            node_tags, coords, _ = gmsh.model.mesh.getNodes(
                2, tag, includeBoundary=True
            )
            if len(coords) == 0:
                continue
            verts = np.array(coords).reshape(-1, 3)
            node_map = {int(nt): i for i, nt in enumerate(node_tags)}

            etypes, _, enodes = gmsh.model.mesh.getElements(2, tag)
            tris = []
            for etype, en in zip(etypes, enodes):
                if etype == 2:
                    raw = en.reshape(-1, 3)
                    for tri in raw:
                        idx = [node_map.get(int(n)) for n in tri]
                        if any(i is None for i in idx):
                            continue
                        tris.append(idx)

            if not tris:
                continue

            tris = np.array(tris, dtype=np.int32)

            if gname and gname in group_meshes:
                gm = group_meshes[gname]
                offset = gm["offset"]
                gm["verts"].append(verts)
                padded = np.hstack([
                    np.full((len(tris), 1), 3, dtype=np.int32),
                    tris + offset,
                ])
                gm["faces"].append(padded.ravel())
                gm["offset"] += len(verts)
            else:
                padded = np.hstack([
                    np.full((len(tris), 1), 3, dtype=np.int32),
                    tris + untagged_offset,
                ])
                untagged_verts.append(verts)
                untagged_faces.append(padded.ravel())
                untagged_offset += len(verts)

    finally:
        gmsh.finalize()

    # Build PyVista meshes and plot
    pl = pv.Plotter(off_screen=(output_png is not None))

    # Add untagged as context wireframe
    if untagged_verts:
        mesh_u = pv.PolyData(
            np.vstack(untagged_verts), np.concatenate(untagged_faces)
        )
        pl.add_mesh(mesh_u, color="lightgray", opacity=0.3, show_edges=True,
                     edge_color="gray", label="_untagged")

    # Add each BC group
    for gname in group_names:
        gm = group_meshes[gname]
        if not gm["verts"]:
            continue
        mesh_g = pv.PolyData(np.vstack(gm["verts"]), np.concatenate(gm["faces"]))
        pl.add_mesh(mesh_g, color=colors[gname], show_edges=True,
                     edge_color="black", label=gname, opacity=1.0)

    pl.add_legend()
    pl.add_axes()

    if output_png:
        pl.show(screenshot=output_png)
        print(f"Saved: {output_png}")
    else:
        pl.show()
