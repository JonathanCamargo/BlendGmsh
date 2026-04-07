"""
BlendGmsh import operator.

FEMBC_OT_ImportSTEP:
  - Opens a file browser for STEP selection.
  - Calls tessellate_step() to convert STEP -> STL via direct import
    (dependencies installed into Blender's Python at addon register time).
  - Imports the STL into Blender without vertex merging.
  - Creates one vertex group per CAD face island via create_vertex_groups_per_island().
  - Runs fused-face detection via check_fused_faces() and surfaces warnings.

Anti-patterns avoided:
  - Do NOT call bpy.ops.mesh.remove_doubles() -- destroys face identity (CONV-02)
  - Do NOT use bpy.ops.mesh.separate(type='LOOSE') -- creates multiple objects
  - Do NOT store Blender-internal vertex indices in output
"""
import colorsys
import json
import os
import subprocess
import sys
import tempfile
import pathlib

import bpy  # type: ignore  # available inside Blender runtime

from .vertex_groups import create_vertex_groups_per_island
from .mesh_import import parse_stl, parse_obj


# Holds the active subprocess.Popen for mesh generation (polled by timer)
_mesh_proc: subprocess.Popen | None = None
_mesh_output_path: str = ""


class FEMBC_OT_ImportSTEP(bpy.types.Operator):
    bl_idname = "fembc.import_step"
    bl_label = "Import STEP File"
    bl_description = (
        "Import a STEP file and tessellate per-face. "
        "Replaces any existing import."
    )

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default="*.step;*.stp;*.STEP;*.STP",
        options={'HIDDEN'},
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = context.scene.fembc_main

        # Signal "Converting..." in panel
        props.face_count = 0
        props.last_error = ""
        props.step_filepath = self.filepath

        try:
            mesh_data = self._run_tessellation(self.filepath)
        except Exception as exc:
            props.face_count = -1
            props.last_error = (
                f"STEP import failed: {exc}. Check console for details."
            )
            self.report({'ERROR'}, props.last_error)
            return {'CANCELLED'}

        # Build Blender mesh directly from Gmsh tessellation data.
        # No intermediate file — vertex groups are assigned at construction.
        vertices = mesh_data["vertices"]
        faces = mesh_data["faces"]
        face_surface_tags = mesh_data["face_surface_tags"]
        surface_tags = mesh_data["surface_tags"]

        mesh = bpy.data.meshes.new("FEMBC_Mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        # Cache per-polygon surface tag as custom attribute (read at assign time)
        attr = mesh.attributes.new("fembc_surface_tag", 'INT', 'FACE')
        for i, tag in enumerate(face_surface_tags):
            attr.data[i].value = tag

        obj = bpy.data.objects.new("FEMBC_Import", mesh)
        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        # Build per-surface vertex sets and assign vertex groups
        from collections import defaultdict
        tag_to_verts = defaultdict(set)
        for fi, tag in enumerate(face_surface_tags):
            for vi in faces[fi]:
                tag_to_verts[tag].add(vi)

        surface_count = 0
        for tag in surface_tags:
            vert_indices = tag_to_verts.get(tag)
            if not vert_indices:
                continue
            vg = obj.vertex_groups.new(name=f"face_{tag}")
            vg.add(list(vert_indices), 1.0, 'REPLACE')
            surface_count += 1

        # Fused-face detection
        from step_converter import check_fused_faces
        warnings = check_fused_faces(
            expected=mesh_data.get("n_surfaces", 0),
            actual=surface_count,
        )

        props.face_count = surface_count
        props.bc_mode = "brep"
        if warnings:
            expected = mesh_data.get("n_surfaces", 0)
            props.last_error = (
                f"Faces fused: {expected} expected, {surface_count} found. "
                "Re-export from CAD without face merging."
            )
        else:
            props.last_error = ""

        return {'FINISHED'}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_tessellation(self, step_path: str) -> dict:
        """Run tessellate_step_to_mesh() directly — deps are installed at register time."""
        import sys
        # step_converter/ is bundled inside the addon directory
        addon_dir = os.path.dirname(os.path.abspath(__file__))
        if addon_dir not in sys.path:
            sys.path.insert(0, addon_dir)

        from step_converter import tessellate_step_to_mesh
        return tessellate_step_to_mesh(step_path)


class FEMBC_OT_ImportMesh(bpy.types.Operator):
    bl_idname = "fembc.import_mesh"
    bl_label = "Import STL/OBJ File"
    bl_description = (
        "Import an STL or OBJ surface mesh. "
        "Sets mesh mode -- assign BC groups manually after import."
    )

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default="*.stl;*.STL;*.obj;*.OBJ",
        options={'HIDDEN'},
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = context.scene.fembc_main
        props.face_count = 0
        props.last_error = ""

        ext = os.path.splitext(self.filepath)[1].lower()
        try:
            if ext == '.stl':
                vertices, faces = parse_stl(self.filepath)
            elif ext == '.obj':
                vertices, faces = parse_obj(self.filepath)
            else:
                props.last_error = f"Unsupported format: {ext}"
                self.report({'ERROR'}, props.last_error)
                return {'CANCELLED'}
        except Exception as exc:
            props.face_count = -1
            props.last_error = f"Import failed: {exc}"
            self.report({'ERROR'}, props.last_error)
            return {'CANCELLED'}

        mesh = bpy.data.meshes.new("FEMBC_Mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        obj = bpy.data.objects.new("FEMBC_Import", mesh)
        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        props.face_count = len(faces)
        props.bc_mode = "mesh"
        props.step_filepath = self.filepath
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Group management helpers
# ---------------------------------------------------------------------------

def rebuild_group_materials(obj, groups):
    """Create/sync one material slot per group with distinct per-group color.

    Must be called whenever groups are added or removed.
    Material names: FEMBC_{group_name}. mat.use_nodes = False so
    diffuse_color shows in Solid viewport mode.
    """
    n = len(groups)
    while len(obj.data.materials) < n:
        obj.data.materials.append(None)
    while len(obj.data.materials) > n:
        obj.data.materials.pop()

    for i, group_item in enumerate(groups):
        r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.9)
        group_item.color = (r, g, b)

        mat_name = f"FEMBC_{group_item.name}"
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = False
        mat.diffuse_color = (r, g, b, 1.0)
        obj.data.materials[i] = mat


def update_face_material_indices(obj, groups):
    """Set polygon.material_index to match the face's vertex group membership.

    Must be in Object Mode to write polygon data.
    """
    mesh = obj.data
    group_name_to_idx = {g.name: i for i, g in enumerate(groups)}

    vert_to_group = {}
    for vg in obj.vertex_groups:
        if vg.name not in group_name_to_idx:
            continue
        mat_idx = group_name_to_idx[vg.name]
        for v in mesh.vertices:
            try:
                w = vg.weight(v.index)
                if w > 0.0:
                    vert_to_group[v.index] = mat_idx
            except RuntimeError:
                pass

    for poly in mesh.polygons:
        # Only color a face if ALL its vertices belong to the same group
        groups_for_poly = [vert_to_group[vi] for vi in poly.vertices if vi in vert_to_group]
        if len(groups_for_poly) == len(poly.vertices):
            poly.material_index = groups_for_poly[0]
    mesh.update()


# ---------------------------------------------------------------------------
# Group operators
# ---------------------------------------------------------------------------

class FEMBC_OT_AddGroup(bpy.types.Operator):
    bl_idname = "fembc.add_group"
    bl_label = "Add Group"
    bl_description = "Add a new named face group"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == 'MESH')

    def execute(self, context):
        groups = context.scene.fembc_groups
        item = groups.add()
        item.name = f"Group.{len(groups):03d}"
        context.scene.fembc_groups_active_index = len(groups) - 1
        obj = context.active_object
        if obj and obj.type == 'MESH':
            rebuild_group_materials(obj, groups)
        return {'FINISHED'}


class FEMBC_OT_RemoveGroup(bpy.types.Operator):
    bl_idname = "fembc.remove_group"
    bl_label = "Remove Group"
    bl_description = "Remove the active face group"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.fembc_groups) > 0

    def execute(self, context):
        scene = context.scene
        idx = scene.fembc_groups_active_index
        group_name = scene.fembc_groups[idx].name

        obj = context.active_object
        if obj and obj.type == 'MESH':
            vg = obj.vertex_groups.get(group_name)
            if vg:
                obj.vertex_groups.remove(vg)

        scene.fembc_groups.remove(idx)
        scene.fembc_groups_active_index = max(0, idx - 1)

        if obj and obj.type == 'MESH':
            rebuild_group_materials(obj, scene.fembc_groups)
            update_face_material_indices(obj, scene.fembc_groups)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Face assignment and JSON export operators
# ---------------------------------------------------------------------------

def _update_cached_surface_tags(obj, group_item):
    """Accumulate BREP surface tags from selected polygons into group cache.

    Reads the fembc_surface_tag custom polygon attribute (set at STEP import).
    Must be in Object Mode.  No-op if the attribute doesn't exist (mesh mode).
    """
    mesh = obj.data
    attr = mesh.attributes.get("fembc_surface_tag")
    if attr is None:
        return

    existing = set()
    if group_item.surface_tags:
        existing = set(json.loads(group_item.surface_tags))

    for poly in mesh.polygons:
        if poly.select:
            existing.add(attr.data[poly.index].value)

    group_item.surface_tags = json.dumps(sorted(existing))


class FEMBC_OT_AssignFaces(bpy.types.Operator):
    bl_idname = "fembc.assign_faces"
    bl_label = "Assign Selected Faces"
    bl_description = "Assign currently selected faces to the active group"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (
            context.active_object is not None
            and context.active_object.type == 'MESH'
            and context.mode == 'EDIT_MESH'
            and len(context.scene.fembc_groups) > 0
        )

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        idx = scene.fembc_groups_active_index
        group_item = scene.fembc_groups[idx]
        group_name = group_item.name

        # Ensure vertex group exists on the object
        vg = obj.vertex_groups.get(group_name)
        if vg is None:
            vg = obj.vertex_groups.new(name=group_name)

        # Set active vertex group so vertex_group_assign uses it
        obj.vertex_groups.active_index = obj.vertex_groups.find(group_name)

        # Assign selected faces' vertices to active group (Edit Mode operator)
        bpy.ops.object.vertex_group_assign()

        # Switch to Object Mode to update material indices, then back to Edit.
        # update_face_material_indices requires Object Mode to write polygon.material_index.
        bpy.ops.object.mode_set(mode='OBJECT')
        update_face_material_indices(obj, scene.fembc_groups)
        _update_cached_surface_tags(obj, group_item)
        bpy.ops.object.mode_set(mode='EDIT')

        return {'FINISHED'}


class FEMBC_OT_ExportJSON(bpy.types.Operator):
    bl_idname = "fembc.export_json"
    bl_label = "Export to JSON"
    bl_description = "Export all face groups to a JSON file conforming to the bc_groups_v1 schema"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return (
            context.active_object is not None
            and context.active_object.type == 'MESH'
            and any(
                True for g in context.scene.fembc_groups
                if context.active_object.vertex_groups.get(g.name) is not None
            )
        )

    def invoke(self, context, event):
        self.filepath = "bc_groups.json"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        import json
        from mathutils import Matrix
        from .export import build_bc_groups_dict

        obj = context.active_object
        scene = context.scene

        # Non-identity transform warning (per UI-SPEC and Pitfall 6)
        if obj.matrix_world != Matrix.Identity(4):
            self.report(
                {'WARNING'},
                "Object has a non-identity transform. Apply scale/rotation "
                "before export for correct world-space coordinates."
            )

        try:
            data = build_bc_groups_dict(
                obj=obj,
                groups=scene.fembc_groups,
                step_filepath=scene.fembc_main.step_filepath,
                blender_version=bpy.app.version_string,
            )
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            self.report(
                {'ERROR'},
                f"Export failed: could not write to {self.filepath}. "
                f"Check the file path and permissions."
            )
            return {'CANCELLED'}

        # Per-group feedback
        mode = data.get("mode", "brep")
        parts = []
        for gname, gdata in data.get("groups", {}).items():
            if mode == "brep":
                n = len(gdata.get("surface_tags", []))
                parts.append(f"{gname}: {n} BREP surfaces")
            else:
                nv = gdata.get("vertex_count", 0)
                nf = gdata.get("face_count", 0)
                parts.append(f"{gname}: {nv} verts, {nf} faces")

        summary = ", ".join(parts) if parts else "no groups"
        self.report({'INFO'}, f"Exported ({mode} mode) {summary} -> {self.filepath}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Timer callback for subprocess polling
# ---------------------------------------------------------------------------

def _poll_mesh_subprocess():
    """Timer callback: poll _mesh_proc and update props on main thread.

    Returns 0.5 to reschedule while subprocess is running, or None to
    unregister once it completes.

    Uses 'import bpy as _bpy' inside the function body (not a closure over
    the execute() context argument) to avoid Pitfall 7: stale context
    reference causing ReferenceError after operator completes.
    """
    global _mesh_proc, _mesh_output_path
    import bpy as _bpy

    if _mesh_proc is None:
        return None  # nothing to poll

    retcode = _mesh_proc.poll()
    if retcode is None:
        return 0.5  # still running, reschedule

    props = _bpy.context.scene.fembc_main
    if retcode == 0:
        props.mesh_status = f"Mesh saved: {_mesh_output_path}"
        props.msh_output_path = _mesh_output_path
    else:
        stderr = _mesh_proc.stderr.read() if _mesh_proc.stderr else ""
        # Print full error to Blender console for debugging
        if stderr:
            print(f"[BlendGmsh] Mesh generation failed:\n{stderr}")
        # Show short message in panel
        short_err = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
        props.mesh_status = f"Mesh generation failed: {short_err}"
        props.msh_output_path = ""

    _mesh_proc = None

    # Force VIEW_3D panel redraw — Blender won't auto-redraw from a timer.
    for window in _bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    return None  # unregister timer


# ---------------------------------------------------------------------------
# Generate Mesh operator
# ---------------------------------------------------------------------------

class FEMBC_OT_GenerateMesh(bpy.types.Operator):
    bl_idname = "fembc.generate_mesh"
    bl_label = "Generate Mesh"
    bl_description = "Generate a tagged .msh file from the imported geometry"

    @classmethod
    def poll(cls, context):
        props = context.scene.fembc_main
        return (
            props.bc_mode in ("brep", "mesh")
            and bool(props.step_filepath)
            and props.mesh_status != "Generating mesh..."
        )

    def execute(self, context):
        global _mesh_proc, _mesh_output_path
        props = context.scene.fembc_main
        obj = context.active_object
        scene = context.scene

        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected.")
            return {'CANCELLED'}

        if props.bc_mode == "brep":
            # BREP mode: read cached surface tags — zero vertex iteration.
            bc_groups = {}
            for group_item in scene.fembc_groups:
                if group_item.surface_tags:
                    tags = json.loads(group_item.surface_tags)
                    if tags:
                        bc_groups[group_item.name] = {"surface_tags": tags}
            mesh_data = obj.data
            bc_dict = {
                "schema_version": 1,
                "source": "blendgmsh",
                "mode": "brep",
                "step_file": props.step_filepath,
                "units": "meters",
                "groups": bc_groups,
                "mesh_stats": {
                    "total_vertices": len(mesh_data.vertices),
                    "total_faces": len(mesh_data.polygons),
                    "bounding_box": {
                        "min": [0.0, 0.0, 0.0],
                        "max": [0.0, 0.0, 0.0],
                    },
                },
            }
        else:
            # Mesh mode: need vertex coords for KDTree matching.
            from .export import build_bc_groups_dict
            bc_dict = build_bc_groups_dict(
                obj=obj,
                groups=scene.fembc_groups,
                step_filepath=props.step_filepath,
                blender_version=bpy.app.version_string,
            )

        # Compute output path: /tmp/<stem>.msh (per MESH-04)
        stem = pathlib.Path(props.step_filepath).stem
        output_msh = str(pathlib.Path(tempfile.gettempdir()) / f"{stem}.msh")

        # Write temp JSON for run_full_pipeline() which takes a file path
        tmp_json = str(pathlib.Path(tempfile.gettempdir()) / "fembc_live_bc_groups.json")
        with open(tmp_json, "w") as f:
            json.dump(bc_dict, f)

        # Signal "Generating mesh..." before subprocess starts (per STAT-03)
        props.mesh_status = "Generating mesh..."
        props.msh_output_path = ""

        # Use subprocess instead of thread: gmsh.initialize() calls
        # signal.signal() which is main-thread-only, so Gmsh cannot run
        # in a background thread. The CLI entry point
        # `python -m matching_library run` is already available.
        addon_dir = os.path.dirname(os.path.abspath(__file__))
        _mesh_output_path = output_msh
        _mesh_proc = subprocess.Popen(
            [sys.executable, "-m", "matching_library", "run",
             tmp_json, props.step_filepath, output_msh],
            env={**os.environ, "PYTHONPATH": addon_dir},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Register timer to poll subprocess on main thread (per MESH-03)
        bpy.app.timers.register(_poll_mesh_subprocess, first_interval=0.5)
        return {'FINISHED'}
