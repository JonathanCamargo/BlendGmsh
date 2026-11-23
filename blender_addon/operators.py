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
import time
from contextlib import contextmanager
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
_mesh_log_fh = None          # log file the subprocess writes to (avoids a full-pipe hang)
_mesh_scene: str = ""        # scene that started the run
_mesh_expected: list = []    # group names sent to the mesher, checked against the .msh
_mesh_t0: float = 0.0        # start time (a float32 Blender property can't hold a timestamp)


def lib_dir():
    """Folder holding the matching_library / step_converter packages.

    Installed zip (export.ps1/.sh): bundled inside the add-on folder.
    Source checkout (e.g. VS Code 'Blender Development' links blender_addon/):
    they sit next to it, in the repository root.
    """
    # realpath: a linked add-on's parent must be the repository, not the link's folder
    addon_dir = os.path.dirname(os.path.realpath(__file__))
    for d in (addon_dir, os.path.dirname(addon_dir)):
        if os.path.isfile(os.path.join(d, "matching_library", "__main__.py")):
            return d
    raise RuntimeError(
        "BlendGmsh is incomplete: 'matching_library' was not found in "
        f"{addon_dir} or its parent folder. Reinstall the add-on from the zip made by export.ps1.")


def _ensure_lib_path():
    d = lib_dir()
    if d not in sys.path:
        sys.path.insert(0, d)


def _start_tagging(op, context, obj):
    """After an import: keep the group names, empty them (old tags belong to the
    previous part) and show the new part grey = nothing assigned yet."""
    scene = context.scene
    had = [g.name for g in scene.fembc_groups if g.surface_tags or g.assigned > 0]
    for g in scene.fembc_groups:
        g.surface_tags = ""
        g.assigned = 0
        g.prev_name = g.name
    scene.fembc_main.groups_object = obj.name
    scene.fembc_main.msh_output_path = ""
    scene.fembc_main.mesh_status = ""
    scene.fembc_main.mesh_warning = ""
    if context.mode == 'OBJECT':
        update_face_material_indices(obj, scene.fembc_groups)
    if had:
        op.report({'INFO'}, f"New part: groups {', '.join(had)} kept but emptied - assign their faces again")


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
        obj["fembc_source"] = self.filepath
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
        _ensure_lib_path()
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

        _start_tagging(self, context, obj)
        return {'FINISHED'}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_tessellation(self, step_path: str) -> dict:
        """Run tessellate_step_to_mesh() directly — deps are installed at register time."""
        _ensure_lib_path()
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
        obj["fembc_source"] = self.filepath
        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        props.face_count = len(faces)
        props.bc_mode = "mesh"
        props.step_filepath = self.filepath
        _start_tagging(self, context, obj)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Group management helpers
# ---------------------------------------------------------------------------
#
# Source of truth for what a group contains:
#   BREP (STEP import): group_item.surface_tags (CAD surface ids). Vertex groups are
#     derived from it. Surfaces share edge vertices, so vertex membership alone
#     would leak neighbouring triangles into a group.
#   Mesh (STL/OBJ/native object): the vertex group named like the group.
#
# Colours: material slot 0 is neutral grey for unassigned faces, slot i+1 is group i,
# so an object only shows a group colour where faces were really assigned.

_UNASSIGNED_MAT = "FEMBC__unassigned"
_UNASSIGNED_RGB = (0.55, 0.55, 0.55)
_DEFAULT_GROUP_NAMES = ("support", "load")


def _is_brep(obj):
    return (obj is not None and obj.type == 'MESH'
            and obj.data.attributes.get("fembc_surface_tag") is not None)


def _tags(group_item):
    try:
        return set(json.loads(group_item.surface_tags)) if group_item.surface_tags else set()
    except ValueError:
        return set()


def _set_tags(group_item, tags):
    group_item.surface_tags = json.dumps(sorted(int(t) for t in tags))


def group_count(group_item):
    """Assigned surfaces/faces, or -1 when unknown (groups from older .blend files)."""
    if group_item.assigned >= 0:
        return group_item.assigned
    return len(_tags(group_item)) if group_item.surface_tags else -1


@contextmanager
def _object_mode(context):
    """Polygon data written in Edit Mode is discarded on leaving it; work in Object Mode."""
    was_edit = context.mode == 'EDIT_MESH'
    if was_edit:
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        yield
    finally:
        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')


def _sync_group_names(scene):
    """Carry list renames (double-click) over to vertex groups and materials."""
    for g in scene.fembc_groups:
        old = g.prev_name
        if old and old != g.name:
            for ob in bpy.data.objects:
                if ob.type == 'MESH':
                    vg = ob.vertex_groups.get(old)
                    if vg is not None and ob.vertex_groups.get(g.name) is None:
                        vg.name = g.name
            mat = bpy.data.materials.get(f"FEMBC_{old}")
            if mat is not None and bpy.data.materials.get(f"FEMBC_{g.name}") is None:
                mat.name = f"FEMBC_{g.name}"
        g.prev_name = g.name


def _poly_tags(mesh):
    attr = mesh.attributes["fembc_surface_tag"]
    vals = [0] * len(attr.data)
    attr.data.foreach_get("value", vals)
    return vals


def rebuild_group_materials(obj, groups):
    """Sync material slots: 0 = unassigned (grey), i+1 = group i (its colour)."""
    mats = obj.data.materials
    n = len(groups) + 1
    while len(mats) < n:
        mats.append(None)
    while len(mats) > n:
        mats.pop()

    def material(name, rgb):
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
        try:
            mat.use_nodes = False   # diffuse_color shows in Solid view (no-op in Blender 5)
        except (AttributeError, TypeError):
            pass
        mat.diffuse_color = (*rgb, 1.0)
        return mat

    mats[0] = material(_UNASSIGNED_MAT, _UNASSIGNED_RGB)
    for i, g in enumerate(groups):
        if tuple(round(c, 4) for c in g.color) == (1.0, 1.0, 1.0):
            # Never coloured: golden-ratio hues stay stable as groups are added
            g.color = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.85, 0.9)
        mats[i + 1] = material(f"FEMBC_{g.name}", tuple(g.color))


def _group_members(obj, groups):
    """Per group: set of polygon indices in it. Object Mode."""
    mesh = obj.data
    if _is_brep(obj):
        ptags = _poly_tags(mesh)
        out = []
        for g in groups:
            t = _tags(g)
            out.append({i for i, tag in enumerate(ptags) if tag in t} if t else set())
        return out

    vg_to_k = {}
    for k, g in enumerate(groups):
        vg = obj.vertex_groups.get(g.name)
        if vg is not None:
            vg_to_k[vg.index] = k
    verts = [set() for _ in groups]
    if vg_to_k:
        for v in mesh.vertices:
            for ge in v.groups:
                k = vg_to_k.get(ge.group)
                if k is not None and ge.weight > 0.0:
                    verts[k].add(v.index)
    return [{p.index for p in mesh.polygons if vs.issuperset(p.vertices)} if vs else set()
            for vs in verts]


def update_face_material_indices(obj, groups):
    """Colour faces by group (grey = unassigned) and refresh the per-group counts.

    Must be in Object Mode to write polygon data.
    """
    rebuild_group_materials(obj, groups)
    mesh = obj.data
    members = _group_members(obj, groups)
    idx = [0] * len(mesh.polygons)
    for k in reversed(range(len(groups))):   # first group wins where groups overlap
        for pi in members[k]:
            idx[pi] = k + 1
    mesh.polygons.foreach_set("material_index", idx)
    brep = _is_brep(obj)
    for g, m in zip(groups, members):
        g.assigned = len(_tags(g)) if brep else len(m)
    mesh.update()


def _rebuild_brep_vertex_group(obj, group_item):
    """Make the group's vertex group hold exactly the vertices of its CAD surfaces."""
    mesh = obj.data
    vg = obj.vertex_groups.get(group_item.name) or obj.vertex_groups.new(name=group_item.name)
    vg.remove(list(range(len(mesh.vertices))))
    t = _tags(group_item)
    if not t:
        return
    ptags = _poly_tags(mesh)
    verts = set()
    for p in mesh.polygons:
        if ptags[p.index] in t:
            verts.update(p.vertices)
    if verts:
        vg.add(list(verts), 1.0, 'REPLACE')


def _edit_selected_tags(obj, expand=True):
    """Edit Mode: CAD surface tags under the selection. With expand, the whole CAD
    face gets selected so the user sees what a group will really contain."""
    import bmesh
    bm = bmesh.from_edit_mesh(obj.data)
    layer = bm.faces.layers.int.get("fembc_surface_tag")
    if layer is None:
        return set()
    sel = {f[layer] for f in bm.faces if f.select}
    if expand and sel:
        for f in bm.faces:
            if f[layer] in sel and not f.select:
                f.select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    return sel


def _edit_has_selected_faces(obj):
    import bmesh
    return any(f.select for f in bmesh.from_edit_mesh(obj.data).faces)


def _active_group(scene):
    groups = scene.fembc_groups
    i = scene.fembc_groups_active_index
    return groups[i] if 0 <= i < len(groups) else None


def _tag_poll(cls, context, need_edit=True):
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        cls.poll_message_set("Select the part (a mesh object) first")
        return False
    if _active_group(context.scene) is None:
        cls.poll_message_set("Add a face group first (+)")
        return False
    if need_edit and context.mode != 'EDIT_MESH':
        cls.poll_message_set("Enter Edit Mode (Tab) and select faces first")
        return False
    return True


# ---------------------------------------------------------------------------
# Group operators
# ---------------------------------------------------------------------------

class FEMBC_OT_AddGroup(bpy.types.Operator):
    bl_idname = "fembc.add_group"
    bl_label = "Add Group"
    bl_description = "Add a new face group (double-click its name to rename it)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.active_object is None or context.active_object.type != 'MESH':
            cls.poll_message_set("Select the part (a mesh object) first")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        _sync_group_names(scene)
        groups = scene.fembc_groups
        names = {g.name for g in groups}
        name = next((n for n in _DEFAULT_GROUP_NAMES if n not in names), None)
        i = len(groups) + 1
        while name is None or name in names:
            name, i = f"group_{i}", i + 1
        item = groups.add()
        item.name = item.prev_name = name
        item.assigned = 0
        scene.fembc_groups_active_index = len(groups) - 1
        obj = context.active_object
        with _object_mode(context):
            update_face_material_indices(obj, groups)
        return {'FINISHED'}


class FEMBC_OT_RemoveGroup(bpy.types.Operator):
    bl_idname = "fembc.remove_group"
    bl_label = "Remove Group"
    bl_description = "Delete the active face group (its faces become unassigned)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.fembc_groups) > 0

    def execute(self, context):
        scene = context.scene
        _sync_group_names(scene)
        idx = scene.fembc_groups_active_index
        group_name = scene.fembc_groups[idx].name

        obj = context.active_object
        with _object_mode(context):
            if obj and obj.type == 'MESH':
                vg = obj.vertex_groups.get(group_name)
                if vg:
                    obj.vertex_groups.remove(vg)
            scene.fembc_groups.remove(idx)
            scene.fembc_groups_active_index = max(0, idx - 1)
            if obj and obj.type == 'MESH':
                update_face_material_indices(obj, scene.fembc_groups)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Face assignment operators
# ---------------------------------------------------------------------------

def _change_membership(op, context, add):
    obj = context.active_object
    scene = context.scene
    _sync_group_names(scene)
    g = _active_group(scene)
    if not _edit_has_selected_faces(obj):
        op.report({'WARNING'}, "No faces selected - click faces in the viewport first")
        return {'CANCELLED'}

    if _is_brep(obj):
        sel = _edit_selected_tags(obj, expand=True)
        before = _tags(g)
        after = (before | sel) if add else (before - sel)
        _set_tags(g, after)
        with _object_mode(context):
            _rebuild_brep_vertex_group(obj, g)
            update_face_material_indices(obj, scene.fembc_groups)
        changed = len(after - before) if add else len(before - after)
        what = f"{changed} CAD surface{'s' if changed != 1 else ''}"
        total = f"{len(after)} total"
    else:
        vg = obj.vertex_groups.get(g.name) or obj.vertex_groups.new(name=g.name)
        obj.vertex_groups.active_index = vg.index
        before = g.assigned
        if add:
            bpy.ops.object.vertex_group_assign()
        else:
            bpy.ops.object.vertex_group_remove_from()
        with _object_mode(context):
            update_face_material_indices(obj, scene.fembc_groups)
        changed = abs(g.assigned - max(before, 0))
        what = f"{changed} face{'s' if changed != 1 else ''}"
        total = f"{g.assigned} total"

    scene.fembc_main.groups_object = obj.name
    verb = "added to" if add else "removed from"
    op.report({'INFO'}, f"{what} {verb} '{g.name}' ({total})")
    return {'FINISHED'}


class FEMBC_OT_AssignFaces(bpy.types.Operator):
    bl_idname = "fembc.assign_faces"
    bl_label = "Assign Selected Faces"
    bl_description = ("Add the selected faces to the active group. On a STEP part, any "
                      "selected triangle adds its whole CAD face")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _tag_poll(cls, context)

    def execute(self, context):
        return _change_membership(self, context, add=True)


class FEMBC_OT_UnassignFaces(bpy.types.Operator):
    bl_idname = "fembc.unassign_faces"
    bl_label = "Remove Selected Faces"
    bl_description = "Take the selected faces out of the active group"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _tag_poll(cls, context)

    def execute(self, context):
        return _change_membership(self, context, add=False)


class FEMBC_OT_SelectGroup(bpy.types.Operator):
    bl_idname = "fembc.select_group"
    bl_label = "Select Group Faces"
    bl_description = "Select exactly the faces in the active group (enters Edit Mode)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _tag_poll(cls, context, need_edit=False)

    def execute(self, context):
        import bmesh
        obj = context.active_object
        scene = context.scene
        _sync_group_names(scene)
        g = _active_group(scene)
        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.select_all(action='DESELECT')
        if _is_brep(obj):
            t = _tags(g)
            bm = bmesh.from_edit_mesh(obj.data)
            layer = bm.faces.layers.int.get("fembc_surface_tag")
            n = 0
            for f in bm.faces:
                if f[layer] in t:
                    f.select_set(True)
                    n += 1
            bm.select_flush_mode()
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        else:
            vg = obj.vertex_groups.get(g.name)
            if vg is not None:
                obj.vertex_groups.active_index = vg.index
                bpy.ops.object.vertex_group_select()
            n = sum(1 for f in bmesh.from_edit_mesh(obj.data).faces if f.select)
        if n == 0:
            self.report({'WARNING'}, f"'{g.name}' has no faces yet")
        else:
            self.report({'INFO'}, f"Selected {n} faces of '{g.name}'")
        return {'FINISHED'}


class FEMBC_OT_EditFaces(bpy.types.Operator):
    bl_idname = "fembc.edit_faces"
    bl_label = "Pick Faces"
    bl_description = ("Enter Edit Mode with face selection and nothing selected, ready to "
                      "click the faces for the active group")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            cls.poll_message_set("Select the part (a mesh object) first")
            return False
        return True

    def execute(self, context):
        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')
            # A fresh import enters Edit Mode fully selected; one Assign click would
            # then put the whole part in the group.
            bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='FACE')
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

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

def _redraw_view3d():
    import bpy as _bpy
    wm = _bpy.context.window_manager
    for window in (wm.windows if wm else []):
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def read_msh_surface_groups(path):
    """Names of the dim-2 physical groups in a Gmsh .msh ($PhysicalNames is ASCII
    even in binary files)."""
    import re
    names = []
    try:
        with open(path, "rb") as fh:
            inside = False
            for raw in fh:
                line = raw.strip()
                if line == b"$PhysicalNames":
                    inside = True
                    continue
                if line == b"$EndPhysicalNames" or (not inside and line.startswith(b"$Nodes")):
                    break
                m = re.match(rb'(\d+)\s+\d+\s+"(.*)"', line) if inside else None
                if m and m.group(1) == b"2":
                    names.append(m.group(2).decode(errors="replace"))
    except OSError:
        pass
    return names


def _log_tail(path, n=40):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()[-n:]
    except OSError:
        return []


def mesh_running():
    return _mesh_proc is not None


def mesh_elapsed():
    return time.time() - _mesh_t0 if _mesh_proc is not None and _mesh_t0 else 0.0


def _poll_mesh_subprocess():
    """Timer callback: poll _mesh_proc and update props on main thread.

    Returns 0.5 to reschedule while subprocess is running, or None to
    unregister once it completes.

    Uses 'import bpy as _bpy' inside the function body (not a closure over
    the execute() context argument) to avoid Pitfall 7: stale context
    reference causing ReferenceError after operator completes.
    """
    global _mesh_proc, _mesh_output_path, _mesh_log_fh
    import bpy as _bpy

    if _mesh_proc is None:
        return None  # nothing to poll

    retcode = _mesh_proc.poll()
    if retcode is None:
        _redraw_view3d()  # elapsed time in the panel
        return 0.5  # still running, reschedule

    if _mesh_log_fh is not None:
        _mesh_log_fh.close()
        _mesh_log_fh = None

    scene = _bpy.data.scenes.get(_mesh_scene) or _bpy.context.scene
    props = scene.fembc_main
    elapsed = time.time() - _mesh_t0 if _mesh_t0 else 0.0
    if retcode == 0:
        found = read_msh_surface_groups(_mesh_output_path)
        tagged = [n for n in found if n != "_untagged"]
        missing = [n for n in _mesh_expected if n not in found]
        props.mesh_status = (f"Mesh ready in {elapsed:.0f} s - groups: "
                             + (", ".join(tagged) if tagged else "none"))
        props.mesh_warning = (
            f"Not found in the mesh: {', '.join(missing)}. Their faces did not match the "
            "geometry - see the log." if missing else "")
        props.msh_output_path = _mesh_output_path
    else:
        tail = _log_tail(props.mesh_log_path)
        if tail:
            print("[BlendGmsh] Mesh generation failed:\n" + "\n".join(tail))
        short_err = next((l.strip() for l in reversed(tail) if l.strip()), "unknown error")
        props.mesh_status = f"Mesh generation failed: {short_err}"
        props.mesh_warning = ""
        props.msh_output_path = ""

    _mesh_proc = None
    _redraw_view3d()
    return None  # unregister timer


def _open_path(path):
    """Open a file or folder with the system's default application."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - local file chosen by the add-on
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


class FEMBC_OT_OpenMeshLog(bpy.types.Operator):
    bl_idname = "fembc.open_mesh_log"
    bl_label = "Open Log"
    bl_description = "Open the mesher's log (matching details, Gmsh errors)"

    @classmethod
    def poll(cls, context):
        p = context.scene.fembc_main.mesh_log_path
        return bool(p) and os.path.isfile(p)

    def execute(self, context):
        _open_path(context.scene.fembc_main.mesh_log_path)
        return {'FINISHED'}


class FEMBC_OT_OpenMeshFolder(bpy.types.Operator):
    bl_idname = "fembc.open_mesh_folder"
    bl_label = "Open Folder"
    bl_description = "Show the folder that holds the generated .msh"

    @classmethod
    def poll(cls, context):
        p = context.scene.fembc_main.msh_output_path
        return bool(p) and os.path.isfile(p)

    def execute(self, context):
        _open_path(os.path.dirname(context.scene.fembc_main.msh_output_path))
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Source geometry resolution
# ---------------------------------------------------------------------------

def _resolve_source_file(obj, props):
    """Return the geometry file the active object came from, or None.

    Imported objects carry a "fembc_source" custom property. Objects imported
    before that property existed fall back to the scene's last import path.
    Anything else (modelled in Blender) returns None.
    """
    src = obj.get("fembc_source")
    if src is None and obj.name.startswith("FEMBC_Import"):
        src = props.step_filepath or None
    if src and os.path.isfile(src):
        return src
    return None


def _export_native_object_stl(obj):
    """Write obj as a triangulated world-space binary STL in the temp dir.

    Returns (stl_path, triangles); triangles are vertex-index triples into
    obj.data.vertices, so BC groups are built with the same triangulation
    Gmsh will see (quad centroids would not match STL triangle centroids).
    """
    from .export import write_binary_stl

    mesh = obj.data
    mesh.calc_loop_triangles()
    triangles = [tuple(t.vertices) for t in mesh.loop_triangles]
    if not triangles:
        raise ValueError(f"Object '{obj.name}' has no faces to mesh.")

    mw = obj.matrix_world
    world_verts = [tuple(mw @ v.co) for v in mesh.vertices]

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in obj.name)
    stl_path = str(pathlib.Path(tempfile.gettempdir()) / f"{safe_name}.stl")
    write_binary_stl(stl_path, world_verts, triangles)
    return stl_path, triangles


# ---------------------------------------------------------------------------
# Generate Mesh operator
# ---------------------------------------------------------------------------

def generate_blocker(context):
    """Why Generate Mesh can't run right now (shown under the button), or ''."""
    obj = context.active_object
    scene = context.scene
    if mesh_running():
        return "A mesh is already being generated"
    if obj is None or obj.type != 'MESH':
        return "Select the part to mesh"
    if "fea_job" in obj:
        return f"'{obj.name}' is an FEA result - select the original part"
    tagged_on = scene.fembc_main.groups_object
    if tagged_on and tagged_on != obj.name and tagged_on in bpy.data.objects:
        return f"The groups were assigned on '{tagged_on}' - select that object"
    if not scene.fembc_groups:
        return "Add a face group (+) and assign faces to it"
    if not any(group_count(g) > 0 or (group_count(g) < 0 and obj.vertex_groups.get(g.name))
               for g in scene.fembc_groups):
        return "No faces assigned yet - select faces, then Assign"
    return ""


class FEMBC_OT_GenerateMesh(bpy.types.Operator):
    bl_idname = "fembc.generate_mesh"
    bl_label = "Generate Mesh"
    bl_description = ("Build a tetrahedral .msh with one named surface group per face group "
                      "(runs in the background)")

    @classmethod
    def poll(cls, context):
        why = generate_blocker(context)
        if why:
            cls.poll_message_set(why)
        return not why

    def execute(self, context):
        global _mesh_proc, _mesh_output_path, _mesh_log_fh, _mesh_scene, _mesh_expected, _mesh_t0
        props = context.scene.fembc_main
        obj = context.active_object
        scene = context.scene

        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected.")
            return {'CANCELLED'}
        _sync_group_names(scene)

        # Flush Edit Mode changes (vertex groups) to obj.data.
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        source_path = _resolve_source_file(obj, props)
        is_brep = (
            source_path is not None
            and source_path.lower().endswith((".step", ".stp"))
            and obj.data.attributes.get("fembc_surface_tag") is not None
        )

        if source_path is None:
            # Object modelled in Blender: no file on disk. Write it out as a
            # triangulated world-space STL and use the STL/OBJ pipeline.
            try:
                source_path, triangles = _export_native_object_stl(obj)
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            from .export import build_bc_groups_dict
            bc_dict = build_bc_groups_dict(
                obj=obj,
                groups=scene.fembc_groups,
                step_filepath=source_path,
                blender_version=bpy.app.version_string,
                triangles=triangles,
            )
            bc_dict["mode"] = "mesh"
        elif is_brep:
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
                "step_file": source_path,
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
                step_filepath=source_path,
                blender_version=bpy.app.version_string,
            )

        if not bc_dict.get("groups"):
            self.report({'ERROR'}, "No faces assigned to any group on the active object.")
            return {'CANCELLED'}
        skipped = [g.name for g in scene.fembc_groups if g.name not in bc_dict["groups"]]
        if skipped:
            self.report({'WARNING'}, f"Empty groups left out: {', '.join(skipped)}")

        # Compute output path: /tmp/<stem>.msh (per MESH-04)
        stem = pathlib.Path(source_path).stem
        output_msh = str(pathlib.Path(tempfile.gettempdir()) / f"{stem}.msh")

        # Write temp JSON for run_full_pipeline() which takes a file path
        tmp_json = str(pathlib.Path(tempfile.gettempdir()) / "fembc_live_bc_groups.json")
        with open(tmp_json, "w") as f:
            json.dump(bc_dict, f)

        # Signal "Generating mesh..." before subprocess starts (per STAT-03)
        props.mesh_status = "Generating mesh..."
        props.mesh_warning = ""
        props.msh_output_path = ""
        _mesh_t0 = time.time()
        props.mesh_log_path = str(pathlib.Path(tempfile.gettempdir()) / f"{stem}.mesh.log")
        props.groups_object = obj.name

        # Use subprocess instead of thread: gmsh.initialize() calls
        # signal.signal() which is main-thread-only, so Gmsh cannot run
        # in a background thread. The CLI entry point
        # `python -m matching_library run` is already available.
        # Output goes to a log file: an unread PIPE can fill up and stall the mesher.
        try:
            lib = lib_dir()
        except RuntimeError as exc:
            props.mesh_status = f"Mesh generation failed: {exc}"
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        _mesh_output_path = output_msh
        _mesh_scene = scene.name
        _mesh_expected = list(bc_dict["groups"])
        _mesh_log_fh = open(props.mesh_log_path, "w", encoding="utf-8")
        _mesh_proc = subprocess.Popen(
            [sys.executable, "-m", "matching_library", "run",
             tmp_json, source_path, output_msh],
            env={**os.environ, "PYTHONPATH": lib, "PYTHONUNBUFFERED": "1"},
            stdout=_mesh_log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Register timer to poll subprocess on main thread (per MESH-03)
        bpy.app.timers.register(_poll_mesh_subprocess, first_interval=0.5)
        return {'FINISHED'}
