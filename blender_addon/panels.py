"""
BlendGmsh addon panel.

One panel, three numbered steps (part -> face groups -> volume mesh), each with a
done/pending marker, so the next action is always visible. It lives in the "FEA"
sidebar tab, above the FEniCSx add-on's panels when that is installed.
"""
import os
import textwrap

import bpy  # type: ignore  # available inside Blender runtime

from . import operators as ops


def _wrap(layout, text, width=42, icon='NONE', alert=False, active=True):
    col = layout.column(align=True)
    col.alert = alert
    col.active = active
    for i, line in enumerate(textwrap.wrap(text, width) or [""]):
        col.label(text=line, icon=icon if i == 0 else ('BLANK1' if icon != 'NONE' else 'NONE'))
    return col


def _step(layout, number, title, done):
    box = layout.box()
    box.label(text=f"{number}. {title}", icon='CHECKMARK' if done else 'RADIOBUT_OFF')
    return box


class FEMBC_UL_GroupList(bpy.types.UIList):
    bl_idname = "FEMBC_UL_group_list"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "color", text="", emboss=True)
            row.prop(item, "name", text="", emboss=False)
            n = ops.group_count(item)
            sub = row.row()
            sub.alignment = 'RIGHT'
            if n == 0:
                sub.alert = True
                sub.label(text="empty")
            elif n > 0:
                unit = "surf" if ops._is_brep(context.active_object) else "faces"
                sub.label(text=f"{n} {unit}")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)


class FEMBC_PT_MainPanel(bpy.types.Panel):
    bl_label = "Mesh & Face Groups"
    bl_category = "FEA"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.fembc_main
        obj = context.active_object
        is_mesh = obj is not None and obj.type == 'MESH'
        is_result = is_mesh and "fea_job" in obj
        brep = ops._is_brep(obj)

        if props.last_error:
            box = layout.box()
            _wrap(box, props.last_error, icon='ERROR', alert=True)

        # --- 1. Part --------------------------------------------------------
        box = _step(layout, 1, "Part", is_mesh and not is_result)
        row = box.row(align=True)
        row.operator("fembc.import_step", text="STEP", icon='IMPORT')
        row.operator("fembc.import_mesh", text="STL / OBJ", icon='IMPORT')
        if not is_mesh:
            _wrap(box, "Import a part, or select a mesh object", icon='INFO')
        elif is_result:
            _wrap(box, f"'{obj.name}' is an FEA result. Select the original part.",
                  icon='ERROR', alert=True)
        else:
            if brep:
                kind = (f"{props.face_count} CAD surfaces" if props.face_count > 0
                        else "CAD part (STEP)")
            elif obj.get("fembc_source"):
                kind = f"{len(obj.data.polygons)} triangles (STL/OBJ)"
            else:
                kind = "Blender object"
            box.label(text=f"{obj.name} · {kind}",
                      icon='MESH_ICOSPHERE' if brep else 'MESH_GRID')

        # --- 2. Face groups -------------------------------------------------
        groups = scene.fembc_groups
        counts = [ops.group_count(g) for g in groups]
        box = _step(layout, 2, "Face groups", any(c > 0 for c in counts))
        row = box.row()
        row.template_list("FEMBC_UL_group_list", "", scene, "fembc_groups",
                          scene, "fembc_groups_active_index", rows=3)
        col = row.column(align=True)
        col.operator("fembc.add_group", text="", icon='ADD')
        col.operator("fembc.remove_group", text="", icon='REMOVE')

        tagged_on = props.groups_object
        if is_mesh and tagged_on and tagged_on != obj.name and tagged_on in bpy.data.objects:
            _wrap(box, f"These groups belong to '{tagged_on}'. Select it to edit them.",
                  icon='ERROR', alert=True)

        r = box.row(align=True)
        r.scale_y = 1.3
        if context.mode == 'EDIT_MESH':
            r.operator("fembc.assign_faces", text="Assign", icon='ADD')
            r.operator("fembc.unassign_faces", text="Remove", icon='REMOVE')
            r.operator("fembc.select_group", text="Select", icon='RESTRICT_SELECT_OFF')
            _wrap(box, "Clicking one triangle picks the whole CAD face" if brep
                  else "Only the selected faces are used", icon='INFO', active=False)
        else:
            r.operator("fembc.edit_faces", text="Pick Faces", icon='EDITMODE_HLT')
            r.operator("fembc.select_group", text="Show Group", icon='RESTRICT_SELECT_OFF')

        if not groups:
            _wrap(box, "Add a group (+) for each support and each load",
                  icon='INFO', active=False)
        elif not any(c > 0 for c in counts):
            _wrap(box, "Pick faces, then Assign them to the highlighted group. "
                       "Grey faces are in no group", icon='INFO', active=False)
        else:
            _wrap(box, "Double-click a name to rename. Grey faces stay free in the analysis",
                  icon='INFO', active=False)

        # --- 3. Volume mesh -------------------------------------------------
        have_msh = bool(props.msh_output_path) and os.path.isfile(props.msh_output_path)
        running = ops.mesh_running()
        box = _step(layout, 3, "Volume mesh", have_msh and not running)
        r = box.row()
        r.scale_y = 1.4
        r.operator("fembc.generate_mesh", text="Generate Mesh", icon='MOD_SOLIDIFY')
        if running:
            box.label(text=f"Generating mesh ... {ops.mesh_elapsed():.0f} s", icon='TIME')
        else:
            why = ops.generate_blocker(context)
            if why:
                _wrap(box, why, icon='INFO', active=False)
            if props.mesh_status and props.mesh_status != "Generating mesh...":
                failed = "failed" in props.mesh_status
                _wrap(box, props.mesh_status, icon='ERROR' if failed else 'CHECKMARK',
                      alert=failed)
            if props.mesh_warning:
                _wrap(box, props.mesh_warning, icon='ERROR', alert=True)
        if have_msh or props.mesh_log_path:
            r = box.row(align=True)
            r.label(text=os.path.basename(props.msh_output_path) if have_msh else "Log",
                    icon='FILE' if have_msh else 'TEXT')
            r.operator("fembc.open_mesh_folder", text="", icon='FILE_FOLDER')
            r.operator("fembc.open_mesh_log", text="", icon='TEXT')
        if have_msh and not running and hasattr(scene, "fx_cae"):
            _wrap(box, "Next: supports & loads in the FEniCSx Job panel below",
                  icon='FORWARD', active=False)

        # --- More -----------------------------------------------------------
        layout.prop(props, "show_more", emboss=False,
                    icon='DISCLOSURE_TRI_DOWN' if props.show_more else 'DISCLOSURE_TRI_RIGHT')
        if props.show_more:
            layout.operator("fembc.export_json", text="Export Groups to JSON", icon='EXPORT')
