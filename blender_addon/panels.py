"""
BlendGmsh addon panel.

Implements FEMBC_PT_MainPanel per the UI-SPEC Panel Layout Contract.
All copy strings are taken verbatim from the UI-SPEC Copywriting Contract.
"""
import textwrap

import bpy  # type: ignore  # available inside Blender runtime


class FEMBC_UL_GroupList(bpy.types.UIList):
    bl_idname = "FEMBC_UL_group_list"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "color", text="", emboss=True)
            row.prop(item, "name", text="", emboss=False, icon_value=icon)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)


class FEMBC_PT_MainPanel(bpy.types.Panel):
    bl_label = "FEM BC"
    bl_category = "FEM BC"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        props = context.scene.fembc_main

        # --- Error / warning alert box ---
        if props.last_error:
            box = layout.box()
            box.alert = True
            box.label(text="Warning:", icon='ERROR')
            for line in textwrap.wrap(props.last_error, width=80):
                box.label(text=line)

        # --- Import section ---
        layout.label(text="Import STEP")
        layout.operator("fembc.import_step", text="Import STEP File", icon='IMPORT')
        layout.operator("fembc.import_mesh", text="Import STL/OBJ", icon='IMPORT')

        layout.separator(factor=2.0)

        # --- Status section ---
        layout.label(text="Status")
        face_count = props.face_count
        bc_mode = props.bc_mode if props.bc_mode else self._detect_mode(context)

        if face_count == -1:
            layout.label(text="No geometry imported", icon='INFO')
        elif face_count == 0:
            layout.label(text="Converting...", icon='TIME')
        else:
            if bc_mode == "brep":
                layout.label(text=f"Ready: {face_count} BREP surfaces", icon='CHECKMARK')
                box = layout.box()
                box.label(text="BREP mode (STEP import)", icon='MESH_ICOSPHERE')
                box.label(text="Selection maps to whole CAD surfaces.")
                box.label(text="Select any triangle on a surface to")
                box.label(text="include the entire BREP face in the group.")
            elif bc_mode == "mesh":
                layout.label(text=f"Ready: {face_count} faces", icon='CHECKMARK')
                box = layout.box()
                box.label(text="Mesh mode (STL/OBJ)", icon='MESH_GRID')
                box.label(text="Selection maps to individual triangles.")
                box.label(text="Only selected triangles are included.")
            else:
                layout.label(text=f"Ready: {face_count} faces detected", icon='CHECKMARK')

        # --- Face Groups section ---
        layout.separator()
        layout.label(text="Face Groups", icon='GROUP_VERTEX')
        box = layout.box()
        box.template_list(
            "FEMBC_UL_group_list", "",
            context.scene, "fembc_groups",
            context.scene, "fembc_groups_active_index",
            rows=3,
        )
        row = box.row(align=True)
        row.operator("fembc.add_group", text="", icon='ADD')
        row.operator("fembc.remove_group", text="", icon='REMOVE')

        layout.separator()
        if bc_mode == "brep":
            layout.operator("fembc.assign_faces",
                            text="Assign Selected Faces (BREP surfaces)")
        else:
            layout.operator("fembc.assign_faces",
                            text="Assign Selected Faces")
        layout.separator()

        # --- Export section ---
        layout.operator("fembc.export_json", text="Export to JSON")

        # --- Generate Mesh section ---
        layout.separator()
        layout.label(text="Generate Mesh")
        layout.operator("fembc.generate_mesh", text="Generate Mesh", icon='MOD_SOLIDIFY')
        if props.mesh_status:
            layout.label(text=props.mesh_status, icon='INFO')
        if props.msh_output_path:
            for line in textwrap.wrap(props.msh_output_path, width=40):
                layout.label(text=line)

    def _detect_mode(self, context):
        """Detect BREP vs mesh mode from the active object's vertex groups."""
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            return ""
        import re
        for vg in obj.vertex_groups:
            if re.match(r"^face_\d+$", vg.name):
                return "brep"
        # If we have vertex groups but none are face_N, it's mesh mode
        if len(obj.vertex_groups) > 0:
            return "mesh"
        return ""
