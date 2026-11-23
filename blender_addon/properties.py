"""
BlendGmsh addon property group.

Stores import state that the panel reads for status display.
Attached to bpy.types.Scene as scene.fembc_main by the register() call in __init__.py.

FEMBCGroupItem is the data model for named face groups with a color property.
"""
import bpy  # type: ignore  # available inside Blender runtime


class FEMBCMainProps(bpy.types.PropertyGroup):
    last_error: bpy.props.StringProperty(
        name="Last Error",
        description="Most recent error/warning message",
        default="",
    )
    face_count: bpy.props.IntProperty(
        name="Face Count",
        description="Number of faces detected after conversion",
        default=-1,
    )
    step_filepath: bpy.props.StringProperty(
        name="STEP File",
        description="Path of most recently imported geometry file",
        subtype='FILE_PATH',
        default="",
    )
    bc_mode: bpy.props.StringProperty(
        name="BC Mode",
        description="brep = STEP import (tags whole CAD surfaces), mesh = STL/OBJ (tags individual triangles)",
        default="",
    )
    msh_output_path: bpy.props.StringProperty(
        name="MSH Output Path",
        description="Path of the most recently generated .msh file",
        subtype='FILE_PATH',
        default="",
    )
    mesh_status: bpy.props.StringProperty(
        name="Mesh Status",
        description="Status of the last mesh generation attempt",
        default="",
    )
    mesh_warning: bpy.props.StringProperty(
        name="Mesh Warning",
        description="Problems found when checking the generated .msh (e.g. groups that did not match)",
        default="",
    )
    mesh_log_path: bpy.props.StringProperty(
        name="Mesh Log",
        description="Log file of the last mesh generation run",
        subtype='FILE_PATH',
        default="",
    )
    groups_object: bpy.props.StringProperty(
        name="Tagged Object",
        description="Object the face groups were last assigned on",
        default="",
    )
    show_more: bpy.props.BoolProperty(name="More", default=False)


def _color_changed(self, context):
    mat = bpy.data.materials.get(f"FEMBC_{self.name}")
    if mat is not None:
        mat.diffuse_color = (*self.color, 1.0)


class FEMBCGroupItem(bpy.types.PropertyGroup):
    # name is inherited from PropertyGroup -- used by UIList for display and double-click rename.
    # Renames are propagated to the vertex group / material by operators._sync_group_names().
    color: bpy.props.FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        size=3,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0),
        update=_color_changed,
    )
    surface_tags: bpy.props.StringProperty(
        name="Surface Tags",
        description="Cached JSON list of BREP surface tags assigned to this group",
        default="",
    )
    assigned: bpy.props.IntProperty(
        name="Assigned",
        description="CAD surfaces (BREP) or faces (mesh) in this group; -1 = not counted yet",
        default=-1,
    )
    prev_name: bpy.props.StringProperty(
        description="Name the vertex group / material currently use (rename tracking)",
        default="",
    )


# Registered on Scene in __init__.py:
# bpy.types.Scene.fembc_groups = bpy.props.CollectionProperty(type=FEMBCGroupItem)
# bpy.types.Scene.fembc_groups_active_index = bpy.props.IntProperty(default=0)
