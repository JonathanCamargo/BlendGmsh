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


class FEMBCGroupItem(bpy.types.PropertyGroup):
    # name is inherited from PropertyGroup -- used by UIList for display and ctrl-click rename
    color: bpy.props.FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        size=3,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0),
    )
    surface_tags: bpy.props.StringProperty(
        name="Surface Tags",
        description="Cached JSON list of BREP surface tags assigned to this group",
        default="",
    )


# Registered on Scene in __init__.py:
# bpy.types.Scene.fembc_groups = bpy.props.CollectionProperty(type=FEMBCGroupItem)
# bpy.types.Scene.fembc_groups_active_index = bpy.props.IntProperty(default=0)
