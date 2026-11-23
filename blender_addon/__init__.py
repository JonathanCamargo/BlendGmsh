"""
BlendGmsh addon entry point.

register() / unregister() are called by Blender when the addon is
enabled or disabled.

Registration order:
  1. FEMBCMainProps (PropertyGroup)
  2. FEMBCGroupItem (PropertyGroup -- must be registered before CollectionProperty)
  3. FEMBC_OT_ImportSTEP (Operator)
  4. FEMBC_OT_AddGroup (Operator)
  5. FEMBC_OT_RemoveGroup (Operator)
  6. FEMBC_OT_AssignFaces (Operator)
  7. FEMBC_OT_ExportJSON (Operator)
  8. FEMBC_OT_GenerateMesh (Operator -- Phase 7)
  9. FEMBC_UL_GroupList (UIList)
  10. FEMBC_PT_MainPanel (Panel -- after all PropertyGroups, Operators, and UIList)
  11. Scene properties: fembc_main, fembc_groups, fembc_groups_active_index

Unregistration is the strict reverse of registration.
"""
from __future__ import annotations

try:
    import bpy  # type: ignore  # available inside Blender runtime
    _BPY_AVAILABLE = True
except ModuleNotFoundError:
    _BPY_AVAILABLE = False


def _ensure_dependencies():
    """Install missing Python dependencies into Blender's Python."""
    import importlib.util
    import subprocess
    import sys

    missing = []
    for pkg_import, pkg_pip in [
        ("gmsh", "gmsh"),
        ("scipy", "scipy"),
        ("jsonschema", "jsonschema"),
        ("numpy", "numpy"),
    ]:
        if importlib.util.find_spec(pkg_import) is None:
            missing.append(pkg_pip)

    if missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing],
        )

if _BPY_AVAILABLE:
    from .properties import FEMBCMainProps, FEMBCGroupItem
    from .operators import (
        FEMBC_OT_ImportSTEP,
        FEMBC_OT_ImportMesh,
        FEMBC_OT_AddGroup,
        FEMBC_OT_RemoveGroup,
        FEMBC_OT_AssignFaces,
        FEMBC_OT_UnassignFaces,
        FEMBC_OT_SelectGroup,
        FEMBC_OT_EditFaces,
        FEMBC_OT_ExportJSON,
        FEMBC_OT_GenerateMesh,   # NEW Phase 7
        FEMBC_OT_OpenMeshLog,
        FEMBC_OT_OpenMeshFolder,
    )
    from .panels import FEMBC_UL_GroupList, FEMBC_PT_MainPanel

    classes = (
        FEMBCMainProps,
        FEMBCGroupItem,
        FEMBC_OT_ImportSTEP,
        FEMBC_OT_ImportMesh,
        FEMBC_OT_AddGroup,
        FEMBC_OT_RemoveGroup,
        FEMBC_OT_AssignFaces,
        FEMBC_OT_UnassignFaces,
        FEMBC_OT_SelectGroup,
        FEMBC_OT_EditFaces,
        FEMBC_OT_ExportJSON,
        FEMBC_OT_GenerateMesh,   # NEW Phase 7
        FEMBC_OT_OpenMeshLog,
        FEMBC_OT_OpenMeshFolder,
        FEMBC_UL_GroupList,
        FEMBC_PT_MainPanel,
    )

    @bpy.app.handlers.persistent
    def _clear_stale_mesh_status(_dummy):
        """A .blend saved mid-run still says 'Generating mesh...' with no process behind it."""
        for scene in bpy.data.scenes:
            p = getattr(scene, "fembc_main", None)
            if p is not None and p.mesh_status == "Generating mesh...":
                p.mesh_status = "Mesh generation was interrupted - generate again"
else:
    classes = ()  # type: ignore[assignment]


def register():
    if not _BPY_AVAILABLE:
        return
    import bpy  # noqa: F811 -- re-import inside function for clarity
    from bpy.utils import register_class

    _ensure_dependencies()

    for cls in classes:
        register_class(cls)

    bpy.types.Scene.fembc_main = bpy.props.PointerProperty(type=FEMBCMainProps)
    bpy.types.Scene.fembc_groups = bpy.props.CollectionProperty(type=FEMBCGroupItem)
    bpy.types.Scene.fembc_groups_active_index = bpy.props.IntProperty(default=0)
    if _clear_stale_mesh_status not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_clear_stale_mesh_status)


def unregister():
    if not _BPY_AVAILABLE:
        return
    import bpy  # noqa: F811
    from bpy.utils import unregister_class

    if _clear_stale_mesh_status in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_clear_stale_mesh_status)
    if hasattr(bpy.types.Scene, "fembc_groups_active_index"):
        del bpy.types.Scene.fembc_groups_active_index
    if hasattr(bpy.types.Scene, "fembc_groups"):
        del bpy.types.Scene.fembc_groups
    if hasattr(bpy.types.Scene, "fembc_main"):
        del bpy.types.Scene.fembc_main

    for cls in reversed(classes):
        unregister_class(cls)
