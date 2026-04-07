"""
Unit tests for Generate Mesh operator output path logic.

Tests the pure-Python path computation that FEMBC_OT_GenerateMesh uses
to determine where to write the .msh file. No bpy required.

Per project guidelines (MEMORY.md "Don't simulate Blender"), we do NOT
mock bpy context here. The build_bc_groups_dict function is already
thoroughly tested in tests/test_export.py.
"""
import pathlib
import tempfile


def compute_output_path(step_filepath: str) -> str:
    """Compute the output .msh path from a STEP/STP filepath.

    Logic mirrors FEMBC_OT_GenerateMesh.execute():
        stem = pathlib.Path(props.step_filepath).stem
        output_msh = str(pathlib.Path(tempfile.gettempdir()) / f"{stem}.msh")
    """
    stem = pathlib.Path(step_filepath).stem
    return str(pathlib.Path(tempfile.gettempdir()) / f"{stem}.msh")


def test_output_path_construction():
    """Given step_filepath='/home/user/bracket.step', output path ends with '/bracket.msh'."""
    result = compute_output_path("/home/user/bracket.step")
    assert result.startswith(tempfile.gettempdir()), (
        f"Expected path to start with {tempfile.gettempdir()!r}, got {result!r}"
    )
    assert result.endswith("/bracket.msh"), (
        f"Expected path to end with '/bracket.msh', got {result!r}"
    )


def test_output_path_construction_nested():
    """Given step_filepath='/deep/path/model.stp', output path stem is 'model' not 'model.stp'."""
    result = compute_output_path("/deep/path/model.stp")
    assert result.endswith("/model.msh"), (
        f"Expected path to end with '/model.msh', got {result!r}"
    )
    # Ensure double-extension trap is avoided: 'model.stp.msh' would be wrong
    assert "model.stp" not in result, (
        f"Stem should not include .stp extension, got {result!r}"
    )


def test_output_path_construction_spaces():
    """Given step_filepath='/path/my model.step', output path contains 'my model.msh'."""
    result = compute_output_path("/path/my model.step")
    assert "my model.msh" in result, (
        f"Expected 'my model.msh' in output path, got {result!r}"
    )
