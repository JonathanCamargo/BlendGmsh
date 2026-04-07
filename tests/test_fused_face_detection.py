"""
Tests for CONV-03: fused-face detection by comparing expected vs actual face counts.
"""

from step_converter import check_fused_faces


def test_warning_on_fusion():
    """When actual count < expected, detect fused-face condition."""
    warnings = check_fused_faces(expected=6, actual=4)
    assert len(warnings) > 0, "Expected at least one warning for fused-face condition"
    combined = " ".join(warnings)
    assert "Fused face condition detected" in combined, (
        f"Expected 'Fused face condition detected' in warnings, got: {warnings}"
    )


def test_no_warning_clean_geometry():
    """When actual count == expected, return empty list."""
    warnings = check_fused_faces(expected=6, actual=6)
    assert warnings == [], f"Expected empty list for clean geometry, got: {warnings}"


def test_warning_on_extra_islands():
    """When actual count > expected, warn about extra islands."""
    warnings = check_fused_faces(expected=6, actual=8)
    assert len(warnings) > 0, "Expected at least one warning for extra islands"
    combined = " ".join(warnings)
    assert "More islands than expected" in combined, (
        f"Expected 'More islands than expected' in warnings, got: {warnings}"
    )


def test_warning_message_contains_counts():
    """Warning message for fused faces must reference both expected and actual counts."""
    warnings = check_fused_faces(expected=6, actual=4)
    combined = " ".join(warnings)
    assert "6" in combined, f"Expected count '6' not found in warning: {combined}"
    assert "4" in combined, f"Actual count '4' not found in warning: {combined}"
