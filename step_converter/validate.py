"""
Fused-face detection for STEP tessellation validation.

Compares the expected BREP face count (from Gmsh) against the actual island count
(from Blender or STL parser) and emits warnings when they differ.

Behavior: warn and continue, NOT abort (per CONV-03).
"""


def check_fused_faces(expected: int, actual: int) -> list[str]:
    """
    Compare expected BREP surface count against actual mesh island count.

    Parameters
    ----------
    expected:
        Number of BREP surfaces reported by Gmsh (``len(gmsh.model.getEntities(2))``).
    actual:
        Number of disconnected mesh islands found after STL import.

    Returns
    -------
    list[str]
        Empty list when counts match (clean geometry).
        One or more warning strings when counts differ.

    Notes
    -----
    This function warns and continues -- it does NOT raise an exception.
    Callers should log the returned list and proceed with the available data.
    """
    warnings: list[str] = []

    if actual < expected:
        warnings.append(
            f"Fused face condition detected: expected {expected} surfaces "
            f"but only {actual} mesh islands found. "
            "Re-export from CAD without face merging."
        )

    elif actual > expected:
        warnings.append(
            f"More islands than expected: expected {expected} surfaces "
            f"but found {actual} mesh islands. "
            "Check for duplicate or split faces in the STEP file."
        )

    return warnings
