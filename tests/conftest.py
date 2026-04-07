"""
Shared pytest fixtures providing programmatic STEP test geometry.
"""

import os
import sys
import pytest

# Ensure examples/ is importable regardless of how pytest is invoked
_EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from generate_test_geometry import make_box_step, make_cylinder_step


@pytest.fixture
def tmp_step_box(tmp_path):
    """Generate a unit box STEP file (6 BREP surfaces) in a temp directory."""
    step_path = str(tmp_path / "box.step")
    make_box_step(step_path)
    return step_path


@pytest.fixture
def tmp_step_cylinder(tmp_path):
    """Generate a cylinder STEP file (3 BREP surfaces) in a temp directory."""
    step_path = str(tmp_path / "cylinder.step")
    make_cylinder_step(step_path)
    return step_path
