"""
Demo: end-to-end visualization of a tagged mesh.

Usage (from project root):
    python examples/demo_visualization.py

Produces:
    output_tagged.png  -- categorical mesh coloring (VIZ-01 + VIZ-02)
    comparison_top.png -- side-by-side Blender vs matched (VIZ-03)
"""

import json
import os
import sys
import tempfile

# Ensure project root is on sys.path so local packages are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pyvista as pv

from matching_library import run_full_pipeline
from tests.test_pipeline import _extract_face_verts_by_z, _make_synthetic_bc_groups
from visualization import (
    visualize_tagged_mesh,
    read_tagged_msh,
    load_group_point_cloud,
    plot_comparison,
)

STEP_FILE = os.path.join(os.path.dirname(__file__), "box.step")


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = type("P", (), {"__truediv__": lambda s, n: os.path.join(tmp_dir, n)})()

        print("Step 1: Extracting surface vertices from box.step...")
        top_verts, top_tris, bot_verts, bot_tris = _extract_face_verts_by_z(STEP_FILE)

        print("Step 2: Creating bc_groups JSON (same strategy as tests)...")
        json_path = _make_synthetic_bc_groups(
            STEP_FILE,
            {"top": (top_verts, top_tris), "bottom": (bot_verts, bot_tris)},
            tmp_path,
        )

        print("Step 3: Running full pipeline (STEP -> tagged .msh)...")
        output_msh = os.path.join(tmp_dir, "tagged.msh")
        report = run_full_pipeline(json_path, STEP_FILE, output_msh)
        print(f"  Pipeline complete. Coverage: {report}")

        print("Step 4: Visualizing tagged mesh (VIZ-01 + VIZ-02)...")
        visualize_tagged_mesh(output_msh, "output_tagged.png")
        print(f"  Saved: output_tagged.png ({os.path.getsize('output_tagged.png')} bytes)")

        print("Step 5: Creating side-by-side comparison for 'top' group (VIZ-03)...")
        blender_points = load_group_point_cloud(json_path, "top")
        groups = read_tagged_msh(output_msh)
        top_data = groups["top"]
        tris = top_data["tris"].astype(np.int32)
        padded = np.hstack([np.full((len(tris), 1), 3, dtype=np.int32), tris]).ravel()
        matched_mesh = pv.PolyData(top_data["verts"], padded)
        plot_comparison(blender_points, matched_mesh, "top", output_png="comparison_top.png")
        print(f"  Saved: comparison_top.png ({os.path.getsize('comparison_top.png')} bytes)")

        print("\nAll done! Open the PNG files to inspect.")


if __name__ == "__main__":
    main()
