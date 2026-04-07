"""
Generate programmatic STEP test geometry using Gmsh OCC kernel.

Provides:
  make_box_step(output_path)      -> 6 BREP surfaces
  make_cylinder_step(output_path) -> 3 BREP surfaces

Run directly to produce examples/box.step and examples/cylinder.step.
"""

import gmsh
import os


def make_box_step(output_path: str) -> None:
    """Generate a unit box STEP file with 6 BREP surfaces."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("box")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.write(output_path)
    gmsh.finalize()


def make_cylinder_step(output_path: str) -> None:
    """Generate a cylinder STEP file with 3 BREP surfaces (top cap, bottom cap, lateral)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("cylinder")
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 1, 0.5)
    gmsh.model.occ.synchronize()
    gmsh.write(output_path)
    gmsh.finalize()


if __name__ == "__main__":
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    box_path = os.path.join(examples_dir, "box.step")
    cylinder_path = os.path.join(examples_dir, "cylinder.step")

    print(f"Generating {box_path} ...")
    make_box_step(box_path)
    print(f"  -> {os.path.getsize(box_path)} bytes")

    print(f"Generating {cylinder_path} ...")
    make_cylinder_step(cylinder_path)
    print(f"  -> {os.path.getsize(cylinder_path)} bytes")

    print("Done.")
