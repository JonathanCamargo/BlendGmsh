"""
CLI entry point for the matching library.

Usage:
  python -m matching_library run <json> <step> <output>
  python -m matching_library tag <json> <input_msh> <output>
  python -m matching_library inspect <msh_file>
  python -m matching_library inspect-bc <json> <step>
  python -m matching_library visualize <json> <msh> [--output <png>]

Subcommands:
  run          Full pipeline: STEP file + bc_groups JSON -> tagged .msh v4
  tag          Tagging-only: existing .msh + bc_groups JSON -> re-tagged .msh v4
  inspect      Inspect a .msh file: element counts, physical groups, surfaces
  inspect-bc   Cross-reference bc_groups.json surface tags against STEP geometry
  visualize    Visualize BC group surfaces on the mesh with PyVista
"""

import argparse
import sys

from matching_library import run_full_pipeline, tag_existing_mesh
from matching_library.debug import inspect_msh, inspect_bc_groups, visualize_bc_groups


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m matching_library",
        description=(
            "BlendGmsh: assign Blender face selections as Gmsh physical groups"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="subcommand", metavar="{run,tag,inspect,inspect-bc,visualize}"
    )

    # ---- run subcommand ----
    run_parser = subparsers.add_parser(
        "run",
        help="Full pipeline: STEP file + bc_groups JSON -> tagged .msh v4",
        description=(
            "Import a STEP file, assign physical groups by BREP surface tag, "
            "generate tet mesh, and write a tagged .msh v4 file."
        ),
    )
    run_parser.add_argument("json", help="Path to bc_groups JSON file")
    run_parser.add_argument("step", help="Path to the STEP file to mesh")
    run_parser.add_argument("output", help="Path for the output .msh file")

    # ---- tag subcommand ----
    tag_parser = subparsers.add_parser(
        "tag",
        help="Tagging-only: existing .msh + bc_groups JSON -> re-tagged .msh v4",
        description=(
            "Load an existing .msh file, assign physical groups by BREP surface tag, "
            "and write a re-tagged .msh v4 file."
        ),
    )
    tag_parser.add_argument("json", help="Path to bc_groups JSON file")
    tag_parser.add_argument("input_msh", help="Path to an existing .msh file to re-tag")
    tag_parser.add_argument("output", help="Path for the output .msh file")

    # ---- inspect subcommand ----
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a .msh file: element counts, physical groups, surfaces",
    )
    inspect_parser.add_argument("msh", help="Path to a .msh file")

    # ---- inspect-bc subcommand ----
    inspect_bc_parser = subparsers.add_parser(
        "inspect-bc",
        help="Cross-reference bc_groups.json against STEP geometry",
    )
    inspect_bc_parser.add_argument("json", help="Path to bc_groups JSON file")
    inspect_bc_parser.add_argument("step", help="Path to the STEP file")

    # ---- visualize subcommand ----
    vis_parser = subparsers.add_parser(
        "visualize",
        help="Visualize BC group surfaces on the mesh with PyVista",
    )
    vis_parser.add_argument("json", help="Path to bc_groups JSON file")
    vis_parser.add_argument("msh", help="Path to a .msh file")
    vis_parser.add_argument(
        "--output", "-o", default=None,
        help="Path for output PNG (interactive window if omitted)",
    )

    return parser


def main() -> int:
    """Parse arguments, dispatch to the appropriate pipeline function, return exit code."""
    import jsonschema

    parser = _build_parser()
    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        return 1

    try:
        if args.subcommand == "run":
            run_full_pipeline(args.json, args.step, args.output)
            print(f"Output written to {args.output}")
        elif args.subcommand == "tag":
            tag_existing_mesh(args.json, args.input_msh, args.output)
            print(f"Output written to {args.output}")
        elif args.subcommand == "inspect":
            inspect_msh(args.msh)
        elif args.subcommand == "inspect-bc":
            inspect_bc_groups(args.json, args.step)
        elif args.subcommand == "visualize":
            visualize_bc_groups(args.json, args.msh, args.output)
        return 0
    except jsonschema.ValidationError as e:
        print(f"Error: invalid JSON: {e.message}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: file not found: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
