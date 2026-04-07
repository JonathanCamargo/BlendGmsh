"""
Tests for bc_groups_v1.json schema validation.
Validates that the JSON schema accepts valid data and rejects invalid data.
"""
import json
import pathlib

import jsonschema
import pytest

SCHEMA_PATH = pathlib.Path(__file__).parent.parent / "schema" / "bc_groups_v1.json"


def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def make_valid_bc_groups():
    return {
        "schema_version": 1,
        "source": "blendgmsh-test",
        "step_file": "test.step",
        "units": "meters",
        "groups": {
            "inlet": {
                "surface_tags": [1, 2, 3],
            }
        },
        "mesh_stats": {
            "total_vertices": 100,
            "total_faces": 200,
            "bounding_box": {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            },
        },
    }


def test_valid_export():
    """A fully valid bc_groups dict should pass schema validation without raising."""
    schema = load_schema()
    data = make_valid_bc_groups()
    jsonschema.validate(instance=data, schema=schema)


def test_invalid_missing_schema_version():
    """Data missing the required schema_version field must raise ValidationError."""
    schema = load_schema()
    data = make_valid_bc_groups()
    del data["schema_version"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=data, schema=schema)


def test_invalid_wrong_schema_version():
    """Data with schema_version=2 must raise ValidationError (const=1 constraint)."""
    schema = load_schema()
    data = make_valid_bc_groups()
    data["schema_version"] = 2
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=data, schema=schema)


def test_invalid_missing_groups():
    """Data missing the required groups field must raise ValidationError."""
    schema = load_schema()
    data = make_valid_bc_groups()
    del data["groups"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=data, schema=schema)


def test_invalid_missing_surface_tags():
    """A group missing the required surface_tags field must raise ValidationError."""
    schema = load_schema()
    data = make_valid_bc_groups()
    del data["groups"]["inlet"]["surface_tags"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=data, schema=schema)


def test_valid_multiple_groups():
    """A valid dict with two groups (inlet and wall) must pass schema validation."""
    schema = load_schema()
    data = make_valid_bc_groups()
    data["groups"]["wall"] = {
        "surface_tags": [4, 5, 6],
    }
    jsonschema.validate(instance=data, schema=schema)


def make_valid_mesh_mode():
    return {
        "schema_version": 1,
        "source": "blendgmsh-test",
        "mode": "mesh",
        "step_file": "test.stl",
        "units": "meters",
        "groups": {
            "inlet": {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                "face_vertex_indices": [[0, 1, 2]],
                "vertex_count": 3,
                "face_count": 1,
            }
        },
        "mesh_stats": {
            "total_vertices": 100,
            "total_faces": 200,
            "bounding_box": {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            },
        },
    }


def test_valid_mesh_mode():
    """A mesh-mode bc_groups dict should pass schema validation."""
    schema = load_schema()
    data = make_valid_mesh_mode()
    jsonschema.validate(instance=data, schema=schema)


def test_mesh_mode_missing_vertices():
    """Mesh mode group missing vertices must raise ValidationError."""
    schema = load_schema()
    data = make_valid_mesh_mode()
    del data["groups"]["inlet"]["vertices"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=data, schema=schema)


def test_brep_mode_field():
    """BREP mode with explicit mode field should pass."""
    schema = load_schema()
    data = make_valid_bc_groups()
    data["mode"] = "brep"
    jsonschema.validate(instance=data, schema=schema)
