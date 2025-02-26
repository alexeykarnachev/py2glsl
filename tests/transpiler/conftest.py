"""
Pytest configuration and shared fixtures for transpiler tests.

This module contains fixtures that are shared across multiple test modules.
"""

import ast

import pytest

from py2glsl.transpiler.models import (
    CollectedInfo,
    FunctionInfo,
    StructDefinition,
    StructField,
)


@pytest.fixture
def symbols():
    """Fixture providing a sample symbol table."""
    return {
        "uv": "vec2",
        "color": "vec4",
        "time": "float",
        "count": "int",
        "flag": "bool",
        "test_struct": "TestStruct",
    }


@pytest.fixture
def collected_info():
    """Fixture providing a sample collected info structure."""
    info = CollectedInfo()

    # Add a test struct
    info.structs["TestStruct"] = StructDefinition(
        name="TestStruct",
        fields=[
            StructField(name="position", type_name="vec3"),
            StructField(name="value", type_name="float"),
        ],
    )

    # Add a test function
    dummy_node = ast.FunctionDef(
        name="test_func",
        args=ast.arguments(
            args=[], posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[],
        decorator_list=[],
    )
    info.functions["test_func"] = FunctionInfo(
        name="test_func",
        return_type="vec3",
        param_types=["vec2", "float"],
        node=dummy_node,
    )

    return info
