"""Tests for the transpiler code_gen_stmt module."""

import ast

import pytest

from py2glsl.transpiler.code_gen_stmt import (
    generate_annotated_assignment,
    generate_assignment,
    generate_augmented_assignment,
    generate_body,
    generate_for_loop,
    generate_if_statement,
    generate_return_statement,
    generate_while_loop,
    get_annotation_type,
)
from py2glsl.transpiler.errors import TranspilerError
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


class TestGetAnnotationType:
    """Tests for the get_annotation_type function."""

    def test_get_annotation_type_name(self):
        """Test getting type from name annotation."""
        # Arrange
        annotation = ast.Name(id="vec2", ctx=ast.Load())

        # Act
        result = get_annotation_type(annotation)

        # Assert
        assert result == "vec2"

    def test_get_annotation_type_string(self):
        """Test getting type from string annotation."""
        # Arrange
        annotation = ast.Constant(value="float", kind=None)

        # Act
        result = get_annotation_type(annotation)

        # Assert
        assert result == "float"

    def test_get_annotation_type_unsupported(self):
        """Test that unsupported annotation types raise an error."""
        # Arrange
        annotation = ast.List(elts=[], ctx=ast.Load())

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported annotation type"):
            get_annotation_type(annotation)


class TestGenerateAssignment:
    """Tests for the generate_assignment function."""

    def test_generate_assignment_new_var(self, symbols, collected_info):
        """Test generating code for assignment to a new variable."""
        # Arrange
        node = ast.parse("x = 1.0").body[0]

        # Act
        result = generate_assignment(node, symbols, "    ", collected_info)

        # Assert
        assert result == "    float x = 1.0;"
        assert "x" in symbols
        assert symbols["x"] == "float"

    def test_generate_assignment_existing_var(self, symbols, collected_info):
        """Test generating code for assignment to an existing variable."""
        # Arrange
        node = ast.parse("time = 2.0").body[0]

        # Act
        result = generate_assignment(node, symbols, "    ", collected_info)

        # Assert
        assert result == "    time = 2.0;"

    def test_generate_assignment_attribute(self, symbols, collected_info):
        """Test generating code for assignment to an attribute."""
        # Arrange
        node = ast.parse("test_struct.value = 5.0").body[0]

        # Act
        result = generate_assignment(node, symbols, "    ", collected_info)

        # Assert
        assert result == "    test_struct.value = 5.0;"

    def test_generate_assignment_unsupported_target(self, symbols, collected_info):
        """Test that assignment to unsupported targets raises an error."""
        # Arrange - list element assignment
        node = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id="list", ctx=ast.Load()),
                    slice=ast.Index(value=ast.Constant(value=0)),
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=1.0),
        )

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported assignment target"):
            generate_assignment(node, symbols, "    ", collected_info)


class TestGenerateAnnotatedAssignment:
    """Tests for the generate_annotated_assignment function."""

    def test_generate_annotated_assignment_with_value(self, symbols, collected_info):
        """Test generating code for annotated assignment with a value."""
        # Arrange
        node = ast.parse("x: 'float' = 1.0").body[0]

        # Act
        result = generate_annotated_assignment(node, symbols, "    ", collected_info)

        # Assert
        assert result == "    float x = 1.0;"
        assert "x" in symbols
        assert symbols["x"] == "float"

    def test_generate_annotated_assignment_without_value(self, symbols, collected_info):
        """Test generating code for annotated assignment without a value."""
        # Arrange
        node = ast.parse("x: 'vec3'").body[0]

        # Act
        result = generate_annotated_assignment(node, symbols, "    ", collected_info)

        # Assert
        assert result == "    vec3 x;"
        assert "x" in symbols
        assert symbols["x"] == "vec3"

    def test_generate_annotated_assignment_unsupported_target(
        self, symbols, collected_info
    ):
        """Test that annotated assignment to unsupported targets raises an error."""
        # Arrange - we need to create this manually as it's not valid Python syntax
        node = ast.AnnAssign(
            target=ast.Attribute(
                value=ast.Name(id="test_struct", ctx=ast.Load()),
                attr="new_field",
                ctx=ast.Store(),
            ),
            annotation=ast.Constant(value="float"),
            value=None,
            simple=0,
        )

        # Act & Assert
        with pytest.raises(
            TranspilerError, match="Unsupported annotated assignment target"
        ):
            generate_annotated_assignment(node, symbols, "    ", collected_info)


class TestGenerateAugmentedAssignment:
    """Tests for the generate_augmented_assignment function."""

    def test_generate_augmented_assignment_add(self, symbols, collected_info):
        """Test generating code for augmented addition assignment."""
        # Arrange
        node = ast.parse("count += 1").body[0]

        # Act
        result = generate_augmented_assignment(node, symbols, "    ", collected_info)

        # Assert
        assert result == "    count = count + 1;"

    def test_generate_augmented_assignment_subtract(self, symbols, collected_info):
        """Test generating code for augmented subtraction assignment."""
        # Arrange
        node = ast.parse("time -= 0.1").body[0]

        # Act
        result = generate_augmented_assignment(node, symbols, "    ", collected_info)

        # Assert
        assert result == "    time = time - 0.1;"

    def test_generate_augmented_assignment_multiply(self, symbols, collected_info):
        """Test generating code for augmented multiplication assignment."""
        # Arrange
        node = ast.parse("uv *= 2.0").body[0]

        # Act
        result = generate_augmented_assignment(node, symbols, "    ", collected_info)

        # Assert
        assert result == "    uv = uv * 2.0;"

    def test_generate_augmented_assignment_divide(self, symbols, collected_info):
        """Test generating code for augmented division assignment."""
        # Arrange
        node = ast.parse("time /= 2.0").body[0]

        # Act
        result = generate_augmented_assignment(node, symbols, "    ", collected_info)

        # Assert
        assert result == "    time = time / 2.0;"

    def test_generate_augmented_assignment_unsupported_op(
        self, symbols, collected_info
    ):
        """Test that unsupported augmented assignment operators raise an error."""
        # Arrange - modulo operator isn't directly supported
        node = ast.AugAssign(
            target=ast.Name(id="count", ctx=ast.Store()),
            op=ast.Mod(),
            value=ast.Constant(value=10),
        )

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported augmented operator"):
            generate_augmented_assignment(node, symbols, "    ", collected_info)


class TestGenerateForLoop:
    """Tests for the generate_for_loop function."""

    def test_generate_for_loop_simple(self, symbols, collected_info):
        """Test generating code for a simple for loop."""
        # Arrange
        node = ast.parse(
            """
for i in range(5):
    count += i
"""
        ).body[0]

        # Act
        result = generate_for_loop(node, symbols, "    ", collected_info)

        # Assert
        assert len(result) == 3
        assert result[0] == "    for (int i = 0; i < 5; i += 1) {"
        assert "count = count + i;" in result[1]
        assert result[2] == "    }"

    def test_generate_for_loop_start_end(self, symbols, collected_info):
        """Test generating code for a for loop with start and end."""
        # Arrange
        node = ast.parse(
            """
for i in range(2, 10):
    count += i
"""
        ).body[0]

        # Act
        result = generate_for_loop(node, symbols, "    ", collected_info)

        # Assert
        assert result[0] == "    for (int i = 2; i < 10; i += 1) {"

    def test_generate_for_loop_start_end_step(self, symbols, collected_info):
        """Test generating code for a for loop with start, end, and step."""
        # Arrange
        node = ast.parse(
            """
for i in range(0, 10, 2):
    count += i
"""
        ).body[0]

        # Act
        result = generate_for_loop(node, symbols, "    ", collected_info)

        # Assert
        assert result[0] == "    for (int i = 0; i < 10; i += 2) {"

    def test_generate_for_loop_pass(self, symbols, collected_info):
        """Test generating code for a for loop with just a pass statement."""
        # Arrange
        node = ast.parse(
            """
for i in range(5):
    pass
"""
        ).body[0]

        # Act
        result = generate_for_loop(node, symbols, "    ", collected_info)

        # Assert
        assert len(result) == 3
        assert result[0] == "    for (int i = 0; i < 5; i += 1) {"
        assert "// Pass statement" in result[1]
        assert result[2] == "    }"

    def test_generate_for_loop_non_range(self, symbols, collected_info):
        """Test that non-range-based for loops raise an error."""
        # Arrange
        node = ast.parse(
            """
for item in items:
    count += 1
"""
        ).body[0]

        # Act & Assert
        with pytest.raises(
            TranspilerError, match="Only range-based for loops are supported"
        ):
            generate_for_loop(node, symbols, "    ", collected_info)


class TestGenerateWhileLoop:
    """Tests for the generate_while_loop function."""

    def test_generate_while_loop(self, symbols, collected_info):
        """Test generating code for a while loop."""
        # Arrange
        node = ast.parse(
            """
while count < 10:
    count += 1
"""
        ).body[0]

        # Act
        result = generate_while_loop(node, symbols, "    ", collected_info)

        # Assert
        assert len(result) == 3
        assert result[0] == "    while (count < 10) {"
        assert "count = count + 1;" in result[1]
        assert result[2] == "    }"


class TestGenerateIfStatement:
    """Tests for the generate_if_statement function."""

    def test_generate_if_statement_no_else(self, symbols, collected_info):
        """Test generating code for an if statement without an else clause."""
        # Arrange
        node = ast.parse(
            """
if count > 5:
    count = 0
"""
        ).body[0]

        # Act
        result = generate_if_statement(node, symbols, "    ", collected_info)

        # Assert
        assert len(result) == 3
        assert result[0] == "    if (count > 5) {"
        assert "count = 0;" in result[1]
        assert result[2] == "    }"

    def test_generate_if_statement_with_else(self, symbols, collected_info):
        """Test generating code for an if statement with an else clause."""
        # Arrange
        node = ast.parse(
            """
if count > 5:
    count = 0
else:
    count += 1
"""
        ).body[0]

        # Act
        result = generate_if_statement(node, symbols, "    ", collected_info)

        # Assert
        assert len(result) == 5
        assert result[0] == "    if (count > 5) {"
        assert "count = 0;" in result[1]
        assert result[2] == "    } else {"
        assert "count = count + 1;" in result[3]
        assert result[4] == "    }"


class TestGenerateReturnStatement:
    """Tests for the generate_return_statement function."""

    def test_generate_return_statement(self, symbols, collected_info):
        """Test generating code for a return statement."""
        # Arrange
        node = ast.parse("return vec4(uv, 0.0, 1.0)").body[0]

        # Act
        result = generate_return_statement(node, symbols, "    ", collected_info)

        # Assert
        assert result == "    return vec4(uv, 0.0, 1.0);"


class TestGenerateBody:
    """Tests for the generate_body function."""

    def test_generate_body_mixed_statements(self, symbols, collected_info):
        """Test generating code for a body with mixed statements."""
        # Arrange
        code = """
x: 'float' = 0.0
for i in range(5):
    x += 1.0
if x > 3.0:
    return vec4(x, 0.0, 0.0, 1.0)
else:
    return vec4(0.0, x, 0.0, 1.0)
"""
        node = ast.parse(code).body

        # Act
        result = generate_body(node, symbols.copy(), collected_info)

        # Assert
        assert len(result) == 7
        assert "float x = 0.0;" in result[0]
        assert "for (int i = 0; i < 5; i += 1) {" in result[1]
        assert "x = x + 1.0;" in result[2]
        assert "}" in result[3]
        assert "if (x > 3.0) {" in result[4]
        assert "return vec4(x, 0.0, 0.0, 1.0);" in result[5]
        assert "} else {" in result[6]
        # Note: The actual list will be longer, but we're just checking key elements

    def test_generate_body_empty(self, symbols, collected_info):
        """Test generating code for an empty body."""
        # Arrange
        node = []

        # Act
        result = generate_body(node, symbols.copy(), collected_info)

        # Assert
        assert result == []

    def test_generate_body_only_pass(self, symbols, collected_info):
        """Test that a body with only a pass statement raises an error."""
        # Arrange
        node = ast.parse("pass").body

        # Act & Assert
        with pytest.raises(TranspilerError, match="Pass statements are not supported"):
            generate_body(node, symbols.copy(), collected_info)

    def test_generate_body_with_break(self, symbols, collected_info):
        """Test generating code for a body with a break statement."""
        # Arrange
        code = """
while True:
    if count > 5:
        break
    count += 1
"""
        node = ast.parse(code).body

        # Act
        result = generate_body(node, symbols.copy(), collected_info)

        # Assert
        assert any("break;" in line for line in result)

    def test_generate_body_unsupported_stmt(self, symbols, collected_info):
        """Test that unsupported statements raise an error."""
        # Arrange - with statement isn't supported
        node = [ast.With(items=[], body=[], type_comment=None)]

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported statement: With"):
            generate_body(node, symbols.copy(), collected_info)
