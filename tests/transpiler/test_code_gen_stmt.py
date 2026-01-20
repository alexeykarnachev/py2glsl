"""Tests for the transpiler code_gen_stmt module."""

import ast

import pytest

from py2glsl.transpiler.ast_parser import get_annotation_type
from py2glsl.transpiler.code_gen_stmt import (
    generate_annotated_assignment,
    generate_assignment,
    generate_augmented_assignment,
    generate_body,
    generate_for_loop,
    generate_if_statement,
    generate_list_declaration,
    generate_return_statement,
    generate_while_loop,
)
from py2glsl.transpiler.models import (
    CollectedInfo,
    FunctionInfo,
    StructDefinition,
    StructField,
    TranspilerError,
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
        """Test that unsupported annotation types return None."""
        annotation = ast.List(elts=[], ctx=ast.Load())
        result = get_annotation_type(annotation)
        assert result is None


class TestGenerateAssignment:
    """Tests for the generate_assignment function."""

    def test_generate_assignment_new_var(self, symbols, collected_info):
        """Test generating code for assignment to a new variable."""
        node = ast.parse("x = 1.0").body[0]
        result = generate_assignment(node, symbols, collected_info)
        assert result == "float x = 1.0;"
        assert symbols["x"] == "float"

    def test_generate_assignment_existing_var(self, symbols, collected_info):
        """Test generating code for assignment to an existing variable."""
        node = ast.parse("time = 2.0").body[0]
        result = generate_assignment(node, symbols, collected_info)
        assert result == "time = 2.0;"

    def test_generate_assignment_attribute(self, symbols, collected_info):
        """Test generating code for assignment to an attribute."""
        node = ast.parse("test_struct.value = 5.0").body[0]
        result = generate_assignment(node, symbols, collected_info)
        assert result == "test_struct.value = 5.0;"

    def test_generate_assignment_unsupported_target(self, symbols, collected_info):
        """Test that assignment to unsupported targets raises an error."""
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
        with pytest.raises(TranspilerError, match="Unsupported assignment target"):
            generate_assignment(node, symbols, collected_info)


class TestGenerateAnnotatedAssignment:
    """Tests for the generate_annotated_assignment function."""

    def test_generate_annotated_assignment_with_value(self, symbols, collected_info):
        """Test generating code for annotated assignment with a value."""
        node = ast.parse("x: 'float' = 1.0").body[0]
        result = generate_annotated_assignment(node, symbols, collected_info)
        assert result == "float x = 1.0;"
        assert symbols["x"] == "float"

    def test_generate_annotated_assignment_without_value(self, symbols, collected_info):
        """Test generating code for annotated assignment without a value."""
        node = ast.parse("x: 'vec3'").body[0]
        result = generate_annotated_assignment(node, symbols, collected_info)
        assert result == "vec3 x;"
        assert symbols["x"] == "vec3"

    def test_generate_annotated_assignment_unsupported_target(
        self, symbols, collected_info
    ):
        """Test that annotated assignment to unsupported targets raises an error."""
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
        with pytest.raises(
            TranspilerError, match="Unsupported annotated assignment target"
        ):
            generate_annotated_assignment(node, symbols, collected_info)


class TestGenerateAugmentedAssignment:
    """Tests for the generate_augmented_assignment function."""

    def test_generate_augmented_assignment_add(self, symbols, collected_info):
        """Test generating code for augmented addition assignment."""
        node = ast.parse("count += 1").body[0]
        result = generate_augmented_assignment(node, symbols, collected_info)
        assert result == "count = count + 1;"

    def test_generate_augmented_assignment_subtract(self, symbols, collected_info):
        """Test generating code for augmented subtraction assignment."""
        node = ast.parse("time -= 0.1").body[0]
        result = generate_augmented_assignment(node, symbols, collected_info)
        assert result == "time = time - 0.1;"

    def test_generate_augmented_assignment_multiply(self, symbols, collected_info):
        """Test generating code for augmented multiplication assignment."""
        node = ast.parse("uv *= 2.0").body[0]
        result = generate_augmented_assignment(node, symbols, collected_info)
        assert result == "uv = uv * 2.0;"

    def test_generate_augmented_assignment_divide(self, symbols, collected_info):
        """Test generating code for augmented division assignment."""
        node = ast.parse("time /= 2.0").body[0]
        result = generate_augmented_assignment(node, symbols, collected_info)
        assert result == "time = time / 2.0;"

    def test_generate_augmented_assignment_unsupported_op(
        self, symbols, collected_info
    ):
        """Test that unsupported augmented assignment operators raise an error."""
        node = ast.AugAssign(
            target=ast.Name(id="count", ctx=ast.Store()),
            op=ast.Mod(),
            value=ast.Constant(value=10),
        )
        with pytest.raises(TranspilerError, match="Unsupported augmented operator"):
            generate_augmented_assignment(node, symbols, collected_info)


class TestGenerateForLoop:
    """Tests for the generate_for_loop function."""

    def test_generate_for_loop_simple(self, symbols, collected_info):
        """Test generating code for a simple for loop."""
        node = ast.parse("for i in range(5):\n    count += i").body[0]
        result = generate_for_loop(node, symbols, collected_info)
        assert len(result) == 3
        assert result[0] == "for (int i = 0; i < 5; i += 1) {"
        assert "count = count + i;" in result[1]
        assert result[2] == "}"

    def test_generate_for_loop_start_end(self, symbols, collected_info):
        """Test generating code for a for loop with start and end."""
        node = ast.parse("for i in range(2, 10):\n    count += i").body[0]
        result = generate_for_loop(node, symbols, collected_info)
        assert result[0] == "for (int i = 2; i < 10; i += 1) {"

    def test_generate_for_loop_start_end_step(self, symbols, collected_info):
        """Test generating code for a for loop with start, end, and step."""
        node = ast.parse("for i in range(0, 10, 2):\n    count += i").body[0]
        result = generate_for_loop(node, symbols, collected_info)
        assert result[0] == "for (int i = 0; i < 10; i += 2) {"

    def test_generate_for_loop_pass(self, symbols, collected_info):
        """Test generating code for a for loop with just a pass statement."""
        node = ast.parse("for i in range(5):\n    pass").body[0]
        result = generate_for_loop(node, symbols, collected_info)
        assert len(result) == 3
        assert result[0] == "for (int i = 0; i < 5; i += 1) {"
        assert "// Pass statement" in result[1]
        assert result[2] == "}"

    def test_generate_for_loop_non_range(self, symbols, collected_info):
        """Test that non-range-based for loops raise an error."""
        node = ast.parse("for item in items:\n    count += 1").body[0]
        with pytest.raises(TranspilerError, match="Unsupported iterable: unknown"):
            generate_for_loop(node, symbols, collected_info)


class TestGenerateWhileLoop:
    """Tests for the generate_while_loop function."""

    def test_generate_while_loop(self, symbols, collected_info):
        """Test generating code for a while loop."""
        node = ast.parse("while count < 10:\n    count += 1").body[0]
        result = generate_while_loop(node, symbols, collected_info)
        assert len(result) == 3
        assert result[0] == "while (count < 10) {"
        assert "count = count + 1;" in result[1]
        assert result[2] == "}"


class TestGenerateIfStatement:
    """Tests for the generate_if_statement function."""

    def test_generate_if_statement_no_else(self, symbols, collected_info):
        """Test generating code for an if statement without an else clause."""
        node = ast.parse("if count > 5:\n    count = 0").body[0]
        result = generate_if_statement(node, symbols, collected_info)
        assert len(result) == 3
        assert result[0] == "if (count > 5) {"
        assert "count = 0;" in result[1]
        assert result[2] == "}"

    def test_generate_if_statement_with_else(self, symbols, collected_info):
        """Test generating code for an if statement with an else clause."""
        node = ast.parse("if count > 5:\n    count = 0\nelse:\n    count += 1").body[0]
        result = generate_if_statement(node, symbols, collected_info)
        assert len(result) == 5
        assert result[0] == "if (count > 5) {"
        assert "count = 0;" in result[1]
        assert result[2] == "} else {"
        assert "count = count + 1;" in result[3]
        assert result[4] == "}"

    def test_generate_if_statement_variable_hoisting(self, symbols, collected_info):
        """Test that variables assigned in all branches are hoisted."""
        code = "if time > 0.0:\n    result = time * 2.0\nelse:\n    result = time * 3.0"
        node = ast.parse(code).body[0]
        result = generate_if_statement(node, symbols, collected_info)
        # Variable should be declared before if statement
        assert "float result;" in result[0]
        assert result[1] == "if (time > 0.0) {"
        assert "result = time * 2.0;" in result[2]
        assert result[3] == "} else {"
        assert "result = time * 3.0;" in result[4]
        assert result[-1] == "}"

    def test_generate_if_elif_else_variable_hoisting(self, symbols, collected_info):
        """Test variable hoisting with if/elif/else."""
        code = "if time > 1.0:\n    val = 1.0\nelif time > 0.0:\n    val = 2.0\nelse:\n    val = 3.0"
        node = ast.parse(code).body[0]
        result = generate_if_statement(node, symbols, collected_info)
        # val should be declared before if
        assert "float val;" in result[0]
        assert "if (time > 1.0)" in result[1]
        assert "val = 1.0;" in result[2]
        assert "} else {" in result[3]
        assert "if (time > 0.0)" in result[4]
        assert "val = 2.0;" in result[5]


class TestGenerateReturnStatement:
    """Tests for the generate_return_statement function."""

    def test_generate_return_statement(self, symbols, collected_info):
        """Test generating code for a return statement."""
        node = ast.parse("return vec4(uv, 0.0, 1.0)").body[0]
        result = generate_return_statement(node, symbols, collected_info)
        assert result == "return vec4(uv, 0.0, 1.0);"


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
        assert len(result) == 9
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
        """Test that a body with only a pass statement generates a no-op comment."""
        # Arrange
        node = ast.parse("pass").body

        # Act
        result = generate_body(node, symbols.copy(), collected_info)

        # Assert
        assert len(result) == 1
        assert "// Pass statement" in result[0]

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

    def test_generate_for_loop_static_list(self, symbols, collected_info):
        """Test generating code for a for loop over a static list."""
        # Arrange
        code = """
some_list = [vec3(1.0, 0.0, 0.0), vec3(2.0, 0.0, 0.0), vec3(3.0, 0.0, 0.0)]
total = vec3(0.0)
for item in some_list:
    total += item
        """
        node = ast.parse(code).body
        symbols.update({"some_list": "list[vec3]"})
        collected_info.globals["some_list_size"] = ("int", "3")

        # Act
        result = generate_body(node, symbols.copy(), collected_info)

        # Assert
        expected = [
            "vec3 some_list[3] = vec3[3](vec3(1.0, 0.0, 0.0), "
            "vec3(2.0, 0.0, 0.0), vec3(3.0, 0.0, 0.0));",
            "vec3 total = vec3(0.0);",
            "for (int i_some_list = 0; i_some_list < some_list_size; ++i_some_list) {",
            "    vec3 item = some_list[i_some_list];",
            "    total = total + item;",  # Code gen uses regular assignment, not +=
            "}",
        ]
        for line in expected:
            assert line in result, f"Expected line not found: {line}"

    def test_generate_for_loop_dynamic_list(self, symbols, collected_info):
        """Test generating code for a for loop over a dynamic list."""
        # Arrange
        code = """
total = vec3(0.0)
for item in some_list:
    total += item
        """
        node = ast.parse(code).body
        symbols.update({"some_list": "list[vec3]"})
        # Use globals instead of uniforms since CollectedInfo doesn't have uniforms
        collected_info.globals["some_list_size"] = ("int", "3")

        # Act
        result = generate_body(node, symbols.copy(), collected_info)

        # Assert
        expected = [
            "vec3 total = vec3(0.0);",
            "for (int i_some_list = 0; i_some_list < some_list_size; ++i_some_list) {",
            "    vec3 item = some_list[i_some_list];",
            "    total = total + item;",
            "}",
        ]
        for line in expected:
            assert line in result, f"Expected line not found: {line}"

    def test_generate_for_loop_nested(self, symbols, collected_info):
        """Test generating code for nested for loops over lists."""
        # Arrange
        code = """
outer_list = [vec3(1.0), vec3(2.0)]
inner_list = [vec3(3.0), vec3(4.0)]
total = vec3(0.0)
for outer in outer_list:
    for inner in inner_list:
        total += outer + inner
        """
        node = ast.parse(code).body
        symbols.update({"outer_list": "list[vec3]", "inner_list": "list[vec3]"})
        collected_info.globals["outer_list_size"] = ("int", "2")
        collected_info.globals["inner_list_size"] = ("int", "2")

        # Act
        result = generate_body(node, symbols.copy(), collected_info)

        # Assert
        expected = [
            "vec3 outer_list[2] = vec3[2](vec3(1.0), vec3(2.0));",
            "vec3 inner_list[2] = vec3[2](vec3(3.0), vec3(4.0));",
            "vec3 total = vec3(0.0);",
            # For loop iteration over outer list
            "for (int i_outer_list = 0; i_outer_list < outer_list_size; "
            "++i_outer_list) {",
            "    vec3 outer = outer_list[i_outer_list];",
            # Nested for loop iteration over inner list
            "    for (int i_inner_list = 0; i_inner_list < inner_list_size; "
            "++i_inner_list) {",
            "        vec3 inner = inner_list[i_inner_list];",
            "        total = total + outer + inner;",
            "    }",
            "}",
        ]
        for line in expected:
            assert line in result, f"Expected line not found: {line}"

    def test_generate_for_loop_empty_list(self, symbols, collected_info):
        """Test generating code for a for loop over an empty list."""
        # Arrange
        code = """
some_list = []
total = vec3(0.0)
for item in some_list:
    total += item
        """
        node = ast.parse(code).body
        symbols.update({"some_list": "list[vec3]"})
        collected_info.globals["some_list_size"] = ("int", "0")

        # Act
        result = generate_body(node, symbols.copy(), collected_info)

        # Assert
        expected = [
            "vec3 some_list[0];",
            "vec3 total = vec3(0.0);",
            "for (int i_some_list = 0; i_some_list < some_list_size; ++i_some_list) {",
            "    vec3 item = some_list[i_some_list];",
            "    total = total + item;",
            "}",
        ]
        for line in expected:
            assert line in result, f"Expected line not found: {line}"

    def test_generate_for_loop_type_mismatch(self, symbols, collected_info):
        """Test that type mismatch in list elements raises an error."""
        # Arrange
        code = """
some_list = [vec3(1.0), 2.0]  # Mixed types
for item in some_list:
    total += item
        """
        node = ast.parse(code).body
        symbols.update({"some_list": "list[vec3]"})

        # Act & Assert
        with pytest.raises(TranspilerError, match="Type mismatch in list elements"):
            generate_body(node, symbols.copy(), collected_info)

    def test_generate_body_with_continue(self, symbols, collected_info):
        """Test generating code for a body with a continue statement in a loop."""
        # Arrange
        code = """
while count < 10:
    if count < 5:
        count += 1
        continue
    count += 2
"""
        node = ast.parse(code).body

        # Act
        result = generate_body(node, symbols.copy(), collected_info)

        # Assert
        expected = [
            "while (count < 10) {",
            "    if (count < 5) {",
            "        count = count + 1;",
            "        continue;",
            "    }",
            "    count = count + 2;",
            "}",
        ]
        for line in expected:
            assert line in result, f"Expected line not found: {line}"


class TestGenerateListDeclaration:
    """Tests for the generate_list_declaration function."""

    def test_generate_list_declaration_static(self, symbols, collected_info):
        """Test generating code for a static list assignment."""
        node = ast.parse("some_list = [vec3(1.0), vec3(2.0), vec3(3.0)]").body[0]
        result = generate_list_declaration(node, symbols, collected_info)
        assert result == "vec3 some_list[3] = vec3[3](vec3(1.0), vec3(2.0), vec3(3.0));"
        assert symbols["some_list"] == "list[vec3]"
        assert collected_info.globals["some_list_size"] == ("int", "3")

    def test_generate_list_declaration_empty(self, symbols, collected_info):
        """Test generating code for an empty list assignment."""
        node = ast.parse("some_list = []").body[0]
        result = generate_list_declaration(node, symbols, collected_info)
        assert result == "vec3 some_list[0];"
        assert symbols["some_list"] == "list[vec3]"
        assert collected_info.globals["some_list_size"] == ("int", "0")

    def test_generate_list_declaration_type_mismatch(self, symbols, collected_info):
        """Test that type mismatch in list elements raises an error."""
        node = ast.parse("some_list = [vec3(1.0), 2.0]").body[0]  # Mixed types
        with pytest.raises(TranspilerError, match="Type mismatch in list elements"):
            generate_list_declaration(node, symbols, collected_info)
