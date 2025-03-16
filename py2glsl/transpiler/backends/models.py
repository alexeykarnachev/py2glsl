from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class BackendType(Enum):
    """Supported GLSL backend types."""

    STANDARD = auto()
    SHADERTOY = auto()


@dataclass
class EntryPointConfig:
    """Configuration for shader entry point generation."""

    input_variables: dict[str, str] = field(default_factory=dict)
    output_variables: dict[str, str] = field(default_factory=dict)
    main_wrapper_template: str = ""


@dataclass
class BackendConfig:
    """Configuration for a GLSL backend."""

    name: str
    version_directive: str
    entry_point: EntryPointConfig
    predefined_uniforms: dict[str, str] = field(default_factory=dict)
    extensions: list[str] = field(default_factory=list)
    preprocessor_defines: dict[str, str | None] = field(default_factory=dict)
    additional_options: dict[str, Any] = field(default_factory=dict)
