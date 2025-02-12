class GLSLError(Exception):
    """Base error for GLSL type system."""


class GLSLTypeError(GLSLError):
    """Error related to type compatibility."""


class GLSLOperationError(GLSLError):
    """Error in GLSL operations."""


class GLSLSwizzleError(GLSLError):
    """Error in vector swizzling."""
