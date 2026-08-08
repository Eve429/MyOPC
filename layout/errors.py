"""Layout-domain exceptions with stable, caller-friendly failure semantics."""


class LayoutError(RuntimeError):
    """Base class for layout database failures."""


class LayoutOpenError(LayoutError):
    """Raised when an input layout cannot be opened or parsed."""


class AmbiguousTopCellError(LayoutError):
    """Raised when a layout has multiple top cells and none was selected."""


class CellNotFoundError(LayoutError):
    """Raised when a requested cell does not exist."""


class LayerNotFoundError(LayoutError):
    """Raised when a requested layer/datatype pair does not exist."""


class ClosedLayoutError(LayoutError):
    """Raised when an operation targets an already closed LayoutDB."""
