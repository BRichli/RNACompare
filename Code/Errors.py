class StructureError(Exception):
    """Custom exception raised for structural issues in the graph."""

    def __init__(self, message: str = "A structural graph error has occurred ") -> None:
        super().__init__(message)


class LexError(Exception):
    """Custom exception raised for structural issues in the graph."""

    def __init__(self, message: str = "A structural graph error has occurred ") -> None:
        super().__init__(message)
