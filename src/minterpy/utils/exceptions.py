"""
This module contains custom exceptions used across Minterpy.
"""

class InvalidDomainBoundsError(ValueError):
    """Raised when domain bounds are invalid."""
    pass


class InvalidDerivativeOrderError(ValueError):
    """Raised when order of derivatives specification is invalid."""
    pass
