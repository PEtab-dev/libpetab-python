"""Various internal helper functions."""

from ..v1.core import to_float_if_float  # noqa: F401
from ..v1.lint import is_valid_identifier as _matches_id_pattern
from .C import RESERVED_KEYWORDS


def is_reserved_keyword(x: str) -> bool:
    """Check whether `x` is a PEtab v2 reserved keyword (case-insensitive)."""
    return isinstance(x, str) and x.lower() in RESERVED_KEYWORDS


def is_valid_identifier(x: str) -> bool:
    """Check whether `x` is a valid PEtab v2 identifier.

    A valid identifier matches the PEtab identifier pattern and is not a
    reserved keyword.
    """
    return _matches_id_pattern(x) and not is_reserved_keyword(x)
