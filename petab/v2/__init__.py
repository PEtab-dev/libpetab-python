"""The PEtab 2.0 subpackage.

Contains all functionality related to handling PEtab 2.0 problems.
"""
from warnings import warn

from ..v1 import *  # noqa: F403, F401, E402

# import after v1
from .problem import Problem  # noqa: F401

warn(
    "Support for PEtab2.0 and all of petab.v2 is experimental "
    "and subject to changes!",
    stacklevel=1,
)
