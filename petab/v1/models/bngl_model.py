"""Deprecated: use :mod:`petab.v2.models.bngl_model`.

BNGL models are only supported with PEtab v2.
"""

from ..._models import bngl_model as _impl
from ..._models.bngl_model import *
from ..._models.bngl_model import __all__  # noqa: F401


def __getattr__(name):
    return getattr(_impl, name)
