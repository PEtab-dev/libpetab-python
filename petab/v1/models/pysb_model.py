"""Deprecated: use :mod:`petab.v2.models.pysb_model`.

PySB models are only supported with PEtab v2.
"""

from ..._models import pysb_model as _impl
from ..._models.pysb_model import *
from ..._models.pysb_model import __all__  # noqa: F401


def __getattr__(name):
    return getattr(_impl, name)
