"""Functions for handling PySB models"""

from ..._models import pysb_model as _impl
from ..._models.pysb_model import *
from ..._models.pysb_model import __all__  # noqa: F401


def __getattr__(name):
    return getattr(_impl, name)
