"""Functions for handling SBML models"""

from ..._models import sbml_model as _impl
from ..._models.sbml_model import *
from ..._models.sbml_model import __all__  # noqa: F401


def __getattr__(name):
    return getattr(_impl, name)
