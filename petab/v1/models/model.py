"""PEtab model abstraction"""

from ..._models import model as _impl
from ..._models.model import *
from ..._models.model import __all__  # noqa: F401


def __getattr__(name):
    return getattr(_impl, name)
