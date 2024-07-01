"""
PEtab global
============

.. warning::

    All functions in here are deprecated. Use the respective functions from
    :mod:`petab.v1` instead.

Attributes:
    ENV_NUM_THREADS:
        Name of environment variable to set number of threads or processes
        PEtab should use for operations that can be performed in parallel.
        By default, all operations are performed sequentially.
"""
import functools
import inspect
import sys
import warnings
from warnings import warn

# deprecated imports
from petab.v1 import *  # noqa: F403, F401, E402

from .v1.format_version import __format_version__  # noqa: F401, E402

# __all__ = [
#     'ENV_NUM_THREADS',
# ]

ENV_NUM_THREADS = "PETAB_NUM_THREADS"


def _deprecated_v1(func):
    """Decorator for deprecation warnings for functions."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn(
            f"petab.{func.__name__} is deprecated, "
            f"please use petab.v1.{func.__name__} instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return new_func


def _deprecated_import_v1(module_name: str):
    """Decorator for deprecation warnings for modules."""
    warn(
        f"The '{module_name}' module is deprecated and will be removed "
        f"in the next major release. Please use "
        f"'petab.v1.{module_name.removeprefix('petab.')}' "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )


__all__ = [
    x
    for x in dir(sys.modules[__name__])
    if not x.startswith("_")
    and x not in {"sys", "warnings", "functools", "warn", "inspect"}
]


# apply decorator to all functions in the module
for name in __all__:
    obj = globals().get(name)
    if callable(obj) and inspect.isfunction(obj):
        globals()[name] = _deprecated_v1(obj)
del name, obj
