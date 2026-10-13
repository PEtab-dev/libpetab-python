"""
PEtab global
============

Attributes:
    ENV_NUM_THREADS:
        Name of environment variable to set number of threads or processes
        PEtab should use for operations that can be performed in parallel.
        By default, all operations are performed sequentially.
"""

import importlib

from .version import __version__  # noqa: F401

ENV_NUM_THREADS = "PETAB_NUM_THREADS"
__all__ = ["ENV_NUM_THREADS"]


def __getattr__(name):
    """Lazily import the `petab.v1` and `petab.v2` subpackages."""
    if name in ("v1", "v2"):
        return importlib.import_module(f"petab.{name}")
    raise AttributeError(f"module 'petab' has no attribute '{name}'")
