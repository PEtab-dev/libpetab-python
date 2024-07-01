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
import sys
from pathlib import Path
from warnings import warn

ENV_NUM_THREADS = "PETAB_NUM_THREADS"
__all__ = ["ENV_NUM_THREADS"]


def __getattr__(name):
    if attr := globals().get(name):
        return attr

    warn(
        f"Accessing `petab.{name}` is deprecated and will be removed in "
        f"the next major release. Please use `petab.v1.{name}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return getattr(importlib.import_module("petab.v1"), name)


deprecated_modules = [
    f.stem for f in (Path(__file__).parent / "v1").glob("*") if f.is_file()
]
for deprecated_module in deprecated_modules:
    sys.modules[f"petab.{deprecated_module}"] = globals().get(
        deprecated_module
    )
