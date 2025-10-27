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
from functools import partial
from pathlib import Path
from warnings import warn

ENV_NUM_THREADS = "PETAB_NUM_THREADS"
__all__ = ["ENV_NUM_THREADS"]


def __getattr__(name):
    if attr := globals().get(name):
        return attr
    if name == "v1":
        return importlib.import_module("petab.v1")
    if name == "v2":
        return importlib.import_module("petab.v2")
    if name not in ("__path__", "__all__", "__wrapped__"):
        warn(
            f"Accessing `petab.{name}` is deprecated and will be removed in "
            f"the next major release. Please use `petab.v1.{name}` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return getattr(importlib.import_module("petab.v1"), name)


def _v1getattr(name, module):
    if name not in ("__path__", "__all__", "__wrapped__"):
        warn(
            f"Accessing `petab.{name}` is deprecated and will be removed in "
            f"the next major release. Please use `petab.v1.{name}` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    try:
        return module.__dict__[name]
    except KeyError:
        raise AttributeError(name) from None


# Create dummy modules for all old modules
v1_root = Path(__file__).resolve().parent / "v1"
v1_objects = [f.relative_to(v1_root) for f in v1_root.rglob("*")]
for v1_object in v1_objects:
    if "__pycache__" in str(v1_object):
        continue
    if v1_object.suffix not in ["", ".py"]:
        continue
    if not (v1_root / v1_object).exists():
        raise ValueError(v1_root / v1_object)
    v1_object_parts = [*v1_object.parts[:-1], v1_object.stem]
    module_name = ".".join(["petab", *v1_object_parts])

    try:
        real_module = importlib.import_module(
            f"petab.v1.{'.'.join(v1_object_parts)}"
        )
        real_module.__getattr__ = partial(_v1getattr, module=real_module)
        sys.modules[module_name] = real_module
    except ModuleNotFoundError:
        pass
