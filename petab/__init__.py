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


# Create dummy modules for all old modules
v1_root = Path(__file__).resolve().parent / "v1"
v1_objects = [f.relative_to(v1_root) for f in v1_root.rglob("*")]
for v1_object in v1_objects:
    abs_v1_object = v1_root / v1_object
    if abs_v1_object.is_file():
        module_name = ".".join(
            ["petab", *v1_object.parts[:-1], v1_object.stem]
        )
    elif abs_v1_object.is_dir():
        module_name = ".".join(["petab", *v1_object.parts])
    else:
        raise ValueError(abs_v1_object)
    sys.modules[module_name] = globals().get(module_name)
