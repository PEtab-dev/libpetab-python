"""Deprecated module for reading and writing PEtab YAML files.

Use petab.v1.yaml instead."""
from petab import _deprecated_import_v1
from petab.v1.yaml import *  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
