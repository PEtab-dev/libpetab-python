"""Deprecated module for PySB models.

Use petab.v1.models.pysb_model instead."""
from petab import _deprecated_import_v1
from petab.v1.models.pysb_model import *  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
