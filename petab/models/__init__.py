"""Deprecated module for PEtab models.

Use petab.v1.models instead"""
from petab import _deprecated_import_v1
from petab.v1.models import *  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
