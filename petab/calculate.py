"""Deprecated module for calculating residuals and log-likelihoods.

Use petab.v1.calculate instead."""
from petab import _deprecated_import_v1
from petab.v1.calculate import *  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
