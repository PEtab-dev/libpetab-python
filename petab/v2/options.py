"""Options controlling the behavior of the PEtab v2 API.

Options can be set globally via :func:`set_options`, or temporarily via
the :func:`option_context` context manager:

>>> import petab.v2 as petab
>>> petab.get_options().table_columns
<TableColumns.ALL: 'all'>
>>> with petab.option_context(table_columns="non_empty"):
...     petab.get_options().table_columns
<TableColumns.NON_EMPTY: 'non_empty'>
>>> petab.get_options().table_columns
<TableColumns.ALL: 'all'>
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum

__all__ = [
    "Options",
    "TableColumns",
    "get_options",
    "option_context",
    "set_options",
]


class TableColumns(StrEnum):
    """Columns to include when converting PEtab tables to DataFrames.

    Required columns, and any columns containing any
    non-default values, are always included.
    """

    #: All standard columns and all additional columns.
    ALL = "all"
    #: Only columns that have a non-empty value in at least one row.
    NON_EMPTY = "non_empty"
    #: Only non-empty columns that differ from the default value in at least
    #: one row.
    NON_DEFAULT = "non_default"


@dataclass(frozen=True)
class Options:
    """Options controlling the behavior of the PEtab v2 API."""

    #: Columns to include when converting tables to DataFrames or files.
    #: See :class:`TableColumns`.
    table_columns: TableColumns = TableColumns.ALL

    def __post_init__(self):
        object.__setattr__(
            self, "table_columns", TableColumns(self.table_columns)
        )


_global_options = Options()
# Temporary overrides set via `option_context`.
# A ContextVar keeps them local to the current thread / async task.
_context_options: ContextVar[Options | None] = ContextVar(
    "petab_v2_options", default=None
)


def get_options() -> Options:
    """Get the currently active options."""
    return _context_options.get() or _global_options


def set_options(**kwargs) -> None:
    """Set options globally.

    Options set inside an active :func:`option_context` only apply until
    that context is exited.

    :param kwargs: Options to set. See :class:`Options`.
    """
    global _global_options

    if (current := _context_options.get()) is not None:
        _context_options.set(replace(current, **kwargs))
    else:
        _global_options = replace(_global_options, **kwargs)


@contextmanager
def option_context(**kwargs) -> Iterator[Options]:
    """Temporarily set options.

    The previous options are restored when the context is exited.

    :param kwargs: Options to set. See :class:`Options`.
    :return: The options active inside the context.
    """
    token = _context_options.set(replace(get_options(), **kwargs))
    try:
        yield get_options()
    finally:
        _context_options.reset(token)
