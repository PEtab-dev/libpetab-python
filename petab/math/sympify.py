"""PEtab math to sympy conversion."""

import sympy as sp
from sympy.abc import _clash


def sympify_petab(expr: str) -> sp.Expr:
    """
    Convert a PEtab math expression to a sympy expression.

    Parameters
    ----------
    expr:
        The PEtab math expression.

    Returns
    -------
    The sympy expression corresponding to ``expr``.
    """
    return sp.sympify(expr, locals=_clash)
