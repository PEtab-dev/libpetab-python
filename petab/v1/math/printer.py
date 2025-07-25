"""A PEtab-compatible sympy string-printer."""

from itertools import chain, islice

import sympy as sp
from sympy.printing.str import StrPrinter

__all__ = ["PetabStrPrinter", "petab_math_str"]


class PetabStrPrinter(StrPrinter):
    """A PEtab-compatible sympy string-printer."""

    #: Mapping of sympy functions to PEtab functions
    _func_map = {
        "asin": "arcsin",
        "acos": "arccos",
        "atan": "arctan",
        "acot": "arccot",
        "asec": "arcsec",
        "acsc": "arccsc",
        "asinh": "arcsinh",
        "acosh": "arccosh",
        "atanh": "arctanh",
        "acoth": "arccoth",
        "asech": "arcsech",
        "acsch": "arccsch",
        "Abs": "abs",
    }

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    def _print_Pow(self, expr: sp.Pow):
        """Custom printing for the power operator"""
        base, exp = expr.as_base_exp()
        str_base = self._print(base)
        str_exp = self._print(exp)
        if not base.is_Atom:
            str_base = f"({str_base})"
        if not exp.is_Atom:
            str_exp = f"({str_exp})"
        return f"{str_base} ^ {str_exp}"

    def _print_Infinity(self, expr):
        """Custom printing for infinity"""
        return "inf"

    def _print_NegativeInfinity(self, expr):
        """Custom printing for negative infinity"""
        return "-inf"

    def _print_Function(self, expr):
        """Custom printing for specific functions"""

        if expr.func.__name__ == "Piecewise":
            return self._print_Piecewise(expr)

        if func := self._func_map.get(expr.func.__name__):
            return f"{func}({', '.join(map(self._print, expr.args))})"

        return super()._print_Function(expr)

    def _print_Piecewise(self, expr):
        """Custom printing for Piecewise function"""
        # merge the tuples and drop the final `True` condition
        str_args = map(
            self._print,
            islice(chain.from_iterable(expr.args), 2 * len(expr.args) - 1),
        )
        return f"piecewise({', '.join(str_args)})"

    def _print_Min(self, expr):
        """Custom printing for Min function"""
        return f"min({', '.join(map(self._print, expr.args))})"

    def _print_Max(self, expr):
        """Custom printing for Max function"""
        return f"max({', '.join(map(self._print, expr.args))})"


def petab_math_str(expr: sp.Basic | sp.Expr | None) -> str:
    """Convert a sympy expression to a PEtab-compatible math expression string.

    :example:
    >>> expr = sp.sympify("x**2 + sin(y)")
    >>> petab_math_str(expr)
    'x ^ 2 + sin(y)'
    >>> expr = sp.sympify("Piecewise((1, x > 0), (0, True))")
    >>> petab_math_str(expr)
    'piecewise(1, x > 0, 0)'
    """
    if expr is None:
        return ""

    return PetabStrPrinter().doprint(expr)
