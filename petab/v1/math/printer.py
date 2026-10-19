"""A PEtab-compatible sympy string-printer."""

import typing
from functools import reduce
from itertools import chain, islice

import sympy as sp
from sympy.logic.boolalg import Boolean
from sympy.printing.str import StrPrinter

__all__ = ["PetabStrPrinter", "petab_math_str"]


class PetabStrPrinter(StrPrinter):
    """A PEtab-compatible sympy string-printer."""

    #: Mapping of sympy functions to PEtab functions
    _func_map: typing.ClassVar[dict[str, str]] = {
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

    def _print_Float(self, expr: sp.Float):
        """Print double-precision floats such that they round-trip exactly."""
        if expr._prec != 53:
            return super()._print_Float(expr)
        value = float(expr)
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value))
        return repr(value)

    def _print_bool_operand(self, expr) -> str:
        """Print an operand of a boolean or relational operator."""
        if isinstance(expr, Boolean) and not expr.is_Atom:
            return f"({self._print(expr)})"
        return self._print(expr)

    def _print_Relational(self, expr):
        """Custom printing for comparisons (sympy prints ``Eq(a, b)``)"""
        lhs = self._print_bool_operand(expr.lhs)
        rhs = self._print_bool_operand(expr.rhs)
        return f"{lhs} {expr.rel_op} {rhs}"

    def _print_And(self, expr):
        """Custom printing for logical AND"""
        return " && ".join(map(self._print_bool_operand, expr.args))

    def _print_Or(self, expr):
        """Custom printing for logical OR"""
        return " || ".join(map(self._print_bool_operand, expr.args))

    def _print_Not(self, expr):
        """Custom printing for logical NOT"""
        return f"!{self._print_bool_operand(expr.args[0])}"

    def _print_Pow(self, expr: sp.Pow):
        """Custom printing for the power operator"""
        base, exp = expr.as_base_exp()
        str_base = self._print(base)
        str_exp = self._print(exp)
        # A negative or non-integer rational base is an Atom, but without
        # parentheses "-2 ^ x" or "1/2 ^ x" would re-parse as -(2^x) or 1/(2^x)
        if (
            not base.is_Atom
            or (base.is_number and base.is_negative)
            or (base.is_Rational and not base.is_Integer)
        ):
            str_base = f"({str_base})"
        # A non-integer Rational exponent (e.g. sqrt -> 1/2) is an Atom but
        # prints as the multi-token "1/2", so without parentheses "x ^ 1/2"
        # re-parses as (x^1)/2. Parenthesize it explicitly.
        if not exp.is_Atom or (exp.is_Rational and not exp.is_Integer):
            str_exp = f"({str_exp})"
        return f"{str_base} ^ {str_exp}"

    def _print_Pi(self, expr):
        """PEtab has no pi constant; print its double-precision value"""
        return repr(float(expr))

    def _print_Exp1(self, expr):
        """Custom printing for Euler's number (no constant in PEtab)"""
        return "exp(1)"

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
        if expr.args[-1].cond != sp.true:
            raise ValueError(
                "Piecewise without a default value cannot be represented "
                f"in PEtab: {expr}"
            )
        # merge the tuples and drop the final `True` condition
        str_args = map(
            self._print,
            islice(chain.from_iterable(expr.args), 2 * len(expr.args) - 1),
        )
        return f"piecewise({', '.join(str_args)})"

    def _print_Min(self, expr):
        """Custom printing for Min function (binary in PEtab)"""
        return reduce(
            lambda acc, arg: f"min({acc}, {arg})",
            map(self._print, expr.args),
        )

    def _print_Max(self, expr):
        """Custom printing for Max function (binary in PEtab)"""
        return reduce(
            lambda acc, arg: f"max({acc}, {arg})",
            map(self._print, expr.args),
        )


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
