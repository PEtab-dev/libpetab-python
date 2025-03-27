"""PEtab math to sympy conversion."""

import numpy as np
import sympy as sp
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from ._generated.PetabMathExprLexer import PetabMathExprLexer
from ._generated.PetabMathExprParser import PetabMathExprParser
from .SympyVisitor import MathVisitorSympy, bool2num

__all__ = ["sympify_petab"]


def sympify_petab(
    expr: str | int | float, evaluate: bool = True
) -> sp.Expr | sp.Basic:
    """Convert PEtab math expression to sympy expression.

    .. note::

        All symbols in the returned expression will have the `real=True`
        assumption.

    Args:
        expr: PEtab math expression.
        evaluate: Whether to evaluate the expression.

    Raises:
        ValueError: Upon lexer/parser errors or if the expression is
        otherwise invalid.

    Returns:
        The sympy expression corresponding to `expr`.
        Boolean values are converted to numeric values.


    :example:
    >>> from petab.math import sympify_petab
    >>> sympify_petab("sin(0)")
    0
    >>> sympify_petab("sin(0)", evaluate=False)
    sin(0.0)
    >>> sympify_petab("sin(0)", evaluate=True)
    0
    >>> sympify_petab("1 + 2", evaluate=True)
    3.00000000000000
    >>> sympify_petab("1 + 2", evaluate=False)
    1.0 + 2.0
    >>> sympify_petab("piecewise(1, 1 > 2, 0)", evaluate=True)
    0.0
    >>> sympify_petab("piecewise(1, 1 > 2, 0)", evaluate=False)
    Piecewise((1.0, 1.0 > 2.0), (0.0, True))
    >>> # currently, boolean values are converted to numeric values
    >>> #  independent of the `evaluate` flag
    >>> sympify_petab("true", evaluate=True)
    1.00000000000000
    >>> sympify_petab("true", evaluate=False)
    1.00000000000000
    >>> # ... and integer values are converted to floats
    >>> sympify_petab("2", evaluate=True)
    2.00000000000000
    """
    if isinstance(expr, sp.Expr):
        # TODO: check if only PEtab-compatible symbols and functions are used
        return expr

    if isinstance(expr, int) or isinstance(expr, np.integer):
        return sp.Integer(expr)
    if isinstance(expr, float) or isinstance(expr, np.floating):
        return sp.Float(expr)

    try:
        input_stream = InputStream(expr)
    except TypeError as e:
        raise TypeError(f"Error parsing {expr!r}: {e.args[0]}") from e

    lexer = PetabMathExprLexer(input_stream)
    # Set error listeners
    lexer.removeErrorListeners()
    lexer.addErrorListener(MathErrorListener())

    stream = CommonTokenStream(lexer)
    parser = PetabMathExprParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(MathErrorListener())

    # Parse expression
    try:
        tree = parser.petabExpression()
    except ValueError as e:
        raise ValueError(f"Error parsing {expr!r}: {e.args[0]}") from None

    # Convert to sympy expression
    visitor = MathVisitorSympy(evaluate=evaluate)
    expr = visitor.visit(tree)
    expr = bool2num(expr)
    # check for `False`, we'll accept both `True` and `None`
    if expr.is_extended_real is False:
        raise ValueError(f"Expression {expr} is not real-valued.")

    return expr


class MathErrorListener(ErrorListener):
    """Error listener for math expression parser/lexer."""

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):  # noqa N803
        raise ValueError(f"Syntax error at {line}:{column}: {msg}")
