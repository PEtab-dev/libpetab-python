"""PEtab math to sympy conversion."""

import numpy as np
import sympy as sp
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from ._generated.PetabMathExprLexer import PetabMathExprLexer
from ._generated.PetabMathExprParser import PetabMathExprParser
from .SympyVisitor import MathVisitorSympy, bool2num

__all__ = ["sympify_petab"]


def sympify_petab(expr: str | int | float) -> sp.Expr | sp.Basic:
    """Convert PEtab math expression to sympy expression.

    Args:
        expr: PEtab math expression.

    Raises:
        ValueError: Upon lexer/parser errors or if the expression is
        otherwise invalid.

    Returns:
        The sympy expression corresponding to `expr`.
        Boolean values are converted to numeric values.
    """
    if isinstance(expr, int) or isinstance(expr, np.integer):
        return sp.Integer(expr)
    if isinstance(expr, float) or isinstance(expr, np.floating):
        return sp.Float(expr)

    # Set error listeners
    input_stream = InputStream(expr)
    lexer = PetabMathExprLexer(input_stream)
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
    visitor = MathVisitorSympy()
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
