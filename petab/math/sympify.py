"""PEtab math to sympy conversion."""

import sympy as sp
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from petab.math._generated.PetabMathExprLexer import PetabMathExprLexer
from petab.math._generated.PetabMathExprParser import PetabMathExprParser
from petab.math.SympyVisitor import MathVisitorSympy, bool2num


def sympify_petab(expr: str | int | float) -> sp.Expr:
    """Convert PEtab math expression to sympy expression.

    Args:
        expr: PEtab math expression.

    Returns:
        Sympy expression.
    """
    if isinstance(expr, int):
        return sp.Integer(expr)
    if isinstance(expr, float):
        return sp.Float(expr)

    input_stream = InputStream(expr)
    lexer = PetabMathExprLexer(input_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(MathErrorListener())

    stream = CommonTokenStream(lexer)
    parser = PetabMathExprParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(MathErrorListener())
    try:
        tree = parser.prog()
    except ValueError as e:
        raise ValueError(f"Error parsing {expr!r}: {e.args[0]}") from None
    visitor = MathVisitorSympy()

    expr = visitor.visit(tree)
    expr = bool2num(expr)
    if not expr.is_real:
        if hasattr(expr, "is_infinite") and expr.is_infinite:
            return expr
        raise ValueError(f"Expression {expr} is not real-valued.")

    return expr


class MathErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):  # noqa N803
        raise ValueError(f"Syntax error at {line}:{column}: {msg}")
