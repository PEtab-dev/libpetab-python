"""PEtab-math to sympy conversion."""
import sympy as sp
from sympy.logic.boolalg import Boolean

from ._generated.PetabMathExprParser import PetabMathExprParser
from ._generated.PetabMathExprParserVisitor import PetabMathExprParserVisitor

__all__ = ["MathVisitorSympy"]

_trig_funcs = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "sech": sp.sech,
    "csch": sp.csch,
    "coth": sp.coth,
    "arccos": sp.acos,
    "arcsin": sp.asin,
    "arctan": sp.atan,
    "arcsec": sp.asec,
    "arccsc": sp.acsc,
    "arccot": sp.acot,
    "arcsinh": sp.asinh,
    "arccosh": sp.acosh,
    "arctanh": sp.atanh,
    "arcsech": sp.asech,
    "arccsch": sp.acsch,
    "arccoth": sp.acoth,
}
_unary_funcs = {
    "exp": sp.exp,
    "log10": lambda x: sp.log(x, 10),
    "log2": lambda x: sp.log(x, 2),
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "sign": sp.sign,
}
_binary_funcs = {
    "pow": sp.Pow,
    "min": sp.Min,
    "max": sp.Max,
}


class MathVisitorSympy(PetabMathExprParserVisitor):
    """
    Sympy-based visitor for the math expression parser.

    Visitor for the math expression parser that converts the parse tree to a
    sympy expression.
    """

    def visitNumber(self, ctx: PetabMathExprParser.NumberContext):
        return sp.Float(ctx.getText())

    def visitVar(self, ctx: PetabMathExprParser.VarContext):
        return sp.Symbol(ctx.getText())

    def visitMultExpr(self, ctx: PetabMathExprParser.MultExprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))
        if ctx.getChildCount() == 3:
            if ctx.MUL():
                return self.visit(ctx.getChild(0)) * self.visit(
                    ctx.getChild(2)
                )
            if ctx.DIV():
                return self.visit(ctx.getChild(0)) / self.visit(
                    ctx.getChild(2)
                )
        raise AssertionError(f"Unexpected expression: {ctx.getText()}")

    def visitAddExpr(self, ctx: PetabMathExprParser.AddExprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))
        if ctx.PLUS():
            return self.visit(ctx.getChild(0)) + self.visit(ctx.getChild(2))
        if ctx.MINUS():
            return self.visit(ctx.getChild(0)) - self.visit(ctx.getChild(2))
        raise AssertionError(
            f"Unexpected operator: {ctx.getChild(1).getText()} "
            f"in {ctx.getText()}"
        )

    def visitArgumentList(self, ctx: PetabMathExprParser.ArgumentListContext):
        return [self.visit(c) for c in ctx.children[::2]]

    def visitFunc_expr(self, ctx: PetabMathExprParser.Func_exprContext):
        if ctx.getChildCount() < 4:
            raise AssertionError(f"Unexpected expression: {ctx.getText()}")
        func_name = ctx.getChild(0).getText()
        args = self.visit(ctx.getChild(2))
        if func_name in _trig_funcs:
            if len(args) != 1:
                raise AssertionError(
                    f"Unexpected number of arguments: {len(args)} "
                    f"in {ctx.getText()}"
                )
            return _trig_funcs[func_name](*args)
        if func_name in _unary_funcs:
            if len(args) != 1:
                raise AssertionError(
                    f"Unexpected number of arguments: {len(args)} "
                    f"in {ctx.getText()}"
                )
            return _unary_funcs[func_name](*args)
        if func_name in _binary_funcs:
            if len(args) != 2:
                raise AssertionError(
                    f"Unexpected number of arguments: {len(args)} "
                    f"in {ctx.getText()}"
                )
            return _binary_funcs[func_name](*args)
        if func_name == "log":
            if len(args) not in [1, 2]:
                raise AssertionError(
                    f"Unexpected number of arguments: {len(args)} "
                    f"in {ctx.getText()}"
                )
            return sp.log(*args)

        if func_name == "piecewise":
            if (len(args) - 1) % 2 != 0:
                raise AssertionError(
                    f"Unexpected number of arguments: {len(args)} "
                    f"in {ctx.getText()}"
                )
            if not all(isinstance(arg, Boolean) for arg in args[1::2]):
                raise AssertionError(
                    f"Expected boolean conditions in {ctx.getText()}"
                )
            # sympy's Piecewise requires an explicit condition for the final
            # `else` case
            args.append(sp.true)
            sp_args = (
                (true_expr, condition)
                for true_expr, condition in zip(
                    args[::2], args[1::2], strict=True
                )
            )
            return sp.Piecewise(*sp_args)

        raise ValueError(f"Unknown function: {ctx.getText()}")

    def visitParenExpr(self, ctx: PetabMathExprParser.ParenExprContext):
        return self.visit(ctx.getChild(1))

    def visitHatExpr(self, ctx: PetabMathExprParser.HatExprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))
        if ctx.getChildCount() != 3:
            raise AssertionError(
                f"Unexpected number of children: {ctx.getChildCount()} "
                f"in {ctx.getText()}"
            )
        return sp.Pow(self.visit(ctx.getChild(0)), self.visit(ctx.getChild(2)))

    def visitUnaryExpr(self, ctx: PetabMathExprParser.UnaryExprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))
        if ctx.getChildCount() == 2:
            match ctx.getChild(0).getText():
                case "-":
                    return -self.visit(ctx.getChild(1))
                case "+":
                    return self.visit(ctx.getChild(1))
        raise AssertionError(f"Unexpected expression: {ctx.getText()}")

    def visitProg(self, ctx: PetabMathExprParser.ProgContext):
        return self.visit(ctx.getChild(0))

    def visitBooleanAtom(self, ctx: PetabMathExprParser.BooleanAtomContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))
        if ctx.getChildCount() == 3 and ctx.OPEN_PAREN() and ctx.CLOSE_PAREN():
            return self.visit(ctx.getChild(1))
        raise AssertionError(f"Unexpected boolean atom: {ctx.getText()}")

    def visitComparisonExpr(
        self, ctx: PetabMathExprParser.ComparisonExprContext
    ):
        if ctx.getChildCount() != 1:
            raise AssertionError(f"Unexpected expression: {ctx.getText()}")
        ctx = ctx.getChild(0)
        if ctx.getChildCount() != 3:
            raise AssertionError(f"Unexpected expression: {ctx.getText()}")
        lhs = self.visit(ctx.getChild(0))
        op = ctx.getChild(1).getText()
        rhs = self.visit(ctx.getChild(2))

        ops = {
            "==": sp.Equality,
            "!=": sp.Unequality,
            "<": sp.StrictLessThan,
            ">": sp.StrictGreaterThan,
            "<=": sp.LessThan,
            ">=": sp.GreaterThan,
        }
        if op in ops:
            return ops[op](lhs, rhs)
        raise AssertionError(f"Unexpected operator: {op}")

    def visitBooleanNotExpr(
        self, ctx: PetabMathExprParser.BooleanNotExprContext
    ):
        if ctx.getChildCount() == 2:
            return ~self.visit(ctx.getChild(1))
        elif ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))
        raise AssertionError(f"Unexpected expression: {ctx.getText()}")

    def visitBooleanAndOrExpr(
        self, ctx: PetabMathExprParser.BooleanAndOrExprContext
    ):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))
        if ctx.getChildCount() != 3:
            raise AssertionError(f"Unexpected expression: {ctx.getText()}")

        if ctx.BOOLEAN_AND():
            return self.visit(ctx.getChild(0)) & self.visit(ctx.getChild(2))
        if ctx.BOOLEAN_OR():
            return self.visit(ctx.getChild(0)) | self.visit(ctx.getChild(2))

        raise AssertionError(f"Unexpected expression: {ctx.getText()}")

    def visitBooleanLiteral(
        self, ctx: PetabMathExprParser.BooleanLiteralContext
    ):
        if ctx.TRUE():
            return sp.true
        if ctx.FALSE():
            return sp.false
        raise AssertionError(f"Unexpected boolean literal: {ctx.getText()}")
