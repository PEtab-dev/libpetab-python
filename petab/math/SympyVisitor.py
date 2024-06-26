"""PEtab-math to sympy conversion."""
import sympy as sp
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

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

_reserved_names = {
    "inf",
    "nan",
    "true",
    "false",
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
        if ctx.getText().lower() in _reserved_names:
            raise ValueError(f"Use of reserved name {ctx.getText()!r}")
        return sp.Symbol(ctx.getText())

    def visitMultExpr(self, ctx: PetabMathExprParser.MultExprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.getChild(0))
        if ctx.getChildCount() == 3:
            operand1 = bool2num(self.visit(ctx.getChild(0)))
            operand2 = bool2num(self.visit(ctx.getChild(2)))
            if ctx.MUL():
                return operand1 * operand2
            if ctx.DIV():
                return operand1 / operand2
        raise AssertionError(f"Unexpected expression: {ctx.getText()}")

    def visitAddExpr(self, ctx: PetabMathExprParser.AddExprContext):
        if ctx.getChildCount() == 1:
            return bool2num(self.visit(ctx.getChild(0)))
        op1 = bool2num(self.visit(ctx.getChild(0)))
        op2 = bool2num(self.visit(ctx.getChild(2)))
        if ctx.PLUS():
            return op1 + op2
        if ctx.MINUS():
            return op1 - op2
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

        if func_name != "piecewise":
            # all functions except piecewise expect numerical arguments
            args = list(map(bool2num, args))

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
            # sympy's Piecewise requires an explicit condition for the final
            # `else` case
            args.append(sp.true)
            sp_args = (
                (true_expr, num2bool(condition))
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
            operand = bool2num(self.visit(ctx.getChild(1)))
            match ctx.getChild(0).getText():
                case "-":
                    return -operand
                case "+":
                    return operand
        raise AssertionError(f"Unexpected expression: {ctx.getText()}")

    def visitProg(self, ctx: PetabMathExprParser.ProgContext):
        return self.visit(ctx.getChild(0))

    def visitComparisonExpr(
        self, ctx: PetabMathExprParser.ComparisonExprContext
    ):
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
            lhs = bool2num(lhs)
            rhs = bool2num(rhs)
            return ops[op](lhs, rhs)
        raise AssertionError(f"Unexpected operator: {op}")

    def visitBooleanNotExpr(
        self, ctx: PetabMathExprParser.BooleanNotExprContext
    ):
        if ctx.getChildCount() == 2:
            return ~num2bool(self.visit(ctx.getChild(1)))
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

        operand1 = num2bool(self.visit(ctx.getChild(0)))
        operand2 = num2bool(self.visit(ctx.getChild(2)))

        if ctx.BOOLEAN_AND():
            return operand1 & operand2
        if ctx.BOOLEAN_OR():
            return operand1 | operand2

        raise AssertionError(f"Unexpected expression: {ctx.getText()}")

    def visitBooleanLiteral(
        self, ctx: PetabMathExprParser.BooleanLiteralContext
    ):
        if ctx.TRUE():
            return sp.true
        if ctx.FALSE():
            return sp.false
        raise AssertionError(f"Unexpected boolean literal: {ctx.getText()}")


def bool2num(x: sp.Basic):
    """Convert sympy Booleans to Floats."""
    if isinstance(x, BooleanFalse):
        return sp.Float(0)
    if isinstance(x, BooleanTrue):
        return sp.Float(1)
    return x


def num2bool(x: sp.Basic):
    """Convert sympy Floats to booleans."""
    if isinstance(x, BooleanTrue | BooleanFalse):
        return x
    # Note: sp.Float(0) == 0 is False in sympy>=1.13
    if x.is_zero is True:
        return sp.false
    if x.is_zero is False:
        return sp.true
    return sp.Piecewise((True, x != 0.0), (False, True))
