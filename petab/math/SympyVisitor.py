"""PEtab-math to sympy conversion."""
import sympy as sp
from sympy.logic.boolalg import Boolean, BooleanFalse, BooleanTrue

from ._generated.PetabMathExprParser import PetabMathExprParser
from ._generated.PetabMathExprParserVisitor import PetabMathExprParserVisitor

__all__ = ["MathVisitorSympy"]

# Mappings of PEtab math functions to sympy functions

# trigonometric functions
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
    "log10": lambda x: -sp.oo if x.is_zero is True else sp.log(x, 10),
    "log2": lambda x: -sp.oo if x.is_zero is True else sp.log(x, 2),
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

# reserved names that cannot be used as variable names
_reserved_names = {
    "inf",
    "nan",
    "true",
    "false",
}


class MathVisitorSympy(PetabMathExprParserVisitor):
    """
    ANTLR4 visitor for PEtab-math-to-sympy conversion.

    Visitor for PEtab math expression AST generated using ANTLR4.
    Converts PEtab math expressions to sympy expressions.

    Most users will not need to interact with this class directly, but rather
    use :func:`petab.math.sympify_petab`.

    Evaluation of any sub-expressions currently relies on sympy's defaults.

    For a general introduction to ANTLR4 visitors, see:
    https://github.com/antlr/antlr4/blob/7d4cea92bc3f7d709f09c3f1ac77c5bbc71a6749/doc/python-target.md
    """

    def visitPetabExpression(
        self, ctx: PetabMathExprParser.PetabExpressionContext
    ) -> sp.Expr | sp.Basic:
        """Visit the root of the expression tree."""
        return self.visit(ctx.getChild(0))

    def visitNumber(self, ctx: PetabMathExprParser.NumberContext) -> sp.Float:
        """Convert number to sympy Float."""
        return sp.Float(ctx.getText())

    def visitVar(self, ctx: PetabMathExprParser.VarContext) -> sp.Symbol:
        """Convert identifier to sympy Symbol."""
        if ctx.getText().lower() in _reserved_names:
            raise ValueError(f"Use of reserved name {ctx.getText()!r}")
        return sp.Symbol(ctx.getText(), real=True)

    def visitMultExpr(
        self, ctx: PetabMathExprParser.MultExprContext
    ) -> sp.Expr:
        """Convert multiplication and division expressions to sympy."""
        if ctx.getChildCount() == 3:
            operand1 = bool2num(self.visit(ctx.getChild(0)))
            operand2 = bool2num(self.visit(ctx.getChild(2)))
            if ctx.ASTERISK():
                return operand1 * operand2
            if ctx.SLASH():
                return operand1 / operand2

        raise AssertionError(f"Unexpected expression: {ctx.getText()}")

    def visitAddExpr(self, ctx: PetabMathExprParser.AddExprContext) -> sp.Expr:
        """Convert addition and subtraction expressions to sympy."""
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

    def visitArgumentList(
        self, ctx: PetabMathExprParser.ArgumentListContext
    ) -> list[sp.Basic | sp.Expr]:
        """Convert function argument lists to a list of sympy expressions."""
        return [self.visit(c) for c in ctx.children[::2]]

    def visitFunctionCall(
        self, ctx: PetabMathExprParser.FunctionCallContext
    ) -> sp.Expr:
        """Convert function call to sympy expression."""
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
            return -sp.oo if args[0].is_zero is True else sp.log(*args)

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
        """Convert parenthesized expression to sympy."""
        return self.visit(ctx.getChild(1))

    def visitPowerExpr(
        self, ctx: PetabMathExprParser.PowerExprContext
    ) -> sp.Pow:
        """Convert power expression to sympy."""
        if ctx.getChildCount() != 3:
            raise AssertionError(
                f"Unexpected number of children: {ctx.getChildCount()} "
                f"in {ctx.getText()}"
            )
        operand1 = bool2num(self.visit(ctx.getChild(0)))
        operand2 = bool2num(self.visit(ctx.getChild(2)))
        return sp.Pow(operand1, operand2)

    def visitUnaryExpr(
        self, ctx: PetabMathExprParser.UnaryExprContext
    ) -> sp.Basic | sp.Expr:
        """Convert unary expressions to sympy."""
        if ctx.getChildCount() == 2:
            operand = bool2num(self.visit(ctx.getChild(1)))
            match ctx.getChild(0).getText():
                case "-":
                    return -operand
                case "+":
                    return operand

        raise AssertionError(f"Unexpected expression: {ctx.getText()}")

    def visitComparisonExpr(
        self, ctx: PetabMathExprParser.ComparisonExprContext
    ) -> sp.Basic | sp.Expr:
        """Convert comparison expressions to sympy."""
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
    ) -> sp.Basic | sp.Expr:
        """Convert boolean NOT expressions to sympy."""
        if ctx.getChildCount() == 2:
            return ~num2bool(self.visit(ctx.getChild(1)))

        raise AssertionError(f"Unexpected expression: {ctx.getText()}")

    def visitBooleanAndOrExpr(
        self, ctx: PetabMathExprParser.BooleanAndOrExprContext
    ) -> sp.Basic | sp.Expr:
        """Convert boolean AND and OR expressions to sympy."""
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
    ) -> Boolean:
        """Convert boolean literals to sympy."""
        if ctx.TRUE():
            return sp.true

        if ctx.FALSE():
            return sp.false

        raise AssertionError(f"Unexpected boolean literal: {ctx.getText()}")


def bool2num(x: sp.Basic | sp.Expr) -> sp.Basic | sp.Expr:
    """Convert sympy Booleans to Floats."""
    if isinstance(x, BooleanFalse):
        return sp.Float(0)
    if isinstance(x, BooleanTrue):
        return sp.Float(1)
    return x


def num2bool(x: sp.Basic | sp.Expr) -> sp.Basic | sp.Expr:
    """Convert sympy Floats to booleans."""
    if isinstance(x, BooleanTrue | BooleanFalse):
        return x
    # Note: sp.Float(0) == 0 is False in sympy>=1.13
    if x.is_zero is True:
        return sp.false
    if x.is_zero is False:
        return sp.true
    return sp.Piecewise((True, x != 0.0), (False, True))
