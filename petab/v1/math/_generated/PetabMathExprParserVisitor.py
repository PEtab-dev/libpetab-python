# Generated from PetabMathExprParser.g4 by ANTLR 4.13.1
from antlr4 import *

if "." in __name__:
    from .PetabMathExprParser import PetabMathExprParser
else:
    from PetabMathExprParser import PetabMathExprParser

# This class defines a complete generic visitor for a parse tree produced by PetabMathExprParser.


class PetabMathExprParserVisitor(ParseTreeVisitor):
    # Visit a parse tree produced by PetabMathExprParser#petabExpression.
    def visitPetabExpression(
        self, ctx: PetabMathExprParser.PetabExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#PowerExpr.
    def visitPowerExpr(self, ctx: PetabMathExprParser.PowerExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#BooleanAndOrExpr.
    def visitBooleanAndOrExpr(
        self, ctx: PetabMathExprParser.BooleanAndOrExprContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#ComparisonExpr.
    def visitComparisonExpr(
        self, ctx: PetabMathExprParser.ComparisonExprContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#MultExpr.
    def visitMultExpr(self, ctx: PetabMathExprParser.MultExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#BooleanLiteral_.
    def visitBooleanLiteral_(
        self, ctx: PetabMathExprParser.BooleanLiteral_Context
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#AddExpr.
    def visitAddExpr(self, ctx: PetabMathExprParser.AddExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#BooleanNotExpr.
    def visitBooleanNotExpr(
        self, ctx: PetabMathExprParser.BooleanNotExprContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#ParenExpr.
    def visitParenExpr(self, ctx: PetabMathExprParser.ParenExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#functionCall_.
    def visitFunctionCall_(
        self, ctx: PetabMathExprParser.FunctionCall_Context
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#UnaryExpr.
    def visitUnaryExpr(self, ctx: PetabMathExprParser.UnaryExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#Number_.
    def visitNumber_(self, ctx: PetabMathExprParser.Number_Context):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#VarExpr_.
    def visitVarExpr_(self, ctx: PetabMathExprParser.VarExpr_Context):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#comp_op.
    def visitComp_op(self, ctx: PetabMathExprParser.Comp_opContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#argumentList.
    def visitArgumentList(self, ctx: PetabMathExprParser.ArgumentListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#functionCall.
    def visitFunctionCall(self, ctx: PetabMathExprParser.FunctionCallContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#booleanLiteral.
    def visitBooleanLiteral(
        self, ctx: PetabMathExprParser.BooleanLiteralContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#number.
    def visitNumber(self, ctx: PetabMathExprParser.NumberContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PetabMathExprParser#var.
    def visitVar(self, ctx: PetabMathExprParser.VarContext):
        return self.visitChildren(ctx)


del PetabMathExprParser
