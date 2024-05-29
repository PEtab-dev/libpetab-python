# Generated from PetabMathExprParser.g4 by ANTLR 4.13.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,27,118,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        1,0,1,0,1,0,1,1,1,1,3,1,34,8,1,1,2,1,2,1,3,1,3,1,3,1,3,1,3,1,3,1,
        3,1,3,1,3,1,3,3,3,48,8,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,5,3,
        59,8,3,10,3,12,3,62,9,3,1,4,1,4,1,4,5,4,67,8,4,10,4,12,4,70,9,4,
        1,5,1,5,1,5,1,5,1,5,1,6,1,6,1,6,1,6,3,6,81,8,6,1,6,1,6,1,6,5,6,86,
        8,6,10,6,12,6,89,9,6,1,7,1,7,1,7,1,7,1,7,1,7,1,7,3,7,98,8,7,1,8,
        1,8,3,8,102,8,8,1,9,1,9,1,9,1,9,1,10,1,10,1,10,1,10,1,11,1,11,1,
        12,1,12,1,13,1,13,1,13,0,2,6,12,14,0,2,4,6,8,10,12,14,16,18,20,22,
        24,26,0,5,1,0,15,20,1,0,21,22,1,0,23,24,1,0,13,14,1,0,7,8,118,0,
        28,1,0,0,0,2,33,1,0,0,0,4,35,1,0,0,0,6,47,1,0,0,0,8,63,1,0,0,0,10,
        71,1,0,0,0,12,80,1,0,0,0,14,97,1,0,0,0,16,101,1,0,0,0,18,103,1,0,
        0,0,20,107,1,0,0,0,22,111,1,0,0,0,24,113,1,0,0,0,26,115,1,0,0,0,
        28,29,3,2,1,0,29,30,5,0,0,1,30,1,1,0,0,0,31,34,3,6,3,0,32,34,3,12,
        6,0,33,31,1,0,0,0,33,32,1,0,0,0,34,3,1,0,0,0,35,36,7,0,0,0,36,5,
        1,0,0,0,37,38,6,3,-1,0,38,39,7,1,0,0,39,48,3,6,3,7,40,41,5,11,0,
        0,41,42,3,6,3,0,42,43,5,12,0,0,43,48,1,0,0,0,44,48,3,24,12,0,45,
        48,3,10,5,0,46,48,3,26,13,0,47,37,1,0,0,0,47,40,1,0,0,0,47,44,1,
        0,0,0,47,45,1,0,0,0,47,46,1,0,0,0,48,60,1,0,0,0,49,50,10,8,0,0,50,
        51,5,25,0,0,51,59,3,6,3,8,52,53,10,6,0,0,53,54,7,2,0,0,54,59,3,6,
        3,7,55,56,10,5,0,0,56,57,7,1,0,0,57,59,3,6,3,6,58,49,1,0,0,0,58,
        52,1,0,0,0,58,55,1,0,0,0,59,62,1,0,0,0,60,58,1,0,0,0,60,61,1,0,0,
        0,61,7,1,0,0,0,62,60,1,0,0,0,63,68,3,2,1,0,64,65,5,27,0,0,65,67,
        3,2,1,0,66,64,1,0,0,0,67,70,1,0,0,0,68,66,1,0,0,0,68,69,1,0,0,0,
        69,9,1,0,0,0,70,68,1,0,0,0,71,72,5,10,0,0,72,73,5,11,0,0,73,74,3,
        8,4,0,74,75,5,12,0,0,75,11,1,0,0,0,76,77,6,6,-1,0,77,78,5,26,0,0,
        78,81,3,14,7,0,79,81,3,14,7,0,80,76,1,0,0,0,80,79,1,0,0,0,81,87,
        1,0,0,0,82,83,10,2,0,0,83,84,7,3,0,0,84,86,3,12,6,3,85,82,1,0,0,
        0,86,89,1,0,0,0,87,85,1,0,0,0,87,88,1,0,0,0,88,13,1,0,0,0,89,87,
        1,0,0,0,90,98,3,22,11,0,91,92,5,11,0,0,92,93,3,12,6,0,93,94,5,12,
        0,0,94,98,1,0,0,0,95,98,3,16,8,0,96,98,3,26,13,0,97,90,1,0,0,0,97,
        91,1,0,0,0,97,95,1,0,0,0,97,96,1,0,0,0,98,15,1,0,0,0,99,102,3,20,
        10,0,100,102,3,18,9,0,101,99,1,0,0,0,101,100,1,0,0,0,102,17,1,0,
        0,0,103,104,3,22,11,0,104,105,3,4,2,0,105,106,3,22,11,0,106,19,1,
        0,0,0,107,108,3,6,3,0,108,109,3,4,2,0,109,110,3,6,3,0,110,21,1,0,
        0,0,111,112,7,4,0,0,112,23,1,0,0,0,113,114,5,1,0,0,114,25,1,0,0,
        0,115,116,5,10,0,0,116,27,1,0,0,0,9,33,47,58,60,68,80,87,97,101
    ]

class PetabMathExprParser ( Parser ):

    grammarFileName = "PetabMathExprParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                     "<INVALID>", "<INVALID>", "<INVALID>", "'true'", "'false'",
                     "'inf'", "<INVALID>", "'('", "')'", "'||'", "'&&'",
                     "'>'", "'<'", "'>='", "'<='", "'=='", "'!='", "'+'",
                     "'-'", "'*'", "'/'", "'^'", "'!'", "','" ]

    symbolicNames = [ "<INVALID>", "NUMBER", "INTEGER", "EXPONENT_FLOAT",
                      "POINT_FLOAT", "FLOAT_NUMBER", "WS", "TRUE", "FALSE",
                      "INF", "NAME", "OPEN_PAREN", "CLOSE_PAREN", "BOOLEAN_OR",
                      "BOOLEAN_AND", "GT", "LT", "GTE", "LTE", "EQ", "NEQ",
                      "PLUS", "MINUS", "MUL", "DIV", "HAT", "NOT", "COMMA" ]

    RULE_prog = 0
    RULE_expr = 1
    RULE_comp_op = 2
    RULE_arithmeticExpr = 3
    RULE_argumentList = 4
    RULE_func_expr = 5
    RULE_booleanExpr = 6
    RULE_booleanAtom = 7
    RULE_comparisonExpr = 8
    RULE_boolComparisonExpr = 9
    RULE_floatComparisonExpr = 10
    RULE_booleanLiteral = 11
    RULE_number = 12
    RULE_var = 13

    ruleNames =  [ "prog", "expr", "comp_op", "arithmeticExpr", "argumentList",
                   "func_expr", "booleanExpr", "booleanAtom", "comparisonExpr",
                   "boolComparisonExpr", "floatComparisonExpr", "booleanLiteral",
                   "number", "var" ]

    EOF = Token.EOF
    NUMBER=1
    INTEGER=2
    EXPONENT_FLOAT=3
    POINT_FLOAT=4
    FLOAT_NUMBER=5
    WS=6
    TRUE=7
    FALSE=8
    INF=9
    NAME=10
    OPEN_PAREN=11
    CLOSE_PAREN=12
    BOOLEAN_OR=13
    BOOLEAN_AND=14
    GT=15
    LT=16
    GTE=17
    LTE=18
    EQ=19
    NEQ=20
    PLUS=21
    MINUS=22
    MUL=23
    DIV=24
    HAT=25
    NOT=26
    COMMA=27

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(PetabMathExprParser.ExprContext,0)


        def EOF(self):
            return self.getToken(PetabMathExprParser.EOF, 0)

        def getRuleIndex(self):
            return PetabMathExprParser.RULE_prog

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProg" ):
                return visitor.visitProg(self)
            else:
                return visitor.visitChildren(self)




    def prog(self):

        localctx = PetabMathExprParser.ProgContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_prog)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 28
            self.expr()
            self.state = 29
            self.match(PetabMathExprParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def arithmeticExpr(self):
            return self.getTypedRuleContext(PetabMathExprParser.ArithmeticExprContext,0)


        def booleanExpr(self):
            return self.getTypedRuleContext(PetabMathExprParser.BooleanExprContext,0)


        def getRuleIndex(self):
            return PetabMathExprParser.RULE_expr

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExpr" ):
                return visitor.visitExpr(self)
            else:
                return visitor.visitChildren(self)




    def expr(self):

        localctx = PetabMathExprParser.ExprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_expr)
        try:
            self.state = 33
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 31
                self.arithmeticExpr(0)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 32
                self.booleanExpr(0)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Comp_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def GT(self):
            return self.getToken(PetabMathExprParser.GT, 0)

        def LT(self):
            return self.getToken(PetabMathExprParser.LT, 0)

        def GTE(self):
            return self.getToken(PetabMathExprParser.GTE, 0)

        def LTE(self):
            return self.getToken(PetabMathExprParser.LTE, 0)

        def EQ(self):
            return self.getToken(PetabMathExprParser.EQ, 0)

        def NEQ(self):
            return self.getToken(PetabMathExprParser.NEQ, 0)

        def getRuleIndex(self):
            return PetabMathExprParser.RULE_comp_op

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitComp_op" ):
                return visitor.visitComp_op(self)
            else:
                return visitor.visitChildren(self)




    def comp_op(self):

        localctx = PetabMathExprParser.Comp_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_comp_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 35
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 2064384) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ArithmeticExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return PetabMathExprParser.RULE_arithmeticExpr


        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)


    class FuncExpr_Context(ArithmeticExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.ArithmeticExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def func_expr(self):
            return self.getTypedRuleContext(PetabMathExprParser.Func_exprContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFuncExpr_" ):
                return visitor.visitFuncExpr_(self)
            else:
                return visitor.visitChildren(self)


    class MultExprContext(ArithmeticExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.ArithmeticExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def arithmeticExpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PetabMathExprParser.ArithmeticExprContext)
            else:
                return self.getTypedRuleContext(PetabMathExprParser.ArithmeticExprContext,i)

        def MUL(self):
            return self.getToken(PetabMathExprParser.MUL, 0)
        def DIV(self):
            return self.getToken(PetabMathExprParser.DIV, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMultExpr" ):
                return visitor.visitMultExpr(self)
            else:
                return visitor.visitChildren(self)


    class HatExprContext(ArithmeticExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.ArithmeticExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def arithmeticExpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PetabMathExprParser.ArithmeticExprContext)
            else:
                return self.getTypedRuleContext(PetabMathExprParser.ArithmeticExprContext,i)

        def HAT(self):
            return self.getToken(PetabMathExprParser.HAT, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitHatExpr" ):
                return visitor.visitHatExpr(self)
            else:
                return visitor.visitChildren(self)


    class AddExprContext(ArithmeticExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.ArithmeticExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def arithmeticExpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PetabMathExprParser.ArithmeticExprContext)
            else:
                return self.getTypedRuleContext(PetabMathExprParser.ArithmeticExprContext,i)

        def PLUS(self):
            return self.getToken(PetabMathExprParser.PLUS, 0)
        def MINUS(self):
            return self.getToken(PetabMathExprParser.MINUS, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAddExpr" ):
                return visitor.visitAddExpr(self)
            else:
                return visitor.visitChildren(self)


    class ParenExprContext(ArithmeticExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.ArithmeticExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def OPEN_PAREN(self):
            return self.getToken(PetabMathExprParser.OPEN_PAREN, 0)
        def arithmeticExpr(self):
            return self.getTypedRuleContext(PetabMathExprParser.ArithmeticExprContext,0)

        def CLOSE_PAREN(self):
            return self.getToken(PetabMathExprParser.CLOSE_PAREN, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitParenExpr" ):
                return visitor.visitParenExpr(self)
            else:
                return visitor.visitChildren(self)


    class UnaryExprContext(ArithmeticExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.ArithmeticExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def arithmeticExpr(self):
            return self.getTypedRuleContext(PetabMathExprParser.ArithmeticExprContext,0)

        def PLUS(self):
            return self.getToken(PetabMathExprParser.PLUS, 0)
        def MINUS(self):
            return self.getToken(PetabMathExprParser.MINUS, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitUnaryExpr" ):
                return visitor.visitUnaryExpr(self)
            else:
                return visitor.visitChildren(self)


    class Number_Context(ArithmeticExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.ArithmeticExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def number(self):
            return self.getTypedRuleContext(PetabMathExprParser.NumberContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitNumber_" ):
                return visitor.visitNumber_(self)
            else:
                return visitor.visitChildren(self)


    class VarExpr_Context(ArithmeticExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.ArithmeticExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def var(self):
            return self.getTypedRuleContext(PetabMathExprParser.VarContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVarExpr_" ):
                return visitor.visitVarExpr_(self)
            else:
                return visitor.visitChildren(self)



    def arithmeticExpr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = PetabMathExprParser.ArithmeticExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 6
        self.enterRecursionRule(localctx, 6, self.RULE_arithmeticExpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 47
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
            if la_ == 1:
                localctx = PetabMathExprParser.UnaryExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 38
                _la = self._input.LA(1)
                if not(_la==21 or _la==22):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 39
                self.arithmeticExpr(7)
                pass

            elif la_ == 2:
                localctx = PetabMathExprParser.ParenExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 40
                self.match(PetabMathExprParser.OPEN_PAREN)
                self.state = 41
                self.arithmeticExpr(0)
                self.state = 42
                self.match(PetabMathExprParser.CLOSE_PAREN)
                pass

            elif la_ == 3:
                localctx = PetabMathExprParser.Number_Context(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 44
                self.number()
                pass

            elif la_ == 4:
                localctx = PetabMathExprParser.FuncExpr_Context(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 45
                self.func_expr()
                pass

            elif la_ == 5:
                localctx = PetabMathExprParser.VarExpr_Context(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 46
                self.var()
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 60
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,3,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 58
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
                    if la_ == 1:
                        localctx = PetabMathExprParser.HatExprContext(self, PetabMathExprParser.ArithmeticExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_arithmeticExpr)
                        self.state = 49
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                        self.state = 50
                        self.match(PetabMathExprParser.HAT)
                        self.state = 51
                        self.arithmeticExpr(8)
                        pass

                    elif la_ == 2:
                        localctx = PetabMathExprParser.MultExprContext(self, PetabMathExprParser.ArithmeticExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_arithmeticExpr)
                        self.state = 52
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 53
                        _la = self._input.LA(1)
                        if not(_la==23 or _la==24):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 54
                        self.arithmeticExpr(7)
                        pass

                    elif la_ == 3:
                        localctx = PetabMathExprParser.AddExprContext(self, PetabMathExprParser.ArithmeticExprContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_arithmeticExpr)
                        self.state = 55
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 56
                        _la = self._input.LA(1)
                        if not(_la==21 or _la==22):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 57
                        self.arithmeticExpr(6)
                        pass


                self.state = 62
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,3,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class ArgumentListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PetabMathExprParser.ExprContext)
            else:
                return self.getTypedRuleContext(PetabMathExprParser.ExprContext,i)


        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(PetabMathExprParser.COMMA)
            else:
                return self.getToken(PetabMathExprParser.COMMA, i)

        def getRuleIndex(self):
            return PetabMathExprParser.RULE_argumentList

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitArgumentList" ):
                return visitor.visitArgumentList(self)
            else:
                return visitor.visitChildren(self)




    def argumentList(self):

        localctx = PetabMathExprParser.ArgumentListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_argumentList)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 63
            self.expr()
            self.state = 68
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==27:
                self.state = 64
                self.match(PetabMathExprParser.COMMA)
                self.state = 65
                self.expr()
                self.state = 70
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Func_exprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NAME(self):
            return self.getToken(PetabMathExprParser.NAME, 0)

        def OPEN_PAREN(self):
            return self.getToken(PetabMathExprParser.OPEN_PAREN, 0)

        def argumentList(self):
            return self.getTypedRuleContext(PetabMathExprParser.ArgumentListContext,0)


        def CLOSE_PAREN(self):
            return self.getToken(PetabMathExprParser.CLOSE_PAREN, 0)

        def getRuleIndex(self):
            return PetabMathExprParser.RULE_func_expr

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFunc_expr" ):
                return visitor.visitFunc_expr(self)
            else:
                return visitor.visitChildren(self)




    def func_expr(self):

        localctx = PetabMathExprParser.Func_exprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_func_expr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 71
            self.match(PetabMathExprParser.NAME)
            self.state = 72
            self.match(PetabMathExprParser.OPEN_PAREN)
            self.state = 73
            self.argumentList()
            self.state = 74
            self.match(PetabMathExprParser.CLOSE_PAREN)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BooleanExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return PetabMathExprParser.RULE_booleanExpr


        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)


    class BooleanAtomExprContext(BooleanExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.BooleanExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def booleanAtom(self):
            return self.getTypedRuleContext(PetabMathExprParser.BooleanAtomContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBooleanAtomExpr" ):
                return visitor.visitBooleanAtomExpr(self)
            else:
                return visitor.visitChildren(self)


    class BooleanAndOrExprContext(BooleanExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.BooleanExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def booleanExpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PetabMathExprParser.BooleanExprContext)
            else:
                return self.getTypedRuleContext(PetabMathExprParser.BooleanExprContext,i)

        def BOOLEAN_AND(self):
            return self.getToken(PetabMathExprParser.BOOLEAN_AND, 0)
        def BOOLEAN_OR(self):
            return self.getToken(PetabMathExprParser.BOOLEAN_OR, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBooleanAndOrExpr" ):
                return visitor.visitBooleanAndOrExpr(self)
            else:
                return visitor.visitChildren(self)


    class BooleanNotExprContext(BooleanExprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a PetabMathExprParser.BooleanExprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def NOT(self):
            return self.getToken(PetabMathExprParser.NOT, 0)
        def booleanAtom(self):
            return self.getTypedRuleContext(PetabMathExprParser.BooleanAtomContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBooleanNotExpr" ):
                return visitor.visitBooleanNotExpr(self)
            else:
                return visitor.visitChildren(self)



    def booleanExpr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = PetabMathExprParser.BooleanExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 12
        self.enterRecursionRule(localctx, 12, self.RULE_booleanExpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 80
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [26]:
                localctx = PetabMathExprParser.BooleanNotExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 77
                self.match(PetabMathExprParser.NOT)
                self.state = 78
                self.booleanAtom()
                pass
            elif token in [1, 7, 8, 10, 11, 21, 22]:
                localctx = PetabMathExprParser.BooleanAtomExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 79
                self.booleanAtom()
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 87
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,6,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = PetabMathExprParser.BooleanAndOrExprContext(self, PetabMathExprParser.BooleanExprContext(self, _parentctx, _parentState))
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_booleanExpr)
                    self.state = 82
                    if not self.precpred(self._ctx, 2):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                    self.state = 83
                    _la = self._input.LA(1)
                    if not(_la==13 or _la==14):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 84
                    self.booleanExpr(3)
                self.state = 89
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,6,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class BooleanAtomContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def booleanLiteral(self):
            return self.getTypedRuleContext(PetabMathExprParser.BooleanLiteralContext,0)


        def OPEN_PAREN(self):
            return self.getToken(PetabMathExprParser.OPEN_PAREN, 0)

        def booleanExpr(self):
            return self.getTypedRuleContext(PetabMathExprParser.BooleanExprContext,0)


        def CLOSE_PAREN(self):
            return self.getToken(PetabMathExprParser.CLOSE_PAREN, 0)

        def comparisonExpr(self):
            return self.getTypedRuleContext(PetabMathExprParser.ComparisonExprContext,0)


        def var(self):
            return self.getTypedRuleContext(PetabMathExprParser.VarContext,0)


        def getRuleIndex(self):
            return PetabMathExprParser.RULE_booleanAtom

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBooleanAtom" ):
                return visitor.visitBooleanAtom(self)
            else:
                return visitor.visitChildren(self)




    def booleanAtom(self):

        localctx = PetabMathExprParser.BooleanAtomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_booleanAtom)
        try:
            self.state = 97
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 90
                self.booleanLiteral()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 91
                self.match(PetabMathExprParser.OPEN_PAREN)
                self.state = 92
                self.booleanExpr(0)
                self.state = 93
                self.match(PetabMathExprParser.CLOSE_PAREN)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 95
                self.comparisonExpr()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 96
                self.var()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ComparisonExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def floatComparisonExpr(self):
            return self.getTypedRuleContext(PetabMathExprParser.FloatComparisonExprContext,0)


        def boolComparisonExpr(self):
            return self.getTypedRuleContext(PetabMathExprParser.BoolComparisonExprContext,0)


        def getRuleIndex(self):
            return PetabMathExprParser.RULE_comparisonExpr

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitComparisonExpr" ):
                return visitor.visitComparisonExpr(self)
            else:
                return visitor.visitChildren(self)




    def comparisonExpr(self):

        localctx = PetabMathExprParser.ComparisonExprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_comparisonExpr)
        try:
            self.state = 101
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [1, 10, 11, 21, 22]:
                self.enterOuterAlt(localctx, 1)
                self.state = 99
                self.floatComparisonExpr()
                pass
            elif token in [7, 8]:
                self.enterOuterAlt(localctx, 2)
                self.state = 100
                self.boolComparisonExpr()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BoolComparisonExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def booleanLiteral(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PetabMathExprParser.BooleanLiteralContext)
            else:
                return self.getTypedRuleContext(PetabMathExprParser.BooleanLiteralContext,i)


        def comp_op(self):
            return self.getTypedRuleContext(PetabMathExprParser.Comp_opContext,0)


        def getRuleIndex(self):
            return PetabMathExprParser.RULE_boolComparisonExpr

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBoolComparisonExpr" ):
                return visitor.visitBoolComparisonExpr(self)
            else:
                return visitor.visitChildren(self)




    def boolComparisonExpr(self):

        localctx = PetabMathExprParser.BoolComparisonExprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_boolComparisonExpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 103
            self.booleanLiteral()
            self.state = 104
            self.comp_op()
            self.state = 105
            self.booleanLiteral()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FloatComparisonExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def arithmeticExpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PetabMathExprParser.ArithmeticExprContext)
            else:
                return self.getTypedRuleContext(PetabMathExprParser.ArithmeticExprContext,i)


        def comp_op(self):
            return self.getTypedRuleContext(PetabMathExprParser.Comp_opContext,0)


        def getRuleIndex(self):
            return PetabMathExprParser.RULE_floatComparisonExpr

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFloatComparisonExpr" ):
                return visitor.visitFloatComparisonExpr(self)
            else:
                return visitor.visitChildren(self)




    def floatComparisonExpr(self):

        localctx = PetabMathExprParser.FloatComparisonExprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_floatComparisonExpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 107
            self.arithmeticExpr(0)
            self.state = 108
            self.comp_op()
            self.state = 109
            self.arithmeticExpr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BooleanLiteralContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TRUE(self):
            return self.getToken(PetabMathExprParser.TRUE, 0)

        def FALSE(self):
            return self.getToken(PetabMathExprParser.FALSE, 0)

        def getRuleIndex(self):
            return PetabMathExprParser.RULE_booleanLiteral

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBooleanLiteral" ):
                return visitor.visitBooleanLiteral(self)
            else:
                return visitor.visitChildren(self)




    def booleanLiteral(self):

        localctx = PetabMathExprParser.BooleanLiteralContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_booleanLiteral)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 111
            _la = self._input.LA(1)
            if not(_la==7 or _la==8):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class NumberContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUMBER(self):
            return self.getToken(PetabMathExprParser.NUMBER, 0)

        def getRuleIndex(self):
            return PetabMathExprParser.RULE_number

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitNumber" ):
                return visitor.visitNumber(self)
            else:
                return visitor.visitChildren(self)




    def number(self):

        localctx = PetabMathExprParser.NumberContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_number)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 113
            self.match(PetabMathExprParser.NUMBER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VarContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NAME(self):
            return self.getToken(PetabMathExprParser.NAME, 0)

        def getRuleIndex(self):
            return PetabMathExprParser.RULE_var

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVar" ):
                return visitor.visitVar(self)
            else:
                return visitor.visitChildren(self)




    def var(self):

        localctx = PetabMathExprParser.VarContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_var)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 115
            self.match(PetabMathExprParser.NAME)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[3] = self.arithmeticExpr_sempred
        self._predicates[6] = self.booleanExpr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def arithmeticExpr_sempred(self, localctx:ArithmeticExprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 8)


            if predIndex == 1:
                return self.precpred(self._ctx, 6)


            if predIndex == 2:
                return self.precpred(self._ctx, 5)


    def booleanExpr_sempred(self, localctx:BooleanExprContext, predIndex:int):
            if predIndex == 3:
                return self.precpred(self._ctx, 2)
