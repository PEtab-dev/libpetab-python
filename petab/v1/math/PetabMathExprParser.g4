// Parser grammar for PEtab math expressions
// run `regenerate.sh` to regenerate the parser
parser grammar PetabMathExprParser;

options { tokenVocab=PetabMathExprLexer; }

petabExpression:
    expr EOF ;

expr:
    <assoc=right> expr '^' expr             # PowerExpr
    | ('+'|'-') expr                        # UnaryExpr
    | '!' expr                              # BooleanNotExpr
    | expr ('*'|'/') expr                   # MultExpr
    | expr ('+'|'-') expr                   # AddExpr
    | '(' expr ')'                          # ParenExpr
    | expr comp_op expr                     # ComparisonExpr
    | expr (BOOLEAN_AND | BOOLEAN_OR) expr  # BooleanAndOrExpr
    | number                                # Number_
    | booleanLiteral                        # BooleanLiteral_
    | functionCall                          # functionCall_
    | var                                   # VarExpr_
    ;

comp_op:
    GT
    | LT
    | GTE
    | LTE
    | EQ
    | NEQ
    ;

argumentList: expr (',' expr)* ;
functionCall: NAME OPEN_PAREN argumentList CLOSE_PAREN ;

booleanLiteral:
    TRUE
    | FALSE
    ;
number: NUMBER ;
var: NAME ;
