// Parser grammar for PEtab math expressions
// run `regenerate.sh` to regenerate the parser
parser grammar PetabMathExprParser;

options { tokenVocab=PetabMathExprLexer; }

prog:
    expr EOF ;

expr:
    arithmeticExpr
    | booleanExpr
    ;

comp_op:
    GT
    | LT
    | GTE
    | LTE
    | EQ
    | NEQ
    ;

arithmeticExpr:
    <assoc=right> arithmeticExpr '^' arithmeticExpr  # HatExpr
    | ('+'|'-') arithmeticExpr                       # UnaryExpr
    | arithmeticExpr ('*'|'/') arithmeticExpr        # MultExpr
    | arithmeticExpr ('+'|'-') arithmeticExpr        # AddExpr
    | '(' arithmeticExpr ')'                         # ParenExpr
    | number                                         # Number_
    | func_expr                                      # FuncExpr_
    | var                                            # VarExpr_
    ;

argumentList: expr (',' expr)* ;
func_expr: NAME OPEN_PAREN argumentList CLOSE_PAREN ;

booleanExpr:
    '!' booleanAtom                                       # BooleanNotExpr
    | booleanExpr (BOOLEAN_AND | BOOLEAN_OR) booleanExpr  # BooleanAndOrExpr
    | booleanAtom                                         # BooleanAtomExpr
;

booleanAtom:
    booleanLiteral
    | '(' booleanExpr ')'
    | comparisonExpr
    | var
    ;
comparisonExpr:
    floatComparisonExpr
    | boolComparisonExpr
    ;
boolComparisonExpr: booleanLiteral comp_op booleanLiteral;
floatComparisonExpr: arithmeticExpr comp_op arithmeticExpr;
booleanLiteral:
    TRUE
    | FALSE
    ;
number: NUMBER ;
var: NAME ;
