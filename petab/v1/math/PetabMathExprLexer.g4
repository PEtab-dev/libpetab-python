// Lexer grammar for PEtab math expressions
// run `regenerate.sh` to regenerate the lexer
lexer grammar PetabMathExprLexer;


NUMBER          : EXPONENT_FLOAT | INTEGER | POINT_FLOAT | INF;
INTEGER         : DIGITS ;
EXPONENT_FLOAT   : (INTEGER | POINT_FLOAT) EXPONENT ;
POINT_FLOAT      : DIGITS '.' DIGITS ;
fragment EXPONENT: ('e' | 'E') ('+' | '-')? DIGITS ;
FLOAT_NUMBER: POINT_FLOAT | EXPONENT_FLOAT;
fragment DIGITS : [0-9]+ ;

WS      : [ \t\r\n]+ -> skip ;
TRUE    : 'true' ;
FALSE   : 'false' ;
INF     : 'inf' ;
NAME : [a-zA-Z_][a-zA-Z0-9_]* ;
OPEN_PAREN : '(' ;
CLOSE_PAREN : ')' ;
BOOLEAN_OR : '||' ;
BOOLEAN_AND : '&&' ;
GT : '>' ;
LT : '<' ;
GTE : '>=' ;
LTE : '<=' ;
EQ : '==' ;
NEQ : '!=' ;
PLUS : '+' ;
MINUS : '-' ;
ASTERISK : '*' ;
SLASH : '/' ;
CARET: '^';
EXCLAMATION_MARK: '!';
COMMA: ',';
