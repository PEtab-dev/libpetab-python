import importlib.resources
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import yaml
from sympy.abc import _clash
from sympy.logic.boolalg import Boolean, BooleanFalse, BooleanTrue

from petab.v1.math import petab_math_str, sympify_petab


def test_sympify_numpy():
    assert sympify_petab(np.float64(1.0)) == sp.Float(1.0)


def test_parse_simple():
    """Test simple numeric expressions."""
    assert float(sympify_petab("1 + 2")) == 3
    assert float(sympify_petab("1 + 2 * 3")) == 7
    assert float(sympify_petab("(1 + 2) * 3")) == 9
    assert float(sympify_petab("1 + 2 * (3 + 4)")) == 15
    assert float(sympify_petab("1 + 2 * (3 + 4) / 2")) == 8


def test_evaluate():
    act = sympify_petab("piecewise(1, 1 > 2, 0)", evaluate=False)
    assert str(act) == "Piecewise((1.0, 1.0 > 2.0), (0.0, True))"


def test_assumptions():
    # in PEtab, all symbols are expected to be real-valued
    assert sympify_petab("x").is_real

    # non-real symbols are changed to real
    assert sympify_petab(sp.Symbol("x", real=False)).is_real


def test_printer():
    a, b, c, d = sp.symbols("a b c d", real=True)
    assert petab_math_str(None) == ""
    assert petab_math_str(BooleanTrue()) == "true"
    assert petab_math_str(BooleanFalse()) == "false"
    assert petab_math_str((a + b) ** (c + d)) == "(a + b) ^ (c + d)"
    # A non-integer rational exponent must be parenthesized, else "a ^ 1/2"
    # re-parses as (a^1)/2 (i.e. sqrt(a) would round-trip to a/2).
    assert petab_math_str(sp.sqrt(a)) == "a ^ (1/2)"
    assert petab_math_str(a ** sp.Rational(2, 3)) == "a ^ (2/3)"
    # negative or non-integer rational bases must be parenthesized
    assert petab_math_str(sp.Float(-2) ** a) == "(-2) ^ a"
    assert petab_math_str(sp.Rational(2, 3) ** a) == "(2/3) ^ a"
    # boolean and relational operators
    assert petab_math_str(sp.And(a > 0, b < 1)) == "(a > 0) && (b < 1)"
    assert petab_math_str(sp.Or(a > 0, b < 1)) == "(a > 0) || (b < 1)"
    assert petab_math_str(sp.Not(sp.And(a > 0, b < 1))) == (
        "!((a > 0) && (b < 1))"
    )
    assert petab_math_str(sp.Eq(a, 1)) == "a == 1"
    assert petab_math_str(sp.Ne(a, 1)) == "a != 1"
    # min/max are binary in PEtab
    assert petab_math_str(sp.Min(a, b, c)) == "min(min(a, b), c)"
    assert petab_math_str(sp.Max(a, b, c)) == "max(max(a, b), c)"
    # no pi / e constants in PEtab
    assert petab_math_str(sp.pi) == "3.141592653589793"
    assert petab_math_str(sp.E) == "exp(1)"
    assert petab_math_str(sp.E * a) == "exp(1)*a"
    # floats round-trip exactly; integer-valued floats are printed as integers
    assert petab_math_str(sp.Float(2.0) * a) == "2*a"
    assert petab_math_str(sp.Float(2 / 3) * a) == "0.6666666666666666*a"
    assert petab_math_str(sp.Float(1e-300)) == "1e-300"
    assert petab_math_str(sympify_petab("1.2345678901234567 * a")) == (
        "1.2345678901234567*a"
    )

    with pytest.raises(ValueError, match="default"):
        petab_math_str(sp.Piecewise((1, a > 0)))


@pytest.mark.parametrize(
    "expr_str",
    [
        "100 * x",
        "x ^ 2",
        "2 * x / 3",
        "log(x, 10)",
        "1.2345678901234567 * x",
        "(-2) ^ x",
        "(1 / 3) ^ x",
        "x ^ -0.5 * y",
        "piecewise(1, x > 0 && y < 2 || x == y, 0)",
        "piecewise(1, !(x > 0 && y < 2), 0)",
        "piecewise(1, x != y, 0)",
        "min(min(x, y), 3) + max(x, max(y, 2))",
        "2 * arccos(0) + exp(1)",
    ],
)
@pytest.mark.parametrize("evaluate", [True, False])
def test_print_parse_round_trip(expr_str, evaluate):
    """Parsing, printing, and parsing again gives the same value."""
    expr = sympify_petab(expr_str, evaluate=evaluate)
    printed = petab_math_str(expr)
    reparsed = sympify_petab(printed, evaluate=evaluate)
    x, y = sp.symbols("x y", real=True)
    values = {x: 0.7, y: -1.3}
    expected = complex(expr.subs(values))
    actual = complex(reparsed.subs(values))
    assert np.isclose(actual, expected, rtol=1e-15, atol=0), printed


def read_cases():
    """Read test cases from YAML file in the petab_test_suite package."""
    yaml_file = importlib.resources.files("petabtests.cases").joinpath(
        str(Path("v2.0.0", "math", "math_tests.yaml"))
    )
    with importlib.resources.as_file(yaml_file) as file, open(file) as file:
        data = yaml.safe_load(file)

    cases = []
    for item in data["cases"]:
        expr_str = item["expression"]
        if item["expected"] is True or item["expected"] is False:
            expected = item["expected"]
        else:
            try:
                expected = float(item["expected"])
            except ValueError:
                expected = sp.sympify(item["expected"], locals=_clash)
                expected = expected.subs(
                    {
                        s: sp.Symbol(s.name, real=True)
                        for s in expected.free_symbols
                    }
                )
        cases.append((expr_str, expected))
    return cases


@pytest.mark.parametrize("expr_str, expected", read_cases())
def test_parse_cases(expr_str, expected):
    """Test PEtab math expressions for the PEtab test suite."""
    sym_expr = sympify_petab(expr_str)
    if isinstance(sym_expr, Boolean):
        assert sym_expr == expected
    else:
        try:
            result = float(sym_expr.evalf())
            assert np.isclose(result, expected), (
                f"{expr_str}: Expected {expected}, got {result}"
            )
        except TypeError:
            assert sym_expr == expected, (
                f"{expr_str}: Expected {expected}, got {result}"
            )

    # test parsing, printing, and parsing again
    resympified = sympify_petab(petab_math_str(sym_expr))
    if sym_expr.is_number:
        assert np.isclose(float(resympified), float(sym_expr))
    else:
        assert resympified.equals(sym_expr), (sym_expr, resympified)


def test_ids():
    """Test symbols in expressions."""
    assert sympify_petab("bla * 2") == 2.0 * sp.Symbol("bla", real=True)

    # test that sympy expressions that are invalid in PEtab raise an error
    with pytest.raises(ValueError):
        sympify_petab(sp.Symbol("föö"))


def test_syntax_error():
    """Test exceptions upon syntax errors."""
    # parser error
    with pytest.raises(ValueError, match="Syntax error"):
        sympify_petab("1 + ")

    # lexer error
    with pytest.raises(ValueError, match="Syntax error"):
        sympify_petab("0.")


def test_complex():
    """Test expressions producing (unsupported) complex numbers."""
    with pytest.raises(ValueError, match="not real-valued"):
        sympify_petab("sqrt(-1)")
    with pytest.raises(ValueError, match="not real-valued"):
        sympify_petab("arctanh(inf)")
