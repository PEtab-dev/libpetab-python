import importlib.resources
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import yaml
from sympy.abc import _clash
from sympy.logic.boolalg import Boolean

from petab.math import sympify_petab


def test_sympify_numpy():
    assert sympify_petab(np.float64(1.0)) == sp.Float(1.0)


def test_parse_simple():
    """Test simple numeric expressions."""
    assert float(sympify_petab("1 + 2")) == 3
    assert float(sympify_petab("1 + 2 * 3")) == 7
    assert float(sympify_petab("(1 + 2) * 3")) == 9
    assert float(sympify_petab("1 + 2 * (3 + 4)")) == 15
    assert float(sympify_petab("1 + 2 * (3 + 4) / 2")) == 8


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
    result = sympify_petab(expr_str)
    if isinstance(result, Boolean):
        assert result == expected
    else:
        try:
            result = float(result.evalf())
            assert np.isclose(result, expected), (
                f"{expr_str}: Expected {expected}, got {result}"
            )
        except TypeError:
            assert result == expected, (
                f"{expr_str}: Expected {expected}, got {result}"
            )


def test_ids():
    """Test symbols in expressions."""
    assert sympify_petab("bla * 2") == 2.0 * sp.Symbol("bla", real=True)

    # test that sympy expressions that are invalid in PEtab raise an error
    # TODO: handle these cases after
    #   https://github.com/PEtab-dev/libpetab-python/pull/364
    # sympify_petab(sp.Symbol("föö"))


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
