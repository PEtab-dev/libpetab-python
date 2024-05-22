from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import yaml

from petab.math import sympify_petab


def test_parse_simple():
    assert sympify_petab("1 + 2") == 3
    assert sympify_petab("1 + 2 * 3") == 7
    assert sympify_petab("(1 + 2) * 3") == 9
    assert sympify_petab("1 + 2 * (3 + 4)") == 15
    assert sympify_petab("1 + 2 * (3 + 4) / 2") == 8


def read_cases():
    with open(Path(__file__).parent / "math_tests.yaml") as file:
        data = yaml.safe_load(file)
    cases = []
    for item in data["cases"]:
        expr_str = item["expression"]
        expected = float(item["expected"])
        cases.append((expr_str, expected))
    return cases


@pytest.mark.parametrize("expr_str, expected", read_cases())
def test_parse_cases(expr_str, expected):
    result = sympify_petab(expr_str)
    result = float(result.evalf())
    assert np.isclose(
        result, expected
    ), f"{expr_str}: Expected {expected}, got {result}"


def test_ids():
    assert sympify_petab("bla * 2") == 2.0 * sp.Symbol("bla")


def test_syntax_error():
    with pytest.raises(ValueError, match="Syntax error"):
        sympify_petab("1 + ")
