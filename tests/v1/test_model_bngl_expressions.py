"""BNGL parameter-expression evaluation.

The semantics here are not guesses. :data:`BNG_VERIFIED` is a table of
expressions with the value BNG2.pl 2.9.3 actually computes for them,
obtained by running each through ``writeNET({evaluate_expressions=>1})``,
the only export path that emits numbers rather than echoing the source
text.

The table is checked from both sides. :func:`test_bng_verified_table` pins
the evaluator against it with no BNG2.pl needed, so the contract holds in
ordinary continuous integration. :func:`test_table_still_matches_bng2pl`
re-derives the same values from a real BNG2.pl where one is available, so
the table cannot quietly rot if BioNetGen changes.
"""

import math
import re
import subprocess

import pytest

from petab.v1.models.bngl_model import (
    BnglExpressionError,
    BnglModel,
    CircularParameterError,
    _locate_bng2,
    evaluate_bngl_expression,
    evaluate_bngl_parameters,
    evaluate_bngl_parameters_partial,
    parse_bngl,
)

#: ``(expression, the value BNG2.pl computes)``. Self-contained, so each one
#: can be dropped straight into a parameters block.
BNG_VERIFIED = [
    # -- operators --------------------------------------------------------
    ("2^3", 8.0),
    ("2**3", 8.0),  # BNG2.pl accepts ** as a synonym for ^
    ("1/2", 0.5),  # float division, never integer
    ("8/4/2", 1.0),  # / is left associative
    ("1-2-3", -4.0),  # - is left associative
    ("1+2*3", 7.0),
    ("(1+2)*3", 9.0),
    ("2*-3", -6.0),
    # Unary minus binds TIGHTER than ^, so this is (-2)^2, not -(2^2).
    ("-2^2", 4.0),
    ("-2^3", -8.0),
    ("3*-2^2", 12.0),
    ("-(2^2)", -4.0),  # explicit parens do give -(2^2)
    ("0-2^2", -4.0),  # binary minus is looser, as usual
    ("-exp(0)^2", 1.0),  # the rule covers function calls too
    # ^ is LEFT associative: (2^3)^2, not 2^(3^2).
    ("2^3^2", 64.0),
    ("2^2^3", 64.0),
    ("4^0.5^2", 4.0),
    ("2^(3^2)", 512.0),
    ("2^-2", 0.25),
    ("2^-2^2", 0.0625),
    # -- comparison and logical, which yield 1.0/0.0 ----------------------
    ("1<2", 1.0),
    ("2<1", 0.0),
    ("1==1", 1.0),
    ("1!=1", 0.0),
    ("1~=2", 1.0),  # ~= is BNG2.pl's alias for !=
    ("2&&3", 1.0),  # normalised, unlike Perl's own &&
    ("0&&3", 0.0),
    ("0||5", 1.0),  # 1.0, not 5
    ("1+2>2", 1.0),  # + binds tighter than >
    ("1<2&&2<3", 1.0),  # comparison binds tighter than &&
    ("if(1,5,7)", 5.0),
    ("if(0,5,7)", 7.0),
    ("if(2>1,5,7)", 5.0),
    # -- functions --------------------------------------------------------
    ("_pi()", math.pi),  # zero-argument functions, not bare names
    ("_e()", math.e),
    ("ln(_e())", 1.0),
    ("exp(1)", math.e),
    ("log10(1000)", 3.0),
    ("log2(8)", 3.0),
    ("sqrt(4)", 2.0),
    ("abs(-3)", 3.0),
    ("sin(1)", math.sin(1)),
    ("cos(1)", math.cos(1)),
    ("tan(1)", math.tan(1)),
    ("asin(0.5)", math.asin(0.5)),
    ("acos(0.5)", math.acos(0.5)),
    ("atan(0.5)", math.atan(0.5)),
    ("sinh(1)", math.sinh(1)),
    ("cosh(1)", math.cosh(1)),
    ("tanh(1)", math.tanh(1)),
    ("asinh(1)", math.asinh(1)),
    ("acosh(2)", math.acosh(2)),
    ("atanh(0.5)", math.atanh(0.5)),
    ("min(1,2)", 1.0),
    ("min(3,1,2)", 1.0),  # min/max/sum/avg are variadic
    ("max(1,2)", 2.0),
    ("sum(1,2,3,4)", 10.0),
    ("avg(2,4)", 3.0),
    # rint is floor(x + 0.5), rounding a half up, where Python's round
    # sends a half to the nearest even number.
    ("rint(0.5)", 1.0),
    ("rint(1.5)", 2.0),
    ("rint(2.5)", 3.0),
    ("rint(-0.5)", 0.0),
    ("rint(-2.5)", -2.0),
]

#: Expressions BNG2.pl refuses. Rejecting them keeps a typo an error rather
#: than a plausible wrong number.
BNG_REJECTS = [
    "log(10)",  # BNGL's natural log is ln, and there is no bare log
    "floor(1.7)",  # commented out in Expression.pm as unsupported
    "ceil(1.2)",
    "_pi",  # the constants are functions, written _pi()
    "_e",
    "foo(1)",
    "1/0",
    "2 @ 3",
    "if(1,5,1/0)",  # BNG2.pl evaluates all three arguments
]


@pytest.mark.parametrize(
    "text, expected", BNG_VERIFIED, ids=[e for e, _ in BNG_VERIFIED]
)
def test_bng_verified_table(text, expected):
    """The evaluator reproduces what BNG2.pl computes."""
    assert evaluate_bngl_parameters({"z": text})["z"] == pytest.approx(
        expected
    )


@pytest.mark.parametrize("text", BNG_REJECTS)
def test_bng_rejected_expressions_are_rejected_here_too(text):
    with pytest.raises(BnglExpressionError):
        evaluate_bngl_parameters({"z": text})


# -- the differential against a real BNG2.pl ---------------------------------

_NET_PARAM = re.compile(r"^\s*\d+\s+(\w+)\s+(\S+)")

_PROBE_MODEL = """\
begin model
begin parameters
{block}
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() 1
end seed species
begin reaction rules
  A() -> B() 1
end reaction rules
end model
generate_network({{overwrite=>1}})
writeNET({{evaluate_expressions=>1,prefix=>"ev"}})
"""


def test_table_still_matches_bng2pl(tmp_path):
    """Re-derive :data:`BNG_VERIFIED` from BNG2.pl itself, in one run.

    Every expression goes into a single parameters block, so this costs one
    BNG2.pl invocation rather than one per case.
    """
    bng2 = _locate_bng2()
    if bng2 is None:
        pytest.skip("BNG2.pl not available")

    names = {f"p{i}": text for i, (text, _) in enumerate(BNG_VERIFIED)}
    block = "\n".join(f"  {n}  {t}" for n, t in names.items())
    (tmp_path / "probe.bngl").write_text(_PROBE_MODEL.format(block=block))

    proc = subprocess.run(  # noqa: S603
        [bng2, "probe.bngl"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    net = tmp_path / "ev.net"
    assert net.exists(), (
        f"BNG2.pl wrote no network:\n{proc.stdout}\n{proc.stderr}"
    )

    computed, in_block = {}, False
    for line in net.read_text().splitlines():
        if line.strip().startswith("begin parameters"):
            in_block = True
            continue
        if line.strip().startswith("end parameters"):
            break
        if in_block:
            m = _NET_PARAM.match(line.split("#")[0])
            if m:
                computed[m.group(1)] = float(m.group(2))

    mismatched = []
    for i, (text, expected) in enumerate(BNG_VERIFIED):
        actual = computed.get(f"p{i}")
        if actual is None or actual != pytest.approx(expected):
            mismatched.append(
                f"  {text!r}: table says {expected!r}, BNG2.pl says {actual!r}"
            )
    assert not mismatched, (
        "BNG2.pl disagrees with BNG_VERIFIED:\n" + "\n".join(mismatched)
    )


# -- resolution order and cycles ---------------------------------------------


def test_expression_over_other_parameters_resolves():
    text = (
        "begin parameters\n"
        "  NA    6.022e23\n"
        "  V     1e-12\n"
        "  Kd    5.0\n"
        "  koff  0.1\n"
        "  kon   koff/(Kd*NA*V)\n"
        "end parameters\n"
    )
    model = BnglModel(parse_bngl(text), model_id="demo")
    assert model.get_parameter_value("kon") == pytest.approx(
        0.1 / (5.0 * 6.022e23 * 1e-12)
    )
    ids = [name for name, _ in model.get_free_parameter_ids_with_values()]
    assert ids == list(model.get_parameter_ids())


def test_declaration_order_does_not_matter():
    """A parameter may be defined before the ones it depends on."""
    assert evaluate_bngl_parameters({"b": "a*2", "a": "3"}) == {
        "a": 3.0,
        "b": 6.0,
    }


def test_chained_expression_dependencies_resolve():
    values = evaluate_bngl_parameters({"a": "2", "b": "a*3", "c": "b+a"})
    assert values == {"a": 2.0, "b": 6.0, "c": 8.0}


@pytest.mark.parametrize(
    "params, target, expected",
    [
        # Shapes taken from real BioNetGen models.
        (
            {
                "kp18": "2",
                "km18": "1",
                "kp19": "3",
                "km19": "1",
                "kp22": "4",
                "km22": "2",
                "kp20": "5",
                "km20": "1",
                "loop3": "(kp18/km18)*(kp19/km19)/((kp22/km22)*(kp20/km20))",
            },
            "loop3",
            (2 / 1) * (3 / 1) / ((4 / 2) * (5 / 1)),
        ),
        ({"p_RM_AC": "7", "p_RM_A": "p_RM_AC"}, "p_RM_A", 7.0),
        ({"lifetime": "4", "gamma_R": "1/lifetime"}, "gamma_R", 0.25),
        ({"krZapTcr": "3", "krZapCd3e": "10*krZapTcr"}, "krZapCd3e", 30.0),
        (
            {"Kd_BRAF": "20", "Gf_BRAF": "ln(Kd_BRAF)"},
            "Gf_BRAF",
            math.log(20),
        ),
        (
            {
                "LT": "3",
                "RT": "1",
                "excess_ratio": "1",
                "use_excess": "if(LT/(RT+0.01)>=excess_ratio,1,0)",
            },
            "use_excess",
            1.0,
        ),
    ],
)
def test_real_world_expression_shapes(params, target, expected):
    assert evaluate_bngl_parameters(params)[target] == pytest.approx(expected)


def test_circular_definition_names_the_cycle():
    with pytest.raises(CircularParameterError) as excinfo:
        evaluate_bngl_parameters({"a": "b", "b": "a"})
    assert "a -> b -> a" in str(excinfo.value)


def test_self_referential_definition_is_reported():
    with pytest.raises(CircularParameterError):
        evaluate_bngl_parameters({"a": "a+1"})


def test_builtin_name_is_rejected_as_a_parameter_name():
    """BNG2.pl: "Cannot use built-in function name '_e' as a parameter"."""
    with pytest.raises(BnglExpressionError, match="built-in"):
        evaluate_bngl_parameters({"_e": "5"})


def test_evaluate_expression_against_known_symbols():
    assert evaluate_bngl_expression("x*2 + y", {"x": 1.5, "y": 1.0}) == 4.0


# -- partial resolution ------------------------------------------------------


def test_partial_resolution_keeps_the_usable_parameters():
    values, errors = evaluate_bngl_parameters_partial(
        {"a": "2", "b": "a*3", "bad": "nosuch", "c": "4"}
    )
    assert values == {"a": 2.0, "b": 6.0, "c": 4.0}
    assert set(errors) == {"bad"}


def test_partial_resolution_also_drops_dependents_of_a_bad_parameter():
    values, errors = evaluate_bngl_parameters_partial(
        {"bad": "nosuch", "downstream": "bad*2", "fine": "1"}
    )
    assert values == {"fine": 1.0}
    assert set(errors) == {"bad", "downstream"}


def test_every_parameter_is_either_resolved_or_reported():
    params = {"a": "1", "b": "a+1", "c": "oops", "d": "c*2"}
    values, errors = evaluate_bngl_parameters_partial(params)
    assert set(values) | set(errors) == set(params)
    assert not (set(values) & set(errors))


def test_one_unevaluable_parameter_does_not_take_down_the_model():
    """A whole-block failure would lose more than the original bug did."""
    text = (
        "begin parameters\n"
        "  good1  2\n"
        "  good2  good1*3\n"
        "  bad    not_a_parameter\n"
        "end parameters\n"
    )
    model = BnglModel(parse_bngl(text), model_id="demo")
    with pytest.warns(UserWarning, match="could not be evaluated"):
        pairs = dict(model.get_free_parameter_ids_with_values())
    assert pairs == {"good1": 2.0, "good2": 6.0}


def test_unevaluable_parameter_surfaces_from_the_model():
    text = "begin parameters\n  a  b\nend parameters\n"
    model = BnglModel(parse_bngl(text), model_id="demo")
    with pytest.raises(ValueError, match="could not be evaluated"):
        model.get_parameter_value("a")


def test_missing_parameter_still_raises_value_error():
    text = "begin parameters\n  a  1\nend parameters\n"
    model = BnglModel(parse_bngl(text), model_id="demo")
    with pytest.raises(ValueError, match="does not exist"):
        model.get_parameter_value("nope")
