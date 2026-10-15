import math

import libsbml
import numpy as np
import pytest
import sympy as sp

from petab.v1.math import sympify_petab
from petab.v2 import C, Problem, reparametrize_log
from petab.v2.models.sbml_model import SbmlModel


def _problem() -> Problem:
    """A problem with a model parameter, an output parameter, and a
    placeholder override."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony(
        """
        k1 = 1
        k2 = 1
        A' = -k1 * A
        A = 1
        """
    )
    problem.add_observable(
        "obs_A",
        "scaling * A + offset_A",
        noise_formula="noiseParameter1_obs_A",
        observable_placeholders=["offset_A"],
        noise_placeholders=["noiseParameter1_obs_A"],
    )
    problem.add_condition("c1", k2="k2_c1")
    problem.add_experiment("e1", 0, "c1")
    for offset in ("offset1", "offset2"):
        problem.add_measurement(
            "obs_A",
            experiment_id="e1",
            time=1,
            measurement=1,
            observable_parameters=[offset],
            noise_parameters=[0.5],
        )
    problem.add_parameter(
        "k1",
        nominal_value=2,
        lb=1e-3,
        ub=1e3,
        prior_dist=C.LOG_NORMAL,
        prior_pars=[1, 2],
        parameterName="k one",
    )
    problem.add_parameter(
        "k2_c1",
        nominal_value=3,
        lb=1e-2,
        ub=1e2,
        prior_dist=C.LOG_UNIFORM,
        prior_pars=[1e-1, 1e1],
    )
    problem.add_parameter("scaling", nominal_value=4, lb=1e-3, ub=1e3)
    problem.add_parameter("offset1", nominal_value=5, lb=1e-3, ub=1e3)
    problem.add_parameter("offset2", estimate=False, nominal_value=6)
    assert not problem.validate().has_errors()
    return problem


def _sym(x: str) -> sp.Symbol:
    return sp.Symbol(x, real=True)


@pytest.mark.parametrize("base", [10, "e"])
def test_reparametrize_log_parameter_table(base):
    problem = _problem()
    ln_b = math.log(10) if base == 10 else 1.0
    prefix = "log10_" if base == 10 else "log_"

    id_map = reparametrize_log(
        problem, ["k1", "k2_c1", "scaling", "offset1", "offset2"], base=base
    )
    assert id_map == {
        "k1": f"{prefix}k1",
        "k2_c1": f"{prefix}k2_c1",
        "scaling": f"{prefix}scaling",
        "offset1": f"{prefix}offset1",
        "offset2": f"{prefix}offset2",
    }
    assert [p.id for p in problem.parameters] == list(id_map.values())
    offset2 = problem[f"{prefix}offset2"]
    assert offset2.estimate is False
    assert offset2.nominal_value == pytest.approx(math.log(6) / ln_b)

    k1 = problem[f"{prefix}k1"]
    assert k1.lb == pytest.approx(math.log(1e-3) / ln_b)
    assert k1.ub == pytest.approx(math.log(1e3) / ln_b)
    assert k1.nominal_value == pytest.approx(math.log(2) / ln_b)
    assert k1.estimate is True
    assert k1.prior_distribution == C.NORMAL
    assert k1.prior_parameters == pytest.approx([1 / ln_b, 2 / ln_b])
    assert k1.name == (f"{prefix.removesuffix('_')}(k one)")

    k2 = problem[f"{prefix}k2_c1"]
    assert k2.prior_distribution == C.UNIFORM
    assert k2.prior_parameters == pytest.approx(
        [math.log(1e-1) / ln_b, math.log(1e1) / ln_b]
    )

    scaling = problem[f"{prefix}scaling"]
    assert scaling.prior_distribution is None
    assert scaling.name is None

    if base == 10:
        # powers of 10 are mapped exactly
        assert [k1.lb, k1.ub] == [-3.0, 3.0]
        assert k2.prior_parameters == [-1.0, 1.0]

    assert not problem.validate().has_errors()


def test_reparametrize_log_expressions_and_model():
    problem = _problem()
    reparametrize_log(
        problem, ["k1", "k2_c1", "scaling", "offset1", "offset2"]
    )

    # observable formula
    obs = problem["obs_A"]
    assert obs.formula == (
        10 ** _sym("log10_scaling") * _sym("A") + 10 ** _sym("offset_A")
    )
    assert obs.noise_formula == sympify_petab("noiseParameter1_obs_A")

    # condition table
    assert problem["c1"].changes[0].target_value == 10 ** _sym("log10_k2_c1")

    # overrides
    assert [m.observable_parameters for m in problem.measurements] == [
        [_sym("log10_offset1")],
        [_sym("log10_offset2")],
    ]
    assert [m.noise_parameters for m in problem.measurements] == [
        [0.5],
        [0.5],
    ]

    # model
    sbml_model = problem.model.sbml_model
    assert sbml_model.getParameter("k1").getConstant() is True
    q = sbml_model.getParameter("log10_k1")
    assert q.getConstant() is True
    assert q.getValue() == pytest.approx(math.log10(2))
    assert sbml_model.getAssignmentRuleByVariable("k1") is None
    ia = sbml_model.getInitialAssignmentBySymbol("k1")
    assert ia is not None
    assert sympify_petab(libsbml.formulaToL3String(ia.getMath())) == (
        sympify_petab("10^log10_k1")
    )
    assert problem.model.is_valid()
    # k2 is not reparametrized, it's a condition target
    assert sbml_model.getInitialAssignmentBySymbol("k2") is None

    assert not problem.validate().has_errors()


def test_reparametrize_log_initial_assignment_replaced():
    problem = Problem()
    problem.model = SbmlModel.from_antimony("k1 = 2; A' = -k1 * A; A = 1")
    sbml_model = problem.model.sbml_model
    ia = sbml_model.createInitialAssignment()
    ia.setSymbol("k1")
    ia.setMath(libsbml.parseL3Formula("3"))
    problem.add_parameter("k1", nominal_value=2, lb=1, ub=10)
    reparametrize_log(problem, ["k1"])
    assert sbml_model.getNumInitialAssignments() == 1
    ia = sbml_model.getInitialAssignmentBySymbol("k1")
    assert libsbml.formulaToL3String(ia.getMath()) == "10^log10_k1"
    assert problem.model.is_valid()


def test_reparametrize_log_id_collision():
    problem = _problem()
    problem.add_parameter("log10_scaling", nominal_value=1, lb=1, ub=2)
    problem.observables[0].formula += _sym("log10_scaling")
    problem.model.sbml_model.createParameter().setId("log10_scaling_1")
    id_map = reparametrize_log(problem, ["scaling"])
    assert id_map == {"scaling": "log10_scaling_2"}


def test_reparametrize_log_nominal():
    problem = _problem()
    problem["scaling"].nominal_value = None
    problem["offset1"].nominal_value = 0
    with pytest.warns(UserWarning, match="no positive nominal value"):
        reparametrize_log(problem, ["scaling", "offset1", "offset2"])
    assert problem["log10_scaling"].nominal_value is None
    assert problem["log10_offset1"].nominal_value is None


def test_reparametrize_log_explicit_priors():
    problem = _problem()
    reparametrize_log(
        problem,
        ["k1", "scaling"],
        priors={"k1": (C.LAPLACE, [0, 1]), "scaling": (C.NORMAL, [1, 2])},
    )
    assert problem["log10_k1"].prior_distribution == C.LAPLACE
    assert problem["log10_k1"].prior_parameters == [0, 1]
    assert problem["log10_scaling"].prior_distribution == C.NORMAL
    assert problem["log10_scaling"].prior_parameters == [1, 2]

    with pytest.raises(ValueError, match="unknown parameters"):
        reparametrize_log(_problem(), ["k1"], priors={"k2": (C.NORMAL, [])})


@pytest.mark.parametrize("prior", [C.UNIFORM, C.NORMAL, C.GAMMA])
def test_reparametrize_log_inexpressible_prior(prior):
    problem = _problem()
    problem["scaling"].prior_distribution = prior
    problem["scaling"].prior_parameters = [1, 2]
    with pytest.raises(ValueError, match="cannot be expressed"):
        reparametrize_log(problem, ["scaling"])


def test_reparametrize_log_invalid():
    problem = _problem()
    problem["scaling"].lb = 0
    with pytest.raises(ValueError, match="must be positive"):
        reparametrize_log(problem, ["scaling"])

    with pytest.raises(ValueError, match="not in the parameter table"):
        reparametrize_log(_problem(), ["nonexistent"])

    problem = _problem()
    problem.add_mapping("scaling", "scaling")
    with pytest.raises(NotImplementedError, match="mapping table"):
        reparametrize_log(problem, ["scaling"])

    problem = _problem()
    doc = problem.model.sbml_model.getSBMLDocument()
    assert doc.setLevelAndVersion(2, 1)
    with pytest.raises(NotImplementedError, match="L2V2"):
        reparametrize_log(problem, ["k1"])


def test_reparametrize_log_fixed_nonpositive_bounds():
    """Bounds of non-estimated parameters are irrelevant."""
    problem = _problem()
    problem["offset2"].lb = 0
    problem["offset2"].ub = 10
    reparametrize_log(problem, ["offset1", "offset2"])
    par = problem["log10_offset2"]
    assert par.lb is None
    assert par.ub == 1
    assert par.nominal_value == pytest.approx(math.log10(6))


def test_reparametrize_log_unresolvable_override():
    # offset1 is reparametrized, offset2 is not
    problem = _problem()
    problem["offset2"].lb = 1
    problem["offset2"].ub = 10
    problem["offset2"].estimate = True
    with pytest.raises(NotImplementedError) as excinfo:
        reparametrize_log(problem, ["offset1"])
    msg = str(excinfo.value)
    assert (
        "Observable placeholder `offset_A' of observable `obs_A' "
        "(overridden by `offset1') also receives:" in msg
    )
    assert "not reparametrized: `offset2'\n" in msg
    assert "non-positive" not in msg

    # non-positive numeric overrides, in several placeholders, reported at
    #  once
    problem = _problem()
    problem.measurements[0].noise_parameters = ["offset1"]
    problem.measurements[1].observable_parameters = [0]
    problem.measurements[1].noise_parameters = [-1]
    with pytest.raises(NotImplementedError) as excinfo:
        reparametrize_log(problem, ["offset1"])
    msg = str(excinfo.value)
    assert "Observable placeholder `offset_A'" in msg
    assert "Noise placeholder `noiseParameter1_obs_A'" in msg
    assert "non-positive numbers: 0\n" in msg
    assert "non-positive numbers: -1\n" in msg
    assert "not reparametrized" not in msg


def test_reparametrize_log_noise_override():
    """Observable placeholders in noise formulas and noise overrides."""
    problem = _problem()
    problem.observables[0].noise_formula = sympify_petab(
        "noiseParameter1_obs_A * offset_A"
    )
    for m in problem.measurements:
        m.noise_parameters = ["sigma"]
    problem.add_parameter("sigma", nominal_value=0.1, lb=1e-3, ub=1)
    assert not problem.validate().has_errors()

    reparametrize_log(problem, ["sigma", "offset1", "offset2"])
    assert problem.observables[0].noise_formula == (
        10 ** _sym("noiseParameter1_obs_A") * 10 ** _sym("offset_A")
    )
    assert [m.noise_parameters for m in problem.measurements] == [
        [_sym("log10_sigma")],
        [_sym("log10_sigma")],
    ]
    assert not problem.validate().has_errors()


@pytest.mark.parametrize("base", [10, "e"])
@pytest.mark.parametrize(
    "prior",
    [
        (C.LOG_NORMAL, [0.5, 1.5]),
        (C.LOG_LAPLACE, [0.5, 1.5]),
        (C.LOG_UNIFORM, [1e-2, 1e2]),
    ],
)
def test_reparametrize_log_prior_densities(base, prior):
    """The derived prior over q is the pushforward of the prior over p."""
    problem = _problem()
    (
        problem["scaling"].prior_distribution,
        problem["scaling"].prior_parameters,
    ) = prior
    old_dist = problem["scaling"].prior_dist
    id_map = reparametrize_log(problem, ["scaling"], base=base)
    new_dist = problem[id_map["scaling"]].prior_dist

    b = 10 if base == 10 else math.e
    thetas = np.linspace(-1.5, 1.5, 7)
    xs = b**thetas
    # density transformation: pi_q(theta) = pi_p(b^theta) * b^theta * ln(b)
    expected = old_dist.pdf(xs) * xs * math.log(b)
    np.testing.assert_allclose(new_dist.pdf(thetas), expected, rtol=1e-10)
