import math

import libsbml
import numpy as np
import pytest
from sbmlmath import sbml_math_to_sympy

from petab.v1.math import sympify_petab
from petab.v2 import C, Problem, reparametrize_log
from petab.v2.core import ProblemConfig
from petab.v2.math import petab_symbol
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


@pytest.mark.parametrize("base", [10, "e"])
def test_reparametrize_log_parameter_table(base):
    problem = _problem()
    ln_b = math.log(10) if base == 10 else 1.0
    prefix = "log10_" if base == 10 else "log_"

    problem, id_map = reparametrize_log(
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
    assert k1.name == (f"{prefix.removesuffix('_')}(k one)")
    ia = problem.model.sbml_model.getInitialAssignmentBySymbol("k1")
    assert str(sbml_math_to_sympy(ia)) == (
        "10**log10_k1" if base == 10 else "exp(log_k1)"
    )

    scaling = problem[f"{prefix}scaling"]
    assert scaling.prior_distribution is None
    assert scaling.name is None

    if base == 10:
        # powers of 10 are mapped exactly
        assert [k1.lb, k1.ub] == [-3.0, 3.0]

    assert not problem.validate().has_errors()


def test_reparametrize_log_expressions_and_model():
    problem, _ = reparametrize_log(
        _problem(), ["k1", "k2_c1", "scaling", "offset1", "offset2"]
    )

    # observable formula
    obs = problem["obs_A"]
    assert obs.formula == (
        10 ** petab_symbol("log10_scaling") * petab_symbol("A")
        + 10 ** petab_symbol("offset_A")
    )
    assert obs.noise_formula == sympify_petab("noiseParameter1_obs_A")

    # condition table
    assert problem["c1"].changes[0].target_value == 10 ** petab_symbol(
        "log10_k2_c1"
    )

    # overrides
    assert [m.observable_parameters for m in problem.measurements] == [
        [petab_symbol("log10_offset1")],
        [petab_symbol("log10_offset2")],
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
    assert str(sbml_math_to_sympy(ia)) == "10**log10_k1"
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
    new_problem, _ = reparametrize_log(problem, ["k1"])
    new_sbml_model = new_problem.model.sbml_model
    assert new_sbml_model.getNumInitialAssignments() == 1
    ia = new_sbml_model.getInitialAssignmentBySymbol("k1")
    assert str(sbml_math_to_sympy(ia)) == "10**log10_k1"
    assert new_problem.model.is_valid()

    # the original problem is not modified
    ia = sbml_model.getInitialAssignmentBySymbol("k1")
    assert libsbml.formulaToL3String(ia.getMath()) == "3"
    assert [p.id for p in problem.parameters] == ["k1"]


def test_reparametrize_log_id_collision():
    problem = _problem()
    problem.add_parameter("log10_scaling", nominal_value=1, lb=1, ub=2)
    problem.observables[0].formula += petab_symbol("log10_scaling")
    problem.model.sbml_model.createParameter().setId("log10_scaling_1")
    problem, id_map = reparametrize_log(problem, ["scaling"])
    assert id_map == {"scaling": "log10_scaling_2"}


def test_reparametrize_log_new_ids_unique():
    """New IDs don't clash with each other."""
    problem = _problem()
    problem.add_parameter("log10_scaling", nominal_value=1, lb=1, ub=2)
    problem.add_parameter(
        "scaling_1", estimate=True, nominal_value=1, lb=1, ub=2
    )
    problem.observables[0].formula += petab_symbol(
        "log10_scaling"
    ) + petab_symbol("scaling_1")
    problem, id_map = reparametrize_log(problem, ["scaling", "scaling_1"])
    assert id_map == {
        "scaling": "log10_scaling_1",
        "scaling_1": "log10_scaling_1_1",
    }


def test_reparametrize_log_nominal_and_fixed_bounds():
    """Missing or negative nominal values of estimated parameters are
    dropped. A nominal value of 0 becomes -inf. Bounds of non-estimated
    parameters are irrelevant."""
    problem = _problem()
    problem["scaling"].nominal_value = None
    problem["offset1"].nominal_value = -1
    problem["k2_c1"].nominal_value = 0
    problem["offset2"].lb = -1
    problem["offset2"].ub = 10
    with pytest.warns(UserWarning, match="no non-negative nominal value"):
        problem, _ = reparametrize_log(
            problem, ["scaling", "offset1", "k2_c1", "offset2"]
        )
    assert problem["log10_scaling"].nominal_value is None
    assert problem["log10_offset1"].nominal_value is None
    assert problem["log10_k2_c1"].nominal_value == -math.inf
    offset2 = problem["log10_offset2"]
    assert (offset2.lb, offset2.ub) == (None, None)
    assert offset2.nominal_value == pytest.approx(math.log10(6))

    problem = _problem()
    problem["offset2"].nominal_value = -1
    with pytest.raises(ValueError, match="non-negative nominal value"):
        reparametrize_log(problem, ["offset2"])


def test_reparametrize_log_fixed_zero():
    """A non-estimated model parameter fixed at 0 is fixed at -inf on log
    scale."""
    problem = _problem()
    problem["k1"].estimate = False
    problem["k1"].nominal_value = 0
    problem, _ = reparametrize_log(problem, ["k1"], priors={"k1": None})
    assert problem["log10_k1"].nominal_value == -math.inf

    sbml_model = problem.model.sbml_model
    assert sbml_model.getParameter("log10_k1").getValue() == -math.inf
    ia = sbml_model.getInitialAssignmentBySymbol("k1")
    assert str(sbml_math_to_sympy(ia)) == "10**log10_k1"
    assert problem.model.is_valid()
    assert not problem.validate().has_errors()

    # -inf survives SBML serialization
    doc = libsbml.readSBMLFromString(problem.model.to_sbml_str())
    assert doc.getModel().getParameter("log10_k1").getValue() == -math.inf


def test_reparametrize_log_explicit_priors():
    problem, _ = reparametrize_log(
        _problem(),
        ["k1", "scaling"],
        priors={"k1": (C.LAPLACE, [0, 1]), "scaling": (C.NORMAL, [1, 2])},
    )
    assert problem["log10_k1"].prior_distribution == C.LAPLACE
    assert problem["log10_k1"].prior_parameters == [0, 1]
    assert problem["log10_scaling"].prior_distribution == C.NORMAL
    assert problem["log10_scaling"].prior_parameters == [1, 2]

    with pytest.raises(ValueError, match="unknown parameters"):
        reparametrize_log(_problem(), ["k1"], priors={"k2": (C.NORMAL, [])})


def test_reparametrize_log_fixed_with_prior():
    """Priors of non-estimated parameters are not converted."""
    problem = _problem()
    problem["offset2"].prior_distribution = C.LOG_NORMAL
    problem["offset2"].prior_parameters = [0, 1]
    with pytest.raises(NotImplementedError, match="non-estimated"):
        reparametrize_log(problem, ["offset1", "offset2"])

    problem, _ = reparametrize_log(
        problem, ["offset1", "offset2"], priors={"offset2": None}
    )
    assert problem["log10_offset2"].prior_distribution is None


@pytest.mark.parametrize("prior", [C.UNIFORM, C.NORMAL, C.GAMMA])
def test_reparametrize_log_inexpressible_prior(prior):
    """Priors of ``p`` without a supported counterpart for ``log(p)`` are
    rejected. E.g., a uniform prior on ``p`` is not uniform on ``log(p)``."""
    problem = _problem()
    problem["scaling"].prior_distribution = prior
    problem["scaling"].prior_parameters = [1, 2]
    with pytest.raises(ValueError, match="cannot be expressed"):
        reparametrize_log(problem, ["scaling"])


def test_reparametrize_log_zero_lower_bound():
    """A lower bound of 0 becomes -inf. This requires a prior if other
    parameters have priors."""
    problem = _problem()
    problem["k1"].lb = 0
    problem, _ = reparametrize_log(problem, ["k1"])
    assert problem["log10_k1"].lb == -math.inf
    assert problem["log10_k1"].prior_distribution == C.NORMAL
    assert not problem.validate().has_errors()

    problem = _problem()
    problem["scaling"].lb = 0
    with pytest.raises(ValueError, match="infinite bounds and no prior"):
        reparametrize_log(problem, ["scaling"])
    problem, _ = reparametrize_log(
        problem, ["scaling"], priors={"scaling": (C.NORMAL, [0, 1])}
    )
    assert problem["log10_scaling"].lb == -math.inf

    # priors of non-estimated parameters are ignored
    problem = _problem()
    problem["scaling"].lb = 0
    problem["offset2"].prior_distribution = C.NORMAL
    problem["offset2"].prior_parameters = [0, 1]
    problem, _ = reparametrize_log(
        problem,
        ["k1", "k2_c1", "scaling"],
        priors={"k1": None, "k2_c1": None},
    )
    assert problem["log10_scaling"].lb == -math.inf
    assert problem["log10_scaling"].prior_distribution is None


def test_reparametrize_log_invalid():
    problem = _problem()
    problem["scaling"].lb = -1
    with pytest.raises(ValueError, match="non-negative lower bound"):
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

    problem = _problem()
    problem.config = ProblemConfig(
        extensions={"some_ext": {"version": "1.0.0", "required": False}}
    )
    reparametrize_log(problem, ["k1"])
    problem.config.extensions["some_ext"].required = True
    with pytest.raises(NotImplementedError, match="some_ext"):
        reparametrize_log(problem, ["k1"])


@pytest.mark.parametrize("base", [10, "e"])
def test_reparametrize_log_sbml_l2(base):
    problem = _problem()
    doc = problem.model.sbml_model.getSBMLDocument()
    assert doc.setLevelAndVersion(2, 4)

    problem, id_map = reparametrize_log(problem, ["k1"], base=base)

    sbml_model = problem.model.sbml_model
    assert sbml_model.getLevel() == 2
    ia = sbml_model.getInitialAssignmentBySymbol("k1")
    assert str(sbml_math_to_sympy(ia)) == (
        f"10**{id_map['k1']}" if base == 10 else f"exp({id_map['k1']})"
    )
    assert problem.model.is_valid()


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
    assert "- negative numbers" not in msg

    # negative numeric overrides, in several placeholders, reported at once
    problem = _problem()
    problem.measurements[0].noise_parameters = ["offset1"]
    problem.measurements[1].observable_parameters = [-2]
    problem.measurements[1].noise_parameters = [-1]
    with pytest.raises(NotImplementedError) as excinfo:
        reparametrize_log(problem, ["offset1"])
    msg = str(excinfo.value)
    assert "Observable placeholder `offset_A'" in msg
    assert "Noise placeholder `noiseParameter1_obs_A'" in msg
    assert "negative numbers: -2\n" in msg
    assert "negative numbers: -1\n" in msg
    assert "not reparametrized" not in msg


def test_reparametrize_log_zero_override():
    """Zero overrides are replaced by a fixed parameter with value -inf."""
    problem = _problem()
    problem.measurements[0].noise_parameters = ["offset1"]
    problem.measurements[1].observable_parameters = [0]
    problem.measurements[1].noise_parameters = [0]
    problem.parameter_tables[0].elements = [
        p for p in problem.parameters if p.id != "offset2"
    ]
    # the preferred ID is taken
    sbml_par = problem.model.sbml_model.createParameter()
    sbml_par.setId("negative_infinity")
    sbml_par.setConstant(True)
    assert not problem.validate().has_errors()

    problem, id_map = reparametrize_log(problem, ["offset1"])
    assert id_map == {"offset1": "log10_offset1"}
    # a single parameter for all zero overrides
    neg_inf = petab_symbol("negative_infinity_1")
    assert [
        (m.observable_parameters, m.noise_parameters)
        for m in problem.measurements
    ] == [
        ([petab_symbol("log10_offset1")], [petab_symbol("log10_offset1")]),
        ([neg_inf], [neg_inf]),
    ]
    par = problem["negative_infinity_1"]
    assert not par.estimate
    assert par.nominal_value == -math.inf
    assert "negative_infinity_1" in [p.id for p in problem.parameters]
    assert problem.observables[0].formula == (
        petab_symbol("scaling") * petab_symbol("A")
        + 10 ** petab_symbol("offset_A")
    )
    assert not problem.validate().has_errors()


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

    problem, _ = reparametrize_log(problem, ["sigma", "offset1", "offset2"])
    assert problem.observables[0].noise_formula == (
        10 ** petab_symbol("noiseParameter1_obs_A")
        * 10 ** petab_symbol("offset_A")
    )
    assert [m.noise_parameters for m in problem.measurements] == [
        [petab_symbol("log10_sigma")],
        [petab_symbol("log10_sigma")],
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
    problem, id_map = reparametrize_log(problem, ["scaling"], base=base)
    new_dist = problem[id_map["scaling"]].prior_dist

    b = 10 if base == 10 else math.e
    thetas = np.linspace(-1.5, 1.5, 7)
    xs = b**thetas
    # density transformation: pi_q(theta) = pi_p(b^theta) * b^theta * ln(b)
    expected = old_dist.pdf(xs) * xs * math.log(b)
    np.testing.assert_allclose(new_dist.pdf(thetas), expected, rtol=1e-10)
