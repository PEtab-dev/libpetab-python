import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from petab import v1, v2
from petab.v1.calculate import calculate_llh as calculate_llh_v1
from petab.v1.models.sbml_model import SbmlModel
from petab.v1.priors import Prior
from petab.v2 import Problem
from petab.v2.calculate import calculate_llh as calculate_llh_v2
from petab.v2.petab1to2 import petab1to2, v1v2_observable_df


def test_v1v2_observable_df_noise_distribution():
    """Test that noiseDistribution is correctly merged with
    observableTransformation."""
    observable_df = pd.DataFrame(
        data={
            v1.C.OBSERVABLE_ID: ["obs1", "obs2"],
            v1.C.OBSERVABLE_FORMULA: ["a", "b"],
            v1.C.NOISE_FORMULA: ["1", "1"],
            v1.C.OBSERVABLE_TRANSFORMATION: [v1.C.LIN, v1.C.LOG],
            v1.C.NOISE_DISTRIBUTION: [v1.C.NORMAL, v1.C.NORMAL],
        }
    ).set_index(v1.C.OBSERVABLE_ID)

    new_df = v1v2_observable_df(observable_df)

    assert list(new_df[v2.C.NOISE_DISTRIBUTION]) == [
        v2.C.NORMAL,
        v2.C.LOG_NORMAL,
    ]


def test_v1v2_observable_df_log10():
    """Test that log10-transformed observables are converted to
    log-transformed ones with the noise formula scaled by ln(10)."""
    observable_df = pd.DataFrame(
        data={
            v1.C.OBSERVABLE_ID: ["obs1", "obs2", "obs3"],
            v1.C.OBSERVABLE_FORMULA: ["a", "b", "c"],
            v1.C.NOISE_FORMULA: [
                "noiseParameter1_obs1 + 0.1",
                "noiseParameter1_obs2 + 0.1",
                "noiseParameter1_obs3 + 0.1",
            ],
            v1.C.OBSERVABLE_TRANSFORMATION: [
                v1.C.LOG,
                v1.C.LOG10,
                v1.C.LOG10,
            ],
            v1.C.NOISE_DISTRIBUTION: [
                v1.C.NORMAL,
                v1.C.NORMAL,
                v1.C.LAPLACE,
            ],
        }
    ).set_index(v1.C.OBSERVABLE_ID)

    with pytest.warns(UserWarning, match=r"observables obs2, obs3 were"):
        new_df = v1v2_observable_df(observable_df)

    assert list(new_df[v2.C.NOISE_DISTRIBUTION]) == [
        v2.C.LOG_NORMAL,
        v2.C.LOG_NORMAL,
        v2.C.LOG_LAPLACE,
    ]
    assert list(new_df[v2.C.NOISE_FORMULA]) == [
        "noiseParameter1_obs1 + 0.1",
        "log(10) * (noiseParameter1_obs2 + 0.1)",
        "log(10) * (noiseParameter1_obs3 + 0.1)",
    ]
    assert list(new_df[v2.C.NOISE_PLACEHOLDERS]) == [
        "noiseParameter1_obs1",
        "noiseParameter1_obs2",
        "noiseParameter1_obs3",
    ]


def _write_v1_problem(
    path: Path,
    *,
    observables: list[dict] | None = None,
    measurements: list[dict] | None = None,
    parameters: list[dict],
) -> str:
    """Write a PEtab v1 problem for the conversion A -> B with the given
    tables (as lists of rows) to ``path`` and return the YAML file name.

    By default, there is one measurement of the observable ``obs_B = B``.
    """
    observables = observables or [
        {
            v1.C.OBSERVABLE_ID: "obs_B",
            v1.C.OBSERVABLE_FORMULA: "B",
            v1.C.NOISE_FORMULA: "0.1",
        }
    ]
    measurements = measurements or [
        {
            v1.C.OBSERVABLE_ID: "obs_B",
            v1.C.SIMULATION_CONDITION_ID: "c0",
            v1.C.TIME: 1.0,
            v1.C.MEASUREMENT: 0.3,
        }
    ]
    problem = v1.Problem(
        model=SbmlModel.from_antimony(
            "r1: A -> B; k1 * A; A = 1; B = 0; k1 = 0.5"
        ),
        condition_df=v1.get_condition_df(
            pd.DataFrame([{v1.C.CONDITION_ID: "c0", "A": 1.0}])
        ),
        observable_df=v1.get_observable_df(pd.DataFrame(observables)),
        measurement_df=v1.get_measurement_df(pd.DataFrame(measurements)),
        parameter_df=v1.get_parameter_df(pd.DataFrame(parameters)),
    )
    return problem.to_files_generic(path)


_K1 = {
    v1.C.PARAMETER_ID: "k1",
    v1.C.PARAMETER_SCALE: v1.C.LIN,
    v1.C.LOWER_BOUND: 1e-3,
    v1.C.UPPER_BOUND: 1e2,
    v1.C.NOMINAL_VALUE: 0.5,
    v1.C.ESTIMATE: 1,
}


@pytest.mark.parametrize("noise_distribution", [v1.C.NORMAL, v1.C.LAPLACE])
@pytest.mark.parametrize("noise_placeholder", [False, True])
def test_petab1to2_log10_noise_llh(
    tmp_path, noise_distribution, noise_placeholder
):
    """Test that the conversion of log10-transformed observables preserves
    the likelihood."""
    if noise_placeholder:
        noise_formula = "noiseParameter1_obs_B + offset"
        noise_parameters = "sigma_B"
        noise_parameter_rows = [
            {
                v1.C.PARAMETER_ID: "sigma_B",
                v1.C.PARAMETER_SCALE: v1.C.LIN,
                v1.C.LOWER_BOUND: 1e-3,
                v1.C.UPPER_BOUND: 1e2,
                v1.C.NOMINAL_VALUE: 0.1,
                v1.C.ESTIMATE: 1,
            },
            {
                v1.C.PARAMETER_ID: "offset",
                v1.C.PARAMETER_SCALE: v1.C.LIN,
                v1.C.LOWER_BOUND: 0,
                v1.C.UPPER_BOUND: 1,
                v1.C.NOMINAL_VALUE: 0.05,
                v1.C.ESTIMATE: 0,
            },
        ]
    else:
        noise_formula = "0.1"
        noise_parameters = ""
        noise_parameter_rows = []

    yaml_file = _write_v1_problem(
        tmp_path,
        observables=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.OBSERVABLE_FORMULA: "B",
                v1.C.OBSERVABLE_TRANSFORMATION: v1.C.LOG10,
                v1.C.NOISE_FORMULA: noise_formula,
                v1.C.NOISE_DISTRIBUTION: noise_distribution,
            }
        ],
        measurements=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.SIMULATION_CONDITION_ID: "c0",
                v1.C.TIME: t,
                v1.C.MEASUREMENT: y,
                v1.C.NOISE_PARAMETERS: noise_parameters,
            }
            for t, y in ((1.0, 0.3), (2.0, 0.5))
        ],
        parameters=[_K1, *noise_parameter_rows],
    )
    v1_problem = v1.Problem.from_yaml(yaml_file)
    with pytest.warns(UserWarning, match=r"observables obs_B were"):
        v2_problem = petab1to2(yaml_file)

    def simulate(measurement_df):
        return measurement_df.rename(
            columns={v1.C.MEASUREMENT: v1.C.SIMULATION}
        ).assign(**{v1.C.SIMULATION: [0.35, 0.45]})

    llh_v1 = calculate_llh_v1(
        v1_problem.measurement_df,
        simulate(v1_problem.measurement_df),
        v1_problem.observable_df,
        v1_problem.parameter_df,
    )
    llh_v2 = calculate_llh_v2(
        v2_problem.measurement_df,
        simulate(v2_problem.measurement_df),
        v2_problem.observable_df,
        v2_problem.parameter_df,
    )
    assert llh_v2 == pytest.approx(llh_v1, rel=1e-12)


@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning"
)
@pytest.mark.parametrize(
    "parameter_scale, prior_type, prior_parameters",
    [
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_NORMAL, "-0.5;0.7"),
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_LAPLACE, "-0.5;0.7"),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_NORMAL, "-0.5;0.7"),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_LAPLACE, "-0.5;0.7"),
        (v1.C.LIN, v1.C.PARAMETER_SCALE_NORMAL, "-0.5;0.7"),
        # parameterScaleUniform within the bounds, exceeding the bounds
        #  (also beyond the floating-point range when unscaled),
        #  and defaulting to the bounds
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_UNIFORM, "-0.5;0.7"),
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_UNIFORM, "-5;0.7"),
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_UNIFORM, "-400;400"),
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_UNIFORM, np.nan),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_UNIFORM, "-0.5;0.7"),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_UNIFORM, "-10;0.7"),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_UNIFORM, "-1000;1000"),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_UNIFORM, np.nan),
        (v1.C.LIN, v1.C.PARAMETER_SCALE_UNIFORM, "0.1;0.7"),
        # logNormal / logLaplace do not depend on the parameter scale
        (v1.C.LOG10, v1.C.LOG_NORMAL, "-0.5;0.7"),
        (v1.C.LOG10, v1.C.LOG_LAPLACE, "-0.5;0.7"),
        (v1.C.LIN, v1.C.LOG_NORMAL, "-0.5;0.7"),
        (v1.C.LIN, v1.C.LOG_LAPLACE, "-0.5;0.7"),
    ],
)
def test_petab1to2_priors(
    tmp_path, parameter_scale, prior_type, prior_parameters
):
    """Test that objective priors are converted to priors with the same
    density."""
    k1 = _K1 | {
        v1.C.PARAMETER_SCALE: parameter_scale,
        v1.C.OBJECTIVE_PRIOR_TYPE: prior_type,
        v1.C.OBJECTIVE_PRIOR_PARAMETERS: prior_parameters,
    }
    yaml_file = _write_v1_problem(
        tmp_path,
        observables=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.OBSERVABLE_FORMULA: "B",
                v1.C.NOISE_FORMULA: "0.1",
            }
        ],
        measurements=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.SIMULATION_CONDITION_ID: "c0",
                v1.C.TIME: 1.0,
                v1.C.MEASUREMENT: 0.3,
            }
        ],
        parameters=[k1],
    )
    v1_prior = Prior.from_par_dict(k1, type_="objective")
    v2_prior = petab1to2(yaml_file)["k1"].prior_dist

    x = np.logspace(-3, 2, 21)
    assert v2_prior.pdf(x) == pytest.approx(v1_prior.pdf(x), rel=1e-12)


@pytest.mark.parametrize(
    "estimate_k1, sigma_prior", [(0, False), (1, False), (1, True)]
)
def test_petab1to2_default_priors(tmp_path, estimate_k1, sigma_prior):
    """The default prior is only made explicit for estimated parameters,
    and only if any estimated parameter has a prior. Priors of
    non-estimated parameters are kept."""
    fixed = {
        v1.C.PARAMETER_SCALE: v1.C.LIN,
        v1.C.LOWER_BOUND: np.nan,
        v1.C.UPPER_BOUND: np.nan,
        v1.C.NOMINAL_VALUE: 0.05,
        v1.C.ESTIMATE: 0,
    }
    sigma = _K1 | {v1.C.PARAMETER_ID: "sigma"}
    if sigma_prior:
        sigma |= {
            v1.C.OBJECTIVE_PRIOR_TYPE: v1.C.NORMAL,
            v1.C.OBJECTIVE_PRIOR_PARAMETERS: "0.1;1",
        }
    yaml_file = _write_v1_problem(
        tmp_path,
        observables=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.OBSERVABLE_FORMULA: "B + offset1 + offset2",
                v1.C.NOISE_FORMULA: "sigma",
            }
        ],
        parameters=[
            _K1 | {v1.C.ESTIMATE: estimate_k1},
            sigma,
            # (the prior is ignored for non-estimated parameters)
            fixed
            | {
                v1.C.PARAMETER_ID: "offset1",
                v1.C.OBJECTIVE_PRIOR_TYPE: v1.C.NORMAL,
                v1.C.OBJECTIVE_PRIOR_PARAMETERS: "0;1",
            },
            fixed | {v1.C.PARAMETER_ID: "offset2"},
        ],
    )
    problem = petab1to2(yaml_file)

    if estimate_k1 and sigma_prior:
        assert problem["k1"].prior_distribution == v2.C.UNIFORM
        assert problem["sigma"].prior_distribution == v2.C.NORMAL
    else:
        assert problem["k1"].prior_distribution is None
        assert problem["sigma"].prior_distribution is None
    assert problem["offset1"].prior_distribution == v2.C.NORMAL
    assert problem["offset1"].prior_parameters == [0, 1]
    assert problem["offset2"].prior_distribution is None


@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning"
)
@pytest.mark.parametrize(
    "parameter_scale, prior_type, prior_parameters, reparametrize",
    [
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_NORMAL, "-0.5;0.7", "priors"),
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_LAPLACE, "-0.5;0.7", "priors"),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_NORMAL, "-0.5;0.7", "priors"),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_LAPLACE, "-0.5;0.7", "all"),
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_UNIFORM, "-0.5;0.7", "priors"),
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_UNIFORM, "-400;400", "priors"),
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_UNIFORM, np.nan, "all"),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_UNIFORM, "-1000;1000", "priors"),
        (v1.C.LOG10, np.nan, np.nan, "priors"),
        (v1.C.LOG10, np.nan, np.nan, "all"),
    ],
)
def test_petab1to2_reparametrize_priors(
    tmp_path, parameter_scale, prior_type, prior_parameters, reparametrize
):
    """Test that reparametrization preserves parameter-scale priors,
    including their normalization constant up to truncation."""
    k1 = _K1 | {
        v1.C.PARAMETER_SCALE: parameter_scale,
        v1.C.OBJECTIVE_PRIOR_TYPE: prior_type,
        v1.C.OBJECTIVE_PRIOR_PARAMETERS: prior_parameters,
    }
    yaml_file = _write_v1_problem(
        tmp_path,
        parameters=[k1],
    )
    problem = petab1to2(yaml_file, reparametrize=reparametrize)

    if pd.isna(prior_type) and reparametrize == "priors":
        # the default prior is only reparametrized with "all"
        assert [p.id for p in problem.parameters] == ["k1"]
        return

    new_id = "log10_k1" if parameter_scale == v1.C.LOG10 else "log_k1"
    assert [p.id for p in problem.parameters] == [new_id]
    par = problem[new_id]
    lb, ub = (
        v1.parameters.scale(x, parameter_scale)
        for x in (_K1[v1.C.LOWER_BOUND], _K1[v1.C.UPPER_BOUND])
    )
    assert (par.lb, par.ub) == pytest.approx((lb, ub))
    assert par.nominal_value == pytest.approx(
        v1.parameters.scale(_K1[v1.C.NOMINAL_VALUE], parameter_scale)
    )

    # k1 = b^q in the model
    sbml_model = problem.model.sbml_model
    assert sbml_model.getInitialAssignmentBySymbol("k1") is not None
    assert sbml_model.getParameter(new_id).getConstant()

    if pd.isna(prior_type):
        # no priors at all
        assert par.prior_distribution is None
        return

    # the v2 prior of the new parameter is the v1 prior density over the
    #  scaled parameter (both truncated to the bounds)
    v1_prior = Prior.from_par_dict(k1, type_="objective")
    if prior_type == v1.C.PARAMETER_SCALE_UNIFORM and not pd.isna(
        prior_parameters
    ):
        # evaluate within the support
        low, high = map(float, prior_parameters.split(";"))
        lb, ub = max(lb, low), min(ub, high)
    theta = np.linspace(lb, ub, 13)[1:-1]
    v1_pdf = v1_prior.pdf(theta, x_scaled=True, rescale=True)
    v2_pdf = par.prior_dist.pdf(theta)
    if prior_type in (
        v1.C.PARAMETER_SCALE_NORMAL,
        v1.C.PARAMETER_SCALE_LAPLACE,
    ):
        assert np.ptp(np.log(v1_pdf)) > 1
    else:
        # uniform within the bounds
        assert np.ptp(np.log(v1_pdf)) < 1e-12
    assert np.all(v1_pdf > 0)
    np.testing.assert_allclose(v2_pdf, v1_pdf, rtol=1e-12)


@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning"
)
def test_petab1to2_reparametrize_llh(tmp_path):
    """Test that reparametrizing overrides and parameters used in formulas
    preserves the likelihood, for both bases."""

    def par(par_id, scale, nominal_value):
        return {
            v1.C.PARAMETER_ID: par_id,
            v1.C.PARAMETER_SCALE: scale,
            v1.C.LOWER_BOUND: 1e-3,
            v1.C.UPPER_BOUND: 1e3,
            v1.C.NOMINAL_VALUE: nominal_value,
            v1.C.ESTIMATE: 1,
            v1.C.OBJECTIVE_PRIOR_TYPE: v1.C.PARAMETER_SCALE_NORMAL,
            v1.C.OBJECTIVE_PRIOR_PARAMETERS: "0;1",
        }

    yaml_file = _write_v1_problem(
        tmp_path,
        observables=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.OBSERVABLE_FORMULA: "observableParameter1_obs_B * B "
                "+ offset",
                v1.C.NOISE_FORMULA: "noiseParameter1_obs_B",
            }
        ],
        measurements=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.SIMULATION_CONDITION_ID: "c0",
                v1.C.TIME: float(t),
                v1.C.MEASUREMENT: 0.3 + t,
                v1.C.OBSERVABLE_PARAMETERS: obs_par,
                v1.C.NOISE_PARAMETERS: noise_par,
            }
            for t, (obs_par, noise_par) in enumerate(
                [("s1", "sd"), ("s2", "0.2"), ("4", "sd")]
            )
        ],
        parameters=[
            _K1,
            par("s1", v1.C.LOG10, 3.0),
            par("s2", v1.C.LOG10, 0.7),
            par("sd", v1.C.LOG, 0.25),
            par("offset", v1.C.LOG10, 0.13),
        ],
    )
    v1_problem = v1.Problem.from_yaml(yaml_file)
    v2_problem = petab1to2(yaml_file, reparametrize="priors")
    assert [p.id for p in v2_problem.parameters] == [
        "k1",
        "log10_s1",
        "log10_s2",
        "log_sd",
        "log10_offset",
    ]

    def simulate(measurement_df):
        return measurement_df.rename(
            columns={v1.C.MEASUREMENT: v1.C.SIMULATION}
        ).assign(**{v1.C.SIMULATION: [0.5, 0.6, 0.7]})

    llh_v1 = calculate_llh_v1(
        v1_problem.measurement_df,
        simulate(v1_problem.measurement_df),
        v1_problem.observable_df,
        v1_problem.parameter_df,
    )
    llh_v2 = calculate_llh_v2(
        v2_problem.measurement_df,
        simulate(v2_problem.measurement_df),
        v2_problem.observable_df,
        v2_problem.parameter_df,
    )
    assert np.isfinite(llh_v1)
    assert llh_v2 == pytest.approx(llh_v1, rel=1e-12)


def _write_v1_problem_zero_lower_bound(
    tmp_path, prior_type=None, prior_parameters=None, sigma_prior=False
) -> str:
    """Write a v1 problem with a log10-scaled `k1` with lower bound 0.

    If `sigma_prior`, the noise parameter `sigma` has a prior.
    """
    # v1 allows a lower bound of 0 for log-scaled parameters
    #  if there is an initialization prior
    k1 = _K1 | {
        v1.C.PARAMETER_SCALE: v1.C.LOG10,
        v1.C.LOWER_BOUND: 0.0,
        v1.C.INITIALIZATION_PRIOR_TYPE: v1.C.PARAMETER_SCALE_UNIFORM,
        v1.C.INITIALIZATION_PRIOR_PARAMETERS: "-3;2",
    }
    if prior_type is not None:
        k1[v1.C.OBJECTIVE_PRIOR_TYPE] = prior_type
        k1[v1.C.OBJECTIVE_PRIOR_PARAMETERS] = prior_parameters
    parameters = [k1]
    observables = None
    if sigma_prior:
        parameters.append(
            _K1
            | {
                v1.C.PARAMETER_ID: "sigma",
                v1.C.OBJECTIVE_PRIOR_TYPE: v1.C.NORMAL,
                v1.C.OBJECTIVE_PRIOR_PARAMETERS: "0.1;1",
            }
        )
        observables = [
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.OBSERVABLE_FORMULA: "B",
                v1.C.NOISE_FORMULA: "sigma",
            }
        ]
    yaml_file = _write_v1_problem(
        tmp_path, observables=observables, parameters=parameters
    )
    assert not v1.lint_problem(v1.Problem.from_yaml(yaml_file))
    return yaml_file


@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning",
    "ignore:Initialisation priors.*:UserWarning",
)
@pytest.mark.parametrize(
    "prior_type,prior_parameters,reparametrize,expected_prior",
    [
        (
            v1.C.PARAMETER_SCALE_NORMAL,
            "-1;0.5",
            "priors",
            (v2.C.NORMAL, [-1, 0.5]),
        ),
        (
            v1.C.PARAMETER_SCALE_NORMAL,
            "-1;0.5",
            "all",
            (v2.C.NORMAL, [-1, 0.5]),
        ),
        (
            v1.C.PARAMETER_SCALE_UNIFORM,
            "-3;1",
            "priors",
            (v2.C.UNIFORM, [-3, 1]),
        ),
        # no priors at all
        (None, None, "all", (None, [])),
        # no priors, but an empty prior column
        (np.nan, np.nan, "all", (None, [])),
    ],
)
def test_petab1to2_reparametrize_zero_lower_bound(
    tmp_path, prior_type, prior_parameters, reparametrize, expected_prior
):
    """Parameters with a lower bound of 0 and a proper prior on log scale
    are reparametrized, with a lower bound of -inf."""
    yaml_file = _write_v1_problem_zero_lower_bound(
        tmp_path, prior_type, prior_parameters
    )

    problem = petab1to2(yaml_file, reparametrize=reparametrize)

    assert [p.id for p in problem.parameters] == ["log10_k1"]
    par = problem["log10_k1"]
    assert (par.lb, par.ub) == (-np.inf, 2)
    assert (par.prior_distribution, par.prior_parameters) == expected_prior
    ia = problem.model.sbml_model.getInitialAssignmentBySymbol("k1")
    assert ia is not None


@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning",
    "ignore:Initialisation priors.*:UserWarning",
)
def test_petab1to2_reparametrize_zero_lower_bound_improper(tmp_path):
    """Parameters with a lower bound of 0 and an improper prior on log scale
    are skipped."""
    # the v1 default prior (uniform on parameter scale) is made explicit
    #  if other estimated parameters have priors
    yaml_file = _write_v1_problem_zero_lower_bound(
        tmp_path, np.nan, np.nan, sigma_prior=True
    )

    with pytest.warns(UserWarning, match=r"k1 were not reparametrized"):
        problem = petab1to2(yaml_file, reparametrize="all")

    # as with reparametrize="none"
    assert [p.id for p in problem.parameters] == ["k1", "sigma"]
    par = problem["k1"]
    assert (par.lb, par.ub) == (0.0, _K1[v1.C.UPPER_BOUND])
    assert par.prior_distribution == v2.C.UNIFORM
    assert par.prior_parameters == [0.0, _K1[v1.C.UPPER_BOUND]]
    sbml_model = problem.model.sbml_model
    assert sbml_model.getInitialAssignmentBySymbol("k1") is None
    assert sbml_model.getAssignmentRuleByVariable("k1") is None


def _write_v1_problem_with_overrides(
    tmp_path, parameters: list[dict], overrides: list[str]
) -> str:
    return _write_v1_problem(
        tmp_path,
        observables=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.OBSERVABLE_FORMULA: "observableParameter1_obs_B * B",
                v1.C.NOISE_FORMULA: "0.1",
            }
        ],
        measurements=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.SIMULATION_CONDITION_ID: "c0",
                v1.C.TIME: float(i),
                v1.C.MEASUREMENT: 0.3,
                v1.C.OBSERVABLE_PARAMETERS: override,
            }
            for i, override in enumerate(overrides)
        ],
        parameters=[_K1, *parameters],
    )


_LOG10_PAR = {
    v1.C.PARAMETER_SCALE: v1.C.LOG10,
    v1.C.LOWER_BOUND: 1e-3,
    v1.C.UPPER_BOUND: 1e3,
    v1.C.NOMINAL_VALUE: 2,
    v1.C.ESTIMATE: 1,
}


@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning"
)
def test_petab1to2_reparametrize_overrides(tmp_path):
    """Parameters that share a placeholder with selected parameters are
    reparametrized as well."""
    yaml_file = _write_v1_problem_with_overrides(
        tmp_path,
        parameters=[
            _LOG10_PAR
            | {
                v1.C.PARAMETER_ID: "s1",
                v1.C.OBJECTIVE_PRIOR_TYPE: v1.C.PARAMETER_SCALE_NORMAL,
                v1.C.OBJECTIVE_PRIOR_PARAMETERS: "0;1",
            },
            _LOG10_PAR | {v1.C.PARAMETER_ID: "s2"},
            # non-estimated, fixed at 0, with a prior
            _LOG10_PAR
            | {
                v1.C.PARAMETER_ID: "s3",
                v1.C.ESTIMATE: 0,
                v1.C.NOMINAL_VALUE: 0,
                v1.C.OBJECTIVE_PRIOR_TYPE: v1.C.PARAMETER_SCALE_NORMAL,
                v1.C.OBJECTIVE_PRIOR_PARAMETERS: "0;1",
            },
            # non-estimated, with a prior that depends on missing bounds
            _LOG10_PAR
            | {
                v1.C.PARAMETER_ID: "s4",
                v1.C.ESTIMATE: 0,
                v1.C.LOWER_BOUND: np.nan,
                v1.C.UPPER_BOUND: np.nan,
                v1.C.OBJECTIVE_PRIOR_TYPE: v1.C.PARAMETER_SCALE_UNIFORM,
            },
        ],
        overrides=["s1", "s2", "100", "s3", "s4"],
    )
    assert not v1.lint_problem(v1.Problem.from_yaml(yaml_file))
    # s2-s4 aren't selected, but share the placeholder with s1
    with pytest.warns(UserWarning, match="non-estimated parameters s4 were"):
        problem = petab1to2(yaml_file, reparametrize="priors")
    assert [p.id for p in problem.parameters] == [
        "k1",
        "log10_s1",
        "log10_s2",
        "log10_s3",
        "log10_s4",
    ]
    assert problem["log10_s1"].prior_distribution == v2.C.NORMAL
    assert problem["log10_s1"].prior_parameters == [0, 1]
    # explicit default prior, as other parameters have priors
    assert problem["log10_s2"].prior_distribution == v2.C.UNIFORM
    assert problem["log10_s2"].prior_parameters == pytest.approx([-3, 3])
    s3 = problem["log10_s3"]
    assert not s3.estimate
    assert s3.nominal_value == -np.inf
    # the parameter-scale prior is kept
    assert s3.prior_distribution == v2.C.NORMAL
    assert s3.prior_parameters == [0, 1]
    s4 = problem["log10_s4"]
    assert s4.nominal_value == pytest.approx(np.log10(2))
    assert s4.prior_distribution is None


@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning"
)
def test_petab1to2_reparametrize_all_skips_unresolvable(tmp_path):
    """With "all", parameters without parameter-scale priors are skipped if
    their placeholder can't be log-transformed."""
    yaml_file = _write_v1_problem_with_overrides(
        tmp_path,
        parameters=[
            _LOG10_PAR | {v1.C.PARAMETER_ID: "s1"},
            _LOG10_PAR | {v1.C.PARAMETER_ID: "s2"},
        ],
        overrides=["s1", "s2", "0"],
    )
    with pytest.warns(UserWarning, match="s1, s2 were not reparametrized"):
        problem = petab1to2(yaml_file, reparametrize="all")
    assert [p.id for p in problem.parameters] == ["k1", "s1", "s2"]

    yaml_file = _write_v1_problem_with_overrides(
        tmp_path,
        parameters=[
            _LOG10_PAR
            | {
                v1.C.PARAMETER_ID: "s1",
                v1.C.OBJECTIVE_PRIOR_TYPE: v1.C.PARAMETER_SCALE_NORMAL,
                v1.C.OBJECTIVE_PRIOR_PARAMETERS: "0;1",
            },
            _LOG10_PAR | {v1.C.PARAMETER_ID: "s2"},
        ],
        overrides=["s1", "s2", "0"],
    )
    # s1 has a prior, so it must be reparametrized
    with (
        pytest.raises(NotImplementedError) as excinfo,
        pytest.warns(UserWarning, match="s2 were not reparametrized"),
    ):
        petab1to2(yaml_file, reparametrize="all")
    msg = str(excinfo.value)
    assert "Use `reparametrize='none'`" in msg
    assert "(overridden by `s1')" in msg
    assert "not reparametrized: `s2'" in msg
    assert "non-positive numbers: 0" in msg


@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning"
)
def test_petab1to2_reparametrize_output_parameter(tmp_path):
    """Test reparametrization of parameters that are not model entities,
    and that x-space priors are not reparametrized."""
    yaml_file = _write_v1_problem(
        tmp_path,
        observables=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.OBSERVABLE_FORMULA: "scaling * B",
                v1.C.NOISE_FORMULA: "sigma",
            }
        ],
        parameters=[
            _K1,
            _LOG10_PAR | {v1.C.PARAMETER_ID: "scaling"},
            _LOG10_PAR
            | {
                v1.C.PARAMETER_ID: "sigma",
                v1.C.OBJECTIVE_PRIOR_TYPE: v1.C.LOG_NORMAL,
                v1.C.OBJECTIVE_PRIOR_PARAMETERS: "0;1",
            },
        ],
    )
    problem = petab1to2(yaml_file, reparametrize="all")
    assert [p.id for p in problem.parameters] == [
        "k1",
        "log10_scaling",
        "sigma",
    ]
    assert problem["log10_scaling"].prior_distribution == v2.C.UNIFORM


@pytest.mark.parametrize(
    "prior_type",
    [
        v1.C.PARAMETER_SCALE_NORMAL,
        v1.C.PARAMETER_SCALE_LAPLACE,
        v1.C.PARAMETER_SCALE_UNIFORM,
    ],
)
def test_petab1to2_reparametrize_warnings_and_errors(tmp_path, prior_type):
    k1 = _K1 | {
        v1.C.PARAMETER_SCALE: v1.C.LOG10,
        v1.C.OBJECTIVE_PRIOR_TYPE: prior_type,
        v1.C.OBJECTIVE_PRIOR_PARAMETERS: "-1;1",
    }
    yaml_file = _write_v1_problem(
        tmp_path,
        parameters=[k1],
    )
    with pytest.warns(UserWarning, match=r"of k1 were converted.*'priors'"):
        petab1to2(yaml_file)

    with pytest.raises(ValueError, match="Invalid value"):
        petab1to2(yaml_file, reparametrize="some")


def test_petab1to2_placeholder_order(tmp_path):
    """Test that placeholders are ordered by index, not as text, so that the
    positional overrides still match for 10+ placeholders."""
    obs_id = "obs_B"
    n_obs, n_noise = 12, 11
    observable_df = pd.DataFrame(
        data={
            v1.C.OBSERVABLE_ID: [obs_id],
            v1.C.OBSERVABLE_FORMULA: [
                " + ".join(
                    f"observableParameter{i}_{obs_id} * B^{i}"
                    for i in range(1, n_obs + 1)
                )
            ],
            v1.C.NOISE_FORMULA: [
                " + ".join(
                    f"noiseParameter{i}_{obs_id}"
                    for i in range(1, n_noise + 1)
                )
            ],
        }
    ).set_index(v1.C.OBSERVABLE_ID)
    # the i-th observable override is i, the i-th noise override is 100 + i
    measurement_df = pd.DataFrame(
        data={
            v1.C.OBSERVABLE_ID: [obs_id, obs_id],
            v1.C.SIMULATION_CONDITION_ID: ["c0", "c0"],
            v1.C.TIME: [1.0, 2.0],
            v1.C.MEASUREMENT: [0.3, 0.5],
            v1.C.OBSERVABLE_PARAMETERS: ";".join(
                str(i) for i in range(1, n_obs + 1)
            ),
            v1.C.NOISE_PARAMETERS: ";".join(
                str(100 + i) for i in range(1, n_noise + 1)
            ),
        }
    )
    condition_df = pd.DataFrame(
        data={v1.C.CONDITION_ID: ["c0"], "A": [1.0]}
    ).set_index(v1.C.CONDITION_ID)
    parameter_df = pd.DataFrame(
        data={
            v1.C.PARAMETER_ID: ["k1"],
            v1.C.PARAMETER_SCALE: [v1.C.LIN],
            v1.C.LOWER_BOUND: [1e-3],
            v1.C.UPPER_BOUND: [1e3],
            v1.C.NOMINAL_VALUE: [0.5],
            v1.C.ESTIMATE: [1],
        }
    ).set_index(v1.C.PARAMETER_ID)
    v1_problem = v1.Problem(
        model=v1.models.sbml_model.SbmlModel.from_antimony(
            "A -> B; k1 * A; A = 1; B = 0; k1 = 0.5"
        ),
        condition_df=condition_df,
        measurement_df=measurement_df,
        parameter_df=parameter_df,
        observable_df=observable_df,
    )

    problem = petab1to2(v1_problem.to_files_generic(tmp_path))

    observable = problem[obs_id]
    assert len(problem.measurements) == 2
    for measurement in problem.measurements:
        assert {
            str(placeholder): float(override)
            for placeholder, override in zip(
                observable.observable_placeholders,
                measurement.observable_parameters,
                strict=True,
            )
        } == {
            f"observableParameter{i}_{obs_id}": i for i in range(1, n_obs + 1)
        }
        assert {
            str(placeholder): float(override)
            for placeholder, override in zip(
                observable.noise_placeholders,
                measurement.noise_parameters,
                strict=True,
            )
        } == {
            f"noiseParameter{i}_{obs_id}": 100 + i
            for i in range(1, n_noise + 1)
        }


@pytest.mark.parametrize("type_", ["observable", "noise"])
def test_v1v2_observable_df_placeholder_gap(type_):
    """Test that non-consecutively numbered placeholders are rejected,
    because v1 overrides are positional."""
    formula = f"{type_}Parameter1_obs1 + {type_}Parameter3_obs1"
    observable_df = pd.DataFrame(
        data={
            v1.C.OBSERVABLE_ID: ["obs1"],
            v1.C.OBSERVABLE_FORMULA: [
                formula if type_ == "observable" else "a"
            ],
            v1.C.NOISE_FORMULA: [formula if type_ == "noise" else "1"],
        }
    ).set_index(v1.C.OBSERVABLE_ID)

    with pytest.raises(ValueError, match="Non-consecutive numbering"):
        v1v2_observable_df(observable_df)


def test_petab1to2_remote():
    """Test that we can upgrade a remote PEtab 1.0.0 problem."""
    yaml_url = (
        "https://cdn.jsdelivr.net/gh/PEtab-dev/petab_test_suite"
        "@main/petabtests/cases/v1.0.0/sbml/0001/_0001.yaml"
    )

    problem = petab1to2(yaml_url)
    assert isinstance(problem, Problem)
    assert len(problem.measurements)


def _petab1to2_with_conditions(
    tmp_path,
    condition_df: pd.DataFrame,
    conditions: list[tuple[str, str]],
    output_dir: Path | None = None,
) -> Problem:
    """Convert a minimal PEtab v1 problem with the given condition table.

    ``conditions`` are (preequilibrationConditionId, simulationConditionId)
    pairs, one measurement per pair. If ``output_dir`` is given, the
    converted problem is written there.
    """
    problem = v1.Problem(
        model=SbmlModel.from_antimony(
            "A = 1; B = 0; k1 = 0.5; A -> B; k1 * A"
        ),
        condition_df=v1.get_condition_df(condition_df),
        observable_df=v1.get_observable_df(
            pd.DataFrame(
                {
                    v1.C.OBSERVABLE_ID: ["obs_B"],
                    v1.C.OBSERVABLE_FORMULA: ["B"],
                    v1.C.NOISE_FORMULA: [0.1],
                }
            )
        ),
        measurement_df=v1.get_measurement_df(
            pd.DataFrame(
                {
                    v1.C.OBSERVABLE_ID: "obs_B",
                    v1.C.PREEQUILIBRATION_CONDITION_ID: [
                        preeq for preeq, _ in conditions
                    ],
                    v1.C.SIMULATION_CONDITION_ID: [
                        sim for _, sim in conditions
                    ],
                    v1.C.TIME: 1.0,
                    v1.C.MEASUREMENT: 0.5,
                }
            )
        ),
        parameter_df=v1.get_parameter_df(
            pd.DataFrame(
                {
                    v1.C.PARAMETER_ID: ["k1"],
                    v1.C.PARAMETER_SCALE: [v1.C.LIN],
                    v1.C.LOWER_BOUND: [1e-3],
                    v1.C.UPPER_BOUND: [1e3],
                    v1.C.NOMINAL_VALUE: [0.5],
                    v1.C.ESTIMATE: [1],
                }
            )
        ),
    )
    yaml_file = problem.to_files_generic(tmp_path)
    if output_dir is None:
        return petab1to2(yaml_file)
    petab1to2(yaml_file, output_dir)
    return Problem.from_yaml(output_dir / Path(yaml_file).name)


def _measured_periods(
    problem: Problem,
) -> list[list[tuple[float, list[str]]] | None]:
    """The (time, conditionIds) of the experiment periods per measurement."""
    return [
        None
        if m.experiment_id is None
        else [
            (p.time, p.condition_ids) for p in problem[m.experiment_id].periods
        ]
        for m in problem.measurements
    ]


def test_petab1to2_empty_conditions_preequilibration(tmp_path):
    """Pre-equilibration is kept if the condition table sets nothing."""
    problem = _petab1to2_with_conditions(
        tmp_path,
        pd.DataFrame({v1.C.CONDITION_ID: ["c0", "steady"]}),
        [("steady", "c0")],
    )

    assert problem.conditions == []
    assert _measured_periods(problem) == [
        [(v2.C.TIME_PREEQUILIBRATION, []), (0, [])]
    ]
    assert problem.experiments[0].has_preequilibration


def test_petab1to2_mixed_empty_conditions(tmp_path):
    """Conditions that set nothing are handled condition-wise."""
    problem = _petab1to2_with_conditions(
        tmp_path,
        pd.DataFrame(
            {
                v1.C.CONDITION_ID: ["c0", "c1", "preeq_set", "preeq_empty"],
                "A": [1.0, None, 2.0, None],
            }
        ),
        [
            ("preeq_set", "c0"),
            ("preeq_set", "c1"),
            ("preeq_empty", "c0"),
            ("preeq_empty", "c1"),
            ("", "c0"),
            ("", "c1"),
        ],
    )

    assert {c.id for c in problem.conditions} == {"c0", "preeq_set"}
    preeq = v2.C.TIME_PREEQUILIBRATION
    assert _measured_periods(problem) == [
        [(preeq, ["preeq_set"]), (0, ["c0"])],
        [(preeq, ["preeq_set"]), (0, [])],
        [(preeq, []), (0, ["c0"])],
        [(preeq, []), (0, [])],
        [(0, ["c0"])],
        # no pre-equilibration, no changes -> no experiment
        None,
    ]


def test_petab1to2_empty_conditions_no_preequilibration(tmp_path):
    """Without pre-equilibration, conditions that set nothing don't need an
    experiment."""
    problem = _petab1to2_with_conditions(
        tmp_path,
        pd.DataFrame({v1.C.CONDITION_ID: ["c0"]}),
        [("", "c0")],
    )

    assert problem.experiments == []
    assert _measured_periods(problem) == [None]


@pytest.mark.parametrize("has_changes", [False, True])
def test_petab1to2_empty_condition_table_omitted(tmp_path, has_changes):
    """A condition table without any changes is not written."""
    from petab.v1.yaml import load_yaml

    output_dir = tmp_path / "v2"
    problem = _petab1to2_with_conditions(
        tmp_path,
        pd.DataFrame(
            {
                v1.C.CONDITION_ID: ["c0", "steady"],
                "A": [2.0 if has_changes else None, None],
            }
        ),
        [("steady", "c0")],
        output_dir=output_dir,
    )

    yaml_config = load_yaml(output_dir / "problem.yaml")
    condition_file = output_dir / "conditions.tsv"
    if has_changes:
        assert yaml_config[v2.C.CONDITION_FILES] == ["conditions.tsv"]
        assert condition_file.exists()
        assert [c.id for c in problem.conditions] == ["c0"]
    else:
        assert v2.C.CONDITION_FILES not in yaml_config
        assert not condition_file.exists()
        assert problem.conditions == []
    assert problem.validate() == []


def test_petab1to2_validate(tmp_path):
    """Test that `validate=False` allows converting problems that fail
    linting for reasons unrelated to the conversion."""
    v1_problem = v1.Problem(
        model=SbmlModel.from_antimony(
            "A -> B; k1 * A; A = 1; B = 0; k1 = 0.5"
        ),
        condition_df=v1.get_condition_df(
            pd.DataFrame({v1.C.CONDITION_ID: ["c0"], "A": [1.0]})
        ),
        observable_df=v1.get_observable_df(
            pd.DataFrame(
                {
                    v1.C.OBSERVABLE_ID: ["obs_B"],
                    v1.C.OBSERVABLE_FORMULA: ["B"],
                    v1.C.NOISE_FORMULA: [0.1],
                }
            )
        ),
        measurement_df=v1.get_measurement_df(
            pd.DataFrame(
                {
                    v1.C.OBSERVABLE_ID: ["obs_B", "obs_B"],
                    v1.C.SIMULATION_CONDITION_ID: ["c0", "c0"],
                    v1.C.TIME: [1.0, 2.0],
                    v1.C.MEASUREMENT: [0.3, 0.5],
                    v1.C.DATASET_ID: ["d1", "d1"],
                }
            )
        ),
        parameter_df=v1.get_parameter_df(
            pd.DataFrame(
                {
                    v1.C.PARAMETER_ID: ["k1"],
                    v1.C.PARAMETER_SCALE: [v1.C.LIN],
                    v1.C.LOWER_BOUND: [1e-3],
                    v1.C.UPPER_BOUND: [1e3],
                    v1.C.NOMINAL_VALUE: [0.5],
                    v1.C.ESTIMATE: [1],
                }
            )
        ),
        # refers to a dataset that is not in the measurement table
        visualization_df=pd.DataFrame(
            {
                v1.C.PLOT_ID: ["p1"],
                v1.C.DATASET_ID: ["no_such_dataset"],
                v1.C.X_VALUES: [v1.C.TIME],
            }
        ),
    )
    yaml_file = v1_problem.to_files_generic(tmp_path / "v1")

    with pytest.raises(ValueError, match="does not pass linting"):
        petab1to2(yaml_file)

    problem = petab1to2(yaml_file, validate=False)
    assert isinstance(problem, Problem)
    assert len(problem.measurements) == 2

    output_dir = tmp_path / "v2"
    output_dir.mkdir()
    petab1to2(yaml_file, output_dir, validate=False)
    problem = Problem.from_yaml(output_dir / Path(yaml_file).name)
    assert len(problem.measurements) == 2


@pytest.mark.parametrize("clash", [False, True])
def test_petab1to2_existing_experiment_id_column(tmp_path, clash):
    """Test that a pre-existing `experimentId` column is renamed
    deterministically and consistently across measurement files."""
    yaml_file = _write_v1_problem(
        tmp_path / "v1",
        observables=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.OBSERVABLE_FORMULA: "B",
                v1.C.NOISE_FORMULA: 0.1,
            }
        ],
        measurements=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.SIMULATION_CONDITION_ID: "c0",
                v1.C.TIME: 1.0,
                v1.C.MEASUREMENT: 0.5,
                v2.C.EXPERIMENT_ID: "dose_response",
            }
        ],
        parameters=[_K1],
    )
    # second measurement file, which may have a clashing column
    yaml_path = tmp_path / "v1" / yaml_file
    yaml_config = yaml.safe_load(yaml_path.read_text())
    measurement_df = pd.DataFrame(
        {
            v1.C.OBSERVABLE_ID: ["obs_B"],
            v1.C.SIMULATION_CONDITION_ID: ["c0"],
            v1.C.TIME: [2.0],
            v1.C.MEASUREMENT: [0.5],
            v2.C.EXPERIMENT_ID: ["time_course"],
        }
    )
    if clash:
        measurement_df[f"{v2.C.EXPERIMENT_ID}_v1"] = "taken"
    v1.write_measurement_df(measurement_df, tmp_path / "v1" / "meas2.tsv")
    yaml_config["problems"][0]["measurement_files"].append("meas2.tsv")
    yaml_path.write_text(yaml.safe_dump(yaml_config))

    expected_name = f"{v2.C.EXPERIMENT_ID}_v1" + ("_2" if clash else "")
    with pytest.warns(UserWarning, match=f"renamed to `{expected_name}`"):
        problem = petab1to2(yaml_path)

    measurement_df = problem.measurement_df
    assert measurement_df[expected_name].tolist() == [
        "dose_response",
        "time_course",
    ]
    assert (
        measurement_df[v2.C.EXPERIMENT_ID].tolist() == ["experiment____c0"] * 2
    )
    if clash:
        taken = measurement_df[f"{v2.C.EXPERIMENT_ID}_v1"]
        assert pd.isna(taken.iloc[0])
        assert taken.iloc[1] == "taken"


try:
    import benchmark_models_petab

    parametrize_or_skip = pytest.mark.parametrize(
        "problem_id", benchmark_models_petab.MODELS
    )
except ImportError:
    parametrize_or_skip = pytest.mark.skip(
        reason="benchmark_models_petab not installed"
    )


@pytest.mark.filterwarnings(
    "ignore:.*Initialisation priors in parameter table are not supported.*:"
    "UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*does not support log10-transformed observables.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*which is reserved in PEtab v2.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:Log-scaled parameters .* were not reparametrized.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:Parameter .* has no non-negative nominal value.*:UserWarning"
)
@pytest.mark.parametrize("reparametrize", ["none", "all"])
@parametrize_or_skip
def test_benchmark_collection(problem_id, reparametrize):
    """Test that we can upgrade all benchmark collection models."""
    logging.basicConfig(level=logging.DEBUG)

    if problem_id == "Froehlich_CellSystems2018":
        # this is mostly about 6M sympifications in the condition table
        pytest.skip("Too slow. Re-enable once we are faster.")

    yaml_path = benchmark_models_petab.get_problem_yaml_path(problem_id)
    try:
        problem = petab1to2(yaml_path, reparametrize=reparametrize)
    except NotImplementedError as e:
        if "could not be reparametrized" in str(e):
            raise
        pytest.skip(str(e))
    assert isinstance(problem, Problem)
    assert len(problem.measurements)
