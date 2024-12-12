from copy import deepcopy
from itertools import product
from pathlib import Path

import benchmark_models_petab
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstest

import petab.v1
from petab.v1 import (
    ESTIMATE,
    MEASUREMENT,
    OBJECTIVE_PRIOR_TYPE,
    OBSERVABLE_ID,
    SIMULATION,
    C,
    get_simulation_conditions,
    get_simulation_df,
)
from petab.v1.priors import Prior, priors_to_measurements


@pytest.mark.parametrize(
    "problem_id", ["Schwen_PONE2014", "Isensee_JCB2018", "Raimundez_PCB2020"]
)
def test_priors_to_measurements(problem_id):
    """Test the conversion of priors to measurements."""
    # setup
    petab_problem_priors: petab.v1.Problem = (
        benchmark_models_petab.get_problem(problem_id)
    )
    petab_problem_priors.visualization_df = None
    assert petab.v1.lint_problem(petab_problem_priors) is False
    if problem_id == "Isensee_JCB2018":
        # required to match the stored simulation results below
        petab.v1.flatten_timepoint_specific_output_overrides(
            petab_problem_priors
        )
        assert petab.v1.lint_problem(petab_problem_priors) is False

    original_problem = deepcopy(petab_problem_priors)
    # All priors in this test case are defined on parameter scale, hence
    # the dummy measurements will take the scaled nominal values.
    x_scaled_dict = dict(
        zip(
            original_problem.x_free_ids,
            original_problem.x_nominal_free_scaled,
            strict=True,
        )
    )
    x_unscaled_dict = dict(
        zip(
            original_problem.x_free_ids,
            original_problem.x_nominal_free,
            strict=True,
        )
    )

    # convert priors to measurements
    petab_problem_measurements = priors_to_measurements(petab_problem_priors)

    # check that the original problem is not modified
    for attr in [
        "condition_df",
        "parameter_df",
        "observable_df",
        "measurement_df",
    ]:
        assert (
            diff := getattr(petab_problem_priors, attr).compare(
                getattr(original_problem, attr)
            )
        ).empty, diff

    # check that measurements and observables were added
    assert petab.v1.lint_problem(petab_problem_measurements) is False
    assert (
        petab_problem_measurements.parameter_df.shape[0]
        == petab_problem_priors.parameter_df.shape[0]
    )
    assert (
        petab_problem_measurements.observable_df.shape[0]
        > petab_problem_priors.observable_df.shape[0]
    )
    assert (
        petab_problem_measurements.measurement_df.shape[0]
        > petab_problem_priors.measurement_df.shape[0]
    )

    # ensure we didn't introduce any new conditions
    assert len(
        get_simulation_conditions(petab_problem_measurements.measurement_df)
    ) == len(get_simulation_conditions(petab_problem_priors.measurement_df))

    # verify that the objective function value is the same

    # load/construct the simulation results
    simulation_df_priors = get_simulation_df(
        Path(
            benchmark_models_petab.MODELS_DIR,
            problem_id,
            f"simulatedData_{problem_id}.tsv",
        )
    )
    # for the prior observables, we need to "simulate" the model with the
    #  nominal parameter values
    simulated_prior_observables = (
        petab_problem_measurements.measurement_df.rename(
            columns={MEASUREMENT: SIMULATION}
        )[
            petab_problem_measurements.measurement_df[
                OBSERVABLE_ID
            ].str.startswith("prior_")
        ]
    )

    def apply_parameter_values(row):
        # apply the parameter values to the observable formula for the prior
        if row[OBSERVABLE_ID].startswith("prior_"):
            parameter_id = row[OBSERVABLE_ID].removeprefix("prior_")
            if original_problem.parameter_df.loc[
                parameter_id, OBJECTIVE_PRIOR_TYPE
            ].startswith("parameterScale"):
                row[SIMULATION] = x_scaled_dict[parameter_id]
            else:
                row[SIMULATION] = x_unscaled_dict[parameter_id]
        return row

    simulated_prior_observables = simulated_prior_observables.apply(
        apply_parameter_values, axis=1
    )
    simulation_df_measurements = pd.concat(
        [simulation_df_priors, simulated_prior_observables]
    )

    llh_priors = petab.v1.calculate_llh_for_table(
        petab_problem_priors.measurement_df,
        simulation_df_priors,
        petab_problem_priors.observable_df,
        petab_problem_priors.parameter_df,
    )
    llh_measurements = petab.v1.calculate_llh_for_table(
        petab_problem_measurements.measurement_df,
        simulation_df_measurements,
        petab_problem_measurements.observable_df,
        petab_problem_measurements.parameter_df,
    )

    # get prior objective function contribution
    parameter_ids = petab_problem_priors.parameter_df.index.values[
        (petab_problem_priors.parameter_df[ESTIMATE] == 1)
        & petab_problem_priors.parameter_df[OBJECTIVE_PRIOR_TYPE].notna()
    ]
    priors = [
        Prior.from_par_dict(
            petab_problem_priors.parameter_df.loc[par_id],
            type_="objective",
            bounds_truncate=False,
        )
        for par_id in parameter_ids
    ]
    prior_contrib = 0
    for parameter_id, prior in zip(parameter_ids, priors, strict=True):
        prior_contrib -= prior.neglogprior(x_scaled_dict[parameter_id])

    assert np.isclose(
        llh_priors + prior_contrib, llh_measurements, rtol=1e-8, atol=1e-16
    ), (llh_priors + prior_contrib, llh_measurements)
    # check that the tolerance is not too high
    assert np.abs(prior_contrib) > 1e-8 * np.abs(llh_priors)


cases = list(
    product(
        [
            (C.NORMAL, (10, 1)),
            (C.LOG_NORMAL, (2, 1)),
            (C.UNIFORM, (1, 2)),
            (C.LAPLACE, (20, 2)),
            (C.LOG_LAPLACE, (1, 0.5)),
            (C.PARAMETER_SCALE_NORMAL, (1, 1)),
            (C.PARAMETER_SCALE_LAPLACE, (1, 2)),
            (C.PARAMETER_SCALE_UNIFORM, (1, 2)),
        ],
        C.PARAMETER_SCALES,
    )
)
ids = [f"{prior_args[0]}_{transform}" for prior_args, transform in cases]


@pytest.mark.parametrize("prior_args, transform", cases, ids=ids)
def test_sample_matches_pdf(prior_args, transform):
    """Test that the sample matches the PDF."""
    np.random.seed(1)
    N_SAMPLES = 10_000
    prior = Prior(*prior_args, transformation=transform)
    sample = prior.sample(N_SAMPLES)

    # pdf -> cdf
    def cdf(x):
        return cumulative_trapezoid(prior.pdf(x), x)

    # Kolmogorov-Smirnov test to check if the sample is drawn from the CDF
    _, p = kstest(sample, cdf)

    # if p < 0.05:
    #     import matplotlib.pyplot as plt
    #     plt.hist(sample, bins=100, density=True)
    #     x = np.linspace(min(sample), max(sample), 100)
    #     plt.plot(x, distribution.pdf(x))
    #     plt.show()

    assert p > 0.05, (p, prior)
