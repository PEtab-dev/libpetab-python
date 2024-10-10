from pathlib import Path

import benchmark_models_petab
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import petab.v1

# from pypesto.petab import PetabImporter
from petab.v1.priors import priors_to_measurements


@pytest.mark.parametrize(
    "problem_id", ["Schwen_PONE2014", "Isensee_JCB2018", "Raimundez_PCB2020"]
)
def test_priors_to_measurements(problem_id):
    """Test the conversion of priors to measurements."""
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

    petab_problem_measurements = priors_to_measurements(petab_problem_priors)
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

    # verity that the objective function value is the same

    # load/construct the simulation results
    simulation_df_priors = petab.v1.get_simulation_df(
        Path(
            benchmark_models_petab.MODELS_DIR,
            problem_id,
            f"simulatedData_{problem_id}.tsv",
        )
    )
    simulation_df_measurements = pd.concat(
        [
            petab_problem_measurements.measurement_df.rename(
                columns={petab.v1.MEASUREMENT: petab.v1.SIMULATION}
            )[
                petab_problem_measurements.measurement_df[
                    petab.v1.C.OBSERVABLE_ID
                ].str.startswith("prior_")
            ],
            simulation_df_priors,
        ]
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
        (petab_problem_priors.parameter_df[petab.v1.ESTIMATE] == 1)
        & petab_problem_priors.parameter_df[
            petab.v1.OBJECTIVE_PRIOR_TYPE
        ].notna()
    ]
    priors = petab.v1.get_priors_from_df(
        petab_problem_priors.parameter_df,
        mode="objective",
        parameter_ids=parameter_ids,
    )
    prior_contrib = 0
    for parameter_id, prior in zip(parameter_ids, priors, strict=True):
        prior_type, prior_pars, par_scale, par_bounds = prior
        if prior_type == petab.v1.PARAMETER_SCALE_NORMAL:
            prior_contrib += norm.logpdf(
                petab_problem_priors.x_nominal_free_scaled[
                    petab_problem_priors.x_free_ids.index(parameter_id)
                ],
                loc=prior_pars[0],
                scale=prior_pars[1],
            )
        else:
            # enable other models, once libpetab has proper support for
            #  evaluating the prior contribution. until then, two test
            #  problems should suffice
            assert problem_id == "Raimundez_PCB2020"
            pytest.skip(f"Prior type {prior_type} not implemented")

    assert np.isclose(
        llh_priors + prior_contrib, llh_measurements, rtol=1e-3, atol=1e-16
    ), (llh_priors + prior_contrib, llh_measurements)
    # check that the tolerance is not too high
    assert np.abs(prior_contrib) > 1e-3 * np.abs(llh_priors)
