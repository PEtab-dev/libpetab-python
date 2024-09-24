"""Functions related to prior handling."""
import copy

import numpy as np
import pandas as pd

from . import (
    ESTIMATE,
    LAPLACE,
    LIN,
    LOG,
    LOG10,
    LOG_LAPLACE,
    LOG_NORMAL,
    MEASUREMENT,
    NOISE_DISTRIBUTION,
    NOISE_FORMULA,
    NOISE_PARAMETERS,
    NORMAL,
    OBJECTIVE_PRIOR_PARAMETERS,
    OBJECTIVE_PRIOR_TYPE,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    OBSERVABLE_TRANSFORMATION,
    PARAMETER_SCALE,
    PARAMETER_SCALE_LAPLACE,
    PARAMETER_SCALE_NORMAL,
    PARAMETER_SEPARATOR,
    SIMULATION_CONDITION_ID,
    TIME,
    Problem,
)


def priors_to_measurements(problem: Problem):
    """Convert priors to measurements.

    Reformulate the given problem such that the objective priors are converted
    to measurements. This is done by adding a new observable for each prior
    and adding a corresponding measurement to the measurement table.
    The new measurement is the prior distribution itself. The resulting
    optimization problem will be equivalent to the original problem.
    This is meant to be used for tools that do not support priors.

    Arguments
    ---------
    problem:
        The problem to be converted.

    Returns
    -------
    The new problem with the priors converted to measurements.
    """
    new_problem = copy.deepcopy(problem)

    # we only need to consider parameters that are estimated
    par_df_tmp = problem.parameter_df.loc[problem.parameter_df[ESTIMATE] == 1]

    if (
        OBJECTIVE_PRIOR_TYPE not in par_df_tmp
        or par_df_tmp.get(OBJECTIVE_PRIOR_TYPE).isna().all()
        or OBJECTIVE_PRIOR_PARAMETERS not in par_df_tmp
        or par_df_tmp.get(OBJECTIVE_PRIOR_PARAMETERS).isna().all()
    ):
        # nothing to do
        return new_problem

    def scaled_observable_formula(parameter_id, parameter_scale):
        if parameter_scale == LIN:
            return parameter_id
        if parameter_scale == LOG:
            return f"log({parameter_id})"
        if parameter_scale == LOG10:
            return f"log10({parameter_id})"
        raise ValueError(f"Unknown parameter scale {parameter_scale}.")

    for i, row in par_df_tmp.iterrows():
        prior_type = row[OBJECTIVE_PRIOR_TYPE]
        parameter_scale = row.get(PARAMETER_SCALE, LIN)
        if pd.isna(prior_type):
            assert pd.isna(row[OBJECTIVE_PRIOR_PARAMETERS])
            continue

        if "uniform" in prior_type.lower():
            # for measurements, "uniform" is not supported yet
            #  if necessary, this could still be implemented by adding another
            #  observable/measurement that will produce a constant objective
            #  offset
            raise NotImplementedError("Uniform priors are not supported.")

        parameter_id = row.name
        prior_parameters = tuple(
            map(
                float,
                row[OBJECTIVE_PRIOR_PARAMETERS].split(PARAMETER_SEPARATOR),
            )
        )
        assert len(prior_parameters) == 2

        # create new observable
        new_obs_id = f"prior_{i}"
        if new_obs_id in new_problem.observable_df.index:
            raise ValueError(
                f"Observable ID {new_obs_id}, which is to be "
                "created, already exists."
            )
        new_observable = {
            OBSERVABLE_FORMULA: scaled_observable_formula(
                parameter_id,
                parameter_scale if "parameterScale" in prior_type else LIN,
            ),
            NOISE_FORMULA: f"noiseParameter1_{new_obs_id}",
        }
        if prior_type in (LOG_NORMAL, LOG_LAPLACE):
            new_observable[OBSERVABLE_TRANSFORMATION] = LOG
        elif OBSERVABLE_TRANSFORMATION in new_problem.observable_df:
            # only set default if the column is already present
            new_observable[OBSERVABLE_TRANSFORMATION] = LIN

        if prior_type in (NORMAL, PARAMETER_SCALE_NORMAL, LOG_NORMAL):
            new_observable[NOISE_DISTRIBUTION] = NORMAL
        elif prior_type in (LAPLACE, PARAMETER_SCALE_LAPLACE, LOG_LAPLACE):
            new_observable[NOISE_DISTRIBUTION] = LAPLACE
        else:
            raise NotImplementedError(
                f"Objective prior type {prior_type} is not implemented."
            )

        new_problem.observable_df = pd.concat(
            [
                pd.DataFrame([new_observable], index=[new_obs_id]),
                new_problem.observable_df,
            ]
        )

        # add measurement
        # we can just use any condition and time point since the parameter
        # value is constant
        sim_cond_id = new_problem.condition_df.index[0]
        new_problem.measurement_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        OBSERVABLE_ID: new_obs_id,
                        TIME: 0,
                        MEASUREMENT: prior_parameters[0],
                        NOISE_PARAMETERS: prior_parameters[1],
                        SIMULATION_CONDITION_ID: sim_cond_id,
                    },
                    index=[0],
                ),
                new_problem.measurement_df,
            ]
        )

        # remove prior from parameter table
        new_problem.parameter_df.loc[
            parameter_id, OBJECTIVE_PRIOR_TYPE
        ] = np.nan
        new_problem.parameter_df.loc[
            parameter_id, OBJECTIVE_PRIOR_PARAMETERS
        ] = np.nan

    return new_problem
