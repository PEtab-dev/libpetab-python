"""Functionality for simplifying PEtab problems"""
from math import nan

import pandas as pd

import petab
from . import Problem
from .C import *  # noqa: F403
from .lint import lint_problem

__all__ = [
    "remove_nan_measurements",
    "remove_unused_observables",
    "remove_unused_conditions",
    "simplify_problem",
    "condition_parameters_to_parameter_table",
]


def remove_nan_measurements(problem: Problem):
    """Drop any measurements that are NaN"""
    problem.measurement_df = problem.measurement_df[
        ~problem.measurement_df[MEASUREMENT].isna()
    ]
    problem.measurement_df.reset_index(inplace=True, drop=True)


def remove_unused_observables(problem: Problem):
    """Remove observables that have no measurements"""
    measured_observables = set(problem.measurement_df[OBSERVABLE_ID].unique())
    problem.observable_df = problem.observable_df[
        problem.observable_df.index.isin(measured_observables)]


def remove_unused_conditions(problem: Problem):
    """Remove conditions that have no measurements"""
    measured_conditions = \
        set(problem.measurement_df[SIMULATION_CONDITION_ID].unique())
    if PREEQUILIBRATION_CONDITION_ID in problem.measurement_df:
        measured_conditions |= \
            set(problem.measurement_df[PREEQUILIBRATION_CONDITION_ID].unique())

    problem.condition_df = problem.condition_df[
        problem.condition_df.index.isin(measured_conditions)]


def simplify_problem(problem: Problem):
    if lint_problem(problem):
        raise ValueError("Invalid PEtab problem supplied.")

    remove_unused_observables(problem)
    remove_unused_conditions(problem)
    condition_parameters_to_parameter_table(problem)

    if lint_problem(problem):
        raise AssertionError("Invalid PEtab problem generated.")


def condition_parameters_to_parameter_table(problem: Problem):
    """Move parameters from the condition table to the parameters table, if
    the same parameter value is used for all conditions."""
    if problem.condition_df is None or problem.condition_df.empty \
            or problem.model is None:
        return

    replacements = {}
    for parameter_id in problem.condition_df:
        if parameter_id == CONDITION_NAME:
            continue

        if problem.model.is_state_variable(parameter_id):
            # initial states can't go the parameters table
            continue

        series = problem.condition_df[parameter_id]
        value = petab.to_float_if_float(series[0])

        # same value for all conditions and no parametric overrides (str)?
        if isinstance(value, float) and len(series.unique()) == 1:
            replacements[parameter_id] = series[0]

    if not replacements:
        return

    rows = [
        {
            PARAMETER_ID: parameter_id,
            PARAMETER_SCALE: LIN,
            LOWER_BOUND: nan,
            UPPER_BOUND: nan,
            NOMINAL_VALUE: value,
            ESTIMATE: 0
        }
        for parameter_id, value in replacements.items()
    ]
    rows = pd.DataFrame(rows)
    rows.set_index(PARAMETER_ID, inplace=True)

    if problem.parameter_df is None:
        problem.parameter_df = rows
    else:
        problem.parameter_df = pd.concat([problem.parameter_df, rows])

    problem.condition_df = \
        problem.condition_df.drop(columns=replacements.keys())
