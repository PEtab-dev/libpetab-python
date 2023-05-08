"""Functions related to mapping parameter from model to parameter estimation
problem"""

import logging
import numbers
import os
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Literal

import libsbml
import numpy as np
import pandas as pd

from . import ENV_NUM_THREADS, core, lint, measurements, observables, \
    parameters
from .C import *  # noqa: F403
from .models import Model
from .mapping import resolve_mapping

logger = logging.getLogger(__name__)
__all__ = ['get_optimization_to_simulation_parameter_mapping',
           'get_parameter_mapping_for_condition',
           'handle_missing_overrides',
           'merge_preeq_and_sim_pars',
           'merge_preeq_and_sim_pars_condition',
           'ParMappingDict', 'ParMappingDictTuple',
           'ScaleMappingDict', 'ScaleMappingDictTuple',
           'ParMappingDictQuadruple']


# Parameter mapping for condition
ParMappingDict = Dict[str, Union[str, numbers.Number]]
# Parameter mapping for combination of preequilibration and simulation
# condition
ParMappingDictTuple = Tuple[ParMappingDict, ParMappingDict]
# Same for scale mapping
ScaleMappingDict = Dict[str, str]
ScaleMappingDictTuple = Tuple[ScaleMappingDict, ScaleMappingDict]
# Parameter mapping for combination of preequilibration and simulation
# conditions, for parameter and scale mapping
ParMappingDictQuadruple = Tuple[ParMappingDict, ParMappingDict,
                                ScaleMappingDict, ScaleMappingDict]


def get_optimization_to_simulation_parameter_mapping(
        condition_df: pd.DataFrame,
        measurement_df: pd.DataFrame,
        parameter_df: Optional[pd.DataFrame] = None,
        observable_df: Optional[pd.DataFrame] = None,
        mapping_df: Optional[pd.DataFrame] = None,
        sbml_model: libsbml.Model = None,
        simulation_conditions: Optional[pd.DataFrame] = None,
        warn_unmapped: Optional[bool] = True,
        scaled_parameters: bool = False,
        fill_fixed_parameters: bool = True,
        allow_timepoint_specific_numeric_noise_parameters: bool = False,
        model: Model = None,
) -> List[ParMappingDictQuadruple]:
    """
    Create list of mapping dicts from PEtab-problem to model parameters.

    Mapping can be performed in parallel. The number of threads is controlled
    by the environment variable with the name of
    :py:data:`petab.ENV_NUM_THREADS`.

    Parameters:
        condition_df, measurement_df, parameter_df, observable_df:
            The dataframes in the PEtab format.
        sbml_model:
            The SBML model (deprecated)
        model:
            The model.
        simulation_conditions:
            Table of simulation conditions as created by
            ``petab.get_simulation_conditions``.
        warn_unmapped:
            If ``True``, log warning regarding unmapped parameters
        scaled_parameters:
            Whether parameter values should be scaled.
        fill_fixed_parameters:
            Whether to fill in nominal values for fixed parameters
            (estimate=0 in parameters table).
        allow_timepoint_specific_numeric_noise_parameters:
            Mapping of timepoint-specific parameters overrides is generally
            not supported. If this option is set to True, this function will
            not fail in case of timepoint-specific fixed noise parameters,
            if the noise formula consists only of one single parameter.
            It is expected that the respective mapping is performed elsewhere.
            The value mapped to the respective parameter here is undefined.

    Returns:
        Parameter value and parameter scale mapping for all conditions.

        The length of the returned array is the number of unique combinations
        of ``simulationConditionId`` s and ``preequilibrationConditionId`` s
        from the measurement table. Each entry is a tuple of four dicts of
        length equal to the number of model parameters.
        The first two dicts map simulation parameter IDs to optimization
        parameter IDs or values (where values are fixed) for preequilibration
        and simulation condition, respectively.
        The last two dicts map simulation parameter IDs to the parameter scale
        of the respective parameter, again for preequilibration and simulation
        condition.
        If no preequilibration condition is defined, the respective dicts will
        be empty. ``NaN`` is used where no mapping exists.
    """
    if sbml_model:
        warnings.warn("Passing a model via the `sbml_model` argument is "
                      "deprecated, use `model=petab.models.sbml_model."
                      "SbmlModel(...)` instead.", DeprecationWarning,
                      stacklevel=2)
        from petab.models.sbml_model import SbmlModel
        if model:
            raise ValueError("Arguments `model` and `sbml_model` are "
                             "mutually exclusive.")
        model = SbmlModel(sbml_model=sbml_model)

    # Ensure inputs are okay
    _perform_mapping_checks(
        measurement_df,
        allow_timepoint_specific_numeric_noise_parameters=  # noqa: E251,E501
        allow_timepoint_specific_numeric_noise_parameters)

    if simulation_conditions is None:
        simulation_conditions = measurements.get_simulation_conditions(
            measurement_df)

    simulation_parameters = dict(model.get_free_parameter_ids_with_values())
    # Add output parameters that are not already defined in the model
    if observable_df is not None:
        output_parameters = observables.get_output_parameters(
            observable_df=observable_df, model=model, mapping_df=mapping_df
        )
        for par_id in output_parameters:
            simulation_parameters[par_id] = np.nan

    num_threads = int(os.environ.get(ENV_NUM_THREADS, 1))

    # If sequential execution is requested, let's not create any
    # thread-allocation overhead
    if num_threads == 1:
        mapping = map(
            _map_condition,
            _map_condition_arg_packer(
                simulation_conditions, measurement_df, condition_df,
                parameter_df, mapping_df,
                model, simulation_parameters, warn_unmapped, scaled_parameters,
                fill_fixed_parameters,
                allow_timepoint_specific_numeric_noise_parameters))
        return list(mapping)

    # Run multi-threaded
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        mapping = executor.map(
            _map_condition,
            _map_condition_arg_packer(
                simulation_conditions, measurement_df, condition_df,
                parameter_df, mapping_df, model, simulation_parameters,
                warn_unmapped, scaled_parameters, fill_fixed_parameters,
                allow_timepoint_specific_numeric_noise_parameters))
    return list(mapping)


def _map_condition_arg_packer(
        simulation_conditions,
        measurement_df,
        condition_df,
        parameter_df,
        mapping_df,
        model,
        simulation_parameters,
        warn_unmapped,
        scaled_parameters,
        fill_fixed_parameters,
        allow_timepoint_specific_numeric_noise_parameters
):
    """Helper function to pack extra arguments for _map_condition"""
    for _, condition in simulation_conditions.iterrows():
        yield (
            condition, measurement_df, condition_df, parameter_df, mapping_df,
            model, simulation_parameters, warn_unmapped, scaled_parameters,
            fill_fixed_parameters,
            allow_timepoint_specific_numeric_noise_parameters
        )


def _map_condition(packed_args):
    """Helper function for parallel condition mapping.

    For arguments see
    :py:func:`get_optimization_to_simulation_parameter_mapping`.
    """

    (condition, measurement_df, condition_df, parameter_df, mapping_df, model,
     simulation_parameters, warn_unmapped, scaled_parameters,
     fill_fixed_parameters,
     allow_timepoint_specific_numeric_noise_parameters) = packed_args

    cur_measurement_df = None
    # Get the condition specific measurements for the current condition, but
    # only if relevant for parameter mapping
    if (OBSERVABLE_PARAMETERS in measurement_df
        and measurement_df[OBSERVABLE_PARAMETERS].notna().any()) \
        or (NOISE_PARAMETERS in measurement_df
            and measurement_df[NOISE_PARAMETERS].notna().any()):
        cur_measurement_df = measurements.get_rows_for_condition(
            measurement_df, condition)

    if PREEQUILIBRATION_CONDITION_ID not in condition \
            or not isinstance(condition[PREEQUILIBRATION_CONDITION_ID], str) \
            or not condition[PREEQUILIBRATION_CONDITION_ID]:
        par_map_preeq = {}
        scale_map_preeq = {}
    else:
        par_map_preeq, scale_map_preeq = get_parameter_mapping_for_condition(
            condition_id=condition[PREEQUILIBRATION_CONDITION_ID],
            is_preeq=True,
            cur_measurement_df=cur_measurement_df,
            model=model,
            condition_df=condition_df,
            parameter_df=parameter_df,
            mapping_df=mapping_df,
            simulation_parameters=simulation_parameters,
            warn_unmapped=warn_unmapped,
            scaled_parameters=scaled_parameters,
            fill_fixed_parameters=fill_fixed_parameters,
            allow_timepoint_specific_numeric_noise_parameters=  # noqa: E251,E501
            allow_timepoint_specific_numeric_noise_parameters
        )

    par_map_sim, scale_map_sim = get_parameter_mapping_for_condition(
        condition_id=condition[SIMULATION_CONDITION_ID],
        is_preeq=False,
        cur_measurement_df=cur_measurement_df,
        model=model,
        condition_df=condition_df,
        parameter_df=parameter_df,
        mapping_df=mapping_df,
        simulation_parameters=simulation_parameters,
        warn_unmapped=warn_unmapped,
        scaled_parameters=scaled_parameters,
        fill_fixed_parameters=fill_fixed_parameters,
        allow_timepoint_specific_numeric_noise_parameters=  # noqa: E251,E501
        allow_timepoint_specific_numeric_noise_parameters
    )

    return par_map_preeq, par_map_sim, scale_map_preeq, scale_map_sim


def get_parameter_mapping_for_condition(
        condition_id: str,
        is_preeq: bool,
        cur_measurement_df: Optional[pd.DataFrame],
        sbml_model: libsbml.Model = None,
        condition_df: pd.DataFrame = None,
        parameter_df: pd.DataFrame = None,
        mapping_df: Optional[pd.DataFrame] = None,
        simulation_parameters: Optional[Dict[str, str]] = None,
        warn_unmapped: bool = True,
        scaled_parameters: bool = False,
        fill_fixed_parameters: bool = True,
        allow_timepoint_specific_numeric_noise_parameters: bool = False,
        model: Model = None,
) -> Tuple[ParMappingDict, ScaleMappingDict]:
    """
    Create dictionary of parameter value and parameter scale mappings from
    PEtab-problem to SBML parameters for the given condition.

    Parameters:
        condition_id:
            Condition ID for which to perform mapping
        is_preeq:
            If ``True``, output parameters will not be mapped
        cur_measurement_df:
            Measurement sub-table for current condition, can be ``None`` if
            not relevant for parameter mapping
        condition_df:
            PEtab condition DataFrame
        parameter_df:
            PEtab parameter DataFrame
        mapping_df:
            PEtab mapping DataFrame
        sbml_model:
            The SBML model (deprecated)
        model:
            The model.
        simulation_parameters:
            Model simulation parameter IDs mapped to parameter values (output
            of ``petab.sbml.get_model_parameters(.., with_values=True)``).
            Optional, saves time if precomputed.
        warn_unmapped:
            If ``True``, log warning regarding unmapped parameters
        scaled_parameters:
            Whether parameter values should be scaled.
        fill_fixed_parameters:
            Whether to fill in nominal values for fixed parameters
            (estimate=0 in parameters table).
        allow_timepoint_specific_numeric_noise_parameters:
            Mapping of timepoint-specific parameters overrides is generally
            not supported. If this option is set to True, this function will
            not fail in case of timepoint-specific fixed noise parameters,
            if the noise formula consists only of one single parameter.
            It is expected that the respective mapping is performed elsewhere.
            The value mapped to the respective parameter here is undefined.

    Returns:
        Tuple of two dictionaries. First dictionary mapping model parameter IDs
        to mapped parameters IDs to be estimated or to filled-in values in case
        of non-estimated parameters.
        Second dictionary mapping model parameter IDs to their scale.
        ``NaN`` is used where no mapping exists.
    """
    if sbml_model:
        warnings.warn("Passing a model via the `sbml_model` argument is "
                      "deprecated, use `model=petab.models.sbml_model."
                      "SbmlModel(...)` instead.", DeprecationWarning,
                      stacklevel=2)
        from petab.models.sbml_model import SbmlModel
        if model:
            raise ValueError("Arguments `model` and `sbml_model` are "
                             "mutually exclusive.")
        model = SbmlModel(sbml_model=sbml_model)

    if cur_measurement_df is not None:
        _perform_mapping_checks(
            cur_measurement_df,
            allow_timepoint_specific_numeric_noise_parameters=  # noqa: E251,E501
            allow_timepoint_specific_numeric_noise_parameters)

    if simulation_parameters is None:
        simulation_parameters = dict(
            model.get_free_parameter_ids_with_values())

    # NOTE: order matters here - the former is overwritten by the latter:
    #  model < condition table < measurement < table parameter table

    # initialize mapping dicts
    # for the case of matching simulation and optimization parameter vector
    par_mapping = simulation_parameters.copy()
    scale_mapping = {par_id: LIN for par_id in par_mapping.keys()}
    _output_parameters_to_nan(par_mapping)

    # not strictly necessary for preequilibration, be we do it to have
    # same length of parameter vectors
    if cur_measurement_df is not None:
        _apply_output_parameter_overrides(par_mapping, cur_measurement_df)

    if not is_preeq:
        handle_missing_overrides(par_mapping, warn=warn_unmapped)

    _apply_condition_parameters(par_mapping, scale_mapping, condition_id,
                                condition_df, model, mapping_df)
    _apply_parameter_table(par_mapping, scale_mapping,
                           parameter_df, scaled_parameters,
                           fill_fixed_parameters)

    return par_mapping, scale_mapping


def _output_parameters_to_nan(mapping: ParMappingDict) -> None:
    """Set output parameters in mapping dictionary to nan"""
    rex = re.compile("^(noise|observable)Parameter[0-9]+_")
    for key in mapping.keys():
        try:
            matches = rex.match(key)
        except TypeError:
            continue

        if matches:
            mapping[key] = np.nan


def _apply_output_parameter_overrides(
        mapping: ParMappingDict,
        cur_measurement_df: pd.DataFrame
) -> None:
    """
    Apply output parameter overrides to the parameter mapping dict for a given
    condition as defined in the measurement table (``observableParameter``,
    ``noiseParameters``).

    Arguments:
        mapping: parameter mapping dict as obtained from
            :py:func:`get_parameter_mapping_for_condition`.
        cur_measurement_df:
            Subset of the measurement table for the current condition
    """
    for _, row in cur_measurement_df.iterrows():
        # we trust that the number of overrides matches (see above)
        overrides = measurements.split_parameter_replacement_list(
            row.get(OBSERVABLE_PARAMETERS, None))
        _apply_overrides_for_observable(mapping, row[OBSERVABLE_ID],
                                        'observable', overrides)

        overrides = measurements.split_parameter_replacement_list(
            row.get(NOISE_PARAMETERS, None))
        _apply_overrides_for_observable(mapping, row[OBSERVABLE_ID], 'noise',
                                        overrides)


def _apply_overrides_for_observable(
        mapping: ParMappingDict,
        observable_id: str,
        override_type: Literal['observable', 'noise'],
        overrides: List[str],
) -> None:
    """
    Apply parameter-overrides for observables and noises to mapping
    matrix.

    Arguments:
        mapping: mapping dict to which to apply overrides
        observable_id: observable ID
        override_type: ``'observable'`` or ``'noise'``
        overrides: list of overrides for noise or observable parameters
    """
    for i, override in enumerate(overrides):
        overridee_id = f'{override_type}Parameter{i+1}_{observable_id}'
        mapping[overridee_id] = override


def _apply_condition_parameters(
        par_mapping: ParMappingDict,
        scale_mapping: ScaleMappingDict,
        condition_id: str,
        condition_df: pd.DataFrame,
        model: Model,
        mapping_df: Optional[pd.DataFrame] = None,
) -> None:
    """Replace parameter IDs in parameter mapping dictionary by condition
    table parameter values (in-place).

    Arguments:
        par_mapping: see :py:func:`get_parameter_mapping_for_condition`
        condition_id: ID of condition to work on
        condition_df: PEtab condition table
    """
    for overridee_id in condition_df.columns:
        if overridee_id == CONDITION_NAME:
            continue

        overridee_id = resolve_mapping(mapping_df, overridee_id)

        # Species, compartments, and rule targets are handled elsewhere
        if model.is_state_variable(overridee_id):
            continue

        par_mapping[overridee_id] = core.to_float_if_float(
            condition_df.loc[condition_id, overridee_id])

        if isinstance(par_mapping[overridee_id], numbers.Number) \
                and np.isnan(par_mapping[overridee_id]):
            # NaN in the condition table for an entity without time derivative
            #  indicates that the model value should be used
            try:
                par_mapping[overridee_id] = \
                    model.get_parameter_value(overridee_id)
            except ValueError as e:
                raise NotImplementedError(
                    "Not sure how to handle NaN in condition table for "
                    f"{overridee_id}."
                ) from e

        scale_mapping[overridee_id] = LIN


def _apply_parameter_table(
        par_mapping: ParMappingDict,
        scale_mapping: ScaleMappingDict,
        parameter_df: Optional[pd.DataFrame] = None,
        scaled_parameters: bool = False,
        fill_fixed_parameters: bool = True,
) -> None:
    """Replace parameters from parameter table in mapping list for a given
    condition and set the corresponding scale.

    Replace non-estimated parameters by ``nominalValues``
    (un-scaled / lin-scaled), replace estimated parameters by the respective
    ID.

    Arguments:
        par_mapping:
            mapping dict obtained from
            :py:func:`get_parameter_mapping_for_condition`
        parameter_df:
            PEtab parameter table
    """

    if parameter_df is None:
        return

    for row in parameter_df.itertuples():
        if row.Index not in par_mapping:
            # The current parameter is not required for this condition
            continue

        scale = getattr(row, PARAMETER_SCALE, LIN)
        scale_mapping[row.Index] = scale
        if fill_fixed_parameters and getattr(row, ESTIMATE) == 0:
            val = getattr(row, NOMINAL_VALUE)
            if scaled_parameters:
                val = parameters.scale(val, scale)
            else:
                scale_mapping[row.Index] = LIN
            par_mapping[row.Index] = val
        else:
            par_mapping[row.Index] = row.Index

    # Replace any leftover mapped parameter coming from condition table
    for problem_par, sim_par in par_mapping.items():
        # string indicates unmapped
        if not isinstance(sim_par, str):
            continue

        try:
            # the overridee is a model parameter
            par_mapping[problem_par] = par_mapping[sim_par]
            scale_mapping[problem_par] = scale_mapping[sim_par]
        except KeyError:
            if parameter_df is None:
                raise

            # or the overridee is only defined in the parameter table
            scale = parameter_df.loc[sim_par, PARAMETER_SCALE] \
                if PARAMETER_SCALE in parameter_df else LIN

            if fill_fixed_parameters \
                    and ESTIMATE in parameter_df \
                    and parameter_df.loc[sim_par, ESTIMATE] == 0:
                val = parameter_df.loc[sim_par, NOMINAL_VALUE]
                if scaled_parameters:
                    val = parameters.scale(val, scale)
                else:
                    scale = LIN
                par_mapping[problem_par] = val

            scale_mapping[problem_par] = scale


def _perform_mapping_checks(
        measurement_df: pd.DataFrame,
        allow_timepoint_specific_numeric_noise_parameters: bool = False
) -> None:
    """Check for PEtab features which we can't account for during parameter
    mapping."""

    if lint.measurement_table_has_timepoint_specific_mappings(
            measurement_df,
            allow_scalar_numeric_noise_parameters=  # noqa: E251,E501
            allow_timepoint_specific_numeric_noise_parameters):
        # we could allow that for floats, since they don't matter in this
        # function and would be simply ignored
        raise ValueError(
            "Timepoint-specific parameter overrides currently unsupported.")


def handle_missing_overrides(
        mapping_par_opt_to_par_sim: ParMappingDict,
        warn: bool = True,
        condition_id: str = None,
) -> None:
    """
    Find all observable parameters and noise parameters that were not mapped
    and set their mapping to np.nan.

    Assumes that parameters matching the regular expression
    ``(noise|observable)Parameter[0-9]+_`` were all supposed to be overwritten.

    Parameters:
        mapping_par_opt_to_par_sim:
            Output of :py:func:`get_parameter_mapping_for_condition`
        warn:
            If True, log warning regarding unmapped parameters
        condition_id:
            Optional condition ID for more informative output
    """
    _missed_vals = []
    rex = re.compile("^(noise|observable)Parameter[0-9]+_")
    for key, val in mapping_par_opt_to_par_sim.items():
        try:
            matches = rex.match(val)
        except TypeError:
            continue

        if matches:
            mapping_par_opt_to_par_sim[key] = np.nan
            _missed_vals.append(key)

    if _missed_vals and warn:
        logger.warning(f"Could not map the following overrides for condition "
                       f"{condition_id}: "
                       f"{_missed_vals}. Usually, this is just due to missing "
                       f"data points.")


def merge_preeq_and_sim_pars_condition(
        condition_map_preeq: ParMappingDict,
        condition_map_sim: ParMappingDict,
        condition_scale_map_preeq: ScaleMappingDict,
        condition_scale_map_sim: ScaleMappingDict,
        condition: Any,
) -> None:
    """Merge preequilibration and simulation parameters and scales for a single
    condition while checking for compatibility.

    This function is meant for the case where we cannot have different
    parameters (and scales) for preequilibration and simulation. Therefore,
    merge both and ensure matching scales and parameters.
    ``condition_map_sim`` and ``condition_scale_map_sim`` will be modified in
    place.

    Arguments:
        condition_map_preeq, condition_map_sim:
            Parameter mapping as obtained from
            :py:func:`get_parameter_mapping_for_condition`
        condition_scale_map_preeq, condition_scale_map_sim:
            Parameter scale mapping as obtained from
            :py:func:`get_parameter_mapping_for_condition`
        condition: Condition identifier for more informative error messages
    """
    if not condition_map_preeq:
        # nothing to do
        return

    all_par_ids = set(condition_map_sim.keys()) \
        | set(condition_map_preeq.keys())

    for par_id in all_par_ids:
        if par_id not in condition_map_preeq:
            # nothing to do
            continue

        if par_id not in condition_map_sim:
            # unmapped for simulation -> just use preeq values
            condition_map_sim[par_id] = condition_map_preeq[par_id]
            condition_scale_map_sim[par_id] = condition_scale_map_preeq[par_id]
            continue

        # present in both
        par_preeq = condition_map_preeq[par_id]
        par_sim = condition_map_sim[par_id]
        if par_preeq != par_sim \
                and not (core.is_empty(par_sim) and core.is_empty(par_preeq)):
            # both identical or both nan is okay
            if core.is_empty(par_sim):
                # unmapped for simulation
                condition_map_sim[par_id] = par_preeq
            elif core.is_empty(par_preeq):
                # unmapped for preeq is okay
                pass
            else:
                raise ValueError(
                    'Cannot handle different values for dynamic '
                    f'parameters: for condition {condition} '
                    f'parameter {par_id} is {par_preeq} for preeq '
                    f'and {par_sim} for simulation.')

        scale_preeq = condition_scale_map_preeq[par_id]
        scale_sim = condition_scale_map_sim[par_id]

        if scale_preeq != scale_sim:
            # both identical is okay
            if core.is_empty(par_sim):
                # unmapped for simulation
                condition_scale_map_sim[par_id] = scale_preeq
            elif core.is_empty(par_preeq):
                # unmapped for preeq is okay
                pass
            else:
                raise ValueError(
                    'Cannot handle different parameter scales '
                    f'parameters: for condition {condition} '
                    f'scale for parameter {par_id} is {scale_preeq} for preeq '
                    f'and {scale_sim} for simulation.')


def merge_preeq_and_sim_pars(
        parameter_mappings: Iterable[ParMappingDictTuple],
        scale_mappings: Iterable[ScaleMappingDictTuple]
) -> Tuple[List[ParMappingDictTuple], List[ScaleMappingDictTuple]]:
    """Merge preequilibration and simulation parameters and scales for a list
    of conditions while checking for compatibility.

    Parameters:
        parameter_mappings:
            As returned by
            :py:func:`petab.get_optimization_to_simulation_parameter_mapping`.
        scale_mappings:
            As returned by
            :py:func:`petab.get_optimization_to_simulation_parameter_mapping`.

    Returns:
        The parameter and scale simulation mappings, modified and checked.
    """
    parameter_mapping = []
    scale_mapping = []
    for ic, ((map_preeq, map_sim), (scale_map_preeq, scale_map_sim)) in \
            enumerate(zip(parameter_mappings, scale_mappings)):
        merge_preeq_and_sim_pars_condition(
            condition_map_preeq=map_preeq,
            condition_map_sim=map_sim,
            condition_scale_map_preeq=scale_map_preeq,
            condition_scale_map_sim=scale_map_sim,
            condition=ic)
        parameter_mapping.append(map_sim)
        scale_mapping.append(scale_map_sim)

    return parameter_mapping, scale_mapping
