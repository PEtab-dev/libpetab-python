"""Functions operating on the PEtab parameter table"""

import numbers
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import (
    Dict, Iterable, List, Set, Tuple, Union, Optional, Literal, Sequence
)

import libsbml
import numpy as np
import pandas as pd

from . import conditions, core, lint, measurements, observables
from .C import *  # noqa: F403
from .models import Model

__all__ = ['create_parameter_df',
           'get_optimization_parameter_scaling',
           'get_optimization_parameters',
           'get_parameter_df',
           'get_priors_from_df',
           'get_valid_parameters_for_parameter_table',
           'map_scale',
           'map_unscale',
           'normalize_parameter_df',
           'scale',
           'unscale',
           'write_parameter_df']

PARAMETER_SCALE_ARGS = Literal['', 'lin', 'log', 'log10']


def get_parameter_df(
        parameter_file: Union[str, Path, pd.DataFrame,
                              Iterable[Union[str, Path, pd.DataFrame]], None]
) -> Union[pd.DataFrame, None]:
    """
    Read the provided parameter file into a ``pandas.Dataframe``.

    Arguments:
        parameter_file: Name of the file to read from or pandas.Dataframe,
        or an Iterable.

    Returns:
        Parameter ``DataFrame``, or ``None`` if ``None`` was passed.
    """
    if parameter_file is None:
        return None
    if isinstance(parameter_file, pd.DataFrame):
        parameter_df = parameter_file
    elif isinstance(parameter_file, (str, Path)):
        parameter_df = pd.read_csv(parameter_file, sep='\t',
                                   float_precision='round_trip')
    elif isinstance(parameter_file, Iterable):
        dfs = [get_parameter_df(x) for x in parameter_file if x]

        if not dfs:
            return None

        parameter_df = pd.concat(dfs)
        # Check for contradicting parameter definitions
        _check_for_contradicting_parameter_definitions(parameter_df)

        return parameter_df

    lint.assert_no_leading_trailing_whitespace(
        parameter_df.columns.values, "parameter")

    if not isinstance(parameter_df.index, pd.RangeIndex):
        parameter_df.reset_index(inplace=True)

    try:
        parameter_df.set_index([PARAMETER_ID], inplace=True)
    except KeyError as e:
        raise KeyError(
            f"Parameter table missing mandatory field {PARAMETER_ID}.") from e
    _check_for_contradicting_parameter_definitions(parameter_df)

    return parameter_df


def _check_for_contradicting_parameter_definitions(parameter_df: pd.DataFrame):
    """
    Raises a ValueError for non-unique parameter IDs
    """
    parameter_duplicates = set(parameter_df.index.values[
                                    parameter_df.index.duplicated()])
    if parameter_duplicates:
        raise ValueError(
            f'The values of `{PARAMETER_ID}` must be unique. The '
            f'following duplicates were found:\n{parameter_duplicates}'
        )


def write_parameter_df(df: pd.DataFrame, filename: Union[str, Path]) -> None:
    """Write PEtab parameter table

    Arguments:
        df: PEtab parameter table
        filename: Destination file name
    """
    df = get_parameter_df(df)
    df.to_csv(filename, sep='\t', index=True)


def get_optimization_parameters(parameter_df: pd.DataFrame) -> List[str]:
    """
    Get list of optimization parameter IDs from parameter table.

    Arguments:
        parameter_df: PEtab parameter DataFrame

    Returns:
        List of IDs of parameters selected for optimization.
    """
    return list(parameter_df.index[parameter_df[ESTIMATE] == 1])


def get_optimization_parameter_scaling(
        parameter_df: pd.DataFrame) -> Dict[str, str]:
    """
    Get Dictionary with optimization parameter IDs mapped to parameter scaling
    strings.

    Arguments:
        parameter_df: PEtab parameter DataFrame

    Returns:
        Dictionary with optimization parameter IDs mapped to parameter scaling
        strings.
    """
    estimated_df = parameter_df.loc[parameter_df[ESTIMATE] == 1]
    return dict(zip(estimated_df.index, estimated_df[PARAMETER_SCALE]))


def create_parameter_df(
        sbml_model: Optional[libsbml.Model] = None,
        condition_df: Optional[pd.DataFrame] = None,
        observable_df: Optional[pd.DataFrame] = None,
        measurement_df: Optional[pd.DataFrame] = None,
        model: Optional[Model] = None,
        include_optional: bool = False,
        parameter_scale: str = LOG10,
        lower_bound: Iterable = None,
        upper_bound: Iterable = None,
        mapping_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create a new PEtab parameter table

    All table entries can be provided as string or list-like with length
    matching the number of parameters

    Arguments:
        sbml_model: SBML Model (deprecated, mutually exclusive with ``model``)
        model: PEtab model (mutually exclusive with ``sbml_model``)
        condition_df: PEtab condition DataFrame
        observable_df: PEtab observable DataFrame
        measurement_df: PEtab measurement DataFrame
        include_optional: By default this only returns parameters that are
            required to be present in the parameter table. If set to ``True``,
            this returns all parameters that are allowed to be present in the
            parameter table (i.e. also including parameters specified in the
            model).
        parameter_scale: parameter scaling
        lower_bound: lower bound for parameter value
        upper_bound: upper bound for parameter value
        mapping_df: PEtab mapping DataFrame

    Returns:
        The created parameter DataFrame
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
    if include_optional:
        parameter_ids = list(get_valid_parameters_for_parameter_table(
            model=model, condition_df=condition_df,
            observable_df=observable_df, measurement_df=measurement_df))
    else:
        parameter_ids = list(get_required_parameters_for_parameter_table(
            model=model, condition_df=condition_df,
            observable_df=observable_df, measurement_df=measurement_df,
            mapping_df=mapping_df
        ))

    df = pd.DataFrame(
        data={
            PARAMETER_ID: parameter_ids,
            PARAMETER_NAME: parameter_ids,
            PARAMETER_SCALE: parameter_scale,
            LOWER_BOUND: lower_bound,
            UPPER_BOUND: upper_bound,
            NOMINAL_VALUE: np.nan,
            ESTIMATE: 1,
            INITIALIZATION_PRIOR_TYPE: '',
            INITIALIZATION_PRIOR_PARAMETERS: '',
            OBJECTIVE_PRIOR_TYPE: '',
            OBJECTIVE_PRIOR_PARAMETERS: '',
        })
    df.set_index([PARAMETER_ID], inplace=True)

    # For model parameters, set nominal values as defined in the model
    for parameter_id in df.index:
        try:
            df.loc[parameter_id, NOMINAL_VALUE] = \
                model.get_parameter_value(parameter_id)
        except ValueError:
            # parameter was introduced as condition-specific override and
            # is potentially not present in the model
            pass
    return df


def get_required_parameters_for_parameter_table(
        model: Model,
        condition_df: pd.DataFrame,
        observable_df: pd.DataFrame,
        measurement_df: pd.DataFrame,
        mapping_df: pd.DataFrame = None
) -> Set[str]:
    """
    Get set of parameters which need to go into the parameter table

    Arguments:
        model: PEtab model
        condition_df: PEtab condition table
        observable_df: PEtab observable table
        measurement_df: PEtab measurement table
        mapping_df: PEtab mapping table

    Returns:
        Set of parameter IDs which PEtab requires to be present in the
        parameter table. That is all {observable,noise}Parameters from the
        measurement table as well as all parametric condition table overrides
        that are not defined in the model.
    """
    # use ordered dict as proxy for ordered set
    parameter_ids = OrderedDict()

    # Add parameters from measurement table, unless they are fixed parameters
    def append_overrides(overrides):
        for p in overrides:
            if isinstance(p, str) and p not in condition_df.columns:
                parameter_ids[p] = None

    for _, row in measurement_df.iterrows():
        # we trust that the number of overrides matches
        append_overrides(measurements.split_parameter_replacement_list(
            row.get(OBSERVABLE_PARAMETERS, None)))
        append_overrides(measurements.split_parameter_replacement_list(
            row.get(NOISE_PARAMETERS, None)))

    # Add output parameters except for placeholders
    for kwargs in [dict(observables=True, noise=False),
                   dict(observables=False, noise=True)]:
        output_parameters = observables.get_output_parameters(
            observable_df, model, mapping_df=mapping_df, **kwargs)
        placeholders = observables.get_placeholders(
            observable_df, **kwargs)
        for p in output_parameters:
            if p not in placeholders:
                parameter_ids[p] = None

    # Add condition table parametric overrides unless already defined in the
    #  model
    for p in conditions.get_parametric_overrides(condition_df):
        if not model.has_entity_with_id(p):
            parameter_ids[p] = None

    # remove parameters that occur in the condition table and are overridden
    #  for ALL conditions
    for p in condition_df.columns[~condition_df.isnull().any()]:
        try:
            del parameter_ids[p]
        except KeyError:
            pass
    return parameter_ids.keys()


def get_valid_parameters_for_parameter_table(
        model: Model,
        condition_df: pd.DataFrame,
        observable_df: pd.DataFrame,
        measurement_df: pd.DataFrame,
        mapping_df: pd.DataFrame = None,
) -> Set[str]:
    """
    Get set of parameters which may be present inside the parameter table

    Arguments:
        model: PEtab model
        condition_df: PEtab condition table
        observable_df: PEtab observable table
        measurement_df: PEtab measurement table
        mapping_df: PEtab mapping table for additional checks

    Returns:
        Set of parameter IDs which PEtab allows to be present in the
        parameter table.
    """
    # - grab all allowed model parameters
    # - grab corresponding names from mapping table
    # - grab all output parameters defined in {observable,noise}Formula
    # - grab all parameters from measurement table
    # - grab all parametric overrides from condition table
    # - remove parameters for which condition table columns exist
    # - remove placeholder parameters
    #   (only partial overrides are not supported)

    # must not go into parameter table
    blackset = set()

    if observable_df is not None:
        placeholders = set(observables.get_placeholders(observable_df))

        # collect assignment targets
        blackset |= placeholders

    if condition_df is not None:
        blackset |= set(condition_df.columns.values) - {CONDITION_NAME}

    # don't use sets here, to have deterministic ordering,
    #  e.g. for creating parameter tables
    parameter_ids = OrderedDict.fromkeys(
        p for p in model.get_valid_parameters_for_parameter_table()
        if p not in blackset
    )

    if mapping_df is not None:
        for from_id, to_id in zip(mapping_df.index.values,
                                  mapping_df[MODEL_ENTITY_ID]):
            if to_id in parameter_ids.keys():
                parameter_ids[from_id] = None

    if observable_df is not None:
        # add output parameters from observables table
        output_parameters = observables.get_output_parameters(
            observable_df=observable_df, model=model)
        for p in output_parameters:
            if p not in blackset:
                parameter_ids[p] = None

    # Append parameters from measurement table, unless they occur as condition
    # table columns
    def append_overrides(overrides):
        for p in overrides:
            if isinstance(p, str) and p not in blackset:
                parameter_ids[p] = None

    if measurement_df is not None:
        for _, row in measurement_df.iterrows():
            # we trust that the number of overrides matches
            append_overrides(measurements.split_parameter_replacement_list(
                row.get(OBSERVABLE_PARAMETERS, None)))
            append_overrides(measurements.split_parameter_replacement_list(
                row.get(NOISE_PARAMETERS, None)))

    # Append parameter overrides from condition table
    if condition_df is not None:
        for p in conditions.get_parametric_overrides(condition_df):
            parameter_ids[p] = None

    return parameter_ids.keys()


def get_priors_from_df(
        parameter_df: pd.DataFrame,
        mode: Literal['initialization', 'objective'],
) -> List[Tuple]:
    """Create list with information about the parameter priors

    Arguments:
        parameter_df: PEtab parameter table
        mode: ``'initialization'`` or ``'objective'``

    Returns:
        List with prior information.
    """

    # get types and parameters of priors from dataframe
    par_to_estimate = parameter_df.loc[parameter_df[ESTIMATE] == 1]

    prior_list = []
    for _, row in par_to_estimate.iterrows():
        # retrieve info about type
        prior_type = str(row.get(f'{mode}PriorType', ''))
        if core.is_empty(prior_type):
            prior_type = PARAMETER_SCALE_UNIFORM

        # retrieve info about parameters of priors, make it a tuple of floats
        pars_str = str(row.get(f'{mode}PriorParameters', ''))
        if core.is_empty(pars_str):
            lb, ub = map_scale([row[LOWER_BOUND], row[UPPER_BOUND]],
                               [row[PARAMETER_SCALE]] * 2)
            pars_str = f'{lb}{PARAMETER_SEPARATOR}{ub}'
        prior_pars = tuple(
            float(entry) for entry in pars_str.split(PARAMETER_SEPARATOR)
        )

        # add parameter scale and bounds, as this may be needed
        par_scale = row[PARAMETER_SCALE]
        par_bounds = (row[LOWER_BOUND], row[UPPER_BOUND])

        # if no prior is specified, we assume a non-informative (uniform) one
        if prior_type == 'nan':
            prior_type = PARAMETER_SCALE_UNIFORM
            prior_pars = (scale(row[LOWER_BOUND], par_scale),
                          scale(row[UPPER_BOUND], par_scale))

        prior_list.append((prior_type, prior_pars, par_scale, par_bounds))

    return prior_list


def scale(
        parameter: numbers.Number,
        scale_str: PARAMETER_SCALE_ARGS,
) -> numbers.Number:
    """Scale parameter according to ``scale_str``.

    Arguments:
        parameter:
            Parameter to be scaled.
        scale_str:
            One of ``'lin'`` (synonymous with ``''``), ``'log'``, ``'log10'``.

    Returns:
        The scaled parameter.
    """
    if scale_str == LIN or not scale_str:
        return parameter
    if scale_str == LOG:
        return np.log(parameter)
    if scale_str == LOG10:
        return np.log10(parameter)
    raise ValueError(f"Invalid parameter scaling: {scale_str}")


def unscale(
        parameter: numbers.Number,
        scale_str: PARAMETER_SCALE_ARGS,
) -> numbers.Number:
    """Unscale parameter according to ``scale_str``.

    Arguments:
        parameter:
            Parameter to be unscaled.
        scale_str:
            One of ``'lin'`` (synonymous with ``''``), ``'log'``, ``'log10'``.

    Returns:
        The unscaled parameter.
    """
    if scale_str == LIN or not scale_str:
        return parameter
    if scale_str == LOG:
        return np.exp(parameter)
    if scale_str == LOG10:
        return 10**parameter
    raise ValueError(f"Invalid parameter scaling: {scale_str}")


def map_scale(
    parameters: Sequence[numbers.Number],
    scale_strs: Union[Iterable[PARAMETER_SCALE_ARGS], PARAMETER_SCALE_ARGS],
) -> Iterable[numbers.Number]:
    """Scale the parameters, i.e. as :func:`scale`, but for Sequences.

    Arguments:
        parameters:
            Parameters to be scaled.
        scale_strs:
            Scales to apply. Broadcast if a single string.

    Returns:
        The scaled parameters.
    """
    if isinstance(scale_strs, str):
        scale_strs = [scale_strs] * len(parameters)
    return map(lambda x: scale(x[0], x[1]), zip(parameters, scale_strs))


def map_unscale(
    parameters: Sequence[numbers.Number],
    scale_strs: Union[Iterable[PARAMETER_SCALE_ARGS], PARAMETER_SCALE_ARGS],
) -> Iterable[numbers.Number]:
    """Unscale the parameters, i.e. as :func:`unscale`, but for Sequences.

    Arguments:
        parameters:
            Parameters to be unscaled.
        scale_strs:
            Scales that the parameters are currently on.
            Broadcast if a single string.

    Returns:
        The unscaled parameters.
    """
    if isinstance(scale_strs, str):
        scale_strs = [scale_strs] * len(parameters)
    return map(lambda x: unscale(x[0], x[1]), zip(parameters, scale_strs))


def normalize_parameter_df(parameter_df: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns and fill in default values."""
    df = parameter_df.copy(deep=True)

    if PARAMETER_NAME not in df:
        df[PARAMETER_NAME] = df.reset_index()[PARAMETER_ID]

    prior_type_cols = [INITIALIZATION_PRIOR_TYPE,
                       OBJECTIVE_PRIOR_TYPE]
    prior_par_cols = [INITIALIZATION_PRIOR_PARAMETERS,
                      OBJECTIVE_PRIOR_PARAMETERS]
    # iterate over initialization and objective priors
    for prior_type_col, prior_par_col in zip(prior_type_cols, prior_par_cols):
        # fill in default values for prior type
        if prior_type_col not in df:
            df[prior_type_col] = PARAMETER_SCALE_UNIFORM
        else:
            for irow, row in df.iterrows():
                if core.is_empty(row[prior_type_col]):
                    df.loc[irow, prior_type_col] = PARAMETER_SCALE_UNIFORM
        if prior_par_col not in df:
            df[prior_par_col] = None
        for irow, row in df.iterrows():
            if core.is_empty(row[prior_par_col]) \
                    and row[prior_type_col] == PARAMETER_SCALE_UNIFORM:
                lb, ub = map_scale([row[LOWER_BOUND], row[UPPER_BOUND]],
                                   [row[PARAMETER_SCALE]] * 2)
                df.loc[irow, prior_par_col] = f'{lb}{PARAMETER_SEPARATOR}{ub}'

    return df
