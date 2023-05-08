"""Functions for working with the PEtab observables table"""

import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Union

import pandas as pd
import sympy as sp
from sympy.abc import _clash

from . import core, lint
from .C import *  # noqa: F403
from .models import Model

__all__ = [
    'create_observable_df',
    'get_formula_placeholders',
    'get_observable_df',
    'get_output_parameters',
    'get_placeholders',
    'write_observable_df'
]


def get_observable_df(
        observable_file: Union[str, pd.DataFrame, Path, None]
) -> Union[pd.DataFrame, None]:
    """
    Read the provided observable file into a ``pandas.Dataframe``.

    Arguments:
        observable_file: Name of the file to read from or pandas.Dataframe.

    Returns:
        Observable DataFrame
    """
    if observable_file is None:
        return observable_file

    if isinstance(observable_file, (str, Path)):
        observable_file = pd.read_csv(observable_file, sep='\t',
                                      float_precision='round_trip')

    lint.assert_no_leading_trailing_whitespace(
        observable_file.columns.values, "observable")

    if not isinstance(observable_file.index, pd.RangeIndex):
        observable_file.reset_index(inplace=True)

    try:
        observable_file.set_index([OBSERVABLE_ID], inplace=True)
    except KeyError:
        raise KeyError(
            f"Observable table missing mandatory field {OBSERVABLE_ID}.")

    return observable_file


def write_observable_df(df: pd.DataFrame, filename: Union[str, Path]) -> None:
    """Write PEtab observable table

    Arguments:
        df: PEtab observable table
        filename: Destination file name
    """
    df = get_observable_df(df)
    df.to_csv(filename, sep='\t', index=True)


def get_output_parameters(
        observable_df: pd.DataFrame,
        model: Model,
        observables: bool = True,
        noise: bool = True,
        mapping_df: pd.DataFrame = None
) -> List[str]:
    """Get output parameters

    Returns IDs of parameters used in observable and noise formulas that are
    not defined in the model.

    Arguments:
        observable_df: PEtab observable table
        model: The underlying model
        observables: Include parameters from observableFormulas
        noise: Include parameters from noiseFormulas
        mapping_df: PEtab mapping table

    Returns:
        List of output parameter IDs
    """
    formulas = []
    if observables:
        formulas.extend(observable_df[OBSERVABLE_FORMULA])
    if noise and NOISE_FORMULA in observable_df:
        formulas.extend(observable_df[NOISE_FORMULA])
    output_parameters = OrderedDict()

    for formula in formulas:
        free_syms = sorted(sp.sympify(formula, locals=_clash).free_symbols,
                           key=lambda symbol: symbol.name)
        for free_sym in free_syms:
            sym = str(free_sym)
            if model.symbol_allowed_in_observable_formula(sym):
                continue

            # does it mapping to a model entity?
            if mapping_df is not None \
                    and sym in mapping_df.index \
                    and model.symbol_allowed_in_observable_formula(
                    mapping_df.loc[sym, MODEL_ENTITY_ID]):
                continue

            output_parameters[sym] = None

    return list(output_parameters.keys())


def get_formula_placeholders(
        formula_string: str,
        observable_id: str,
        override_type: Literal['observable', 'noise'],
) -> List[str]:
    """
    Get placeholder variables in noise or observable definition for the
    given observable ID.

    Arguments:
        formula_string: observable formula
        observable_id: ID of current observable
        override_type: ``'observable'`` or ``'noise'``, depending on whether
            ``formula`` is for observable or for noise model

    Returns:
        List of placeholder parameter IDs in the order expected in the
        observableParameter column of the measurement table.
    """
    if not formula_string:
        return []

    if not isinstance(formula_string, str):
        return []

    pattern = re.compile(r'(?:^|\W)(' + re.escape(override_type)
                         + r'Parameter\d+_' + re.escape(observable_id)
                         + r')(?=\W|$)')
    placeholder_set = set(pattern.findall(formula_string))

    # need to sort and check that there are no gaps in numbering
    placeholders = [f"{override_type}Parameter{i}_{observable_id}"
                    for i in range(1, len(placeholder_set) + 1)]

    if placeholder_set != set(placeholders):
        raise AssertionError("Non-consecutive numbering of placeholder "
                             f"parameter for {placeholder_set}")

    return placeholders


def get_placeholders(
        observable_df: pd.DataFrame,
        observables: bool = True,
        noise: bool = True,
) -> List[str]:
    """Get all placeholder parameters from observable table observableFormulas
    and noiseFormulas

    Arguments:
        observable_df: PEtab observable table
        observables: Include parameters from observableFormulas
        noise: Include parameters from noiseFormulas

    Returns:
        List of placeholder parameters from observable table observableFormulas
        and noiseFormulas.
    """

    # collect placeholder parameters overwritten by
    # {observable,noise}Parameters
    placeholder_types = []
    formula_columns = []
    if observables:
        placeholder_types.append('observable')
        formula_columns.append(OBSERVABLE_FORMULA)
    if noise:
        placeholder_types.append('noise')
        formula_columns.append(NOISE_FORMULA)

    placeholders = []
    for _, row in observable_df.iterrows():
        for placeholder_type, formula_column \
                in zip(placeholder_types, formula_columns):
            if formula_column not in row:
                continue

            cur_placeholders = get_formula_placeholders(
                row[formula_column], row.name, placeholder_type)
            placeholders.extend(cur_placeholders)
    return core.unique_preserve_order(placeholders)


def create_observable_df() -> pd.DataFrame:
    """Create empty observable dataframe

    Returns:
        Created DataFrame
    """

    return pd.DataFrame(data={col: [] for col in OBSERVABLE_DF_COLS})
