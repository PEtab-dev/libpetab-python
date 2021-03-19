"""PEtab core functions (or functions that don't fit anywhere else)"""

import logging
import os
import re
from typing import Iterable, Optional, Callable, Union, Any, Sequence, List
from warnings import warn

import numpy as np
import pandas as pd

from . import yaml
from .C import *  # noqa: F403

logger = logging.getLogger(__name__)


def get_simulation_df(simulation_file: str) -> pd.DataFrame:
    """Read PEtab simulation table

    Arguments:
        simulation_file: URL or filename of PEtab simulation table

    Returns:
        Simulation DataFrame
    """
    return pd.read_csv(simulation_file, sep="\t", index_col=None,
                       float_precision='round_trip')


def write_simulation_df(df: pd.DataFrame, filename: str) -> None:
    """Write PEtab simulation table

    Arguments:
        df: PEtab simulation table
        filename: Destination file name
    """
    with open(filename, 'w') as fh:
        df.to_csv(fh, sep='\t', index=False)


def get_visualization_df(visualization_file: str) -> pd.DataFrame:
    """Read PEtab visualization table

    Arguments:
        visualization_file: URL or filename of PEtab visualization table

    Returns:
        Visualization DataFrame
    """
    try:
        vis_spec = pd.read_csv(visualization_file, sep="\t", index_col=None,
                               float_precision='round_trip')
    except pd.errors.EmptyDataError:
        warn("Visualization table is empty. Defaults will be used. "
             "Refer to the documentation for details.")
        vis_spec = pd.DataFrame()
    return vis_spec


def write_visualization_df(df: pd.DataFrame, filename: str) -> None:
    """Write PEtab visualization table

    Arguments:
        df: PEtab visualization table
        filename: Destination file name
    """
    with open(filename, 'w') as fh:
        df.to_csv(fh, sep='\t', index=False)


def get_notnull_columns(df: pd.DataFrame, candidates: Iterable):
    """
    Return list of ``df``-columns in ``candidates`` which are not all null/nan.

    The output can e.g. be used as input for ``pandas.DataFrame.groupby``.

    Arguments:
        df:
            Dataframe
        candidates:
            Columns of ``df`` to consider
    """
    return [col for col in candidates
            if col in df and not np.all(df[col].isnull())]


def get_observable_id(parameter_id: str) -> str:
    """Get PEtab observable ID from PEtab-style sigma or observable
    `AssignmentRule`-target ``parameter_id``.

    e.g. for 'observable_obs1' -> 'obs1', for 'sigma_obs1' -> 'obs1'

    Arguments:
        parameter_id: Some parameter ID

    Returns:
        Observable ID
    """
    warn("This function will be removed in future releases.",
         DeprecationWarning)

    if parameter_id.startswith(r'observable_'):
        return parameter_id[len('observable_'):]

    if parameter_id.startswith(r'sigma_'):
        return parameter_id[len('sigma_'):]

    raise ValueError('Cannot extract observable id from: ' + parameter_id)


def flatten_timepoint_specific_output_overrides(
        petab_problem: 'petab.problem.Problem') -> None:
    """Flatten timepoint-specific output parameter overrides.

    If the PEtab problem definition has timepoint-specific
    `observableParameters` or `noiseParameters` for the same observable,
    replace those by replicating the respective observable.

    This is a helper function for some tools which may not support such
    timepoint-specific mappings. The observable table and measurement table
    are modified in place.

    Arguments:
        petab_problem:
            PEtab problem to work on
    """
    new_measurement_dfs = []
    new_observable_dfs = []
    possible_groupvars = [OBSERVABLE_ID, OBSERVABLE_PARAMETERS,
                          NOISE_PARAMETERS, SIMULATION_CONDITION_ID,
                          PREEQUILIBRATION_CONDITION_ID]
    groupvars = get_notnull_columns(petab_problem.measurement_df,
                                    possible_groupvars)
    for groupvar, measurements in \
            petab_problem.measurement_df.groupby(groupvars, dropna=False):
        obs_id = groupvar[groupvars.index(OBSERVABLE_ID)]
        # construct replacement id
        replacement_id = ''
        for field in possible_groupvars:
            if field in groupvars:
                val = groupvar[groupvars.index(field)
                               ].replace(';', '_').replace('.', '_')
                if replacement_id == '':
                    replacement_id = val
                elif val != '':
                    replacement_id += f'__{val}'

        logger.debug(f'Creating synthetic observable {obs_id}')
        if replacement_id in petab_problem.observable_df.index:
            raise RuntimeError('could not create synthetic observables '
                               f'since {replacement_id} was already '
                               'present in observable table')
        observable = petab_problem.observable_df.loc[obs_id].copy()
        observable.name = replacement_id
        for field, parname, target in [
            (NOISE_PARAMETERS, 'noiseParameter', NOISE_FORMULA),
            (OBSERVABLE_PARAMETERS, 'observableParameter', OBSERVABLE_FORMULA)
        ]:
            if field in measurements:
                observable[target] = re.sub(
                    fr'{parname}([0-9]+)_{obs_id}',
                    f'{parname}\\1_{replacement_id}',
                    observable[target]
                )

        measurements[OBSERVABLE_ID] = replacement_id
        new_measurement_dfs.append(measurements)
        new_observable_dfs.append(observable)

    petab_problem.observable_df = pd.concat(new_observable_dfs, axis=1).T
    petab_problem.observable_df.index.name = OBSERVABLE_ID
    petab_problem.measurement_df = pd.concat(new_measurement_dfs)


def concat_tables(
        tables: Union[str, pd.DataFrame, Iterable[Union[pd.DataFrame, str]]],
        file_parser: Optional[Callable] = None
) -> pd.DataFrame:
    """Concatenate DataFrames provided as DataFrames or filenames, and a parser

    Arguments:
        tables:
            Iterable of tables to join, as DataFrame or filename.
        file_parser:
            Function used to read the table in case filenames are provided,
            accepting a filename as only argument.

    Returns:
        The concatenated DataFrames
    """

    if isinstance(tables, pd.DataFrame):
        return tables

    if isinstance(tables, str):
        return file_parser(tables)

    df = pd.DataFrame()

    for tmp_df in tables:
        # load from file, if necessary
        if isinstance(tmp_df, str):
            tmp_df = file_parser(tmp_df)

        df = df.append(tmp_df, sort=False,
                       ignore_index=isinstance(tmp_df.index, pd.RangeIndex))

    return df


def to_float_if_float(x: Any) -> Any:
    """Return input as float if possible, otherwise return as is

    Arguments:
        x: Anything

    Returns:
        ``x`` as float if possible, otherwise ``x``
    """

    try:
        return float(x)
    except (ValueError, TypeError):
        return x


def is_empty(val) -> bool:
    """Check if the value `val`, e.g. a table entry, is empty.

    Arguments:
        val: The value to check.

    Returns:
        empty: Whether the field is to be considered empty.
    """
    return val == '' or pd.isnull(val)


def create_combine_archive(
        yaml_file: str, filename: str,
        family_name: Optional[str] = None,
        given_name: Optional[str] = None,
        email: Optional[str] = None,
        organization: Optional[str] = None,
) -> None:
    """Create COMBINE archive (http://co.mbine.org/documents/archive) based
    on PEtab YAML file.

    Arguments:
        yaml_file: Path to PEtab YAML file
        family_name: Family name of archive creator
        given_name: Given name of archive creator
        email: E-mail address of archive creator
        organization: Organization of archive creator
    """

    path_prefix = os.path.dirname(yaml_file)
    yaml_config = yaml.load_yaml(yaml_file)

    # function-level import, because module-level import interfered with
    # other SWIG interfaces
    try:
        import libcombine
    except ImportError:
        raise ImportError(
            "To use PEtab's COMBINE functionality, libcombine "
            "(python-libcombine) must be installed.")

    def _add_file_metadata(location: str, description: str = ""):
        """Add metadata to the added file"""
        omex_description = libcombine.OmexDescription()
        omex_description.setAbout(location)
        omex_description.setDescription(description)
        omex_description.setCreated(
            libcombine.OmexDescription.getCurrentDateAndTime())
        archive.addMetadata(location, omex_description)

    archive = libcombine.CombineArchive()

    # Add PEtab files and metadata
    archive.addFile(
        yaml_file,
        os.path.basename(yaml_file),
        libcombine.KnownFormats.lookupFormat("yaml"),
        True
    )
    _add_file_metadata(location=os.path.basename(yaml_file),
                       description="PEtab YAML file")

    # Add parameter file(s) that describe a single parameter table.
    # Works for a single file name, or a list of file names.
    for parameter_subset_file in (
            list(np.array(yaml_config[PARAMETER_FILE]).flat)):
        archive.addFile(
            os.path.join(path_prefix, parameter_subset_file),
            parameter_subset_file,
            libcombine.KnownFormats.lookupFormat("tsv"),
            False
        )
        _add_file_metadata(
            location=parameter_subset_file,
            description="PEtab parameter file"
        )

    for problem in yaml_config[PROBLEMS]:
        for sbml_file in problem[SBML_FILES]:
            archive.addFile(
                os.path.join(path_prefix, sbml_file),
                sbml_file,
                libcombine.KnownFormats.lookupFormat("sbml"),
                False
            )
            _add_file_metadata(location=sbml_file, description="SBML model")

        for field in [MEASUREMENT_FILES, OBSERVABLE_FILES,
                      VISUALIZATION_FILES, CONDITION_FILES]:
            if field not in problem:
                continue

            for file in problem[field]:
                archive.addFile(
                    os.path.join(path_prefix, file),
                    file,
                    libcombine.KnownFormats.lookupFormat("tsv"),
                    False
                )
                desc = field.split("_")[0]
                _add_file_metadata(location=file,
                                   description=f"PEtab {desc} file")

    # Add archive metadata
    description = libcombine.OmexDescription()
    description.setAbout(".")
    description.setDescription("PEtab archive")
    description.setCreated(libcombine.OmexDescription.getCurrentDateAndTime())

    # Add creator info
    creator = libcombine.VCard()
    if family_name:
        creator.setFamilyName(family_name)
    if given_name:
        creator.setGivenName(given_name)
    if email:
        creator.setEmail(email)
    if organization:
        creator.setOrganization(organization)
    description.addCreator(creator)

    archive.addMetadata(".", description)
    archive.writeToFile(filename)


def unique_preserve_order(seq: Sequence) -> List:
    """Return a list of unique elements in Sequence, keeping only the first
    occurrence of each element

    seq: Sequence to prune

    Returns:
        List of unique elements in ``seq``
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
